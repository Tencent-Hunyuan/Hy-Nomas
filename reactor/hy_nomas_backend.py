"""Reuse the released interactive pipeline, sampler, history and PixelUMM windows."""

from __future__ import annotations

import os
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

from hy_nomas_model import MAX_WORLD_FRAMES, HyNomasInput, ModelSettings, WorldStart
from hy_nomas_prompts import shape_prompt


class HyNomasBackend:
    def __init__(self, settings: ModelSettings, *, rank: int, world_size: int) -> None:
        self.settings = settings
        self.rank = rank
        self._scratch = tempfile.TemporaryDirectory(prefix="hy-nomas-")
        # Upstream imports flat configs/ and resolves its native assets from cwd.
        # DistributedRunner confines those process-wide effects to model workers.
        sys.path.insert(0, str(settings.source))
        os.chdir(settings.source)
        import torch

        torch.set_grad_enabled(False)
        torch.cuda.set_device(rank)
        from configs.config import _flatten_cfg
        from hydra import compose, initialize_config_dir
        from hymm.distributed.parallel_states import (
            ParallelDims,
            initialize_sequence_parallel_state_v2,
        )
        from hymm.pipeline.causal_ar import CPTCausalInferencePipeline

        from hy_nomas_attention import install_decoder_fa4, install_fa4

        capability = torch.cuda.get_device_capability()
        if capability not in ((8, 9), (10, 0)):
            raise RuntimeError(
                "This build supports Blackwell SM100 and the SM89 low-memory profile."
            )
        if settings.cpu_offload and (
            world_size != 1
            or settings.cache_int8_weights
            or settings.decoder_batch_size != 1
        ):
            raise ValueError(
                "CPU offload requires one GPU, one decoder window and uncached weights."
            )
        if capability == (10, 0):
            install_fa4()
            install_decoder_fa4()
        overrides = [
            "+experiment=inference",
            "model.dit.seg_ar_attention_backend=magi"
            if capability == (10, 0) and not settings.cpu_offload
            else "model.dit.seg_ar_attention_backend=flex",
            "inference.torch_compile=true",
            "model.vae.compile_decoder=false",
            f"parallelism.sp_size={world_size}",
            f"dit_weight={settings.dit}",
            f"model.vae.checkpoint={settings.decoder}",
            f"model.text_encoder.model_name={settings.text_encoder}",
            "model.vae.stable_prefix_streaming=true",
            "inference.seg_ar_segment_vae_decode=true",
            f"parallelism.text_encoder_cpu_offload={str(settings.cpu_offload).lower()}",
            f"parallelism.dit_cpu_offload={str(settings.cpu_offload).lower()}",
            "parallelism.dit_cpu_offload_for_decode=false",
            f"model.vae.cpu_offload={str(settings.cpu_offload).lower()}",
        ]
        if settings.precision != "bf16":
            from hy_nomas_quantization import install_fp8

            install_fp8(settings.source, settings.cutlass, capability)
            overrides.append("model.vae.cutlass_fp8=true")
        if settings.precision == "int8_fp8":
            overrides += [
                "inference.dit_weight_quantization=int8",
                "inference.kv_cache_quantization=int8",
            ]
        if settings.cpu_offload:
            overrides += [
                "inference.sparse_attention.router_mode=triton_bf16",
                "inference.sparse_attention.flex_sparse_layout=aligned",
                "inference.sparse_attention.memory_efficient=true",
                "inference.sparse_attention.memory_adaptive_heads=true",
            ]
        if world_size == 1:
            overrides += [
                "model.vae.parallel_mode=none",
                "parallelism.shard_text_encoder=false",
            ]
        if settings.history_frames is not None:
            overrides += [
                "inference.sliding_window.enabled=true",
                f"inference.sliding_window.past_frames={settings.history_frames}",
                f"inference.sliding_window.sampling={settings.history_sampling}",
            ]
        with initialize_config_dir(
            config_dir=str(settings.source / "configs"), version_base="1.3"
        ):
            config = _flatten_cfg(compose(config_name="config", overrides=overrides))
        mesh = None
        if world_size > 1:
            parallel = ParallelDims(
                dp_replicate=1,
                dp_shard=1,
                sp=world_size,
                tp=1,
                pp=1,
                world_size=world_size,
            )
            mesh = parallel.build_mesh(device_type="cuda")
            initialize_sequence_parallel_state_v2(parallel, mesh, False)
        self.pipeline = CPTCausalInferencePipeline(
            args=config,
            device=torch.device("cuda", rank),
            dit_model_path=str(settings.dit),
            world_mesh=mesh,
        )
        from hy_nomas_fusions import install_qk_fusion
        from hy_nomas_infra import (
            install_gpu_output,
            install_window_batching,
        )

        if settings.cache_int8_weights:
            from hy_nomas_quantization import cache_int8_weights

            cache_int8_weights(self.pipeline.transformer)
        install_qk_fusion(self.pipeline.vae.decoder)
        install_window_batching(
            self.pipeline.vae,
            settings.decoder_batch_size,
            quantized=settings.precision != "bf16",
        )
        self._gpu_output = not settings.cpu_offload
        if self._gpu_output:
            install_gpu_output(self.pipeline)
        else:
            from hy_nomas_infra import install_cpu_streaming_state

            install_cpu_streaming_state(self.pipeline.vae)
        self._exchange = None
        if world_size > 1:
            from hy_nomas_exchange import SequenceExchange

            self._exchange = SequenceExchange(packed_qkv=True)
        # Compile the repeated region once per shape, avoiding a graph containing
        # all decoder layers. The released DiT compiles its own pre/post regions.
        for layer in self.pipeline.vae.decoder.layers:
            layer.compile(fullgraph=True, dynamic=False)
        # Text encoding is pure for the frozen encoder. Key by all native
        # arguments; cache writes and attention still run for every segment.
        self._text_encoding = lru_cache(maxsize=2)(self.pipeline._encode_single_prompt)
        self.pipeline._encode_single_prompt = self._text_encoding
        self._seed = 42
        self._segment_frames = 120
        self._last_prompt: str | None = None
        self._generated = 0
        self._emitted = 0

    def start(self, start: WorldStart) -> None:
        self._seed = start.seed
        self._segment_frames = start.segment_frames

    def step(self, input: HyNomasInput) -> tuple[np.ndarray, int, bool]:
        import torch
        from hymm.utils.video_output import video_to_uint8_numpy

        frames = min(self._segment_frames, MAX_WORLD_FRAMES - self._generated)
        camera_path = None
        if input.c2ws is not None:
            camera_path = str(Path(self._scratch.name) / "camera.npz")
            np.savez(camera_path, c2ws=input.c2ws, intrinsics=input.intrinsics)
        if self._exchange is not None:
            self._exchange.begin_step()
        output = self.pipeline.inference(
            prompts=[
                shape_prompt(
                    input.prompt,
                    self.pipeline.args,
                    camera_continuation=(
                        camera_path is not None and input.prompt == self._last_prompt
                    ),
                )
            ],
            num_frames=frames,
            height=self.settings.height,
            width=self.settings.width,
            guidance_scale=1.0,
            camera_guidance_scale=1.0,
            num_inference_steps=self.settings.denoising_steps,
            flow_shift=5.0,
            seed=self._seed,
            negative_prompt=self.pipeline.args.inference.neg_prompt,
            segment_frame_boundaries=[0, frames],
            camera_trajectory=camera_path,
            segment_has_camera=[camera_path is not None],
            preserve_kv_cache=True,
            trajectory_translation_scale=input.translation_strength,
            trajectory_rotation_scale=input.rotation_strength,
        )
        self._generated += frames
        self._last_prompt = input.prompt
        complete = self._generated == MAX_WORLD_FRAMES
        session = self.pipeline._interactive_pixelumm_decode_session
        if self.rank == 0:
            metadata = self.pipeline._last_pixelumm_streaming_metadata
            start = int(metadata["replace_pixel_start"])
            stable_end = int(metadata["stable_pixel_frames"])
            if start != self._emitted:
                raise RuntimeError("PixelUMM stable timeline is discontinuous.")
            # WebRTC is append-only. Emit the exact stable prefix, retaining the
            # future-dependent overlap in the native decoder until the next turn.
            stable = output[:, :, : stable_end - start]
        if complete:
            tail = session.finish()
            if self.rank == 0:
                tail = self.pipeline._postprocess_vae_video(tail)
                stable = torch.cat([stable, tail], dim=2)
        if self.rank != 0:
            return np.empty((0, 0, 0, 3), dtype=np.uint8), frames, complete
        if self._gpu_output:
            rgb = stable[0].permute(1, 2, 3, 0).contiguous().numpy()
        else:
            rgb = np.ascontiguousarray(video_to_uint8_numpy(stable[0], chunk_frames=8))
        self._emitted += len(rgb)
        return rgb, frames, complete

    def reset(self) -> None:
        from hy_nomas_attention import clear_fa4_cache

        clear_fa4_cache()
        self._text_encoding.cache_clear()
        if self._exchange is not None:
            self._exchange.reset()
        pipeline = self.pipeline
        pipeline._finish_interactive_decode_phase()
        pipeline.kv_cache = None
        pipeline.cache_mask_list = None
        pipeline.cache_valid_len = 0
        pipeline.global_frame_counter = 0
        # These lists are first allocated by the native cache initializer.
        pipeline.segment_token_lengths = []
        pipeline._interactive_cached_action_tails = []
        pipeline._interactive_cached_seg_video_ranges = []
        pipeline._interactive_segment_count = 0
        pipeline._interactive_segment_cache_tokens = []
        pipeline._interactive_segment_latent_lengths = []
        pipeline._reset_interactive_noise_state()
        pipeline._reset_interactive_pixelumm_decode_state()
        self._generated = 0
        self._emitted = 0
        self._last_prompt = None
