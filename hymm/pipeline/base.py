"""Shared base class for inference pipelines."""
from hymm.utils.inference_metrics import profiled

from contextlib import contextmanager
import math
import os
import time
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from loguru import logger

from hymm.pipeline.solvers.euler import EulerSolver
from hymm.models.checkpointing.load import load_transformer, load_text_encoder
from hymm.models.vae import load_vae
from hymm.distributed.fsdp_util import apply_fsdp2_text_encoder
from hymm.utils.model_cpu_offload import (
    install_model_cpu_offload, offload_dit_for_decode, validate_model_cpu_offload,
)


def align_to(value, alignment):
    return int(math.ceil(value / alignment) * alignment)


class BasePipeline(nn.Module):
    """Common model loading, text encoding, and VAE decoding for all pipelines."""

    def __init__(
        self,
        args,
        device: torch.device,
        dit_model_path: str,
        world_mesh=None,
    ):
        super().__init__()
        self.args = args
        self.device = device
        self.world_mesh = world_mesh
        validate_model_cpu_offload(args, device)
        from hymm.utils.range_attention_config import validate_range_attention_device
        validate_range_attention_device(args, device)
        self.enable_parallel_vae_decode = (
            os.environ.get("HY_INFER_PARALLEL_VAE_DECODE", "1").strip().lower()
            in {"1", "true", "yes", "on"}
        ) and device.type == "cuda"

        # Force bf16 for inference when training was done with fp32 master
        # weights.  Rationale:
        #   - fp32 master weight is only meaningful for AdamW updates during
        #     training (avoids ~1e-7 grads being rounded away by bf16's ULP).
        #   - Inference has no optimizer step, so fp32 buys nothing.
        #   - FlashAttention requires fp16 / bf16 / fp8 — running the DiT in
        #     fp32 raises at the first attention layer.
        # This lets training configs (which set master_weight_type=fp32) be
        # reused for inference (e.g. via scripts/inference/monitor_and_infer.sh)
        # without a separate inference yaml that overrides it back to bf16.
        from omegaconf import open_dict

        if getattr(args.dit, "master_weight_type", "bf16") == "fp32":
            logger.warning(
                "[BasePipeline] master_weight_type=fp32 is invalid for inference "
                "(FlashAttention requires fp16/bf16/fp8); forcing it to bf16. "
                "fp32 master weights matter only for AdamW training updates."
            )
            with open_dict(args):
                args.dit.master_weight_type = "bf16"
                # Keep the flat alias in sync (set by configs/config.py::_flatten_cfg)
                if hasattr(args, "master_weight_type"):
                    args.master_weight_type = "bf16"

        self.target_dtype = {"fp32": torch.float32, "fp16": torch.float16}.get(
            args.dit.master_weight_type, torch.bfloat16
        )
        self._load_models(dit_model_path)

    @profiled("model_load")
    def _load_models(self, dit_model_path: str):
        args = self.args
        device = self.device
        target_dtype = self.target_dtype

        from hymm.models.dit.modules.int8_stored_bf16_linear import (
            load_w8_bf16_transformer, resolve_w8_bf16_checkpoint,
        )

        from hymm.distributed.parallel_states import nccl_info
        from hymm.distributed.tensor_parallel import (
            parallelize_transformer, validate_tensor_parallel_config,
        )
        validate_tensor_parallel_config(args)
        tp_size = getattr(args.parallelism, "tp_size", 1)
        if tp_size != nccl_info.tp_size:
            raise ValueError("parallelism.tp_size does not match the initialized TP process group")
        if tp_size > 1:
            from hymm.utils.parallel import sync_random_states
            sync_random_states(nccl_info.parallel_dims)

        dit_offload = bool(getattr(args.parallelism, "dit_cpu_offload", False))
        dit_device = torch.device("cpu") if dit_offload else device
        quantized = resolve_w8_bf16_checkpoint(args.inference, dit_model_path)
        lora_adapter = getattr(args.inference, "lora_adapter", None)
        if quantized:
            if lora_adapter:
                raise ValueError("INT8 DiT cannot be combined with inference.lora_adapter")
            if target_dtype != torch.bfloat16:
                raise ValueError("INT8 DiT requires BF16 compute")
            if getattr(args, "convert_state_dict_in_out", False) or getattr(
                args, "convert_state_dict_16168_to_16164", False
            ):
                raise ValueError("INT8 DiT does not support legacy checkpoint conversion")
            self.transformer = load_w8_bf16_transformer(
                args.dit.model_type, dit_model_path, args, dit_device, logger,
            )
        else:
            load_dtype = (
                torch.float32 if args.dit.master_weight_type == "fp32" else target_dtype
            )
            self.transformer = load_transformer(
                args.dit.model_type,
                dit_model_path,
                factor_kwargs={"device": torch.device("cpu") if tp_size > 1 else dit_device, "dtype": load_dtype},
                convert_state_dict_in_out=getattr(args, "convert_state_dict_in_out", False),
                convert_state_dict_16168_to_16164=getattr(
                    args, "convert_state_dict_16168_to_16164", False
                ),
                args=args,
                logger=logger,
            )
            if lora_adapter:
                from hymm.models.lora import load_lora_adapter
                load_lora_adapter(self.transformer, lora_adapter)
            if tp_size > 1:
                parallelize_transformer(self.transformer, nccl_info.tp_group, nccl_info.sp_size)
                logger.info(f"DiT tensor parallel enabled: rank={nccl_info.rank_within_tpgroup}/{tp_size}")
            self.transformer.to(dtype=target_dtype, device=dit_device)
        # Freeze only this inference instance, after optional LoRA loading.
        # The INT8 path deliberately never applies a blanket floating dtype cast.
        self.transformer.eval().requires_grad_(False)
        self.transformer.nomas_sparse_config = dict(getattr(args.inference, "sparse_attention", {}))
        from collections import Counter
        from hymm.models.dit.modules.attention.nomas_inference_optim import validate as validate_optim
        optim = validate_optim(getattr(args.inference, 'sparse_optim', {}))
        if optim.get('enabled', False):
            if (args.dit.seg_ar_attention_backend != 'magi' or args.dit.master_weight_type != 'bf16'
                    or args.inference.rollout_type != 'seg_ar'
                    or not self.transformer.nomas_sparse_config.get('enabled', False)
                    or self.transformer.nomas_sparse_config.get('router_mode') != 'cute_bf16'
                    or args.inference.dit_weight_quantization != 'none'
                    or args.inference.kv_cache_quantization != 'none'
                    or args.inference.sliding_window.enabled):
                raise ValueError('sparse_optim currently requires BF16 pooled CuTe/Magi seg_ar without SWA/quantization')
        self.transformer.sparse_optim = optim
        self.transformer.sparse_optim_counts = Counter()
        compile_cache = bool(getattr(args.inference, "torch_compile", False))
        for block in getattr(self.transformer, "double_blocks", []):
            block.compile_cache_prepost = compile_cache

        if dit_offload:
            install_model_cpu_offload(self.transformer, device, stage_prefix="dit")

        te_offload = getattr(args.parallelism, "text_encoder_cpu_offload", False)
        self.text_encoder = load_text_encoder(
            device=torch.device("cpu") if te_offload else device,
            logger=logger,
            args=args,
        )
        self.text_encoder.eval().requires_grad_(False)
        if te_offload and (args.parallelism.sp_size * tp_size == 1
                           or not getattr(args.parallelism, "shard_text_encoder", False)):
            install_model_cpu_offload(self.text_encoder, device)
        self._maybe_shard_text_encoder()

        from hymm.models.vae.pixelumm_decoder import (
            maybe_load_pixelumm_decoder,
        )

        pixelumm_offload = getattr(args.vae, "cpu_offload", False)
        # Serialized FP8 buffers can load directly on CPU for offload.
        # Only legacy BF16-to-FP8 conversion needs CUDA initialization.
        from hymm.models.vae.pixelumm_decoder.prequantized import is_prequantized_pixelumm
        fp8_enabled = (getattr(args.vae, "cutlass_fp8", False)
                       or getattr(args.vae, "deepgemm_fp8", False))
        pixelumm_load_on_cpu = pixelumm_offload and (
            not fp8_enabled or is_prequantized_pixelumm(args.vae.checkpoint)
        )
        pixelumm_decoder = maybe_load_pixelumm_decoder(
            args,
            device=torch.device("cpu") if pixelumm_load_on_cpu else device,
            logger=logger,
        )
        if pixelumm_decoder is None:
            self.vae, _, self.s_ratio, self.t_ratio = load_vae(
                args.vae_type,
                args.vae.vae_precision,
                vae_path=args.vae.checkpoint,
                logger=logger,
                device=device,
            )
        else:
            self.vae = pixelumm_decoder
            self.s_ratio = pixelumm_decoder.spatial_compression_ratio
            self.t_ratio = pixelumm_decoder.time_compression_ratio

        self.vae.eval().requires_grad_(False)
        if pixelumm_offload:
            install_model_cpu_offload(self.vae, device)

        self.scheduler = EulerSolver(
            shift=args.inference.infer_flow_shift_video,
            reverse=True,
            solver="euler",
        )

    @contextmanager
    def _dit_denoise_residency(self):
        """Keep DiT on this rank's GPU for a whole denoising/cache-write phase.

        Text encoding and VAE/output processing stay outside this scope. No
        per-forward hook is installed, so all denoise steps share one transfer.
        """
        parallelism = getattr(self.args, "parallelism", None)
        if not getattr(parallelism, "dit_cpu_offload", False):
            yield
            return
        if getattr(parallelism, "dit_cpu_offload_for_decode", False):
            raise ValueError(
                "parallelism.dit_cpu_offload and parallelism.dit_cpu_offload_for_decode "
                "cannot both be true. If GPU memory allows, enable only "
                "parallelism.dit_cpu_offload_for_decode."
            )
        controller = getattr(self.transformer, "_model_cpu_offload", None)
        if controller is None or controller.stage_prefix != "dit":
            raise RuntimeError("DiT CPU offload requires an installed denoising-phase controller")
        self.transformer._last_inference_mode = torch.is_inference_mode_enabled()
        with controller.on_device():
            yield

    def _maybe_shard_text_encoder(self):
        """Shard the frozen text encoder during distributed inference."""
        args = self.args
        parallelism = getattr(args, "parallelism", None)
        if parallelism is None or not getattr(parallelism, "shard_text_encoder", False):
            return

        sp_size = getattr(parallelism, "sp_size", 1) * getattr(parallelism, "tp_size", 1)
        if sp_size <= 1:
            logger.warning(
                "parallelism.shard_text_encoder=true but model parallelism is disabled (sp_size * tp_size=1); "
                "skipping text-encoder sharding."
            )
            return

        if self.world_mesh is None:
            if getattr(parallelism, "text_encoder_cpu_offload", False):
                raise ValueError("Sharded TE CPU offload requires the inference model-parallel device mesh")
            logger.warning(
                "parallelism.shard_text_encoder=true but no world_mesh was provided; "
                "skipping text-encoder sharding."
            )
            return

        from hymm.distributed.parallel_states import nccl_info
        sp_mesh = (nccl_info.parallel_dims.mp_mesh if getattr(parallelism, "tp_size", 1) > 1
                   else self.world_mesh["sp"])
        if sp_mesh.size() != sp_size:
            raise ValueError("Text encoder mesh does not match sp_size * tp_size")
        cpu_offload = getattr(parallelism, "text_encoder_cpu_offload", False)
        apply_fsdp2_text_encoder(
            self.text_encoder,
            sp_mesh=sp_mesh,
            param_dtype=self.text_encoder.model.dtype,
            logger=logger,
            cpu_offload=cpu_offload,
        )
        if cpu_offload:
            self.text_encoder._fsdp_cpu_offload_device = self.device
        torch.cuda.empty_cache()
        logger.info(
            f"Text encoder FSDP sharding enabled: model_parallel_size={sp_size}, "
            f"cpu_offload={cpu_offload}. CPU shards are staged per forward; "
            "encoded features and small buffers keep their own device lifetimes."
        )

    def _encode_single_prompt(
        self,
        prompt: str,
        data_type: str = "video",
        is_uncond: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode one text prompt, returning (hidden_states, attention_mask)."""
        text_len = (
            getattr(self.args.text_encoder, "text_len", 256)
            if hasattr(self.args, "text_encoder")
            else getattr(self.args, "text_len", 256)
        )
        text_inputs = self.text_encoder.text2tokens(
            [prompt],
            data_type=data_type,
            max_length=text_len,
        )
        outputs = self.text_encoder.encode(
            text_inputs,
            data_type=data_type,
            device=self.device,
            is_uncond=is_uncond,
        )
        states = outputs.hidden_state.to(dtype=self.target_dtype, device=self.device)
        mask = outputs.attention_mask
        if mask is not None:
            mask = mask.to(dtype=self.target_dtype, device=self.device)
        return states, mask

    def _prepare_vae_decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Apply the shared VAE scaling and dtype conversion."""
        if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor:
            latents = (
                latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
            )
        else:
            latents = latents / self.vae.config.scaling_factor

        vae_dtype = next(self.vae.parameters()).dtype
        return latents.to(vae_dtype)

    @staticmethod
    @profiled("postprocess")
    def _postprocess_vae_video(
        video: torch.Tensor, *, inplace: bool = False,
    ) -> torch.Tensor:
        """Map raw VAE output to [0, 1], optionally consuming owned storage.

        Streaming callers may retain raw pixels for later overlap blending, so
        the default preserves its input. Only an owner that no longer needs
        those raw pixels may request the no-grad, in-place path.
        """
        if inplace:
            if torch.is_grad_enabled() or video.requires_grad:
                raise RuntimeError("In-place video postprocessing requires no-grad output")
            # Keep each operation separate to preserve the existing dtype's
            # intermediate rounding without full-video temporary allocations.
            return video.mul_(0.5).add_(0.5).clamp_(0, 1)
        return (video * 0.5 + 0.5).clamp(0, 1)

    def _create_segment_vae_decode_session(
        self,
        *,
        group=None,
        rank: int = 0,
        world_size: int = 1,
    ):
        """Create the exact DCAE session used by opt-in seg_ar streaming."""
        from hymm.models.vae.autoencoder_dcae_v import AutoencoderDCAEV

        if not isinstance(self.vae, AutoencoderDCAEV):
            raise ValueError(
                "inference.seg_ar_segment_vae_decode=true requires DCAE-V "
                f"(AutoencoderDCAEV), got {type(self.vae).__name__}."
            )
        if world_size > 1:
            from hymm.models.vae.operator_parallel_decode import (
                DistributedDCAEVStreamingDecodeSession,
            )

            return DistributedDCAEVStreamingDecodeSession(
                self.vae.model,
                group=group,
                rank=rank,
                world_size=world_size,
            )
        return self.vae.create_streaming_decode_session()

    @offload_dit_for_decode
    def _decode_latents(self, latents: torch.Tensor, num_frames: int) -> torch.Tensor:
        """Decode latents to pixel-space video tensor in [0, 1]."""
        self.vae.enable_tiling()
        if num_frames > 32:
            self.vae.enable_temporal_tiling()

        latents = self._prepare_vae_decode_latents(latents)
        if self.enable_parallel_vae_decode and not getattr(
            self.vae, "handles_distributed_decode", False
        ):
            import torch.distributed as dist

            from hymm.models.vae.operator_parallel_decode import (
                decode_operator_temporal_parallel,
            )

            from hymm.distributed.parallel_states import get_model_parallel_info
            group, world_size, rank = get_model_parallel_info()
            if not (dist.is_available() and dist.is_initialized() and world_size > 1):
                logger.warning(
                    "HY_INFER_PARALLEL_VAE_DECODE=1 but no multi-rank SP group is "
                    "active; falling back to the baseline VAE decode."
                )
                video = self.vae.decode(latents, return_dict=False)[0]
            else:
                split_multiple_raw = os.environ.get(
                    "HY_INFER_PARALLEL_VAE_SPLIT_MULTIPLE", "10"
                ).strip()
                split_multiple = int(split_multiple_raw) if split_multiple_raw else None
                torch.cuda.synchronize(self.device)
                decode_start = time.perf_counter()
                video = decode_operator_temporal_parallel(
                    self.vae,
                    latents,
                    group=group,
                    rank=rank,
                    world_size=world_size,
                    split_multiple=split_multiple,
                    tile_loop=True,
                    async_cache_send=False,
                    logger=logger.info,
                )
                torch.cuda.synchronize(self.device)
                logger.info(
                    "[VAE decode] strategy=operator_temporal_tiled "
                    f"rank={rank}/{world_size} "
                    f"decode_s={time.perf_counter() - decode_start:.3f}"
                )
        else:
            decode_kwargs = (
                {"pixel_frames": num_frames}
                if getattr(self.vae, "handles_distributed_decode", False)
                else {}
            )
            video = self.vae.decode(
                latents,
                return_dict=False,
                **decode_kwargs,
            )[0]
        # This full-clip decode owns the returned pixels; no streaming session
        # retains them for later blending. Consume them once inference is done.
        return self._postprocess_vae_video(
            video, inplace=not torch.is_grad_enabled() and not video.requires_grad,
        )

    # ------------------------------------------------------------------
    # Intra-segment per-action attention-mask helpers (shared by the
    # seg_ar cache pipeline and the one_shot bidirectional pipeline).
    # Depend only on self.args / self.text_encoder / self.t_ratio.
    # ------------------------------------------------------------------
    def _compute_action_start(self, prompt: str) -> int:
        """Token index (post-crop, S_enc coords) where ``contained_actions`` begins.

        Mirrors ``WM1MTrainingDataset._compute_text_action_start_index``: tokenize
        the full segment JSON and its action-stripped variant (``contained_actions``
        removed from ``events[0]``), take the token-level common-prefix length on
        the RAW (pre-crop) ids, and subtract the encoder ``crop_start``.

        ``contained_actions`` is forced to be the LAST key of the event by the
        prompt builder, so the stripped JSON is an exact prefix of the full JSON
        and the common prefix ends precisely at the action boundary.

        Returns an int in ``[0, text_len)``, or -1 when the segment carries no
        ``contained_actions`` (no head/tail split — whole block always visible).
        """
        import json as _json

        try:
            obj = _json.loads(prompt)
        except Exception:
            return -1
        segs = obj.get("segments") if isinstance(obj, dict) else None
        if not (isinstance(segs, list) and segs and isinstance(segs[0], dict)):
            return -1
        seg0 = segs[0]
        acts = seg0.get("contained_actions")
        has_actions = (isinstance(acts, list) and len(acts) > 0) or (
            isinstance(acts, dict) and len(acts) > 0
        )
        if not has_actions:
            return -1

        # Build the action-stripped variant (drop contained_actions on segments[0]).
        strip_obj = {
            k: (
                [{ek: sv for ek, sv in segs[0].items() if ek != "contained_actions"}]
                if k == "segments"
                else v
            )
            for k, v in obj.items()
        }
        strip_prompt = _json.dumps(strip_obj, ensure_ascii=False)

        text_len = getattr(self.args, "text_len", 256)
        full_ids = (
            self.text_encoder.text2tokens(
                [prompt],
                data_type="video",
                max_length=text_len,
            )["input_ids"]
            .squeeze(0)
            .tolist()
        )
        strip_ids = (
            self.text_encoder.text2tokens(
                [strip_prompt],
                data_type="video",
                max_length=text_len,
            )["input_ids"]
            .squeeze(0)
            .tolist()
        )

        crop_start = 0
        try:
            tmpl = getattr(self.text_encoder, "prompt_template_video", None)
            if isinstance(tmpl, dict):
                cs = int(tmpl.get("crop_start", 0) or 0)
                crop_start = cs if cs > 0 else 0
        except Exception:
            crop_start = 0

        n = min(len(full_ids), len(strip_ids))
        common = 0
        while common < n and full_ids[common] == strip_ids[common]:
            common += 1

        action_start = common - crop_start
        if action_start < 0:
            action_start = 0
        if action_start > text_len - 1:
            action_start = text_len - 1
        return int(action_start)

    def _resolve_action_attention_option(self, name, default=None):
        """Resolve a shared train/inference action-attention option."""
        for obj in (
            getattr(self.args, "inference", None),
            self.args,
            getattr(self.args, "training", None),
            getattr(self.args, "dataset", None),
        ):
            if obj is not None and getattr(obj, name, None) is not None:
                return getattr(obj, name)
        return default

    def _cross_segment_text_head_only_enabled(self) -> bool:
        """Whether other segments may see only a text block's non-action head."""
        return bool(
            self._resolve_action_attention_option(
                "cross_segment_text_head_only",
                True,
            )
        )

    def _intra_seg_enabled(self) -> bool:
        """Whether inference-side intra-segment per-action masking is active.

        Requires the training flag ``intra_segment_action_mask`` AND
        ``json_prompt_normalize_action_timestamp`` (which rebases each action's
        start_time/end_time to the event-local timeline; ``contained_actions``
        stays a plain list), matching the training-side gating in
        ``WM1MTrainingDataset``.
        """
        return bool(
            self._resolve_action_attention_option("intra_segment_action_mask")
        ) and bool(
            self._resolve_action_attention_option(
                "json_prompt_normalize_action_timestamp"
            )
        )

    def _prev_seg_last_frame_only(self) -> bool:
        """Whether segment j's video AND text queries attend only the FINAL
        latent frame of each previous segment's video (video K-ranges only;
        text->text unchanged). Mirrors the training flag
        ``prev_seg_last_frame_only`` (checked flat / under ``training`` /
        ``dataset``), matching how ``_intra_seg_enabled`` resolves its flags.
        """

        def _resolve(name):
            v = getattr(self.args, name, None)
            if v is not None:
                return v
            tr = getattr(self.args, "training", None)
            if tr is not None and getattr(tr, name, None) is not None:
                return getattr(tr, name)
            ds = getattr(self.args, "dataset", None)
            if ds is not None and getattr(ds, name, None) is not None:
                return getattr(ds, name)
            return None

        return bool(_resolve("prev_seg_last_frame_only"))

    def _compute_action_text_offsets(self, prompt: str) -> Optional[List[int]]:
        """Per-action token offsets ``[o_0, ..., o_A]`` within the segment text.

        Mirrors ``WM1MTrainingDataset._compute_text_action_offsets``: build JSON
        byte-prefix variants of the segment truncating ``contained_actions`` to
        keep only actions ``0..k-1``, then token-common-prefix each against the
        full segment text. ``o_0`` is the head end (no actions), and action ``k``
        occupies ``[o_k, o_{k+1})`` with ``o_A`` the valid tail end.

        Returns ``None`` when the segment carries no ``contained_actions`` (legacy
        whole-block visibility). ``contained_actions`` is a plain list of actions
        with event-local timestamps (see ``json_prompt_normalize_action_timestamp``).
        """
        import json as _json

        try:
            obj = _json.loads(prompt)
        except Exception:
            return None
        segs = obj.get("segments") if isinstance(obj, dict) else None
        if not (isinstance(segs, list) and segs and isinstance(segs[0], dict)):
            return None
        seg0 = segs[0]
        acts = seg0.get("contained_actions")
        # Ordered-action dict -> {"<|Action1|>": {...}, ...}; recover ordered list.
        if isinstance(acts, dict) and acts:
            act_items = list(acts.items())
            num_actions = len(act_items)
        elif isinstance(acts, list) and acts:
            act_items = None
            num_actions = len(acts)
        else:
            return None

        def _variant_with_k_actions(k: int) -> str:
            seg_k = dict(seg0)
            if k <= 0:
                seg_k.pop("contained_actions", None)
            elif act_items is not None:
                seg_k["contained_actions"] = {key: val for key, val in act_items[:k]}
            else:
                seg_k["contained_actions"] = acts[:k]
            new_obj = {
                key: ([seg_k] if key == "segments" else val) for key, val in obj.items()
            }
            return _json.dumps(new_obj, ensure_ascii=False)

        text_len = getattr(self.args, "text_len", 256)
        full_ids = (
            self.text_encoder.text2tokens(
                [prompt],
                data_type="video",
                max_length=text_len,
            )["input_ids"]
            .squeeze(0)
            .tolist()
        )

        crop_start = 0
        try:
            tmpl = getattr(self.text_encoder, "prompt_template_video", None)
            if isinstance(tmpl, dict):
                cs = int(tmpl.get("crop_start", 0) or 0)
                crop_start = cs if cs > 0 else 0
        except Exception:
            crop_start = 0

        def _prefix_offset(variant_text: str) -> int:
            v_ids = (
                self.text_encoder.text2tokens(
                    [variant_text],
                    data_type="video",
                    max_length=text_len,
                )["input_ids"]
                .squeeze(0)
                .tolist()
            )
            n = min(len(full_ids), len(v_ids))
            common = 0
            while common < n and full_ids[common] == v_ids[common]:
                common += 1
            off = common - crop_start
            if off < 0:
                off = 0
            if off > text_len - 1:
                off = text_len - 1
            return int(off)

        offsets = [
            _prefix_offset(_variant_with_k_actions(k)) for k in range(num_actions)
        ]
        cleaned = []
        prev = 0
        for o in offsets:
            o = max(o, prev)
            o = min(o, text_len)
            cleaned.append(o)
            prev = o
        # Append the tail end (== valid text length) as the final boundary.
        full_mask = self.text_encoder.text2tokens(
            [prompt],
            data_type="video",
            max_length=text_len,
        ).get("attention_mask", None)
        if full_mask is not None:
            valid = int(full_mask.squeeze(0).sum().item())
        else:
            valid = text_len
        cleaned.append(min(valid, text_len))
        return cleaned

    def _compute_action_video_windows(
        self,
        prompt: str,
        seg_latent_start: int,
        seg_latent_end: int,
    ) -> Optional[List[List[int]]]:
        """Segment-relative per-action video sub-windows ``[vs, ve, aidx]``.

        Mirrors ``WM1MTrainingDataset._compute_action_video_boundaries`` for one
        segment: map each action's relative ``start_time``/``end_time`` to latent
        frames (``pixel = int(t * fps)``, ``latent = pixel // t_ratio``), clamp to
        ``[0, seg_len)`` (SEGMENT-RELATIVE — the cache attention offsets video
        windows from the current chunk's local frame 0), and tile the segment
        with gaps (``aidx == -1``). Timestamps in the normalized-action prompt
        are already event-relative, so no event-start subtraction is needed.

        Returns ``None`` when the segment has no usable actions (legacy whole
        video visibility). Raises ``ValueError`` on overlapping actions.
        """
        import json as _json

        seg_len = int(seg_latent_end) - int(seg_latent_start)
        if seg_len <= 0:
            return None
        from hymm.inference import _parse_timestamp

        try:
            obj = _json.loads(prompt)
        except Exception:
            return None
        segs = obj.get("segments") if isinstance(obj, dict) else None
        if not (isinstance(segs, list) and segs and isinstance(segs[0], dict)):
            return None
        acts = segs[0].get("contained_actions")
        if isinstance(acts, dict):
            action_list = list(acts.values())
        elif isinstance(acts, list):
            action_list = acts
        else:
            return None
        if not action_list:
            return None

        inf_cfg = getattr(self.args, "inference", None)
        fps = None
        if inf_cfg is not None:
            fps = getattr(inf_cfg, "video_fps", None)
        if fps is None:
            fps = getattr(self.args, "video_fps", 16)
        fps = float(fps)
        t_ratio = float(getattr(self, "t_ratio", 4) or 4)

        def _time_to_latent(t: float) -> int:
            t = max(t, 0.0)
            pixel_frame = int(t * fps)
            return pixel_frame // int(t_ratio)

        spans = []  # (vstart, vend, action_idx) in segment-relative latent frames
        for k, a in enumerate(action_list):
            if not isinstance(a, dict) or "start_time" not in a or "end_time" not in a:
                return None
            a_s = _time_to_latent(_parse_timestamp(a["start_time"]))
            a_e = _time_to_latent(_parse_timestamp(a["end_time"]))
            vs = max(0, min(a_s, seg_len))
            ve = max(0, min(a_e, seg_len))
            if ve > vs:
                spans.append([vs, ve, k])
        if not spans:
            return None

        spans.sort(key=lambda s: (s[0], s[1]))
        for p in range(1, len(spans)):
            if spans[p][0] < spans[p - 1][1]:
                raise ValueError(
                    f"Overlapping action video windows {spans[p - 1]} and "
                    f"{spans[p]} not supported (segment prompt: {prompt[:80]})"
                )

        windows: List[List[int]] = []
        cursor = 0
        for vs, ve, k in spans:
            if vs > cursor:
                windows.append([cursor, vs, -1])
            windows.append([vs, ve, k])
            cursor = ve
        if cursor < seg_len:
            windows.append([cursor, seg_len, -1])

        # Degenerate single-action-covers-segment case == legacy -> skip.
        if len(windows) == 1 and windows[0][2] == 0 and len(action_list) == 1:
            return None
        return windows
