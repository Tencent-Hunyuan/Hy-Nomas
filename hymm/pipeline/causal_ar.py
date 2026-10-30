"""Segment / chunk-level KV-cache autoregressive inference pipeline (seg_ar / chunk_ar)."""

import math
import os
import time
from typing import Callable, List, Optional

import torch
from hymm.utils.inference_metrics import stage
from loguru import logger
from omegaconf import ListConfig

from hymm.pipeline.base import BasePipeline, align_to
from hymm.pipeline.bidirectional import get_cfg_scale_for_step
from hymm.pipeline.solvers.euler import EulerSolver
from hymm.utils.camera_cfg import (
    camera_guidance_is_active,
    combine_camera_text_cfg,
)
from hymm.distributed.parallel_states import (
    get_sequence_parallel_state,
    get_model_parallel_info,
    nccl_info,
)
from hymm.utils.parallel import sync_random_states
from hymm.models.dit.modules.uniform_history_cache import (
    ARCHIVED_HISTORY_SAMPLINGS, HISTORY_SAMPLINGS, validate_swa_streaming_rollback,
)


def _resolve_pixelumm_stable_prefix_streaming(
    *,
    requested: bool,
    segment_vae_decode: bool,
    vae_name: Optional[str],
) -> bool:
    """Validate and resolve the per-request PixelUMM replace-tail decode policy."""
    if vae_name == "pixelumm" and segment_vae_decode and not requested:
        raise ValueError(
            "model.vae.name=pixelumm with "
            "inference.seg_ar_segment_vae_decode=true requires "
            "model.vae.stable_prefix_streaming=true"
        )
    if not requested:
        return False
    if vae_name != "pixelumm":
        raise ValueError(
            "model.vae.stable_prefix_streaming=true requires "
            "model.vae.name=pixelumm"
        )
    if not segment_vae_decode:
        raise ValueError(
            "model.vae.stable_prefix_streaming=true requires "
            "inference.seg_ar_segment_vae_decode=true"
        )
    return True


class PhaseSwappedKVCache(list):
    """Move exact full-layer KV between pinned CPU memory and GPU by phase.

    The transformer sees the original list-of-tensors cache interface, so its
    attention path and BF16 values remain unchanged. KV is resident on GPU
    during denoising/cache writes and parked in preallocated host buffers while
    PixelUMM decodes the generated latent.
    """

    def __init__(
        self,
        *,
        num_blocks: int,
        batch_size: int,
        max_cache_tokens: int,
        cache_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.device = torch.device(device)
        self.capacity = int(max_cache_tokens)
        self.dtype = dtype
        self.cache_heads = int(cache_heads)
        self.head_dim = int(head_dim)
        self._pin_memory = True

        def host_empty(shape, *, tensor_dtype):
            try:
                return torch.empty(
                    shape,
                    dtype=tensor_dtype,
                    device="cpu",
                    pin_memory=True,
                )
            except RuntimeError:
                self._pin_memory = False
                return torch.empty(
                    shape,
                    dtype=tensor_dtype,
                    device="cpu",
                )

        shape = [
            batch_size,
            self.capacity,
            self.cache_heads,
            self.head_dim,
        ]
        self._host_blocks = [
            {
                "k": host_empty(shape, tensor_dtype=dtype),
                "v": host_empty(shape, tensor_dtype=dtype),
            }
            for _ in range(num_blocks)
        ]
        self._host_masks = [
            host_empty(
                [batch_size, self.capacity],
                tensor_dtype=torch.bool,
            ).zero_()
            for _ in range(num_blocks)
        ]
        super().__init__(
            [{"k": block["k"], "v": block["v"]} for block in self._host_blocks]
        )
        self.cache_masks = list(self._host_masks)
        self.active_device = torch.device("cpu")
        self.active_capacity = self.capacity

    @property
    def pin_memory_enabled(self) -> bool:
        return self._pin_memory

    @staticmethod
    def _copy_mask_metadata(
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        ranges = getattr(source, "_action_tail_ranges", None)
        if ranges is not None:
            target._action_tail_ranges = list(ranges)
        elif hasattr(target, "_action_tail_ranges"):
            target._action_tail_ranges = []

    def restore(
        self,
        *,
        required_tokens: int,
        valid_tokens: int,
    ) -> int:
        required_tokens = max(1, int(required_tokens))
        valid_tokens = int(valid_tokens)
        if (
            self.active_device.type == "cuda"
            and self.active_capacity >= required_tokens
        ):
            return 0
        if self.active_device.type == "cuda":
            self.offload(valid_tokens=valid_tokens)

        active_capacity = min(self.capacity, required_tokens)
        shape = [
            self._host_blocks[0]["k"].shape[0],
            active_capacity,
            self.cache_heads,
            self.head_dim,
        ]
        gpu_blocks = [
            {
                "k": torch.empty(
                    shape,
                    dtype=self.dtype,
                    device=self.device,
                ),
                "v": torch.empty(
                    shape,
                    dtype=self.dtype,
                    device=self.device,
                ),
            }
            for _ in range(len(self))
        ]
        gpu_masks = [
            torch.empty(
                [shape[0], active_capacity],
                dtype=torch.bool,
                device=self.device,
            )
            for _ in range(len(self))
        ]
        stream = torch.cuda.Stream(device=self.device)
        with torch.cuda.stream(stream):
            if valid_tokens > 0:
                for index, gpu_cache in enumerate(gpu_blocks):
                    gpu_cache["k"][:, :valid_tokens].copy_(
                        self._host_blocks[index]["k"][:, :valid_tokens],
                        non_blocking=self._pin_memory,
                    )
                    gpu_cache["v"][:, :valid_tokens].copy_(
                        self._host_blocks[index]["v"][:, :valid_tokens],
                        non_blocking=self._pin_memory,
                    )
                    gpu_masks[index][:, :valid_tokens].copy_(
                        self._host_masks[index][:, :valid_tokens],
                        non_blocking=self._pin_memory,
                    )
            event = torch.cuda.Event()
            event.record(stream)
        event.synchronize()
        for index in range(len(self)):
            self[index] = gpu_blocks[index]
            self._copy_mask_metadata(
                self._host_masks[index],
                gpu_masks[index],
            )
        self.cache_masks = gpu_masks
        self.active_device = self.device
        self.active_capacity = active_capacity
        return sum(
            tensor.numel() * tensor.element_size()
            for cache in gpu_blocks
            for tensor in cache.values()
        )

    def offload(self, *, valid_tokens: int) -> int:
        valid_tokens = int(valid_tokens)
        if self.active_device.type == "cpu":
            return 0

        stream = torch.cuda.Stream(device=self.device)
        moved_bytes = 0
        with torch.cuda.stream(stream):
            if valid_tokens > 0:
                for index, gpu_cache in enumerate(self):
                    for key in ("k", "v"):
                        self._host_blocks[index][key][:, :valid_tokens].copy_(
                            gpu_cache[key][:, :valid_tokens],
                            non_blocking=self._pin_memory,
                        )
                        moved_bytes += (
                            gpu_cache[key][:, :valid_tokens].numel()
                            * gpu_cache[key].element_size()
                        )
                    self._host_masks[index][:, :valid_tokens].copy_(
                        self.cache_masks[index][:, :valid_tokens],
                        non_blocking=self._pin_memory,
                    )
                event = torch.cuda.Event()
                event.record(stream)
            else:
                event = None
        if event is not None:
            event.synchronize()
        for index in range(len(self)):
            self._copy_mask_metadata(
                self.cache_masks[index],
                self._host_masks[index],
            )
            self[index] = self._host_blocks[index]
        self.cache_masks = list(self._host_masks)
        self.active_device = torch.device("cpu")
        self.active_capacity = self.capacity
        return moved_bytes

    def reset(self) -> None:
        self.offload(valid_tokens=0)
        for mask in self._host_masks:
            mask.zero_()
            if hasattr(mask, "_action_tail_ranges"):
                mask._action_tail_ranges = []

    def grow(self, required_tokens: int, valid_tokens: int) -> None:
        if self.capacity >= required_tokens:
            return
        self.offload(valid_tokens=valid_tokens)
        new_capacity = int(required_tokens)
        for block in self._host_blocks:
            for key in ("k", "v"):
                old = block[key]
                grown = torch.empty(
                    [old.shape[0], new_capacity, *old.shape[2:]],
                    dtype=old.dtype,
                    device="cpu",
                    pin_memory=self._pin_memory,
                )
                grown[:, :valid_tokens].copy_(old[:, :valid_tokens])
                block[key] = grown
        for index, old in enumerate(self._host_masks):
            grown = torch.zeros(
                [old.shape[0], new_capacity],
                dtype=old.dtype,
                device="cpu",
                pin_memory=self._pin_memory,
            )
            grown[:, :valid_tokens].copy_(old[:, :valid_tokens])
            self._copy_mask_metadata(old, grown)
            self._host_masks[index] = grown
        for index in range(len(self)):
            self[index] = self._host_blocks[index]
        self.cache_masks = list(self._host_masks)
        self.capacity = new_capacity
        self.active_capacity = new_capacity


class CPTCausalInferencePipeline(BasePipeline):
    """
    Segment / chunk-level KV-cache autoregressive inference for CPT models.

    Two rollout modes share this pipeline:
      - seg_ar:   Segment-level AR. Each segment is denoised as a whole.
      - chunk_ar: Chunk-level AR. Each segment is further split into chunks
                  of `inference_chunk_size` latent frames. Requires
                  inference_chunk_size > 0.
    """

    def __init__(
        self,
        args,
        device: torch.device,
        dit_model_path: str,
        world_mesh=None,
    ):
        logger.info("Initializing CPT KV-Cache AR Pipeline...")
        super().__init__(
            args, device, dit_model_path, world_mesh=world_mesh
        )

        self.hidden_size = self.transformer.hidden_size
        self.heads_num = self.transformer.heads_num
        self.num_double_blocks = len(self.transformer.double_blocks)
        self.patch_size = self.transformer.patch_size
        self.in_channels = self.transformer.config.in_channels

        self.max_cache_segments = getattr(args, "max_cache_segments", None)
        self.kv_cache = None
        self.cache_valid_len = 0
        self.global_frame_counter = 0
        self._interactive_cached_action_tails = []
        self._interactive_cached_seg_video_ranges = []
        self._interactive_segment_count = 0
        self._interactive_segment_cache_tokens = []
        self._interactive_segment_latent_lengths = []
        self._interactive_noise_bank = None
        self._interactive_noise_generator = None
        self._interactive_noise_seed = None
        self._interactive_segment_noise_rng_states = []
        self._interactive_pixelumm_decode_session = None
        self._last_pixelumm_streaming_metadata = None
        self._interactive_pixelumm_offloaded = False

        guidance_scale = getattr(args, "cfg_scale_video", 6.0)
        self.do_classifier_free_guidance = guidance_scale > 1.0

        self.denoise_mode = getattr(args, "denoise_mode", "ode")
        self.cache_at_step = getattr(args, "cache_at_step", -1)
        inf_cfg = getattr(args, "inference", None)
        from hymm.models.dit.modules.kv_cache_storage import quantization_mode, validate_kv_cache_config

        validate_kv_cache_config(args)
        self.kv_storage_format = quantization_mode(inf_cfg)
        self.int8_kv_storage = self.kv_storage_format == "int8"
        self.qvg_kv_storage = self.kv_storage_format == "qvg"
        self.cache_noise_sigma = float(
            getattr(
                inf_cfg,
                "cache_noise_sigma",
                getattr(args, "cache_noise_sigma", 0.0),
            )
            or 0.0
        )
        if not 0.0 <= self.cache_noise_sigma <= 1.0:
            raise ValueError(
                f"cache_noise_sigma must be in [0, 1], got {self.cache_noise_sigma}"
            )
        if self.cache_noise_sigma > 0.0 and self.cache_at_step >= 0:
            raise ValueError(
                "cache_noise_sigma and cache_at_step are mutually exclusive "
                f"(got cache_noise_sigma={self.cache_noise_sigma}, "
                f"cache_at_step={self.cache_at_step}); cache_at_step already "
                "snapshots a partially-denoised latent."
            )
        self.use_global_camera_rope_offset = bool(
            getattr(inf_cfg, "use_global_camera_rope_offset", False)
        )

        logger.info(
            f"Pipeline: {self.num_double_blocks} double blocks, "
            f"hidden={self.hidden_size}, heads={self.heads_num}"
        )
        logger.info(f"CFG: {self.do_classifier_free_guidance}, scale={guidance_scale}")
        if self.cache_at_step >= 0:
            logger.info(f"Cache at intermediate denoise step: {self.cache_at_step}")
        if self.cache_noise_sigma > 0.0:
            logger.info(
                f"Cache noise: sigma={self.cache_noise_sigma:.6f} "
                f"(t={self.cache_noise_sigma * 1000:.1f}) applied to each "
                f"finished segment before the KV write"
            )
        logger.info(
            f"Camera RoPE offset: "
            f"{'global_frame_counter' if self.use_global_camera_rope_offset else 'local'}"
        )

    def prewarm_interactive_kv_cache(self) -> bool:
        """Allocate fixed phase-swapped CPU KV storage before ready."""
        if not bool(
            getattr(
                self.args.inference,
                "interactive_kv_cache_layerwise_cpu_offload",
                False,
            )
        ):
            return False
        cpu_mode = (
            str(
                getattr(
                    self.args.inference,
                    "interactive_kv_cache_cpu_mode",
                    "",
                )
            )
            .strip()
            .lower()
        )
        if cpu_mode != "phase_swap":
            raise ValueError(
                "interactive CPU KV offload supports only explicit "
                "inference.interactive_kv_cache_cpu_mode=phase_swap"
            )
        video_size = getattr(self.args.inference, "video_size", None)
        if not isinstance(video_size, (list, tuple, ListConfig)) or not video_size:
            raise ValueError("interactive KV prewarm requires inference.video_size")
        target_height = align_to(int(video_size[0]), 16)
        target_width = align_to(
            int(video_size[1] if len(video_size) > 1 else video_size[0]),
            16,
        )
        latent_h = target_height // self.s_ratio
        latent_w = target_width // self.s_ratio
        tokens_per_frame = (latent_h // self.patch_size[1]) * (
            latent_w // self.patch_size[2]
        )
        text_len = int(getattr(self.args, "text_len", 256))
        max_cache_tokens = self._interactive_kv_preallocate_tokens(
            tokens_per_frame=tokens_per_frame,
            text_len=text_len,
        )
        if max_cache_tokens <= 0:
            return False
        started = time.perf_counter()
        self._initialize_kv_cache(
            batch_size=1,
            max_cache_tokens=max_cache_tokens,
            dtype=self.target_dtype,
            device=self.device,
        )
        if not isinstance(self.kv_cache, PhaseSwappedKVCache):
            raise RuntimeError("phase-swapped KV prewarm was not initialized")
        self.kv_cache.offload(valid_tokens=0)
        self._offload_interactive_pixelumm()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(
            "Interactive phase-swapped CPU KV cache prewarmed before worker "
            f"ready: tokens={max_cache_tokens}, "
            f"elapsed={time.perf_counter() - started:.2f}s"
        )
        return True

    def _move_interactive_pixelumm_session_state(
        self,
        device: torch.device | str,
    ) -> None:
        session = self._interactive_pixelumm_decode_session
        if session is not None:
            session._move_decode_state(device)

    def _restore_interactive_pixelumm(self) -> None:
        if getattr(self.vae, "_model_cpu_offload", None) is not None:
            return
        if not self._interactive_pixelumm_offloaded:
            return
        started = time.perf_counter()
        self.vae.to(device=self.device)
        self._move_interactive_pixelumm_session_state(self.device)
        self._interactive_pixelumm_offloaded = False
        logger.info(
            "Restored PixelUMM decoder for decode phase in "
            f"{time.perf_counter() - started:.3f}s"
        )

    def _offload_interactive_pixelumm(self) -> None:
        if getattr(self.vae, "_model_cpu_offload", None) is not None:
            return
        if self._interactive_pixelumm_offloaded or not getattr(
            self.vae, "handles_distributed_decode", False
        ):
            return
        started = time.perf_counter()
        self._move_interactive_pixelumm_session_state("cpu")
        self.vae.to(device="cpu")
        self._interactive_pixelumm_offloaded = True
        logger.info(
            "Offloaded PixelUMM decoder between decode phases in "
            f"{time.perf_counter() - started:.3f}s"
        )

    def _prepare_interactive_decode_phase(self) -> int:
        if not isinstance(self.kv_cache, PhaseSwappedKVCache):
            return 0
        released_bytes = self.kv_cache.offload(valid_tokens=self.cache_valid_len)
        self.cache_mask_list = self.kv_cache.cache_masks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._restore_interactive_pixelumm()
        return released_bytes

    def _finish_interactive_decode_phase(self) -> None:
        if not isinstance(self.kv_cache, PhaseSwappedKVCache):
            return
        self._offload_interactive_pixelumm()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _initialize_kv_cache(self, batch_size, max_cache_tokens, dtype, device):
        qvg_storage = getattr(self, "qvg_kv_storage", False)
        head_dim = self.hidden_size // self.heads_num
        head_shards = nccl_info.sp_size * nccl_info.tp_size
        if self.heads_num % head_shards:
            raise ValueError("KV cache heads must be divisible by sp_size * tp_size")
        cache_heads = self.heads_num // head_shards

        cpu_offload = bool(
            getattr(
                self.args.inference,
                "interactive_kv_cache_layerwise_cpu_offload",
                False,
            )
        )
        cpu_mode = (
            str(
                getattr(
                    self.args.inference,
                    "interactive_kv_cache_cpu_mode",
                    "",
                )
            )
            .strip()
            .lower()
        )
        if cpu_offload and cpu_mode != "phase_swap":
            raise ValueError(
                "interactive CPU KV offload supports only explicit "
                "inference.interactive_kv_cache_cpu_mode=phase_swap"
            )
        if cpu_offload:
            if self.int8_kv_storage or qvg_storage:
                raise ValueError("Compressed KV cannot use PhaseSwappedKVCache")
            reusable = (
                isinstance(self.kv_cache, PhaseSwappedKVCache)
                and self.kv_cache.capacity >= max_cache_tokens
                and self.kv_cache[0]["k"].shape[0] == batch_size
                and self.kv_cache.cache_heads == cache_heads
                and self.kv_cache.head_dim == head_dim
                and self.kv_cache.dtype == dtype
            )
            if reusable:
                self.kv_cache.reset()
            else:
                self.kv_cache = PhaseSwappedKVCache(
                    num_blocks=self.num_double_blocks,
                    batch_size=batch_size,
                    max_cache_tokens=max_cache_tokens,
                    cache_heads=cache_heads,
                    head_dim=head_dim,
                    dtype=dtype,
                    device=device,
                )
            self.cache_mask_list = self.kv_cache.cache_masks
        else:
            if qvg_storage:
                from hymm.models.dit.modules.qvg_kv_cache import QVGOptions, make_qvg_kv_block

                options = QVGOptions.from_config(getattr(self.args.inference, "qvg_kv_cache", {}))
                self.kv_cache = [
                    make_qvg_kv_block(batch_size, max_cache_tokens, cache_heads, head_dim,
                                      device=device, options=options, layer=layer)
                    for layer in range(self.num_double_blocks)
                ]
            elif self.int8_kv_storage:
                from hymm.models.dit.modules.int8_kv_cache import make_int8_kv_block
                self.kv_cache = [
                    make_int8_kv_block(batch_size, max_cache_tokens, cache_heads, head_dim, device=device)
                    for _ in range(self.num_double_blocks)
                ]
            else:
                self.kv_cache = [
                    {
                        # Unwritten cache capacity is never read: every attention
                        # path slices to cache_valid_len.
                        "k": torch.empty(
                            [batch_size, max_cache_tokens, cache_heads, head_dim],
                            dtype=dtype,
                            device=device,
                        ),
                        "v": torch.empty(
                            [batch_size, max_cache_tokens, cache_heads, head_dim],
                            dtype=dtype,
                            device=device,
                        ),
                    }
                    for _ in range(self.num_double_blocks)
                ]
            self.cache_mask_list = [
                torch.zeros(
                    [batch_size, max_cache_tokens],
                    dtype=torch.bool,
                    device=device,
                )
                for _ in range(self.num_double_blocks)
            ]
        self.cache_valid_len = 0
        self.global_frame_counter = 0
        self.segment_token_lengths = []
        logger.info(
            f"KV cache initialized: {self.num_double_blocks} blocks, "
            f"max_tokens={max_cache_tokens}, heads={cache_heads}, "
            f"format={'qvg' if qvg_storage else 'int8' if self.int8_kv_storage else 'none'}, "
            f"storage={'cpu-phase_swap' if cpu_offload else 'gpu'}, "
            f"pinned={getattr(self.kv_cache, 'pin_memory_enabled', False)}"
        )

    def _ensure_kv_cache_capacity(self, required_tokens: int) -> None:
        """Grow persistent history storage, excluding SWA's current-segment KV."""
        if self.kv_cache is None or self.cache_mask_list is None:
            raise RuntimeError("KV cache must be initialized before it can grow")
        if isinstance(self.kv_cache, PhaseSwappedKVCache):
            if self.kv_cache.capacity < required_tokens:
                old_capacity = self.kv_cache.capacity
                self.kv_cache.grow(required_tokens, self.cache_valid_len)
                logger.info(
                    "Persistent phase-swapped CPU KV backing grown: "
                    f"{old_capacity} -> {self.kv_cache.capacity} tokens"
                )
            return
        from hymm.models.dit.modules.kv_cache_storage import cache_geometry
        current_capacity = int(cache_geometry(self.kv_cache[0])[0][1])
        if current_capacity >= required_tokens:
            return
        if "swa_capacity" in self.kv_cache[0]:
            # Only the selected segments' text budget can grow between turns;
            # geometric over-allocation would exceed the bounded SWA capacity.
            new_capacity = required_tokens
        else:
            growth_factor = float(
                getattr(
                    self.args.inference,
                    "interactive_kv_cache_growth_factor",
                    1.5,
                )
            )
            if growth_factor < 1.0:
                raise ValueError(
                    "inference.interactive_kv_cache_growth_factor must be >= 1.0"
                )
            new_capacity = max(
                required_tokens,
                int(math.ceil(current_capacity * growth_factor)),
            )
        for cache in self.kv_cache:
            if cache.get("storage_format") == "qvg":
                cache["shape"] = (cache["shape"][0], new_capacity, *cache["shape"][2:])
            for key in ("k", "v", "k_scale", "v_scale"):
                if key not in cache:
                    continue
                old = cache[key]
                grown = torch.empty(
                    [old.shape[0], new_capacity, *old.shape[2:]],
                    dtype=old.dtype,
                    device=old.device,
                )
                grown[:, : self.cache_valid_len].copy_(old[:, : self.cache_valid_len])
                cache[key] = grown
        for index, old in enumerate(self.cache_mask_list):
            grown = torch.zeros(
                [old.shape[0], new_capacity],
                dtype=old.dtype,
                device=old.device,
            )
            grown[:, : self.cache_valid_len].copy_(old[:, : self.cache_valid_len])
            PhaseSwappedKVCache._copy_mask_metadata(old, grown)
            self.cache_mask_list[index] = grown
        logger.info(
            f"Persistent KV cache grown: {current_capacity} -> {new_capacity} tokens"
        )

    def _interactive_kv_preallocate_tokens(
        self,
        *,
        tokens_per_frame: int,
        text_len: int,
    ) -> int:
        """Return fixed cache capacity for the configured interactive horizon.

        The cache stores one video-token row per latent frame plus one text
        block per prompt turn. The text-turn upper bound uses PixelUMM's minimum
        aligned segment duration, so arbitrary normal Web turns within the
        horizon do not trigger a realloc/copy. This changes capacity only, never
        valid cache contents or attention math.
        """
        seconds = float(
            getattr(
                self.args.inference,
                "interactive_kv_cache_preallocate_seconds",
                0.0,
            )
            or 0.0
        )
        if seconds <= 0:
            return 0
        fps = int(getattr(self.args.inference, "video_fps", 24))
        pixel_frames = max(1, int(math.ceil(seconds * fps)))
        latent_frames = (pixel_frames - 1) // 4 + 1
        minimum_segment_frames = int(
            getattr(
                self.args.inference,
                "interactive_kv_cache_min_segment_frames",
                80,
            )
            or 80
        )
        maximum_segments = max(
            1,
            int(math.ceil(pixel_frames / max(minimum_segment_frames, 1))),
        )
        return latent_frames * int(tokens_per_frame) + maximum_segments * int(text_len)

    @torch.inference_mode()
    def rollback_last_interactive_segment(self) -> dict:
        """Discard the most recently cached interactive segment in-place."""
        if (
            self.kv_cache is None
            or self.cache_mask_list is None
            or self._interactive_segment_count <= 0
            or not self._interactive_segment_cache_tokens
            or not self._interactive_segment_latent_lengths
            or not self._interactive_segment_noise_rng_states
        ):
            raise RuntimeError("there is no interactive segment to roll back")
        validate_swa_streaming_rollback(self.kv_cache)
        if any(len(items) != self._interactive_segment_count for items in (
            self._interactive_segment_cache_tokens,
            self._interactive_segment_latent_lengths,
            self._interactive_segment_noise_rng_states,
        )):
            raise RuntimeError("interactive rollback bookkeeping is inconsistent")
        token_count = int(self._interactive_segment_cache_tokens[-1])
        latent_frames = int(self._interactive_segment_latent_lengths[-1])
        noise_rng_state, noise_capacity = self._interactive_segment_noise_rng_states[-1]
        if latent_frames <= 0 or latent_frames > self.global_frame_counter:
            raise RuntimeError("invalid interactive rollback frame count")
        if (self._interactive_noise_bank is None or self._interactive_noise_generator is None
                or not self.global_frame_counter - latent_frames <= noise_capacity
                <= self._interactive_noise_bank.shape[2]):
            raise RuntimeError("interactive rollback noise timeline is inconsistent")
        pixelumm_session = getattr(
            self,
            "_interactive_pixelumm_decode_session",
            None,
        )
        if pixelumm_session is not None and (
            pixelumm_session.finished
            or len(pixelumm_session.push_latent_frames) != self._interactive_segment_count
            or pixelumm_session.push_latent_frames[-1] != latent_frames
            or pixelumm_session.available_latent_frames != self.global_frame_counter
        ):
            raise RuntimeError("interactive PixelUMM timeline does not match the segment to roll back")

        old_valid_len = int(self.cache_valid_len)
        archived = self.kv_cache[0].get("swa_sampling") in ARCHIVED_HISTORY_SAMPLINGS
        if len(self.kv_cache) != len(self.cache_mask_list):
            raise RuntimeError("interactive KV layers and masks disagree")
        if archived:
            from hymm.models.dit.modules.uniform_history_cache import (
                apply_history_rollback, prepare_history_rollback, selected_action_tails,
            )
            # Validate every layer before dropping any archive or RNG record.
            rollbacks = [
                prepare_history_rollback(
                    cache, mask, segment_index=self._interactive_segment_count - 1,
                    latent_frames=latent_frames, total_frames=self.global_frame_counter,
                )
                for cache, mask in zip(self.kv_cache, self.cache_mask_list)
            ]
            first = rollbacks[0]
            if any((plan.frame_ids, plan.valid_tokens, plan.removed_tokens, plan.tokens_per_frame)
                   != (first.frame_ids, first.valid_tokens, first.removed_tokens, first.tokens_per_frame)
                   for plan in rollbacks):
                raise RuntimeError("interactive SWA rollback selection disagrees across layers")
            new_valid_len = first.valid_tokens
            token_count = first.removed_tokens
        else:
            if any("swa_capacity" in cache for cache in self.kv_cache):
                raise ValueError("Streaming SWA rollback requires an archived history sampling strategy")
            if token_count <= 0 or token_count > old_valid_len:
                raise RuntimeError(
                    "invalid interactive cache rollback size: "
                    f"tokens={token_count}, cache_valid_len={old_valid_len}"
                )
            removed_tokens = removed_chunks = 0
            for count in reversed(self.segment_token_lengths):
                removed_tokens += int(count)
                removed_chunks += 1
                if removed_tokens >= token_count:
                    break
            if removed_tokens != token_count:
                raise RuntimeError("interactive cache bookkeeping is inconsistent")
            new_valid_len = old_valid_len - token_count

        if pixelumm_session is not None:
            pixelumm_session.rollback_last_push()
        if archived:
            for cache, mask, plan in zip(self.kv_cache, self.cache_mask_list, rollbacks):
                apply_history_rollback(cache, mask, plan)
            self._interactive_cached_action_tails = selected_action_tails(
                self.kv_cache[0]["swa_uniform_layout"], self.cache_mask_list[0],
            )
            self._interactive_cached_seg_video_ranges = []
        else:
            for cache in self.kv_cache:
                for key in ("k", "v", "k_scale", "v_scale"):
                    if key in cache:
                        cache[key][:, new_valid_len:old_valid_len].zero_()
            for cache_mask in self.cache_mask_list:
                cache_mask[:, new_valid_len:old_valid_len].zero_()
            del self.segment_token_lengths[-removed_chunks:]

        self._interactive_segment_cache_tokens.pop()
        self._interactive_segment_latent_lengths.pop()
        self._interactive_segment_noise_rng_states.pop()

        self.cache_valid_len = new_valid_len
        self.global_frame_counter = max(
            0, int(self.global_frame_counter) - latent_frames
        )
        if self._interactive_noise_generator is not None:
            self._interactive_noise_generator.set_state(noise_rng_state)
        if self._interactive_noise_bank.shape[2] > noise_capacity:
            self._interactive_noise_bank = self._interactive_noise_bank[:, :, :noise_capacity].contiguous()
        self._interactive_segment_count -= 1
        remaining_segments = int(self._interactive_segment_count)
        self._interactive_cached_action_tails = [
            item
            for item in self._interactive_cached_action_tails
            if int(item[0]) < remaining_segments
        ]
        self._interactive_cached_seg_video_ranges = [
            item
            for item in self._interactive_cached_seg_video_ranges
            if int(item[0]) < remaining_segments
        ]
        if pixelumm_session is not None:
            self._last_pixelumm_streaming_metadata = {
                "stable_pixel_frames": int(pixelumm_session.stable_pixel_frames),
                "provisional_pixel_frames": int(pixelumm_session.withheld_pixel_frames),
                "withheld_pixel_frames": 0,
                "replace_pixel_frames": 0,
                "generated_pixel_frames": int(pixelumm_session.available_pixel_frames),
            }
        logger.info(
            "Rolled back interactive AR segment: "
            f"segments={remaining_segments}, tokens={self.cache_valid_len}, "
            f"global_frames={self.global_frame_counter}"
        )
        return {
            "remaining_segments": remaining_segments,
            # Archived SWA removes a full CPU record, then reselects GPU KV;
            # this count is not the decrease in GPU cache_valid_len.
            "removed_cache_tokens": token_count,
            "removed_latent_frames": latent_frames,
        }

    def _reset_interactive_noise_state(self) -> None:
        """Drop the noise timeline used by an interactive Web session."""
        self._interactive_noise_bank = None
        self._interactive_noise_generator = None
        self._interactive_noise_seed = None
        self._interactive_segment_noise_rng_states = []

    def _reset_interactive_pixelumm_decode_state(self) -> None:
        """Drop the PixelUMM global-window timeline used by an interactive session."""
        self._interactive_pixelumm_decode_session = None
        self._last_pixelumm_streaming_metadata = None

    def _fixed_initial_noise_timeline(
        self,
        *,
        batch_size: int,
        latent_frames: int,
        latent_h: int,
        latent_w: int,
        seed: Optional[int],
        continuing_session: bool,
    ) -> tuple[torch.Tensor, torch.Generator]:
        """Return a slice from one fixed-duration GPU noise timeline.

        Both a regular batch and incremental Web calls using the Web config
        sample the same fixed-size tensor. This matters because torch.randn on
        [B,C,T,H,W] is layout-dependent: sampling T=current_length is not equal
        to slicing a T=max_length tensor when C > 1. Incremental calls retain
        the bank and advance by global_frame_counter; a regular batch uses its
        prefix. The bank is sampled once on CPU for RNG parity, copied to GPU
        once, and then sliced directly without per-segment H2D copies.
        """
        resolved_seed = int(seed if seed is not None else 42)
        initial_seconds = float(
            getattr(
                self.args.inference,
                "interactive_noise_max_seconds",
                120.0,
            )
        )
        hard_max_seconds = float(
            getattr(
                self.args.inference,
                "interactive_noise_hard_max_seconds",
                initial_seconds,
            )
        )
        growth_seconds = float(
            getattr(
                self.args.inference,
                "interactive_noise_growth_seconds",
                30.0,
            )
        )
        if hard_max_seconds < initial_seconds:
            raise ValueError(
                "inference.interactive_noise_hard_max_seconds must be >= "
                "interactive_noise_max_seconds"
            )
        video_fps = int(getattr(self.args.inference, "video_fps", 24))
        max_pixel_frames = max(1, int(round(initial_seconds * video_fps)))
        max_latent_frames = (max_pixel_frames - 1) // 4 + 1
        required_shape = (
            int(batch_size),
            int(self.in_channels),
            int(latent_h),
            int(latent_w),
        )

        if not continuing_session:
            self._reset_interactive_noise_state()
            generator = torch.Generator(device="cpu").manual_seed(resolved_seed)
            cpu_noise_bank = torch.randn(
                [
                    batch_size,
                    self.in_channels,
                    max_latent_frames,
                    latent_h,
                    latent_w,
                ],
                generator=generator,
                dtype=self.target_dtype,
                device="cpu",
            )
            self._interactive_noise_bank = cpu_noise_bank.to(device=self.device)
            self._interactive_noise_generator = generator
            self._interactive_noise_seed = resolved_seed
            logger.info(
                "Initialized persistent GPU noise timeline: "
                f"seed={resolved_seed}, initial_seconds={initial_seconds:g}, "
                f"latent_frames={max_latent_frames}, "
                f"shape={tuple(self._interactive_noise_bank.shape)}"
            )
        else:
            if self._interactive_noise_bank is None:
                raise RuntimeError(
                    "interactive KV cache exists without its GPU noise timeline"
                )
            if self._interactive_noise_generator is None:
                raise RuntimeError("interactive noise timeline has no CPU RNG state")
            if self._interactive_noise_seed != resolved_seed:
                raise ValueError(
                    "interactive continuation changed seed: "
                    f"existing={self._interactive_noise_seed}, "
                    f"requested={resolved_seed}"
                )
            bank_shape = self._interactive_noise_bank.shape
            existing_shape = (
                int(bank_shape[0]),
                int(bank_shape[1]),
                int(bank_shape[3]),
                int(bank_shape[4]),
            )
            if existing_shape != required_shape:
                raise ValueError(
                    "interactive continuation changed latent shape: "
                    f"existing={existing_shape}, "
                    f"requested={required_shape}"
                )

        start = int(self.global_frame_counter)
        end = start + int(latent_frames)
        capacity = int(self._interactive_noise_bank.shape[2])
        if end > capacity:
            hard_max_pixel_frames = max(
                1,
                int(round(hard_max_seconds * video_fps)),
            )
            hard_capacity = (hard_max_pixel_frames - 1) // 4 + 1
            if end > hard_capacity:
                raise ValueError(
                    "interactive session exceeded its configured noise limit: "
                    f"requested latent range [{start}:{end}], "
                    f"hard_capacity={hard_capacity}. Increase "
                    "inference.interactive_noise_hard_max_seconds."
                )
            growth_pixel_frames = max(
                1,
                int(round(max(growth_seconds, 1.0) * video_fps)),
            )
            growth_latent_frames = max(
                1,
                (growth_pixel_frames - 1) // 4 + 1,
            )
            new_capacity = min(
                hard_capacity,
                max(end, capacity + growth_latent_frames),
            )
            extension = torch.randn(
                [
                    batch_size,
                    self.in_channels,
                    new_capacity - capacity,
                    latent_h,
                    latent_w,
                ],
                generator=self._interactive_noise_generator,
                dtype=self.target_dtype,
                device="cpu",
            ).to(device=self.device)
            self._interactive_noise_bank = torch.cat(
                [self._interactive_noise_bank, extension],
                dim=2,
            )
            logger.info(
                "Grew persistent noise timeline without changing its prefix: "
                f"{capacity} -> {new_capacity} latent frames"
            )
            capacity = new_capacity
        logger.info(
            "Using persistent initial-noise slice: "
            f"latent_frames=[{start}:{end}]/{capacity}"
        )
        return (
            self._interactive_noise_bank[:, :, start:end, :, :],
            self._interactive_noise_generator,
        )

    def _evict_oldest_segment(self):
        """Evict history while keeping quantized values and their scales aligned."""
        if not self.segment_token_lengths:
            return
        tokens_to_evict = self.segment_token_lengths.pop(0)
        remaining = self.cache_valid_len - tokens_to_evict

        if remaining > 0:
            for cache in self.kv_cache:
                keys = (
                    ("k", "v", "k_scale", "v_scale")
                    if cache["k"].dtype == torch.int8 else ("k", "v")
                )
                for key in keys:
                    cache[key][:, :remaining].copy_(
                        cache[key][:, tokens_to_evict : self.cache_valid_len].clone()
                    )
            for mask in self.cache_mask_list:
                mask[:, :remaining] = mask[
                    :, tokens_to_evict : self.cache_valid_len
                ].clone()

        self.cache_valid_len = remaining
        logger.info(
            f"Evicted oldest segment ({tokens_to_evict} tokens), cache_valid_len={self.cache_valid_len}"
        )

    def _sync_segment_vae_status(
        self,
        local_error: Optional[Exception],
        *,
        context: str,
        status_tensor: Optional[torch.Tensor] = None,
    ) -> None:
        """Synchronize rank-local VAE failures without stranding SP peers."""
        import torch.distributed as dist

        failed = local_error is not None
        sp_group, sp_size, _ = get_model_parallel_info()
        if dist.is_available() and dist.is_initialized() and sp_size > 1:
            if sp_group is None:
                raise RuntimeError(
                    "Segment VAE decode requires an initialized SP process "
                    "group when sp_size > 1."
                )
            if status_tensor is None:
                raise RuntimeError(
                    "Segment VAE decode status tensor was not initialized."
                )
            status_tensor.fill_(int(failed))
            dist.all_reduce(
                status_tensor,
                op=dist.ReduceOp.MAX,
                group=sp_group,
            )
            failed = bool(status_tensor.item())

        if failed:
            message = f"Segment VAE decode failed during {context}."
            if local_error is not None:
                raise RuntimeError(message) from local_error
            raise RuntimeError(f"{message} The SP output rank reported an error.")

    def _encode_prompt(
        self,
        prompt,
        negative_prompt=None,
        data_type="video",
        encode_negative=None,
    ):
        text_states, text_mask = self._encode_single_prompt(
            prompt,
            data_type=data_type,
        )

        neg_states = None
        neg_mask = None
        if (
            self.do_classifier_free_guidance
            if encode_negative is None
            else bool(encode_negative)
        ):
            neg_text = negative_prompt or ""
            neg_states, neg_mask = self._encode_single_prompt(
                neg_text,
                data_type=data_type,
                is_uncond=True,
            )

        return {
            "text_states": text_states,
            "text_mask": text_mask,
            "neg_states": neg_states,
            "neg_mask": neg_mask,
        }

    # NOTE: the intra-segment per-action mask helpers
    # (_compute_action_start, _intra_seg_enabled, _compute_action_text_offsets,
    # _compute_action_video_windows) now live on BasePipeline so the one_shot
    # bidirectional pipeline can share the exact same implementation.

    def _cam_flag(self, name: str, default: bool = False) -> bool:
        """Resolve a camera normalization flag (inference.* > root-level).

        Mirrors the resolution used by the one_shot pipeline. The training
        camera yaml sets these under ``dataset.*``; ``_flatten_cfg`` promotes
        them to root level, so root is the fallback after ``inference.*``.
        """
        inf_cfg = getattr(self.args, "inference", None)
        if inf_cfg is not None and getattr(inf_cfg, name, None) is not None:
            return bool(getattr(inf_cfg, name))
        return bool(getattr(self.args, name, default))

    @torch.no_grad()
    def _prepare_camera_keyboard(
        self,
        camera_trajectory: Optional[str],
        num_frames: int,
        latent_h: int,
        latent_w: int,
        segment_frame_boundaries: Optional[List[int]],
        trajectory_translation_scale: Optional[float] = None,
        trajectory_rotation_scale: Optional[float] = None,
    ) -> Optional[torch.Tensor]:
        """Build the full-clip keyboard embedding for camera control.

        Returns ``[1, T', C_kb]`` (one keyboard token per latent frame, since
        this model uses temporal patch_size==1) for action_attn /
        extrinsics_attn, or ``None`` when camera control is inactive /
        unsupported for the AR cache path.
        """
        cam_mode = getattr(self.transformer, "camera_injection_mode", None)
        if camera_trajectory is None:
            if cam_mode is not None:
                logger.warning(
                    f"  Model has camera_injection_mode='{cam_mode}' but no "
                    f"camera_trajectory was provided; camera modules stay inactive."
                )
            return None
        if cam_mode is None:
            logger.warning(
                f"  camera_trajectory='{camera_trajectory}' is set, but the model "
                f"has camera_injection_mode=None. Camera control will have NO effect."
            )
            return None
        if cam_mode not in ("action_attn", "extrinsics_attn"):
            logger.warning(
                f"  camera_injection_mode='{cam_mode}' is not supported by the "
                f"seg_ar / chunk_ar cache path (only action_attn / extrinsics_attn). "
                f"Camera control will be skipped for this run."
            )
            return None

        from hymm.pipeline.bidirectional import prepare_camera_inputs

        norm_first = self._cam_flag("normalize_relative_to_first_frame", False)
        norm_fw = self._cam_flag("normalize_framewise", False)
        norm_disp = self._cam_flag("normalize_displacement", False)
        norm_disp_per_video = self._cam_flag("normalize_displacement_per_video", False)
        inf_cfg = getattr(self.args, "inference", None)
        raw_scale = trajectory_translation_scale
        if raw_scale is None:
            raw_scale = getattr(inf_cfg, "trajectory_translation_scale", None)
        traj_scale = float(raw_scale) if raw_scale is not None else None
        raw_rotation_scale = trajectory_rotation_scale
        if raw_rotation_scale is None:
            raw_rotation_scale = getattr(inf_cfg, "trajectory_rotation_scale", 1.0)
        rotation_scale = (
            float(raw_rotation_scale) if raw_rotation_scale is not None else 1.0
        )
        if rotation_scale != 1.0 and cam_mode != "extrinsics_attn":
            raise ValueError("Rotation gains currently require extrinsics_attn")
        logger.info(
            f"  Camera config: normalize_first={norm_first}, framewise={norm_fw}, "
            f"displacement={norm_disp}, "
            f"displacement_per_video={norm_disp_per_video}, "
            f"trajectory_translation_scale={traj_scale}, "
            f"trajectory_rotation_scale={rotation_scale}"
        )

        cam = prepare_camera_inputs(
            trajectory_source=camera_trajectory,
            num_frames=num_frames,
            device=self.device,
            dtype=self.target_dtype,
            normalize_relative_to_first_frame=norm_first,
            normalize_framewise=norm_fw,
            normalize_displacement=norm_disp,
            normalize_displacement_per_video=norm_disp_per_video,
            segment_boundaries_pixel=segment_frame_boundaries,
            translation_scale=traj_scale,
            rotation_scale=rotation_scale,
        )
        if cam["num_frames"] != num_frames:
            logger.warning(
                f"  Camera trajectory length ({cam['num_frames']} pixel frames) "
                f"differs from video num_frames ({num_frames}). Keeping the video's "
                f"segment boundaries; per-chunk keyboard slices are clamped to the "
                f"available keyboard length."
            )

        # Preprocess into the per-latent-frame keyboard embedding. Non-FSDP at
        # inference, so calling the model helper directly is safe.
        with torch.autocast(device_type="cuda", dtype=self.target_dtype):
            if "turn_camera_offsets" in cam:
                if cam_mode != "extrinsics_attn":
                    raise ValueError("Precomputed demo turns require extrinsics_attn")
                # Each original turn has its own first-frame repeat/padding.
                # Padding concatenated poses just once would mix the four
                # matrices belonging to different turns into a camera token.
                offsets = cam["turn_camera_offsets"]
                embeddings = []
                for start, end in zip(offsets[:-1], offsets[1:]):
                    embeddings.append(
                        self.transformer._preprocess_camera(
                            norm_extrinsics=cam["norm_extrinsics"][:, start:end],
                            norm_intrinsics=cam["norm_intrinsics"][:, start:end],
                            action_onehot=cam["action_onehot"][:, start:end],
                            oh=latent_h,
                            ow=latent_w,
                            img_dtype=self.target_dtype,
                            device=self.device,
                        )
                    )
                keyboard_full = torch.cat(embeddings, dim=1)
                expected = (num_frames - 1) // 4 + 1
                if keyboard_full.shape[1] != expected:
                    raise ValueError(
                        "Precomputed camera tokens do not match offline latent length"
                    )
            else:
                keyboard_full = self.transformer._preprocess_camera(
                    norm_extrinsics=cam["norm_extrinsics"],
                    norm_intrinsics=cam["norm_intrinsics"],
                    action_onehot=cam["action_onehot"],
                    oh=latent_h,
                    ow=latent_w,
                    img_dtype=self.target_dtype,
                    device=self.device,
                )
        logger.info(
            f"  Camera control enabled (seg_ar): mode={cam_mode}, "
            f"trajectory='{camera_trajectory}', keyboard={tuple(keyboard_full.shape)}"
        )
        return keyboard_full

    def _slice_chunk_camera(
        self,
        camera_embedding_full: Optional[torch.Tensor],
        cam_mode: Optional[str],
        chunk_latent_start: int,
        chunk_latent_end: int,
        seg_idx: int,
        segment_has_camera: Optional[List[bool]],
        guidance_branches: int,
    ):
        """Slice the full keyboard for one chunk and build its camera-off gate.

        Returns ``(chunk_camera_embedding, chunk_camera_has_mask)``. The
        embedding is ``[cfg_batch, n_frames_chunk, C_kb]`` and the mask is a
        1D per-sample ``[cfg_batch]`` bool (or ``None`` when the segment is
        camera-on). Since each chunk is a single segment, the legacy 1D
        per-sample gate is the correct shape (the 2D per-segment mask path in
        ActionModule requires segment boundaries, which we don't use here).
        """
        if camera_embedding_full is None or cam_mode not in (
            "action_attn",
            "extrinsics_attn",
        ):
            return None, None

        kb_len = camera_embedding_full.shape[1]
        cs = max(0, min(int(chunk_latent_start), kb_len))
        ce = max(cs, min(int(chunk_latent_end), kb_len))
        expected = int(chunk_latent_end) - int(chunk_latent_start)
        if ce <= cs:
            logger.warning(
                f"  Camera: empty keyboard slice for chunk "
                f"[{chunk_latent_start}:{chunk_latent_end}] (kb_len={kb_len}); "
                f"skipping camera for this chunk."
            )
            return None, None
        if (ce - cs) != expected:
            logger.warning(
                f"  Camera: keyboard slice length {ce - cs} != chunk latent frames "
                f"{expected} (kb_len={kb_len}); skipping camera for this chunk to "
                f"avoid keyboard/image frame misalignment."
            )
            return None, None

        emb = camera_embedding_full[:, cs:ce, :]
        if guidance_branches > 1:
            emb = emb.repeat(guidance_branches, 1, 1)

        seg_on = True
        if segment_has_camera is not None and seg_idx < len(segment_has_camera):
            seg_on = bool(segment_has_camera[seg_idx])

        has_mask = None
        if guidance_branches == 3:
            # Separate camera/text CFG, matching one_shot:
            #   base   = negative text, camera off
            #   camera = negative text, camera on
            #   text   = positive text, camera on
            camera_mask = [False, seg_on, seg_on]
            has_mask = torch.tensor(
                camera_mask,
                dtype=torch.bool,
                device=self.device,
            )
            logger.info(
                f"  Separate camera CFG gate for segment {seg_idx}: "
                f"base/camera/text={camera_mask}."
            )
        elif guidance_branches == 2:
            # Text is always negative/empty on the base branch. Camera can
            # optionally remain active there, allowing CFG to guide text while
            # preserving the same current-segment camera condition in both
            # branches. Both branches read canonical history at segment start.
            ar_uncond_with_camera = bool(
                getattr(self.args.inference, "ar_uncond_with_camera", True)
            )
            camera_mask = [seg_on if ar_uncond_with_camera else False, seg_on]
            has_mask = torch.tensor(
                camera_mask,
                dtype=torch.bool,
                device=self.device,
            )
            logger.info(
                f"  Text CFG camera gate for segment {seg_idx}: "
                f"mask={camera_mask}, "
                f"ar_uncond_with_camera={ar_uncond_with_camera}."
            )
        elif not seg_on:
            has_mask = torch.zeros(
                [emb.shape[0]],
                dtype=torch.bool,
                device=self.device,
            )
            logger.info(
                f"  Camera OFF for segment {seg_idx} (segment_control='prompt'): "
                f"cross-attention output gated to zero."
            )
        else:
            logger.info(
                f"  Camera ON  for segment {seg_idx}: injecting keyboard "
                f"[{cs}:{ce}] ({ce - cs} latent frames) per double block."
            )
        return emb, has_mask

    @torch.no_grad()
    def inference(
        self,
        prompts: List[str],
        num_frames: int,
        height: int,
        width: int,
        guidance_scale: float,
        num_inference_steps: int,
        flow_shift: float,
        seed: int,
        camera_guidance_scale: float = 1.0,
        negative_prompt: str = "",
        segment_frame_boundaries: Optional[List[int]] = None,
        camera_trajectory: Optional[str] = None,
        segment_has_camera: Optional[List[bool]] = None,
        cfg_schedule: Optional[str] = None,
        cfg_skip_steps: int = 0,
        segment_callback: Optional[
            Callable[[int, int, str, torch.Tensor], None]
        ] = None,
        preserve_kv_cache: bool = False,
        trajectory_translation_scale: Optional[float] = None,
        trajectory_rotation_scale: Optional[float] = None,
    ):
        """Run autoregressive inference on a list of prompts."""
        return self._process_single_prompt(
            prompts=prompts,
            num_frames=num_frames,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            camera_guidance_scale=camera_guidance_scale,
            num_inference_steps=num_inference_steps,
            flow_shift=flow_shift,
            seed=seed,
            negative_prompt=negative_prompt,
            segment_frame_boundaries=segment_frame_boundaries,
            camera_trajectory=camera_trajectory,
            trajectory_translation_scale=trajectory_translation_scale,
            trajectory_rotation_scale=trajectory_rotation_scale,
            segment_has_camera=segment_has_camera,
            cfg_schedule=cfg_schedule,
            cfg_skip_steps=cfg_skip_steps,
            segment_callback=segment_callback,
            preserve_kv_cache=preserve_kv_cache,
        )

    @torch.no_grad()
    def _process_single_prompt(
        self,
        prompts: List[str],
        num_frames: int,
        height: int,
        width: int,
        guidance_scale: float,
        camera_guidance_scale: float,
        num_inference_steps: int,
        flow_shift: float,
        seed: int,
        negative_prompt: str = "",
        segment_frame_boundaries: Optional[List[int]] = None,
        camera_trajectory: Optional[str] = None,
        segment_has_camera: Optional[List[bool]] = None,
        cfg_schedule: Optional[str] = None,
        cfg_skip_steps: int = 0,
        segment_callback: Optional[
            Callable[[int, int, str, torch.Tensor], None]
        ] = None,
        preserve_kv_cache: bool = False,
        trajectory_translation_scale: Optional[float] = None,
        trajectory_rotation_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Generate video autoregressively, chunk-by-chunk within each segment."""
        self.transformer.sparse_optim_counts.clear()
        if segment_frame_boundaries is not None and len(segment_frame_boundaries) >= 2:
            num_segments = len(segment_frame_boundaries) - 1
        else:
            num_segments = len(prompts)
        batch_size = 1

        rollout_type = getattr(self.args.inference, "rollout_type", "seg_ar")
        is_chunk_ar = rollout_type == "chunk_ar"
        chunk_size = (
            (getattr(self.args, "inference_chunk_size", 0) or 0) if is_chunk_ar else 0
        )
        segment_vae_decode = bool(
            getattr(
                self.args.inference,
                "seg_ar_segment_vae_decode",
                False,
            )
        )

        if is_chunk_ar and chunk_size <= 0:
            raise ValueError(
                f"rollout_type='chunk_ar' requires inference_chunk_size > 0, "
                f"got {chunk_size}. Use rollout_type='seg_ar' for segment-level AR."
            )
        if segment_vae_decode and rollout_type != "seg_ar":
            raise ValueError(
                "inference.seg_ar_segment_vae_decode=true is only supported "
                f"with rollout_type='seg_ar', got {rollout_type!r}."
            )

        segment_vae_session = None
        segment_vae_group, segment_vae_sp_size, segment_vae_rank = get_model_parallel_info()
        segment_vae_output_rank = segment_vae_rank == 0
        decoded_video_cpu: Optional[torch.Tensor] = None
        decoded_pixel_frames = 0
        # Coordinates are local to this call. Preserve the PixelUMM timeline for a
        # later call only when preserve_kv_cache is requested.
        pixelumm_request_pixel_offset = 0
        pixelumm_output_pixel_start = 0
        emitted_latent_frames = 0
        pixel_frames_per_latent = None
        segment_vae_status = None
        segment_vae_cache_offload = segment_vae_sp_size < 16
        segment_vae_uses_pixelumm = (
            getattr(getattr(self.args, "vae", None), "name", None) == "pixelumm"
        )
        if segment_vae_uses_pixelumm and not bool(
            getattr(self.vae, "handles_distributed_decode", False)
        ):
            raise RuntimeError(
                "model.vae.name=pixelumm did not construct a PixelUMM distributed decoder"
            )
        pixelumm_stable_streaming = _resolve_pixelumm_stable_prefix_streaming(
            requested=bool(
                getattr(
                    self.args.vae,
                    "stable_prefix_streaming",
                    False,
                )
            ),
            segment_vae_decode=segment_vae_decode,
            vae_name=getattr(getattr(self.args, "vae", None), "name", None),
        )
        if segment_vae_decode:
            segment_vae_status = torch.zeros(
                1,
                dtype=torch.int32,
                device=self.device,
            )
            if segment_vae_uses_pixelumm:
                if pixelumm_stable_streaming:
                    if self._interactive_pixelumm_decode_session is None:
                        self._interactive_pixelumm_decode_session = (
                            self.vae.create_streaming_decode_session(
                                to_cpu=True,
                                retain_history=False,
                            )
                        )
                    # A retained PixelUMM timeline uses global coordinates; output
                    # starts at the previous provisional tail.
                    pixelumm_request_pixel_offset = int(
                        self._interactive_pixelumm_decode_session.available_pixel_frames
                    )
                    pixelumm_output_pixel_start = int(
                        self._interactive_pixelumm_decode_session.stable_pixel_frames
                    )
                    logger.info(
                        "  PixelUMM replace-tail streaming ON: publish "
                        "each complete segment immediately, retain the final "
                        "temporal overlap provisionally, and in-place replace "
                        "that old tail after the next segment completes its "
                        f"cross-boundary window over SP={segment_vae_sp_size}."
                    )
                else:
                    logger.info(
                        "  seg_ar incremental PixelUMM decode ON: decode the cumulative "
                        "finalized latent timeline at every segment boundary using "
                        f"the PixelUMM distributed backend over SP={segment_vae_sp_size}."
                    )
            else:
                # Every SP rank owns a session facade. Complete newly-arrived
                # temporal tiles are pipelined across the group while rank 0 keeps
                # the canonical feature cache between segment boundaries.
                segment_vae_session = self._create_segment_vae_decode_session(
                    group=segment_vae_group,
                    rank=segment_vae_rank,
                    world_size=segment_vae_sp_size,
                )
                logger.info(
                    "  seg_ar synchronous speculative VAE decode ON: canonical "
                    "global 10-latent tiles preserve cache state; incomplete "
                    "segment tails use disposable repeat-last padding previews "
                    f"(complete tiles distributed over SP={segment_vae_sp_size}); "
                    "feature-cache strategy="
                    f"{'CPU round-trip' if segment_vae_cache_offload else 'GPU copy'} "
                    f"for SP={segment_vae_sp_size}."
                )

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        target_height = align_to(height, 16)
        target_width = align_to(width, 16)

        latent_frames = (num_frames - 1) // 4 + 1
        latent_h = target_height // self.s_ratio
        latent_w = target_width // self.s_ratio

        tt_total = latent_frames // self.patch_size[0]
        th = latent_h // self.patch_size[1]
        tw = latent_w // self.patch_size[2]
        tokens_per_frame = th * tw
        swa_config = getattr(self.args.inference, "sliding_window", {})
        swa_enabled = bool(swa_config.get("enabled", False))
        swa_sampling = swa_config.get("sampling", "recent")
        swa_archived = swa_enabled and swa_sampling in ARCHIVED_HISTORY_SAMPLINGS
        past_frames = swa_config.get("past_frames", 36)
        swa_gamma = swa_config.get("gamma", 2.0)
        swa_capacity = 0
        if getattr(self, "qvg_kv_storage", False):
            if (preserve_kv_cache or rollout_type != "seg_ar"
                    or guidance_scale != 1.0 or camera_guidance_scale != 1.0
                    or self.max_cache_segments is not None
                    or self._prev_seg_last_frame_only()):
                raise ValueError("QVG supports offline full-segment attention with CFG=1, "
                                 "no interactive preservation, and no segment eviction")
        if swa_enabled:
            if swa_sampling not in HISTORY_SAMPLINGS:
                raise ValueError(f"sliding_window.sampling must be one of {HISTORY_SAMPLINGS}")
            if type(past_frames) is not int or past_frames < 1:
                raise ValueError("sliding_window.past_frames must be a positive integer")
            if swa_archived and (
                getattr(self.transformer, "nomas_sparse_config", {}).get("enabled", False)
                or self._prev_seg_last_frame_only()
            ):
                raise ValueError("Sampled history requires dense attention and prev_seg_last_frame_only=false")
            swa_capacity = past_frames * tokens_per_frame
            logger.info(f"SWA enabled: sampling={swa_sampling}, past_frames={past_frames}, "
                        f"video_capacity={swa_capacity}, tokens_per_frame={tokens_per_frame}")


        ar_mode = "chunk-level" if is_chunk_ar else "segment-level"
        logger.info(f"CPT KV-Cache AR Inference ({ar_mode}):")
        logger.info(f"  Segments: {num_segments}")
        logger.info(f"  rollout_type: {rollout_type}, chunk_size: {chunk_size}")
        logger.info(
            f"  Resolution: {target_height}x{target_width}, frames={num_frames}"
        )
        logger.info(f"  Latent: T={latent_frames}, H={latent_h}, W={latent_w}")
        logger.info(f"  Patched: tt={tt_total}, th={th}, tw={tw}")
        logger.info(f"  tokens_per_frame={tokens_per_frame}")
        if cfg_schedule and cfg_schedule != "constant":
            logger.info(
                f"  CFG schedule: {cfg_schedule} (base cfg={guidance_scale}, skip_steps={cfg_skip_steps})"
            )

        # Opt-in ActionModule I/O debug hooks (default OFF). Enable with
        # DEBUG_ACTION_MODULE=1 (see tools_tmp/action_module_debug_hooks.py).
        if os.environ.get("DEBUG_ACTION_MODULE", "").strip() not in (
            "",
            "0",
            "false",
            "False",
        ):
            try:
                from tools_tmp.action_module_debug_hooks import (
                    attach_action_module_hooks,
                )

                attach_action_module_hooks(
                    self.transformer,
                    rank=getattr(nccl_info, "global_rank", 0),
                    log_path=os.environ.get("DEBUG_ACTION_MODULE_LOG") or None,
                )
            except Exception as _e:  # noqa: BLE001
                logger.warning(f"[AM-DEBUG] failed to attach hooks: {_e}")

        # Opt-in ActionModule I/O debug hooks (default OFF). Enable with
        # DEBUG_ACTION_MODULE=1 (see tools_tmp/action_module_debug_hooks.py).
        if os.environ.get("DEBUG_ACTION_MODULE", "").strip() not in (
            "",
            "0",
            "false",
            "False",
        ):
            try:
                from tools_tmp.action_module_debug_hooks import (
                    attach_action_module_hooks,
                )

                attach_action_module_hooks(
                    self.transformer,
                    rank=getattr(nccl_info, "global_rank", 0),
                    log_path=os.environ.get("DEBUG_ACTION_MODULE_LOG") or None,
                )
            except Exception as _e:  # noqa: BLE001
                logger.warning(f"[AM-DEBUG] failed to attach hooks: {_e}")

        # Opt-in ActionModule I/O debug hooks (default OFF). Enable with
        # DEBUG_ACTION_MODULE=1 (see tools_tmp/action_module_debug_hooks.py).
        if os.environ.get("DEBUG_ACTION_MODULE", "").strip() not in (
            "",
            "0",
            "false",
            "False",
        ):
            try:
                from tools_tmp.action_module_debug_hooks import (
                    attach_action_module_hooks,
                )

                attach_action_module_hooks(
                    self.transformer,
                    rank=getattr(nccl_info, "global_rank", 0),
                    log_path=os.environ.get("DEBUG_ACTION_MODULE_LOG") or None,
                )
            except Exception as _e:  # noqa: BLE001
                logger.warning(f"[AM-DEBUG] failed to attach hooks: {_e}")

        if segment_frame_boundaries is not None:
            latent_boundaries = segment_frame_boundaries
            if latent_boundaries[-1] != latent_frames:
                latent_boundaries = []
                for b in segment_frame_boundaries:
                    if b == 0:
                        latent_boundaries.append(0)
                    else:
                        # Match training (wm_1m_torch_dataset.py): pixel_frame // vae_temporal_ratio,
                        # kept strictly increasing so no zero-width segment is produced.
                        latent_idx = b // 4
                        latent_idx = max(
                            latent_boundaries[-1] + 1, min(latent_idx, latent_frames)
                        )
                        latent_boundaries.append(latent_idx)
                latent_boundaries[0] = 0
                latent_boundaries[-1] = latent_frames
                logger.info(
                    f"  Converted pixel boundaries to latent: {segment_frame_boundaries} -> {latent_boundaries}"
                )
        else:
            assert latent_frames % num_segments == 0, (
                f"latent_frames ({latent_frames}) must be divisible by num_segments ({num_segments}). "
                f"Provide --segment-frame-boundaries for uneven splits."
            )
            frames_per_seg = latent_frames // num_segments
            latent_boundaries = [i * frames_per_seg for i in range(num_segments + 1)]

        logger.info(f"  Latent frame boundaries: {latent_boundaries}")

        # Optionally pad each segment's latent length UP to a whole multiple of
        # chunk_size so segment boundaries align with the global chunk grid used
        # in training (training chunks globally, not per-segment). This removes
        # partial trailing chunks. It lengthens the video (each event extended to
        # a whole chunk); tensors below are sized from the updated latent_frames.
        pad_segment_to_chunk_size = getattr(
            self.args.inference, "pad_segment_to_chunk_size", False
        )
        if is_chunk_ar and chunk_size > 0 and pad_segment_to_chunk_size:
            padded_boundaries = [0]
            for seg_idx in range(num_segments):
                seg_frames = latent_boundaries[seg_idx + 1] - latent_boundaries[seg_idx]
                padded_len = ((seg_frames + chunk_size - 1) // chunk_size) * chunk_size
                padded_boundaries.append(padded_boundaries[-1] + padded_len)
            if padded_boundaries != latent_boundaries:
                latent_boundaries = padded_boundaries
                latent_frames = latent_boundaries[-1]
                tt_total = latent_frames // self.patch_size[0]
                num_frames = (latent_frames - 1) * 4 + 1
                logger.info(
                    f"  pad_segment_to_chunk_size: padded boundaries -> {latent_boundaries} "
                    f"(latent_frames={latent_frames}, num_frames={num_frames})"
                )

        chunk_schedule = []
        for seg_idx in range(num_segments):
            seg_start = latent_boundaries[seg_idx]
            seg_end = latent_boundaries[seg_idx + 1]
            seg_frames = seg_end - seg_start

            if chunk_size > 0 and seg_frames > chunk_size:
                chunk_starts = list(range(seg_start, seg_end, chunk_size))
                for ci, cs in enumerate(chunk_starts):
                    ce = min(cs + chunk_size, seg_end)
                    if ci == len(chunk_starts) - 2 and (seg_end - ce) < chunk_size // 2:
                        ce = seg_end
                        chunk_schedule.append((seg_idx, cs, ce))
                        break
                    chunk_schedule.append((seg_idx, cs, ce))
            else:
                chunk_schedule.append((seg_idx, seg_start, seg_end))

        total_chunks = len(chunk_schedule)
        logger.info(f"  Total chunks: {total_chunks}")
        for ci, (si, cs, ce) in enumerate(chunk_schedule):
            logger.info(
                f"    chunk[{ci}]: seg={si}, latent_frames=[{cs}:{ce}] ({ce - cs} frames)"
            )

        continuing_session = bool(
            preserve_kv_cache
            and self.kv_cache is not None
            and (self.cache_valid_len > 0 or swa_enabled)
            and self._interactive_segment_count > 0
        )
        session_segment_offset = (
            int(self._interactive_segment_count) if continuing_session else 0
        )
        text_len = getattr(self.args, "text_len", 256)
        max_chunk_latent_frames = max(ce - cs for _, cs, ce in chunk_schedule)
        max_img_tokens_per_chunk = max_chunk_latent_frames * tokens_per_frame
        if self.max_cache_segments is not None:
            max_cache_tokens = self.max_cache_segments * (
                max_img_tokens_per_chunk + text_len
            )
        else:
            max_cache_tokens = (
                total_chunks * max_img_tokens_per_chunk + num_segments * text_len
            )
        request_cache_tokens = max_cache_tokens
        max_cache_tokens += max_img_tokens_per_chunk + text_len
        if preserve_kv_cache:
            preallocated_tokens = self._interactive_kv_preallocate_tokens(
                tokens_per_frame=tokens_per_frame,
                text_len=text_len,
            )
            if preallocated_tokens > max_cache_tokens:
                logger.info(
                    "Interactive KV fixed preallocation: "
                    f"{max_cache_tokens} -> {preallocated_tokens} tokens"
                )
                max_cache_tokens = preallocated_tokens

        if swa_enabled:
            if swa_archived:
                # Each represented segment contributes its text once, outside
                # the exact latent-frame budget. At most past_frames segments
                # can be represented by past_frames selected video frames,
                # including segments from earlier interactive requests.
                swa_capacity += min(past_frames, session_segment_offset + num_segments) * text_len
                logger.info(f"{swa_sampling} history GPU capacity: {swa_capacity} tokens; archive=cpu")
            max_cache_tokens = swa_capacity
        cam_mode = getattr(self.transformer, "camera_injection_mode", None)
        do_text_cfg = guidance_scale > 1.0
        has_camera_condition = (
            camera_trajectory is not None
            and cam_mode in ("action_attn", "extrinsics_attn")
            and (segment_has_camera is None or any(segment_has_camera))
        )
        do_camera_cfg = camera_guidance_is_active(
            camera_guidance_scale,
            has_camera_condition,
        )
        guidance_branches = 3 if do_camera_cfg else (2 if do_text_cfg else 1)
        needs_guidance = guidance_branches > 1
        cfg_batch = batch_size * guidance_branches
        single_slot_cfg_cache_enabled = bool(
            getattr(
                self.args.inference,
                "ar_single_slot_cfg_cache",
                True,
            )
        )
        single_slot_cfg_cache = (
            needs_guidance and not is_chunk_ar and single_slot_cfg_cache_enabled
        )
        persistent_cache_batch = batch_size if single_slot_cfg_cache else cfg_batch
        # Roll the cache forward with the conditional velocity only, the way
        # segment self-forcing training does. Without CFG the prediction already
        # is conditional, so there is nothing to separate.
        cache_latent_cond_only = needs_guidance and bool(
            getattr(self.args.inference, "ar_cache_latent_cond_only", False)
        )

        logger.info(
            "AR prompt entry: "
            f"preserve_kv_cache={preserve_kv_cache}, "
            f"continuing_session={continuing_session}, "
            f"cache_present={self.kv_cache is not None}, "
            f"cache_valid_len={self.cache_valid_len}, "
            f"global_frame_counter={self.global_frame_counter}"
        )
        if continuing_session:
            expected_batch = int(self.kv_cache[0]["k"].shape[0])
            if expected_batch != persistent_cache_batch:
                raise RuntimeError(
                    "Interactive continuation changed CFG cache batch size: "
                    f"existing={expected_batch}, requested={persistent_cache_batch}."
                )
            self._ensure_kv_cache_capacity(
                swa_capacity if swa_enabled else self.cache_valid_len + request_cache_tokens
            )
            logger.info(
                "Continuing persistent AR session: "
                f"cached_segments={session_segment_offset}, "
                f"cache_tokens={self.cache_valid_len}, "
                f"global_frames={self.global_frame_counter}"
            )
        else:
            self._initialize_kv_cache(
                batch_size=persistent_cache_batch,
                max_cache_tokens=max_cache_tokens,
                dtype=self.target_dtype,
                device=self.device,
            )
            self._interactive_cached_action_tails = []
            self._interactive_cached_seg_video_ranges = []
            self._interactive_segment_count = 0
            self._interactive_segment_cache_tokens = []
            self._interactive_segment_latent_lengths = []
            if not preserve_kv_cache:
                self._reset_interactive_noise_state()
        if swa_enabled:
            for cache in self.kv_cache:
                cache["swa_capacity"] = swa_capacity
                if swa_archived:
                    cache["swa_sampling"] = swa_sampling
                    cache["swa_past_frames"] = past_frames
                    cache["swa_gamma"] = swa_gamma
        if isinstance(self.kv_cache, PhaseSwappedKVCache):
            restored_bytes = self.kv_cache.restore(
                required_tokens=(
                    swa_capacity if swa_enabled else self.cache_valid_len + request_cache_tokens
                ),
                valid_tokens=self.cache_valid_len,
            )
            self.cache_mask_list = self.kv_cache.cache_masks
            self._offload_interactive_pixelumm()
            logger.info(
                "Prepared phase-swapped GPU KV working set: "
                f"capacity={self.kv_cache.active_capacity}, "
                f"restored={restored_bytes / (1024**3):.2f} GiB"
            )
        if single_slot_cfg_cache:
            logger.info(
                "  seg_ar CFG cache optimization: one persistent canonical KV "
                "slot; history is broadcast read-only to all guidance branches."
            )
        elif needs_guidance and not is_chunk_ar:
            logger.info(
                f"  seg_ar CFG cache optimization disabled: keeping "
                f"{guidance_branches} physical guidance KV slots."
            )
        if cache_latent_cond_only:
            logger.info(
                "  ar_cache_latent_cond_only ON: the KV cache is written from a "
                "conditional-velocity latent (train parity), the decoded video "
                "still follows the CFG-guided trajectory."
            )

        if (nccl_info.sp_size > 1 or nccl_info.tp_size > 1) and nccl_info.parallel_dims is not None:
            sync_random_states(nccl_info.parallel_dims)
        fixed_noise_timeline = getattr(
            self.args.inference,
            "interactive_noise_max_seconds",
            None,
        )
        noise_growth_rollback = None
        if (preserve_kv_cache and continuing_session
                and self._interactive_noise_bank is not None
                and self.global_frame_counter + latent_frames > self._interactive_noise_bank.shape[2]):
            # Undoing the first segment of this request must also undo RNG
            # consumption from growing the shared noise timeline.
            noise_growth_rollback = (
                self._interactive_noise_generator.get_state().clone(),
                int(self._interactive_noise_bank.shape[2]),
            )
        if preserve_kv_cache or fixed_noise_timeline is not None:
            all_noise, generator = self._fixed_initial_noise_timeline(
                batch_size=batch_size,
                latent_frames=latent_frames,
                latent_h=latent_h,
                latent_w=latent_w,
                seed=seed,
                continuing_session=continuing_session,
            )
        else:
            generator = torch.Generator(device="cpu").manual_seed(
                seed if seed is not None else 42
            )
            all_noise = torch.randn(
                [
                    batch_size,
                    self.in_channels,
                    latent_frames,
                    latent_h,
                    latent_w,
                ],
                generator=generator,
                dtype=self.target_dtype,
            ).to(device=self.device)

        logger.info("Encoding text prompts...")
        encoded_prompts = []
        # Per-segment action-start (post-crop S_enc token index where
        # contained_actions begins; -1 = no action -> no head/tail split).
        # Used below to hide each segment's action tail from OTHER segments'
        # video in the KV cache, matching the training magi_mp_causal mask.
        action_starts = []
        cross_segment_text_head_only = self._cross_segment_text_head_only_enabled()

        # Previous-segment last-frame-only video restriction (seg_ar only;
        # chunk_size == 0). Mirrors the training flag prev_seg_last_frame_only
        # (create_magi_mp_causal_ranges) and the one_shot pipeline: each cached
        # PREVIOUS segment's video keys are masked down to only its FINAL latent
        # frame (tokens_per_frame tokens); its text keys stay fully visible. The
        # KV mask is query-agnostic, so this restricts BOTH the current chunk's
        # video and text queries — matching training where video AND text of
        # segment j attend only previous segments' last video frame (text->text
        # is unchanged). No effect in chunk_ar (chunk_size > 0), matching the
        # training helper which skips the restriction for chunk-wise causal.
        prev_seg_last_frame = self._prev_seg_last_frame_only() and chunk_size == 0
        if prev_seg_last_frame:
            logger.info(
                f"  prev_seg_last_frame_only ON (seg_ar): previous segments' "
                f"video masked to last {tokens_per_frame} tokens (1 latent frame)"
            )

        # Intra-segment per-action masking (current-chunk self-attention). When
        # active, precompute per-segment text offsets [o_0..o_A] and per-segment
        # video sub-windows [[vs,ve,aidx],...] (segment-relative latent frames).
        # None entries -> that segment uses legacy full-visibility attention.
        intra_seg_active = self._intra_seg_enabled()
        intra_seg_txt_offsets_per_seg: List[Optional[List[int]]] = []
        intra_seg_vid_windows_per_seg: List[Optional[List[List[int]]]] = []

        for i, prompt in enumerate(prompts):
            enc = self._encode_prompt(
                prompt,
                negative_prompt=negative_prompt,
                encode_negative=needs_guidance,
            )
            encoded_prompts.append(enc)
            a_start = (
                self._compute_action_start(prompt)
                if cross_segment_text_head_only
                else -1
            )
            action_starts.append(a_start)

            offs = None
            windows = None
            if intra_seg_active:
                try:
                    offs = self._compute_action_text_offsets(prompt)
                    if offs is not None:
                        windows = self._compute_action_video_windows(
                            prompt,
                            seg_latent_start=latent_boundaries[i],
                            seg_latent_end=latent_boundaries[i + 1],
                        )
                        if windows is None:
                            offs = None  # no per-action video footprint -> legacy
                except ValueError as _e:
                    logger.warning(
                        f"  [{i}] intra-seg masking disabled for this segment: {_e}"
                    )
                    offs, windows = None, None
            intra_seg_txt_offsets_per_seg.append(offs)
            intra_seg_vid_windows_per_seg.append(windows)
            logger.info(
                f"  [{i}] Encoded (cross_segment_head_only="
                f"{cross_segment_text_head_only}, action_start={a_start}, "
                f"intra_seg={'on' if offs is not None else 'off'}): {prompt[:80]}..."
            )

        # Cached text-block action-tail ranges, in absolute cache-token coords:
        #   list of (seg_idx, tail_start, tail_end).
        # A later segment's chunk must NOT attend any tail whose seg_idx != its
        # own; we toggle these positions in cache_mask_list before each chunk's
        # denoise (and restore them afterwards so cache writes stay intact).
        cached_action_tails = (
            self._interactive_cached_action_tails if continuing_session else []
        )

        # Cached previous-segment VIDEO token ranges, in absolute cache-token
        # coords: list of (seg_idx, vid_start, vid_end). When
        # prev_seg_last_frame is on, every chunk masks these ranges down to
        # their final tokens_per_frame tokens for all seg_idx != current, so the
        # current chunk's video AND text only attend previous segments' last
        # latent frame. Video keys are otherwise always valid; unlike the action
        # tails there is nothing to restore (a cached segment is "previous" for
        # every later chunk), so re-applying the mask each chunk is idempotent.
        cached_seg_video_ranges = (
            self._interactive_cached_seg_video_ranges if continuing_session else []
        )

        # All text encoding is complete. Keep camera projections and every
        # denoise/cache-write forward in one DiT residency phase, before VAE.
        with self._dit_denoise_residency():
            # ========== Camera control: preprocess full keyboard once ==========
            # Mirrors the one_shot (bidirectional) pipeline: build the full-clip
            # camera tensors and preprocess them into the per-latent-frame keyboard
            # embedding ONCE. Each chunk then slices the keyboard for its own latent
            # frames and injects it per double block via ActionModule
            # (forward_with_cache). Only action_attn / extrinsics_attn are supported
            # in the AR cache path; residual_norm is not (it adds a token-aligned
            # plucker residual to img before SP chunking, which the cache path does
            # not currently thread through).
            camera_embedding_full = self._prepare_camera_keyboard(
                camera_trajectory=camera_trajectory,
                trajectory_translation_scale=trajectory_translation_scale,
                trajectory_rotation_scale=trajectory_rotation_scale,
                num_frames=num_frames,
                latent_h=latent_h,
                latent_w=latent_w,
                segment_frame_boundaries=segment_frame_boundaries,
            )
            cam_mode = getattr(self.transformer, "camera_injection_mode", None)

            out_channels = self.transformer.config.out_channels
            save_latents_path = os.environ.get(
                "HY_INFER_SAVE_FINAL_LATENTS_PATH", ""
            ).strip()
            retain_output_latents = (
                not segment_vae_decode or segment_vae_uses_pixelumm or bool(save_latents_path)
            )
            output_latents = (
                torch.zeros(
                    [batch_size, out_channels, latent_frames, latent_h, latent_w],
                    dtype=self.target_dtype,
                    device=self.device,
                )
                if retain_output_latents
                else None
            )

            total_denoise_time = 0.0
            total_cache_time = 0.0
            total_segment_vae_decode_time = 0.0
            total_speculative_padding_latents = 0
            first_decoded_frame_time = None
            generation_start = time.time()

            from hymm.utils.segment_attention_profiler import (
                seg_attn_profiler as _seg_profiler,
            )

            _seg_profiler.set_total_steps(num_inference_steps)
            _seg_profiler._profiling_active = False

            segments_with_cached_txt = set()
            segment_cache_start = self.cache_valid_len
            segment_added_tokens = 0
            segment_added_latent_frames = 0
            segment_noise_rng_states = {}

            for chunk_idx, (seg_idx, chunk_latent_start, chunk_latent_end) in enumerate(
                chunk_schedule
            ):
                chunk_start_time = time.time()
                chunk_latent_frames = chunk_latent_end - chunk_latent_start
                is_last_chunk_of_seg = (
                    chunk_idx + 1 == len(chunk_schedule)
                    or chunk_schedule[chunk_idx + 1][0] != seg_idx
                )
                is_last_chunk = chunk_idx == total_chunks - 1
                should_cache_chunk = not is_last_chunk or preserve_kv_cache
                global_seg_idx = session_segment_offset + seg_idx

                logger.info(f"\n{'=' * 60}")
                logger.info(
                    f"Chunk {chunk_idx + 1}/{total_chunks} (segment {seg_idx + 1}/{num_segments})"
                )
                logger.info(
                    f"  Latent frames: [{chunk_latent_start}:{chunk_latent_end}] ({chunk_latent_frames} frames)"
                )
                logger.info(
                    f"  Cache valid: {self.cache_valid_len} tokens, global_frame_counter={self.global_frame_counter}"
                )
                logger.info(f"{'=' * 60}")

                chunk_noise = all_noise[:, :, chunk_latent_start:chunk_latent_end, :, :]

                if chunk_idx == 1 and _seg_profiler.enabled:
                    _ps = self.transformer.patch_size
                    _seg_profiler.tokens_per_frame = (chunk_noise.shape[3] // _ps[1]) * (
                        chunk_noise.shape[4] // _ps[2]
                    )
                    _seg_profiler._profiling_active = True

                enc = encoded_prompts[seg_idx]
                text_states = enc["text_states"]
                text_mask = enc["text_mask"]

                # Whether this segment's text is already present in the KV cache
                # (written by an earlier chunk of the same segment). For the first
                # chunk of a segment the text is new -> it must be attended (and
                # later cached). For subsequent chunks the text is already in the
                # cache, so re-appending it as the current chunk's text would
                # duplicate the segment's text keys in attention (a train/inference
                # mismatch: training keeps exactly one text block per segment).
                is_first_chunk_of_seg = seg_idx not in segments_with_cached_txt
                if preserve_kv_cache and is_first_chunk_of_seg:
                    segment_noise_rng_states[seg_idx] = (
                        noise_growth_rollback if chunk_idx == 0 and noise_growth_rollback is not None
                        else (generator.get_state().clone(), int(self._interactive_noise_bank.shape[2]))
                    )

                if do_camera_cfg:
                    text_states_cfg = torch.cat(
                        [enc["neg_states"], enc["neg_states"], text_states], dim=0
                    )
                    text_mask_cfg = torch.cat(
                        [enc["neg_mask"], enc["neg_mask"], text_mask], dim=0
                    )
                elif do_text_cfg:
                    text_states_cfg = torch.cat([enc["neg_states"], text_states], dim=0)
                    text_mask_cfg = torch.cat([enc["neg_mask"], text_mask], dim=0)
                else:
                    text_states_cfg = text_states
                    text_mask_cfg = text_mask

                # ---- Intra-segment per-action masking for THIS chunk ----
                # Only the first chunk of a segment carries the segment text in the
                # current chunk (cache_txt=True); subsequent chunks read it from the
                # cache and the attention helper is inert (its gate requires
                # cache_txt). Build per-CFG-sample text offsets: the negative (uncond)
                # sample has no actions -> None (legacy full attention); the positive
                # sample gets the segment offsets. Video windows are shared (the
                # helper only applies them when that sample's offsets is not None).
                chunk_intra_offsets = None
                chunk_intra_windows = None
                if intra_seg_active and is_first_chunk_of_seg:
                    seg_offs = intra_seg_txt_offsets_per_seg[seg_idx]
                    seg_wins = intra_seg_vid_windows_per_seg[seg_idx]
                    if seg_offs is not None and seg_wins is not None:
                        chunk_intra_windows = seg_wins
                        if do_camera_cfg:
                            chunk_intra_offsets = [None, None, seg_offs]
                        elif do_text_cfg:
                            chunk_intra_offsets = [None, seg_offs]
                        else:
                            chunk_intra_offsets = [seg_offs]

                # ---- Camera: slice this chunk's keyboard + per-segment gate ----
                chunk_cam_emb, chunk_cam_mask = self._slice_chunk_camera(
                    camera_embedding_full=camera_embedding_full,
                    cam_mode=cam_mode,
                    chunk_latent_start=chunk_latent_start,
                    chunk_latent_end=chunk_latent_end,
                    seg_idx=seg_idx,
                    segment_has_camera=segment_has_camera,
                    guidance_branches=guidance_branches,
                )

                latents = chunk_noise[:, :out_channels, ...].clone()

                solver_type = getattr(self.args.inference, "solver_type", "euler")
                if solver_type != "euler":
                    raise ValueError(
                        f"Unsupported inference solver: {solver_type!r}; expected 'euler'"
                    )
                scheduler = EulerSolver(
                    shift=flow_shift,
                    reverse=True,
                    solver="euler",
                )
                logger.info("  Solver: Euler")
                scheduler.set_timesteps(
                    num_inference_steps=num_inference_steps, device=self.device
                )
                timesteps = scheduler.timesteps

                # Conditional-only companion latent for the cache write. It needs its
                # own solver instance. Euler carries a step index, so stepping one
                # object twice per denoising step would corrupt the guided trajectory.
                if cache_latent_cond_only:
                    cache_scheduler = EulerSolver(
                        shift=flow_shift,
                        reverse=True,
                        solver="euler",
                    )
                    cache_scheduler.set_timesteps(
                        num_inference_steps=num_inference_steps, device=self.device
                    )
                    cache_cond_latents = latents.clone()
                else:
                    cache_scheduler = None
                    cache_cond_latents = None

                cache_snapshot_latents = None
                cache_snapshot_t = None

                # ---- Action-tail visibility (train/inference parity) ----
                # Training (magi_mp_causal) hides each segment's contained_actions
                # text tail from EVERY other segment's video. Replicate that here by
                # toggling the cached tail tokens in cache_mask_list: a tail is
                # visible (True) only to its OWN segment's chunks, masked (False) for
                # all others. Re-applied every chunk; cache writes use the genuine
                # text_mask and are corrected at the next chunk's top.
                if cached_action_tails and self.cache_mask_list is not None:
                    for _si, _ts, _te, _tvalid in cached_action_tails:
                        if _si == global_seg_idx:
                            for _cm in self.cache_mask_list:
                                _cm[:, _ts:_te] = _tvalid.to(_cm.device)
                        else:
                            for _cm in self.cache_mask_list:
                                _cm[:, _ts:_te] = False

                # ---- Previous-segment last-frame-only video (train/inference parity) ----
                # Mirror the training prev_seg_last_frame_only mask: mask every
                # PREVIOUS segment's cached video keys down to only their final
                # tokens_per_frame tokens (one latent frame). The CURRENT segment's
                # own chunks are not yet in the cache (its video is attended live),
                # so only seg_idx != _si is restricted. This is query-agnostic, so it
                # applies to both the current chunk's video and text queries. Video
                # tokens carry no padding, so we simply zero the non-final positions.
                if (
                    prev_seg_last_frame
                    and cached_seg_video_ranges
                    and self.cache_mask_list is not None
                ):
                    for _si, _vs, _ve in cached_seg_video_ranges:
                        if _si == global_seg_idx:
                            continue  # own segment (shouldn't be cached yet, but be safe)
                        keep_start = max(_vs, _ve - tokens_per_frame)
                        for _cm in self.cache_mask_list:
                            _cm[:, _vs:keep_start] = False
                            _cm[:, keep_start:_ve] = True

                denoise_start = time.time()
                with stage("denoise"):
                    for step_idx, t in enumerate(timesteps):
                        self.transformer._sparse_step_index = step_idx
                        if chunk_idx == 1:
                            _seg_profiler.new_step(step_idx, t.item())
                        t_expand = t.expand(batch_size).to(self.device)

                        latent_input = latents

                        # Pad latents to in_channels before feeding into transformer
                        if latent_input.shape[1] < self.in_channels:
                            pad = torch.zeros(
                                [
                                    *latent_input.shape[:1],
                                    self.in_channels - latent_input.shape[1],
                                    *latent_input.shape[2:],
                                ],
                                device=latent_input.device,
                                dtype=latent_input.dtype,
                            )
                            latent_input = torch.cat([latent_input, pad], dim=1)

                        if needs_guidance:
                            latent_input = latent_input.repeat(guidance_branches, 1, 1, 1, 1)
                            t_expand = t_expand.repeat(guidance_branches)

                        with torch.autocast(device_type="cuda", dtype=self.target_dtype):
                            output = self.transformer.forward_with_cache(
                                hidden_states=latent_input,
                                timestep=t_expand,
                                text_states=text_states_cfg,
                                text_mask=text_mask_cfg,
                                kv_cache_list=self.kv_cache,
                                cache_valid_len=self.cache_valid_len,
                                update_cache=False,
                                global_frame_counter=self.global_frame_counter,
                                guidance=None,
                                cache_txt=is_first_chunk_of_seg,
                                cache_mask_list=self.cache_mask_list,
                                camera_embedding=chunk_cam_emb,
                                camera_has_mask=chunk_cam_mask,
                                intra_seg_txt_offsets=chunk_intra_offsets,
                                intra_seg_vid_windows=chunk_intra_windows,
                                camera_rope_time_offset=(
                                    self.global_frame_counter
                                    if self.use_global_camera_rope_offset
                                    else 0
                                ),
                            )
                            noise_pred = output[0]

                        if needs_guidance:
                            gs = guidance_scale
                            if cfg_schedule and cfg_schedule != "constant":
                                gs = get_cfg_scale_for_step(
                                    cfg_schedule,
                                    step_idx,
                                    num_inference_steps,
                                    guidance_scale,
                                    cfg_skip_steps=cfg_skip_steps,
                                )
                            if do_camera_cfg:
                                (
                                    noise_pred_base,
                                    noise_pred_camera,
                                    noise_pred_cond,
                                ) = noise_pred.chunk(3)
                                noise_pred = combine_camera_text_cfg(
                                    noise_pred_base,
                                    noise_pred_camera,
                                    noise_pred_cond,
                                    camera_guidance_scale,
                                    gs,
                                )
                            else:
                                noise_pred_base, noise_pred_cond = noise_pred.chunk(2)
                                noise_pred = noise_pred_base + gs * (
                                    noise_pred_cond - noise_pred_base
                                )

                        latents = scheduler.step(
                            model_output=noise_pred,
                            timestep=t,
                            sample=latents,
                            return_dict=False,
                        )[0].to(self.target_dtype)

                        if cache_cond_latents is not None:
                            # Same timestep grid, same conditional prediction the CFG
                            # combination above consumed -- only the guidance term is
                            # left out, so this latent carries the segment's content
                            # without the amplification training never cached.
                            cache_cond_latents = cache_scheduler.step(
                                model_output=noise_pred_cond,
                                timestep=t,
                                sample=cache_cond_latents,
                                return_dict=False,
                            )[0].to(self.target_dtype)

                        if (
                            self.cache_at_step >= 0
                            and step_idx == self.cache_at_step
                            and should_cache_chunk
                        ):
                            cache_snapshot_latents = (
                                cache_cond_latents
                                if cache_cond_latents is not None
                                else latents
                            ).clone()
                            # ``latents`` is the result *after* scheduler.step(), so it
                            # lives at the next sigma, not at the model-input timestep
                            # ``t`` used for this denoising step. Keep the timestep
                            # embedding used to write KV aligned with that post-step
                            # latent state.
                            post_step_sigma = scheduler.sigmas[scheduler.step_index]
                            cache_snapshot_t = (
                                post_step_sigma * scheduler.config.num_train_timesteps
                            ).to(device=t.device, dtype=t.dtype)
                            logger.info(
                                f"  Snapshot for cache at step {step_idx}, "
                                f"post_step_timestep={cache_snapshot_t.item():.1f}"
                            )

                        if (step_idx + 1) % 10 == 0 or step_idx == 0:
                            logger.info(f"  Step {step_idx + 1}/{num_inference_steps}")

                torch.cuda.synchronize()
                denoise_end = time.time()
                total_denoise_time += denoise_end - denoise_start

                if chunk_idx == 1 and _seg_profiler.data:
                    _profile_dir = os.environ.get(
                        "PROFILE_SEGMENT_ATTENTION_VIS_DIR", "./outputs"
                    )
                    os.makedirs(_profile_dir, exist_ok=True)
                    _plot_idx = getattr(_seg_profiler, "_plot_counter", 0)
                    _seg_profiler._plot_counter = _plot_idx + 1
                    _seg_profiler.plot_and_save(
                        os.path.join(
                            _profile_dir, f"seg_attn_profile_ar_{_plot_idx:04d}.png"
                        )
                    )
                    _seg_profiler.reset()
                    _seg_profiler._profiling_active = False

                if output_latents is not None:
                    output_latents[:, :, chunk_latent_start:chunk_latent_end, :, :] = (
                        latents
                    )

                cache_start = time.time()

                with stage("cache_write"):
                    if should_cache_chunk:
                        if self.max_cache_segments is not None:
                            while len(self.segment_token_lengths) >= self.max_cache_segments:
                                _evicted = (
                                    self.segment_token_lengths[0]
                                    if self.segment_token_lengths
                                    else 0
                                )
                                self._evict_oldest_segment()
                                segment_cache_start = max(0, segment_cache_start - _evicted)
                                # Shift cached action-tail offsets left by the evicted
                                # token count; drop any tail that fell out of the cache.
                                if _evicted > 0 and cached_action_tails:
                                    _shifted = []
                                    for _si, _ts, _te, _tvalid in cached_action_tails:
                                        _ts2, _te2 = _ts - _evicted, _te - _evicted
                                        if _te2 <= 0:
                                            continue  # fully evicted
                                        if _ts2 < 0:
                                            # Front partially evicted: trim the valid mask
                                            # to stay aligned with the surviving tokens.
                                            _tvalid = _tvalid[:, -_ts2:]
                                            _ts2 = 0
                                        _shifted.append((_si, _ts2, _te2, _tvalid))
                                    cached_action_tails = _shifted

                                # Same left-shift for cached video ranges; drop any that
                                # fell fully out of the cache, clamp a partial front.
                                if _evicted > 0 and cached_seg_video_ranges:
                                    _shifted_v = []
                                    for _si, _vs, _ve in cached_seg_video_ranges:
                                        _vs2, _ve2 = _vs - _evicted, _ve - _evicted
                                        if _ve2 <= 0:
                                            continue  # fully evicted
                                        if _vs2 < 0:
                                            _vs2 = 0
                                        _shifted_v.append((_si, _vs2, _ve2))
                                    cached_seg_video_ranges = _shifted_v

                        if is_first_chunk_of_seg:
                            segment_cache_start = self.cache_valid_len

                        if cache_snapshot_latents is not None:
                            cache_latents = cache_snapshot_latents
                            cache_t = cache_snapshot_t.expand(batch_size).to(self.device)
                            logger.info(
                                f"  Using intermediate latents for cache "
                                f"(step {self.cache_at_step}, t={cache_snapshot_t.item():.1f})"
                            )
                        else:
                            cache_latents = (
                                cache_cond_latents
                                if cache_cond_latents is not None
                                else latents
                            )
                            cache_t = torch.zeros(
                                [batch_size], device=self.device, dtype=torch.long
                            )

                        if self.cache_noise_sigma > 0.0:
                            # Use the CPU generator shared by the whole rollout so every
                            # sequence-parallel rank writes identical noisy history.
                            cache_eps = torch.randn(
                                cache_latents.shape,
                                generator=generator,
                                dtype=cache_latents.dtype,
                            ).to(device=cache_latents.device)
                            sigma = self.cache_noise_sigma
                            cache_latents = (
                                (1.0 - sigma) * cache_latents.float()
                                + sigma * cache_eps.float()
                            ).to(cache_latents.dtype)
                            cache_t = torch.full(
                                [batch_size],
                                sigma * 1000.0,
                                device=self.device,
                                dtype=torch.float32,
                            )
                            logger.info(
                                f"  Cache noised: sigma={sigma:.6f}, "
                                f"cache_t={sigma * 1000.0:.1f}"
                            )

                        cache_input = cache_latents

                        # Pad cache_input to in_channels before feeding into transformer
                        if cache_input.shape[1] < self.in_channels:
                            pad = torch.zeros(
                                [
                                    *cache_input.shape[:1],
                                    self.in_channels - cache_input.shape[1],
                                    *cache_input.shape[2:],
                                ],
                                device=cache_input.device,
                                dtype=cache_input.dtype,
                            )
                            cache_input = torch.cat([cache_input, pad], dim=1)

                        cache_write_text_states = text_states_cfg
                        cache_write_text_mask = text_mask_cfg
                        cache_write_intra_offsets = chunk_intra_offsets
                        cache_write_cam_emb = chunk_cam_emb
                        cache_write_cam_mask = chunk_cam_mask
                        if single_slot_cfg_cache:
                            # seg_ar has exactly one chunk per segment. Persist only the
                            # fully conditioned final branch; the other branches exist
                            # solely for the current CFG prediction.
                            cache_write_text_states = text_states
                            cache_write_text_mask = text_mask
                            if chunk_intra_offsets is not None:
                                cache_write_intra_offsets = [chunk_intra_offsets[-1]]
                            if chunk_cam_emb is not None:
                                cache_write_cam_emb = chunk_cam_emb[-batch_size:].contiguous()
                            if chunk_cam_mask is not None:
                                cache_write_cam_mask = chunk_cam_mask[-batch_size:].contiguous()
                        elif needs_guidance:
                            cache_input = cache_input.repeat(guidance_branches, 1, 1, 1, 1)
                            cache_t = cache_t.repeat(guidance_branches)

                        if swa_archived:
                            for cache in self.kv_cache:
                                cache["swa_segment_index"] = global_seg_idx
                                cache["swa_action_start"] = action_starts[seg_idx]

                        with torch.autocast(device_type="cuda", dtype=self.target_dtype):
                            _, new_tokens = self.transformer.forward_with_cache(
                                hidden_states=cache_input,
                                timestep=cache_t,
                                text_states=cache_write_text_states,
                                text_mask=cache_write_text_mask,
                                kv_cache_list=self.kv_cache,
                                cache_valid_len=self.cache_valid_len,
                                update_cache=True,
                                global_frame_counter=self.global_frame_counter,
                                cache_txt=is_first_chunk_of_seg,
                                cache_mask_list=self.cache_mask_list,
                                camera_embedding=cache_write_cam_emb,
                                camera_has_mask=cache_write_cam_mask,
                                intra_seg_txt_offsets=cache_write_intra_offsets,
                                intra_seg_vid_windows=chunk_intra_windows,
                                camera_rope_time_offset=(
                                    self.global_frame_counter
                                    if self.use_global_camera_rope_offset
                                    else 0
                                ),
                            )

                        # Keep one canonical generated history between prompt segments.
                        # Branch-local KV is preserved within a multi-chunk segment so
                        # its later chunks still receive cond/uncond text respectively.
                        # At the segment boundary the fully conditioned branch becomes
                        # the actual history seen by all branches in the next segment.
                        if (
                            needs_guidance
                            and not single_slot_cfg_cache
                            and is_last_chunk_of_seg
                        ):
                            cache_write_start = segment_cache_start
                            cache_write_end = self.cache_valid_len + int(new_tokens)
                            canonical_history = slice(
                                (guidance_branches - 1) * batch_size,
                                guidance_branches * batch_size,
                            )
                            for cache in self.kv_cache:
                                for branch_idx in range(guidance_branches - 1):
                                    branch_history = slice(
                                        branch_idx * batch_size,
                                        (branch_idx + 1) * batch_size,
                                    )
                                    cache["k"][
                                        branch_history, cache_write_start:cache_write_end
                                    ].copy_(
                                        cache["k"][
                                            canonical_history,
                                            cache_write_start:cache_write_end,
                                        ]
                                    )
                                    cache["v"][
                                        branch_history, cache_write_start:cache_write_end
                                    ].copy_(
                                        cache["v"][
                                            canonical_history,
                                            cache_write_start:cache_write_end,
                                        ]
                                    )
                            for cache_mask in self.cache_mask_list:
                                for branch_idx in range(guidance_branches - 1):
                                    branch_history = slice(
                                        branch_idx * batch_size,
                                        (branch_idx + 1) * batch_size,
                                    )
                                    cache_mask[
                                        branch_history, cache_write_start:cache_write_end
                                    ].copy_(
                                        cache_mask[
                                            canonical_history,
                                            cache_write_start:cache_write_end,
                                        ]
                                    )
                            logger.info(
                                "  Canonical CFG history synchronized from fully "
                                "conditioned branch "
                                f"at segment {seg_idx} boundary for cache tokens "
                                f"[{cache_write_start}:{cache_write_end}]."
                            )

                        if swa_archived:
                            from hymm.models.dit.modules.uniform_history_cache import selected_action_tails

                            first_cache = self.kv_cache[0]
                            lengths = {cache["swa_valid_len"] for cache in self.kv_cache}
                            if len(lengths) != 1 or any(
                                cache["swa_frame_ids"] != first_cache["swa_frame_ids"]
                                or cache["swa_uniform_layout"] != first_cache["swa_uniform_layout"]
                                for cache in self.kv_cache
                            ):
                                raise RuntimeError("Sampled history selection disagrees across layers")
                            self.cache_valid_len = lengths.pop()
                            cached_action_tails = selected_action_tails(
                                first_cache["swa_uniform_layout"], self.cache_mask_list[0])
                            cached_seg_video_ranges = []
                            segments_with_cached_txt.add(seg_idx)
                            self.segment_token_lengths.clear()
                            archive_bytes = sum(cache["swa_archive_bytes"] for cache in self.kv_cache)
                            logger.info(
                                f"{swa_sampling} history: total_frames={first_cache['swa_total_frames']}, "
                                f"selected_frames={len(first_cache['swa_frame_ids'])}, "
                                f"frame_ids={list(first_cache['swa_frame_ids'])}, "
                                f"valid_tokens={self.cache_valid_len}, gpu_capacity={swa_capacity}, "
                                f"cpu_archive_gib={archive_bytes / 2**30:.3f}"
                            )
                            new_tokens = 0
                        elif swa_enabled:
                            old_valid_len = self.cache_valid_len
                            cut = self.kv_cache[0]["swa_evicted"]
                            shifted_tails = []
                            for si, start, end, valid in cached_action_tails:
                                if end <= cut:
                                    continue
                                trim = max(0, cut - start)
                                shifted_tails.append(
                                    (si, max(0, start - cut), end - cut, valid[:, trim:])
                                )
                            cached_action_tails = shifted_tails
                            cached_seg_video_ranges = [
                                (si, max(0, start - cut), end - cut)
                                for si, start, end in cached_seg_video_ranges if end > cut
                            ]
                            lengths = {cache["swa_valid_len"] for cache in self.kv_cache}
                            if len(lengths) != 1:
                                raise RuntimeError("SWA cache lengths disagree across layers")
                            self.cache_valid_len = lengths.pop()
                            text_count = int(cache_write_text_states.shape[1])
                            image_count = int(new_tokens) - text_count
                            video_start = max(0, old_valid_len - cut)
                            video_end = old_valid_len + image_count - cut
                            if video_end > video_start:
                                if prev_seg_last_frame:
                                    cached_seg_video_ranges.append(
                                        (global_seg_idx, video_start, video_end)
                                    )
                                action_start = action_starts[seg_idx]
                                if action_start is not None and action_start >= 0:
                                    cached_action_tails.append((
                                        global_seg_idx, video_end + int(action_start),
                                        video_end + text_count,
                                        cache_write_text_mask[:, int(action_start):].bool().clone(),
                                    ))
                            segments_with_cached_txt.add(seg_idx)
                            self.segment_token_lengths.clear()
                            logger.info(
                                f"SWA cache: valid={self.cache_valid_len}, capacity={swa_capacity}, "
                                f"evicted={self.kv_cache[0]['swa_evicted']}"
                            )
                            new_tokens = 0

                        if is_first_chunk_of_seg and not swa_enabled:
                            segments_with_cached_txt.add(seg_idx)
                            logger.info(
                                f"  Cached img+txt KV (first chunk of segment {seg_idx})"
                            )

                            # Record this segment's cached VIDEO token range (absolute
                            # cache coords) so prev_seg_last_frame can later mask it down
                            # to its final latent frame for subsequent segments. In
                            # seg_ar (chunk_size == 0) each segment is a single chunk, so
                            # this one img block spans the whole segment's video.
                            if prev_seg_last_frame:
                                s_enc_v = int(cache_write_text_states.shape[1])
                                img_tokens_v = int(new_tokens) - s_enc_v
                                if img_tokens_v > 0:
                                    vid_start = self.cache_valid_len
                                    vid_end = self.cache_valid_len + img_tokens_v
                                    cached_seg_video_ranges.append(
                                        (global_seg_idx, vid_start, vid_end)
                                    )
                                    logger.info(
                                        f"  Video range cached for segment {seg_idx}: "
                                        f"cache tokens [{vid_start}:{vid_end}] "
                                        f"({img_tokens_v} tokens) -> restricted to last "
                                        f"{tokens_per_frame} for later segments"
                                    )

                            # Record this segment's action-tail token range in absolute
                            # cache coords so later segments can be masked off it. The
                            # txt block (length S_enc, post-crop coords matching
                            # action_start) is written AFTER this chunk's img tokens:
                            #   img: [cvl_before, cvl_before + img_tokens)
                            #   txt: [cvl_before + img_tokens, cvl_before + new_tokens)
                            a_start = (
                                action_starts[seg_idx] if seg_idx < len(action_starts) else -1
                            )
                            if a_start is not None and a_start >= 0:
                                s_enc = int(cache_write_text_states.shape[1])
                                img_tokens = int(new_tokens) - s_enc
                                if img_tokens >= 0:
                                    txt_block_start = self.cache_valid_len + img_tokens
                                    tail_start = txt_block_start + int(a_start)
                                    tail_end = txt_block_start + s_enc
                                    if tail_end > tail_start:
                                        # Genuine validity of the tail tokens (excludes
                                        # text padding) so that when this segment's own
                                        # chunks make the tail visible we restore the
                                        # real mask rather than un-masking padding.
                                        tail_valid = (
                                            cache_write_text_mask[:, int(a_start) : s_enc]
                                            .bool()
                                            .clone()
                                        )
                                        cached_action_tails.append(
                                            (
                                                global_seg_idx,
                                                tail_start,
                                                tail_end,
                                                tail_valid,
                                            )
                                        )
                                        logger.info(
                                            f"  Action-tail cached for segment {seg_idx}: "
                                            f"cache tokens [{tail_start}:{tail_end}] "
                                            f"(action_start={a_start}, S_enc={s_enc}) "
                                            f"-> hidden from later segments' video"
                                        )
                        elif not swa_enabled:
                            logger.info(
                                f"  Cached img KV only (segment {seg_idx} text already cached)"
                            )

                        if getattr(self, "qvg_kv_storage", False) and not swa_enabled:
                            expected_len = self.cache_valid_len + int(new_tokens)
                            if any(cache["qvg_valid_len"] != expected_len for cache in self.kv_cache):
                                raise RuntimeError("QVG full-history cache lengths disagree across layers")
                        self.cache_valid_len += new_tokens
                        if not swa_enabled:
                            self.segment_token_lengths.append(new_tokens)
                        self.global_frame_counter += chunk_latent_frames
                        segment_added_tokens += int(new_tokens)
                        segment_added_latent_frames += int(chunk_latent_frames)
                        if is_last_chunk_of_seg and preserve_kv_cache:
                            self._interactive_segment_cache_tokens.append(segment_added_tokens)
                            self._interactive_segment_latent_lengths.append(
                                segment_added_latent_frames
                            )
                            self._interactive_segment_noise_rng_states.append(
                                segment_noise_rng_states.pop(seg_idx)
                            )
                            segment_added_tokens = 0
                            segment_added_latent_frames = 0
                        logger.info(
                            f"  Cache updated: +{new_tokens} tokens, "
                            f"total={self.cache_valid_len}, cached_chunks={len(self.segment_token_lengths)}"
                        )
                    else:
                        self.global_frame_counter += chunk_latent_frames

                torch.cuda.synchronize()
                cache_end = time.time()
                total_cache_time += cache_end - cache_start

                segment_decode_elapsed = 0.0
                if segment_vae_decode and is_last_chunk_of_seg:
                    # Keep only the finalized guided latent plus persistent AR
                    # state while VAE runs. These per-forward tensors would
                    # otherwise overlap the decoder activation peak.
                    noise_pred = None
                    output = None
                    latent_input = None
                    cache_input = None
                    cache_latents = None
                    cache_snapshot_latents = None
                    cache_cond_latents = None
                    text_states_cfg = None
                    text_mask_cfg = None
                    chunk_cam_emb = None
                    chunk_cam_mask = None
                    cache_write_text_states = None
                    cache_write_text_mask = None
                    cache_write_intra_offsets = None
                    cache_write_cam_emb = None
                    cache_write_cam_mask = None
                    phase_released_bytes = self._prepare_interactive_decode_phase()
                    segment_decode_start = time.time()
                    emitted_frames = 0
                    pending_latent_frames = 0
                    speculative_padding_latents = 0
                    speculative_cache_offload_bytes = 0
                    decode_error = None
                    if segment_vae_uses_pixelumm:
                        try:
                            if pixelumm_stable_streaming:
                                # The AR latent boundaries are derived from
                                # absolute pixel boundaries with floor division.
                                # Therefore an individual source segment can span
                                # more than ``4 * segment_latents`` pixels when its
                                # start/end timestamps have different modulo-4
                                # residues (for example 86 latents and 346 pixels).
                                # PixelUMM streaming pushes, however, each expose at most
                                # ``time_compression_ratio * segment_latents``
                                # decoded pixels. Align intermediate visible ends
                                # to the canonical latent timeline and let only the
                                # final push absorb the video's global 0-3 frame
                                # tail. This preserves the exact final frame count
                                # while keeping every push within PixelUMM's valid range.
                                temporal_ratio = int(
                                    getattr(self.vae, "time_compression_ratio", 4)
                                )
                                target_pixel_end = pixelumm_request_pixel_offset + (
                                    int(num_frames)
                                    if is_last_chunk
                                    else min(
                                        int(num_frames),
                                        int(chunk_latent_end) * temporal_ratio,
                                    )
                                )
                                segment_pixel_frames = target_pixel_end - int(
                                    self._interactive_pixelumm_decode_session.available_pixel_frames
                                )
                                # ``push_preview`` returns the complete current
                                # segment plus the previous provisional overlap,
                                # corrected with the new segment's future context.
                                # The returned range therefore replaces the old
                                # tail in-place and appends the new segment.
                                # Only intermediate pushes are padded/held back; the last
                                # push remains equal to full-clip decode.
                                self._interactive_pixelumm_decode_session.final_segment = is_last_chunk
                                held_pixel_frames = (
                                    0 if is_last_chunk else self.vae.hold_back_pixel_frames
                                )
                                preview_update = (
                                    self._interactive_pixelumm_decode_session.push_preview(
                                        self._prepare_vae_decode_latents(latents.detach()),
                                        pixel_frames=segment_pixel_frames,
                                    )
                                )
                                self._last_pixelumm_streaming_metadata = {
                                    "replace_pixel_start": pixelumm_output_pixel_start,
                                    "replace_pixel_frames": (
                                        pixelumm_request_pixel_offset - pixelumm_output_pixel_start
                                    ),
                                    "stable_pixel_frames": int(
                                        preview_update.stable_pixel_frames
                                    ),
                                    "provisional_pixel_frames": int(
                                        preview_update.provisional_pixel_frames
                                    ),
                                    "withheld_pixel_frames": held_pixel_frames,
                                    "generated_pixel_frames": int(preview_update.pixel_end),
                                }
                                if segment_vae_output_rank:
                                    preview_video_cpu = self._postprocess_vae_video(
                                        preview_update.video
                                    ).cpu()
                                    update_start = (
                                        int(preview_update.pixel_start)
                                        - pixelumm_output_pixel_start
                                    )
                                    update_end = (
                                        int(preview_update.pixel_end)
                                        - pixelumm_output_pixel_start
                                    )
                                    if int(preview_video_cpu.shape[2]) != (
                                        update_end - update_start
                                    ):
                                        raise RuntimeError(
                                            "PixelUMM preview update frame count mismatch: "
                                            f"range=[{update_start}, {update_end}), "
                                            f"video={preview_video_cpu.shape[2]}"
                                        )
                                    if decoded_video_cpu is None:
                                        if update_start != 0:
                                            raise RuntimeError(
                                                "The first PixelUMM preview update must start "
                                                f"at frame 0, got {update_start}."
                                            )
                                        decoded_video_cpu = preview_video_cpu
                                    else:
                                        current_frames = int(decoded_video_cpu.shape[2])
                                        if update_start > current_frames:
                                            raise RuntimeError(
                                                "PixelUMM preview update leaves a gap: "
                                                f"current={current_frames}, "
                                                f"update_start={update_start}."
                                            )
                                        decoded_video_cpu = torch.cat(
                                            [
                                                decoded_video_cpu[
                                                    :, :, :update_start, :, :
                                                ],
                                                preview_video_cpu,
                                            ],
                                            dim=2,
                                        )
                                    decoded_pixel_frames = int(decoded_video_cpu.shape[2])
                                    if decoded_pixel_frames != update_end:
                                        raise RuntimeError(
                                            "PixelUMM assembled preview frame count mismatch: "
                                            f"assembled={decoded_pixel_frames}, "
                                            f"expected={update_end}."
                                        )
                                    emitted_frames = int(preview_video_cpu.shape[2])
                                    emitted_latent_frames = chunk_latent_end
                            else:
                                if output_latents is None:
                                    raise RuntimeError(
                                        "PixelUMM incremental decode requires retained latents."
                                    )
                                preview_num_frames = (
                                    num_frames
                                    if is_last_chunk
                                    else int(segment_frame_boundaries[seg_idx + 1])
                                )
                                cumulative_latents = output_latents[
                                    :, :, :chunk_latent_end, :, :
                                ]
                                decoded_preview = self._decode_latents(
                                    cumulative_latents,
                                    preview_num_frames,
                                )
                                if segment_vae_output_rank:
                                    decoded_video_cpu = decoded_preview.cpu()
                                    decoded_pixel_frames = int(decoded_video_cpu.shape[2])
                                    emitted_frames = max(
                                        decoded_pixel_frames
                                        - (
                                            int(segment_frame_boundaries[seg_idx])
                                            if segment_frame_boundaries is not None
                                            else 0
                                        ),
                                        0,
                                    )
                                    emitted_latent_frames = chunk_latent_end
                            if (
                                segment_vae_output_rank
                                and decoded_pixel_frames > 0
                                and first_decoded_frame_time is None
                            ):
                                first_decoded_frame_time = time.time() - generation_start
                            if pixelumm_stable_streaming:
                                del preview_update
                            else:
                                del decoded_preview
                            torch.cuda.synchronize(self.device)
                        except Exception as exc:  # noqa: BLE001
                            decode_error = exc
                    else:
                        try:
                            prepared_latents = self._prepare_vae_decode_latents(
                                latents.detach()
                            )

                            def _store_decoded_range(
                                decoded_tile,
                                *,
                                latent_start,
                                valid_latent_frames,
                            ):
                                nonlocal decoded_video_cpu, decoded_pixel_frames
                                nonlocal emitted_frames, emitted_latent_frames
                                nonlocal first_decoded_frame_time
                                nonlocal pixel_frames_per_latent
                                latent_end = int(latent_start) + int(valid_latent_frames)
                                if latent_start > emitted_latent_frames:
                                    raise RuntimeError(
                                        "Segment VAE output has a latent gap: "
                                        f"next_start={latent_start}, "
                                        f"emitted={emitted_latent_frames}."
                                    )
                                if decoded_tile.shape[2] % valid_latent_frames != 0:
                                    raise RuntimeError(
                                        "Decoded pixel frames do not map evenly to "
                                        "their real latent frames: "
                                        f"pixels={decoded_tile.shape[2]}, "
                                        f"latents={valid_latent_frames}."
                                    )

                                tile_pixel_frames_per_latent = int(
                                    decoded_tile.shape[2]
                                ) // int(valid_latent_frames)
                                if pixel_frames_per_latent is None:
                                    pixel_frames_per_latent = tile_pixel_frames_per_latent
                                elif (
                                    pixel_frames_per_latent != tile_pixel_frames_per_latent
                                ):
                                    raise RuntimeError(
                                        "DCAE temporal expansion changed between "
                                        "tiles: "
                                        f"expected={pixel_frames_per_latent}, "
                                        "got="
                                        f"{tile_pixel_frames_per_latent}."
                                    )
                                video_tile = self._postprocess_vae_video(decoded_tile)
                                if decoded_video_cpu is None:
                                    decoded_video_cpu = torch.empty(
                                        [
                                            int(video_tile.shape[0]),
                                            int(video_tile.shape[1]),
                                            num_frames,
                                            int(video_tile.shape[3]),
                                            int(video_tile.shape[4]),
                                        ],
                                        dtype=video_tile.dtype,
                                        device="cpu",
                                    )
                                destination_start = (
                                    int(latent_start) * tile_pixel_frames_per_latent
                                )
                                destination_end = min(
                                    destination_start + int(video_tile.shape[2]),
                                    num_frames,
                                )
                                copy_frames = max(
                                    destination_end - destination_start,
                                    0,
                                )
                                if copy_frames > 0:
                                    cpu_tile = video_tile[:, :, :copy_frames, :, :].cpu()
                                    decoded_video_cpu[
                                        :,
                                        :,
                                        destination_start:destination_end,
                                        :,
                                        :,
                                    ].copy_(cpu_tile)
                                    del cpu_tile
                                    decoded_pixel_frames = max(
                                        decoded_pixel_frames,
                                        destination_end,
                                    )

                                previous_emitted_latents = emitted_latent_frames
                                emitted_latent_frames = max(
                                    emitted_latent_frames,
                                    latent_end,
                                )
                                newly_emitted_start = min(
                                    previous_emitted_latents * tile_pixel_frames_per_latent,
                                    num_frames,
                                )
                                newly_emitted_end = min(
                                    emitted_latent_frames * tile_pixel_frames_per_latent,
                                    num_frames,
                                )
                                new_frames = max(
                                    newly_emitted_end - newly_emitted_start,
                                    0,
                                )
                                emitted_frames += new_frames
                                if new_frames > 0 and first_decoded_frame_time is None:
                                    first_decoded_frame_time = (
                                        time.time() - generation_start
                                    )

                            def _collect_canonical_tile(decoded_tile):
                                _store_decoded_range(
                                    decoded_tile,
                                    latent_start=(
                                        segment_vae_session.decoded_latent_frames
                                    ),
                                    valid_latent_frames=(
                                        segment_vae_session.tile_latent_frames
                                    ),
                                )

                            segment_vae_session.push(
                                prepared_latents,
                                output_callback=_collect_canonical_tile,
                            )
                            pending_latent_frames = (
                                segment_vae_session.pending_latent_frames
                            )
                            if pending_latent_frames > 0:
                                speculative_padding_latents = (
                                    segment_vae_session.tile_latent_frames
                                    - pending_latent_frames
                                )
                                total_speculative_padding_latents += (
                                    speculative_padding_latents
                                )
                                preview_start = segment_vae_session.decoded_latent_frames

                                def _collect_speculative_tail(decoded_tail):
                                    _store_decoded_range(
                                        decoded_tail,
                                        latent_start=preview_start,
                                        valid_latent_frames=(pending_latent_frames),
                                    )

                                offload_bytes_before = (
                                    segment_vae_session.speculative_cache_offload_bytes
                                )
                                segment_vae_session.speculative_decode_pending(
                                    output_callback=_collect_speculative_tail,
                                    offload_canonical_cache=(segment_vae_cache_offload),
                                )
                                speculative_cache_offload_bytes = (
                                    segment_vae_session.speculative_cache_offload_bytes
                                    - offload_bytes_before
                                )
                            if is_last_chunk:
                                final_pending_latents = (
                                    segment_vae_session.pending_latent_frames
                                )
                                final_pending_start = (
                                    segment_vae_session.decoded_latent_frames
                                )

                                def _collect_final_canonical_tail(decoded_tail):
                                    _store_decoded_range(
                                        decoded_tail,
                                        latent_start=final_pending_start,
                                        valid_latent_frames=(final_pending_latents),
                                    )

                                segment_vae_session.finish(
                                    output_callback=(
                                        _collect_final_canonical_tail
                                        if final_pending_latents > 0
                                        else None
                                    ),
                                )
                            if (
                                segment_vae_output_rank
                                and emitted_latent_frames
                                != segment_vae_session.received_latent_frames
                            ):
                                raise RuntimeError(
                                    "Speculative segment VAE decode did not emit "
                                    "the complete segment timeline: "
                                    f"emitted={emitted_latent_frames}, "
                                    "received="
                                    f"{segment_vae_session.received_latent_frames}."
                                )
                            del prepared_latents
                            torch.cuda.synchronize(self.device)
                        except Exception as exc:  # noqa: BLE001
                            decode_error = exc

                    try:
                        self._sync_segment_vae_status(
                            decode_error,
                            context=f"segment {seg_idx}",
                            status_tensor=segment_vae_status,
                        )
                    finally:
                        # A decode failure is reported to the caller, but must not
                        # leave PixelUMM resident or skip phase cleanup. The next queued
                        # request can restore its KV working set normally.
                        self._finish_interactive_decode_phase()
                    segment_decode_elapsed = time.time() - segment_decode_start
                    total_segment_vae_decode_time += segment_decode_elapsed
                    if segment_vae_output_rank:
                        if segment_vae_uses_pixelumm:
                            logger.info(
                                "  Segment PixelUMM decode: "
                                f"segment={seg_idx}, emitted_frames="
                                f"{decoded_pixel_frames}, wall_s="
                                f"{segment_decode_elapsed:.3f}, "
                                f"phase_released_gib="
                                f"{phase_released_bytes / (1024**3):.2f}, "
                                f"stable_prefix={getattr(self._interactive_pixelumm_decode_session, 'stable_pixel_frames', decoded_pixel_frames)}, "
                                f"provisional_tail={getattr(self._interactive_pixelumm_decode_session, 'withheld_pixel_frames', 0)}, "
                                f"replace_frames={int((self._last_pixelumm_streaming_metadata or {}).get('replace_pixel_frames', 0))}, "
                                f"peak_allocated_gib="
                                f"{torch.cuda.max_memory_allocated(self.device) / (1024**3):.2f}"
                            )
                        else:
                            logger.info(
                                "  Segment VAE decode: "
                                f"segment={seg_idx}, emitted_frames={emitted_frames}, "
                                f"pending_latents={pending_latent_frames}, "
                                f"speculative_pad_latents="
                                f"{speculative_padding_latents}, "
                                f"feature_cache_strategy="
                                f"{'cpu_roundtrip' if segment_vae_cache_offload else 'gpu_copy'}, "
                                f"cache_roundtrip_gib="
                                f"{speculative_cache_offload_bytes / (1024**3):.2f}, "
                                f"wall_s={segment_decode_elapsed:.3f}, "
                                f"peak_allocated_gib="
                                f"{torch.cuda.max_memory_allocated(self.device) / (1024**3):.2f}"
                            )
                        if segment_callback is not None:
                            try:
                                # The callback receives a CPU view of the complete
                                # timeline decoded so far.  It is intentionally
                                # invoked only on the SP output rank and only after a
                                # segment boundary, so web callers can publish a
                                # playable cumulative preview without affecting the
                                # distributed denoising path on the other ranks.
                                held = 0 if is_last_chunk else getattr(self.vae, "hold_back_pixel_frames", 0)
                                segment_callback(
                                    seg_idx,
                                    num_segments,
                                    prompts[seg_idx],
                                    decoded_video_cpu[:, :, :max(decoded_pixel_frames - held, 0), :, :],
                                )
                            except Exception:  # noqa: BLE001
                                logger.exception(
                                    "Segment callback failed for segment "
                                    f"{seg_idx + 1}/{num_segments}; continuing inference"
                                )

                chunk_end_time = time.time()
                logger.info(
                    f"Chunk {chunk_idx + 1} time: {chunk_end_time - chunk_start_time:.2f}s "
                    f"(denoise: {denoise_end - denoise_start:.2f}s, "
                    f"cache: {cache_end - cache_start:.2f}s, "
                    f"segment_vae: {segment_decode_elapsed:.2f}s)"
                )

            generation_end = time.time()

        logger.info(f"\n{'=' * 60}")
        logger.info(f"PROFILING SUMMARY")
        logger.info(f"{'=' * 60}")
        logger.info(f"Total generation: {generation_end - generation_start:.2f}s")
        logger.info(f"  Denoising: {total_denoise_time:.2f}s")
        logger.info(f"  Cache updates: {total_cache_time:.2f}s")
        if segment_vae_decode:
            logger.info(
                f"  Blocking segment VAE decode: {total_segment_vae_decode_time:.2f}s"
            )
            if segment_vae_output_rank:
                logger.info(
                    "  Total speculative repeat-last padding: "
                    f"{total_speculative_padding_latents} latent frames"
                )
                if first_decoded_frame_time is not None:
                    logger.info(
                        "  First decoded frame available after: "
                        f"{first_decoded_frame_time:.2f}s"
                    )
                else:
                    logger.info("  First decoded frame available after: n/a")

        if preserve_kv_cache:
            self._interactive_cached_action_tails = cached_action_tails
            self._interactive_cached_seg_video_ranges = cached_seg_video_ranges
            self._interactive_segment_count = session_segment_offset + num_segments
            logger.info(
                "Preserving AR cache for interactive continuation: "
                f"segments={self._interactive_segment_count}, "
                f"tokens={self.cache_valid_len}, "
                f"global_frames={self.global_frame_counter}"
            )
        else:
            from hymm.models.dit.modules.kv_cache_storage import cache_geometry, cache_storage_audit
            storage_audits = [cache_storage_audit(cache) for cache in self.kv_cache]
            self._last_kv_optimization_audit = {
                "int8_kv": self.int8_kv_storage, "swa": swa_enabled,
                "qvg_kv": getattr(self, "qvg_kv_storage", False),
                "kv_storage_format": storage_audits[0]["storage_format"],
                "gpu_kv_storage_bytes": sum(row["gpu_storage_bytes"] for row in storage_audits),
                "codec_stats": [row["codec_stats"] for row in storage_audits],
                "kv_storage_layers": storage_audits,
                "gpu_kv_peak_working_bytes_upper_bound": sum(row["peak_working_bytes"] for row in storage_audits),
                "kv_peak_note": (
                    "Sum of per-layer working peaks, not a simultaneous allocator peak; excludes codec temporaries"
                ),
                "qvg_options": storage_audits[0].get("qvg_options"),
                "qvg_version": storage_audits[0].get("qvg_version"),
                "history_sampling": swa_sampling if swa_enabled else None,
                "history_sampling_gamma": (
                    swa_gamma if swa_enabled and swa_sampling == "temporal_context" else None
                ),
                "history_archive": "cpu" if swa_archived else None,
                "history_text_policy": (
                    "selected_segments" if swa_archived else "within_token_budget" if swa_enabled else "all_segments"
                ),
                "history_video_frame_budget": past_frames if swa_archived else None,
                "history_text_capacity": swa_capacity - past_frames * tokens_per_frame if swa_archived else 0,
                "history_total_frames": self.kv_cache[0].get("swa_total_frames", 0) if swa_archived else None,
                "history_frame_ids": list(self.kv_cache[0].get("swa_frame_ids", ())) if swa_archived else [],
                "history_layout": list(self.kv_cache[0].get("swa_uniform_layout", ())) if swa_archived else [],
                "history_updates": len(self.kv_cache[0].get("swa_archive", ())) if swa_archived else 0,
                "cpu_archive_bytes": sum(cache.get("swa_archive_bytes", 0) for cache in self.kv_cache),
                "capacity": swa_capacity if swa_enabled else int(cache_geometry(self.kv_cache[0])[0][1]),
                "valid_tokens": self.cache_valid_len, "tokens_per_frame": tokens_per_frame,
                "block_count": len(self.kv_cache),
                "sparse": bool(getattr(self.transformer, "nomas_sparse_config", {}).get("enabled", False)),
                "sparse_calls": [cache.get("sparse_stats", {}).get("calls", 0) for cache in self.kv_cache],
                "router_calls": [cache.get("router_stats", {}).get("cute_hits", 0) for cache in self.kv_cache],
                "compile_calls": [getattr(block, "_compile_cache_calls", 0)
                                  for block in self.transformer.double_blocks],
                "storage_dtypes": sorted({dtype for row in storage_audits for dtype in row["storage_dtypes"]}),
                "layer_valid_lengths": [
                    cache.get("swa_valid_len", cache.get("qvg_valid_len", self.cache_valid_len))
                    for cache in self.kv_cache],
            }
            if any('dense_layer_stats' in cache for cache in self.kv_cache):
                self._last_kv_optimization_audit['dense_layer_stats'] = [
                    dict(cache.get('dense_layer_stats', {})) for cache in self.kv_cache
                ]
            self._last_kv_optimization_audit['sparse_optim'] = dict(
                config=self.transformer.sparse_optim,
                counts=dict(self.transformer.sparse_optim_counts))
            self._last_kv_optimization_audit['route_reuse'] = [
                dict(cache.get('route_reuse_stats', {})) for cache in self.kv_cache]
            logger.info("Freeing AR cache after generation...")
            self.kv_cache = None
            self.cache_mask_list = None
            self.segment_token_lengths.clear()
            self.cache_valid_len = 0
            self._interactive_cached_action_tails = []
            self._interactive_cached_seg_video_ranges = []
            self._interactive_segment_count = 0
            self._interactive_segment_cache_tokens = []
            self._interactive_segment_latent_lengths = []
            self._reset_interactive_noise_state()
            self._reset_interactive_pixelumm_decode_state()
        final_output_error = None
        if segment_vae_session is not None:
            try:
                if (
                    segment_vae_session.received_latent_frames != latent_frames
                    or (
                        segment_vae_output_rank
                        and emitted_latent_frames != latent_frames
                    )
                    or segment_vae_session.decoded_latent_frames != latent_frames
                    or segment_vae_session.pending_latent_frames != 0
                ):
                    raise RuntimeError(
                        "Segment VAE decoder did not account for the complete "
                        "latent timeline: "
                        f"received={segment_vae_session.received_latent_frames}, "
                        f"canonical={segment_vae_session.decoded_latent_frames}, "
                        f"pending={segment_vae_session.pending_latent_frames}, "
                        f"emitted={emitted_latent_frames}, "
                        f"expected={latent_frames}."
                    )
            except Exception as exc:  # noqa: BLE001
                final_output_error = exc
            finally:
                segment_vae_session.feature_cache = None
                segment_vae_session = None

        # VAE decode is the next peak; drop generation-only tensors first.
        # Release local KV/archive references as well as self.kv_cache.
        cache = first_cache = None
        try:
            del all_noise
        except UnboundLocalError:
            pass
        try:
            del encoded_prompts
        except UnboundLocalError:
            pass
        try:
            del camera_embedding_full
        except UnboundLocalError:
            pass
        try:
            del chunk_noise
        except UnboundLocalError:
            pass
        try:
            del latents
        except UnboundLocalError:
            pass
        try:
            del latent_input
        except UnboundLocalError:
            pass
        try:
            del cache_input
        except UnboundLocalError:
            pass
        try:
            del cache_latents
        except UnboundLocalError:
            pass
        try:
            del cache_snapshot_latents
        except UnboundLocalError:
            pass
        try:
            del cache_cond_latents
        except UnboundLocalError:
            pass
        try:
            del chunk_cam_emb
        except UnboundLocalError:
            pass
        try:
            del chunk_cam_mask
        except UnboundLocalError:
            pass
        try:
            del text_states_cfg
        except UnboundLocalError:
            pass
        try:
            del text_mask_cfg
        except UnboundLocalError:
            pass
        try:
            del noise_pred
        except UnboundLocalError:
            pass
        try:
            del output
        except UnboundLocalError:
            pass
        torch.cuda.empty_cache()

        decode_rank_within_sp = int(getattr(nccl_info, "rank_within_spgroup", 0) or 0)
        if save_latents_path and decode_rank_within_sp == 0:
            try:
                import numpy as np

                save_dir = os.path.dirname(save_latents_path)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                tmp_path = f"{save_latents_path}.tmp.npy"
                np.save(
                    tmp_path,
                    output_latents[0].detach().float().cpu().numpy(),
                )
                os.replace(tmp_path, save_latents_path)
                logger.info(
                    "[VAE decode] saved final latents for benchmark: "
                    f"{save_latents_path}"
                )
            except Exception as exc:  # noqa: BLE001
                if not segment_vae_decode:
                    raise
                final_output_error = exc

        if segment_vae_decode:
            if segment_vae_output_rank and final_output_error is None:
                try:
                    if decoded_video_cpu is None:
                        raise RuntimeError(
                            "Speculative segment VAE decode produced no output."
                        )
                    expected_pixelumm_frames = (
                        int(num_frames)
                        + pixelumm_request_pixel_offset
                        - pixelumm_output_pixel_start
                    )
                    if pixelumm_stable_streaming and decoded_pixel_frames != expected_pixelumm_frames:
                        raise RuntimeError(
                            "PixelUMM replace-tail output frame count mismatch: "
                            f"decoded={decoded_pixel_frames}, "
                            f"requested={num_frames}, "
                            f"carry={pixelumm_request_pixel_offset - pixelumm_output_pixel_start}."
                        )
                    if not pixelumm_stable_streaming and decoded_pixel_frames != num_frames:
                        raise RuntimeError(
                            "Speculative segment VAE output frame count "
                            "mismatch: "
                            f"decoded={decoded_pixel_frames}, "
                            f"requested={num_frames}."
                        )
                except Exception as exc:  # noqa: BLE001
                    final_output_error = exc

            self._sync_segment_vae_status(
                final_output_error,
                context="final output assembly",
                status_tensor=segment_vae_status,
            )
            if not segment_vae_output_rank:
                return torch.empty(0, dtype=self.target_dtype, device="cpu")

            logger.info(
                "[VAE decode] strategy="
                f"{'pixelumm_replace_provisional_tail' if pixelumm_stable_streaming else 'seg_ar_speculative_repeat_last'} "
                f"frames={decoded_video_cpu.shape[2]}, "
                f"device={decoded_video_cpu.device}"
            )
            return decoded_video_cpu

        logger.info("Decoding latents to video...")
        if output_latents is None:
            raise RuntimeError("Full-clip VAE decode requires retained latents.")
        return self._decode_latents(output_latents, num_frames)
