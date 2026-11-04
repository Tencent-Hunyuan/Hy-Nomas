"""
Segment self-forcing SFT trainer for a single Arrow or Parquet data file.

Usage:
    torchrun --nproc_per_node=8 hymm/train.py \
        +experiment=sft \
        dit_weight=/path/to/model.safetensors \
        dataset.data_file=/path/to/train.arrow
"""

import contextlib
import gc
import json
import math
import os
import random
import sys
import time
import datetime

_GC_DISABLE = os.environ.get("HY_GC_DISABLE", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)
if _GC_DISABLE:
    gc.disable()
else:
    gc.set_threshold(7000, 100, 100)

_ENV_TRUE_VALUES = ("1", "true", "yes", "on")
_TORCH_DETERMINISTIC = (
    os.environ.get("TORCH_DETERMINISTIC_MODE", "0").strip().lower() in _ENV_TRUE_VALUES
)
if _TORCH_DETERMINISTIC:
    # Must be set before CuBLAS workspaces are initialized.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.utils.tensorboard import SummaryWriter

from diffusers.optimization import get_scheduler
from accelerate.utils import set_seed

from hymm.utils.parallel import sync_object_for_parallel_training

from configs.config import parse_args
from hymm.models.checkpointing.load import load_transformer, load_text_encoder
from hymm.models.lora import (
    inject_lora,
    load_lora_adapter,
    save_lora_adapter,
)
from hymm.models.vae import load_vae
from hymm.distributed.parallel_states import (
    ParallelDims,
    initialize_sequence_parallel_state_v2,
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
)
from hymm.distributed.fsdp_util import (
    get_dit_fsdp_kwargs_v2,
    apply_fsdp_checkpointing,
    apply_camera_module_checkpointing,
    apply_fsdp2,
    apply_fsdp2_text_encoder,
    resolve_fsdp_param_dtype,
)
from hymm.utils.logging_ import setup_logger
from hymm.models.checkpointing import (
    save_checkpoint,
    load_checkpoint,
    load_rng_states,
    auto_find_latest_checkpoint,
    auto_find_latest_lora_checkpoint,
    load_lora_training_checkpoint,
    poll_pending_lora_checkpoint,
    restore_rng_state,
    save_lora_training_checkpoint,
    shutdown_lora_checkpoint_writer,
    wait_for_pending_lora_checkpoint,
)
from torch.profiler import profile, record_function, ProfilerActivity

from hymm.utils.helpers import ScalarStates, CycleStates
from hymm.utils.torch_utils import resolve_dit_training_precision
from hymm.models.dit.ema import (
    build_ema_states,
    update_ema,
    update_ema_with_rate,
    sync_ema_from_model,
    load_ema_states_from_checkpoint,
    EMA_SCHEDULED_RATE,
)

from hymm.trainer.transport.segment_self_forcing_transport import (
    create_segment_self_forcing_transport,
)
from hymm.dataset.single_file_sft import (
    create_single_file_sft_dataloader,
    RepeatingSFTIterator,
)
from hymm.utils.camera_utils import (
    freeze_base_dit_for_camera,
    freeze_camera_modules,
    print_camera_param_summary,
    ensure_camera_params_fp32,
    assert_camera_local_params_fp32,
)

from hymm.trainer.optimizer.dist_muon import (
    get_dist_muon_optimizer as get_muon_optimizer,
)

MUON_OPTIMIZER_IMPL = "hymm.trainer.optimizer.dist_muon"

try:
    import swanlab
except ImportError:
    swanlab = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _configure_deterministic_algorithms(logger=None):
    if not _TORCH_DETERMINISTIC:
        return None

    msg = (
        "Deterministic algorithm enabled via TORCH_DETERMINISTIC_MODE=1 "
        "(expect perf degradation)."
    )
    if logger is not None:
        logger.info(msg)

    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return msg


def _sync_cuda_time(sync=True, barrier=True):
    if barrier:
        dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])
    if sync:
        torch.cuda.synchronize()
    return time.time()


# ---------------------------------------------------------------------------
# FLOPs estimation (per-block matmul counting, matching train_cpt.py)
# ---------------------------------------------------------------------------

_PFLOPS_SCALE = 1_000_000_000_000_000  # 1e15, for cumulative PFLOPs display


def estimate_n_tokens(latent_shape, patch_size, reorg_token=False):
    """Number of visual tokens after patchification.

    Args:
        latent_shape: [B, C, T, H, W]
        patch_size: [pt, ph, pw]
        reorg_token: whether temporal channel reorg doubles C and halves T
    """
    _, _, T, H, W = latent_shape
    if reorg_token:
        T = (T + 1) // 2
    return (T // patch_size[0]) * (H // patch_size[1]) * (W // patch_size[2])


def _compute_causal_attn_flops(
    S_img, txt_per_block, prompt_video_boundaries_frames, D, sp
):
    """Per-block attention FLOPs for segment-level causal (mp_causal).

    Segment j attends to segments 0..j:
      Q_j = [video_seg_j, text_j]
      KV_j = [video_0..j, text_0..j]
    Returns total attention FLOPs per block per GPU (divided by sp).
    """
    boundaries = prompt_video_boundaries_frames
    total_frames = boundaries[-1]
    tokens_per_frame = S_img // total_frames if total_frames > 0 else 1
    num_prompts = len(boundaries) - 1

    total = 0
    for j in range(num_prompts):
        vid_start = boundaries[j] * tokens_per_frame
        vid_end = boundaries[j + 1] * tokens_per_frame
        q_len = (vid_end - vid_start) + txt_per_block
        kv_len = vid_end + (j + 1) * txt_per_block
        total += 4 * D * q_len * kv_len
    return total // sp


def _compute_chunk_causal_attn_flops(video_len, chunk_size_tokens, txt_tokens, D, sp):
    """Per-block attention FLOPs for GLOBAL chunk-wise causal attention.

    Mirrors ``create_magi_mp_causal_ranges`` (chunk_size_tokens > 0 branch) so
    the FLOPs match the actual magi attention mask used in chunk-level DF:
      - video chunk c (q = [c*cs, c*cs+cs]) attends K = [0, chunk_end]  (causal
        cumulative over video tokens -> lower-triangular -> ~V^2/2, NOT V^2)
      - each video chunk also attends the (single-prompt) text block
      - text attends video [0, V] and text->text

    This is the single-prompt approximation (num_prompts == 1), which covers the
    vast majority of samples. For multi-prompt the video->video term dominates
    and is still correctly chunk-causal, so the approximation stays accurate.

    Without this, the segment-level ``_compute_causal_attn_flops`` with a single
    prompt degenerates to ~full attention (q_len ~= kv_len ~= V), over-counting
    attention by ~2x and inflating MFU on long-sequence steps.
    """
    V = int(video_len)
    cs = int(chunk_size_tokens)
    T = int(txt_tokens)
    if cs <= 0 or V <= 0:
        S = V + T
        return 4 * D * S * S // sp

    num_chunks = (V + cs - 1) // cs
    last_chunk = V - (num_chunks - 1) * cs
    if num_chunks > 1 and last_chunk < cs // 2:
        num_chunks -= 1

    total = 0
    for c in range(num_chunks):
        chunk_start = c * cs
        chunk_end = min(chunk_start + cs, V)
        q_len = chunk_end - chunk_start
        total += q_len * chunk_end  # video chunk -> cumulative video [0, chunk_end]
        total += q_len * T  # video chunk -> text
    total += T * V  # text -> video [0, V]
    total += T * T  # text -> text
    return 4 * D * total // sp


def _block_linear_flops(D, mlp_hidden, S_img_local, S_txt):
    """Per-block linear-projection FLOPs, as ``(double_block, single_block)``.

    Img tokens are SP-split (pass ``S_img_local``); txt tokens are not.
    """
    db = (
        4 * 2 * D * D * S_img_local  # Q/K/V/O
        + 2 * 2 * D * mlp_hidden * S_img_local  # MLP up+down
        + 4 * 2 * D * D * S_txt
        + 2 * 2 * D * mlp_hidden * S_txt
    )
    S_single_local = S_img_local + S_txt
    sb = (
        3 * 2 * D * D * S_single_local  # Q/K/V
        + 2 * D * mlp_hidden * S_single_local  # MLP gate
        + 2 * (D + mlp_hidden) * D * S_single_local  # fused output
    )
    return db, sb


def _last_segment_fwd_flops(
    S_img,
    txt_per_block,
    prompt_video_boundaries_frames,
    D,
    mlp_hidden,
    n_double,
    n_single,
    sp,
):
    """Forward FLOPs contributed by the final segment alone.

    Mirrors the j == num_prompts-1 term of ``_compute_causal_attn_flops`` plus
    the linear projections over that segment's own tokens.  Causal layout makes
    this an exact decomposition: segments 0..N-2 cost the same whether or not
    segment N-1 exists.
    """
    boundaries = prompt_video_boundaries_frames
    num_prompts = len(boundaries) - 1
    if num_prompts < 1:
        return 0

    total_frames = boundaries[-1]
    tokens_per_frame = S_img // total_frames if total_frames > 0 else 1

    j = num_prompts - 1
    vid_start = boundaries[j] * tokens_per_frame
    vid_end = boundaries[j + 1] * tokens_per_frame
    q_len = (vid_end - vid_start) + txt_per_block
    kv_len = vid_end + (j + 1) * txt_per_block
    attn = 4 * D * q_len * kv_len // sp

    db_linear, sb_linear = _block_linear_flops(
        D,
        mlp_hidden,
        (vid_end - vid_start) // sp,
        txt_per_block,
    )
    return n_double * (db_linear + attn) + n_single * (sb_linear + attn)


class _LightweightModelConfig:
    """Lightweight stand-in for a real model, exposing only the attributes
    needed by ``estimate_step_flops``.  Used for offline FLOPs estimation
    (e.g. sorting packing groups) when the full model is not convenient to pass.
    """

    def __init__(
        self,
        hidden_size,
        num_double_blocks,
        num_single_blocks,
        mlp_width_ratio=4.0,
        patch_size=None,
    ):
        self.hidden_size = hidden_size
        self.double_blocks = [None] * num_double_blocks
        self.single_blocks = [None] * num_single_blocks
        self.config = type("C", (), {"mlp_width_ratio": mlp_width_ratio})()
        self.patch_size = patch_size or [1, 2, 2]


def estimate_step_flops(
    model,
    n_img_tokens: int,
    n_txt_tokens: int,
    batch_size: int,
    sp_size: int = 1,
    gc_enabled: bool = False,
    gc_ratio: float = 1.0,
    skip_attn_recompute: bool = False,
    attn_mode: str = "flash",
    prompt_video_boundaries_frames=None,
    txt_tokens_per_block: int = 256,
    chunk_size_tokens: int = 0,
    extra_forward_passes: float = 0.0,
    skip_last_segment_cache_write: bool = False,
    segment_dropout_p: float = 0.0,
):
    """Estimate FLOPs **per GPU** for one training step (fwd + bwd + GC recompute).

    Counts every major matmul in each transformer block:
      - Linear projections: Q/K/V/O and MLP
      - Attention kernels: QK^T and AV

    Sequence Parallelism (Ulysses SP):
      - Img linear projections run on S_img/sp tokens per GPU.
      - Txt linear projections run on full S_txt (not split).
      - Attention: each GPU has H/sp heads on full sequence -> FLOPs / sp.

    ``model`` can be a real nn.Module or a ``_LightweightModelConfig``.
    """
    D = model.hidden_size
    if hasattr(model, "config") and hasattr(model.config, "mlp_width_ratio"):
        mlp_ratio = model.config.mlp_width_ratio
    elif hasattr(model, "config") and callable(getattr(model.config, "get", None)):
        mlp_ratio = model.config.get("mlp_width_ratio", 4.0)
    else:
        mlp_ratio = 4.0
    n_double = len(model.double_blocks) if hasattr(model, "double_blocks") else 0
    n_single = len(model.single_blocks) if hasattr(model, "single_blocks") else 0
    mlp_hidden = int(D * mlp_ratio)

    sp = max(sp_size, 1)
    S_img = n_img_tokens
    S_txt = n_txt_tokens
    S_img_local = S_img // sp
    S_total = S_img + S_txt

    # ---- Linear projection FLOPs per double/single block per GPU ----
    db_linear, sb_linear = _block_linear_flops(D, mlp_hidden, S_img_local, S_txt)

    # ---- Attention FLOPs per double block per GPU ----
    is_causal = "causal" in (attn_mode or "")
    if is_causal and chunk_size_tokens and chunk_size_tokens > 0:
        # Chunk-level DF: global chunk-wise causal (lower-triangular over video
        # chunks -> ~S_img^2/2). MUST match magi_attn.create_magi_mp_causal_ranges.
        db_attn = _compute_chunk_causal_attn_flops(
            S_img,
            chunk_size_tokens,
            S_txt,
            D,
            sp,
        )
    elif is_causal and prompt_video_boundaries_frames is not None:
        db_attn = _compute_causal_attn_flops(
            S_img,
            txt_tokens_per_block,
            prompt_video_boundaries_frames,
            D,
            sp,
        )
    else:
        db_attn = 4 * D * S_total * S_total // sp

    db_fwd = db_linear + db_attn

    # ---- Single block FLOPs per GPU ----
    sb_attn = db_attn
    sb_fwd = sb_linear + sb_attn

    # ---- Total ----
    fwd_flops = n_double * db_fwd + n_single * sb_fwd
    bwd_flops = 2 * fwd_flops

    recompute_flops = 0
    if gc_enabled and gc_ratio > 0:
        if skip_attn_recompute:
            recompute_per_db = db_linear
            recompute_per_sb = sb_linear
        else:
            recompute_per_db = db_fwd
            recompute_per_sb = sb_fwd
        recompute_flops = gc_ratio * (
            n_double * recompute_per_db + n_single * recompute_per_sb
        )

    # ---- Camera ActionModule FLOPs (extrinsics_attn / action_attn) ----
    # Injected after every double block (see nomas_cpt.forward). Only the real
    # model exposes ``action_modules``; the lightweight offline config does not,
    # so this is skipped there. Without this term the camera modules add real
    # wall-clock time but no counted FLOPs, mechanically depressing MFU.
    camera_flops = 0
    cam_modules = getattr(model, "action_modules", None)
    cam_mode = getattr(model, "camera_injection_mode", None)
    if (
        cam_modules is not None
        and len(cam_modules) > 0
        and cam_mode
        in (
            "action_attn",
            "extrinsics_attn",
        )
    ):
        am0 = cam_modules[0]
        # img_attn_q: Linear(img_hidden_size -> keyboard_hidden_dim)
        # proj_keyboard: Linear(keyboard_hidden_dim -> img_hidden_size)
        D_cam = am0.img_attn_q.in_features
        Kd = am0.img_attn_q.out_features
        n_cam = len(cam_modules)

        # Q comes from image tokens (SP-split); the two big linears dominate.
        cam_lin_per_layer = (
            2 * 2 * D_cam * Kd * S_img_local
        )  # img_attn_q + proj_keyboard

        # Cross-attn (QK^T + AV) with K/V = keyboard tokens (~= latent frames).
        # N_kb is derivable from the boundaries (total latent frames). Small
        # relative to the linears; included only when available.
        cam_attn_per_layer = 0
        if (
            prompt_video_boundaries_frames is not None
            and len(prompt_video_boundaries_frames) >= 2
        ):
            n_kb = max(int(prompt_video_boundaries_frames[-1]), 1)
            cam_attn_per_layer = 4 * S_img_local * n_kb * Kd

        cam_fwd = n_cam * (cam_lin_per_layer + cam_attn_per_layer)
        # Camera modules are typically NOT activation-checkpointed
        # (checkpoint_camera_modules defaults off in the camera configs), so no
        # recompute term is added here.
        camera_flops = cam_fwd + 2 * cam_fwd  # fwd + bwd

    # Some training modes run additional no-grad forwards over the same causal
    # layout. Segment self-forcing, for example, performs one gradient-bearing
    # flow-matching forward plus one low-noise KV-cache write forward for every
    # segment. The latter has forward FLOPs but no backward/GC recompute.
    extra_fwd_flops = max(float(extra_forward_passes), 0.0) * fwd_flops
    segment_dropout_p = min(max(float(segment_dropout_p), 0.0), 1.0)

    # Forward FLOPs of the final segment alone, shared by the two segment-level
    # corrections below.
    last_seg_fwd = 0
    if (skip_last_segment_cache_write and extra_fwd_flops > 0) or segment_dropout_p > 0:
        if (
            prompt_video_boundaries_frames is not None
            and len(prompt_video_boundaries_frames) >= 2
        ):
            last_seg_fwd = _last_segment_fwd_flops(
                S_img,
                txt_tokens_per_block,
                prompt_video_boundaries_frames,
                D,
                mlp_hidden,
                n_double,
                n_single,
                sp,
            )
        else:
            # No boundaries -> single segment, which is also the final one.
            last_seg_fwd = fwd_flops

    if skip_last_segment_cache_write and extra_fwd_flops > 0:
        # The final segment's cache entry is never read before the per-sample
        # cache is freed, so that write is skipped at runtime. Subtracting it
        # here matters for more than absolute accuracy: the omitted share is
        # ~1/N of the sample, so leaving it in biases the packing planner
        # against multi-segment samples and unbalances the DP ranks.
        extra_fwd_flops = max(extra_fwd_flops - last_seg_fwd, 0)

    # ---- Per-segment text dropout (segment_self_forcing.segment_dropout) ----
    # A dropped segment runs an extra conditional pass whose velocity only feeds
    # the (detached) cache write, so it is a no-grad forward like the cache write
    # itself: no backward, no GC recompute. Which segment gets dropped is drawn
    # per (global_step, packed slot) at runtime and cannot be known while
    # planning, so this is the expectation: p x the forward FLOPs of the segments
    # that can double. Segments whose cache write is skipped never pay it,
    # because their conditional velocity would be discarded and the transport
    # runs the unconditional pass alone. Causal layout makes the per-segment
    # forwards sum to fwd_flops, so the doubled share is exact up to the negative
    # pass's shorter (empty) prompt.
    dropout_flops = 0
    if segment_dropout_p > 0 and fwd_flops > 0:
        doubled_fwd = (
            fwd_flops - last_seg_fwd if skip_last_segment_cache_write else fwd_flops
        )
        dropout_flops = segment_dropout_p * max(doubled_fwd, 0)

    return (
        fwd_flops
        + bwd_flops
        + recompute_flops
        + extra_fwd_flops
        + dropout_flops
        + camera_flops
    ) * batch_size


def get_gpu_peak_tflops(dtype=torch.bfloat16):
    """Theoretical peak TFLOPS (bf16 Tensor Core)."""
    return 148


class DefaultTrainer:
    """Segment self-forcing SFT trainer.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI / YAML config.
    """

    def __init__(self, args):
        self.args = args
        self.training_mode = getattr(args, "training_mode", None)
        if self.training_mode != "segment_self_forcing":
            raise ValueError(
                "Open-source training supports only training_mode=segment_self_forcing"
            )
        self.finetune_mode = getattr(args, "finetune_mode", "full")
        if self.finetune_mode not in ("full", "lora"):
            raise ValueError("training.finetune_mode must be 'full' or 'lora'")
        if (
            getattr(args.camera, "use_camera_control", False)
            and getattr(args.camera, "injection_mode", None) != "extrinsics_attn"
        ):
            raise ValueError(
                "Public camera SFT supports model.camera.injection_mode=extrinsics_attn"
            )
        data_file = getattr(args.dataset, "data_file", None)
        if (
            not data_file
            or not str(data_file).endswith((".arrow", ".parquet"))
            or not os.path.isfile(data_file)
        ):
            raise ValueError(
                "dataset.data_file must be one existing .arrow or .parquet file"
            )
        if not getattr(args, "dit_weight", None):
            raise ValueError(
                "dit_weight must point to a compatible base model checkpoint"
            )
        self.precision_policy = resolve_dit_training_precision(
            args.dit.master_weight_type
        )

        torch.backends.cuda.matmul.allow_tf32 = True
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self._ckpt_dp_replicate_rank = 0
        self._ckpt_dp_shard_sp_group = None
        self._enable_dcp_checkpoint = bool(
            getattr(args.optimizer, "enable_dcp_checkpoint", False)
        )
        self._pending_rng_resume_state = None
        self._pre_logger_messages = []
        _det_msg = _configure_deterministic_algorithms()
        if _det_msg:
            self._pre_logger_messages.append(("info", _det_msg))

        _t_total_start = time.time()

        _t0 = time.time()
        self._init_distributed()
        self._init_logging()
        self.logger.info(
            "DiT precision policy: "
            f"master={self.precision_policy.master_dtype}, "
            f"compute={self.precision_policy.compute_dtype}, "
            f"reduce={self.precision_policy.reduce_dtype}, "
            f"cache={self.precision_policy.cache_dtype}"
        )
        self._init_seed()
        _t1 = time.time()
        self.logger.info(f"[INIT_TIMER] distributed + logging + seed: {_t1 - _t0:.1f}s")

        _t0 = time.time()
        self._load_models()
        _t1 = time.time()
        self.logger.info(
            f"[INIT_TIMER] load_models (VAE + text_enc + DiT): {_t1 - _t0:.1f}s"
        )

        _t0 = time.time()
        self._create_transport()
        _t1 = time.time()
        self.logger.info(f"[INIT_TIMER] create_transport: {_t1 - _t0:.1f}s")

        _t0 = time.time()
        self._apply_fsdp()
        _t1 = time.time()
        self.logger.info(f"[INIT_TIMER] apply_fsdp: {_t1 - _t0:.1f}s")

        _t0 = time.time()
        self._init_ema()
        _t1 = time.time()
        self.logger.info(f"[INIT_TIMER] init_ema: {_t1 - _t0:.1f}s")

        _t0 = time.time()
        self._create_optimizer()
        _t1 = time.time()
        self.logger.info(f"[INIT_TIMER] create_optimizer: {_t1 - _t0:.1f}s")

        _t0 = time.time()
        self._maybe_resume()
        _t1 = time.time()
        self.logger.info(f"[INIT_TIMER] maybe_resume (checkpoint): {_t1 - _t0:.1f}s")

        gc_enabled = args.parallelism.gradient_checkpointing
        gc_ratio = (
            float(eval(str(getattr(args.parallelism, "selective_checkpointing", 1.0))))
            if gc_enabled
            else 0.0
        )
        self._gc_enabled = gc_enabled
        self._gc_ratio = gc_ratio
        self._skip_attn_recompute = getattr(
            args.parallelism, "skip_attn_recompute", False
        )
        self._gpu_peak_tflops = get_gpu_peak_tflops()
        self._cumulative_flops = 0.0

        _t0 = time.time()
        self._create_dataloader()
        _t1 = time.time()
        self.logger.info(f"[INIT_TIMER] create_dataloader: {_t1 - _t0:.1f}s")

        _t0 = time.time()
        self._init_trackers()
        _t1 = time.time()
        self.logger.info(f"[INIT_TIMER] init_trackers (tb/swanlab): {_t1 - _t0:.1f}s")

        self.logger.info(
            f"[INIT_TIMER] === TOTAL init: {time.time() - _t_total_start:.1f}s ==="
        )

    # ------------------------------------------------------------------
    # Distributed
    # ------------------------------------------------------------------

    def _init_distributed(self):
        args = self.args
        world_size = self.world_size
        sp_size = int(args.parallelism.sp_size)
        if getattr(args.parallelism, "tp_size", 1) != 1:
            raise ValueError("parallelism.tp_size > 1 is currently supported by inference only")
        sharding_strategy = args.parallelism.fsdp_sharding_strategy

        if sharding_strategy == "full":
            dp_replicate = 1
            dp_shard = -1
        else:
            if sharding_strategy != "hybrid":
                raise ValueError(
                    f"Unsupported FSDP sharding strategy: {sharding_strategy}"
                )
            if world_size % sp_size != 0:
                raise ValueError(
                    f"Invalid parallel dims: WORLD_SIZE({world_size}) must be divisible "
                    f"by sp_size({sp_size})"
                )

            data_parallel_size = world_size // sp_size
            target_shard_size = getattr(
                args.parallelism, "fsdp_hybrid_shard_size", None
            )
            if target_shard_size is None:
                dp_replicate = world_size // 8
                if dp_replicate < 1 or world_size % (dp_replicate * sp_size) != 0:
                    raise ValueError(
                        f"Invalid hybrid parallel dims: WORLD_SIZE({world_size}), "
                        f"sp_size({sp_size}), legacy node shard size(8) cannot form "
                        "a valid dp_replicate * dp_shard * sp mesh"
                    )
                dp_shard = world_size // (dp_replicate * sp_size)
                if dp_shard < 1:
                    raise ValueError(
                        f"Invalid hybrid parallel dims: WORLD_SIZE({world_size}), "
                        f"sp_size({sp_size}) inferred dp_shard({dp_shard})"
                    )
            else:
                target_shard_size = int(target_shard_size)
                if target_shard_size < 1:
                    raise ValueError(
                        f"parallelism.fsdp_hybrid_shard_size must be >= 1, got {target_shard_size}"
                    )
                dp_shard = min(target_shard_size, data_parallel_size)
                while data_parallel_size % dp_shard != 0:
                    dp_shard -= 1
                dp_replicate = data_parallel_size // dp_shard

        if (
            world_size == 8
            and sharding_strategy == "hybrid"
            and getattr(args.parallelism, "fsdp_hybrid_shard_size", None) is None
        ):
            args.parallelism.fsdp_sharding_strategy = "full"
            dp_replicate = 1
            dp_shard = -1
            print("node=1 fall back hybrid to full-shard")

        self.parallel_dims = ParallelDims(
            dp_replicate=dp_replicate,
            dp_shard=dp_shard,
            sp=sp_size,
            tp=1,
            pp=1,
            world_size=world_size,
        )
        msg = (
            "Resolved parallel mesh: "
            f"dp_replicate={self.parallel_dims.dp_replicate}, "
            f"dp_shard={self.parallel_dims.dp_shard}, sp={self.parallel_dims.sp}, "
            f"tp={self.parallel_dims.tp}, pp={self.parallel_dims.pp}, "
            f"world_size={world_size}, "
            f"fsdp_sharding_strategy={args.parallelism.fsdp_sharding_strategy}"
        )
        if hasattr(self, "logger"):
            self.logger.info(msg)
        else:
            print(msg)

        dist.init_process_group(
            backend="nccl", timeout=datetime.timedelta(seconds=3600)
        )
        torch.cuda.set_device(self.local_rank)
        self.device = torch.cuda.current_device()
        self.world_mesh = self.parallel_dims.build_mesh(device_type="cuda")

        if self.parallel_dims.dp_enabled:
            dp_mesh = self.world_mesh["dp"]
            self.dp_degree = dp_mesh.size()
            self.dp_rank = dp_mesh.get_local_rank()
        else:
            self.dp_degree = 1
            self.dp_rank = 0

        initialize_sequence_parallel_state_v2(
            self.parallel_dims,
            self.world_mesh,
            args.parallelism.use_dynamic_ring_attention,
        )

        # Production checkpoint layout: model/optimizer shards are identical
        # across the HSDP replicate dimension, so only replica 0 writes one
        # shard per dp_shard_sp rank.
        if self.parallel_dims.dp_replicate_enabled:
            self._ckpt_dp_replicate_rank = self.world_mesh[
                "dp_replicate"
            ].get_local_rank()
        if self.parallel_dims.dp_shard_enabled or self.parallel_dims.sp_enabled:
            self._ckpt_dp_shard_sp_group = self.world_mesh["dp_shard_sp"].get_group()

        if getattr(args, "attn_mode", None) == "yunchang":
            from hymm.utils.yunchang_attn import initialize_yunchang_sp

            initialize_yunchang_sp(
                sp_ulysses_degree=getattr(
                    args, "yunchang_ulysses_degree", args.parallelism.sp_size
                ),
                sp_ring_degree=getattr(args, "yunchang_ring_degree", 1),
                rank=self.rank,
                world_size=world_size,
            )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _init_logging(self):
        args = self.args

        if self.rank <= 0 and args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

        if args.output_dir is not None:
            obj_list = [args.output_dir]
            dist.broadcast_object_list(obj_list, src=0)
            args.output_dir = obj_list[0]

        self.logger, _ = setup_logger(args.output_dir)

        # Save the fully resolved config to output_dir for reproducibility
        if self.rank <= 0 and args.output_dir is not None:
            try:
                from omegaconf import OmegaConf

                config_save_path = os.path.join(args.output_dir, "resolved_config.yaml")
                with open(config_save_path, "w") as f:
                    f.write(OmegaConf.to_yaml(args, resolve=True))
                self.logger.info(f"Saved resolved config to {config_save_path}")
            except Exception as e:
                self.logger.warning(f"Failed to save resolved config: {e}")

        self.logger.info("=" * 80)
        self.logger.info(f"Training mode: {self.training_mode}")
        self.logger.info("=" * 80)
        self.logger.info(f"--> args {args}")
        self.logger.info(f"dp_degree: {self.dp_degree}, dp_rank: {self.dp_rank}")
        for level, msg in self._pre_logger_messages:
            log_fn = getattr(self.logger, level, self.logger.info)
            log_fn(msg)
        self._pre_logger_messages.clear()

    def _init_seed(self):
        args = self.args
        local_seed = (
            getattr(args, "global_seed", 42) + self.rank // args.parallelism.sp_size
        )
        set_seed(local_seed)
        random.seed(local_seed)
        print(f"dp_rank: {self.dp_rank}, rank: {self.rank}, local_seed: {local_seed}")

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------

    def _load_models(self):
        args, device, logger = self.args, self.device, self.logger

        _t0 = time.time()
        logger.info("Loading VAE...")
        vae, _, s_ratio, t_ratio = load_vae(
            args.vae_type, args.vae.vae_precision, vae_path=args.vae.checkpoint,
            logger=logger, device="cpu"
        )
        vae.requires_grad_(False)
        vae.eval()
        args.vae_spatial_ratio = s_ratio
        args.vae_temporal_ratio = t_ratio
        args.vae_latent_channels = vae.config.latent_channels
        logger.info(
            f"VAE config: spatial_ratio={s_ratio}, temporal_ratio={t_ratio}, latent_channels={args.vae_latent_channels}"
        )
        logger.info(f"[INIT_TIMER]   VAE loaded in {time.time() - _t0:.1f}s")

        _t0 = time.time()
        logger.info(f"Loading text encoder from {args.text_encoder.model_name}")
        text_encoder = load_text_encoder(
            device, logger, args
        )
        logger.info(f"[INIT_TIMER]   Text encoder loaded in {time.time() - _t0:.1f}s")

        # Shard the frozen, inference-only text encoder with FSDP2 over the
        # Ulysses SP mesh (shard size == sp_size). Within an SP group every rank
        # encodes the same prompt, so the full-weight copies are pure redundancy
        # (~13.2GB/GPU for Qwen2.5-VL-7B). Sharding frees ~(1 - 1/sp) of that for
        # the DiT. Done before the DiT loads so the freed memory also lowers the
        # initialization peak. No gradients/optimizer state are involved.
        if getattr(args.parallelism, "shard_text_encoder", False):
            if self.parallel_dims.sp_enabled:
                _t0 = time.time()
                te_param_dtype = text_encoder.model.dtype
                apply_fsdp2_text_encoder(
                    text_encoder,
                    sp_mesh=self.world_mesh["sp"],
                    param_dtype=te_param_dtype,
                    logger=logger,
                )
                torch.cuda.empty_cache()
                logger.info(
                    f"[INIT_TIMER]   Text encoder FSDP sharded in {time.time() - _t0:.1f}s"
                )
            else:
                logger.warning(
                    "parallelism.shard_text_encoder=true but sp is disabled (sp_size=1); "
                    "skipping text-encoder sharding."
                )

        factor_kwargs = {
            "device": device,
            "dtype": self.precision_policy.master_dtype,
        }

        _t0 = time.time()
        logger.info(
            f"Loading model from {args.dit_weight}..."
        )
        model = load_transformer(
            args.dit.model_type
            if hasattr(args, "dit") and hasattr(args.dit, "model_type")
            else args.dit_type,
            args.dit_weight,
            factor_kwargs,
            getattr(args, "convert_state_dict_in_out", False),
            getattr(args, "convert_state_dict_16168_to_16164", False),
            args,
            logger,
        )
        model.train()

        # Ensure uniform dtype across all parameters.  Checkpoint weights may
        # be stored in a different precision (e.g. fp32) than the model's
        # initialization dtype, causing mixed-dtype errors in FSDP.
        target_dtype = factor_kwargs["dtype"]
        model.to(dtype=target_dtype)
        logger.info(
            f"[INIT_TIMER]   DiT loaded + dtype cast in {time.time() - _t0:.1f}s"
        )
        logger.info(f"Unified model dtype to {target_dtype}")

        # Camera master parameters must remain fp32 for every training role.
        # They stay in the parent DiT's root FSDP2 unit (no separate FSDP unit),
        # so the blanket ``model.to(dtype=target_dtype)`` above would otherwise
        # demote them whenever master_weight_type is bf16.
        if getattr(getattr(args, "camera", None), "use_camera_control", False):
            ensure_camera_params_fp32(
                model,
                role="student/generator",
                logger=logger,
                force=True,
            )

        self.vae = vae
        self.text_encoder = text_encoder
        self.model = model

        # Freeze entire DiT (no trainable params in DiT)
        if getattr(args.dit, "freeze", False):
            for param in model.parameters():
                param.requires_grad = False
            logger.info("model.dit.freeze=True: froze ALL DiT parameters")

        # Camera control: freeze base DiT, only camera injection modules trainable
        freeze_base_for_camera = bool(
            getattr(args.camera, "freeze_dit_for_camera", False)
        )
        freeze_student_camera = bool(
            getattr(args.camera, "freeze_camera_modules", False)
        )
        if freeze_base_for_camera and freeze_student_camera:
            raise ValueError(
                "model.camera.freeze_dit_for_camera and "
                "model.camera.freeze_camera_modules cannot both be true: "
                "the combination would freeze the entire student DiT."
            )

        if freeze_base_for_camera:
            freeze_base_dit_for_camera(model, logger)
        # Universal student-camera freeze (all training/distillation modes).
        # Distillation inherits stage6's camera modules and keeps them fixed so
        # only the video branch is distilled.
        elif freeze_student_camera:
            freeze_camera_modules(model, logger)

        if self.finetune_mode == "lora":
            adapter_path = getattr(args.lora, "adapter_path", None)
            if adapter_path:
                self.lora_config = load_lora_adapter(model, adapter_path)
            else:
                replaced = inject_lora(
                    model,
                    rank=int(args.lora.rank),
                    alpha=float(args.lora.alpha),
                    dropout=float(args.lora.dropout),
                )
                self.lora_config = {
                    "rank": int(args.lora.rank),
                    "alpha": float(args.lora.alpha),
                    "dropout": float(args.lora.dropout),
                }
                logger.info(f"LoRA adapted linear layers: {len(replaced)}")
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"LoRA trainable parameters: {trainable:,}")

    # ------------------------------------------------------------------
    # Transport / denoiser
    # ------------------------------------------------------------------

    def _create_transport(self):
        self.transport = create_segment_self_forcing_transport(
            self.args,
            logger=self.logger,
        )

    # ------------------------------------------------------------------
    # FSDP
    # ------------------------------------------------------------------

    def _apply_fsdp(self):
        args, logger, model = self.args, self.logger, self.model

        logger.info(
            f"--> Initializing FSDP with sharding strategy: {args.parallelism.fsdp_sharding_strategy}"
        )
        fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs_v2(
            model,
            self.parallel_dims,
            self.world_mesh,
            args.parallelism.fsdp_sharding_strategy,
            use_lora=self.finetune_mode == "lora",
            cpu_offload=getattr(args, "use_cpu_offload", False),
            master_weight_type=args.dit.master_weight_type,
        )

        if args.parallelism.gradient_checkpointing:
            # ---- Activation offload: build the manager BEFORE checkpointing so
            # apply_fsdp_checkpointing can install the custom wrapper that offloads
            # each double block's img/txt INPUTS to pinned CPU (Phase 2). The
            # attention-out offload (Phase 1) uses the same manager via the SAC
            # context_fn. ----
            offload_on = getattr(args.parallelism, "offload_activations", False)
            if offload_on:
                from hymm.distributed.activation_offload import (
                    ActivationOffloadManager,
                    set_offload_manager,
                    register_prefetch_hooks,
                )

                if not getattr(args.parallelism, "skip_attn_recompute", False):
                    logger.info(
                        "--> Activation offload with skip_attn_recompute=false: "
                        "offloading checkpoint block img/txt inputs only; "
                        "attention forward will be recomputed in backward."
                    )
                dev = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
                offload_async = bool(
                    getattr(args.parallelism, "offload_activations_async", False)
                )
                self.offload_manager = ActivationOffloadManager(
                    enabled=True,
                    device=dev,
                    async_enabled=offload_async,
                )
                set_offload_manager(self.offload_manager)
                # Let the shared backward helper close the manager's per-step
                # leases after every successful backward/optimizer boundary.
                model._activation_offload_manager = self.offload_manager

            apply_fsdp_checkpointing(
                model,
                no_split_modules,
                args.parallelism.selective_checkpointing,
                skip_attn_recompute=getattr(
                    args.parallelism, "skip_attn_recompute", False
                ),
            )

            if offload_on:
                # reset per-step slot counters at the start of every model forward
                model.register_forward_pre_hook(
                    lambda m, a: (
                        self.offload_manager.begin_forward(
                            grad_enabled=torch.is_grad_enabled()
                        ),
                        None,
                    )[1]
                )
                # backward pre-hooks prefetch each layer's attn `out` one layer
                # ahead (no-op in sync mode; kept for the async path).
                blocks = getattr(model, "double_blocks", None)
                if blocks is not None and len(blocks) > 0:
                    register_prefetch_hooks(blocks, name="attn_out")
                    logger.info(
                        f"--> Activation offload ON (attn out + img/txt -> CPU), "
                        f"{len(blocks)} double blocks, "
                        f"async_d2h={self.offload_manager._async_d2h}, "
                        f"async_h2d={self.offload_manager._async_h2d}, "
                        f"resid_min_bytes={self.offload_manager._resid_min_bytes}"
                    )
                else:
                    logger.warning(
                        "--> Activation offload ON but model.double_blocks not found; "
                        "using synchronous H2D fallback (no backward overlap)"
                    )
            if getattr(args.camera, "use_camera_control", False) and getattr(
                args.parallelism, "checkpoint_camera_modules", True
            ):
                apply_camera_module_checkpointing(
                    model,
                    p=getattr(args.parallelism, "camera_selective_checkpointing", 1.0),
                )

        apply_fsdp2(
            model=model,
            dp_mesh=fsdp_kwargs["device_mesh"],
            # ``parallelism.fsdp_param_dtype`` (default null) opts out of the
            # precision policy's fixed bf16 forward dtype. Distillation recipes
            # need fp32 all-gather so the student matches the teacher's numerics
            # exactly; leaving the key null keeps master's policy verbatim.
            param_dtype=(
                resolve_fsdp_param_dtype(args)
                if getattr(args.parallelism, "fsdp_param_dtype", None) is not None
                else self.precision_policy.compute_dtype
            ),
            reduce_dtype=self.precision_policy.reduce_dtype,
            pp_enabled=self.parallel_dims.pp_enabled,
            cpu_offload=fsdp_kwargs["cpu_offload"],
            reshard_after_forward_policy="default",
            overlap=getattr(args.parallelism, "fsdp_overlap", True),
        )
        if getattr(getattr(args, "camera", None), "use_camera_control", False):
            assert_camera_local_params_fp32(
                model,
                role="student/generator",
                logger=logger,
            )
        logger.info("--> Model FSDP initialized")

        if (
            args.parallelism.fsdp_sharding_strategy == "hybrid"
            and self.parallel_dims.dp_replicate_enabled
            and (self.parallel_dims.dp_shard_enabled or self.parallel_dims.sp_enabled)
        ):
            replicate_group = self.world_mesh["dp_replicate"].get_group()
            for name, param in model.named_parameters():
                group_rank0_rank = dist.get_process_group_ranks(replicate_group)[0]
                if isinstance(param, DTensor):
                    dist.broadcast(
                        param._local_tensor, src=group_rank0_rank, group=replicate_group
                    )
                else:
                    dist.broadcast(param, src=group_rank0_rank, group=replicate_group)

    # ------------------------------------------------------------------
    # Self-Forcing (lazy init after FSDP)
    # ------------------------------------------------------------------

    def _init_ema(self):
        ema_rates = getattr(self.args, "ema_rates", None) or []
        ema_schedule = getattr(self.args, "ema_schedule", None) or []

        # ema_schedule overrides ema_rates when set.
        # Format: [[step, rate], ...] sorted by step ascending.
        # Maintains a single EMA shadow whose decay rate changes over time.
        if ema_schedule:
            # Parse and sort schedule by step
            self.ema_schedule = sorted(
                [[int(s), float(r)] for s, r in ema_schedule], key=lambda x: x[0]
            )
            # Use a fixed sentinel key so the checkpoint folder is always "ema_scheduled/"
            # regardless of the schedule configuration (safe for resume after schedule changes).
            self._ema_canonical_rate = EMA_SCHEDULED_RATE
            self.ema_rates = [EMA_SCHEDULED_RATE]
            self.ema_states = (
                build_ema_states(self.model, self.ema_rates) if self.ema_rates else {}
            )
            self.logger.info(f"--> EMA schedule initialized: {self.ema_schedule}")
        else:
            self.ema_schedule = []
            self._ema_canonical_rate = None
            self.ema_rates = list(ema_rates)
            self.ema_states = (
                build_ema_states(self.model, self.ema_rates) if self.ema_rates else {}
            )
            if self.ema_rates:
                self.logger.info(f"--> EMA initialized with rates: {self.ema_rates}")

    def _get_ema_rate_for_step(self, step: int) -> float:
        """Get the EMA decay rate for the given update step based on ema_schedule.

        Returns the rate from the last schedule entry whose step <= current step.
        """
        if not self.ema_schedule:
            return self.ema_rates[0] if self.ema_rates else 0.9999
        rate = self.ema_schedule[0][1]
        for s, r in self.ema_schedule:
            if step >= s:
                rate = r
            else:
                break
        return rate

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------

    def _create_optimizer(self):
        args, logger, model = self.args, self.logger, self.model

        optimizer_type = getattr(args, "optimizer_type", "muon").lower()
        if optimizer_type not in ("muon", "adamw"):
            raise ValueError("training.optimizer.name must be 'muon' or 'adamw'")
        if optimizer_type == "muon":
            logger.info(f"--> Using Muon optimizer ({MUON_OPTIMIZER_IMPL})")
            self.optimizer = get_muon_optimizer(
                model,
                lr=args.optimizer.learning_rate,
                weight_decay=args.optimizer.weight_decay,
                momentum=getattr(args.optimizer, "momentum", 0.95),
                adamw_betas=tuple(getattr(args.optimizer, "adamw_betas", (0.9, 0.95))),
                adamw_eps=getattr(args.optimizer, "adamw_eps", 1e-8),
                adamw_weight_decay=args.optimizer.adamw_weight_decay,
                muon_exclude_patterns=getattr(
                    args.optimizer, "muon_exclude_patterns", []
                ),
                muon_comm_mode=getattr(args.optimizer, "muon_comm_mode", "legacy"),
                muon_canzona_fused_comm=getattr(
                    args.optimizer, "muon_canzona_fused_comm", True
                ),
                muon_canzona_balance=getattr(
                    args.optimizer, "muon_canzona_balance", "global"
                ),
                muon_canzona_balance_cost=getattr(
                    args.optimizer, "muon_canzona_balance_cost", "numel"
                ),
                muon_canzona_overlap=getattr(
                    args.optimizer, "muon_canzona_overlap", "none"
                ),
            )
            if self.finetune_mode == "lora":
                muon_params = sum(
                    id(param) in self.optimizer._muon_param_ids
                    for group in self.optimizer.param_groups
                    for param in group["params"]
                )
                if muon_params == 0:
                    raise ValueError("LoRA Muon has no matrix parameters to optimize")
                logger.info(f"LoRA Muon matrix parameters: {muon_params}")
        else:
            logger.info("--> Using AdamW optimizer")
            self.optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.optimizer.learning_rate,
                betas=tuple(getattr(args.optimizer, "adamw_betas", (0.9, 0.95))),
                weight_decay=args.optimizer.weight_decay,
                eps=getattr(args.optimizer, "adamw_eps", 1e-8),
            )

        self.init_steps = 0
        self.lr_scheduler = get_scheduler(
            args.optimizer.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=args.optimizer.lr_warmup_steps,
            num_training_steps=args.optimizer.max_train_steps,
            last_epoch=self.init_steps - 1,
        )

        # Initialize adaptive gradient clipping (if enabled)
        from hymm.utils.train_utils import init_adaptive_grad_clipper

        init_adaptive_grad_clipper(args)

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    def _maybe_resume(self):
        args, logger = self.args, self.logger
        self.save_optimizer_state = getattr(args, "save_optimizer_state", False)
        if (
            self.finetune_mode == "lora"
            and self._enable_dcp_checkpoint
            and not self.save_optimizer_state
        ):
            raise ValueError(
                "LoRA production checkpointing requires "
                "training.optimizer.save_optimizer_state=true"
            )
        self.scalar_states = None
        # Stash the resolved checkpoint path so train() can call
        # load_rng_states() after _create_dataloader / _init_trackers.
        # Loading RNG state here would be too early: third-party libraries
        # invoked later (notably swanlab.init in _init_trackers) consume the
        # global Python random state, which would advance the just-restored
        # state and break bitwise resume reproducibility.
        self._pending_rng_resume_path = None

        # When True, never load EMA shadow weights from the checkpoint (in both
        # auto- and explicit-resume paths) and don't require an EMA folder to be
        # present for auto-resume validation. After the regular model checkpoint
        # is restored, EMA is initialized from those resumed model weights.
        # Useful for resuming a checkpoint that has no EMA shards while EMA
        # tracking is configured for the new run.
        skip_ema_load = bool(getattr(args, "skip_ema_load", False))

        resume_path = getattr(args, "resume", None)
        if self.finetune_mode == "lora" and resume_path:
            if not self._enable_dcp_checkpoint:
                raise ValueError(
                    "LoRA optimizer resume requires "
                    "training.optimizer.enable_dcp_checkpoint=true"
                )
            if str(resume_path).strip().lower() == "auto":
                resume_path = auto_find_latest_lora_checkpoint(
                    args.output_dir, logger=logger
                )
            if resume_path:
                scalar_state, rng_state, scheduler_state = (
                    load_lora_training_checkpoint(
                        self.model,
                        self.optimizer,
                        resume_path,
                        shard_group=self._ckpt_dp_shard_sp_group,
                        logger=logger,
                    )
                )
                self.scalar_states = ScalarStates.from_pretrained(scalar_state)
                self.init_steps = self.scalar_states.train_steps
                if scheduler_state is not None:
                    self.lr_scheduler.load_state_dict(scheduler_state)
                self._pending_rng_resume_state = rng_state
                logger.info(
                    f"Resumed LoRA training from step {self.init_steps}: {resume_path}"
                )
                return
            logger.info(
                "[auto-resume] No complete LoRA checkpoint found; starting fresh"
            )
            resume_path = None
        if resume_path and str(resume_path).strip().lower() == "auto":
            resume_path = auto_find_latest_checkpoint(
                args.output_dir,
                logger=logger,
                required_ema_rates=[] if skip_ema_load else self.ema_rates,
                require_optimizer=self.save_optimizer_state,
            )
            if resume_path is None:
                logger.info(
                    "[auto-resume] No checkpoint found, starting training from scratch."
                )

        if resume_path and str(resume_path) != "None":
            if not os.path.exists(resume_path):
                raise ValueError("Invalid resume path: %s" % resume_path)
            logger.info(f"Resuming training from {resume_path}")
            _t_ckpt = time.time()
            self.model, _ss_list, optimizer_loaded = load_checkpoint(
                self.model,
                self.optimizer,
                resume_path,
                load_optimizer_state=self.save_optimizer_state,
            )
            if self.save_optimizer_state:
                logger.info(
                    "Loaded optimizer state"
                    if optimizer_loaded
                    else "Optimizer state not found"
                )
            logger.info(
                f"[INIT_TIMER]   checkpoint model+optim loaded in {time.time() - _t_ckpt:.1f}s"
            )

            reset_progress = bool(getattr(args, "reset_progress_on_resume", False))
            if reset_progress:
                # Keep the loaded model weights + optimizer momentum, but treat
                # this as the start of a fresh training run: zero the step
                # counter, dataset cursor (consumed_* / epoch), and LR schedule.
                # Used when resuming weights onto a *different* dataset and we
                # want iter / dataset index / LR warmup to all start from 0.
                self.scalar_states = ScalarStates(lr=args.optimizer.learning_rate)
                self.init_steps = 0
                ss = self.scalar_states
                logger.info(
                    "[resume] reset_progress_on_resume=True: loaded model+optimizer "
                    "from checkpoint but reset train_steps / dataset cursor / LR "
                    "schedule to 0 (fresh-run semantics)."
                )
            else:
                self.scalar_states = ScalarStates.from_pretrained(
                    _ss_list,
                    rank=self.rank,
                    world_size=self.world_size,
                    default_rank0_ss=False,
                )
                self.init_steps = self.scalar_states.train_steps
                ss = self.scalar_states
            logger.info(
                f"Resumed from step {self.init_steps} | "
                f"consumed_groups_per_dp={dict(ss.consumed_groups_per_dp)} | "
                f"consumed_samples_per_dp={dict(ss.consumed_samples_per_dp)} | "
                f"epoch={dict(ss.epoch)} | "
                f"save_optimizer_state={self.save_optimizer_state}"
            )
            self.lr_scheduler = get_scheduler(
                args.optimizer.lr_scheduler,
                optimizer=self.optimizer,
                num_warmup_steps=args.optimizer.lr_warmup_steps,
                num_training_steps=args.optimizer.max_train_steps,
                last_epoch=self.init_steps - 1,
            )
            for param_group in self.optimizer.param_groups:
                param_group["initial_lr"] = args.optimizer.learning_rate
            # NOTE: RNG state restore is deferred to train() (just before the
            # main loop) so that the resumed Python/CPU/CUDA/numpy RNG state
            # is not advanced by post-resume init steps such as swanlab.init,
            # which internally consumes Python's global random module.
            # When resetting progress (fresh-run semantics on a new dataset),
            # skip RNG restore so shuffling starts from a clean state.
            self._pending_rng_resume_path = None if reset_progress else resume_path
            if self.ema_rates and not skip_ema_load:
                _t_ema = time.time()
                load_ema_states_from_checkpoint(
                    self.ema_states, resume_path, self.model, self.ema_rates
                )
                logger.info(
                    f"[INIT_TIMER]   ema states loaded in {time.time() - _t_ema:.1f}s"
                )
            elif self.ema_rates and skip_ema_load:
                # EMA was allocated before _maybe_resume(), when self.model still
                # held the initial dit_weight. Synchronize the existing shadows
                # in-place now that the regular resume checkpoint has been
                # restored. Reusing the allocation avoids holding two fp32 EMA
                # copies at once.
                sync_ema_from_model(self.ema_states, self.model)
                logger.info(
                    "[resume] skip_ema_load=True: not loading EMA shadow weights "
                    "from checkpoint; EMA synchronized from resumed model weights."
                )
            # Distillation transports resume their critic weights/optimizer from
            # the SAME resolved checkpoint dir (which may have been auto-detected
            # as "latest"), so hand it over explicitly rather than re-resolving.
            if hasattr(self.transport, "set_resolved_resume_dir"):
                self.transport.set_resolved_resume_dir(resume_path)

    # ------------------------------------------------------------------
    # Dataloader
    # ------------------------------------------------------------------

    def _create_dataloader(self):
        args = self.args
        if self.scalar_states is None:
            self.scalar_states = ScalarStates(lr=args.optimizer.learning_rate)
        dataset, sampler, video_loader = create_single_file_sft_dataloader(
            args,
            text_encoder=self.text_encoder,
            dp_rank=self.dp_rank,
            dp_degree=self.dp_degree,
        )
        self.logger.info(
            f"Single-file segment SFT dataset: {args.dataset.data_file} "
            f"({len(dataset)} samples)"
        )
        loader = RepeatingSFTIterator(video_loader, sampler, self.scalar_states)
        self.data_iter = iter(loader)
        self._loader = loader

    # ------------------------------------------------------------------
    # Experiment trackers (TensorBoard / W&B)
    # ------------------------------------------------------------------

    def _init_trackers(self):
        args, logger = self.args, self.logger

        _raw_tracker = getattr(args, "tracker", None)
        tracker = (_raw_tracker or "tensorboard").lower()
        use_tb = tracker in ("tensorboard", "both")
        use_swanlab = tracker in ("swanlab", "both")
        if self.rank == 0:
            logger.info(
                f"[TRACKER] raw={_raw_tracker!r} resolved={tracker!r} use_tb={use_tb} use_swanlab={use_swanlab}"
            )

        self.tb_writer = None
        if self.rank == 0 and use_tb:
            self.tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tb"))

        self.swanlab_active = False
        if self.rank == 0 and use_swanlab:
            if swanlab is None:
                logger.warning("[SWANLAB] swanlab requested but not installed!")
            else:
                swanlab_api_key = getattr(args, "swanlab_api_key", None)
                swanlab_project = getattr(
                    args, "swanlab_project", "huanyuan-worldmodel"
                )
                swanlab_workspace = getattr(args, "swanlab_workspace", None)
                # Custom host support (for self-hosted / internal swanlab deployments)
                # Set EITHER swanlab_host (single base URL, we derive web/api from it)
                # OR swanlab_api_host + swanlab_web_host (for fine-grained control).
                swanlab_host = getattr(args, "swanlab_host", None)
                swanlab_api_host = getattr(args, "swanlab_api_host", None)
                swanlab_web_host = getattr(args, "swanlab_web_host", None)
                if swanlab_host:
                    base = swanlab_host.rstrip("/")
                    if not swanlab_api_host:
                        swanlab_api_host = f"{base}/api"
                    if not swanlab_web_host:
                        swanlab_web_host = base
                if swanlab_api_host:
                    os.environ["SWANLAB_API_HOST"] = swanlab_api_host
                if swanlab_web_host:
                    os.environ["SWANLAB_WEB_HOST"] = swanlab_web_host
                if swanlab_api_key:
                    os.environ["SWANLAB_API_KEY"] = swanlab_api_key
                try:
                    if swanlab_api_key:
                        # swanlab.login signature (verified against 0.7.19,
                        # swanlab/data/sdk.py): api_key / host / web_host / save.
                        # There is NO `api_host` kwarg — passing it raises
                        # TypeError that the outer try/except silently swallows
                        # and disables the tracker.  api_host is consumed via
                        # the SWANLAB_API_HOST env var set above instead.
                        login_kwargs = dict(api_key=swanlab_api_key, save=False)
                        if swanlab_web_host:
                            login_kwargs["web_host"] = swanlab_web_host
                        swanlab.login(**login_kwargs)
                    run_name = getattr(args, "name", None) or os.path.basename(
                        args.output_dir
                    )
                    init_kwargs = dict(
                        project=swanlab_project,
                        experiment_name=run_name,
                        config=vars(args) if hasattr(args, "__dict__") else {},
                        logdir=args.output_dir,
                    )
                    if swanlab_workspace:
                        init_kwargs["workspace"] = swanlab_workspace
                    swanlab.init(**init_kwargs)
                    self.swanlab_active = True
                    logger.info(
                        f"[SWANLAB] Initialized: project={swanlab_project} "
                        f"experiment={run_name} host={swanlab_web_host or '<default>'}"
                    )
                except Exception as e:
                    logger.error(f"[SWANLAB] Failed: {e}")

    # ------------------------------------------------------------------
    # Training config summary
    # ------------------------------------------------------------------

    def _print_config(self):
        args, logger, model = self.args, self.logger, self.model
        params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.params_count = params_count
        training_mode = self.training_mode

        logger.info(
            "****************************** Running Segment Self-Forcing SFT ******************************"
        )

        logger.info("=" * 80)
        logger.info("SYSTEM & HARDWARE CONFIGURATION")
        logger.info("=" * 80)
        logger.info(
            f"Number of GPUs                                   : {self.world_size}"
        )
        logger.info(
            f"Local rank                                       : {self.local_rank}"
        )
        logger.info(f"Global rank                                      : {self.rank}")
        logger.info(
            f"Device                                           : cuda:{self.local_rank}"
        )
        logger.info(
            f"Master weight dtype                              : {model.parameters().__next__().dtype}"
        )

        logger.info("-" * 80)
        logger.info("MODEL PARAMETERS & ARCHITECTURE")
        logger.info("-" * 80)
        logger.info(
            f"Trainable parameters                             : {params_count / 1e9:.3f}B"
        )
        logger.info(
            f"Model type                                       : {args.dit.model_type if hasattr(args, 'dit') and hasattr(args.dit, 'model_type') else 'N/A'}"
        )
        logger.info(
            f"Training mode                                    : {training_mode}"
        )
        logger.info(
            f"DIT model path                                   : {args.dit_weight if hasattr(args, 'dit_weight') else 'N/A'}"
        )
        logger.info(
            f"Gradient checkpointing                           : {args.parallelism.gradient_checkpointing}"
        )
        logger.info(
            f"Camera module checkpointing                      : {getattr(args.parallelism, 'checkpoint_camera_modules', True) and getattr(args.camera, 'use_camera_control', False) and args.parallelism.gradient_checkpointing}"
        )

        # FLOPs estimation config (attributes are initialized in train() for all ranks)
        gc_enabled = self._gc_enabled
        gc_ratio = self._gc_ratio
        flops_factor = 6 + 2 * gc_ratio
        logger.info(
            f"GPU peak bf16 TFLOPS                             : {self._gpu_peak_tflops}"
        )
        logger.info(
            f"FLOPs factor                                     : {flops_factor} (gc={gc_enabled}, sel_ckpt={getattr(args.parallelism, 'selective_checkpointing', 1.0)})"
        )
        _tr = getattr(self, "transport", None)
        _pse_mp = getattr(_tr, "pse_max_prob", 0.0) if _tr is not None else 0.0
        if (
            _tr is not None
            and getattr(_tr, "enable", False)
            and _pse_mp
            and _pse_mp > 0
        ):
            logger.info(
                f"PSE forward passes (FLOPs x)                     : 2 "
                f"(pse_max_prob={_pse_mp}, pse_ramp_steps={getattr(_tr, 'pse_ramp_steps', 5000)}; "
                f"two-pass active once p>0, FLOPs/MFU scaled x2)"
            )
        else:
            logger.info(
                "PSE forward passes (FLOPs x)                     : 1 (PSE off)"
            )

        _attn_mode = getattr(args.dit, "attn_mode", "flash")
        _df_cs = int(getattr(_tr, "chunk_size", 0) or 0) if _tr is not None else 0
        if "causal" in (_attn_mode or "") and _df_cs > 0:
            logger.info(
                f"Attention FLOPs model                            : chunk-causal "
                f"(chunk_size={_df_cs} frames, ~S^2/2; was ~S^2 full -> fixes inflated MFU)"
            )
        elif "causal" in (_attn_mode or ""):
            logger.info(
                "Attention FLOPs model                            : segment-causal"
            )
        else:
            logger.info(
                "Attention FLOPs model                            : full (non-causal)"
            )

        logger.info("-" * 80)
        logger.info("DISTRIBUTED TRAINING CONFIGURATION")
        logger.info("-" * 80)
        logger.info(
            f"World size                                       : {self.world_size}"
        )
        logger.info(
            f"DP degree                                        : {self.dp_degree}"
        )
        logger.info(
            f"DP rank                                          : {self.dp_rank}"
        )
        logger.info(
            f"SP size                                          : {args.parallelism.sp_size}"
        )
        logger.info(
            f"FSDP sharding strategy                           : {args.parallelism.fsdp_sharding_strategy}"
        )

        total_batch_size = (
            args.dataset.video_batch_size
            * self.dp_degree
            * int(getattr(args.parallelism, "gradient_accumulation_steps", 1))
        )
        logger.info("-" * 80)
        logger.info("VIDEO CONFIGURATION")
        logger.info("-" * 80)
        logger.info(
            f"Video size (H x W)                               : {args.inference.video_size if hasattr(args, 'inference') else 'N/A'}"
        )
        logger.info(
            "Sample n frames                                  : "
            f"{getattr(args.dataset, 'sample_n_frames', 'variable (from data file)')}"
        )
        logger.info(
            f"Video batch size                                 : {args.dataset.video_batch_size}"
        )
        logger.info(
            f"Total batch size                                 : {total_batch_size}"
        )

        logger.info("-" * 80)
        logger.info("OPTIMIZATION CONFIGURATION")
        logger.info("-" * 80)
        optimizer_type = getattr(
            args, "optimizer_type", getattr(args.optimizer, "name", "muon")
        )
        adamw_betas = tuple(getattr(args.optimizer, "adamw_betas", (0.9, 0.95)))
        adamw_eps = getattr(args.optimizer, "adamw_eps", 1e-8)
        adamw_weight_decay = (
            getattr(args.optimizer, "adamw_weight_decay", args.optimizer.weight_decay)
            if str(optimizer_type).lower() == "muon"
            else args.optimizer.weight_decay
        )
        logger.info(
            f"Optimizer                                        : {optimizer_type}"
        )
        logger.info(
            f"Learning rate                                    : {args.optimizer.learning_rate}"
        )
        logger.info(
            f"Weight decay                                     : {args.optimizer.weight_decay}"
        )
        logger.info(f"AdamW betas                                      : {adamw_betas}")
        logger.info(f"AdamW eps                                        : {adamw_eps}")
        logger.info(
            f"AdamW weight decay                               : {adamw_weight_decay}"
        )
        logger.info(
            f"Muon momentum                                    : {getattr(args.optimizer, 'momentum', 0.95)}"
        )
        logger.info(
            f"Muon communication mode                         : {getattr(args.optimizer, 'muon_comm_mode', 'legacy')}"
        )
        logger.info(
            f"Muon exclude patterns                            : {getattr(args.optimizer, 'muon_exclude_patterns', [])}"
        )
        logger.info(
            f"Max gradient norm                                : {args.optimizer.max_grad_norm}"
        )
        logger.info(
            f"Enable gradient norm clipping                    : {getattr(args, 'enable_grad_norm_clip', True)}"
        )
        logger.info(
            f"Adaptive gradient clipping                       : {getattr(args.optimizer, 'adaptive_grad_clip', False)}"
        )
        if getattr(args.optimizer, "adaptive_grad_clip", False):
            logger.info(
                f"  factor={args.optimizer.adaptive_grad_clip_factor}, min={args.optimizer.adaptive_grad_clip_min}, ema_decay={args.optimizer.adaptive_grad_clip_ema_decay}"
            )
        logger.info(
            f"Gradient accumulation steps                      : {args.parallelism.gradient_accumulation_steps}"
        )
        logger.info(
            f"LR scheduler                                     : {args.optimizer.lr_scheduler}"
        )
        logger.info(
            f"Warmup steps                                     : {args.optimizer.lr_warmup_steps}"
        )
        logger.info(
            f"Checkpointing steps                              : {args.optimizer.checkpointing_steps}"
        )
        logger.info(
            f"Save optimizer state                             : {args.optimizer.save_optimizer_state}"
        )
        logger.info(
            f"Enable production checkpoint                    : {self._enable_dcp_checkpoint}"
        )
        checkpoint_format = (
            "async LoRA HSDP shards"
            if self.finetune_mode == "lora" and self._enable_dcp_checkpoint
            else "synchronous safetensors"
        )
        logger.info(
            f"Checkpoint format                               : {checkpoint_format}"
        )
        logger.info(
            f"Dataset type                                     : {getattr(args, 'dataset_type', 'default')}"
        )
        logger.info("=" * 80)

        if getattr(args.camera, "use_camera_control", False):
            logger.info("-" * 80)
            logger.info("CAMERA CONTROL MODULES")
            logger.info("-" * 80)
            logger.info(
                f"Injection mode                                   : {getattr(args.camera, 'injection_mode', None)}"
            )
            logger.info(
                f"Freeze DiT for camera                            : {getattr(args.camera, 'freeze_dit_for_camera', False)}"
            )
            print_camera_param_summary(model, logger)
            logger.info("=" * 80)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self):
        start = time.perf_counter()
        ss = self.scalar_states
        if self.finetune_mode == "lora":
            if self._enable_dcp_checkpoint:
                adapter_dir = save_lora_training_checkpoint(
                    self.model,
                    self.optimizer,
                    self.lr_scheduler,
                    self.args.output_dir,
                    ss,
                    self.lora_config,
                    dp_replicate_rank=self._ckpt_dp_replicate_rank,
                    shard_group=self._ckpt_dp_shard_sp_group,
                    logger=self.logger,
                )
                if self.rank == 0:
                    self.logger.info(
                        f"[checkpoint] async LoRA checkpoint staged in "
                        f"{time.perf_counter() - start:.2f}s at {adapter_dir}"
                    )
                return
            adapter_dir = os.path.join(
                self.args.output_dir, "checkpoints", f"global_step-{ss.update_steps}"
            )
            save_lora_adapter(
                self.model,
                adapter_dir,
                rank=self.lora_config["rank"],
                alpha=self.lora_config["alpha"],
                dropout=self.lora_config["dropout"],
            )
            error = None
            if self.rank == 0:
                try:
                    with open(
                        os.path.join(adapter_dir, "training_state.json"), "w"
                    ) as file:
                        json.dump({"update_steps": ss.update_steps}, file)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            messages = [error]
            dist.broadcast_object_list(messages, src=0)
            if messages[0]:
                raise RuntimeError(f"LoRA checkpoint write failed: {messages[0]}")
        else:
            save_checkpoint(
                self.model,
                self.optimizer,
                self.args.output_dir,
                ss,
                checkpoint_step=ss.update_steps,
                save_optimizer_state=self.save_optimizer_state,
                ema_states=self.ema_states,
                ema_rates=self.ema_rates,
                logger=self.logger,
            )

        if self.rank == 0:
            self.logger.info(
                f"[checkpoint] synchronous save finished in "
                f"{time.perf_counter() - start:.2f}s at step {ss.update_steps}"
            )

    # ------------------------------------------------------------------
    # FLOPs estimation (per-block matmul counting)
    # ------------------------------------------------------------------

    def _pse_forward_passes(self, global_step=None):
        """Number of full model forward+backward passes per step.

        PSE (Progressive Self-Exposure) runs a TWO-pass forward whenever the
        replacement probability ``p = min(pse_max_prob, step/ramp)`` is > 0:
        every token goes through the model twice (pass-1 noisy + pass-2 mixed)
        with two backwards. ``p`` only decides which tokens get their input
        swapped, NOT whether pass-2 runs, so the multiplier is exactly 2 as soon
        as ``p > 0``. Without this, MFU is under-counted by ~2x for PSE runs.
        """
        tr = getattr(self, "transport", None)
        if tr is None:
            return 1
        pse_max_prob = getattr(tr, "pse_max_prob", 0.0) or 0.0
        if not (getattr(tr, "enable", False) and pse_max_prob > 0):
            return 1
        if global_step is None:
            global_step = getattr(
                getattr(self, "scalar_states", None), "train_steps", 0
            )
        ramp = getattr(tr, "pse_ramp_steps", 5000)
        p = min(pse_max_prob, global_step / max(1, ramp))
        return 2 if p > 0 else 1

    def _estimate_step_flops_from_batch(self, batch, global_step=None):
        """Compute per-GPU FLOPs for one training step from the batch.

        Returns (step_flops, n_img_tokens) or (None, None) if unavailable.
        """
        args = self.args
        model = self.model
        if batch is None:
            return None, None

        # PSE two-pass forward runs the model twice per step (see
        # _pse_forward_passes); scale the single-pass estimate accordingly.
        pse_passes = self._pse_forward_passes(global_step)

        attn_mode = getattr(args.dit, "attn_mode", "flash")
        reorg_token = "888" in args.dit.model_type or "16168" in args.dit.model_type
        txt_per_block = getattr(args.text_encoder, "text_len", 256) or 256
        patch_size = model.patch_size if hasattr(model, "patch_size") else [1, 1, 1]

        # Chunk-level DF uses global chunk-wise causal attention (~S^2/2). The
        # chunk granularity (in latent frames) must match unified.py's
        # chunk_size_tokens = chunk_size * tokens_per_frame so the FLOPs estimate
        # reflects the actual magi mask instead of ~full attention.
        df_chunk_size = int(
            getattr(getattr(self, "transport", None), "chunk_size", 0) or 0
        )

        def _chunk_size_tokens(tokens_per_frame):
            if df_chunk_size == 1:
                return tokens_per_frame
            if df_chunk_size > 1:
                return df_chunk_size * tokens_per_frame
            return 0

        # Token-packed batch: attention is block-diagonal (per-sample),
        # so we compute FLOPs per sample and sum.
        if isinstance(batch, dict) and batch.get("is_token_packed"):
            sample_shapes = batch.get("sample_shapes", [])
            kwargs_list = batch.get("kwargs_list", [{}] * len(sample_shapes))
            text_mask = batch.get("packed_text_mask", None)
            num_packed = len(sample_shapes)
            pt, ph, pw = patch_size

            total_flops = 0
            total_img_tokens = 0
            for i in range(num_packed):
                T_i, H_i, W_i = sample_shapes[i]
                tokens_per_frame_i = (H_i // ph) * (W_i // pw)
                n_img_i = (T_i // pt) * tokens_per_frame_i
                total_img_tokens += n_img_i

                # Per-sample text token count
                if (
                    text_mask is not None
                    and text_mask.dim() >= 2
                    and i < text_mask.shape[0]
                ):
                    n_txt_i = int(text_mask[i].sum().item())
                else:
                    n_txt_i = txt_per_block

                # Per-sample prompt_video_boundaries (in latent frames)
                kw_i = kwargs_list[i] if i < len(kwargs_list) else {}
                if not isinstance(kw_i, dict):
                    kw_i = {}
                pvb_i = kw_i.get("prompt_video_boundaries", None)
                if pvb_i is not None:
                    if isinstance(pvb_i, torch.Tensor):
                        pvb_i = pvb_i.squeeze().tolist()
                    if (
                        isinstance(pvb_i, (list, tuple))
                        and pvb_i
                        and isinstance(pvb_i[0], list)
                    ):
                        pvb_i = pvb_i[0]
                    pvb_i = [int(b) for b in pvb_i]

                total_flops += estimate_step_flops(
                    model=model,
                    n_img_tokens=n_img_i,
                    n_txt_tokens=n_txt_i,
                    batch_size=1,
                    sp_size=args.parallelism.sp_size,
                    gc_enabled=self._gc_enabled,
                    gc_ratio=self._gc_ratio,
                    skip_attn_recompute=self._skip_attn_recompute,
                    attn_mode=attn_mode,
                    prompt_video_boundaries_frames=pvb_i,
                    txt_tokens_per_block=txt_per_block,
                    chunk_size_tokens=_chunk_size_tokens(tokens_per_frame_i),
                )
            return total_flops * pse_passes, total_img_tokens

        # Standard (non-packed) batch: (video_rgb, latents, text_ids, text_mask, *extra)
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            latents = batch[1]
            if not hasattr(latents, "shape") or latents.dim() != 5:
                return None, None

            if reorg_token:
                orig_shape = list(latents.shape)
                orig_shape[2] = orig_shape[2] + 1
                n_img_tokens = estimate_n_tokens(
                    orig_shape, patch_size, reorg_token=True
                )
            else:
                n_img_tokens = estimate_n_tokens(
                    latents.shape, patch_size, reorg_token=False
                )

            text_mask = batch[3] if len(batch) > 3 else None
            if text_mask is not None and text_mask.dim() >= 2:
                n_txt_tokens = int(text_mask[0].sum().item())
            else:
                text_ids = batch[2] if len(batch) > 2 else None
                n_txt_tokens = text_ids.shape[-1] if text_ids is not None else 256

            batch_kwargs = batch[4] if len(batch) > 4 else {}
            if not isinstance(batch_kwargs, dict):
                batch_kwargs = {}
            pvb = batch_kwargs.get("prompt_video_boundaries", None)
            if pvb is not None:
                if isinstance(pvb, torch.Tensor):
                    pvb = pvb.tolist()
                if isinstance(pvb[0], list):
                    pvb = pvb[0]
                pvb = [int(b) for b in pvb]

            batch_size = latents.shape[0]
            # latents: [B, C, T, H, W] -> tokens/frame after patchify
            ph_, pw_ = patch_size[1], patch_size[2]
            tokens_per_frame = (latents.shape[3] // ph_) * (latents.shape[4] // pw_)
            step_flops = estimate_step_flops(
                model=model,
                n_img_tokens=n_img_tokens,
                n_txt_tokens=n_txt_tokens,
                batch_size=batch_size,
                sp_size=args.parallelism.sp_size,
                gc_enabled=self._gc_enabled,
                gc_ratio=self._gc_ratio,
                skip_attn_recompute=self._skip_attn_recompute,
                attn_mode=attn_mode,
                prompt_video_boundaries_frames=pvb,
                txt_tokens_per_block=txt_per_block,
                chunk_size_tokens=_chunk_size_tokens(tokens_per_frame),
            )
            return step_flops * pse_passes, n_img_tokens

        return None, None

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_metrics(
        self,
        loss,
        grad_norm,
        loss_dict_or_terms,
        sec_per_step,
        steps_per_sec,
        batch=None,
        data_load_time=None,
        train_step_time=None,
    ):
        ss = self.scalar_states
        lr = self.optimizer.param_groups[0]["lr"]

        vis_dict = {}
        if isinstance(loss_dict_or_terms, dict) and "vis_dict" in loss_dict_or_terms:
            vis_dict = loss_dict_or_terms["vis_dict"]

        # Build timing detail string
        _timing = ""
        if data_load_time is not None and train_step_time is not None:
            _timing = (
                f" data_load={data_load_time:.2f}s train_step={train_step_time:.2f}s"
            )

        # Build packing / data_source detail string
        _extra = ""
        if batch is not None:
            if isinstance(batch, dict) and batch.get("is_token_packed"):
                _num_packed = batch.get("num_packed_samples", "?")
                _total_tokens = batch.get("total_packed_tokens", "?")
                _extra = f" | packed: {_num_packed} samples, {_total_tokens} tokens"
                _kw_list = batch.get("kwargs_list", [])
                from collections import Counter

                _source_counts = Counter()
                _videoids = []
                for _kw in _kw_list:
                    if isinstance(_kw, dict):
                        if "data_source" in _kw:
                            _source_counts[str(_kw["data_source"])] += 1
                        if "videoid" in _kw:
                            _videoids.append(str(_kw["videoid"]))
                if _source_counts:
                    _breakdown = ", ".join(
                        f"{src}×{cnt}" for src, cnt in sorted(_source_counts.items())
                    )
                    _extra += f" | sources: [{_breakdown}]"
                if _videoids:
                    _extra += f" | videoids: [{', '.join(_videoids)}]"
            elif isinstance(batch, (list, tuple)) and len(batch) > 0:
                _kw = batch[-1]
                if isinstance(_kw, dict) and "data_source" in _kw:
                    _src = _kw["data_source"]
                    if isinstance(_src, (list, tuple)):
                        _src = _src[0] if len(_src) == 1 else str(_src)
                    _extra = f" | source: {_src}"

        # Compute per-GPU TFLOPs and MFU using per-block matmul counting
        step_flops, _n_img_tokens = self._estimate_step_flops_from_batch(batch)
        _step_time_for_flops = (
            train_step_time if train_step_time and train_step_time > 0 else sec_per_step
        )
        if step_flops is not None and _step_time_for_flops > 0:
            tflops_per_gpu = step_flops / _step_time_for_flops / 1e12
            gpu_peak = getattr(self, "_gpu_peak_tflops", 0)
            mfu = tflops_per_gpu / gpu_peak if gpu_peak > 0 else 0.0
            step_tflops = step_flops / 1e12
            _tflops_str = f" est_tflops={step_tflops:.1f} tflops/gpu={tflops_per_gpu:.1f} mfu={mfu * 100:.1f}%"
        else:
            tflops_per_gpu = None
            mfu = None
            step_tflops = None
            _tflops_str = ""

        from hymm.utils.train_utils import get_clip_value

        _effective_clip = get_clip_value(self.args)
        _clip_str = ""
        if _effective_clip < float("inf") and grad_norm > _effective_clip:
            _clip_str = (
                f" [CLIPPED {grad_norm / _effective_clip:.1f}x → {_effective_clip:.3f}]"
            )

        _camera_str = ""
        if "camera/has_camera_ratio" in vis_dict:
            _camera_str += f" cam_ratio={vis_dict['camera/has_camera_ratio']:.2f}"
        if "camera/grad_rms_local" in vis_dict:
            _camera_str += f" cam_grms={vis_dict['camera/grad_rms_local']:.4g}"
        if "backbone/grad_rms_local" in vis_dict:
            _camera_str += f" bb_grms={vis_dict['backbone/grad_rms_local']:.4g}"
        if "camera/seg_dropout_samples" in vis_dict:
            _camera_str += (
                f" cam_drop={vis_dict['camera/seg_dropout_samples']:.0f}/"
                f"{vis_dict.get('camera/seg_dropout_segments', 0):.0f}"
            )

        _tail = (
            f"step_time={sec_per_step:.2f}s steps/s={steps_per_sec:.2f}"
            f"{_tflops_str}{_timing}{_extra}"
        )
        self.logger.info(
            f"Progress: {ss.update_steps}/{self.args.optimizer.max_train_steps} | "
            f"loss={loss:.8f} grad_norm={grad_norm:.6f}{_clip_str} lr={lr:.8f}"
            f"{_camera_str} {_tail}"
        )

        if self.tb_writer is not None:
            try:
                self.tb_writer.add_scalar("train/loss", loss, ss.update_steps)
                self.tb_writer.add_scalar("train/grad_norm", grad_norm, ss.update_steps)
                self.tb_writer.add_scalar("train/lr", lr, ss.update_steps)
                self.tb_writer.add_scalar(
                    "train/epoch", ss.epoch.get("video", 0), ss.update_steps
                )
                self.tb_writer.add_scalar(
                    "Time/step_time", sec_per_step, ss.update_steps
                )
                self.tb_writer.add_scalar(
                    "Time/steps_per_sec", steps_per_sec, ss.update_steps
                )
                if tflops_per_gpu is not None:
                    self.tb_writer.add_scalar(
                        "Performance/est_tflops", step_tflops, ss.update_steps
                    )
                    self.tb_writer.add_scalar(
                        "Performance/tflops_per_gpu", tflops_per_gpu, ss.update_steps
                    )
                    self.tb_writer.add_scalar("Performance/mfu", mfu, ss.update_steps)
                    cum_pflops = getattr(self, "_cumulative_flops", 0) / _PFLOPS_SCALE
                    self.tb_writer.add_scalar(
                        "Performance/cumulative_pflops", cum_pflops, ss.update_steps
                    )
                for key, value in vis_dict.items():
                    if not isinstance(value, (int, float)):
                        continue
                    self.tb_writer.add_scalar(f"train/{key}", value, ss.update_steps)
            except OSError as _tb_err:
                if not getattr(self, "_tb_warn_logged", False):
                    self.logger.warning(
                        f"TensorBoard write failed (will suppress future warnings): {_tb_err}"
                    )
                    self._tb_warn_logged = True

        if self.swanlab_active:
            log_data = {
                "train/loss": loss,
                "train/grad_norm": grad_norm,
                "train/lr": lr,
                "train/epoch": ss.epoch.get("video", 0),
                "Time/step_time": sec_per_step,
                "Time/steps_per_sec": steps_per_sec,
            }
            if tflops_per_gpu is not None:
                log_data["Performance/est_tflops"] = step_tflops
                log_data["Performance/tflops_per_gpu"] = tflops_per_gpu
                log_data["Performance/mfu"] = mfu
                log_data["Performance/cumulative_pflops"] = (
                    getattr(self, "_cumulative_flops", 0) / _PFLOPS_SCALE
                )
            for key, value in vis_dict.items():
                if not isinstance(value, (int, float)):
                    continue
                log_data[f"train/{key}"] = value
            if self.ema_schedule:
                log_data["train/ema_rate"] = self._get_ema_rate_for_step(
                    ss.update_steps
                )
            swanlab.log(log_data, step=ss.update_steps)

    # ------------------------------------------------------------------
    # Benchmark summary
    # ------------------------------------------------------------------

    def _print_benchmark_summary(self, step_times, warmup_steps, active_steps, args):
        """Print a wall-clock benchmark summary table and exit."""
        import statistics

        if not step_times:
            self.logger.warning("[Benchmark] No active steps recorded.")
            return

        data_times = [t[0] for t in step_times]
        train_times = [t[1] for t in step_times]
        total_times = [t[2] for t in step_times]

        peak_mem_gb = torch.cuda.max_memory_allocated(self.device) / (1024**3)
        peak_mem_reserved_gb = torch.cuda.max_memory_reserved(self.device) / (1024**3)

        is_token_packed = getattr(args, "token_packing", False)
        mode_str = "Token Packing" if is_token_packed else "Standard (no packing)"

        if self.rank == 0:
            print(f"\n{'=' * 80}")
            print(f"  BENCHMARK SUMMARY — {mode_str}")
            print(f"{'=' * 80}")
            print(f"  Warmup steps (excluded):  {warmup_steps}")
            print(
                f"  Active steps (measured):  {active_steps} (actual: {len(step_times)})"
            )
            print(f"{'─' * 80}")
            print(f"  {'Metric':<30} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
            print(f"  {'─' * 70}")

            for name, vals in [
                ("Data loading (s)", data_times),
                ("Train step (s)", train_times),
                ("Total step (s)", total_times),
            ]:
                mean = statistics.mean(vals)
                std = statistics.stdev(vals) if len(vals) > 1 else 0.0
                mn = min(vals)
                mx = max(vals)
                print(
                    f"  {name:<30} {mean:>10.3f} {std:>10.3f} {mn:>10.3f} {mx:>10.3f}"
                )

            avg_total = statistics.mean(total_times)
            avg_train = statistics.mean(train_times)
            _bench_step_flops, _bench_n_img = self._estimate_step_flops_from_batch(
                getattr(self, "_last_batch", None)
            )
            # Recover the token counts that feed the FLOPs/MFU estimate so the
            # benchmark log records S_img / S_txt / S_img_local / S_total.
            _bench_sp = max(int(args.parallelism.sp_size), 1)
            _bench_s_img = _bench_n_img
            _bench_s_txt = None
            _lb = getattr(self, "_last_batch", None)
            if isinstance(_lb, dict) and _lb.get("is_token_packed"):
                _pm = _lb.get("packed_text_mask", None)
                if _pm is not None and hasattr(_pm, "dim") and _pm.dim() >= 2:
                    _bench_s_txt = int(_pm.sum().item())
            elif isinstance(_lb, (list, tuple)) and len(_lb) > 3:
                _tm = _lb[3]
                if _tm is not None and hasattr(_tm, "dim") and _tm.dim() >= 2:
                    _bench_s_txt = int(_tm[0].sum().item())
            print(f"{'─' * 80}")
            print(
                f"  Throughput:  {1.0 / avg_total:.2f} steps/s  ({avg_total:.3f} s/step)"
            )
            if _bench_s_img is not None:
                _s_img_local = _bench_s_img // _bench_sp
                _s_total = _bench_s_img + (_bench_s_txt or 0)
                print(f"  S_img (image tokens):       {_bench_s_img}")
                print(
                    f"  S_txt (text tokens):        {_bench_s_txt if _bench_s_txt is not None else 'n/a'}"
                )
                print(f"  S_img_local (S_img/sp):     {_s_img_local}  (sp={_bench_sp})")
                print(f"  S_total (S_img+S_txt):      {_s_total}")
            if _bench_step_flops is not None and avg_train > 0:
                _bench_tflops = _bench_step_flops / avg_train / 1e12
                _bench_mfu = (
                    _bench_tflops / self._gpu_peak_tflops
                    if self._gpu_peak_tflops > 0
                    else 0.0
                )
                print(f"  TFLOPs/GPU:                 {_bench_tflops:.1f}")
                print(f"  MFU:                        {_bench_mfu * 100:.1f}%")
            print(f"  Peak GPU memory allocated:  {peak_mem_gb:.2f} GB")
            print(f"  Peak GPU memory reserved:   {peak_mem_reserved_gb:.2f} GB")
            print(f"{'─' * 80}")
            print(
                f"  Config: sp_size={args.parallelism.sp_size}, "
                f"batch_size={args.dataset.video_batch_size}, "
                f"n_frames={getattr(args.dataset, 'sample_n_frames', 'variable')}, "
                f"token_packing={is_token_packed}"
            )
            print(f"{'=' * 80}\n")

        self.logger.info(
            f"[Benchmark] Done. avg_step={statistics.mean(total_times):.3f}s, "
            f"peak_mem={peak_mem_gb:.2f}GB, mode={mode_str}"
        )

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        args = self.args
        logger = self.logger
        ss = self.scalar_states

        if self.rank == 0:
            self._print_config()

        max_train_steps = int(args.optimizer.max_train_steps)
        gradient_accumulation_steps = int(args.parallelism.gradient_accumulation_steps)

        cycle_states = CycleStates()
        self.optimizer.zero_grad()
        global_start_time = _sync_cuda_time()

        _is_first_iter = True
        logger.info(f"Starting {self.training_mode} training...")
        logger.info(f"  Max train steps: {max_train_steps}")
        logger.info(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
        logger.info(f"  Resumed step: {ss.train_steps}")
        self.model.train()

        # ---- Profiler setup ----
        is_profiler = getattr(args, "is_profiler", False)
        profiler_skip_first = getattr(args, "profiler_skip_first", 3)
        profiler_wait = getattr(args, "profiler_wait", 1)
        profiler_warmup = getattr(args, "profiler_warmup", 1)
        profiler_active = getattr(args, "profiler_active", 3)
        profiler_repeat = getattr(args, "profiler_repeat", 1)
        profiler_exit_after = getattr(args, "profiler_exit_after", True)
        profiler_record_memory = getattr(args, "profiler_record_memory", False)
        profiler_memory_max_entries = getattr(
            args, "profiler_memory_max_entries", 100000
        )

        if is_profiler:
            profiler_dir = os.path.join(args.output_dir, "profiler")
            if self.rank == 0:
                os.makedirs(profiler_dir, exist_ok=True)
            dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])

            total_profiler_steps = (
                profiler_skip_first
                + (profiler_wait + profiler_warmup + profiler_active) * profiler_repeat
            )
            logger.info(
                f"[Profiler] Enabled. skip_first={profiler_skip_first}, "
                f"wait={profiler_wait}, warmup={profiler_warmup}, "
                f"active={profiler_active}, repeat={profiler_repeat}. "
                f"Total profiled steps: {total_profiler_steps}. "
                f"Traces will be saved to {profiler_dir}"
            )

            # Restrict trace export to one rank per node (LOCAL_RANK==0) to
            # avoid dumping one large trace per GPU on multi-node runs. The
            # global rank is kept in the filename so per-node traces stay
            # distinguishable on the shared output_dir.
            _local_rank = int(os.environ.get("LOCAL_RANK", 0))

            def _on_trace_ready(prof):
                """Called when a profiler trace is ready."""
                if _local_rank != 0:
                    return
                # Export gzip-compressed Chrome trace. Perfetto UI
                # (ui.perfetto.dev) opens .json.gz natively and converts it to
                # its internal protobuf representation on load. The .gz suffix
                # makes export_chrome_trace emit a binary, compressed file
                # (typically 5-10x smaller than raw .json).
                trace_path = os.path.join(
                    profiler_dir,
                    f"trace_rank{self.rank}_step{prof.step_num}.json.gz",
                )
                prof.export_chrome_trace(trace_path)
                if self.rank == 0:
                    print(f"\n{'=' * 80}")
                    print(
                        f"[Profiler] Perfetto-compatible trace (gzip) saved to {trace_path}"
                    )
                    print(f"{'=' * 80}")
                    print(
                        prof.key_averages().table(
                            sort_by="cuda_time_total",
                            row_limit=30,
                        )
                    )
                    print(f"{'=' * 80}")
                    # Also print memory summary
                    print(
                        prof.key_averages().table(
                            sort_by="self_cuda_memory_usage",
                            row_limit=15,
                        )
                    )
                    print(f"{'=' * 80}\n")

            # ---- CUDA memory snapshot (pytorch.org/memory_viz) ----
            # Complements torch.profiler's op-level memory table: records every
            # allocation with its call stack so you can see WHICH tensors / code
            # paths hold the most memory. Only LOCAL_RANK==0 dumps, mirroring the
            # per-node trace policy above.
            _mem_dumped = [False]

            def _dump_memory_snapshot():
                if not profiler_record_memory or _local_rank != 0 or _mem_dumped[0]:
                    return
                _mem_dumped[0] = True
                snap_path = os.path.join(
                    profiler_dir, f"mem_snapshot_rank{self.rank}.pickle"
                )
                # Authoritative peak counters (cheap; independent of the history
                # ring-buffer size). Reset when recording started, so these reflect
                # the profiled training window's peak.
                _peak_alloc = torch.cuda.max_memory_allocated(self.device) / 1e9
                _peak_resv = torch.cuda.max_memory_reserved(self.device) / 1e9
                logger.info(
                    f"[Profiler] Peak CUDA memory (rank{self.rank}): "
                    f"allocated={_peak_alloc:.2f} GB, reserved={_peak_resv:.2f} GB"
                )
                try:
                    torch.cuda.memory._dump_snapshot(snap_path)
                    if self.rank == 0:
                        print(
                            f"[Profiler] CUDA memory snapshot saved to {snap_path} "
                            f"-> view at https://pytorch.org/memory_viz"
                        )
                except Exception as e:
                    logger.warning(f"[Profiler] memory snapshot dump failed: {e}")
                finally:
                    torch.cuda.memory._record_memory_history(enabled=None)  # stop

            torch_profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(
                    skip_first=profiler_skip_first,
                    wait=profiler_wait,
                    warmup=profiler_warmup,
                    active=profiler_active,
                    repeat=profiler_repeat,
                ),
                on_trace_ready=_on_trace_ready,
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
            torch_profiler.__enter__()
            # Gate the START on LOCAL_RANK==0 to match _dump_memory_snapshot(),
            # which only dumps/stops on rank 0. Starting on every local rank
            # while only stopping on rank 0 would leave ranks 1..N recording
            # memory history forever (growing overhead), which is especially
            # bad when profiler_exit_after=false and training continues.
            if profiler_record_memory and _local_rank == 0:
                torch.cuda.memory._record_memory_history(
                    max_entries=profiler_memory_max_entries
                )
                # Reset peak counters so the peak logged at dump time reflects the
                # profiled training window (not model-load transients).
                torch.cuda.reset_peak_memory_stats(self.device)
                logger.info(
                    f"[Profiler] CUDA memory history recording ON "
                    f"(max_entries={profiler_memory_max_entries})"
                )
        else:
            torch_profiler = None

        # ---- Benchmark (wall-clock) setup ----
        is_benchmark = getattr(args, "is_benchmark", False)
        benchmark_warmup_steps = getattr(args, "benchmark_warmup_steps", 3)
        benchmark_active_steps = getattr(args, "benchmark_active_steps", 5)
        benchmark_total_steps = benchmark_warmup_steps + benchmark_active_steps
        benchmark_step_times = []  # (data_load_time, train_step_time, total_time)
        if is_benchmark:
            logger.info(
                f"[Benchmark] Enabled. warmup={benchmark_warmup_steps}, "
                f"active={benchmark_active_steps}, total={benchmark_total_steps} steps. "
                f"Will print wall-clock summary and exit."
            )
            # Allocate peak memory tracker
            torch.cuda.reset_peak_memory_stats(self.device)
            max_train_steps = min(max_train_steps, benchmark_total_steps)

        if ss.train_steps >= max_train_steps and self.rank == 0:
            logger.warning(
                "No training iterations will run: train_steps=%s >= max_train_steps=%s. "
                "After resume, raise max_train_steps above the resumed step count.",
                ss.train_steps,
                max_train_steps,
            )

        # Deferred RNG-state restore for resumed runs.  This must happen AFTER
        # _create_dataloader / _init_trackers (called in __init__) because
        # third-party libraries invoked there (notably swanlab.init) consume
        # the global Python random module.  Restoring before those calls would
        # let those side-effects advance the just-restored state and break
        # bitwise resume reproducibility.  See `_maybe_resume` for context.
        _pending_rng_resume_path = getattr(self, "_pending_rng_resume_path", None)
        if _pending_rng_resume_path:
            _t_rng = time.time()
            load_rng_states(_pending_rng_resume_path, self.rank, logger=logger)
            logger.info(
                f"[INIT_TIMER]   rng states loaded (deferred) in {time.time() - _t_rng:.1f}s"
            )
            self._pending_rng_resume_path = None
        _pending_rng_resume_state = getattr(self, "_pending_rng_resume_state", None)
        if _pending_rng_resume_state is not None:
            restore_rng_state(_pending_rng_resume_state, logger=logger)
            self._pending_rng_resume_state = None

        while ss.train_steps < max_train_steps:
            if self.finetune_mode == "lora" and self._enable_dcp_checkpoint:
                poll_pending_lora_checkpoint()
            start_time = _sync_cuda_time()

            with (
                record_function("data_loading")
                if torch_profiler
                else contextlib.nullcontext()
            ):
                batch = next(self.data_iter)

                ss.consumed_samples_per_dp["video"] = (
                    self._loader.scalar_states.consumed_samples_per_dp["video"]
                )
                ss.consumed_groups_per_dp["video"] = (
                    self._loader.scalar_states.consumed_groups_per_dp["video"]
                )
                ss.epoch_consumed_samples["video"] = (
                    self._loader.scalar_states.consumed_samples_per_dp["video"]
                )
                ss.epoch["video"] = self._loader.scalar_states.epoch["video"]

                batch = sync_object_for_parallel_training(
                    batch, parallel_dims=self.parallel_dims
                )

            dataloader_time_ = _sync_cuda_time()

            with (
                record_function("forward_backward_optim")
                if torch_profiler
                else contextlib.nullcontext()
            ):
                loss, loss_dict_or_terms, grad_norm = self.transport.forward_step(
                    args=args,
                    model=self.model,
                    vae=self.vae,
                    text_encoder=self.text_encoder,
                    optimizer=self.optimizer,
                    lr_scheduler=self.lr_scheduler,
                    batch=batch,
                    device=self.device,
                    global_step=ss.train_steps,
                    scalar_states=ss,
                    logger=logger,
                    parallel_dims=self.parallel_dims,
                )

            forward_time_ = _sync_cuda_time()
            data_load_time = dataloader_time_ - start_time
            train_step_time = forward_time_ - dataloader_time_

            total_step_time = forward_time_ - start_time

            if _is_first_iter:
                _is_first_iter = False
                logger.info(
                    f"[STEP_TIMER] first step (step={ss.train_steps + 1}) breakdown: "
                    f"data_load={data_load_time:.2f}s fwd+bwd+optim={train_step_time:.2f}s "
                    f"total={total_step_time:.2f}s"
                )
                # ------------------------------------------------------------------
                # Debug: dump the batch prompts on the very first training iteration.
                # RE-ENABLED — the earlier RuntimeError was caused by the default
                # DataLoader collate wrapping the per-sample list[str] one level
                # deeper as list[list[str]]; the unwrap guard below (~30 lines
                # down) now strips that redundant nesting before validating, so
                # this rank-0 dump no longer crashes the run. Rank-0 only to avoid
                # log flooding; also writes output_dir/debug/first_iter_prompts_*.txt.
                # ------------------------------------------------------------------
                if self.rank == 0:
                    # Goal: dump the **exact strings fed into the text
                    # encoder** for the first iteration. The dataset side
                    # writes that list under ``kwargs["text_encoder_input"]``
                    # (a list[str], one entry per segment, already AFTER
                    # CFG dropout). This is the ONLY source we trust here --
                    # any other field (``text``, top-level ``prompt`` ...)
                    # is a pre-dropout display string and would silently
                    # disagree with what the tokenizer actually saw, so we
                    # fail loudly instead of falling back.
                    def _extract_kwargs_list(_batch):
                        """Return list[kwargs_dict] across batch shapes."""
                        if isinstance(_batch, dict):
                            kl = _batch.get("kwargs_list")
                            if isinstance(kl, (list, tuple)):
                                return list(kl)
                        if isinstance(_batch, (list, tuple)) and len(_batch) > 0:
                            _kw = _batch[-1]
                            if isinstance(_kw, dict):
                                return [_kw]
                        raise RuntimeError(
                            f"[FIRST_ITER_PROMPT] unsupported batch shape: "
                            f"type={type(_batch).__name__}, "
                            f"keys={list(_batch.keys()) if isinstance(_batch, dict) else 'n/a'}"
                        )

                    _kw_list = _extract_kwargs_list(batch)
                    _prompts = []
                    for _i, _kw in enumerate(_kw_list):
                        if not isinstance(_kw, dict):
                            raise RuntimeError(
                                f"[FIRST_ITER_PROMPT] kwargs_list[{_i}] is not a dict: "
                                f"got {type(_kw).__name__}"
                            )
                        if "text_encoder_input" not in _kw:
                            raise RuntimeError(
                                f"[FIRST_ITER_PROMPT] kwargs_list[{_i}] missing "
                                f"'text_encoder_input' key. Available keys: {list(_kw.keys())}. "
                                f"This field must be emitted by the dataset (see "
                                f"WM1MTrainingDataset.__getitem__)."
                            )
                        _val = _kw["text_encoder_input"]
                        # The default DataLoader collate (used on the
                        # non-token-packing path) transposes list fields, so a
                        # per-sample list[str] arrives here wrapped one level
                        # deeper. Two shapes occur:
                        #   single-segment : [["prompt"]]
                        #   multi-segment  : [("seg0",), ("seg1",)]   <- transposed
                        # The real training forward already tolerates both (see
                        # the data_source unwrap at ~line 1905); this debug-only
                        # dump must too. Unwrap instead of crashing the run.
                        if (
                            isinstance(_val, (list, tuple))
                            and len(_val) == 1
                            and isinstance(_val[0], (list, tuple))
                        ):
                            _val = _val[0]
                        if (
                            isinstance(_val, (list, tuple))
                            and _val
                            and all(
                                isinstance(s, (list, tuple))
                                and len(s) == 1
                                and isinstance(s[0], str)
                                for s in _val
                            )
                        ):
                            # Transposed per-segment collate: one 1-element
                            # container per segment -> flatten to list[str].
                            _val = [s[0] for s in _val]
                        if not isinstance(_val, (list, tuple)) or not all(
                            isinstance(s, str) for s in _val
                        ):
                            raise RuntimeError(
                                f"[FIRST_ITER_PROMPT] kwargs_list[{_i}]['text_encoder_input'] "
                                f"must be list[str], got {type(_val).__name__}: {_val!r}"
                            )
                        _prompts.append(list(_val))

                    _src = "kwargs.text_encoder_input (actual text encoder input)"
                    logger.info(f"[FIRST_ITER_PROMPT] prompt source: {_src}")
                    logger.info(
                        f"[FIRST_ITER_PROMPT] step={ss.train_steps} batch_size={len(_prompts)}"
                    )

                    _MAX_CHARS = 2000
                    for _i, _segs in enumerate(_prompts):
                        _joined = "\n---\n".join(_segs)
                        _truncated = (
                            _joined
                            if len(_joined) <= _MAX_CHARS
                            else _joined[:_MAX_CHARS]
                            + f"... [+{len(_joined) - _MAX_CHARS} chars truncated]"
                        )
                        logger.info(
                            f"[FIRST_ITER_PROMPT][sample {_i}] "
                            f"n_segments={len(_segs)} total_chars={len(_joined)}\n{_truncated}"
                        )
                    # Persist the full per-segment input to a file. Each
                    # segment is written on its own line so the dump
                    # matches exactly what the text encoder saw.
                    _dump_dir = os.path.join(args.output_dir, "debug")
                    os.makedirs(_dump_dir, exist_ok=True)
                    _dump_path = os.path.join(
                        _dump_dir, f"first_iter_prompts_step{ss.train_steps}.txt"
                    )
                    with open(_dump_path, "w", encoding="utf-8") as _f:
                        _f.write(f"# source: {_src}\n")
                        _f.write(f"# batch_size: {len(_prompts)}\n\n")
                        for _i, _segs in enumerate(_prompts):
                            _f.write(
                                f"===== sample {_i} (n_segments={len(_segs)}) =====\n"
                            )
                            for _si, _seg in enumerate(_segs):
                                _f.write(f"----- segment {_si} -----\n")
                                _f.write(_seg)
                                _f.write("\n")
                            _f.write("\n")
                    logger.info(
                        f"[FIRST_ITER_PROMPT] full prompts saved to {_dump_path}"
                    )

                    # Persist the exact sparse attention payload/ranges assembled
                    # by the transport for this same batch. MagiAttention consumes
                    # q/k ranges directly; dumping those is lossless and avoids an
                    # enormous dense [video+text, video+text] boolean matrix.
                    _attn_debug = (
                        batch.get("_first_iter_attention_debug")
                        if isinstance(batch, dict)
                        else None
                    )
                    if _attn_debug is None:
                        _attn_debug = {
                            "schema_version": 1,
                            "global_step": int(ss.train_steps),
                            "status": "not_available",
                            "reason": (
                                "transport did not expose a first-iteration "
                                "attention snapshot (non-packed/non-Magi path)"
                            ),
                        }
                    _attn_dump_path = os.path.join(
                        _dump_dir,
                        f"first_iter_attention_step{ss.train_steps}.json",
                    )
                    with open(_attn_dump_path, "w", encoding="utf-8") as _f:
                        json.dump(_attn_debug, _f, ensure_ascii=False, indent=2)
                    logger.info(
                        f"[FIRST_ITER_ATTENTION] sparse mask params/ranges saved "
                        f"to {_attn_dump_path}"
                    )

            # ---- Benchmark: collect timing ----
            if is_benchmark and ss.train_steps >= benchmark_warmup_steps:
                benchmark_step_times.append(
                    (data_load_time, train_step_time, total_step_time)
                )

            # Store batch info for _log_metrics (packing details, data_source, timing)
            self._last_batch = batch
            self._last_data_load_time = data_load_time
            self._last_train_step_time = train_step_time

            # Accumulate FLOPs for this step
            _step_flops, _ = self._estimate_step_flops_from_batch(
                batch, global_step=ss.train_steps
            )
            if _step_flops is not None:
                self._cumulative_flops = (
                    getattr(self, "_cumulative_flops", 0) + _step_flops
                )

            ss.add(train_steps=1)
            is_update_step = ss.train_steps % gradient_accumulation_steps == 0

            if is_update_step:
                ss.add(update_steps=1)
                ss.lr = self.optimizer.param_groups[0]["lr"]
                if self.ema_rates:
                    _gn_val = (
                        float(grad_norm.item())
                        if hasattr(grad_norm, "item")
                        else float(grad_norm)
                    )
                    if math.isfinite(_gn_val) and _gn_val > 0:
                        if self.ema_schedule:
                            _current_rate = self._get_ema_rate_for_step(ss.update_steps)
                            update_ema_with_rate(
                                self.ema_states,
                                self.model,
                                self._ema_canonical_rate,
                                _current_rate,
                            )
                        else:
                            update_ema(self.ema_states, self.model, self.ema_rates)
                    else:
                        logger.warning(
                            f"Skipping EMA update due to non-finite grad_norm={_gn_val}"
                        )

            self.scalar_states = sync_object_for_parallel_training(
                ss, parallel_dims=self.parallel_dims, force_object=True
            )
            ss = self.scalar_states

            cycle_states.add(log_steps=1, running_loss=loss)

            # Accumulate per-t-bin loss
            if isinstance(loss_dict_or_terms, dict) and "t" in loss_dict_or_terms:
                _t_val = loss_dict_or_terms["t"]

                # 统一处理逻辑：统计每个bin中的样本数量，不取平均值
                _t_bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]

                # ----- Preferred path: exact per-token loss & per-token t -----
                # Exposed by the token-packed forward (unified.py).  Lets us
                # compute the *true* mean loss inside each t-bin instead of
                # just step-mean weighted by bin hit count.
                _ptl = loss_dict_or_terms.get("per_token_loss", None)
                _ptt = loss_dict_or_terms.get("t_per_token", None)
                _have_per_token = (
                    isinstance(_ptl, torch.Tensor)
                    and isinstance(_ptt, torch.Tensor)
                    and _ptl.numel() == _ptt.numel()
                    and _ptl.numel() > 0
                )

                if _have_per_token:
                    _ptl_f = _ptl.detach().float().flatten()
                    _ptt_f = _ptt.detach().float().flatten()
                    # Optional keep-mask (exclude masked-out tokens from both
                    # the numerator and denominator, matching how the actual
                    # training loss was computed).
                    _keep = loss_dict_or_terms.get("per_token_keep_mask", None)
                    if (
                        isinstance(_keep, torch.Tensor)
                        and _keep.numel() == _ptl_f.numel()
                    ):
                        _keep_b = _keep.detach().bool().flatten()
                        _ptl_f = _ptl_f[_keep_b]
                        _ptt_f = _ptt_f[_keep_b]

                    if _ptl_f.numel() > 0:
                        # Compute (sum, count) per bin on-device, stack, and
                        # do ONE GPU->CPU sync at the end to keep step latency
                        # flat (vs. 5x .item() inside the loop).
                        _sum_list = []
                        _cnt_list = []
                        for _bi, (_lo, _hi) in enumerate(_t_bins):
                            if _bi < len(_t_bins) - 1:
                                _mask = (_ptt_f >= _lo) & (_ptt_f < _hi)
                            else:
                                _mask = _ptt_f >= _lo
                            _sum_list.append((_ptl_f * _mask.float()).sum())
                            _cnt_list.append(_mask.sum())
                        _sums_cpu = torch.stack(_sum_list).detach().cpu().tolist()
                        _cnts_cpu = torch.stack(_cnt_list).detach().cpu().tolist()
                        for _bi, (_lo, _hi) in enumerate(_t_bins):
                            _cnt = int(_cnts_cpu[_bi])
                            if _cnt > 0:
                                _bin_key = f"{_lo:.1f}_{_hi:.1f}"
                                cycle_states.running_loss_by_t_bin[_bin_key] += float(
                                    _sums_cpu[_bi]
                                )
                                cycle_states.running_count_by_t_bin[_bin_key] += _cnt
                # ----- Fallback: only step-mean loss + t tensor available -----
                elif hasattr(_t_val, "item") and _t_val.dim() == 0:
                    # 标量t值：当作单个样本处理
                    _t_scalar = _t_val.item()
                    for _bi, (_lo, _hi) in enumerate(_t_bins):
                        if _lo <= _t_scalar < _hi or (
                            _bi == len(_t_bins) - 1 and _t_scalar >= _lo
                        ):
                            _bin_key = f"{_lo:.1f}_{_hi:.1f}"
                            cycle_states.running_loss_by_t_bin[_bin_key] += loss
                            cycle_states.running_count_by_t_bin[_bin_key] += 1
                            break
                else:
                    # 张量t值：没有 per-token loss，只能用 step-mean 当作每个
                    # 命中样本的代表 loss 累加 loss*bin_count，
                    # _avg = sum_loss/sum_count 是命中频次加权的 step-mean，
                    # 反映 t-bin 上长期 loss 水平（近似，非精确）。
                    _t_flat = _t_val.flatten()

                    for _bi, (_lo, _hi) in enumerate(_t_bins):
                        # 计算当前bin中的样本数
                        if _bi < len(_t_bins) - 1:
                            _in_bin_mask = (_t_flat >= _lo) & (_t_flat < _hi)
                        else:
                            _in_bin_mask = _t_flat >= _lo

                        _bin_count = _in_bin_mask.sum().item()
                        if _bin_count > 0:
                            _bin_key = f"{_lo:.1f}_{_hi:.1f}"
                            cycle_states.running_loss_by_t_bin[_bin_key] += (
                                loss * _bin_count
                            )
                            cycle_states.running_count_by_t_bin[_bin_key] += _bin_count

            log_interval = getattr(args, "log_interval", 10)

            if is_update_step and ss.update_steps % log_interval == 0:
                end_time = _sync_cuda_time()
                sec_per_step = (end_time - global_start_time) / cycle_states.log_steps
                steps_per_sec = cycle_states.log_steps / (end_time - global_start_time)

                if self.rank == 0:
                    self._log_metrics(
                        loss,
                        grad_norm,
                        loss_dict_or_terms,
                        sec_per_step,
                        steps_per_sec,
                        batch=getattr(self, "_last_batch", None),
                        data_load_time=getattr(self, "_last_data_load_time", None),
                        train_step_time=getattr(self, "_last_train_step_time", None),
                    )

                    # Log per-t-bin average losses
                    if cycle_states.running_count_by_t_bin:
                        for (
                            _bin_key,
                            _bin_loss,
                        ) in cycle_states.running_loss_by_t_bin.items():
                            _bin_count = cycle_states.running_count_by_t_bin[_bin_key]
                            if _bin_count > 0:
                                _avg = _bin_loss / _bin_count
                                if self.tb_writer is not None:
                                    try:
                                        self.tb_writer.add_scalar(
                                            f"train/loss_t_{_bin_key}",
                                            _avg,
                                            ss.update_steps,
                                        )
                                    except OSError:
                                        pass
                                if self.swanlab_active:
                                    swanlab.log(
                                        {f"train/loss_t_{_bin_key}": _avg},
                                        step=ss.update_steps,
                                    )

                cycle_states.reset_epoch_based_states()
                global_start_time = _sync_cuda_time()

            checkpointing_steps = getattr(args, "checkpointing_steps", 500)
            if (
                is_update_step
                and ss.update_steps % checkpointing_steps == 0
                and ss.update_steps > 0
            ):
                logger.info(f"--> save checkpoint at update_step {ss.update_steps}")
                self._save_checkpoint()
                if _GC_DISABLE:
                    _gc_t0 = time.time()
                    gc.collect()
                    if self.rank == 0:
                        logger.info(
                            f"[GC] manual gc.collect after checkpoint in {time.time() - _gc_t0:.2f}s"
                        )
                dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])

            # ---- Profiler step ----
            if torch_profiler is not None:
                torch_profiler.step()
                # Exit after profiling is done to avoid unnecessary training
                if profiler_exit_after and ss.train_steps >= total_profiler_steps:
                    logger.info(
                        f"[Profiler] Profiling complete after {ss.train_steps} steps. "
                        f"Traces saved to {profiler_dir}. Exiting."
                    )
                    torch_profiler.__exit__(None, None, None)
                    _dump_memory_snapshot()
                    if self.tb_writer:
                        try:
                            self.tb_writer.close()
                        except OSError:
                            pass
                    dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])
                    sys.exit(0)

            # ---- Benchmark: print summary and exit ----
            if is_benchmark and ss.train_steps >= benchmark_total_steps:
                self._print_benchmark_summary(
                    benchmark_step_times,
                    benchmark_warmup_steps,
                    benchmark_active_steps,
                    args,
                )
                dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])
                sys.exit(0)

        # ---- Cleanup profiler if still active ----
        if torch_profiler is not None:
            torch_profiler.__exit__(None, None, None)
            _dump_memory_snapshot()

        logger.info(f"--> save final checkpoint at update_step {ss.update_steps}")
        self._save_checkpoint()
        if self.finetune_mode == "lora" and self._enable_dcp_checkpoint:
            shutdown_lora_checkpoint_writer()

        if self.tb_writer:
            self.tb_writer.close()
        if self.swanlab_active:
            try:
                swanlab.finish()
            except Exception:
                pass
        if get_sequence_parallel_state():
            destroy_sequence_parallel_group()
        logger.info("Done!")


def main():
    args = parse_args()
    trainer = None

    # ---- Register SIGTERM handler so we can log before being killed ----
    import signal

    def _sigterm_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        msg = f"Training terminated by signal {sig_name} (signum={signum})"
        if (
            trainer is not None
            and hasattr(trainer, "logger")
            and trainer.logger is not None
        ):
            trainer.logger.error(msg)
        else:
            from loguru import logger as _root_logger

            _root_logger.error(msg)
        # Flush loguru sinks before exiting
        from loguru import logger as _flush_logger

        _flush_logger.complete()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    try:
        trainer = DefaultTrainer(args)
        trainer.train()
    except BaseException:
        # Use BaseException to also catch KeyboardInterrupt / SystemExit,
        # which are the common ways distributed training gets terminated.
        import traceback

        tb_str = traceback.format_exc()
        if (
            trainer is not None
            and hasattr(trainer, "logger")
            and trainer.logger is not None
        ):
            trainer.logger.error(f"Training crashed with exception:\n{tb_str}")
        else:
            # Logger not yet initialised — fall back to loguru root + stderr.
            from loguru import logger as _root_logger

            _root_logger.error(f"Training crashed during init:\n{tb_str}")
        raise
    finally:
        # Always flush loguru sinks, even on normal exit
        from loguru import logger as _flush_logger

        _flush_logger.complete()


if __name__ == "__main__":
    main()
