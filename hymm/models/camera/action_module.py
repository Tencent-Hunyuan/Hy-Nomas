import contextlib
from typing import Any, List, Tuple, Optional, Union, Dict

from flash_attn import flash_attn_func
from loguru import logger
import torch
import torch.nn as nn
from torch.profiler import record_function as _record_function
from ..dit.modules.posemb_layers import apply_rotary_emb, get_nd_rotary_pos_embed
from ..dit.modules.norm_layers import get_norm_layer
from hymm.distributed.parallel_states import get_sequence_parallel_state, nccl_info


def _prof(name):
    """record_function span that is a no-op unless a profiler is active."""
    try:
        active = torch.autograd._profiler_enabled()
    except Exception:
        active = False
    return _record_function(name) if active else contextlib.nullcontext()


# Optional: per-segment block-diagonal cross-attention via MagiAttention's
# flex_flash_attn_func. The legacy flash_attn_func path stays as the default
# fallback when ``per_segment_camera_attn`` is off or segment boundaries are
# not provided at runtime.
from ..dit.modules.attention.backends import magi_available
from ..dit.modules.attention.range_attention import (
    range_attention,
    flex_camera_range_attention,
)

_MAGI_ATTENTION_AVAILABLE = magi_available()


def _torch_chunk_local_start(total_tokens: int, sp_size: int, sp_rank: int) -> int:
    """Global start index of SP rank ``sp_rank`` under ``torch.chunk`` semantics.

    The image sequence is sharded across SP ranks with
    ``torch.chunk(img, sp_size, dim=1)`` in ``nomas_cpt.forward`` /
    ``forward_with_cache``. ``torch.chunk`` splits a length-``L`` axis into
    chunks of ``split_size = ceil(L / sp_size)``; rank ``r`` therefore starts at
    ``r * split_size`` (clamped to ``L``), and the *last* non-empty rank absorbs
    the smaller remainder. This differs from the "balanced" distribution
    (``r * (L // sp_size) + min(r, L % sp_size)``) whenever
    ``0 < L % sp_size < sp_size - 1``; the camera cross-attention keeps Q local
    (no all-to-all), so it must reproduce the exact same offset or later ranks
    mis-map their tokens to keyboard frames / segments. Keeping these in sync is
    the whole point of this helper.
    """
    if sp_size <= 1:
        return 0
    split_size = (total_tokens + sp_size - 1) // sp_size  # ceil(total / sp_size)
    return min(sp_rank * split_size, total_tokens)


# ---------------------------------------------------------------------------
# Range-tensor cache for per-segment cross-attention
# ---------------------------------------------------------------------------
#
# ``_blockwise_cross_attn`` would otherwise rebuild q_ranges / k_ranges /
# attn_type_map at every layer (54 ActionModule blocks × every step), each
# call paying for a Python list-of-lists → ``torch.tensor`` host→device copy
# plus a small allocation. Since the inputs to range computation are
# step-stable (same prompt_video_boundaries, same SP rank, same
# tt / th*tw / N_frames within one block stage), we cache the resulting
# tensors keyed by their inputs and reuse across layers.
#
# Cache size is bounded in practice by the number of distinct
# (boundaries, local_q_window, S, N_frames, device) tuples seen during a run.
# For typical multi-prompt training (a few hundred unique boundary patterns)
# the cache plateaus at <1MB. Mirrors ``_magi_mask_cache`` in
# ``hymm/models/dit/modules/attention/magi_attn.py``.
_segmented_attn_range_cache: Dict[
    str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
] = {}

# Packed + per-segment (multi-prompt) camera cross-attention via flex_flash_attn_func.
_packed_seg_flex_range_cache: Dict[
    str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
] = {}


def _coerce_packed_seg_latent_list(raw: Optional[Any]) -> Optional[List[int]]:
    """Normalize one sample's ``prompt_video_boundaries_latent`` for flex cache keys.

    Accepts ``None``, a flat list/tuple of int-like values, a nested ``[[...]]``
    layout (same as ``unified._normalize_pvb_latent_for_packed_camera``), or a
    1-D tensor.  Returns ``None`` if absent or empty so downstream uses a single
    segment ``[0, T_lat]``.
    """
    if raw is None:
        return None
    if torch.is_tensor(raw):
        raw = raw.detach().cpu().squeeze().tolist()
    if not isinstance(raw, (list, tuple)) or len(raw) == 0:
        return None
    if isinstance(raw[0], (list, tuple)):
        raw = raw[0]
    out: List[int] = []
    for x in raw:
        if torch.is_tensor(x):
            x = x.detach().cpu().item()
        out.append(int(x))
    return out if out else None


def _normalize_sample_latent_boundaries(
    raw_b: Optional[List[int]],
    T_lat: int,
) -> List[int]:
    """Monotonic latent-frame boundaries: start at 0, end at ``T_lat``."""
    if T_lat <= 0:
        return [0, 0]
    if raw_b is None or len(raw_b) < 2:
        return [0, T_lat]
    boundaries = [max(0, min(int(x), T_lat)) for x in raw_b]
    boundaries = sorted(set(boundaries))
    if not boundaries or boundaries[0] != 0:
        boundaries = [0] + [b for b in boundaries if b > 0]
        boundaries = sorted(set(boundaries))
    fixed: List[int] = [boundaries[0]]
    for b in boundaries[1:]:
        if b > fixed[-1]:
            fixed.append(b)
    if fixed[-1] != T_lat:
        fixed.append(T_lat)
    return fixed


def _segment_aware_chunk_boundaries(
    prompt_boundaries: Optional[List[int]],
    T_lat: int,
    chunk_size: int,
) -> List[int]:
    """Segment-aware fixed-size chunk boundaries (latent-frame units).

    Each prompt segment ``[a, b)`` (from ``prompt_boundaries``) is independently
    subdivided into chunks of ``chunk_size`` latent frames; the result is the
    sorted union of all segment boundaries and the intra-segment chunk cuts, so
    **no chunk ever crosses a prompt-segment boundary**. This keeps per-chunk
    camera cross-attention finer-grained while preserving segment isolation —
    critical when per-segment camera dropout is active, otherwise a chunk that
    straddles a segment boundary would leak a dropped (camera-OFF) segment's
    keyboard into a neighbouring kept (camera-ON) segment.

    When ``prompt_boundaries`` is missing this degenerates to a single segment
    ``[0, T_lat]`` → plain fixed-size chunks ``[0, cs, 2*cs, ..., T_lat]``.
    The trailing chunk of each segment may be smaller than ``chunk_size``.
    """
    segs = _normalize_sample_latent_boundaries(prompt_boundaries, T_lat)
    if chunk_size <= 0:
        return segs
    out: List[int] = [segs[0]]
    for s in range(len(segs) - 1):
        a, b = segs[s], segs[s + 1]
        c = a + chunk_size
        while c < b:
            out.append(c)
            c += chunk_size
        out.append(b)
    return out


def _build_packed_segment_flex_ranges(
    *,
    packed_img_token_counts: List[int],
    packed_kb_token_counts: List[int],
    packed_sample_shapes: List[Tuple[int, int, int]],
    packed_has_camera: List[bool],
    packed_segment_boundaries_latent: Optional[List[Optional[List[int]]]],
    local_start: int,
    local_end: int,
    patch_size: Tuple[int, int, int],
    device: torch.device,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Build (q_ranges, k_ranges, attn_type_map) for packed + per-segment camera cross-attn.

    q_ranges use **local** Q indices (0 .. TS_local-1) like ``_get_cached_segment_attn_ranges``.
    k_ranges use **global** indices along the concatenated keyboard tensor (all packed samples).

    **Indexing assumption:** for each segment, ``k0_l`` / ``k1_l`` are taken from the same
    latent-frame boundaries as image tokens (clamped to ``K_i``). This matches one
    keyboard slot per latent frame; it can truncate when ``K_i`` differs from ``T_lat``
    (e.g. subsampled extrinsics with ``K_i > T_lat``). Callers that need a different
    Q/K alignment should use the flash fallback or extend the mapping here.

    Samples with ``K_i == 0`` are skipped (no keyboard slice). Rows with ``has_camera``
    False but non-zero ``K_i`` still build ranges so cross-attn runs on the provided
    tensors (e.g. zeros) for gradient flow.
    """
    pt, ph, pw = int(patch_size[0]), int(patch_size[1]), int(patch_size[2])
    q_list: List[List[int]] = []
    k_list: List[List[int]] = []

    img_cum = 0
    kb_cum = 0
    n = len(packed_img_token_counts)
    seg_lists = packed_segment_boundaries_latent or [None] * n
    assert len(packed_has_camera) == n, (
        f"packed_has_camera len {len(packed_has_camera)} != n_pack {n}"
    )

    for i in range(n):
        L_i = packed_img_token_counts[i]
        K_i = packed_kb_token_counts[i]
        T_i, H_i, W_i = packed_sample_shapes[i]
        S_i = max((H_i // ph) * (W_i // pw), 1)
        T_lat = max(T_i // pt, 1)

        if K_i <= 0:
            img_cum += L_i
            kb_cum += K_i
            continue

        raw_b: Optional[List[int]] = None
        if i < len(seg_lists):
            raw_b = seg_lists[i]
        boundaries = _normalize_sample_latent_boundaries(raw_b, T_lat)

        for j in range(len(boundaries) - 1):
            g_q0 = img_cum + boundaries[j] * S_i
            g_q1 = img_cum + boundaries[j + 1] * S_i
            k0_l = max(0, min(boundaries[j], K_i))
            k1_l = max(0, min(boundaries[j + 1], K_i))
            g_k0 = kb_cum + k0_l
            g_k1 = kb_cum + k1_l

            local_q_start = max(g_q0, local_start) - local_start
            local_q_end = min(g_q1, local_end) - local_start
            if local_q_start >= local_q_end:
                continue
            if k0_l >= k1_l:
                continue
            q_list.append([local_q_start, local_q_end])
            k_list.append([g_k0, g_k1])

        img_cum += L_i
        kb_cum += K_i

    if not q_list:
        return None
    q_ranges = torch.tensor(q_list, dtype=torch.int32, device=device)
    k_ranges = torch.tensor(k_list, dtype=torch.int32, device=device)
    attn_type_map = torch.zeros(len(q_list), dtype=torch.int32, device=device)
    return q_ranges, k_ranges, attn_type_map


def _get_cached_packed_segment_flex_ranges(
    *,
    packed_img_token_counts: Tuple[int, ...],
    packed_kb_token_counts: Tuple[int, ...],
    packed_sample_shapes: Tuple[Tuple[int, int, int], ...],
    packed_has_camera: Tuple[bool, ...],
    packed_seg_boundaries: Tuple[Tuple[int, ...], ...],
    local_start: int,
    local_end: int,
    patch_size: Tuple[int, int, int],
    device: torch.device,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    cache_key = (
        f"pkcam_{local_start}_{local_end}_"
        f"img{','.join(str(x) for x in packed_img_token_counts)}_"
        f"kb{','.join(str(x) for x in packed_kb_token_counts)}_"
        f"sh{'_'.join(f'{a}x{b}x{c}' for a, b, c in packed_sample_shapes)}_"
        f"hc{''.join('1' if x else '0' for x in packed_has_camera)}_"
        f"seg{'|'.join(','.join(str(b) for b in tup) for tup in packed_seg_boundaries)}_"
        f"pt{patch_size[0]}_{patch_size[1]}_{patch_size[2]}_"
        f"{device}"
    )
    cached = _packed_seg_flex_range_cache.get(cache_key)
    if cached is not None:
        return cached
    built = _build_packed_segment_flex_ranges(
        packed_img_token_counts=list(packed_img_token_counts),
        packed_kb_token_counts=list(packed_kb_token_counts),
        packed_sample_shapes=list(packed_sample_shapes),
        packed_has_camera=list(packed_has_camera),
        packed_segment_boundaries_latent=[
            (list(t) if len(t) > 0 else None) for t in packed_seg_boundaries
        ],
        local_start=local_start,
        local_end=local_end,
        patch_size=patch_size,
        device=device,
    )
    if built is not None:
        _packed_seg_flex_range_cache[cache_key] = built
    return built


def _get_cached_segment_attn_ranges(
    boundaries: Tuple[int, ...],
    local_start: int,
    local_end: int,
    tokens_per_frame: int,
    n_frames: int,
    device: torch.device,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return cached (q_ranges, k_ranges, attn_type_map) for per-segment
    block-diagonal cross-attention, building them on first miss.

    Returns ``None`` when the local rank holds no Q tokens for any segment
    (caller should treat this as "skip the cross-attn for this rank/sample").
    """
    cache_key = (
        f"cam_seg_{'_'.join(str(b) for b in boundaries)}"
        f"_ls{local_start}_le{local_end}"
        f"_tpf{tokens_per_frame}_N{n_frames}_dev{device}"
    )
    cached = _segmented_attn_range_cache.get(cache_key)
    if cached is not None:
        return cached

    q_list: List[List[int]] = []
    k_list: List[List[int]] = []
    for j in range(len(boundaries) - 1):
        seg_q_start = boundaries[j] * tokens_per_frame
        seg_q_end = boundaries[j + 1] * tokens_per_frame
        local_q_start = max(seg_q_start, local_start) - local_start
        local_q_end = min(seg_q_end, local_end) - local_start
        if local_q_start >= local_q_end:
            continue
        # Clamp K range to [0, n_frames] to be safe against off-by-one at the
        # tail (front-padded ÷4 reshape can shift boundaries by 1 latent frame).
        k_start = max(0, min(boundaries[j], n_frames))
        k_end = max(0, min(boundaries[j + 1], n_frames))
        if k_start >= k_end:
            continue
        q_list.append([local_q_start, local_q_end])
        k_list.append([k_start, k_end])

    if not q_list:
        # Don't cache the empty case — it's defensive / shouldn't normally
        # happen. Caller will return zeros.
        return None

    q_ranges = torch.tensor(q_list, dtype=torch.int32, device=device)
    k_ranges = torch.tensor(k_list, dtype=torch.int32, device=device)
    attn_type_map = torch.zeros(len(q_list), dtype=torch.int32, device=device)
    triple = (q_ranges, k_ranges, attn_type_map)
    _segmented_attn_range_cache[cache_key] = triple
    return triple


class ActionModule(nn.Module):
    """
    Action module from https://arxiv.org/pdf/2501.08325

    The keyboard control signal input is a L*D vector. By default each image
    token attends to **all** keyboard tokens (legacy ``flash_attn_func`` path).
    When ``per_segment_camera_attn=True`` AND ``segment_boundaries_latent`` is
    passed at forward time, the cross-attention switches to a per-segment
    block-diagonal pattern via MagiAttention's ``flex_flash_attn_func``:

        For segment j with latent-frame boundaries [vb[j], vb[j+1)):
          Q range  = image tokens     [vb[j]*th*tw, vb[j+1]*th*tw)
          K range  = keyboard tokens  [vb[j],       vb[j+1])
          attn_type = 0 (full softmax within block, no causality)

    Image tokens of segment j only attend to keyboard tokens of the same
    segment. This matches the multi-prompt segment design — each prompt and
    its corresponding camera trajectory are scoped to the same temporal
    range.

    With **NaViT token packing** (flat latent concat), when ``per_segment_camera_attn``
    is enabled and ``magi_attention`` is available, cross-attention uses the same
    ``flex_flash_attn_func`` primitive with **per-sample global Q/K offsets** so
    that (i) packed samples do not cross-attend each other, and (ii) within each
    sample, segment j image tokens only attend to segment j keyboard tokens.
    """

    def __init__(
        self,
        keyboard_dim_in: int = 8,
        hidden_size: int = 128,
        img_hidden_size: int = 2048,
        keyboard_hidden_dim: int = 1024,
        heads_num: int = 16,
        patch_size: list = [1, 1, 1],
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qkv_bias: bool = False,
        rope_dim_list: list = [8, 28, 28],
        rope_theta: int = 256,
        per_segment_camera_attn: bool = False,
        per_chunk_camera_attn: bool = False,
        camera_chunk_size: int = 5,
        seg_ar_attention_backend: str = "magi",
    ):
        super().__init__()

        self.rope_dim_list = rope_dim_list
        self.rope_theta = rope_theta
        self.heads_num = heads_num
        self.patch_size = patch_size
        if seg_ar_attention_backend not in ("magi", "flex"):
            raise ValueError("Invalid camera seg_ar_attention_backend")
        self.seg_ar_attention_backend = seg_ar_attention_backend
        self.per_segment_camera_attn = bool(per_segment_camera_attn)
        # Per-chunk camera cross-attention: fixed latent-frame chunks instead of
        # prompt segments. Block-diagonal (each chunk only sees its own keyboard
        # chunk). Decoupled from the per-segment dropout gate, which still uses
        # prompt_video_boundaries.
        self.per_chunk_camera_attn = bool(per_chunk_camera_attn)
        self.camera_chunk_size = int(camera_chunk_size)
        if self.per_chunk_camera_attn and self.camera_chunk_size <= 0:
            raise ValueError(
                f"camera_chunk_size must be >= 1 when per_chunk_camera_attn=True, "
                f"got {self.camera_chunk_size}."
            )
        if (
            (self.per_segment_camera_attn or self.per_chunk_camera_attn)
            and seg_ar_attention_backend == "magi"
            and not _MAGI_ATTENTION_AVAILABLE
        ):
            raise ImportError(
                "ActionModule(per_segment_camera_attn / per_chunk_camera_attn) "
                "requires the magi_attention library (flex_flash_attn_func). "
                "Install it, or disable both to fall back to the legacy "
                "flash_attn_func path."
            )

        # keyboard embedding
        self.keyboard_embed = nn.Sequential(
            nn.Linear(keyboard_dim_in, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

        # Get norm layer
        qk_norm_layer = get_norm_layer(qk_norm_type)

        head_dim_key = keyboard_hidden_dim // heads_num
        self.key_attn_q_norm = (
            qk_norm_layer(head_dim_key, elementwise_affine=True, eps=1e-6)
            if qk_norm
            else nn.Identity()
        )
        self.key_attn_k_norm = (
            qk_norm_layer(head_dim_key, elementwise_affine=True, eps=1e-6)
            if qk_norm
            else nn.Identity()
        )

        # Cross-attention projections. Keyboard is expected to already be aligned
        # to the image latent's temporal axis (one keyboard token per latent
        # frame); upstream preprocessing (e.g. _preprocess_camera in nomas_cpt.py
        # for extrinsics_attn mode) is responsible for that alignment.
        self.img_attn_q = nn.Linear(img_hidden_size, keyboard_hidden_dim, bias=qkv_bias)
        self.keyboard_attn_kv = nn.Linear(
            hidden_size, keyboard_hidden_dim * 2, bias=qkv_bias
        )
        self.proj_keyboard = nn.Linear(
            keyboard_hidden_dim, img_hidden_size, bias=qkv_bias
        )

        # Pre-compute rope embeddings. ``get_nd_rotary_pos_embed`` builds these
        # on CPU; we register them as (non-persistent) buffers so they ride
        # along with ``.to(device)`` / FSDP module placement and are NOT copied
        # host->device on every forward (the previous ``self.freqs_cos.to(...)``
        # in forward triggered a (7500, head_dim) CPU->GPU copy + sync at every
        # layer of every step, serializing the pipeline and tanking MFU).
        freqs_cos, freqs_sin = self.get_rotary_pos_embed(
            7500, self.patch_size[1], self.patch_size[2], self.rope_dim_list
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def _rope_tables_on(self, device):
        """Return RoPE tables on ``device``, lazily migrating them once.

        Guards against the module's buffers still living on CPU (e.g. when the
        model was never explicitly moved with ``.to(device)``). After the first
        call the buffers stay on the target device, so subsequent forwards do
        no host->device copy.
        """
        if self.freqs_cos.device != device:
            self.freqs_cos = self.freqs_cos.to(device)
            self.freqs_sin = self.freqs_sin.to(device)
        return self.freqs_cos, self.freqs_sin

    def get_rotary_pos_embed(self, video_length, height, width, rope_dim_list=None):
        """
        Get rotary position embedding.
        Uses the existing get_nd_rotary_pos_embed function.
        """
        target_ndim = 3

        latents_size = [video_length, height, width]

        if isinstance(self.patch_size, int):
            rope_sizes = [s // self.patch_size for s in latents_size]
        elif isinstance(self.patch_size, list):
            rope_sizes = [
                s // self.patch_size[idx] for idx, s in enumerate(latents_size)
            ]

        if len(rope_sizes) != target_ndim:
            rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes

        # forward() indexes freqs_cos/sin by frame_ids = token_ids // (th*tw) only;
        # that matches get_nd_rotary_pos_embed's table iff spatial rope grid is 1x1.
        assert rope_sizes[1] == 1 and rope_sizes[2] == 1, (
            "ActionModule RoPE requires rope spatial sizes (h, w) == (1, 1); "
            f"got rope_sizes={rope_sizes} from latents_size={latents_size}, "
            f"patch_size={self.patch_size}"
        )

        head_dim = sum(rope_dim_list) if rope_dim_list else 64
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert sum(rope_dim_list) == head_dim, (
            "sum(rope_dim_list) should equal to head_dim of attention layer"
        )

        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list,
            rope_sizes,
            theta=self.rope_theta,
            use_real=True,
            theta_rescale_factor=1,
        )
        return freqs_cos, freqs_sin

    def _range_kernel(self, **kwargs):
        if self.seg_ar_attention_backend == "flex":
            # Q ranges are already local to the SP rank; K geometry is unchanged.
            return flex_camera_range_attention(**kwargs)
        return range_attention(backend="magi", scope="camera", **kwargs)

    def _blockwise_cross_attn(
        self,
        q: torch.Tensor,  # [B, TS_local, H, D]   — Q (image tokens)
        k: torch.Tensor,  # [B, N_frames, H, D]   — K (keyboard, full)
        v: torch.Tensor,  # [B, N_frames, H, D]
        tt: int,  # global latent-frame count
        th: int,
        tw: int,
        attn_boundaries_latent: List[int],  # [num_blocks + 1] in latent-frame units
    ) -> torch.Tensor:
        """Block-diagonal cross-attention via flex_flash_attn_func.

        ``attn_boundaries_latent`` partitions the latent-frame axis into blocks;
        a block is a prompt segment (per_segment_camera_attn) or a fixed-size
        chunk (per_chunk_camera_attn) — this method is agnostic to which.

        Builds (q_ranges, k_ranges) such that image tokens of block j only
        attend to keyboard tokens of the same block:

            q_ranges[j] = [vb[j] * th*tw, vb[j+1] * th*tw)   # image-token units
            k_ranges[j] = [vb[j],         vb[j+1])           # keyboard-token units
            attn_type   = 0 (full attention within block)

        SP-aware: q is the LOCAL chunk along token dim (size TS_local), k is
        the FULL keyboard (size N_frames). q_ranges are intersected with the
        local rank's [local_start, local_end) and remapped to local
        coordinates. k_ranges stay global. Mirrors the SP layout used by
        the legacy ``flash_attn_func`` path.
        """
        B, TS_local, H, D = q.shape
        N_frames = k.shape[1]
        S = th * tw

        # Determine this rank's local Q range (global token indices).
        # MUST match how the image sequence is sharded across SP ranks upstream
        # (``torch.chunk(img, sp_size, dim=1)`` in nomas_cpt.forward /
        # forward_with_cache). ``torch.chunk`` uses split_size=ceil(L/sp_size)
        # for the first ranks and gives the remainder to the last — NOT the
        # "balanced" (first ``rem`` ranks get +1) distribution. Using the
        # balanced formula here silently mis-offsets later ranks whenever
        # total_tokens % sp_size is not 0 or sp_size-1.
        local_start, local_end = self._sp_local_window(tt * S, TS_local)

        # Cached range tensors — avoids rebuilding (and host→device sync)
        # at every layer. Inputs are step-stable: attn_boundaries_latent
        # is fixed for one sample / one step, SP rank's local window is
        # fixed for the whole step, S and N_frames are model-config-fixed.
        boundaries_tuple = tuple(int(b) for b in attn_boundaries_latent)
        cached = _get_cached_segment_attn_ranges(
            boundaries=boundaries_tuple,
            local_start=local_start,
            local_end=local_end,
            tokens_per_frame=S,
            n_frames=N_frames,
            device=q.device,
        )
        if cached is None:
            # Defensive: this rank has no overlap at all (should not happen
            # in normal training). Return zeros so the residual add is a no-op.
            return torch.zeros(B, TS_local, H, D, dtype=q.dtype, device=q.device)
        q_ranges, k_ranges, attn_type_map = cached

        # flex_flash_attn_func expects flat [N, H, D] tensors. Loop over batch
        # for B>1 (per-sample boundaries are not supported here yet — caller
        # currently passes a single boundary list shared across the batch,
        # consistent with how WM1MCameraDataset emits per-sample data).
        if B == 1:
            out_flat, _ = self._range_kernel(
                q=q.squeeze(0),
                k=k.squeeze(0),
                v=v.squeeze(0),
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                attn_type_map=attn_type_map,
            )
            return out_flat.unsqueeze(0)

        outs = []
        for b in range(B):
            out_b, _ = self._range_kernel(
                q=q[b],
                k=k[b],
                v=v[b],
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                attn_type_map=attn_type_map,
            )
            outs.append(out_b)
        return torch.stack(outs, dim=0)

    def _sp_local_window(self, total_tokens: int, TS_local: int) -> Tuple[int, int]:
        """Return [local_start, local_end) for this SP-then-TP token shard.

        ``local_start`` is computed to match ``torch.chunk(img, sp_size, dim=1)``
        (the upstream SP image shard), i.e. split_size=ceil(total/sp_size) with
        the remainder absorbed by the last rank — see ``_torch_chunk_local_start``.
        TP then splits that SP shard into balanced contiguous token ranges.
        ``local_end`` is anchored to the actual ``TS_local`` this rank received,
        so the window length always equals the real shard size.
        """
        local_start = 0
        local_tokens = total_tokens
        if get_sequence_parallel_state():
            sp_size = nccl_info.sp_size
            sp_rank = nccl_info.rank_within_spgroup
            local_start = _torch_chunk_local_start(total_tokens, sp_size, sp_rank)
            local_tokens = min((total_tokens + sp_size - 1) // sp_size,
                               max(0, total_tokens - local_start))
        tp_size = getattr(self, "tensor_parallel_size", 1)
        if tp_size > 1:
            from hymm.distributed.tensor_parallel import sequence_partition
            offset, _ = sequence_partition(local_tokens, tp_size, self.tensor_parallel_rank)
            local_start += offset
        local_end = local_start + TS_local
        return int(local_start), int(local_end)

    def _build_packed_camera_token_gate(
        self,
        *,
        packed_has_camera: List[bool],
        packed_has_camera_per_segment: Optional[List[List[bool]]],
        packed_img_token_counts: List[int],
        packed_sample_shapes: List[Tuple[int, int, int]],
        packed_segment_boundaries_latent: Optional[List[Optional[List[int]]]],
        TS_local: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Per-token gate [TS_local] (1.0=keep / 0.0=drop) combining two OFF signals:

        * whole row OFF when ``packed_has_camera[i]`` is False;
        * per-segment OFF when ``packed_has_camera_per_segment[i][j]`` is False
          (training dropout; only meaningful for camera-ON rows).

        Returns None when nothing is dropped. SP-aware; uses the same
        (sample, segment) -> image-token math as ``_build_packed_segment_flex_ranges``.
        """
        n = len(packed_img_token_counts)
        seg = packed_has_camera_per_segment
        row_on = lambda i: (
            bool(packed_has_camera[i]) if i < len(packed_has_camera) else True
        )
        seg_off = lambda i, j: (
            seg is not None and i < len(seg) and j < len(seg[i]) and not seg[i][j]
        )

        # Fast path: every row has camera and no segment is dropped.
        if all(row_on(i) for i in range(n)) and (
            seg is None
            or not any((i < len(seg)) and (not all(seg[i])) for i in range(n))
        ):
            return None

        pt, ph, pw = (
            int(self.patch_size[0]),
            int(self.patch_size[1]),
            int(self.patch_size[2]),
        )
        total_img = int(sum(packed_img_token_counts))
        local_start, local_end = self._sp_local_window(total_img, TS_local)

        gate = torch.ones(TS_local, device=device, dtype=dtype)
        seg_lists = packed_segment_boundaries_latent or [None] * n
        img_cum = 0
        for i in range(n):
            L_i = packed_img_token_counts[i]
            T_i, H_i, W_i = packed_sample_shapes[i]
            S_i = max((H_i // ph) * (W_i // pw), 1)
            T_lat = max(T_i // pt, 1)
            boundaries = _normalize_sample_latent_boundaries(
                seg_lists[i] if i < len(seg_lists) else None, T_lat
            )
            drop_row = not row_on(i)
            for j in range(len(boundaries) - 1):
                if not (drop_row or seg_off(i, j)):
                    continue
                q0 = max(img_cum + boundaries[j] * S_i, local_start) - local_start
                q1 = min(img_cum + boundaries[j + 1] * S_i, local_end) - local_start
                if q0 < q1:
                    gate[q0:q1] = 0.0
            img_cum += L_i
        return gate

    def _packed_cross_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        packed_img_token_counts: List[int],
        packed_kb_token_counts: List[int],
        packed_sample_shapes: List[Tuple[int, int, int]],
        packed_has_camera: List[bool],
        packed_segment_boundaries_latent: Optional[List[Optional[List[int]]]] = None,
    ) -> torch.Tensor:
        """Cross-attn for NaViT token packing.

        - **Magi path** (``per_segment_camera_attn`` and Magi / flex_flash available):
          one ``flex_flash_attn_func`` with merged (q_ranges, k_ranges) across
          packed samples and latent-frame segments (multi-prompt camera).
        - **Flash fallback** (``else`` branch below): one ``flash_attn_func`` call
          per packed sample (each image token sees that sample's full keyboard).
          Also used when the Magi path is selected but flex range cache is empty
          (e.g. rare geometry / SP edge), instead of returning an all-zero tensor.
        """
        B, TS_local, Hh, D = q.shape
        assert B == 1, (
            "packed camera path currently supports B==1 (fused token packing)"
        )
        _, N_kb_total, Hk, Dk = k.shape
        assert Hh == Hk and D == Dk

        total_img = int(sum(packed_img_token_counts))
        total_kb = int(sum(packed_kb_token_counts))
        assert total_kb == N_kb_total, (
            f"keyboard length {N_kb_total} != sum(packed_kb_token_counts)={total_kb}"
        )

        local_start, local_end = self._sp_local_window(total_img, TS_local)
        assert local_end - local_start == TS_local

        pt, ph, pw = (
            int(self.patch_size[0]),
            int(self.patch_size[1]),
            int(self.patch_size[2]),
        )

        # --- RoPE: per packed-sample local temporal frame indices ---
        freqs_cos, freqs_sin = self._rope_tables_on(q.device)

        token_ids = (
            torch.arange(TS_local, device=q.device, dtype=torch.long) + local_start
        )
        frame_ids_q = torch.zeros(TS_local, dtype=torch.long, device=q.device)
        img_cum = 0
        for i, L_i in enumerate(packed_img_token_counts):
            T_i, H_i, W_i = packed_sample_shapes[i]
            S_i = max((H_i // ph) * (W_i // pw), 1)
            t_tok_i = max(T_i // pt, 1)
            g0, g1 = img_cum, img_cum + L_i
            mask = (token_ids >= g0) & (token_ids < g1)
            if mask.any():
                off = token_ids[mask] - g0
                fid = torch.div(off, S_i, rounding_mode="floor").clamp(max=t_tok_i - 1)
                frame_ids_q[mask] = fid
            img_cum += L_i

        freqs_cos_q = freqs_cos[frame_ids_q]
        freqs_sin_q = freqs_sin[frame_ids_q]
        q, _ = apply_rotary_emb(
            q,
            q,
            (freqs_cos_q, freqs_sin_q),
            head_first=False,
        )

        frame_ids_k = torch.empty(N_kb_total, dtype=torch.long, device=k.device)
        kb_cum = 0
        rope_t_max = max(int(self.freqs_cos.shape[0]) - 1, 0)
        for i, K_i in enumerate(packed_kb_token_counts):
            fid = torch.arange(K_i, device=k.device, dtype=torch.long).clamp(
                max=rope_t_max
            )
            frame_ids_k[kb_cum : kb_cum + K_i] = fid
            kb_cum += K_i
        freqs_cos_k = freqs_cos[frame_ids_k]
        freqs_sin_k = freqs_sin[frame_ids_k]
        _, k = apply_rotary_emb(
            k,
            k,
            (freqs_cos_k, freqs_sin_k),
            head_first=False,
        )

        use_magi_packed = (
            (self.per_segment_camera_attn or self.per_chunk_camera_attn)
            and (self.seg_ar_attention_backend == "flex" or _MAGI_ATTENTION_AVAILABLE)
        )
        if use_magi_packed:
            n = len(packed_img_token_counts)
            # Per-chunk mode: subdivide each sample's prompt segments into
            # fixed-size latent-frame chunks for the cross-attn ranges only.
            # Chunks are SEGMENT-AWARE (never cross a prompt-segment boundary),
            # so a dropped (camera-OFF) segment cannot leak its keyboard into a
            # neighbouring kept segment. The gate (built separately in forward)
            # still uses the prompt boundaries, so dropout semantics are intact.
            seg_adj: List[Optional[List[int]]]
            if self.per_chunk_camera_attn:
                _pb = packed_segment_boundaries_latent
                seg_adj = []
                for i_s, sh in enumerate(packed_sample_shapes):
                    T_lat_i = max(int(sh[0]) // pt, 1)
                    pb_i = _coerce_packed_seg_latent_list(
                        _pb[i_s] if (_pb is not None and i_s < len(_pb)) else None
                    )
                    seg_adj.append(
                        _segment_aware_chunk_boundaries(
                            pb_i, T_lat_i, self.camera_chunk_size
                        )
                    )
            else:
                seg_adj = (
                    list(packed_segment_boundaries_latent)
                    if packed_segment_boundaries_latent is not None
                    else [None] * n
                )
            while len(seg_adj) < n:
                seg_adj.append(None)
            # Tuple-of-tuples for cache key + flex builder; coerce tensors / nested lists.
            packed_seg_tuples = tuple(
                tuple(lst) if lst is not None else tuple()
                for lst in (
                    _coerce_packed_seg_latent_list(
                        seg_adj[i] if i < len(seg_adj) else None
                    )
                    for i in range(n)
                )
            )
            cached = _get_cached_packed_segment_flex_ranges(
                packed_img_token_counts=tuple(packed_img_token_counts),
                packed_kb_token_counts=tuple(packed_kb_token_counts),
                packed_sample_shapes=tuple(tuple(sh) for sh in packed_sample_shapes),
                packed_has_camera=tuple(bool(x) for x in packed_has_camera),
                packed_seg_boundaries=packed_seg_tuples,
                local_start=local_start,
                local_end=local_end,
                patch_size=(pt, ph, pw),
                device=q.device,
            )
            if cached is not None:
                q_ranges, k_ranges, attn_type_map = cached
                out_flat, _ = self._range_kernel(
                    q=q.squeeze(0),
                    k=k.squeeze(0),
                    v=v.squeeze(0),
                    q_ranges=q_ranges,
                    k_ranges=k_ranges,
                    attn_type_map=attn_type_map,
                )
                return out_flat.unsqueeze(0)
            if self.seg_ar_attention_backend == "flex":
                # No local query has any visible keyboard range.
                return torch.zeros_like(q)
            # Keep the legacy Magi fallback unchanged.
        else:
            # ``per_segment_camera_attn`` off or Magi / flex_flash unavailable.
            logger.debug(
                "Packed camera: using per-sample full cross-attention "
                f"with backend={self.seg_ar_attention_backend}."
            )

        acc = torch.zeros(B, TS_local, Hh, D, dtype=q.dtype, device=q.device)
        global_pos = (
            torch.arange(TS_local, device=q.device, dtype=torch.long) + local_start
        )
        img_cum0 = 0
        kb_cum0 = 0
        for i, L_i in enumerate(packed_img_token_counts):
            K_i = packed_kb_token_counts[i]
            g0, g1 = img_cum0, img_cum0 + L_i
            k0, k1 = kb_cum0, kb_cum0 + K_i
            img_cum0 += L_i
            kb_cum0 += K_i

            mask = (global_pos >= g0) & (global_pos < g1)
            if not mask.any():
                continue
            if K_i <= 0:
                continue

            local_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            q_i = q[:, local_idx, :, :]  # [1, n, H, D]
            k_i = k[:, k0:k1, :, :]  # [1, K_i, H, D]
            v_i = v[:, k0:k1, :, :]  # [1, K_i, H, D]

            # flash_attn_func expects 4D (batch, seqlen, nheads, headdim); do not squeeze(0).
            if self.seg_ar_attention_backend == "flex":
                qr = torch.tensor([[0, q_i.shape[1]]], device=q.device, dtype=torch.int32)
                kr = torch.tensor([[0, k_i.shape[1]]], device=q.device, dtype=torch.int32)
                out_i = self._range_kernel(q=q_i.squeeze(0), k=k_i.squeeze(0),
                    v=v_i.squeeze(0), q_ranges=qr, k_ranges=kr)[0].unsqueeze(0)
            else:
                out_i = flash_attn_func(q_i, k_i, v_i, causal=False)
            acc[:, local_idx, :, :] = out_i

        return acc

    def forward(
        self,
        x,
        tt,
        th,
        tw,
        keyboard,
        has_camera_mask=None,
        segment_boundaries_latent: Optional[List[int]] = None,
        packed_img_token_counts: Optional[List[int]] = None,
        packed_kb_token_counts: Optional[List[int]] = None,
        packed_sample_shapes: Optional[List[Tuple[int, int, int]]] = None,
        packed_has_camera: Optional[List[bool]] = None,
        packed_segment_boundaries_latent: Optional[List[Optional[List[int]]]] = None,
        packed_has_camera_per_segment: Optional[List[List[bool]]] = None,
        rope_time_offset: int = 0,
    ):
        """
        Args:
            x:         [B, TS_local, C]   image hidden states. In SP mode TS_local
                       is the rank-local chunk of the global TS = tt * th * tw.
            tt:        global latent-frame count (NOT chunked by SP).
            th, tw:    spatial token grid sizes.
            keyboard:  [B, N_frames, C_kb]  camera/keyboard tokens. Always full
                       (not SP-chunked); upstream ``_preprocess_camera`` aligns
                       N_frames == tt for extrinsics_attn / action_attn modes.
            has_camera_mask: bool/float tensor that gates the cross-attn output
                       (cam_out is multiplied by the expanded mask before the
                       residual add). Two shapes accepted:
                         [B]               legacy per-sample mask. Sample served
                                           as text-only when value is 0/False.
                         [B, N_segments]   per-segment mask (requires
                                           ``segment_boundaries_latent`` so we
                                           can expand to a token-level mask via
                                           the same frame_ids used for RoPE).
                                           Used at inference for the joint
                                           bench, where ``segment_control ==
                                           "prompt"`` segments get camera off.
                       In both cases applies to the dense (non-packed) path; the
                       NaViT packed path keeps per-sample [B] semantics.
            segment_boundaries_latent: optional list of length num_segments+1
                       in latent-frame units, starts at 0, ends at tt. When
                       provided AND ``self.per_segment_camera_attn=True`` AND
                       num_segments > 1, dispatches to per-segment block-
                       diagonal cross-attention via ``flex_flash_attn_func``.
                       Otherwise falls back to the legacy ``flash_attn_func``
                       (image sees all keyboard).
            packed_img_token_counts / packed_kb_token_counts / packed_sample_shapes /
            packed_has_camera: when set, enable NaViT token-packing mode (B must be 1).
                All four lists must have the same length (number of packed samples).
            packed_segment_boundaries_latent: per packed sample, latent-frame
                ``prompt_video_boundaries`` (same convention as ``segment_boundaries_latent``).
                When ``per_segment_camera_attn=True`` and Magi is installed, packed camera
                cross-attention uses ``flex_flash_attn_func`` with merged segment ranges
                across samples; otherwise each sample uses one full cross-attn block
                (``flash_attn_func``). If provided, its length must match the four
                packed count lists above.
            rope_time_offset: optional latent-frame offset for dense AR cache
                inference. Defaults to 0 to preserve the training / one-shot path.
        SP layout (when ``get_sequence_parallel_state()`` is True):
            - x is the rank-local chunk along token dim, size TS_local.
            - keyboard / k / v stay full (size N_frames == tt).
            - RoPE positions for q are offset by ``local_start`` and
              ``rope_time_offset``; RoPE for k uses
              [rope_time_offset, rope_time_offset + N_frames).
        """
        assert x is not None, "hidden_states is required for action module"
        assert keyboard is not None, "keyboard is required for action module"
        if x.shape[1] == 0:
            # Short sequences can leave a TP rank with no query tokens. Camera
            # attention has no collectives; skip kernels that reject empty Q.
            return x

        hidden_states = x

        B, N_frames, _ = keyboard.shape

        # === Project Q from image, K/V from keyboard ===
        with _prof("camera/in_proj"):
            # img_q:   [B, TS_local, keyboard_hidden_dim]
            img_q = self.img_attn_q(hidden_states)
            # keyboard_embed: [B, N_frames, hidden_size]
            keyboard_embed = self.keyboard_embed(keyboard)
            # keyboard_kv:    [B, N_frames, keyboard_hidden_dim * 2]
            keyboard_kv = self.keyboard_attn_kv(keyboard_embed)

            # Reshape to multi-head layout.
            # q: [B, TS_local, H, D]
            B, TS, HD = img_q.shape
            D = HD // self.heads_num
            q = img_q.view(B, TS, self.heads_num, D)
            # k, v: [B, N_frames, H, D]
            k, v = keyboard_kv.view(B, N_frames, 2, self.heads_num, D).permute(
                2, 0, 1, 3, 4
            )

            # QK-norm.
            q = self.key_attn_q_norm(q).to(v)
            k = self.key_attn_k_norm(k).to(v)

        if packed_img_token_counts is not None:
            assert (
                packed_kb_token_counts is not None and packed_sample_shapes is not None
            )
            assert packed_has_camera is not None
            n_pack = len(packed_img_token_counts)
            _img_c = list(packed_img_token_counts)
            _kb_c = list(packed_kb_token_counts)
            _shp = list(packed_sample_shapes)
            _h_cam = list(packed_has_camera)
            _seg_lat = (
                list(packed_segment_boundaries_latent)
                if packed_segment_boundaries_latent is not None
                else None
            )
            assert len(packed_kb_token_counts) == n_pack, (
                f"packed_kb_token_counts len {len(packed_kb_token_counts)} != n_pack {n_pack} | "
                f"packed_img_token_counts={_img_c} packed_kb_token_counts={_kb_c}"
            )
            assert len(packed_sample_shapes) == n_pack, (
                f"packed_sample_shapes len {len(packed_sample_shapes)} != n_pack {n_pack} | "
                f"packed_img_token_counts={_img_c} packed_sample_shapes={_shp}"
            )
            assert len(packed_has_camera) == n_pack, (
                f"packed_has_camera len {len(packed_has_camera)} != n_pack {n_pack} | "
                f"packed_img_token_counts={_img_c} packed_has_camera={_h_cam}"
            )
            if packed_segment_boundaries_latent is not None:
                assert len(packed_segment_boundaries_latent) == n_pack, (
                    f"packed_segment_boundaries_latent len "
                    f"{len(packed_segment_boundaries_latent)} != n_pack {n_pack} | "
                    f"packed_segment_boundaries_latent={_seg_lat}"
                )
            with _prof("camera/cross_attn"):
                cam_heads = self._packed_cross_attention(
                    q,
                    k,
                    v,
                    packed_img_token_counts=_img_c,
                    packed_kb_token_counts=_kb_c,
                    packed_sample_shapes=_shp,
                    packed_has_camera=_h_cam,
                    packed_segment_boundaries_latent=_seg_lat,
                )
            with _prof("camera/out_proj"):
                cam_out = cam_heads.reshape(B, TS, self.heads_num * D)
                cam_out = self.proj_keyboard(cam_out)
            # Combined hard gate: zero cam_out for camera-OFF rows
            # (packed_has_camera[i] False, whole row) and, for camera-ON rows,
            # for per-segment dropout-OFF segments. Returns None (no-op) when
            # every row/segment is kept. This is the packed-path equivalent of
            # the non-packed has_camera_mask.
            with _prof("camera/cam_gate"):
                cam_gate = self._build_packed_camera_token_gate(
                    packed_has_camera=_h_cam,
                    packed_has_camera_per_segment=packed_has_camera_per_segment,
                    packed_img_token_counts=_img_c,
                    packed_sample_shapes=_shp,
                    packed_segment_boundaries_latent=_seg_lat,
                    TS_local=TS,
                    device=cam_out.device,
                    dtype=cam_out.dtype,
                )
            if cam_gate is not None:
                cam_out = cam_out * cam_gate.view(1, TS, 1)
            hidden_states = hidden_states + cam_out
            return hidden_states

        S = th * tw  # tokens per frame

        # === RoPE for Q (image tokens) and K (keyboard tokens) ===
        # q layout stays [B, TS_local, H, D]
        freqs_cos, freqs_sin = self._rope_tables_on(q.device)

        # Camera weights remain replicated; its queries follow the SP x TP
        # residual token range so RoPE and per-chunk visibility stay aligned.
        local_start, _ = self._sp_local_window(tt * S, TS)
        token_ids = torch.arange(TS, device=q.device) + local_start

        # Map each q token to its temporal frame index for temporal RoPE.
        rope_t_max = max(int(self.freqs_cos.shape[0]) - 1, 0)
        frame_ids = torch.div(token_ids, S, rounding_mode="floor").long()
        if rope_time_offset:
            frame_ids = frame_ids + int(rope_time_offset)
        frame_ids = frame_ids.clamp(max=rope_t_max)
        freqs_cos_q = freqs_cos[frame_ids]
        freqs_sin_q = freqs_sin[frame_ids]

        # k uses the full RoPE range for the keyboard, optionally shifted by
        # the AR chunk's global latent-frame offset.
        frame_ids_k = torch.arange(N_frames, device=k.device, dtype=torch.long)
        if rope_time_offset:
            frame_ids_k = frame_ids_k + int(rope_time_offset)
        frame_ids_k = frame_ids_k.clamp(max=rope_t_max)
        freqs_cos_k = freqs_cos[frame_ids_k]
        freqs_sin_k = freqs_sin[frame_ids_k]

        # Apply RoPE separately to q and k.
        q, _ = apply_rotary_emb(
            q,
            q,
            (freqs_cos_q, freqs_sin_q),
            head_first=False,
        )  # only `q` is meaningful
        _, k = apply_rotary_emb(
            k,
            k,
            (freqs_cos_k, freqs_sin_k),
            head_first=False,
        )  # only `k` is meaningful

        # === Cross-attention dispatch ===
        # Use per-segment block-diagonal magi attention only when
        #   1. opted in via ``self.per_segment_camera_attn``, AND
        #   2. caller passed multi-segment boundaries (len > 2 means
        #      at least 2 segments; len == 2 is a single segment [0, tt],
        #      which is equivalent to "image sees all keyboard" — fall
        #      back to flash_attn to skip the magi setup overhead).
        # Per-chunk mode: subdivide each prompt segment into fixed-size latent-frame
        # chunks for the cross-attn ranges. Chunks are SEGMENT-AWARE (never cross a
        # prompt-segment boundary) so a dropped segment cannot leak into a kept one.
        # The has_camera_mask gate below still uses the prompt
        # ``segment_boundaries_latent`` (decoupled), matching the packed path.
        if self.per_chunk_camera_attn:
            cam_attn_boundaries = _segment_aware_chunk_boundaries(
                segment_boundaries_latent, tt, self.camera_chunk_size
            )
        else:
            cam_attn_boundaries = segment_boundaries_latent
        use_segmented = (
            (self.per_segment_camera_attn or self.per_chunk_camera_attn)
            and cam_attn_boundaries is not None
            and len(cam_attn_boundaries) > 2
        )
        with _prof("camera/cross_attn"):
            if use_segmented:
                # attn: [B, TS_local, H, D]
                cam_out = self._blockwise_cross_attn(
                    q=q,
                    k=k,
                    v=v,
                    tt=tt,
                    th=th,
                    tw=tw,
                    attn_boundaries_latent=cam_attn_boundaries,
                )
            else:
                # Legacy: each image token attends to all keyboard tokens.
                # q: [B, TS_local, H, D]   k, v: [B, N_frames, H, D]
                if self.seg_ar_attention_backend == "flex":
                    qr = torch.tensor([[0, q.shape[1]]], device=q.device, dtype=torch.int32)
                    kr = torch.tensor([[0, k.shape[1]]], device=q.device, dtype=torch.int32)
                    cam_out = torch.stack([
                        self._range_kernel(q=q[b], k=k[b], v=v[b],
                                           q_ranges=qr, k_ranges=kr)[0]
                        for b in range(q.shape[0])
                    ])
                else:
                    cam_out = flash_attn_func(q, k, v, causal=False)

        # Merge heads back: [B, TS_local, H*D] → project to img hidden size.
        with _prof("camera/out_proj"):
            cam_out = cam_out.reshape(B, TS, self.heads_num * D)
            cam_out = self.proj_keyboard(cam_out)

        # Zero out cross-attention for samples / segments without real camera data.
        # ``has_camera_mask`` accepts two shapes:
        #   [B]              legacy per-sample (broadcast over all tokens)
        #   [B, N_segments]  per-segment; requires segment_boundaries_latent
        #                    (already computed for RoPE) so we can expand to
        #                    a token-level [B, TS_local] mask.
        if has_camera_mask is not None:
            if has_camera_mask.dim() == 1:
                # [B] -> [B, 1, 1] broadcast over (TS_local, hidden).
                mask = has_camera_mask.float().unsqueeze(-1).unsqueeze(-1)
            elif has_camera_mask.dim() == 2:
                assert (
                    segment_boundaries_latent is not None
                    and len(segment_boundaries_latent) >= 2
                ), (
                    f"Per-segment has_camera_mask (shape={tuple(has_camera_mask.shape)}) "
                    f"requires segment_boundaries_latent, got {segment_boundaries_latent}"
                )
                N_seg = has_camera_mask.shape[1]
                assert len(segment_boundaries_latent) == N_seg + 1, (
                    f"segment_boundaries_latent length {len(segment_boundaries_latent)} "
                    f"!= N_seg+1 ({N_seg + 1})"
                )
                # frame_ids: [TS_local] global tt-frame index per local token,
                # already SP-aware (computed above for RoPE).
                # Use interior boundaries for bucketize: token with frame_id
                # in [boundaries[i], boundaries[i+1]) → segment i.
                interior = torch.tensor(
                    list(segment_boundaries_latent[1:-1]),
                    device=frame_ids.device,
                    dtype=frame_ids.dtype,
                )
                if interior.numel() == 0:
                    seg_ids = torch.zeros_like(frame_ids)
                else:
                    seg_ids = torch.bucketize(frame_ids, interior, right=True)
                seg_ids = seg_ids.clamp(max=N_seg - 1).long()
                # has_camera_mask: [B, N_seg], seg_ids: [TS_local]
                #   → token_mask: [B, TS_local]
                token_mask = has_camera_mask.float()[:, seg_ids]
                mask = token_mask.unsqueeze(-1)  # [B, TS_local, 1]
            else:
                raise ValueError(
                    f"has_camera_mask must be 1D [B] or 2D [B, N_segments], "
                    f"got shape {tuple(has_camera_mask.shape)}"
                )
            cam_out = cam_out * mask

        hidden_states = hidden_states + cam_out
        return hidden_states
