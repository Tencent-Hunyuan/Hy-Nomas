from .modules.attention.nomas_inference_optim import scoped_forward as scoped_sparse_optim_forward
"""
CPT (Continue Pre-Training) model for bidirectional training with per-chunk text modulation support.

Based on Nomas_ar.py architecture but with:
- Bidirectional attention (no causal mask)
- Scalar timestep [B] (expanded to [B, T] for segment text)
- Single blocks ENABLED
- No teacher forcing / diffusion forcing specific logic

The model reuses block classes from Nomas_ar.py (supporting both 2D and 3D vec modes)
and is checkpoint-compatible with both Nomas.py and Nomas_ar.py models.
"""

import math
import contextlib
import itertools
from typing import Any, List, Tuple, Optional, Union, Dict
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function as _record_function

from hymm.utils.inference_metrics import profiled


def _prof(name):
    """record_function span that is a no-op unless a profiler is active.

    Guards the per-block hot path: normal (non-profiled) training pays no
    context-manager overhead, while ``is_profiler`` runs get named spans.
    """
    try:
        active = torch.autograd._profiler_enabled()
    except Exception:
        active = False
    return _record_function(name) if active else contextlib.nullcontext()


from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from .modules.activation_layers import get_activation_layer
from .modules.norm_layers import get_norm_layer, RMSNorm
from .modules.embed_layers import TimestepEmbedder, PatchEmbed, TextProjection
from .modules.mlp_layers import (
    MLP,
    FinalLayerPerBlock,
    LinearWarpforSingle,
    fixed_m_linear,
)
from .modules.attention import (
    parallel_attention,
    preprocess_dynamic_ring_attention,
    postprocess_dynamic_ring_attention,
)
from .modules.modulate_layers import ModulateDiT, modulate, apply_gate
from .modules.nomas_compile_prepost import compiled_pre_attention, compiled_post_attention
from .modules.int8_stored_bf16_linear import Int8StoredBF16Linear, linear_weight
from .modules.uniform_history_cache import ARCHIVED_HISTORY_SAMPLINGS, HISTORY_SAMPLINGS
from .modules.posemb_layers import (
    apply_rotary_emb,
    get_nd_rotary_pos_embed,
    get_nd_rotary_pos_embed_with_temporal_positions,
    get_packed_rotary_pos_embed,
    get_rotary_pos_embed_with_offset,
)
from hymm.models.text_encoder import SingleTokenRefinerPerBlock
from .moe.hunyuan_moe import MoeConfig, HunYuanMoE

from hymm.distributed.parallel_states import get_sequence_parallel_state, nccl_info
from hymm.distributed.communications import all_gather, all_to_all_4D
from hymm.distributed.tensor_parallel import (
    gather_sequence, local_modulation_gate, shard_transformer_streams,
)

try:
    from hymm.utils.segment_attention_profiler import (
        seg_attn_profiler as _seg_profiler_ar,
        compute_seg1_attention_scores_ar as _compute_ar_scores,
    )

    _HAS_AR_VIS_PROFILER = True
except ImportError:
    _HAS_AR_VIS_PROFILER = False

from .modules.attention.backends import (
    magi_attention as _flex_flash_attn_func,
    profile_flash_attention as _ar_profile_fa_func,
)
from .modules.attention.range_attention import range_attention, flex_masked_attention

import os as _os

_INTRA_SEG_CACHE_DETERMINISTIC = (
    _os.environ.get("MAGI_ATTENTION_DETERMINISTIC_MODE", "0") == "1"
)
_FLASH_CACHE_DETERMINISTIC = (
    _os.environ.get("FLASH_ATTENTION_DETERMINISTIC_MODE", "0") == "1"
)


_intra_seg_range_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _build_intra_seg_cache_ranges(
    hvis: int,
    limg: int,
    valid: int,
    offs: List[int],
    windows: List[List[int]],
    device: torch.device,
):
    """Build flex_flash_attn ranges for the CURRENT-chunk intra-segment mask.

    Flat per-sample sequence layout (visible tokens only)::

        [ visible_history (hvis) | cur_img (limg) | cur_txt_valid (valid) ]

    Query coords: img -> ``[0, limg)``, txt -> ``[limg, limg+valid)``.
    Key coords:   hist -> ``[0, hvis)``, img -> ``[hvis, hvis+limg)``,
                  txt  -> ``[hvis+limg, hvis+limg+valid)``.

    ``offs`` is ``[o_0, ..., o_A]`` (post-crop text offsets, monotonic, clamped
    to ``valid``): head is ``[0, o_0)``, action ``k`` is ``[o_k, o_{k+1})``.
    ``windows`` tiles ``[0, limg)`` as ``[ws, we, aidx]`` (aidx == -1 = gap).

    History is a uniformly-visible prefix (cross-segment tail masking is already
    applied upstream via ``cache_mask``); only the current chunk's video<->text
    cross terms are restricted, mirroring the training segment-level magi branch.
    """
    key = (
        f"h{hvis}_i{limg}_v{valid}_"
        f"o{'-'.join(str(x) for x in offs)}_"
        f"w{';'.join(','.join(str(x) for x in w) for w in windows)}"
    )
    cached = _intra_seg_range_cache.get(key)
    if cached is not None:
        if cached[0].device != device:
            cached = (cached[0].to(device), cached[1].to(device), cached[2].to(device))
            _intra_seg_range_cache[key] = cached
        return cached

    img_k0 = hvis
    txt_k0 = hvis + limg
    A = len(offs) - 1
    head_end = min(int(offs[0]), valid)

    q_list: List[List[int]] = []
    k_list: List[List[int]] = []

    def _emit(qs: int, qe: int, own_action: Optional[int]):
        if qe <= qs:
            return
        # History (visible prefix) — skip when empty (first segment).
        if hvis > 0:
            q_list.append([qs, qe])
            k_list.append([0, hvis])
        # All current video.
        q_list.append([qs, qe])
        k_list.append([img_k0, img_k0 + limg])
        # Head text.
        if head_end > 0:
            q_list.append([qs, qe])
            k_list.append([txt_k0, txt_k0 + head_end])
        # Own action text slice.
        if own_action is not None and 0 <= own_action < A:
            s = txt_k0 + min(int(offs[own_action]), valid)
            e = txt_k0 + min(int(offs[own_action + 1]), valid)
            if e > s:
                q_list.append([qs, qe])
                k_list.append([s, e])

    # Video sub-window queries (img coords).
    for win in windows:
        ws, we, aidx = int(win[0]), int(win[1]), int(win[2])
        ws = max(0, min(ws, limg))
        we = max(0, min(we, limg))
        _emit(ws, we, None if aidx < 0 else aidx)

    # Head-text query (txt coords, offset by limg).
    _emit(limg, limg + head_end, None)
    # Per-action text queries.
    for kk in range(A):
        qs = limg + min(int(offs[kk]), valid)
        qe = limg + min(int(offs[kk + 1]), valid)
        _emit(qs, qe, kk)

    q_ranges = torch.tensor(q_list, dtype=torch.int32, device=device)
    k_ranges = torch.tensor(k_list, dtype=torch.int32, device=device)
    attn_type_map = torch.zeros(len(q_list), dtype=torch.int32, device=device)
    _intra_seg_range_cache[key] = (q_ranges, k_ranges, attn_type_map)
    return q_ranges, k_ranges, attn_type_map


@profiled("cache_attention")
def _intra_seg_cache_forward(
    q_full: torch.Tensor,
    full_k: torch.Tensor,
    full_v: torch.Tensor,
    kv_mask: torch.Tensor,
    cur_txt_mask: torch.Tensor,
    cache_valid_len: int,
    img_q_len: int,
    txt_offsets_per_sample: List[Optional[List[int]]],
    vid_windows: List[List[int]],
    seg_ar_attention_backend: str = "magi",
) -> torch.Tensor:
    """Per-sample flex-attention for the current-chunk intra-segment mask.

    Operates on the composed ``full_k``/``full_v`` sequence
    ``[history | cur_img | cur_txt]`` and the query ``q_full = [img_q | txt_q]``
    (post-SP tensors, identical to the flash3 varlen path). History is compacted
    per sample via ``kv_mask`` (cross-segment tail masking already applied), the
    current video is fully visible, and each action's video/text is restricted to
    its own text slice via :func:`_build_intra_seg_cache_ranges`.

    ``txt_offsets_per_sample[b]`` is ``None`` -> that sample uses full (legacy)
    bidirectional attention over its visible tokens; else ``[o_0, ..., o_A]``.
    Returns ``hidden`` ``[B, img_q_len + txt_q_len, H, D]`` (txt padding rows 0).
    """
    B = q_full.size(0)
    H = q_full.size(2)
    D = q_full.size(3)
    S_q = q_full.size(1)
    txt_q_len = S_q - img_q_len
    device = q_full.device
    hidden = torch.zeros(B, S_q, H, D, device=device, dtype=q_full.dtype)

    txt_k0 = cache_valid_len + img_q_len

    for b in range(B):
        # Compact visible history for this sample.
        if cache_valid_len > 0:
            hist_valid = kv_mask[b, :cache_valid_len].bool()
            hist_idx = hist_valid.nonzero(as_tuple=False).squeeze(-1)
        else:
            hist_idx = torch.empty(0, dtype=torch.long, device=device)
        hvis = int(hist_idx.numel())

        # Current-chunk valid text token indices (within the txt block).
        txt_valid = cur_txt_mask[b].bool()
        txt_idx_local = txt_valid.nonzero(as_tuple=False).squeeze(-1)
        valid = int(txt_idx_local.numel())

        if seg_ar_attention_backend == "flex":
            from .modules.attention.range_attention import expand_compact_ranges

            query_mask = torch.cat((
                torch.ones(img_q_len, device=device, dtype=torch.bool), txt_valid,
            ))
            key_mask = torch.cat((kv_mask[b, :cache_valid_len], query_mask))
            offs_b = txt_offsets_per_sample[b] if b < len(txt_offsets_per_sample) else None
            if offs_b is None:
                qr, kr, types = [[0, S_q]], [[0, full_k.shape[1]]], None
            else:
                qr, kr, types = _build_intra_seg_cache_ranges(
                    hvis=hvis, limg=img_q_len, valid=valid,
                    offs=[min(int(o), valid) for o in offs_b],
                    windows=vid_windows, device=device,
                )
                query_positions = torch.cat((torch.arange(img_q_len, device=device),
                                             img_q_len + txt_idx_local))
                key_positions = torch.cat((hist_idx, cache_valid_len + query_positions))
                qr = expand_compact_ranges(qr, query_positions)
                kr = expand_compact_ranges(kr, key_positions)
            out, _ = range_attention(
                q_full[b], full_k[b], full_v[b], qr, kr, types, backend="flex",
                query_mask=query_mask, key_mask=key_mask,
            )
            hidden[b].copy_(out)
            continue

        # Assemble compact K/V: [hist_vis | cur_img | cur_txt_valid].
        parts_k = []
        parts_v = []
        if hvis > 0:
            parts_k.append(full_k[b, :cache_valid_len][hist_idx])
            parts_v.append(full_v[b, :cache_valid_len][hist_idx])
        parts_k.append(full_k[b, cache_valid_len : cache_valid_len + img_q_len])
        parts_v.append(full_v[b, cache_valid_len : cache_valid_len + img_q_len])
        if valid > 0:
            parts_k.append(full_k[b, txt_k0 : txt_k0 + txt_q_len][txt_idx_local])
            parts_v.append(full_v[b, txt_k0 : txt_k0 + txt_q_len][txt_idx_local])
        k_seq = torch.cat(parts_k, dim=0).contiguous()
        v_seq = torch.cat(parts_v, dim=0).contiguous()

        # Assemble compact Q: [img_q | txt_q_valid].
        q_img = q_full[b, :img_q_len]
        if valid > 0:
            q_txt = q_full[b, img_q_len:][txt_idx_local]
            q_seq = torch.cat([q_img, q_txt], dim=0).contiguous()
        else:
            q_seq = q_img.contiguous()

        offs_b = txt_offsets_per_sample[b] if b < len(txt_offsets_per_sample) else None
        if offs_b is None:
            # Legacy full bidirectional attention over the visible tokens.
            n_k = k_seq.size(0)
            n_q = q_seq.size(0)
            q_ranges = torch.tensor([[0, n_q]], dtype=torch.int32, device=device)
            k_ranges = torch.tensor([[0, n_k]], dtype=torch.int32, device=device)
            attn_type_map = torch.zeros(1, dtype=torch.int32, device=device)
        else:
            offs_clamped = [min(int(o), valid) for o in offs_b]
            q_ranges, k_ranges, attn_type_map = _build_intra_seg_cache_ranges(
                hvis=hvis,
                limg=img_q_len,
                valid=valid,
                offs=offs_clamped,
                windows=vid_windows,
                device=device,
            )

        out_seq, _ = range_attention(
            backend=seg_ar_attention_backend,
            q=q_seq,
            k=k_seq,
            v=v_seq,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            softmax_scale=None,
            deterministic=_INTRA_SEG_CACHE_DETERMINISTIC,
        )

        # Scatter back: img rows dense, txt rows at their valid positions.
        hidden[b, :img_q_len] = out_seq[:img_q_len]
        if valid > 0:
            hidden[b, img_q_len:][txt_idx_local] = out_seq[img_q_len:]
    return hidden


def causal_attention_with_cache_mp(
    q,
    k,
    v,
    kv_cache: Dict[str, torch.Tensor],
    cache_valid_len: int = 0,
    attn_mode: str = None,
    text_mask: Optional[torch.Tensor] = None,
    update_cache: bool = False,
    cache_txt: bool = True,
    cache_mask: Optional[torch.Tensor] = None,
    intra_seg_txt_offsets: Optional[List[int]] = None,
    intra_seg_vid_windows: Optional[List[List[int]]] = None,
    tokens_per_frame: int = 0,
    seg_ar_attention_backend: str = "magi",
    sparse_context=None,
):
    """
    Multi-prompt causal attention with KV cache.

    Unlike Nomas_causal's version which caches only img KV (text is always the
    same single prompt), this version caches the **concatenated** img+txt KV
    from each previous segment.  Causality is implicit: only previously
    generated segments are in the cache.

    Cache layout (token-level, not frame-level):
        Within a segment, text KV is cached only once (with the first chunk),
        subsequent chunks in the same segment cache only img KV:
          [seg0_chunk0_img | seg0_txt | seg0_chunk1_img | ... | seg1_chunk0_img | seg1_txt | ...]
        This matches training where each segment has exactly one text block.

    The current chunk's Q attends to:
        - first chunk of a segment (cache_txt=True):
            [cached_history_kv] + [current_img_kv] + [current_txt_kv]
        - subsequent chunk of the same segment (cache_txt=False):
            [cached_history_kv] + [current_img_kv]
          (the segment's txt is already in cached_history_kv, so re-appending
          current_txt here would duplicate the segment's text keys vs. training)
    which is fully bidirectional (no mask needed) since all history is
    already causally correct.

    Text padding tokens are excluded from attention via unpad_input +
    flash_attn_varlen, matching the training behavior where padding never
    participates in attention (flash_attn_no_pad_v3 / magi range-based).

    Args:
        q: (img_q, txt_q) each [B, L, H, D]
        k: (img_k, txt_k) each [B, L, H, D]
        v: (img_v, txt_v) each [B, L, H, D]
        kv_cache: {"k": [B, max_cache_len, H, D], "v": [B, max_cache_len, H, D]}
        cache_valid_len: Number of valid **tokens** (not frames) in cache
        attn_mode: Attention backend ("flash3", "torch")
        text_mask: Text attention mask [B, S] for current segment's text
        update_cache: If True, append current chunk's KV to cache
        cache_txt: If True, include txt KV in cache update (should be True
            only for the first chunk of each segment to avoid duplicating
            text tokens in the cache, matching training behavior).
        cache_mask: Boolean mask [B, max_cache_len] for cached tokens.
            True=valid, False=padding. Updated in-place when update_cache=True.
    """
    if seg_ar_attention_backend not in ("magi", "flex"):
        raise ValueError("Invalid seg_ar_attention_backend")
    if seg_ar_attention_backend == "flex" and torch.is_grad_enabled():
        raise RuntimeError("Flex cached attention requires no-grad inference")
    is_sp = get_sequence_parallel_state()
    sp_group = nccl_info.sp_group

    img_q, txt_q = q
    img_k, txt_k = k
    img_v, txt_v = v
    img_output_dtype = img_q.dtype
    txt_output_dtype = txt_q.dtype

    if is_sp:
        img_q = all_to_all_4D(img_q, group=sp_group, scatter_dim=2, gather_dim=1)
        img_k = all_to_all_4D(img_k, group=sp_group, scatter_dim=2, gather_dim=1)
        img_v = all_to_all_4D(img_v, group=sp_group, scatter_dim=2, gather_dim=1)

        if txt_q.shape[2] != img_q.shape[2]:
            shard_heads = img_q.shape[2]
            rank_in_group = nccl_info.rank_within_spgroup
            start_h = rank_in_group * shard_heads
            end_h = start_h + shard_heads
            txt_q = txt_q[:, :, start_h:end_h, :]
            txt_k = txt_k[:, :, start_h:end_h, :]
            txt_v = txt_v[:, :, start_h:end_h, :]

    from .modules.kv_cache_storage import cache_format, cache_geometry

    storage_format = cache_format(kv_cache)
    qvg_storage = storage_format == "qvg"
    int8_storage = storage_format == "int8"
    cache_shape, cache_device, cache_dtype = cache_geometry(kv_cache)
    if not qvg_storage:
        if kv_cache["k"].dtype != kv_cache["v"].dtype:
            raise RuntimeError("KV cache K/V dtype mismatch")
        if kv_cache["k"].device != kv_cache["v"].device:
            raise RuntimeError("KV cache K/V device mismatch")
    elif (seg_ar_attention_backend != "flex" or is_sp or sparse_context is not None
          or not cache_txt
          or img_q.shape[0] != 1 or cache_shape[0] != 1):
        raise ValueError("QVG requires single-slot dense Flex segment attention")
    if int8_storage:
        from .modules.int8_kv_cache import dequantize_kv, quantize_kv_pair_into
    compressed_storage = int8_storage or qvg_storage
    kernel_dtype = torch.bfloat16 if compressed_storage else cache_dtype
    if (compressed_storage or "swa_capacity" in kv_cache) and torch.is_grad_enabled():
        raise ValueError("Compressed KV/SWA requires no-grad inference")
    if seg_ar_attention_backend == "flex":
        if kernel_dtype != torch.bfloat16:
            raise ValueError("Flex cached attention requires BF16 compute")
        if kv_cache.get("swa_sampling", "recent") not in HISTORY_SAMPLINGS:
            raise ValueError(f"Flex SWA sampling must be one of {HISTORY_SAMPLINGS}")
    if img_q.device != cache_device or txt_q.device != cache_device:
        raise RuntimeError(
            "Current Q and KV cache must be on the same device: "
            f"img_q={img_q.device}, txt_q={txt_q.device}, "
            f"cache={cache_device}"
        )

    # Cache history can be very large, so cast the current chunk to the cache
    # dtype instead of promoting/copying the history. This keeps cache-attention
    # precision independent from the persistent/master parameter dtype.
    img_q = img_q.to(dtype=kernel_dtype)
    txt_q = txt_q.to(dtype=kernel_dtype)
    img_k = img_k.to(dtype=kernel_dtype)
    txt_k = txt_k.to(dtype=kernel_dtype)
    img_v = img_v.to(dtype=kernel_dtype)
    txt_v = txt_v.to(dtype=kernel_dtype)

    # Current chunk's combined KV for attention (always img + txt)
    cur_k = torch.cat([img_k, txt_k], dim=1)  # [B, L_img+L_txt, H, D]
    cur_v = torch.cat([img_v, txt_v], dim=1)
    cur_tokens = cur_k.size(1)

    # KV to write into cache: img + txt (first chunk) or img only (subsequent chunks)
    if cache_txt:
        cache_k = cur_k  # [B, L_img+L_txt, H, D]
        cache_v = cur_v
    else:
        cache_k = img_k  # [B, L_img, H, D]
        cache_v = img_v
    cache_write_tokens = cache_k.size(1)

    # Build mask for current chunk's tokens: img=True, txt=text_mask
    B = img_q.size(0)
    img_token_len = img_k.size(1)
    txt_token_len = txt_k.size(1)
    cur_img_mask = torch.ones(B, img_token_len, device=img_k.device, dtype=torch.bool)
    if text_mask is not None:
        cur_txt_mask = text_mask.bool()  # [B, txt_token_len]
    else:
        cur_txt_mask = torch.ones(
            B, txt_token_len, device=img_k.device, dtype=torch.bool
        )

    # Mask for tokens to write into cache
    if cache_txt:
        cache_write_mask = torch.cat([cur_img_mask, cur_txt_mask], dim=1)
    else:
        cache_write_mask = cur_img_mask

    qvg_append = update_cache and qvg_storage and "swa_capacity" not in kv_cache
    if qvg_append:
        from .modules.qvg_kv_cache import validate_qvg_append

        validate_qvg_append(
            kv_cache, cache_mask, cache_k, cache_v, cache_write_mask,
            cache_valid_len, img_token_len,
        )

    # Both Magi and Flex consume BF16 Q/K/V. INT8 is persistent storage only:
    # decode this layer's valid history, keeping current-segment KV unquantized.
    # Sliding-window eviction still happens after attention has read old history.
    if qvg_storage:
        from .modules.qvg_kv_cache import decode_history_into

        total_len = cache_valid_len + cur_tokens
        full_k = torch.empty((B, total_len, *cache_shape[2:]),
                             dtype=torch.bfloat16, device=cache_device)
        full_v = torch.empty_like(full_k)
        decode_history_into(kv_cache, full_k, full_v, cache_valid_len)
        full_k[:, cache_valid_len:].copy_(cur_k)
        full_v[:, cache_valid_len:].copy_(cur_v)
        history_k, history_v = full_k[:, :cache_valid_len], full_v[:, :cache_valid_len]
    elif int8_storage:
        history_k = dequantize_kv(kv_cache, "k", cache_valid_len)
        history_v = dequantize_kv(kv_cache, "v", cache_valid_len)
    else:
        history_k = kv_cache["k"][:, :cache_valid_len]
        history_v = kv_cache["v"][:, :cache_valid_len]

    # History mask from cache_mask
    if cache_mask is not None and cache_valid_len > 0:
        history_mask = cache_mask[:, :cache_valid_len]  # [B, cache_valid_len]
    else:
        history_mask = torch.ones(
            history_k.shape[0],
            cache_valid_len,
            device=img_k.device,
            dtype=torch.bool,
        )

    # seg_ar CFG keeps one persistent canonical history slot while evaluating
    # the current guidance branches. Broadcast that read-only history across
    # the current batch instead of allocating one full KV slot per branch.
    # Cache writes stay one-to-one and therefore do not broadcast.
    history_batch = history_k.shape[0]
    if history_batch != B:
        if update_cache:
            raise ValueError(
                "KV-cache batch mismatch during update: "
                f"current={B}, cache={history_batch}. "
                "Read-only history broadcasting is supported, cache writes are not."
            )
        if history_batch != 1:
            raise ValueError(
                "KV-cache batch mismatch: "
                f"current={B}, cache={history_batch}; only a single canonical "
                "history slot can be broadcast."
            )
        history_k = history_k.expand(B, -1, -1, -1)
        history_v = history_v.expand(B, -1, -1, -1)
        history_mask = history_mask.expand(B, -1)

    # Segment attention profiling (AR pipeline)
    if (
        seg_ar_attention_backend == "magi"
        and _HAS_AR_VIS_PROFILER
        and _ar_profile_fa_func is not None
        and _seg_profiler_ar.should_profile()
        and cache_valid_len > 0
        and not is_sp
    ):
        _ar_scores = _compute_ar_scores(
            img_q,
            txt_q,
            img_k,
            img_v,
            txt_k,
            txt_v,
            history_k,
            history_v,
            tokens_per_frame=_seg_profiler_ar.tokens_per_frame,
            fa_func=_ar_profile_fa_func,
        )
        if _ar_scores is not None:
            _seg_profiler_ar.record(_ar_scores)

    swa_commit = update_cache and "swa_capacity" in kv_cache
    if kv_cache.get("swa_sampling", "recent") in ARCHIVED_HISTORY_SAMPLINGS and sparse_context is not None:
        raise ValueError("Sampled history requires dense attention")
    if swa_commit and (not cache_txt or cache_mask is None):
        raise ValueError("SWA requires a masked segment-AR cache write")
    if update_cache and not swa_commit and not qvg_append:
        new_end = cache_valid_len + cache_write_tokens
        if int8_storage:
            quantize_kv_pair_into(kv_cache, cache_k.detach(), cache_v.detach(), cache_valid_len)
        else:
            kv_cache["k"][:, cache_valid_len:new_end] = cache_k.detach()
            kv_cache["v"][:, cache_valid_len:new_end] = cache_v.detach()
        if cache_mask is not None:
            cache_mask[:, cache_valid_len:new_end] = cache_write_mask

    # Build full KV for attention.
    # First chunk of a segment (cache_txt=True): attend history + current img+txt.
    # Subsequent chunks (cache_txt=False): the segment's txt is already in the
    # history cache, so attend history + current img only (no txt duplication).
    if qvg_storage:
        pass  # History was decoded directly into the final attention buffers.
    elif int8_storage or swa_commit:
        # Current keys remain BF16 for their own attention. Only history is decoded.
        if cache_txt:
            total_len = cache_valid_len + cur_tokens
            full_k = torch.cat([history_k, cur_k], dim=1)
            full_v = torch.cat([history_v, cur_v], dim=1)
        elif update_cache:
            full_k = torch.cat([history_k, img_k, txt_k], dim=1)
            full_v = torch.cat([history_v, img_v, txt_v], dim=1)
        else:
            full_k = torch.cat([history_k, img_k], dim=1)
            full_v = torch.cat([history_v, img_v], dim=1)
    elif update_cache:
        if cache_txt:
            total_len = cache_valid_len + cur_tokens
            full_k = kv_cache["k"][:, :total_len]
            full_v = kv_cache["v"][:, :total_len]
        else:
            cached_all_k = kv_cache["k"][:, : cache_valid_len + cache_write_tokens]
            cached_all_v = kv_cache["v"][:, : cache_valid_len + cache_write_tokens]
            full_k = torch.cat([cached_all_k, txt_k], dim=1)
            full_v = torch.cat([cached_all_v, txt_v], dim=1)
    else:
        if cache_txt:
            # First chunk of a segment: this segment's txt is not yet in the
            # cache, so attend to it via the current chunk (img + txt).
            full_k = torch.cat([history_k, cur_k], dim=1)
            full_v = torch.cat([history_v, cur_v], dim=1)
        else:
            # Subsequent chunk of the same segment: the segment's txt is already
            # in the history cache (written by the first chunk). Re-appending the
            # current txt here would duplicate the segment's text keys in the
            # attention (train/inference mismatch — training keeps exactly one
            # text block per segment), so attend to the current IMG only.
            full_k = torch.cat([history_k, img_k], dim=1)
            full_v = torch.cat([history_v, img_v], dim=1)

    # Build full masks for Q and KV
    cur_full_mask = torch.cat([cur_img_mask, cur_txt_mask], dim=1)  # [B, img+txt]
    q_mask = cur_full_mask  # Q = current chunk's img + txt

    if swa_commit or qvg_append:
        kv_mask = torch.cat([history_mask, cur_full_mask], dim=1)
    elif update_cache:
        if cache_txt:
            if cache_mask is not None:
                kv_mask = cache_mask[:, :total_len]
            else:
                kv_mask = torch.ones(
                    B, total_len, device=img_k.device, dtype=torch.bool
                )
        else:
            cached_mask = (
                cache_mask[:, : cache_valid_len + cache_write_tokens]
                if cache_mask is not None
                else torch.ones(
                    B,
                    cache_valid_len + cache_write_tokens,
                    device=img_k.device,
                    dtype=torch.bool,
                )
            )
            kv_mask = torch.cat([cached_mask, cur_txt_mask], dim=1)
    else:
        # Mirror the KV composition above: include the current txt mask only on
        # the first chunk of a segment (cache_txt=True). On subsequent chunks the
        # segment's txt lives in the history cache, so the current chunk only
        # contributes img tokens to the attended KV.
        if cache_txt:
            kv_mask = torch.cat([history_mask, cur_full_mask], dim=1)
        else:
            kv_mask = torch.cat([history_mask, cur_img_mask], dim=1)

    # Q = current segment's img + txt
    q_full = torch.cat([img_q, txt_q], dim=1)

    img_q_len = img_q.size(1)

    # ---- Intra-segment per-action masking (current-chunk self-attention) ----
    # Only the current chunk's video<->text cross terms are restricted; the
    # visible history prefix (already cross-segment/tail masked via cache_mask)
    # and video causality are unchanged. Requires the segment's txt to live in
    # the CURRENT chunk (cache_txt=True, i.e. first/only chunk of the segment),
    # per-action offsets/windows, and the flex kernel. Once a caller supplies a
    # real payload, fail loudly instead of silently training with full attention.
    _intra_seg_requested = (
        cache_txt
        and intra_seg_vid_windows is not None
        and len(intra_seg_vid_windows) > 0
        and intra_seg_txt_offsets is not None
        and any(o is not None for o in intra_seg_txt_offsets)
    )
    if _intra_seg_requested and seg_ar_attention_backend == "magi" and _flex_flash_attn_func is None:
        raise RuntimeError(
            "intra_segment_action_mask cache payload requires "
            "magi_attention.functional.flex_flash_attn_func"
        )
    if _intra_seg_requested and attn_mode != "flash3":
        raise RuntimeError(
            "intra_segment_action_mask cache payload requires attn_mode='flash3', "
            f"got {attn_mode!r}"
        )
    _use_intra_seg = _intra_seg_requested
    # Permanently dense layers bypass pooled routing, including KV writes.
    if sparse_context is not None:
        sparse_cfg = sparse_context['config']
        if sparse_context['layer'] < int(sparse_cfg.get('dense_first_n_layers', 1)):
            stats = kv_cache.setdefault('dense_layer_stats', {'denoise': 0, 'write': 0})
            stats['write' if update_cache else 'denoise'] += 1
            sparse_context = None
    dense_sparse_read = sparse_context is not None and sparse_context.get("dense", False)
    if dense_sparse_read and update_cache:
        raise ValueError("Dense warmup reads cannot update sparse cache metadata")
    if sparse_context is not None and not dense_sparse_read:
        sparse_context = dict(sparse_context, backend=seg_ar_attention_backend)
        from .modules.attention.nomas_cache_sparse import cache_sparse_attention
        hidden = cache_sparse_attention(
            q_full, full_k, full_v, kv_mask, cur_txt_mask, kv_cache,
            cache_valid_len, update_cache, cache_txt, sparse_context,
            offsets=intra_seg_txt_offsets if _use_intra_seg else None,
            windows=intra_seg_vid_windows if _use_intra_seg else None,
        )
    elif _use_intra_seg:
        # Convert segment-relative latent-FRAME windows to TOKEN windows.
        # (img_q here is the full, SP-gathered video sequence: img_q_len ==
        # tt * tokens_per_frame.)
        tpf = (
            int(tokens_per_frame) if tokens_per_frame and tokens_per_frame > 0 else None
        )
        if tpf is None and img_q_len > 0:
            # Fallback: infer from the last window's frame extent if provided.
            tpf = 1
        vid_windows_tok = [
            [int(w[0]) * tpf, int(w[1]) * tpf, int(w[2])] for w in intra_seg_vid_windows
        ]
        _orig_dtype = q_full.dtype
        if _orig_dtype not in (torch.float16, torch.bfloat16):
            _qf = q_full.to(torch.bfloat16)
            _kf = full_k.to(torch.bfloat16)
            _vf = full_v.to(torch.bfloat16)
        else:
            _qf, _kf, _vf = q_full, full_k, full_v
        hidden = _intra_seg_cache_forward(
            q_full=_qf,
            full_k=_kf,
            full_v=_vf,
            kv_mask=kv_mask,
            cur_txt_mask=cur_txt_mask,
            cache_valid_len=cache_valid_len,
            img_q_len=img_q_len,
            txt_offsets_per_sample=intra_seg_txt_offsets,
            vid_windows=vid_windows_tok,
            seg_ar_attention_backend=seg_ar_attention_backend,
        )
        if hidden.dtype != _orig_dtype:
            hidden = hidden.to(_orig_dtype)
        del _qf, _kf, _vf
    elif attn_mode == "torch":
        q_t = q_full.transpose(1, 2)
        k_t = full_k.transpose(1, 2)
        v_t = full_v.transpose(1, 2)
        hidden = F.scaled_dot_product_attention(
            q_t, k_t, v_t,
            attn_mask=kv_mask[:, None, None, :] if (compressed_storage or "swa_capacity" in kv_cache) else None,
            dropout_p=0.0, is_causal=False
        )
        hidden = hidden.transpose(1, 2)
        del q_t, k_t, v_t
    elif attn_mode == "flash3" and seg_ar_attention_backend == "flex":
        from hymm.utils.inference_metrics import stage
        with stage("cache_attention"):
            hidden = flex_masked_attention(q_full, full_k, full_v, q_mask, kv_mask)
    elif attn_mode == "flash3":
        from .modules.attention.range_attention import note_non_range_call
        note_non_range_call()
        from flash_attn.bert_padding import pad_input, unpad_input
        from einops import rearrange as _rearrange

        if kernel_dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(
                "Flash3 KV-cache attention requires fp16/bf16 cache dtype, "
                f"got {kernel_dtype} (update_cache={update_cache}, "
                f"cache_txt={cache_txt}, cache_valid_len={cache_valid_len})"
            )
        if not q_full.dtype == full_k.dtype == full_v.dtype == kernel_dtype:
            raise RuntimeError(
                "Flash3 KV-cache Q/K/V dtype mismatch: "
                f"q={q_full.dtype}, k={full_k.dtype}, v={full_v.dtype}, "
                f"cache={kernel_dtype}, update_cache={update_cache}, "
                f"cache_txt={cache_txt}, cache_valid_len={cache_valid_len}"
            )

        H = q_full.shape[2]  # heads
        S_q = q_full.shape[1]

        # Unpad Q
        q_flat = _rearrange(q_full, "b s h d -> b s (h d)")
        q_unpad, q_indices, cu_seqlens_q, max_seqlen_q, _ = unpad_input(q_flat, q_mask)
        q_unpad = _rearrange(q_unpad, "nnz (h d) -> nnz h d", h=H)

        # Unpad K, V
        k_flat = _rearrange(full_k, "b s h d -> b s (h d)")
        v_flat = _rearrange(full_v, "b s h d -> b s (h d)")
        k_unpad, _, cu_seqlens_k, max_seqlen_k, _ = unpad_input(k_flat, kv_mask)
        v_unpad, _, _, _, _ = unpad_input(v_flat, kv_mask)
        k_unpad = _rearrange(k_unpad, "nnz (h d) -> nnz h d", h=H)
        v_unpad = _rearrange(v_unpad, "nnz (h d) -> nnz h d", h=H)

        from flash_attn_interface import flash_attn_varlen_func as _flash_attn_varlen

        out_unpad = _flash_attn_varlen(
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal=False,
            deterministic=_FLASH_CACHE_DETERMINISTIC,
        )

        # Re-pad output to [B, S_q, H, D]
        hidden = pad_input(
            _rearrange(out_unpad, "nnz h d -> nnz (h d)"),
            q_indices,
            B,
            S_q,
        )
        hidden = _rearrange(hidden, "b s (h d) -> b s h d", h=H)
    else:
        raise ValueError(f"Unsupported attn_mode for KV cache: {attn_mode}")

    if qvg_storage:
        # Drop all decoded-history aliases before allocating encoder workspace.
        del full_k, full_v, history_k, history_v, q_full
    if dense_sparse_read and seg_ar_attention_backend == "flex":
        from .modules.attention.nomas_cache_sparse import note_dense_flex
        note_dense_flex()

    if swa_commit:
        if kv_cache.get("swa_sampling", "recent") in ARCHIVED_HISTORY_SAMPLINGS:
            from .modules.uniform_history_cache import commit_uniform_history
            commit_uniform_history(
                kv_cache, cache_mask, cache_k.detach(), cache_v.detach(),
                cache_write_mask, img_token_len, txt_token_len, tokens_per_frame,
            )
        else:
            from .modules.swa_cache import append_dense_record, commit_window
            if sparse_context is None:
                append_dense_record(kv_cache, cache_valid_len, img_token_len, txt_token_len, tokens_per_frame)
            commit_window(kv_cache, cache_mask, cache_k.detach(), cache_v.detach(),
                          cache_write_mask, cache_valid_len)
    elif qvg_append:
        from .modules.qvg_kv_cache import commit_qvg_append

        commit_qvg_append(
            kv_cache, cache_mask, cache_k.detach(), cache_v.detach(),
            cache_write_mask, cache_valid_len, img_token_len,
        )

    img_attn = hidden[:, :img_q_len]
    txt_attn = hidden[:, img_q_len:]

    if is_sp:
        img_attn = all_to_all_4D(
            img_attn.contiguous(), group=sp_group, scatter_dim=1, gather_dim=2
        )

        txt_attn = all_gather(txt_attn.contiguous(), dim=2, group=sp_group)

    # Preserve the block-facing dtype expected by output projections and
    # residual paths even though the cache attention kernel runs in cache dtype.
    if img_attn.dtype != img_output_dtype:
        img_attn = img_attn.to(dtype=img_output_dtype)
    if txt_attn.dtype != txt_output_dtype:
        txt_attn = txt_attn.to(dtype=txt_output_dtype)

    img_attn = img_attn.reshape(img_attn.size(0), img_attn.size(1), -1)
    txt_attn = txt_attn.reshape(txt_attn.size(0), txt_attn.size(1), -1)

    return img_attn, txt_attn


class MMDoubleStreamBlock(nn.Module):
    """
    A multimodal dit block with seperate modulation for
    text and image/video, see more details (SD3): https://arxiv.org/abs/2403.03206
                                     (Flux.1): https://github.com/black-forest-labs/flux
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        mlp_act_type: str = "gelu_tanh",
        attn_mode: str = None,
        moe_config: Optional[MoeConfig] = None,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qkv_bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.seg_ar_attention_backend = "magi"

        self.deterministic = False
        self.heads_num = heads_num
        self.attn_mode = attn_mode
        self.moe_config = moe_config

        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.img_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.img_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.img_attn_q = nn.Linear(
            hidden_size, hidden_size * 1, bias=qkv_bias, **factory_kwargs
        )

        self.img_attn_k = nn.Linear(
            hidden_size, hidden_size * 1, bias=qkv_bias, **factory_kwargs
        )

        self.img_attn_v = nn.Linear(
            hidden_size, hidden_size * 1, bias=qkv_bias, **factory_kwargs
        )

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.img_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.img_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.img_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs
        )

        self.img_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.img_mlp = (
            MLP(
                hidden_size,
                mlp_hidden_dim,
                act_layer=get_activation_layer(mlp_act_type),
                bias=True,
                **factory_kwargs,
            )
            if moe_config is None
            else HunYuanMoE(moe_config)
        )

        self.txt_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.txt_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.txt_attn_q = nn.Linear(
            hidden_size, hidden_size * 1, bias=qkv_bias, **factory_kwargs
        )

        self.txt_attn_k = nn.Linear(
            hidden_size, hidden_size * 1, bias=qkv_bias, **factory_kwargs
        )

        self.txt_attn_v = nn.Linear(
            hidden_size, hidden_size * 1, bias=qkv_bias, **factory_kwargs
        )
        self.txt_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.txt_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.txt_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs
        )

        self.txt_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.txt_mlp = (
            MLP(
                hidden_size,
                mlp_hidden_dim,
                act_layer=get_activation_layer(mlp_act_type),
                bias=True,
                **factory_kwargs,
            )
            if moe_config is None
            else HunYuanMoE(moe_config)
        )
        self.hybrid_seq_parallel_attn = None

        if moe_config is not None:
            self.loss_moe = 0.0

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def get_moe_loss(self):
        return self.loss_moe

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        vec_txt: torch.Tensor,
        freqs_cis: tuple = None,
        text_mask=None,
        attn_param=None,
        is_flash=False,
        txt_tokens_per_block: int = None,
        prompt_video_boundaries: Optional[List[int]] = None,
        # KV cache params (FSDP2-safe delegation from forward_with_cache)
        kv_cache: Dict[str, torch.Tensor] = None,
        cache_valid_len: int = 0,
        update_cache: bool = False,
        cache_attn_mode: str = None,
        cache_txt: bool = True,
        cache_mask: Optional[torch.Tensor] = None,
        intra_seg_txt_offsets: Optional[List[int]] = None,
        intra_seg_vid_windows: Optional[List[List[int]]] = None,
        intra_seg_tokens_per_frame: int = 0,
        sparse_context=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with support for both 2D vec [B, C] and 3D vec [B, num_prompts, C].

        Args:
            img: Image hidden states [B, L, D]
            txt: Text hidden states [B, S, D]
            vec: Modulation vector, [B, C] towards image modulation
            vec_txt: Modulation vector, [B, C] or [B, num_prompts, C] towards text modulation
            freqs_cis: Rotary position embeddings
            text_mask: Text attention mask [B, S]
            attn_param: Attention parameters
            is_flash: Force flash attention mode
            txt_tokens_per_block: Number of text tokens per block
            kv_cache: KV cache dict for autoregressive generation (FSDP2-safe delegation)
            cache_valid_len: Number of valid cached tokens
            update_cache: Whether to update the cache
            cache_attn_mode: Attention mode override for cache path
            cache_txt: Whether to cache text tokens
        """

        # ========== KV cache delegation (FSDP2-safe) ==========
        # When kv_cache is provided, delegate to forward_with_cache logic.
        # This is called via block(..., kv_cache=...) from the model-level
        # forward_with_cache, ensuring FSDP2 hooks fire at the block level.
        if kv_cache is not None:
            return self.forward_with_cache(
                img=img,
                txt=txt,
                vec=vec,
                vec_txt=vec_txt,
                freqs_cis=freqs_cis,
                text_mask=text_mask,
                kv_cache=kv_cache,
                cache_valid_len=cache_valid_len,
                update_cache=update_cache,
                attn_mode=cache_attn_mode,
                cache_txt=cache_txt,
                cache_mask=cache_mask,
                intra_seg_txt_offsets=intra_seg_txt_offsets,
                intra_seg_vid_windows=intra_seg_vid_windows,
                intra_seg_tokens_per_frame=intra_seg_tokens_per_frame,
                sparse_context=sparse_context,
            )

        vec_is_per_token = vec.dim() == 3  # [B, L_img, C]
        vec_txt_is_per_token = vec_txt.dim() == 3  # [B, L_txt, C]

        if vec_is_per_token:
            (
                img_mod1_shift,
                img_mod1_scale,
                img_mod1_gate,
                img_mod2_shift,
                img_mod2_scale,
                img_mod2_gate,
            ) = self.img_mod(vec).chunk(6, dim=-1)

            (
                txt_mod1_shift,
                txt_mod1_scale,
                txt_mod1_gate,
                txt_mod2_shift,
                txt_mod2_scale,
                txt_mod2_gate,
            ) = self.txt_mod(vec_txt).chunk(6, dim=-1)

            # Prepare image for attention.
            img_modulated = self.img_norm1(img)
            # Per-token modulation: shift/scale are 3D [B, L, C], apply directly
            img_modulated = img_modulated * (1 + img_mod1_scale) + img_mod1_shift

            # txt modulation: per-token if vec_txt is 3D, otherwise standard
            txt_modulated = self.txt_norm1(txt)
            if vec_txt_is_per_token:
                txt_modulated = txt_modulated * (1 + txt_mod1_scale) + txt_mod1_shift
            else:
                txt_modulated = modulate(
                    txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale
                )
        else:
            (
                img_mod1_shift,
                img_mod1_scale,
                img_mod1_gate,
                img_mod2_shift,
                img_mod2_scale,
                img_mod2_gate,
            ) = self.img_mod(vec).chunk(6, dim=-1)

            # Text modulation: always uses vec(t)
            (
                txt_mod1_shift,
                txt_mod1_scale,
                txt_mod1_gate,
                txt_mod2_shift,
                txt_mod2_scale,
                txt_mod2_gate,
            ) = self.txt_mod(vec_txt).chunk(6, dim=-1)

            # Prepare image for attention.
            img_modulated = self.img_norm1(img)
            img_modulated = modulate(
                img_modulated, shift=img_mod1_shift, scale=img_mod1_scale
            )

            # Prepare txt for attention.
            txt_modulated = self.txt_norm1(txt)
            txt_modulated = modulate(
                txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale
            )

        img_q = self.img_attn_q(img_modulated)
        img_k = self.img_attn_k(img_modulated)
        img_v = self.img_attn_v(img_modulated)

        img_q = rearrange(img_q, "B L (H D) -> B L H D", H=self.heads_num)
        img_k = rearrange(img_k, "B L (H D) -> B L H D", H=self.heads_num)
        img_v = rearrange(img_v, "B L (H D) -> B L H D", H=self.heads_num)

        # Apply QK-Norm if needed
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)

        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert img_qq.shape == img_q.shape and img_kk.shape == img_k.shape, (
                f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            )
            img_q, img_k = img_qq, img_kk

        # Prepare txt for attention.
        txt_q = self.txt_attn_q(txt_modulated)
        txt_k = self.txt_attn_k(txt_modulated)
        txt_v = self.txt_attn_v(txt_modulated)

        txt_q = rearrange(txt_q, "B L (H D) -> B L H D", H=self.heads_num)
        txt_k = rearrange(txt_k, "B L (H D) -> B L H D", H=self.heads_num)
        txt_v = rearrange(txt_v, "B L (H D) -> B L H D", H=self.heads_num)
        # Apply QK-Norm if needed.
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

        # Determine attention mode with override support for teacher forcing
        if is_flash:
            attn_mode = "flash3"
        elif attn_param is not None and "attn_mode_override" in attn_param:
            attn_mode = attn_param["attn_mode_override"]
        else:
            attn_mode = self.attn_mode

        attn_param_local = attn_param
        if prompt_video_boundaries is not None:
            if attn_param_local is None:
                attn_param_local = {}
            else:
                attn_param_local = dict[Any, Any](attn_param_local)
            attn_param_local["prompt_video_boundaries"] = prompt_video_boundaries
            attn_param_local["num_prompts"] = len(prompt_video_boundaries) - 1
            if txt_tokens_per_block is not None:
                attn_param_local["txt_tokens_per_block"] = txt_tokens_per_block

        attn = parallel_attention(
            (img_q, txt_q),
            (img_k, txt_k),
            (img_v, txt_v),
            img_q_len=img_q.shape[1],
            img_kv_len=img_k.shape[1],
            text_mask=text_mask,
            attn_mode=attn_mode,
            attn_param=attn_param_local,
        )

        img_attn, txt_attn = (
            attn[:, : img_q.shape[1]].contiguous(),
            attn[:, img_q.shape[1] :].contiguous(),
        )

        # Norms gather shifts/scales' full token layout, while residual gates
        # follow the sequence-sharded outputs of the row projections.
        img_mod1_gate = local_modulation_gate(self, img_mod1_gate)
        img_mod2_gate = local_modulation_gate(self, img_mod2_gate)
        txt_mod1_gate = local_modulation_gate(self, txt_mod1_gate)
        txt_mod2_gate = local_modulation_gate(self, txt_mod2_gate)

        if vec_is_per_token:
            # Per-token modulation: gate/shift/scale are 3D [B, L, C].
            # Do NOT use modulate()/apply_gate() — they unsqueeze(1), which would
            # turn the tensors into 4D and break downstream rearrange/attention.
            img = img + self.img_attn_proj(img_attn) * img_mod1_gate
            img_mlp_input = self.img_norm2(img) * (1 + img_mod2_scale) + img_mod2_shift
            img = img + self.img_mlp(img_mlp_input) * img_mod2_gate
        else:
            # Calculate the img bloks.
            img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
            img = img + apply_gate(
                self.img_mlp(
                    modulate(
                        self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale
                    )
                ),
                gate=img_mod2_gate,
            )

        def _run_txt_mlp(_input):
            _fixed_m = int(_os.environ.get("HY_TEXT_MLP_FC2_FIXED_M", "0") or 0)
            if _fixed_m <= 0 or getattr(self, "tensor_parallel_size", 1) > 1:
                # TP reduce-scatter partitions the complete text sequence once.
                return self.txt_mlp(_input)
            _fc1 = self.txt_mlp.fc1(_input)
            _act = self.txt_mlp.act(_fc1)
            _hidden = self.txt_mlp.drop1(_act)
            _hidden = self.txt_mlp.norm(_hidden)
            _fc2 = fixed_m_linear(_hidden, self.txt_mlp.fc2, _fixed_m)
            return self.txt_mlp.drop2(_fc2)

        # Calculate the txt bloks.
        if vec_txt_is_per_token:
            # Per-text-token modulation: shift/scale/gate are 3D [B, L_txt, C]
            txt_attn_proj_out = self.txt_attn_proj(txt_attn)
            txt = txt + txt_attn_proj_out * txt_mod1_gate
            txt_mlp_input = self.txt_norm2(txt) * (1 + txt_mod2_scale) + txt_mod2_shift
            txt_mlp_out = _run_txt_mlp(txt_mlp_input)
            txt = txt + txt_mlp_out * txt_mod2_gate
        else:
            txt_attn_proj_out = self.txt_attn_proj(txt_attn)
            txt = txt + apply_gate(txt_attn_proj_out, gate=txt_mod1_gate)
            txt_mlp_input = modulate(
                self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale
            )
            txt_mlp_out = _run_txt_mlp(txt_mlp_input)
            txt = txt + apply_gate(txt_mlp_out, gate=txt_mod2_gate)

        if self.moe_config is not None:
            self.loss_moe = self.img_mlp.get_moe_loss() + self.txt_mlp.get_moe_loss()

        return img, txt

    def forward_with_cache(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        vec_txt: torch.Tensor = None,
        freqs_cis: tuple = None,
        text_mask=None,
        kv_cache: Dict[str, torch.Tensor] = None,
        cache_valid_len: int = 0,
        update_cache: bool = False,
        attn_mode: str = None,
        cache_txt: bool = True,
        cache_mask: Optional[torch.Tensor] = None,
        intra_seg_txt_offsets: Optional[List[int]] = None,
        intra_seg_vid_windows: Optional[List[List[int]]] = None,
        intra_seg_tokens_per_frame: int = 0,
        sparse_context=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward with KV cache for segment-level autoregressive generation.

        vec must be [B, C] (single timestep per segment).
        txt is [B, S, C] for the current segment's text only.

        Args:
            cache_txt: If True, include txt KV in cache update. Should be True
                only for the first chunk of each segment to avoid duplicating
                text tokens in the cache (matching training behavior).
        """
        assert kv_cache is not None
        assert vec.dim() == 2, f"vec must be [B, C] in cache mode, got {vec.shape}"
        if vec_txt is None:
            vec_txt = vec

        if getattr(self, "compile_cache_prepost", False):
            if torch.is_grad_enabled() or self.training:
                raise ValueError("Compiled cache blocks require frozen no-grad inference")
            self._compile_cache_calls = getattr(self, "_compile_cache_calls", 0) + 1
            if not isinstance(freqs_cis, tuple) or len(freqs_cis) != 2:
                raise ValueError("torch_compile requires tuple cos/sin RoPE")
            if not all(
                isinstance(norm, nn.LayerNorm)
                and not norm.elementwise_affine
                and norm.eps == 1e-6
                for norm in (self.img_norm1, self.img_norm2, self.txt_norm1, self.txt_norm2)
            ):
                raise ValueError("torch_compile requires affine-free LayerNorm")
            if not all(
                isinstance(mod, ModulateDiT) and isinstance(mod.act, nn.SiLU)
                for mod in (self.img_mod, self.txt_mod)
            ):
                raise ValueError("torch_compile requires SiLU modulation")
            if not all(isinstance(norm, RMSNorm) and hasattr(norm, "weight") and norm.eps == 1e-6 for norm in (
                self.img_attn_q_norm, self.img_attn_k_norm,
                self.txt_attn_q_norm, self.txt_attn_k_norm,
            )):
                raise ValueError("torch_compile requires affine QK RMSNorm")
            (img_q, img_k, img_v, txt_q, txt_k, txt_v,
             img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate,
             txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate) = (
                compiled_pre_attention()(
                    img, txt, vec, vec_txt, freqs_cis[0], freqs_cis[1],
                    linear_weight(self.img_mod.linear), self.img_mod.linear.bias,
                    linear_weight(self.txt_mod.linear), self.txt_mod.linear.bias,
                    linear_weight(self.img_attn_q), self.img_attn_q.bias,
                    linear_weight(self.img_attn_k), self.img_attn_k.bias,
                    linear_weight(self.img_attn_v), self.img_attn_v.bias,
                    linear_weight(self.txt_attn_q), self.txt_attn_q.bias,
                    linear_weight(self.txt_attn_k), self.txt_attn_k.bias,
                    linear_weight(self.txt_attn_v), self.txt_attn_v.bias,
                    self.img_attn_q_norm.weight, self.img_attn_k_norm.weight,
                    self.txt_attn_q_norm.weight, self.txt_attn_k_norm.weight,
                    self.heads_num, self.img_attn_q_norm.eps,
                )
            )
        else:
            (img_mod1_shift, img_mod1_scale, img_mod1_gate,
             img_mod2_shift, img_mod2_scale, img_mod2_gate) = self.img_mod(vec).chunk(6, dim=-1)
            (txt_mod1_shift, txt_mod1_scale, txt_mod1_gate,
             txt_mod2_shift, txt_mod2_scale, txt_mod2_gate) = self.txt_mod(vec_txt).chunk(6, dim=-1)

            img_modulated = modulate(
                self.img_norm1(img), shift=img_mod1_shift, scale=img_mod1_scale
            )
            img_q = rearrange(self.img_attn_q(img_modulated), "B L (H D) -> B L H D", H=self.heads_num)
            img_k = rearrange(self.img_attn_k(img_modulated), "B L (H D) -> B L H D", H=self.heads_num)
            img_v = rearrange(self.img_attn_v(img_modulated), "B L (H D) -> B L H D", H=self.heads_num)
            img_q = self.img_attn_q_norm(img_q).to(img_v)
            img_k = self.img_attn_k_norm(img_k).to(img_v)
            if freqs_cis is not None:
                img_q, img_k = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)

            txt_modulated = modulate(
                self.txt_norm1(txt), shift=txt_mod1_shift, scale=txt_mod1_scale
            )
            txt_q = rearrange(self.txt_attn_q(txt_modulated), "B L (H D) -> B L H D", H=self.heads_num)
            txt_k = rearrange(self.txt_attn_k(txt_modulated), "B L (H D) -> B L H D", H=self.heads_num)
            txt_v = rearrange(self.txt_attn_v(txt_modulated), "B L (H D) -> B L H D", H=self.heads_num)
            txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
            txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

        # Attention and communication stay outside compiled graphs.
        img_attn, txt_attn = causal_attention_with_cache_mp(
            (img_q, txt_q),
            (img_k, txt_k),
            (img_v, txt_v),
            kv_cache=kv_cache,
            cache_valid_len=cache_valid_len,
            attn_mode=attn_mode or self.attn_mode,
            text_mask=text_mask,
            update_cache=update_cache,
            cache_txt=cache_txt,
            cache_mask=cache_mask,
            intra_seg_txt_offsets=intra_seg_txt_offsets,
            intra_seg_vid_windows=intra_seg_vid_windows,
            tokens_per_frame=intra_seg_tokens_per_frame,
            seg_ar_attention_backend=self.seg_ar_attention_backend,
            sparse_context=sparse_context,
        )

        if getattr(self, "compile_cache_prepost", False):
            if not all(
                isinstance(mlp, MLP)
                and isinstance(mlp.act, nn.GELU)
                and mlp.act.approximate == "tanh"
                and isinstance(mlp.norm, nn.Identity)
                and isinstance(mlp.fc1, (nn.Linear, Int8StoredBF16Linear))
                and isinstance(mlp.fc2, (nn.Linear, Int8StoredBF16Linear))
                and mlp.drop1.p == 0
                and mlp.drop2.p == 0
                for mlp in (self.img_mlp, self.txt_mlp)
            ):
                raise ValueError("torch_compile requires tanh GELU MLP blocks")
            img, txt = compiled_post_attention()(
                img, txt, img_attn, txt_attn,
                img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate,
                txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate,
                linear_weight(self.img_attn_proj), self.img_attn_proj.bias,
                linear_weight(self.txt_attn_proj), self.txt_attn_proj.bias,
                linear_weight(self.img_mlp.fc1), self.img_mlp.fc1.bias,
                linear_weight(self.img_mlp.fc2), self.img_mlp.fc2.bias,
                linear_weight(self.txt_mlp.fc1), self.txt_mlp.fc1.bias,
                linear_weight(self.txt_mlp.fc2), self.txt_mlp.fc2.bias,
            )
        else:
            img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
            img = img + apply_gate(
                self.img_mlp(
                    modulate(self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale)
                ),
                gate=img_mod2_gate,
            )

            txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
            txt = txt + apply_gate(
                self.txt_mlp(
                    modulate(self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale)
                ),
                gate=txt_mod2_gate,
            )

        return img, txt


class MMSingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    Also refer to (SD3): https://arxiv.org/abs/2403.03206
                  (Flux.1): https://github.com/black-forest-labs/flux
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        attn_mode: str = None,
        moe_config: Optional[MoeConfig] = None,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qk_scale: float = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.attn_mode = attn_mode
        self.moe_config = moe_config

        self.hidden_size = hidden_size
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.scale = qk_scale or head_dim**-0.5

        self.linear1_q = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.linear1_k = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.linear1_v = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        if moe_config is None:
            self.linear1_mlp = nn.Linear(hidden_size, mlp_hidden_dim, **factory_kwargs)
            self.linear2 = LinearWarpforSingle(
                hidden_size + mlp_hidden_dim, hidden_size, bias=True, **factory_kwargs
            )
            self.mlp_act = get_activation_layer(mlp_act_type)()
        else:
            self.mlp = HunYuanMoE(moe_config, "HunYuanMLP2")
            self.loss_moe = None

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )

        self.pre_norm = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.modulation = ModulateDiT(
            hidden_size,
            factor=3,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.hybrid_seq_parallel_attn = None

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def get_moe_loss(self):
        return self.loss_moe

    def forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        txt_len: int,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
        text_mask=None,
        attn_param=None,
        is_flash=False,
    ) -> torch.Tensor:
        mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)
        x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale)

        q = self.linear1_q(x_mod)
        k = self.linear1_k(x_mod)
        v = self.linear1_v(x_mod)

        q = rearrange(q, "B L (H D) -> B L H D", H=self.heads_num)
        k = rearrange(k, "B L (H D) -> B L H D", H=self.heads_num)
        v = rearrange(v, "B L (H D) -> B L H D", H=self.heads_num)

        if self.moe_config is None:
            mlp = self.linear1_mlp(x_mod)

        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
        img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
        img_v, txt_v = v[:, :-txt_len, :, :], v[:, -txt_len:, :, :]
        img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
        assert img_qq.shape == img_q.shape and img_kk.shape == img_k.shape, (
            f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
        )
        img_q, img_k = img_qq, img_kk

        if is_flash:
            attn_mode = "flash3"
        else:
            attn_mode = self.attn_mode
        attn = parallel_attention(
            (img_q, txt_q),
            (img_k, txt_k),
            (img_v, txt_v),
            img_q_len=img_q.shape[1],
            img_kv_len=img_k.shape[1],
            text_mask=text_mask,
            attn_mode=attn_mode,
            attn_param=attn_param,
        )
        if self.moe_config is None:
            output = self.linear2(attn, self.mlp_act(mlp))
        else:
            output = self.mlp(x_mod, attn)
            self.loss_moe = self.mlp.get_moe_loss()

        return x + apply_gate(output, gate=local_modulation_gate(self, mod_gate))


class CPTHYNomasDiffusionTransformer(ModelMixin, ConfigMixin):
    """
    HY-Nomas Transformer backbone for bidirectional CPT with per-chunk text modulation.

    Combines Nomas.py's bidirectional training (single blocks enabled, no causal attention)
    with Nomas_ar.py's per-chunk text modulation (3D vec, per-block text refiner).

    Key differences from Nomas.py:
    - Uses SingleTokenRefinerPerBlock (supports [B,T] timestep -> 4D text output)
    - Uses FinalLayerPerBlock (supports per-block final modulation)
    - Uses Nomas_ar.py's MMDoubleStreamBlock (supports 3D vec with per-block modulation)
    - Expands scalar timestep [B] -> [B, num_chunks] for segment text

    Key differences from Nomas_ar.py:
    - Scalar timestep [B] (not per-chunk [B,T])
    - Bidirectional attention (no causal override, no teacher_forcing/diffusion_forcing logic)
    - Single blocks ENABLED
    - Standard RoPE (no teacher forcing temporal positions)

    Checkpoint-compatible with Nomas.py and Nomas_ar.py (same submodule architecture).

    Reference:
    [1] Flux.1: https://github.com/black-forest-labs/flux
    [2] MMDiT: http://arxiv.org/abs/2403.03206
    """

    @register_to_config
    def __init__(
        self,
        patch_size: list = [1, 2, 2],
        in_channels: int = 4,
        out_channels: int = None,
        hidden_size: int = 3072,
        heads_num: int = 24,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        mm_double_blocks_depth: int = 20,
        mm_single_blocks_depth: int = 40,
        rope_dim_list: List[int] = [16, 56, 56],
        qkv_bias: bool = True,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        guidance_embed: bool = False,
        text_projection: str = "single_refiner",
        use_attention_mask: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        text_states_dim: int = 4096,
        rope_theta: int = 256,
        attn_mode: str = "magi_mp_causal",
        win_size: List[int] = [3, 3, 3],
        tile_size: List[int] = [8, 4, 4],
        win_type: str = "fix",
        win_ratio: int = 9,
        ba_attn_chunk_size: int = 512,
        ba_attn_topk: int = 3,
        ba_attn_causal: bool = False,
        ba_attn_slice_method: str = "3D",
        ba_attn_full_attn_block_num: int = 0,
        ba_attn_full_attn_every_n_blocks: int = 0,
        ssta_topk: int = 0,
        ssta_threshold: float = 0.0,
        ssta_enforce_share_topk_block: int = 0,
        ssta_lambda: float = 0.9,
        ssta_sampling_type: str = "similarity",
        ssta_adaptive_pool: int = 0,
        ptm_attn_sparse_type: str = "none",
        ptm_attn_pad_type: str = "repeat",
        ptm_attn_use_text_mask: int = 1,
        ptm_attn_mask_share_within_head: int = 1,
        df_magi_topk: int = 0,
        df_magi_threshold: float = 0.0,
        df_magi_share_within_head: int = 1,
        df_magi_causal: int = 1,
        df_avg_text_tokens: int = 0,
        moe_config: MoeConfig = None,
        is_reshape_temporal_channels=False,
        unpad_zero_tokens: bool = True,
        # Text segment-level learnable positional embedding (Plan A: fixed
        # max). When > 0, SingleTokenRefinerPerBlock will create an
        # nn.Embedding(text_seg_pos_emb_num, hidden_size) whose lookup is
        # broadcast-added to all tokens of each segment after input_embedder.
        # Set to 0 (default) to keep legacy behavior.
        text_seg_pos_emb_num: int = 0,
        # Camera control
        camera_injection_mode: Optional[
            str
        ] = None,  # null | "action_attn" | "extrinsics_attn" | "residual_norm"
        action_config: Optional[dict] = None,
        seg_ar_attention_backend: str = "magi",
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if seg_ar_attention_backend not in ("magi", "flex"):
            raise ValueError("Invalid seg_ar_attention_backend")
        self.seg_ar_attention_backend = seg_ar_attention_backend


        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.unpatchify_channels = self.out_channels
        self.guidance_embed = guidance_embed
        self.rope_dim_list = rope_dim_list
        self.rope_theta = rope_theta
        self.use_attention_mask = use_attention_mask
        self.text_projection = text_projection
        self.moe_config = moe_config
        self.attn_mode = attn_mode
        self.text_states_dim = text_states_dim
        self.camera_injection_mode = camera_injection_mode

        assert attn_mode != "flash3", (
            f"Multi-segment text is incompatible with attn_mode='{attn_mode}'. "
            f"flash3 performs full bidirectional attention without segment-level masking, "
            f"which defeats the purpose of per-chunk text modulation. "
            f"Use 'magi_mp_causal' or another segment-aware attention mode instead."
        )

        if hidden_size % heads_num != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by heads_num {heads_num}"
            )
        pe_dim = hidden_size // heads_num
        if sum(rope_dim_list) != pe_dim:
            raise ValueError(
                f"Got {rope_dim_list} but expected positional dim {pe_dim}"
            )
        self.hidden_size = hidden_size
        self.heads_num = heads_num

        # image projection
        self.img_in = PatchEmbed(
            self.patch_size,
            self.in_channels,
            self.hidden_size,
            is_reshape_temporal_channels=is_reshape_temporal_channels,
            **factory_kwargs,
        )

        # text projection - use PerBlock version to support [B,T] timestep -> 4D text
        if self.text_projection == "linear":
            self.txt_in = TextProjection(
                text_states_dim,
                self.hidden_size,
                get_activation_layer("silu"),
                **factory_kwargs,
            )
        elif self.text_projection == "single_refiner":
            self.txt_in = SingleTokenRefinerPerBlock(
                text_states_dim,
                hidden_size,
                heads_num,
                depth=2,
                seg_pos_emb_num=text_seg_pos_emb_num,
                **factory_kwargs,
            )
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        # time modulation
        self.time_in = TimestepEmbedder(
            self.hidden_size, get_activation_layer("silu"), **factory_kwargs
        )

        # guidance modulation
        self.guidance_in = (
            TimestepEmbedder(
                self.hidden_size, get_activation_layer("silu"), **factory_kwargs
            )
            if guidance_embed
            else None
        )

        # Double stream blocks (from Nomas_ar.py, support both 2D and 3D vec)
        self.double_blocks = nn.ModuleList(
            [
                MMDoubleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    attn_mode=attn_mode,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    moe_config=moe_config,
                    **factory_kwargs,
                )
                for _ in range(mm_double_blocks_depth)
            ]
        )

        # Single stream blocks (from Nomas_ar.py, ENABLED unlike Nomas_ar.py)
        self.single_blocks = nn.ModuleList(
            [
                MMSingleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    attn_mode=attn_mode,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    moe_config=moe_config,
                    **factory_kwargs,
                )
                for _ in range(mm_single_blocks_depth)
            ]
        )

        # Final layer (PerBlock version, supports both 2D and 3D condition)
        self.final_layer = FinalLayerPerBlock(
            self.hidden_size,
            self.patch_size,
            self.out_channels,
            get_activation_layer("silu"),
            **factory_kwargs,
        )

        for block in self.double_blocks:
            block.seg_ar_attention_backend = seg_ar_attention_backend

        # Attention parameters
        self.attn_param = {
            "seg_ar_attention_backend": seg_ar_attention_backend,
            "flex_mask_cache": {},
            "win_size": [win_size],
            "win_type": win_type,
            "win_ratio": win_ratio,
            "tile_size": tile_size,
            "ssta_topk": ssta_topk,
            "ssta_threshold": ssta_threshold,
            "ssta_enforce_share_topk_block": ssta_enforce_share_topk_block,
            "ssta_lambda": ssta_lambda,
            "ssta_sampling_type": ssta_sampling_type,
            "ssta_adaptive_pool": ssta_adaptive_pool,
            "ba_attn_chunk_size": ba_attn_chunk_size,
            "ba_attn_topk": ba_attn_topk,
            "ba_attn_causal": ba_attn_causal,
            "ba_attn_slice_method": ba_attn_slice_method,
            "ba_attn_full_attn_block_num": ba_attn_full_attn_block_num,
            "ba_attn_full_attn_every_n_blocks": ba_attn_full_attn_every_n_blocks,
            "ptm_attn_sparse_type": ptm_attn_sparse_type,
            "ptm_attn_pad_type": ptm_attn_pad_type,
            "ptm_attn_use_text_mask": ptm_attn_use_text_mask,
            "ptm_attn_mask_share_within_head": ptm_attn_mask_share_within_head,
            "df_magi_topk": df_magi_topk,
            "df_magi_threshold": df_magi_threshold,
            "df_magi_share_within_head": df_magi_share_within_head,
            "df_magi_causal": df_magi_causal,
            "df_avg_text_tokens": df_avg_text_tokens,
            "unpad_zero_tokens": unpad_zero_tokens,
        }

        # ======================== Camera Control Modules ========================
        if action_config is not None:
            action_config = dict(action_config, seg_ar_attention_backend=seg_ar_attention_backend)
        self._init_camera_modules(camera_injection_mode, action_config, factory_kwargs)

    def _init_camera_modules(
        self, camera_injection_mode, action_config, factory_kwargs
    ):
        """Initialize camera injection modules based on camera_injection_mode."""
        if camera_injection_mode is None:
            return

        from hymm.models.camera.camera_modules import CameraEmbed
        from ..dit.modules.embed_layers import TimestepEmbedder
        from ..dit.modules.activation_layers import get_activation_layer

        valid_modes = ("action_attn", "extrinsics_attn", "residual_norm")
        assert camera_injection_mode in valid_modes, (
            f"camera_injection_mode must be one of {valid_modes}, got '{camera_injection_mode}'"
        )

        # Pose-conditioning modules are only part of the residual_norm path.
        if camera_injection_mode == "residual_norm":
            self.action_in = TimestepEmbedder(
                self.hidden_size, get_activation_layer("silu")
            )
            nn.init.zeros_(self.action_in.mlp[2].weight)
            if self.action_in.mlp[2].bias is not None:
                nn.init.zeros_(self.action_in.mlp[2].bias)
            self.camera_embed = CameraEmbed(13, self.hidden_size)

        if camera_injection_mode == "action_attn":
            # ActionModule per block (cross-attention on raw action_onehot)
            from hymm.models.camera.action_module import ActionModule

            assert action_config is not None, (
                "action_config required for action_attn mode"
            )
            self.action_modules = nn.ModuleList(
                [ActionModule(**action_config) for _ in range(len(self.double_blocks))]
            )

        elif camera_injection_mode == "extrinsics_attn":
            # ActionModule per block (cross-attention on extrinsics embedding).
            from hymm.models.camera.action_module import ActionModule

            assert action_config is not None, (
                "action_config required for extrinsics_attn mode"
            )
            self.action_modules = nn.ModuleList(
                [ActionModule(**action_config) for _ in range(len(self.double_blocks))]
            )
            # Extrinsics embedding layers
            self.extrinsics_embedding_camctrl = nn.Linear(64, 64)
            self.c2ws_hidden_states_layer1 = nn.Linear(64, 64)
            self.c2ws_hidden_states_layer2 = nn.Linear(64, 64)
            nn.init.xavier_uniform_(self.extrinsics_embedding_camctrl.weight)
            nn.init.zeros_(self.extrinsics_embedding_camctrl.bias)
            nn.init.xavier_uniform_(self.c2ws_hidden_states_layer1.weight)
            nn.init.zeros_(self.c2ws_hidden_states_layer1.bias)
            nn.init.xavier_uniform_(self.c2ws_hidden_states_layer2.weight)
            nn.init.zeros_(self.c2ws_hidden_states_layer2.bias)

        elif camera_injection_mode == "residual_norm":
            # Plucker ray embedding added to img tokens
            self.patch_embedding_camctrl = nn.Linear(6 * 16 * 16, self.hidden_size)
            self.c2ws_hidden_states_layer1 = nn.Linear(
                self.hidden_size, self.hidden_size
            )
            self.c2ws_hidden_states_layer2 = nn.Linear(
                self.hidden_size, self.hidden_size
            )
            nn.init.xavier_uniform_(self.patch_embedding_camctrl.weight)
            nn.init.zeros_(self.patch_embedding_camctrl.bias)
            nn.init.xavier_uniform_(self.c2ws_hidden_states_layer1.weight)
            nn.init.zeros_(self.c2ws_hidden_states_layer1.bias)
            nn.init.xavier_uniform_(self.c2ws_hidden_states_layer2.weight)
            nn.init.zeros_(self.c2ws_hidden_states_layer2.bias)

    def _preprocess_camera(
        self,
        norm_extrinsics,
        norm_intrinsics,
        action_onehot,
        oh,
        ow,
        img_dtype,
        device,
    ):
        """Preprocess camera inputs based on camera_injection_mode.

        Returns:
            camera_embedding: The processed camera embedding to inject.
                - action_attn: action_onehot passthrough [B, T, 8]
                - extrinsics_attn: extrinsics embedding [B, T', 64]
                - residual_norm: plucker embedding [B, L, hidden_size]
        """
        from hymm.models.camera.camera_modules import ray_condition
        from einops import rearrange

        mode = self.camera_injection_mode

        if mode == "action_attn":
            # Passthrough: action_onehot is used directly as keyboard input
            return action_onehot

        if mode == "extrinsics_attn":
            extrinsics_embedding = norm_extrinsics.to(img_dtype)  # [B, T, 4, 4]
            # Pad front by repeating the first embedding so that total frames is divisible by 4.
            # Training typically has T=4k+1 (pad 3), but inference may have T=4k (pad 4) etc.
            num_pad = (4 - extrinsics_embedding.shape[1] % 4) % 4
            if num_pad > 0:
                first_embedding = extrinsics_embedding[:, :1]
                pad_embeddings = first_embedding.repeat(1, num_pad, 1, 1)
                extrinsics_embedding = torch.cat(
                    [pad_embeddings, extrinsics_embedding], dim=1
                )

            b, f, h, w = extrinsics_embedding.shape
            assert f % 4 == 0, (
                f"After padding, frame count {f} must be divisible by 4 (original T={norm_extrinsics.shape[1]}, padded {num_pad})"
            )
            extrinsics_embedding = extrinsics_embedding.contiguous().view(
                b, f // 4, 4, h * w
            )
            extrinsics_embedding = extrinsics_embedding.contiguous().view(
                b, f // 4, 4 * h * w
            )

            extrinsics_embedding = self.extrinsics_embedding_camctrl(
                extrinsics_embedding
            )
            c2ws_hidden_states = self.c2ws_hidden_states_layer2(
                F.silu(self.c2ws_hidden_states_layer1(extrinsics_embedding))
            )
            extrinsics_embedding = (
                extrinsics_embedding + c2ws_hidden_states
            )  # [B, T', 64]
            return extrinsics_embedding

        elif mode == "residual_norm":
            # Build K from intrinsics
            fx = norm_intrinsics[..., 0, 0]
            fy = norm_intrinsics[..., 1, 1]
            cx = norm_intrinsics[..., 0, 2]
            cy = norm_intrinsics[..., 1, 2]
            K = torch.stack([fx, fy, cx, cy], dim=-1)  # [B, T, 4]

            # Pad and subsample to match latent temporal dim
            norm_extrinsics = torch.cat(
                [norm_extrinsics[:, :1].repeat(1, 3, 1, 1), norm_extrinsics], dim=1
            )
            K = torch.cat([K[:, :1].repeat(1, 3, 1), K], dim=1)
            norm_extrinsics = norm_extrinsics[:, ::4]
            K = K[:, ::4]

            plucker_embedding = ray_condition(
                K, norm_extrinsics, oh * 16, ow * 16, device=device
            )
            plucker_embedding = plucker_embedding.permute(0, 1, 4, 2, 3).to(img_dtype)
            b, f, c, h, w = plucker_embedding.shape
            plucker_embedding = plucker_embedding.transpose(1, 2)
            plucker_embedding = rearrange(
                plucker_embedding,
                "b c f (h c1) (w c2) -> b (f h w) (c c1 c2)",
                c1=16,
                c2=16,
            )

            c2ws_plucker_embedding = self.patch_embedding_camctrl(plucker_embedding)
            c2ws_hidden_states = self.c2ws_hidden_states_layer2(
                F.silu(self.c2ws_hidden_states_layer1(c2ws_plucker_embedding))
            )
            c2ws_plucker_embedding = c2ws_plucker_embedding + c2ws_hidden_states
            return c2ws_plucker_embedding

    def _prepare_token_packed_camera(
        self,
        packed_cam: Dict[str, Any],
        sample_shapes: List[Tuple[int, int, int]],
        img_dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[
        Optional[torch.Tensor], Optional[torch.Tensor], Optional[Dict[str, Any]]
    ]:
        """Build per-sample camera tensors then concatenate along the time / token axis.

        When ``has_camera`` is False for a packed row, we still run ``_preprocess_camera`` on
        the provided tensors (e.g. zeros from the dataloader) so camera modules receive
        gradients instead of being skipped with empty keyboard / detached zero plucker.

        Returns:
            (camera_embedding, residual_norm_add, packed_action_meta)
            - action_attn / extrinsics_attn: ``camera_embedding`` is ``[1, sum_K, C]`` for
              ``ActionModule``; ``residual_norm_add`` is None; ``packed_action_meta`` carries
              per-sample counts for packed cross-attention.
            - residual_norm: ``residual_norm_add`` is ``[1, sum_L, hidden]`` to add to ``img``;
              ``camera_embedding`` is None; ``packed_action_meta`` is None.
        """
        mode = self.camera_injection_mode
        if mode is None:
            return None, None, None

        pt, ph, pw = self.patch_size
        img_token_counts = [
            (T // pt) * (H // ph) * (W // pw) for T, H, W in sample_shapes
        ]
        raw_has = packed_cam.get("has_camera") or [True] * len(sample_shapes)
        has_list: List[bool] = []
        for i in range(len(sample_shapes)):
            if i < len(raw_has):
                h = raw_has[i]
                has_list.append(
                    bool(h.item()) if isinstance(h, torch.Tensor) else bool(h)
                )
            else:
                has_list.append(True)

        if mode == "residual_norm":
            parts: List[torch.Tensor] = []
            for i, (T, H, W) in enumerate(sample_shapes):
                Li = img_token_counts[i]
                ne = packed_cam["norm_extrinsics"][i].to(device=device, dtype=img_dtype)
                ni = packed_cam["norm_intrinsics"][i].to(device=device, dtype=img_dtype)
                max_cam = max((T - 1) * 4 + 1, 1)
                ne = ne[:max_cam]
                ni = ni[:max_cam]
                plucker = self._preprocess_camera(
                    ne.unsqueeze(0),
                    ni.unsqueeze(0),
                    None,
                    oh=H * 16,
                    ow=W * 16,
                    img_dtype=img_dtype,
                    device=device,
                )
                plucker = plucker.squeeze(0)
                if plucker.shape[0] != Li:
                    if plucker.shape[0] > Li:
                        plucker = plucker[:Li]
                    else:
                        pad = torch.zeros(
                            Li - plucker.shape[0],
                            plucker.shape[1],
                            device=device,
                            dtype=img_dtype,
                        )
                        plucker = torch.cat([plucker, pad], dim=0)
                parts.append(plucker)
            res = torch.cat(parts, dim=0).unsqueeze(0)
            return None, res, None

        emb_parts: List[torch.Tensor] = []
        kb_lens: List[int] = []
        for i, (T, H, W) in enumerate(sample_shapes):
            max_cam = max((T - 1) * 4 + 1, 1)
            ne_i = packed_cam["norm_extrinsics"][i].to(device=device, dtype=img_dtype)[
                :max_cam
            ]
            ni_i = packed_cam["norm_intrinsics"][i].to(device=device, dtype=img_dtype)[
                :max_cam
            ]
            ao_i = packed_cam["action_onehot"][i].to(device=device, dtype=img_dtype)[
                :max_cam
            ]

            if mode == "action_attn":
                kb = ao_i.unsqueeze(0) if ao_i.dim() == 1 else ao_i
                if kb.dim() == 2:
                    kb = kb.unsqueeze(0)
                out = self._preprocess_camera(
                    None,
                    None,
                    kb,
                    oh=H * 16,
                    ow=W * 16,
                    img_dtype=img_dtype,
                    device=device,
                )
            elif mode == "extrinsics_attn":
                out = self._preprocess_camera(
                    ne_i.unsqueeze(0),
                    ni_i.unsqueeze(0),
                    None,
                    oh=H * 16,
                    ow=W * 16,
                    img_dtype=img_dtype,
                    device=device,
                )
            else:
                return None, None, None
            emb_parts.append(out)
            kb_lens.append(int(out.shape[1]))

        if not emb_parts:
            return None, None, None
        camera_embedding = torch.cat(emb_parts, dim=1)
        if camera_embedding.shape[1] == 0:
            return None, None, None

        packed_action_meta = {
            "img_token_counts": img_token_counts,
            "kb_token_counts": kb_lens,
            "sample_shapes": list(sample_shapes),
            "has_camera": has_list,
            "packed_segment_boundaries_latent": packed_cam.get(
                "prompt_video_boundaries_latent"
            ),
            # Per-segment camera on/off (training per-segment dropout). When
            # present, the ActionModule hard-gates cam_out to 0 on dropped
            # segments' tokens (per packed sample). None => no per-segment gate.
            "has_camera_per_segment": packed_cam.get("has_camera_per_segment"),
        }
        return camera_embedding, None, packed_action_meta

    def enable_deterministic(self):
        for block in self.double_blocks:
            block.enable_deterministic()
        for block in self.single_blocks:
            block.enable_deterministic()

    def disable_deterministic(self):
        for block in self.double_blocks:
            block.disable_deterministic()
        for block in self.single_blocks:
            block.disable_deterministic()

    def get_rotary_pos_embed(self, rope_sizes, device=None):
        target_ndim = 3
        head_dim = self.hidden_size // self.heads_num
        rope_dim_list = self.rope_dim_list
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
            device=device,
        )
        return freqs_cos, freqs_sin

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        text_states: Union[torch.Tensor, List[torch.Tensor]],
        encoder_attention_mask: Union[torch.Tensor, List[torch.Tensor]],
        vision_states: torch.Tensor = None,
        output_features=False,
        output_features_stride=8,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        return_dict: bool = False,
        guidance=None,
        mask_type="t2v",
        extra_kwargs=None,
        # Camera-specific inputs (only used when camera_injection_mode is set)
        action_labels: Optional[torch.Tensor] = None,
        action_onehot: Optional[torch.Tensor] = None,
        norm_extrinsics: Optional[torch.Tensor] = None,
        norm_intrinsics: Optional[torch.Tensor] = None,
        has_camera_mask: Optional[
            torch.Tensor
        ] = None,  # [B] bool mask, True = has real camera data
        # Discriminator feature hooks (additive, default off). When
        # out_contexts=True, the forward collects the post-double-block hidden
        # state ``img`` at the layer indices in hook_layers and returns a dict
        # {str(index): [B, L, hidden_size]} INSTEAD of the usual output — used by
        # the DMD+ADV discriminator head. Mirrors Wan22 WanModel.forward's
        # out_contexts/hook_layers. Leaving out_contexts=False keeps behaviour
        # byte-identical to before.
        out_contexts: bool = False,
        hook_layers: Optional[List[int]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass supporting segment text inputs.

        text_states can be:
        - torch.Tensor [B, S, C]: standard single text for all chunks
        - torch.Tensor [B, N, S, C]: per-prompt text (N text segments)
        - List[torch.Tensor]: list of N tensors each [B, S, C]

        A single 3D text tensor is treated as one segment.
        """

        # ========== KV cache delegation (FSDP2-safe) ==========
        # When extra_kwargs contains 'kv_cache_params', delegate to
        # forward_with_cache().  This ensures the call goes through
        # model.__call__() (triggering FSDP2 unshard hooks) rather than
        # calling forward_with_cache() directly, which would mix regular
        # Tensors with DTensor parameters.
        _kv_params = extra_kwargs.get("kv_cache_params", None) if extra_kwargs else None
        if _kv_params is not None:
            return self.forward_with_cache(
                hidden_states=hidden_states,
                timestep=timestep,
                text_states=text_states
                if not isinstance(text_states, list)
                else text_states[0],
                text_mask=encoder_attention_mask
                if not isinstance(encoder_attention_mask, list)
                else encoder_attention_mask[0],
                kv_cache_list=_kv_params.get("kv_cache_list", None),
                cache_valid_len=_kv_params.get("cache_valid_len", 0),
                update_cache=_kv_params.get("update_cache", False),
                global_frame_counter=_kv_params.get("global_frame_counter", 0),
                attn_mode_override=_kv_params.get("attn_mode_override", None),
                cache_txt=_kv_params.get("cache_txt", False),
                cache_mask_list=_kv_params.get("cache_mask_list", None),
                intra_seg_txt_offsets=_kv_params.get("intra_seg_txt_offsets", None),
                intra_seg_vid_windows=_kv_params.get("intra_seg_vid_windows", None),
                hide_cached_action_tails=_kv_params.get(
                    "hide_cached_action_tails", False
                ),
                current_action_start=_kv_params.get("current_action_start", -1),
                camera_embedding=_kv_params.get("camera_embedding", None),
                camera_has_mask=_kv_params.get("camera_has_mask", None),
                camera_inputs=_kv_params.get("camera_inputs", None),
                camera_rope_time_offset=_kv_params.get("camera_rope_time_offset", 0),
            )

        if guidance is None:
            guidance = torch.tensor(
                [6016.0], device=hidden_states.device, dtype=torch.bfloat16
            )

        img = hidden_states
        t = timestep  # [B] - scalar timestep for bidirectional training

        # ========== Token packing: pre-patchified flat input ==========
        # When extra_kwargs contains 'token_packing_info', hidden_states is
        # [B, C_in, L_total] (flat packed latent tokens, NOT 5D).
        # 'sample_shapes' is a list of (T_i, H_i, W_i) per packed sample.
        # We use forward_flat for patchify and get_packed_rotary_pos_embed
        # to generate per-sample RoPE concatenated along the token dim.
        _tp_info = (
            extra_kwargs.get("token_packing_info", None) if extra_kwargs else None
        )
        if _tp_info is not None:
            assert self.img_in.is_pointwise, (
                "token_packing_info requires patch_size=(1,1,1) (pointwise PatchEmbed). "
                f"Got patch_size={self.patch_size}"
            )
            sample_shapes = _tp_info["sample_shapes"]  # [(T0,H0,W0), (T1,H1,W1), ...]
            bs = img.shape[0]

            # For downstream code that reads (tt, th, tw) / (ot, oh, ow),
            # use the first sample's shape as a representative.  These are
            # only needed for camera control and unpatchify, which are not
            # used in the fused packed-attention path.
            T0, H0, W0 = sample_shapes[0]
            pt, ph, pw = self.patch_size
            tt, th, tw = T0 // pt, H0 // ph, W0 // pw
            ot, oh, ow = T0, H0, W0
            self.attn_param["thw"] = [tt, th, tw]

            # Project flat tokens: [B, C_in, L] -> [B, L, C_in] -> linear -> [B, L, hidden]
            img = img.permute(0, 2, 1)
            img = self.img_in.forward_flat(img)

            # Generate packed RoPE: per-sample RoPE concatenated along token dim
            freqs_cos, freqs_sin = get_packed_rotary_pos_embed(
                rope_dim_list=self.rope_dim_list,
                sample_shapes=sample_shapes,
                patch_size=tuple(self.patch_size),
                theta=self.rope_theta,
                device=hidden_states.device,
            )

            # Build cu_seqlens for packed block-diagonal attention.
            # Each sample's img and txt tokens form an independent sub-sequence.
            img_token_counts = [
                (T // pt) * (H // ph) * (W // pw) for T, H, W in sample_shapes
            ]
            txt_token_counts = _tp_info["txt_token_counts"]  # [n0, n1, ...]
            assert len(txt_token_counts) == len(sample_shapes), (
                f"txt_token_counts length ({len(txt_token_counts)}) must match "
                f"sample_shapes length ({len(sample_shapes)})"
            )
            img_cu = [0] + list(itertools.accumulate(img_token_counts))
            txt_cu = [0] + list(itertools.accumulate(txt_token_counts))
            self.attn_param["packed_cu_seqlens_img"] = torch.tensor(
                img_cu, dtype=torch.int32, device=hidden_states.device
            )
            self.attn_param["packed_cu_seqlens_txt"] = torch.tensor(
                txt_cu, dtype=torch.int32, device=hidden_states.device
            )

            # Packed attention mode: default to the model's configured attn_mode.
            # For magi_mp_causal, also store per-sample magi parameters.
            self.attn_param["packed_attn_mode"] = self.attn_mode
            if "per_sample_magi_params" in _tp_info:
                self.attn_param["packed_per_sample_magi_params"] = _tp_info[
                    "per_sample_magi_params"
                ]

            # Token packing + Sequence Parallel: chunk img and RoPE along
            # the token dim, just like the normal (non-packed) path.
            # The all-to-all in the attention layer will gather the full
            # sequence back before running block-diagonal flash attention.
            if get_sequence_parallel_state():
                sp_size = nccl_info.sp_size
                sp_rank = nccl_info.rank_within_spgroup
                assert not nccl_info.use_dynamic_ring_attention, (
                    "Token packing is not compatible with dynamic ring attention. "
                    "Use Ulysses SP (use_dynamic_ring_attention=false) instead."
                )
                img = torch.chunk(img, sp_size, dim=1)[sp_rank]
                freqs_cos = torch.chunk(freqs_cos, sp_size, dim=0)[sp_rank]
                freqs_sin = torch.chunk(freqs_sin, sp_size, dim=0)[sp_rank]
        else:
            bs, _, ot, oh, ow = img.shape
            tt, th, tw = (
                ot // self.patch_size[0],
                oh // self.patch_size[1],
                ow // self.patch_size[2],
            )
            self.attn_param["thw"] = [tt, th, tw]
            img = self.img_in(img)

        # Check for segment-level teacher forcing mode
        segment_tf_params = (
            extra_kwargs.get("segment_tf_params", None) if extra_kwargs else None
        )
        attn_mode_override = (
            extra_kwargs.get("attn_mode_override", None) if extra_kwargs else None
        )

        if freqs_cos is None and freqs_sin is None:
            if segment_tf_params is not None:
                # Segment TF RoPE: noisy_seg[j] and clean_seg[j] share same positions.
                # The interleaved sequence has 2x frames: [noisy_seg_0, clean_seg_0, ...]
                # Each segment pair (noisy+clean) maps to the same original frame positions.
                segment_frame_lengths = segment_tf_params["segment_frame_lengths"]
                temporal_positions = []
                frame_offset = 0
                for seg_len in segment_frame_lengths:
                    seg_positions = list(range(frame_offset, frame_offset + seg_len))
                    temporal_positions.extend(seg_positions)  # noisy
                    temporal_positions.extend(seg_positions)  # clean (same positions)
                    frame_offset += seg_len
                temporal_positions = torch.tensor(
                    temporal_positions, dtype=torch.float32, device=hidden_states.device
                )

                head_dim = self.hidden_size // self.heads_num
                rope_dim_list = self.rope_dim_list
                if rope_dim_list is None:
                    target_ndim = 3
                    rope_dim_list = [
                        head_dim // target_ndim for _ in range(target_ndim)
                    ]

                freqs_cos, freqs_sin = get_nd_rotary_pos_embed_with_temporal_positions(
                    rope_dim_list=rope_dim_list,
                    temporal_positions=temporal_positions,
                    spatial_sizes=(th, tw),
                    theta=self.rope_theta,
                    use_real=True,
                    theta_rescale_factor=1.0,
                )
            else:
                # Standard RoPE
                freqs_cos, freqs_sin = self.get_rotary_pos_embed(
                    (tt, th, tw), device=hidden_states.device
                )
        elif freqs_cos is not None and freqs_sin is not None:
            freqs_cos = freqs_cos.to(hidden_states.device)
            freqs_sin = freqs_sin.to(hidden_states.device)

        # ========== Camera control: preprocess and inject ==========
        camera_embedding = None
        packed_action_meta = None
        _pack_cam_raw = _tp_info.get("packed_camera") if _tp_info else None
        if _pack_cam_raw is not None and self.camera_injection_mode is not None:
            with _prof("camera/preprocess"):
                ce, res_add, packed_action_meta = self._prepare_token_packed_camera(
                    _pack_cam_raw,
                    _tp_info["sample_shapes"],
                    img_dtype=img.dtype,
                    device=img.device,
                )
            if self.camera_injection_mode == "residual_norm":
                if res_add is not None:
                    if res_add.dtype != img.dtype:
                        res_add = res_add.to(dtype=img.dtype)
                    img = img + res_add
                camera_embedding = None
            else:
                camera_embedding = ce
        elif self.camera_injection_mode is not None and norm_extrinsics is not None:
            with _prof("camera/preprocess"):
                camera_embedding = self._preprocess_camera(
                    norm_extrinsics=norm_extrinsics,
                    norm_intrinsics=norm_intrinsics,
                    action_onehot=action_onehot,
                    oh=oh,
                    ow=ow,
                    img_dtype=img.dtype,
                    device=img.device,
                )
            if self.camera_injection_mode == "residual_norm":
                # Add plucker embedding directly to img tokens (before SP chunking).
                # camera_embedding: [B, L, C] where L = tt * th * tw.
                if has_camera_mask is not None:
                    if has_camera_mask.dim() == 1:
                        # Legacy [B] -> broadcast over (L, C).
                        mask = has_camera_mask.float()[:, None, None]
                        camera_embedding = camera_embedding * mask
                    elif has_camera_mask.dim() == 2:
                        # [B, N_segments] -> token-level mask via prompt_video_boundaries.
                        N_seg = has_camera_mask.shape[1]
                        _pvb = (
                            extra_kwargs.get("prompt_video_boundaries", None)
                            if extra_kwargs
                            else None
                        )
                        if isinstance(_pvb, torch.Tensor):
                            _pvb = _pvb.tolist()
                            if _pvb and isinstance(_pvb[0], list):
                                _pvb = _pvb[0]
                            _pvb = [int(b) for b in _pvb]
                        assert _pvb is not None and len(_pvb) == N_seg + 1, (
                            f"residual_norm per-segment has_camera_mask "
                            f"(shape={tuple(has_camera_mask.shape)}) requires "
                            f"extra_kwargs['prompt_video_boundaries'] of length "
                            f"N_seg+1={N_seg + 1}, got {_pvb}"
                        )
                        S = th * tw
                        L = tt * S
                        device = camera_embedding.device
                        token_ids = torch.arange(L, device=device)
                        frame_ids = (
                            torch.div(token_ids, S, rounding_mode="floor")
                            .clamp(max=tt - 1)
                            .long()
                        )
                        interior = torch.tensor(
                            list(_pvb[1:-1]),
                            device=device,
                            dtype=frame_ids.dtype,
                        )
                        if interior.numel() == 0:
                            seg_ids = torch.zeros_like(frame_ids)
                        else:
                            seg_ids = torch.bucketize(frame_ids, interior, right=True)
                        seg_ids = seg_ids.clamp(max=N_seg - 1).long()
                        token_mask = has_camera_mask.float()[:, seg_ids]  # [B, L]
                        camera_embedding = camera_embedding * token_mask.unsqueeze(-1)
                    else:
                        raise ValueError(
                            f"has_camera_mask must be 1D [B] or 2D [B, N_segments], "
                            f"got shape {tuple(has_camera_mask.shape)}"
                        )
                img = img + camera_embedding
                camera_embedding = None  # no further injection needed

        # ========== Handle list/4D text_states: convert to 4D tensor ==========
        # Detect whether text_states is a list of tensors or a 4D tensor
        # and prepare txt / text_mask accordingly.
        num_text_segments = 1
        txt_tokens_per_block = None

        if isinstance(text_states, (list, tuple)):
            # List of tensors: each [B, S, C] -> stack to [B, N, S, C]
            num_text_segments = len(text_states)
            txt = torch.stack(text_states, dim=1)  # [B, N, S, C]

            # Handle encoder_attention_mask similarly
            if isinstance(encoder_attention_mask, (list, tuple)):
                text_mask = torch.stack(encoder_attention_mask, dim=1)  # [B, N, S]
            else:
                text_mask = encoder_attention_mask
        elif text_states.dim() == 4:
            # Already 4D [B, N, S, C]
            txt = text_states
            text_mask = encoder_attention_mask
            num_text_segments = txt.shape[1]
        else:
            # Standard 3D [B, S, C]
            txt = text_states
            text_mask = encoder_attention_mask
            # Auto-wrap a single 3D prompt to one 4D segment.
            # Skip the wrap when token packing is active — _refine_packed_text
            # expects flat 3D input and handles the per-sample structure itself.
            if _tp_info is None:
                txt = txt.unsqueeze(1)  # [B, 1, S, C]
                text_mask = (
                    text_mask.unsqueeze(1) if text_mask.dim() == 2 else text_mask
                )  # [B, 1, S]
                num_text_segments = 1

        # Sequence parallel
        # NOTE: when token packing is active, SP chunking of img/freqs was
        # already done in the _tp_info branch above — skip here to avoid
        # double-chunking.
        if get_sequence_parallel_state() and _tp_info is None:
            sp_size = nccl_info.sp_size
            sp_rank = nccl_info.rank_within_spgroup
            if nccl_info.use_dynamic_ring_attention:
                # For dynamic ring attention, use the appropriate text_mask dimension
                _text_mask_for_sp = (
                    text_mask if text_mask.dim() == 2 else text_mask[:, 0, :]
                )
                img, freqs_cos, freqs_sin = preprocess_dynamic_ring_attention(
                    img, _text_mask_for_sp, freqs_cos, freqs_sin
                )
            else:
                if img.shape[1] % sp_size != 0:
                    n_token = img.shape[1]
                    assert n_token > (n_token // sp_size + 1) * (sp_size - 1), (
                        f"Token is too short for SP {sp_size}"
                    )

                img = torch.chunk(img, sp_size, dim=1)[sp_rank]
                freqs_cos = torch.chunk(freqs_cos, sp_size, dim=0)[sp_rank]
                freqs_sin = torch.chunk(freqs_sin, sp_size, dim=0)[sp_rank]

        # ========== Per-chunk text modulation setup ==========
        tokens_per_frame = th * tw
        num_frames = tt

        # ========== Validate text-video chunk relationship ==========
        # Support both fixed-length and variable-length prompt-video assignment.
        # prompt_video_boundaries: List[int] of length num_prompts+1, exclusive-end chunk indices.
        # E.g. [0, 6, 10] means prompt 0 covers chunks [0,6), prompt 1 covers [6,10).
        num_prompts = num_text_segments  # N in [B, N, S, C]
        prompt_video_boundaries = None

        if extra_kwargs is not None:
            prompt_video_boundaries = extra_kwargs.get("prompt_video_boundaries", None)
            if prompt_video_boundaries is not None and isinstance(
                prompt_video_boundaries, torch.Tensor
            ):
                prompt_video_boundaries = prompt_video_boundaries.tolist()
                if isinstance(prompt_video_boundaries[0], list):
                    prompt_video_boundaries = prompt_video_boundaries[0]
                prompt_video_boundaries = [int(b) for b in prompt_video_boundaries]

        if prompt_video_boundaries is not None:
            assert len(prompt_video_boundaries) == num_prompts + 1, (
                f"prompt_video_boundaries length ({len(prompt_video_boundaries)}) must be num_prompts+1 ({num_prompts + 1})"
            )
            assert (
                prompt_video_boundaries[0] == 0
                and prompt_video_boundaries[-1] == num_frames
            ), (
                f"boundaries must start at 0 and end at {num_frames}, got {prompt_video_boundaries}"
            )

        if text_mask.dim() == 2:
            text_mask = repeat(text_mask, "b s -> b t s", t=num_prompts)

        # ========== Prepare modulation vectors ==========
        # chunk_size controls per-group noise granularity:
        #    1: per-frame (each frame gets its own noise level / modulation)
        #    0: segment-level (uses prompt_video_boundaries)
        #   >1: chunk-level (uniform groups of chunk_size latent frames)
        # chunk_size == 1 and chunk_size > 1 share the same abstraction.
        chunk_size = 1
        if extra_kwargs is not None:
            chunk_size = extra_kwargs.get("chunk_size", 1)

        _packed_per_token_t = (
            bool(extra_kwargs.get("packed_per_token_timestep", False))
            if extra_kwargs
            else False
        )

        if _packed_per_token_t:
            # ===== Token-packing diffusion forcing =====
            # t is already a per-token timestep [B, L_total] for the FULL packed
            # sequence (built in unified._df_packed_prepare).  The transport laid
            # out per-segment / per-chunk / per-frame timesteps directly onto the
            # token axis, so here we only need to (a) SP-chunk t to match the img
            # shard, then (b) embed it into a 3D per-token vec [B, L_local, C].
            assert _tp_info is not None, (
                "packed_per_token_timestep requires token_packing_info."
            )
            t_full = t  # [B, L_total]
            if get_sequence_parallel_state():
                sp_size = nccl_info.sp_size
                sp_rank = nccl_info.rank_within_spgroup
                if not nccl_info.use_dynamic_ring_attention:
                    t_local = torch.chunk(t_full, sp_size, dim=1)[
                        sp_rank
                    ]  # [B, L_local]
                else:
                    t_local = t_full
            else:
                t_local = t_full

            # Embed: [B, L_local] -> [B*L_local, C] -> [B, L_local, C]
            B_t, L_local = t_local.shape
            vec_flat = self.time_in(t_local.reshape(-1))  # [B*L_local, C]
            vec = vec_flat.view(B_t, L_local, -1)  # [B, L_local, C] per-token

            # Each packed text segment uses its corresponding video chunk's
            # timestep. Read it from the segment's first video token.
            #
            # We first compute the per-segment VIDEO chunk timestep (the t at each
            # segment's first video token) for alignment logging and text
            # modulation.
            _seg_video_t = None  # list[float] per-segment video chunk t
            _seg_dbg = None  # list[(sample_idx, seg_idx, tok0, video_t)]
            _S_enc = None
            _psm = _tp_info.get("per_sample_magi_params", None) if _tp_info else None
            if _psm is not None:
                _S_enc = int(_psm[0].get("txt_tokens_per_block"))
                _seg_video_t = []
                _seg_dbg = []
                for _i in range(len(sample_shapes)):
                    _magi_i = _psm[_i]
                    _pvb_tok = _magi_i[
                        "prompt_video_boundaries"
                    ]  # within-sample token units
                    _nseg_i = int(_magi_i.get("num_prompts", 1))
                    _samp_start = int(img_cu[_i])
                    for _s in range(_nseg_i):
                        _tok0 = _samp_start + int(_pvb_tok[_s])
                        _tok0 = min(_tok0, t_full.shape[1] - 1)
                        _tv = float(t_full[0, _tok0])
                        _seg_video_t.append(_tv)
                        _seg_dbg.append((_i, _s, _tok0, _tv))

            # Per-segment text timestep = the segment's video chunk timestep.
            assert _seg_video_t is not None, (
                "Packed text modulation needs per_sample_magi_params "
                "(magi_mp_causal attention) to locate per-segment video "
                "token boundaries."
            )
            # [total_prompts] in sample-major / segment order — the SAME
            # order _refine_packed_text batches text blocks and the flat
            # packed text is laid out.
            t_per_prompt = torch.tensor(
                _seg_video_t,
                device=t.device,
                dtype=t.dtype,
            )
            # Per-text-token vec: each segment's S_enc tokens share its t.
            _ttxt = t_per_prompt.repeat_interleave(_S_enc)  # [S_total]
            vec_txt = self.time_in(_ttxt).view(
                bs, -1, vec.shape[-1]
            )  # [1, S_total, C]
            # Pass per-prompt timesteps into the refiner so its per-block
            # modulation also matches the segment's video chunk timestep.
            txt_t = t_per_prompt

            # ---- Logging: text-vs-video timestep alignment ----
            if _seg_dbg is not None and __import__("os").environ.get(
                "LOG_TXT_VIDEO_T", ""
            ).strip() not in ("", "0", "false", "False"):
                _is_rank0 = True
                try:
                    if (
                        torch.distributed.is_available()
                        and torch.distributed.is_initialized()
                    ):
                        _is_rank0 = torch.distributed.get_rank() == 0
                except Exception:
                    pass
                if _is_rank0:
                    import sys as _sys

                    # Each text segment uses the corresponding video timestep.
                    _lines = []
                    _all_match = True
                    for _si, _se, _tk, _vid_t in _seg_dbg:
                        _text_t = _vid_t
                        _m = abs(_vid_t - _text_t) < 1e-6
                        _all_match = _all_match and _m
                        _lines.append(
                            f"s{_si}/seg{_se}: tok0={_tk} "
                            f"video_t={_vid_t:.4f} text_t={_text_t:.4f} "
                            f"match={'Y' if _m else 'N'}"
                        )
                    print(
                        "[LOG_TXT_VIDEO_T] "
                        f"| num_samples={len(sample_shapes)} "
                        f"total_segments={len(_seg_dbg)} S_enc={_S_enc} "
                        f"all_match={'Y' if _all_match else 'N'} | "
                        + " || ".join(_lines),
                        file=_sys.stderr,
                        flush=True,
                    )
        elif t.dim() == 2:
            # t is [B, T_latent] — per-frame timestep from diffusion forcing.
            # Convert to per-group vec: compute one vec per group, then expand to per-token.
            #
            # IMPORTANT: boundaries are in latent frame units (ot),
            # but after PatchEmbed, img has L_img = tt * th * tw tokens where
            # tt = ot // patch_size[0].  We must convert boundaries to patched-frame
            # units for the token expansion step.
            tokens_per_frame = th * tw
            pt = self.patch_size[0]  # temporal patch size
            num_latent_frames = t.shape[1]

            if chunk_size == 1:
                # Per-frame diffusion forcing: each frame is its own group
                _boundaries = list(range(num_latent_frames + 1))
            elif chunk_size == 0 and prompt_video_boundaries is not None:
                # Segment boundaries are data-driven in latent frame units.
                _boundaries = prompt_video_boundaries  # latent frame units (ot)
            else:
                # Chunk-level: uniform boundaries every chunk_size frames
                _boundaries = list(range(0, num_latent_frames, chunk_size))
                if _boundaries[-1] != num_latent_frames:
                    _boundaries.append(num_latent_frames)
                # Merge last chunk into previous if remainder < half a chunk
                if len(_boundaries) > 2:
                    last_size = _boundaries[-1] - _boundaries[-2]
                    if last_size < chunk_size // 2:
                        _boundaries.pop(-2)

            _num_groups = len(_boundaries) - 1

            # Convert boundaries to patched-frame units for token expansion
            _patched_boundaries = [b // pt for b in _boundaries]

            # Compute per-group mean timestep: [B, num_groups]
            t_per_group = torch.zeros(bs, _num_groups, device=t.device, dtype=t.dtype)
            for grp_idx in range(_num_groups):
                grp_start = _boundaries[grp_idx]
                grp_end = _boundaries[grp_idx + 1]
                t_per_group[:, grp_idx] = t[:, grp_start:grp_end].mean(dim=1)

            # Compute per-group vec: flatten [B, num_groups] -> [B*num_groups], then unflatten
            vec_flat = self.time_in(t_per_group.flatten())  # [B*num_groups, C]
            vec_per_group = vec_flat.view(bs, _num_groups, -1)  # [B, num_groups, C]

            # Expand per-group vec to per-img-token: each patched frame in a group
            # gets the same vec.
            vec_per_token_list = []
            for grp_idx in range(_num_groups):
                grp_start_patched = _patched_boundaries[grp_idx]
                grp_end_patched = _patched_boundaries[grp_idx + 1]
                num_patched_frames_in_grp = grp_end_patched - grp_start_patched
                num_tokens_in_grp = num_patched_frames_in_grp * tokens_per_frame
                vec_per_token_list.append(
                    vec_per_group[:, grp_idx : grp_idx + 1, :].expand(
                        -1, num_tokens_in_grp, -1
                    )
                )
            vec = torch.cat(vec_per_token_list, dim=1)  # [B, L_img_full, C]

            # Chunk vec for Sequence Parallel (to match the img chunk on this rank)
            if get_sequence_parallel_state():
                sp_size = nccl_info.sp_size
                sp_rank = nccl_info.rank_within_spgroup
                if not nccl_info.use_dynamic_ring_attention:
                    vec = torch.chunk(vec, sp_size, dim=1)[
                        sp_rank
                    ]  # [B, L_img_chunk, C]

            # Per-segment text modulation uses each group's timestep.
            txt_t = t_per_group  # [B, num_segments]
            vec_txt_flat = self.time_in(txt_t.flatten())
            vec_txt = vec_txt_flat.view(bs, txt_t.shape[1], -1)
        else:
            # Standard path: t is [B] (scalar timestep for bidirectional training)
            vec = self.time_in(t)  # [B, C]

            txt_t = t.unsqueeze(1).expand(-1, num_prompts)  # [B, num_prompts]
            vec_txt = vec  # [B, C] — all segments share the same scalar t

        # vec_3d is the 3D version of vec when using per-token timestep (causal path).
        # In the standard (bidirectional) path, vec is 2D [B, C] and vec_3d is None.
        vec_3d = vec if vec.dim() == 3 else None

        # guidance modulation (for CFG-distilled models)
        if self.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )
            # guidance is a scalar tensor, embed it via TimestepEmbedder
            guidance_vec = self.guidance_in(guidance)
            vec = vec + guidance_vec
            if vec_3d is not None:
                vec_3d = vec_3d + guidance_vec.unsqueeze(1)
            # Also add guidance to vec_txt
            if vec_txt.dim() == 3:
                vec_txt = vec_txt + guidance_vec.unsqueeze(1)
            else:
                vec_txt = vec_txt + guidance_vec

        # ========== Text embedding ==========
        if _tp_info is not None and self.text_projection == "single_refiner":
            # Token packing + single_refiner: process each packed sample's
            # text independently to prevent cross-sample self-attention
            # contamination in the refiner's IndividualTokenRefinerBlock.
            txt, text_mask = self._refine_packed_text(
                txt,
                text_mask,
                txt_t,
                _tp_info,
            )
        elif self.text_projection == "linear":
            if txt.dim() == 4:
                # 4D text [B, T, S, C_in] -> process each chunk independently
                B_txt, T_txt, S_txt, C_txt = txt.shape
                txt = rearrange(txt, "b t s c -> (b t) s c")
                txt = self.txt_in(txt)
                txt = rearrange(txt, "(b t) s c -> b t s c", b=B_txt, t=T_txt)
            else:
                txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            # SingleTokenRefinerPerBlock supports both 3D [B, S, C] and 4D [B, T, S, C] input.
            # When txt is 4D (pre-chunked text), the refiner processes each chunk with
            # its corresponding timestep, preserving per-chunk text diversity.
            # When txt is 3D, the refiner internally expands to 4D if txt_t is [B, T].
            txt = self.txt_in(
                txt, txt_t, text_mask if self.use_attention_mask else None
            )
            # Segment text is 4D [B, T, S, C].
            # Expand text_mask to match: [B, S] -> [B, T, S] (only if not already 3D)
            if txt.dim() == 4:
                assert text_mask.dim() == 3, (
                    f"text_mask must be 3D when txt is 4D, got {text_mask.dim()}"
                )
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
        self.loss_moe = 0.0

        # ========== Pre-set attn_param for MagiMP attention ==========
        # In packed mode the per-sample structure is carried by
        # packed_per_sample_magi_params; the global values below are
        # only used by the non-packed attention path and are set to
        # reasonable defaults so downstream code doesn't break.
        if _tp_info is not None:
            per_sample_magi = _tp_info.get("per_sample_magi_params", None)
            if per_sample_magi is not None and per_sample_magi:
                txt_tokens_per_block = per_sample_magi[0].get(
                    "txt_tokens_per_block", txt.shape[-1]
                )
                num_prompts = per_sample_magi[0].get("num_prompts", 1)
            else:
                txt_tokens_per_block = txt.shape[-1]
                num_prompts = 1
        elif txt.dim() == 4:
            txt_tokens_per_block = txt.shape[2]
        elif txt.dim() == 3:
            assert txt.shape[1] % num_prompts == 0, (
                f"txt length {txt.shape[1]} must be divisible by "
                f"num_prompts {num_prompts}"
            )
            txt_tokens_per_block = txt.shape[1] // num_prompts
        else:
            raise ValueError(f"Unexpected txt tensor dim: {txt.dim()}, expected 3 or 4")

        # Pre-compute token-unit boundaries and chunk_size_tokens so attention.py
        # doesn't need to know about chunk_size details.  Always refresh the
        # boundary-related entries because ``self.attn_param`` is reused across
        # samples; leaving a previous sample's boundaries here can make
        # MagiAttention validate against stale video lengths during validation.
        tokens_per_frame = th * tw
        if prompt_video_boundaries is None:
            # In token-packed mode, per-sample boundaries are carried by
            # ``_tp_info['per_sample_magi_params']`` (token units) and the
            # global ``self.attn_param`` values are only used by the
            # non-packed attention path. Skip the multi-prompt requirement
            # here when packing is active to avoid a false positive.
            if num_prompts != 1 and _tp_info is None:
                raise ValueError(
                    "prompt_video_boundaries is required when "
                    f"num_prompts={num_prompts} for magi_mp_causal attention"
                )
            prompt_video_boundaries_for_attn = [0, num_frames]
        else:
            prompt_video_boundaries_for_attn = prompt_video_boundaries

        self.attn_param["num_prompts"] = len(prompt_video_boundaries_for_attn) - 1
        self.attn_param["txt_tokens_per_block"] = txt_tokens_per_block
        self.attn_param["prompt_video_boundaries"] = prompt_video_boundaries_for_attn
        self.attn_param["prompt_video_boundaries_tokens"] = [
            b * tokens_per_frame for b in prompt_video_boundaries_for_attn
        ]
        # Intra-segment per-action masking params (one_shot). Refresh every
        # call so a previous sample's action layout never leaks via the reused
        # self.attn_param. Video windows arrive already in ABSOLUTE video-token
        # coords from the pipeline -> store as-is (no *tokens_per_frame here).
        self.attn_param["txt_action_start_per_block"] = (
            extra_kwargs.get("txt_action_start_per_block", None)
            if extra_kwargs
            else None
        )
        self.attn_param["txt_action_offsets_per_block"] = (
            extra_kwargs.get("txt_action_offsets_per_block", None)
            if extra_kwargs
            else None
        )
        self.attn_param["vid_action_boundaries_per_block"] = (
            extra_kwargs.get("vid_action_boundaries_per_block", None)
            if extra_kwargs
            else None
        )
        self.attn_param["cross_segment_text_head_only"] = (
            bool(extra_kwargs.get("cross_segment_text_head_only", True))
            if extra_kwargs
            else True
        )
        # Previous-segment last-frame video->video restriction (segment-level
        # causal, chunk_size==0). tokens_per_frame == th*tw (patch_size_t==1
        # so one latent frame == one tt == tokens_per_frame tokens).
        self.attn_param["prev_seg_last_frame_only"] = bool(
            extra_kwargs.get("prev_seg_last_frame_only", False)
            if extra_kwargs
            else False
        )
        self.attn_param["tokens_per_frame"] = tokens_per_frame

        if __import__("os").environ.get("DEBUG_PVB", "").strip() not in (
            "",
            "0",
            "false",
            "False",
        ):
            import sys as _sys

            _img_len = img.shape[1]
            print(
                f"[DEBUG_PVB] nomas_cpt.forward: tt={tt} th={th} tw={tw} "
                f"num_frames={num_frames} tokens_per_frame={tokens_per_frame} "
                f"img.shape[1](S_video_local)={_img_len} "
                f"sp_state={get_sequence_parallel_state()} "
                f"num_prompts={self.attn_param['num_prompts']} num_text_segments={num_text_segments} "
                f"pvb(frames)={prompt_video_boundaries_for_attn} "
                f"pvb_tokens={self.attn_param.get('prompt_video_boundaries_tokens')} "
                f"expected_last_token={num_frames * tokens_per_frame}",
                file=_sys.stderr,
                flush=True,
            )

        chunk_size = extra_kwargs.get("chunk_size", 0) if extra_kwargs else 0
        if chunk_size == 1:
            # Per-frame: each frame is its own chunk for attention
            self.attn_param["chunk_size_tokens"] = tokens_per_frame
        elif chunk_size > 1:
            self.attn_param["chunk_size_tokens"] = chunk_size * tokens_per_frame
        else:
            self.attn_param["chunk_size_tokens"] = 0

        # Forward attn_mode_override from extra_kwargs
        if extra_kwargs is not None:
            override = extra_kwargs.get("attn_mode_override", None)
            if override is not None:
                self.attn_param["attn_mode_override"] = override

            # Pre-compute DF-mode params so attention.py doesn't extract them inline.
            # Only apply the AR-chunk boundary conversion when true AR metadata
            # is present.  Validation loss passes a minimal diffusion_forcing_params
            # dict (enable only); treating it as AR metadata
            # would overwrite segment token boundaries with stale/default chunk
            # units and fail MagiAttention boundary validation.
            df_params = extra_kwargs.get("diffusion_forcing_params", None)
            if df_params is not None and (
                "num_ar_chunks" in df_params or "tokens_per_chunk" in df_params
            ):
                sequence_length = img.shape[1]
                num_ar_chunks = df_params.get("num_ar_chunks", 1)
                tpc = df_params.get("tokens_per_chunk", None)
                if tpc is None or tpc <= 0:
                    tpc = sequence_length // num_ar_chunks
                required = math.ceil(sequence_length / num_ar_chunks)
                if tpc < required:
                    tpc = required

                self.attn_param["df_num_ar_chunks"] = num_ar_chunks
                self.attn_param["df_tokens_per_chunk"] = tpc
                self.attn_param["df_txt_tokens_per_block"] = txt_tokens_per_block
                self.attn_param["diffusion_forcing_params"] = df_params

                # diffusion_forcing_magi_mp uses the same attention pattern as
                # magi_mp_causal, so set chunk_size_tokens and convert boundaries
                # to token units — both modes share the same dispatch path.
                if prompt_video_boundaries is not None:
                    from .modules.attention import (
                        align_prompt_video_boundaries_to_ar_window,
                    )

                    ar_ws = df_params.get("chunk_size", 5)
                    aligned = align_prompt_video_boundaries_to_ar_window(
                        prompt_video_boundaries, ar_ws
                    )
                    self.attn_param["chunk_size_tokens"] = tpc
                    self.attn_param["prompt_video_boundaries_tokens"] = [
                        b * tpc for b in [c // ar_ws for c in aligned]
                    ]

        # ========== Flatten 4D text to 3D before blocks ==========
        # When text has num_prompts != num_chunks, we cannot pass 4D text to the blocks
        # because the block asserts num_blocks_txt == num_blocks (= num_chunks from vec_3d).
        # Instead, flatten txt to [B, num_prompts*S, C] and let MagiMP attention mask
        # handle the window-based text-video assignment.
        if txt.dim() == 4:
            S_txt = txt.shape[2]
            txt = rearrange(txt, "b t s c -> b (t s) c")
            if text_mask.dim() == 3:
                text_mask = rearrange(text_mask, "b t s -> b (t s)")
            # Expand per-segment vec_txt [B, num_prompts, C] to per-text-token [B, num_prompts*S, C]
            if vec_txt.dim() == 3:
                # Each segment's vec is repeated S times for its S text tokens
                vec_txt = vec_txt.unsqueeze(2).expand(
                    -1, -1, S_txt, -1
                )  # [B, num_prompts, S, C]
                vec_txt = rearrange(
                    vec_txt, "b t s c -> b (t s) c"
                )  # [B, num_prompts*S, C]

        # Optional persistent text sequence parallelism.  The text refiner /
        # projection above still sees the full text sequence, but all expensive
        # double-block text operations (norm/modulation/QKV/proj/MLP) run only
        # on this rank's contiguous text-token shard. Attention converts both
        # img and txt from sequence-sharded to head-sharded via all-to-all and
        # reverses that layout after attention.
        text_sequence_parallel = (
            bool(
                extra_kwargs.get("text_sequence_parallel", False)
                if extra_kwargs
                else False
            )
            and get_sequence_parallel_state()
        )
        self.attn_param["text_sequence_parallel"] = text_sequence_parallel
        if text_sequence_parallel:
            if txt.dim() != 3:
                raise ValueError(
                    "text_sequence_parallel requires flattened 3D text "
                    f"[B,L_txt,C], got {tuple(txt.shape)}"
                )
            sp_size = nccl_info.sp_size
            sp_rank = nccl_info.rank_within_spgroup
            if txt.shape[1] % sp_size != 0:
                raise ValueError(
                    "text_sequence_parallel currently requires text length "
                    f"divisible by SP size: L_txt={txt.shape[1]}, sp={sp_size}"
                )
            txt = torch.chunk(txt, sp_size, dim=1)[sp_rank].contiguous()
            if vec_txt.dim() == 3:
                if vec_txt.shape[1] % sp_size != 0:
                    raise ValueError(
                        "Per-token vec_txt length must be divisible by SP size: "
                        f"L_txt={vec_txt.shape[1]}, sp={sp_size}"
                    )
                vec_txt = torch.chunk(vec_txt, sp_size, dim=1)[sp_rank].contiguous()

        # TP residuals stay sequence-sharded across all DiT blocks. Their
        # LayerNorms all-gather before QKV/MLP; masks and RoPE remain full here.
        img, txt = shard_transformer_streams(self, img, txt)

        # --------------------- Pass through DiT double blocks ------------------------
        # Discriminator feature hooks: collect post-block hidden states at the
        # requested layer indices (additive; only when out_contexts=True).
        contexts: Dict[str, torch.Tensor] = {}
        _hook_set = (
            set(hook_layers) if (out_contexts and hook_layers is not None) else None
        )
        for index, block in enumerate(self.double_blocks):
            force_full_attn = False
            sta_force_full_attn = (
                self.attn_mode in ["sta_flex", "ssta_flex", "ptm_sparse_attn"]
                and self.attn_param["win_type"] == "hybrid"
                and self.attn_param["win_ratio"] > 0
                and (
                    (index + 1) % self.attn_param["win_ratio"] == 0
                    or (index + 1) == len(self.double_blocks)
                )
            )
            block_force_full_attn = (
                self.attn_mode == "block_attn"
                and self.attn_param["ba_attn_full_attn_every_n_blocks"] > 0
                and (
                    (index + 1) % self.attn_param["ba_attn_full_attn_every_n_blocks"]
                    == 0
                    or (index + 1) == len(self.double_blocks)
                )
            )
            force_full_attn = sta_force_full_attn or block_force_full_attn
            self.attn_param["layer-name"] = f"double_block_{index + 1}"
            with _prof("dit/double_block"):
                img, txt = block(
                    img=img,
                    txt=txt,
                    vec=vec,
                    vec_txt=vec_txt,
                    freqs_cis=freqs_cis,
                    text_mask=text_mask,
                    attn_param=self.attn_param,
                    is_flash=force_full_attn,
                    txt_tokens_per_block=txt_tokens_per_block,
                    prompt_video_boundaries=prompt_video_boundaries,
                )

            # Camera: action_attn / extrinsics_attn inject via ActionModule after each block.
            if (
                camera_embedding is not None
                and self.camera_injection_mode in ("action_attn", "extrinsics_attn")
                and index < len(self.action_modules)
            ):
                with _prof("camera/action_module"):
                    if packed_action_meta is not None:
                        img = self.action_modules[index](
                            img,
                            tt,
                            th,
                            tw,
                            camera_embedding,
                            has_camera_mask=has_camera_mask,
                            segment_boundaries_latent=prompt_video_boundaries,
                            packed_img_token_counts=packed_action_meta[
                                "img_token_counts"
                            ],
                            packed_kb_token_counts=packed_action_meta[
                                "kb_token_counts"
                            ],
                            packed_sample_shapes=packed_action_meta["sample_shapes"],
                            packed_has_camera=packed_action_meta["has_camera"],
                            packed_segment_boundaries_latent=packed_action_meta.get(
                                "packed_segment_boundaries_latent"
                            ),
                            packed_has_camera_per_segment=packed_action_meta.get(
                                "has_camera_per_segment"
                            ),
                        )
                    else:
                        img = self.action_modules[index](
                            img,
                            tt,
                            th,
                            tw,
                            camera_embedding,
                            has_camera_mask=has_camera_mask,
                            segment_boundaries_latent=prompt_video_boundaries,
                        )

            if self.moe_config is not None:
                self.loss_moe += block.get_moe_loss()

            # Discriminator hook: capture the (post-block, post-camera) hidden
            # state for this layer. img is [B, L, hidden_size].
            if _hook_set is not None and index in _hook_set:
                contexts[str(index)] = (
                    gather_sequence(img, self.tensor_parallel_group,
                                    self.final_layer.norm_final.sequence_length)
                    if getattr(self, "tensor_parallel_size", 1) > 1 else img
                )

        # Discriminator path: return the collected per-layer features before the
        # final layer / unpatchify (mirrors Wan22 out_contexts short-circuit).
        #
        # Under SP, ``img`` was chunked along the token dim (token-packing branch
        # or the plain SP branch above), so the hooked features are only this
        # rank's 1/sp_size slab. Gather them back to full sequence length before
        # handing them to the discriminator head, exactly as the two non-hook
        # return paths below do. Without this each rank's disc sees a different
        # contiguous slab of (t,h,w) tokens and cannot penalise cross-slab
        # temporal incoherence; ``all_gather`` is the differentiable
        # ``_AllGather.apply``, so the generator's adversarial gradient is routed
        # back to the correct local shard.
        if out_contexts:
            if get_sequence_parallel_state():
                if nccl_info.use_dynamic_ring_attention:
                    # Ragged shards: needs the pad-gather-trim helper, not a
                    # bare all_gather (mirrors the non-hook path below).
                    contexts = {
                        key: postprocess_dynamic_ring_attention(value)
                        for key, value in contexts.items()
                    }
                else:
                    contexts = {
                        key: all_gather(value, dim=1, group=nccl_info.sp_group)
                        for key, value in contexts.items()
                    }
            return contexts

        if output_features:
            features_list = []

        # ---------------------------- Final layer ------------------------------
        # FinalLayerPerBlock supports both 2D [B, C] and 3D [B, T, C] condition
        img = self.final_layer(img, vec, tokens_per_frame=tokens_per_frame)

        # Token packing: skip unpatchify, return flat tokens [1, L_img, C_out].
        # Loss is computed in token space by the caller.
        if _tp_info is not None:
            assert return_dict == False, "return_dict is not supported."
            # When SP is active, img is [1, L_img_total/P, C_out] — gather
            # back to full packed sequence before returning to the caller.
            if get_sequence_parallel_state():
                img = all_gather(img, dim=1, group=nccl_info.sp_group)
            # img is [1, L_img_total, C_out] — flat packed tokens
            return (img, None)

        if get_sequence_parallel_state():
            if nccl_info.use_dynamic_ring_attention:
                img = postprocess_dynamic_ring_attention(img)
            else:
                img = all_gather(img, dim=1, group=nccl_info.sp_group)
        img = self.unpatchify(img, tt, th, tw)
        assert return_dict == False, "return_dict is not supported."
        if output_features:
            features_list = torch.stack(features_list, dim=0)
            if get_sequence_parallel_state():
                features_list = all_gather(features_list, dim=2, group=nccl_info.group)
        else:
            features_list = None
        return (img, features_list)

    @scoped_sparse_optim_forward
    def forward_with_cache(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        text_states: torch.Tensor,
        text_mask: torch.Tensor,
        kv_cache_list: Optional[List[Dict[str, torch.Tensor]]] = None,
        cache_valid_len: int = 0,
        update_cache: bool = False,
        global_frame_counter: int = 0,
        guidance=None,
        attn_mode_override: str = None,
        cache_txt: bool = True,
        cache_mask_list: Optional[List[torch.Tensor]] = None,
        # Cross-segment action-tail hiding (distillation seg-AR cache path).
        # ``current_action_start`` is the post-crop text offset where THIS
        # segment's per-action tail begins; the [start, text_len) span is
        # recorded on the cache mask when the segment is written, and hidden
        # again on every later segment's read. Defaults (-1 / False) keep the
        # cache mask untouched, so every existing caller is unaffected.
        hide_cached_action_tails: bool = False,
        current_action_start: int = -1,
        # Camera control (seg_ar / chunk_ar). ``camera_embedding`` is the
        # already-preprocessed keyboard for the CURRENT chunk only:
        #   - extrinsics_attn / action_attn: [B, N_frames_chunk, C_kb] where
        #     N_frames_chunk == tt (one keyboard token per latent frame of the
        #     chunk). Injected per double block via ActionModule (legacy full
        #     cross-attention — the chunk is a single segment, so no
        #     block-diagonal segmenting is needed within it).
        # ``camera_has_mask`` optionally gates the cross-attn output to 0 for
        # camera-off segments (joint bench segment_control == "prompt"). Shape
        # [B, 1] bool.
        camera_embedding: Optional[torch.Tensor] = None,
        camera_has_mask: Optional[torch.Tensor] = None,
        camera_inputs: Optional[Dict[str, torch.Tensor]] = None,
        # Intra-segment per-action masking (current-chunk self-attention only).
        # ``intra_seg_txt_offsets``: per-BATCH-sample [o_0,...,o_A] post-crop text
        #   offsets (None for a sample -> legacy full attention, e.g. CFG neg).
        # ``intra_seg_vid_windows``: [[vstart, vend, action_idx], ...] in
        #   segment-relative LATENT-FRAME coords (converted to tokens inside the
        #   attention using tokens_per_frame). Both None -> feature inert.
        intra_seg_txt_offsets: Optional[List[Optional[List[int]]]] = None,
        intra_seg_vid_windows: Optional[List[List[int]]] = None,
        camera_rope_time_offset: int = 0,
    ) -> Tuple[torch.Tensor, int]:
        """
        Forward pass with KV cache for segment-level AR generation.

        Processes ONE chunk at a time (single prompt, single chunk latents).
        Previous chunks are stored in the KV cache. Causality is implicit:
        only past chunks are in the cache, so full bidirectional attention
        within [cache + current_chunk] is equivalent to causal across chunks.

        Args:
            hidden_states: [B, C, T_chunk, H, W] — latent for current chunk only
            timestep: [B] scalar timestep
            text_states: [B, S, C] — text embedding for current segment only (3D)
            text_mask: [B, S] — text attention mask for current segment
            kv_cache_list: List of KV cache dicts, one per double block.
                Each dict has {"k": [B, max_len, H, D], "v": [B, max_len, H, D]}
            cache_valid_len: Number of valid **tokens** in cache
            update_cache: If True, write current chunk's KV to cache
            global_frame_counter: Total latent frames generated so far (for RoPE offset)
            guidance: Guidance tensor (default: 6016.0)
            attn_mode_override: Override attention mode (e.g. "flash" for cache mode)
            cache_txt: If True, include txt KV in cache update. Should be True
                only for the first chunk of each segment to avoid duplicating
                text tokens in the cache (matching training behavior).
            camera_rope_time_offset: Optional latent-frame offset for camera
                ActionModule RoPE in AR cache inference. Defaults to 0 to keep
                legacy local camera RoPE behavior.

        Returns:
            (output_img, new_cache_tokens): The denoised output [B, C, T_chunk, H, W]
                and the number of tokens added to cache (img, or img+txt) if update_cache.
        """
        if guidance is None:
            guidance = torch.tensor(
                [6016.0], device=hidden_states.device, dtype=torch.bfloat16
            )

        img = hidden_states
        t = timestep  # [B]
        bs, _, ot, oh, ow = img.shape
        tt = ot // self.patch_size[0]
        th = oh // self.patch_size[1]
        tw = ow // self.patch_size[2]

        tokens_per_frame = th * tw

        # Hide previously-cached per-action text tails so the current segment
        # attends only to earlier segments' HEAD text. The ranges were recorded
        # when those segments were written (see the update_cache block below).
        if (
            hide_cached_action_tails
            and cache_mask_list is not None
            and cache_valid_len > 0
        ):
            for cache_mask in cache_mask_list:
                for start, end in getattr(cache_mask, "_action_tail_ranges", []):
                    cache_mask[:, int(start) : int(end)] = False

        # RoPE: use the same function as forward() when offset=0 to ensure
        # bit-exact results. Only use the offset variant for subsequent chunks.
        if global_frame_counter == 0:
            freqs_cos, freqs_sin = self.get_rotary_pos_embed(
                (tt, th, tw), device=hidden_states.device
            )
        else:
            freqs_cos, freqs_sin = get_rotary_pos_embed_with_offset(
                rope_sizes=(tt, th, tw),
                time_offset=global_frame_counter,
                hidden_size=self.hidden_size,
                heads_num=self.heads_num,
                rope_dim_list=self.rope_dim_list,
                rope_theta=self.rope_theta,
            )

        # Patch embedding
        img = self.img_in(img)

        if camera_inputs is not None:
            if camera_embedding is not None:
                raise ValueError(
                    "forward_with_cache received both camera_embedding and "
                    "camera_inputs; pass exactly one camera representation."
                )
            if self.camera_injection_mode not in (
                "action_attn",
                "extrinsics_attn",
            ):
                raise NotImplementedError(
                    "KV-cache camera preprocessing supports action_attn and "
                    f"extrinsics_attn, got {self.camera_injection_mode!r}."
                )
            camera_embedding = self._preprocess_camera(
                norm_extrinsics=camera_inputs.get("norm_extrinsics"),
                norm_intrinsics=camera_inputs.get("norm_intrinsics"),
                action_onehot=camera_inputs.get("action_onehot"),
                oh=oh,
                ow=ow,
                img_dtype=img.dtype,
                device=img.device,
            )
            if camera_embedding.shape[1] != tt:
                raise ValueError(
                    "KV-cache camera/latent alignment mismatch: "
                    f"camera_tokens={camera_embedding.shape[1]}, "
                    f"latent_frames={tt}"
                )

        # Sequence parallel
        if get_sequence_parallel_state():
            sp_size = nccl_info.sp_size
            sp_rank = nccl_info.rank_within_spgroup
            if nccl_info.use_dynamic_ring_attention:
                _text_mask_for_sp = text_mask
                img, freqs_cos, freqs_sin = preprocess_dynamic_ring_attention(
                    img, _text_mask_for_sp, freqs_cos, freqs_sin
                )
            else:
                img = torch.chunk(img, sp_size, dim=1)[sp_rank]
                freqs_cos = torch.chunk(freqs_cos, sp_size, dim=0)[sp_rank]
                freqs_sin = torch.chunk(freqs_sin, sp_size, dim=0)[sp_rank]

        # Timestep embedding: [B] -> [B, C]
        vec = self.time_in(t)
        vec_txt = vec  # Same timestep for text modulation

        # Text projection: standard 3D [B, S, C] path
        txt = text_states
        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(
                txt, t, text_mask if self.use_attention_mask else None
            )
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        # Keep original text length for cache/token bookkeeping; TP residual
        # shards may be shorter than one token or empty on some ranks.
        cache_text_length = txt.shape[1]
        img, txt = shard_transformer_streams(self, img, txt)
        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

        # Use flash3 attention for KV cache mode.
        # Prefer the model's own attn_mode if it's flash3, otherwise default to flash3.
        if attn_mode_override:
            cache_attn_mode = attn_mode_override
        else:
            cache_attn_mode = "flash3"

        # Pass through double blocks with KV cache
        # NOTE: Call block(...) (i.e. __call__) instead of block.forward_with_cache()
        # so FSDP2 unshard hooks fire at the block level.  The block's forward()
        # detects kv_cache != None and delegates to forward_with_cache internally.
        # A fresh segment/request must never reuse routes from an earlier rollout.
        if not update_cache and getattr(self, "_sparse_step_index", 0) == 0:
            for cache in kv_cache_list or []:
                cache.pop("route_reuse", None)
        for index, block in enumerate(self.double_blocks):
            block_kv_cache = kv_cache_list[index] if kv_cache_list is not None else None
            block_cache_mask = (
                cache_mask_list[index] if cache_mask_list is not None else None
            )

            with _prof("dit/double_block"):
                img, txt = block(
                    img=img,
                    txt=txt,
                    vec=vec,
                    vec_txt=vec_txt,
                    freqs_cis=freqs_cis,
                    text_mask=text_mask,
                    kv_cache=block_kv_cache,
                    cache_valid_len=cache_valid_len,
                    update_cache=update_cache,
                    cache_attn_mode=cache_attn_mode,
                    cache_txt=cache_txt,
                    cache_mask=block_cache_mask,
                    intra_seg_txt_offsets=intra_seg_txt_offsets,
                    intra_seg_vid_windows=intra_seg_vid_windows,
                    intra_seg_tokens_per_frame=tokens_per_frame,
                    sparse_context=(
                        dict(config=self.nomas_sparse_config, grid=(tt, th, tw),
                             time_start=global_frame_counter, layer=index,
                             step_index=getattr(self, "_sparse_step_index", 0),
                             dense=(not update_cache and getattr(self, "_sparse_step_index", 0)
                                    < self.nomas_sparse_config.get("dense_first_n_steps", 2)))
                        if getattr(self, "nomas_sparse_config", {}).get("enabled", False) else None
                    ),
                )

            # Camera: inject per block via ActionModule (action_attn /
            # extrinsics_attn). The chunk is a single segment, so we use the
            # legacy full cross-attention path (segment_boundaries_latent=None):
            # every image token of the chunk attends to that chunk's keyboard
            # tokens. ``img`` is the SP-local chunk here; ActionModule is
            # SP-aware (it recomputes the local token window from tt/th/tw).
            if (
                camera_embedding is not None
                and self.camera_injection_mode in ("action_attn", "extrinsics_attn")
                and index < len(self.action_modules)
            ):
                with _prof("camera/action_module"):
                    img = self.action_modules[index](
                        img,
                        tt,
                        th,
                        tw,
                        camera_embedding,
                        has_camera_mask=camera_has_mask,
                        segment_boundaries_latent=None,
                        rope_time_offset=camera_rope_time_offset,
                    )

        tp_enabled = getattr(self, "tensor_parallel_size", 1) > 1
        # Preserve the TP1 path: gather SP tokens before the output layer.
        if get_sequence_parallel_state() and not tp_enabled:
            if nccl_info.use_dynamic_ring_attention:
                img = postprocess_dynamic_ring_attention(img)
            else:
                img = all_gather(img, dim=1, group=nccl_info.sp_group)

        # With TP, final LayerNorm normalizes local tokens and gathers them
        # before the output projection; restore the SP dimension afterwards.
        img = self.final_layer(img, vec, tokens_per_frame=tokens_per_frame)
        if get_sequence_parallel_state() and tp_enabled:
            img = all_gather(img, dim=1, group=nccl_info.sp_group)
        img = self.unpatchify(img, tt, th, tw)

        # Calculate tokens added to cache
        if update_cache:
            img_tokens = tt * tokens_per_frame
            txt_tokens = cache_text_length if cache_txt else 0
            # Account for SP: after all_to_all, img tokens are full but heads are sharded
            if get_sequence_parallel_state():
                new_cache_tokens = img_tokens + txt_tokens
            else:
                new_cache_tokens = img_tokens + txt_tokens
        else:
            new_cache_tokens = 0

        # Record this segment's per-action text tail on the cache mask so later
        # segments can hide it (see the hide_cached_action_tails block above).
        # Coordinates are absolute cache-token indices, matching cache_mask.
        if (
            update_cache
            and cache_txt
            and cache_mask_list is not None
            and current_action_start is not None
            and int(current_action_start) >= 0
            and not (kv_cache_list and "swa_capacity" in kv_cache_list[0])
        ):
            text_len = int(cache_text_length)
            tail_start = (
                int(cache_valid_len)
                + int(tt * tokens_per_frame)
                + min(int(current_action_start), text_len)
            )
            tail_end = int(cache_valid_len) + int(tt * tokens_per_frame) + text_len
            if tail_end > tail_start:
                for cache_mask in cache_mask_list:
                    ranges = list(getattr(cache_mask, "_action_tail_ranges", []))
                    if (tail_start, tail_end) not in ranges:
                        ranges.append((tail_start, tail_end))
                    cache_mask._action_tail_ranges = ranges

        return (img, new_cache_tokens)

    def get_moe_loss(self):
        return self.loss_moe

    def _refine_packed_text(
        self,
        txt: torch.Tensor,
        text_mask: torch.Tensor,
        txt_t: torch.Tensor,
        tp_info: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process packed text through SingleTokenRefiner with per-sample isolation.

        When token packing is active, all packed samples' text tokens are
        concatenated into a single flat sequence ``[1, S_total, C]``.  Passing
        this directly through the refiner lets its self-attention mix text
        tokens from unrelated samples.

        This helper splits the flat text back into per-sample (or per-prompt)
        blocks, runs the refiner on each block independently, and re-
        concatenates the result.

        Returns ``(txt, text_mask)`` both as 3D/2D tensors suitable for the
        packed attention path that follows.
        """
        txt_counts = tp_info["txt_token_counts"]

        assert txt.dim() == 3, (
            f"_refine_packed_text expects 3D txt [1, S_total, C], got {txt.shape}. "
            "The 4D wrap in text format detection should be skipped for token packing."
        )
        txt_3d = txt  # [1, S_total, C_in]
        mask_2d = text_mask if text_mask.dim() == 2 else text_mask.view(1, -1)

        # txt_t carries the per-block refiner timestep(s).
        #   - CPT scalar path: txt_t is a scalar (or
        #     broadcastable) tensor shared by every block -> t_scalar.
        #   - segment-level DF: txt_t is a
        #     per-prompt vector [total_prompts] in sample-major segment order,
        #     i.e. the SAME order the blocks are batched below, so we feed each
        #     block its own timestep instead of a shared scalar.
        _txt_t_flat = txt_t.reshape(-1)
        t_scalar = _txt_t_flat[0]
        _per_prompt_t = _txt_t_flat if _txt_t_flat.numel() > 1 else None

        per_sample_magi = tp_info.get("per_sample_magi_params", None)
        is_multi_prompt = per_sample_magi is not None and any(
            p.get("num_prompts", 1) > 1 for p in per_sample_magi
        )

        if is_multi_prompt:
            # Each sample has actual_np prompts × S_enc tokens.
            # Reshape into [total_prompts, S_enc, C] and batch through refiner.
            S_enc = per_sample_magi[0]["txt_tokens_per_block"]
            all_blocks, all_masks = [], []
            seg_id_chunks = []
            offset = 0
            for count in txt_counts:
                n_p = count // S_enc
                all_blocks.append(
                    txt_3d[0, offset : offset + count].view(n_p, S_enc, -1)
                )
                all_masks.append(mask_2d[0, offset : offset + count].view(n_p, S_enc))
                # Within each sample the prompts are stored in segment order,
                # so segment ids for the batched view are simply 0..n_p-1
                # repeated per sample. This matches the auto-injection on the
                # non-packed 4D path where seg id == chunk index.
                seg_id_chunks.append(torch.arange(n_p, dtype=torch.long))
                offset += count

            batched = torch.cat(all_blocks, dim=0)
            batched_mask = torch.cat(all_masks, dim=0)
            if _per_prompt_t is not None:
                # Each batched block gets its own segment timestep. The block
                # order here (sample-major over prompts) matches how
                # _per_prompt_t was assembled in the forward.
                assert _per_prompt_t.numel() == batched.shape[0], (
                    f"per-prompt timestep count ({_per_prompt_t.numel()}) must "
                    f"match number of text blocks ({batched.shape[0]})"
                )
                t_refiner = _per_prompt_t.to(device=batched.device, dtype=batched.dtype)
            else:
                t_refiner = t_scalar.expand(batched.shape[0])

            # Build explicit segment ids so SingleTokenRefinerPerBlock can
            # inject seg_pos_emb on this flattened 3D path (its automatic
            # injection only fires when ``t`` is 2D, which is not the case
            # here because we have already flattened segments into batch).
            seg_ids = torch.cat(seg_id_chunks, dim=0).to(device=batched.device)

            refined = self.txt_in(
                batched,
                t_refiner,
                batched_mask if self.use_attention_mask else None,
                seg_ids=seg_ids,
            )
            txt_out = refined.reshape(1, -1, refined.shape[-1])
        else:
            # Single-prompt packed: variable-length per sample.
            # Pad to max length, batch through refiner, then unpad.
            max_len = max(txt_counts)
            N = len(txt_counts)
            C_in = txt_3d.shape[-1]

            padded = torch.zeros(
                N, max_len, C_in, device=txt_3d.device, dtype=txt_3d.dtype
            )
            padded_mask = torch.zeros(
                N, max_len, device=mask_2d.device, dtype=mask_2d.dtype
            )
            offset = 0
            for i, count in enumerate(txt_counts):
                padded[i, :count] = txt_3d[0, offset : offset + count]
                padded_mask[i, :count] = mask_2d[0, offset : offset + count]
                offset += count

            t_refiner = t_scalar.expand(N)

            # Single-prompt path: each sample carries exactly one segment,
            # so every row maps to segment id 0 in the seg_pos_emb table.
            # This matches the legacy non-packed 3D single-segment behavior
            # (which skipped injection entirely) up to a constant zero-init
            # bias; with zero-init weights this is a no-op at step 0 and
            # learns a global "segment-0" bias afterwards.
            seg_ids = torch.zeros(N, dtype=torch.long, device=padded.device)

            refined = self.txt_in(
                padded,
                t_refiner,
                padded_mask if self.use_attention_mask else None,
                seg_ids=seg_ids,
            )
            chunks = [refined[i, : txt_counts[i]] for i in range(N)]
            txt_out = torch.cat(chunks, dim=0).unsqueeze(0)

        return txt_out, mask_2d

    def unpatchify(self, x, t, h, w):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.unpatchify_channels
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
        x = torch.einsum("nthwcopq->nctohpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))

        return imgs

    def params_count(self):
        counts = {
            "double": sum(
                [
                    sum(p.numel() for p in block.img_attn_q.parameters())
                    + sum(p.numel() for p in block.img_attn_k.parameters())
                    + sum(p.numel() for p in block.img_attn_v.parameters())
                    + sum(p.numel() for p in block.img_attn_proj.parameters())
                    + sum(p.numel() for p in block.img_mlp.parameters())
                    + sum(p.numel() for p in block.txt_attn_q.parameters())
                    + sum(p.numel() for p in block.txt_attn_k.parameters())
                    + sum(p.numel() for p in block.txt_attn_v.parameters())
                    + sum(p.numel() for p in block.txt_attn_proj.parameters())
                    + sum(p.numel() for p in block.txt_mlp.parameters())
                    for block in self.double_blocks
                ]
            ),
            "single": sum(
                [
                    sum(p.numel() for p in block.linear1_q.parameters())
                    + sum(p.numel() for p in block.linear1_k.parameters())
                    + sum(p.numel() for p in block.linear1_v.parameters())
                    + sum(p.numel() for p in block.linear1_mlp.parameters())
                    + sum(p.numel() for p in block.linear2.parameters())
                    for block in self.single_blocks
                ]
            ),
            "total": sum(p.numel() for p in self.parameters()),
        }
        counts["attn+mlp"] = counts["double"] + counts["single"]
        return counts
