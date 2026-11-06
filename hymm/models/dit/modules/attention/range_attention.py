"""Selectable exact range-attention kernels; Magi remains the default.

Flex supports inference-only BF16 MHA and type-zero rectangle masks. No silent
fallback. Mask caches retain only range tensors/indices, never QKV/history.
"""

from __future__ import annotations

import hashlib
import time
from collections import Counter, OrderedDict
from functools import lru_cache

import torch

from .range_mask import RangeSpec

_MASKS = OrderedDict()
_STATS = Counter()
_MAX_MASKS = 32
_PHASE = "other"


def set_range_phase(phase):
    global _PHASE
    previous = _PHASE
    _PHASE = phase
    return previous


def note_magi_call():
    _STATS["magi_calls"] += 1
    _STATS["magi/" + _PHASE] += 1


def note_non_range_call():
    _STATS["flash3_non_range_calls"] += 1
    _STATS["flash3_non_range/" + _PHASE] += 1


def unpad_positions(mask):
    """Valid-token positions of a [B, S] mask, for ``unpad_tokens``, and their cu_seqlens."""
    lengths = mask.sum(dim=-1, dtype=torch.int32)
    cu_seqlens = torch.cat([lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)])
    return mask.flatten().nonzero().flatten(), cu_seqlens


def unpad_tokens(x, positions):
    """[B, S, H, D] to [nnz, H, D], as flash_attn's unpad_input without its per-element index."""
    return x.flatten(0, 1).index_select(0, positions)


def pad_tokens(x, positions, batch, length):
    """Inverse of ``unpad_tokens``; padding rows are zero."""
    out = x.new_zeros((batch * length, *x.shape[1:]))
    out[positions] = x
    return out.reshape(batch, length, *x.shape[1:])


@lru_cache(maxsize=1)
def compiled_flex():
    from torch.nn.attention.flex_attention import flex_attention

    return torch.compile(flex_attention, dynamic=True, fullgraph=True)


def clear_range_cache():
    _MASKS.clear()
    # Legacy builders also retain range metadata; bound them by sample lifetime
    # without touching their training/default-Magi behavior.
    import sys
    for module_name, cache_names in (
        ("hymm.models.dit.nomas_cpt", ("_intra_seg_range_cache",)),
        ("hymm.models.dit.modules.attention.magi_attn", ("_magi_mask_cache",)),
        ("hymm.models.camera.action_module", ("_segmented_attn_range_cache", "_packed_seg_flex_range_cache")),
    ):
        module = sys.modules.get(module_name)
        if module is not None:
            for name in cache_names:
                cache = getattr(module, name, None)
                if cache is not None:
                    cache.clear()


def reset_range_stats():
    _STATS.clear()


def range_stats():
    return dict(_STATS, cached_masks=len(_MASKS))


def _cached_mask(q, k, qr, kr, types, query_mask=None, key_mask=None):
    start = time.perf_counter()
    spec = RangeSpec(q.shape[0], k.shape[0], qr, kr, types)
    key = (spec.fingerprint, q.shape[0], k.shape[0], str(q.device))
    if query_mask is not None:
        query_mask = query_mask.detach().cpu().numpy()
        key_mask = key_mask.detach().cpu().numpy()
        key += (hashlib.sha256(query_mask.tobytes() + key_mask.tobytes()).digest(),)
    if key in _MASKS:
        _STATS["mask_cache_hits"] += 1
        _MASKS.move_to_end(key)
        return _MASKS[key]
    if spec.overlaps():
        raise ValueError(
            "Overlapping QK rectangles require multiplicity semantics; refusing conversion"
        )
    mask = spec.block_mask(device=q.device, query_mask=query_mask, key_mask=key_mask)
    _MASKS[key] = mask
    while len(_MASKS) > _MAX_MASKS:
        _MASKS.popitem(last=False)
    _STATS["mask_build_seconds"] += time.perf_counter() - start
    _STATS["mask_builds"] += 1
    return mask


def range_attention(
    q,
    k,
    v,
    q_ranges,
    k_ranges,
    attn_type_map=None,
    *,
    backend="magi",
    softmax_scale=None,
    deterministic=False,
    scope="dit",
    query_mask=None,
    key_mask=None,
    **kwargs,
):
    if scope not in ("dit", "camera"):
        raise ValueError(f"Invalid attention audit scope: {scope}")
    if backend not in ("magi", "flex"):
        raise ValueError(f"Unknown range attention backend {backend!r}")
    if backend == "magi":
        if query_mask is not None or key_mask is not None:
            raise ValueError("Token visibility masks require the Flex backend")
        from .backends import magi_kernel

        if scope == "dit":
            note_magi_call()
        else:
            _STATS["camera_magi_calls"] += 1
        return magi_kernel()(
            q=q,
            k=k,
            v=v,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            softmax_scale=softmax_scale,
            deterministic=deterministic,
            **kwargs,
        )
    if kwargs:
        raise ValueError(f"Unsupported Flex arguments: {sorted(kwargs)}")
    if torch.is_grad_enabled():
        raise RuntimeError("Flex range backend supports no-grad inference only")
    if any(
        x.ndim != 3 or x.dtype != torch.bfloat16 or x.device != q.device
        for x in (q, k, v)
    ):
        raise ValueError("Flex expects BF16 [S,H,D] tensors on one device")
    if (
        q.device.type != "cuda"
        or k.shape != v.shape
        or q.shape[1:] != k.shape[1:]
        or q.shape[-1] != 128
    ):
        raise ValueError("Flex range backend requires CUDA MHA D=128 and matching K/V")
    prefix = "flex" if scope == "dit" else "camera_flex"
    if (query_mask is None) != (key_mask is None):
        raise ValueError("Query and key visibility must be supplied together")
    if query_mask is not None:
        for mask, tensor in ((query_mask, q), (key_mask, k)):
            if mask.shape != tensor.shape[:1] or mask.dtype != torch.bool or mask.device != q.device:
                raise ValueError("Invalid token visibility mask")
    _STATS[prefix + "_calls"] += 1
    _STATS[prefix + "/" + _PHASE] += 1
    if not q.shape[0] or not k.shape[0]:
        spec = RangeSpec(q.shape[0], k.shape[0], q_ranges, k_ranges, attn_type_map)
        if spec.overlaps():
            raise ValueError("Overlapping QK rectangles require multiplicity semantics")
        return torch.zeros_like(q), None
    mask = _cached_mask(q, k, q_ranges, k_ranges, attn_type_map, query_mask, key_mask)
    qq, kk, vv = (x.transpose(0, 1).unsqueeze(0) for x in (q, k, v))
    out = compiled_flex()(
        qq,
        kk,
        vv,
        block_mask=mask,
        scale=softmax_scale,
        kernel_options={"ROWS_GUARANTEED_SAFE": False, "BLOCKS_ARE_CONTIGUOUS": False},
    )
    return out.squeeze(0).transpose(0, 1).contiguous(), None


def run_range_attention(q, k, v, *, range_spec, backend, softmax_scale=None):
    import numpy as np

    qr = torch.from_numpy(np.array(range_spec.q_ranges, copy=True)).to(q.device)
    kr = torch.from_numpy(np.array(range_spec.k_ranges, copy=True)).to(q.device)
    types = torch.from_numpy(np.array(range_spec.attn_type_map, copy=True)).to(q.device)
    return range_attention(
        q, k, v, qr, kr, types, backend=backend, softmax_scale=softmax_scale
    )[0]


def flex_masked_attention(q, k, v, query_mask, key_mask):
    """Full attention over valid tokens, with independent Q/K lengths and holes.

    This is the BF16 action-free cached path. Query and key visibility are
    independent; masked query rows remain zero.
    """
    if (
        q.ndim != 4
        or k.ndim != 4
        or v.shape != k.shape
        or q.shape[0] != k.shape[0]
        or q.shape[2:] != k.shape[2:]
        or query_mask.shape != q.shape[:2]
        or key_mask.shape != k.shape[:2]
        or query_mask.dtype != torch.bool
        or key_mask.dtype != torch.bool
        or query_mask.device != q.device
        or key_mask.device != q.device
    ):
        raise ValueError("Invalid padded Flex QKV/mask layout")
    output = torch.empty_like(q)
    for batch in range(q.shape[0]):
        result, _ = range_attention(
            q[batch], k[batch], v[batch], [[0, q.shape[1]]], [[0, k.shape[1]]],
            backend="flex", query_mask=query_mask[batch], key_mask=key_mask[batch],
        )
        output[batch].copy_(result)
    return output


def expand_compact_ranges(ranges, positions):
    """Map compact interval endpoints to original token positions, including holes."""
    ranges = ranges.reshape(-1, 2).long()
    if not positions.numel():
        return torch.zeros_like(ranges, dtype=torch.int32)
    nonempty = ranges[:, 1] > ranges[:, 0]
    starts = positions[ranges[:, 0].clamp(max=positions.numel() - 1)]
    ends = positions[(ranges[:, 1] - 1).clamp(min=0)] + 1
    return torch.stack((starts, torch.where(nonempty, ends, starts)), 1).to(torch.int32)


def flex_camera_range_attention(q, k, v, q_ranges, k_ranges, attn_type_map=None):
    """Keep camera range geometry and 1/sqrt(original_dim) scaling unchanged.

    Camera heads are D=64; zero-padding Q/K/V to the already validated D=128
    kernel is exact algebraically when the original scale is passed explicitly.
    Camera calls use separate audit counters from the DiT self-attention calls.
    """
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape or q.shape[1:] != k.shape[1:]:
        raise ValueError("Invalid camera QKV shape")
    dim = q.shape[-1]
    if dim not in (64, 128):
        raise ValueError("Flex camera attention supports head dimension 64 or 128")
    if dim == 64:
        q, k, v = (torch.nn.functional.pad(x, (0, 64)) for x in (q, k, v))
    result, _ = range_attention(
        q,
        k,
        v,
        q_ranges,
        k_ranges,
        attn_type_map,
        backend="flex",
        softmax_scale=dim**-0.5,
        scope="camera",
    )
    return result[..., :dim].contiguous(), None
