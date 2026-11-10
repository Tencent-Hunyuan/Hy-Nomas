"""FP8 (E4M3) KV cache read directly by the Magi SM100 FP8 forward.

K and V are stored as ``x / descale`` with one power-of-two descale per head. A history
written at different times is read by one kernel call, so the descales are frozen when
the first segment is written and later values are clamped to the representable range.
Q is quantized per call. The kernel folds the Q/K descales into the softmax scale and
multiplies the output by the V descale; the output stays BF16.

Dense full attention only: no sliding window, sparse routing, CFG history broadcast or
multi-chunk segments.

This cache is not an ``inference.kv_cache_quantization`` choice: it is what
``inference.dit_weight_quantization=fp8`` uses, and nothing else selects it.
"""
from functools import lru_cache

import torch

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
# Frozen K/V descales leave this much room above the first segment's largest value.
HEADROOM = 4.0
K_DESCALE, V_DESCALE = "fp8_k_descale", "fp8_v_descale"


def validate_fp8_mode(args):
    """Reject configurations the FP8 DiT (FP8 projections + FP8 KV cache) does not cover."""
    inference, problems = args.inference, []
    if getattr(inference, "kv_cache_quantization", "none") != "none":
        problems.append("inference.kv_cache_quantization must stay none (the FP8 DiT sets its own FP8 KV cache)")
    if getattr(inference, "rollout_type", "seg_ar") != "seg_ar":
        problems.append("inference.rollout_type must be seg_ar")
    if getattr(getattr(inference, "sliding_window", None), "enabled", False):
        problems.append("inference.sliding_window.enabled must be false")
    if getattr(getattr(inference, "sparse_attention", None), "enabled", False):
        problems.append("inference.sparse_attention.enabled must be false")
    if float(getattr(inference, "cfg_scale_video", 1.0)) != 1.0 or float(getattr(inference, "camera_cfg_scale", 1.0)) != 1.0:
        problems.append("cfg_scale_video and camera_cfg_scale must be 1.0")
    if getattr(inference, "interactive_kv_cache_layerwise_cpu_offload", False):
        problems.append("inference.interactive_kv_cache_layerwise_cpu_offload must be false")
    if getattr(args.dit, "seg_ar_attention_backend", "magi") != "magi":
        problems.append("model.dit.seg_ar_attention_backend must be magi")
    if int(getattr(args.parallelism, "sp_size", 1)) != 1:
        problems.append("parallelism.sp_size must be 1 (tensor parallel is supported)")
    if problems:
        raise ValueError("inference.dit_weight_quantization=fp8 is unsupported here: " + "; ".join(problems))
    if torch.cuda.is_available():
        if torch.cuda.get_device_capability()[0] != 10:
            raise RuntimeError("The FP8 DiT requires Blackwell SM100/SM103")
        _kernel()


def head_descale(x, headroom=1.0):
    """Power-of-two descale per head for [..., H, D]."""
    dims = tuple(i for i in range(x.ndim) if i != x.ndim - 2)
    amax = x.detach().abs().amax(dims).float().clamp_min(1e-6) * headroom
    return torch.exp2(torch.ceil(torch.log2(amax / FP8_MAX))).contiguous()


def quantize(x, descale):
    """BF16 [..., H, D] to FP8 with a per-head descale; values past the range saturate."""
    scaled = x / descale.to(x.dtype).view(-1, 1)
    return scaled.clamp_(-FP8_MAX, FP8_MAX).to(FP8)


def cache_descales(cache, cur_k, cur_v, cache_valid_len):
    """Frozen K/V descales of this layer; (re)derived while the cache is empty."""
    if cache_valid_len == 0 or K_DESCALE not in cache:
        if cache_valid_len:
            raise RuntimeError("FP8 KV cache holds history but no descales")
        cache[K_DESCALE] = head_descale(cur_k, HEADROOM)
        cache[V_DESCALE] = head_descale(cur_v, HEADROOM)
    return cache[K_DESCALE], cache[V_DESCALE]


def write_current(cache, cur_k, cur_v, cache_valid_len):
    """Quantize the current K/V in place behind the history; returns the FP8 views read by attention."""
    if cur_k.shape[0] != 1 or cache["k"].shape[0] != 1:
        raise ValueError("FP8 KV cache supports batch 1 (CFG=1) only")
    end = cache_valid_len + cur_k.shape[1]
    if end > cache["k"].shape[1]:
        raise ValueError(f"FP8 KV cache of {cache['k'].shape[1]} tokens has no room for {end}")
    k_descale, v_descale = cache_descales(cache, cur_k, cur_v, cache_valid_len)
    cache["k"][:, cache_valid_len:end] = quantize(cur_k, k_descale)
    cache["v"][:, cache_valid_len:end] = quantize(cur_v, v_descale)
    return cache["k"][:, :end], cache["v"][:, :end]


@lru_cache(maxsize=1)
def _kernel():
    import inspect
    from magi_attention.kernel.cutedsl import flex_flash_attn_func

    if "q_descale" not in inspect.signature(flex_flash_attn_func).parameters:
        raise RuntimeError("FP8 KV cache requires a MagiAttention build with the FP8 SM100 forward")
    return flex_flash_attn_func


def _gather(x, positions):
    """unpad_tokens for FP8 (indexing goes through the raw bytes)."""
    flat = x.flatten(0, 1)
    if positions.shape[0] == flat.shape[0]:
        return flat
    return flat.view(torch.uint8).index_select(0, positions).view(FP8)


def fp8_dense_attention(q_unpad, full_k, full_v, k_indices, cu_seqlens_q, cu_seqlens_k,
                        max_seqlen_q, max_seqlen_k, cache):
    """Unpadded BF16 queries [nq, H, D] over the FP8 cache; returns BF16 [nq, H, D]."""
    if torch.cuda.get_device_capability(q_unpad.device)[0] != 10:
        raise RuntimeError("FP8 KV cache attention requires Blackwell SM100/SM103")
    q_descale = head_descale(q_unpad)
    q_ranges = torch.stack([cu_seqlens_q[:-1], cu_seqlens_q[1:]], 1).to(torch.int32)
    k_ranges = torch.stack([cu_seqlens_k[:-1], cu_seqlens_k[1:]], 1).to(torch.int32)
    out, _ = _kernel()(
        quantize(q_unpad, q_descale), _gather(full_k, k_indices), _gather(full_v, k_indices),
        q_ranges=q_ranges, k_ranges=k_ranges, mask_types=0,
        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
        softmax_scale=q_unpad.shape[-1] ** -0.5, disable_fwd_atomic_reduction=True,
        out_dtype=torch.bfloat16,
        q_descale=q_descale, k_descale=cache[K_DESCALE], v_descale=cache[V_DESCALE],
    )
    return out
