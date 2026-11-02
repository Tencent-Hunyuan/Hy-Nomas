"""Load optional Magi kernels for the active GPU architecture."""

from functools import lru_cache
import importlib.util

import torch


def magi_available():
    try:
        return importlib.util.find_spec("magi_attention") is not None
    except (ImportError, ValueError):
        return False


@lru_cache(maxsize=2)
def _magi_kernel(blackwell):
    try:
        if blackwell:
            from magi_attention.kernel.cutedsl import flex_flash_attn_func
            return flex_flash_attn_func
        from magi_attention.functional import flex_flash_attn_func
    except ImportError as exc:
        raise RuntimeError("The selected Magi backend requires MagiAttention") from exc
    return flex_flash_attn_func


def magi_kernel():
    return magi_attention


def _blackwell_magi(q, k, v, q_ranges=None, k_ranges=None, attn_type_map=None,
                    *, softmax_scale=None, deterministic=False, **kwargs):
    return _magi_kernel(True)(
        q, k, v, q_ranges=q_ranges, k_ranges=k_ranges,
        mask_types=attn_type_map, softmax_scale=softmax_scale,
        deterministic=deterministic, **kwargs,
    )


def magi_attention(*args, **kwargs):
    q = args[0] if args else kwargs["q"]
    blackwell = q.is_cuda and torch.cuda.get_device_capability(q.device)[0] >= 10
    kernel = _blackwell_magi if blackwell else _magi_kernel(False)
    return kernel(*args, **kwargs)


def profile_flash_attention(*args, **kwargs):
    try:
        from flash_attn_interface import flash_attn_func
    except ImportError:
        from flash_attn import flash_attn_func
    return flash_attn_func(*args, **kwargs)
