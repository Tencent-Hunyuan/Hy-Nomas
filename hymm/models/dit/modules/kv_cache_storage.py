"""Storage dispatch shared by dense and quantized inference caches."""

import torch


def cache_format(cache):
    mode = cache.get("storage_format")
    if mode is not None:
        return mode
    if cache["k"].dtype == torch.float8_e4m3fn:
        return "fp8"
    return "int8" if cache["k"].dtype == torch.int8 else "none"


def cache_geometry(cache):
    if cache_format(cache) == "qvg":
        return cache["shape"], cache["device"], torch.bfloat16
    return cache["k"].shape, cache["k"].device, cache["k"].dtype


def quantization_mode(inference):
    mode = getattr(inference, "kv_cache_quantization", "none")
    if mode not in ("none", "int8", "qvg"):
        raise ValueError("inference.kv_cache_quantization must be none, int8, or qvg")
    return mode


def validate_kv_cache_config(args):
    inf = args.inference
    if quantization_mode(inf) != "qvg":
        return
    from .qvg_kv_cache import QVGOptions

    QVGOptions.from_config(getattr(inf, "qvg_kv_cache", {}))
    if (getattr(inf, "rollout_type", None) != "seg_ar"
            or getattr(args.dit, "seg_ar_attention_backend", None) != "flex"
            or getattr(args.dit, "master_weight_type", None) != "bf16"
            or getattr(args.parallelism, "sp_size", 1) != 1
            or getattr(inf, "cfg_scale_video", 1.0) != 1.0
            or getattr(inf, "camera_cfg_scale", 1.0) != 1.0
            or getattr(inf, "max_cache_segments", None) is not None):
        raise ValueError("QVG currently requires offline seg_ar, Flex, BF16, SP1, "
                         "CFG=1, camera CFG=1, and max_cache_segments=null")
    for name in ("preserve_kv_cache", "interactive_kv_cache_layerwise_cpu_offload",
                 "seg_ar_segment_vae_decode"):
        if getattr(inf, name, False):
            raise ValueError(f"QVG does not support {name}")
    if getattr(getattr(args, "vae", None), "stable_prefix_streaming", False):
        raise ValueError("QVG does not support model.vae.stable_prefix_streaming")
    if getattr(inf, "sparse_attention", {}).get("enabled", False):
        raise ValueError("QVG currently requires dense attention")


def storage_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from storage_tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from storage_tensors(child)
    elif hasattr(value, "tensors"):
        yield from value.tensors()


def storage_bytes(value):
    seen = set()
    total = 0
    for tensor in storage_tensors(value):
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        if key not in seen:
            seen.add(key)
            total += storage.nbytes()
    return total


def cache_storage_audit(cache):
    mode = cache_format(cache)
    payload = cache["qvg_working"] if mode == "qvg" else {
        key: cache[key] for key in ("k", "v", "k_scale", "v_scale") if key in cache
    }
    result = {
        "storage_format": mode,
        "gpu_storage_bytes": storage_bytes(payload),
        "storage_dtypes": sorted({str(t.dtype) for t in storage_tensors(payload)}),
        "codec_stats": dict(cache.get("qvg_stats", cache.get("kv_codec_stats", {}))),
    }
    result["peak_working_bytes"] = max(
        result["gpu_storage_bytes"], result["codec_stats"].get("peak_working_bytes", 0),
    )
    if mode == "qvg":
        from dataclasses import asdict
        from .qvg_kv_cache import UPSTREAM_VERSION

        blocks = cache["qvg_working"]
        result["components_bytes"] = {
            name: storage_bytes([getattr(getattr(block, key), name)
                                 for block in blocks for key in ("k", "v")])
            for name in ("centroids", "cluster_ids", "residual", "scales")
        }
        result["components_bytes"]["text"] = storage_bytes([block.text for block in blocks])
        result["qvg_options"] = asdict(cache["qvg_options"])
        result["qvg_version"] = UPSTREAM_VERSION
    result["transfer_bytes_scope"] = "KV payload only; excludes masks"
    return result


def record_qvg_working_peak(cache, replacement):
    """Include overlap with the previous working set during an atomic rebuild."""
    stats = cache["qvg_stats"]
    current = storage_bytes(replacement)
    overlap = storage_bytes((cache["qvg_working"], replacement))
    stats["peak_working_bytes"] = max(stats.get("peak_working_bytes", 0), current)
    stats["peak_rebuild_payload_bytes"] = max(
        stats.get("peak_rebuild_payload_bytes", 0), overlap,
    )
