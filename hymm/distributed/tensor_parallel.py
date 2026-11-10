"""Validation for DiT tensor parallelism.

The implementation lives in ``hymm.models.dit.modules.tensor_parallel``: an inference-only
TP for the dual-stream blocks' KV-cache forward (``seg_ar`` / ``chunk_ar`` rollouts).
"""
from __future__ import annotations


def validate_parallel_sizes(sp_size, tp_size, world_size):
    for name, value in (("sp_size", sp_size), ("tp_size", tp_size), ("world_size", world_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    size = sp_size * tp_size
    if world_size % size:
        raise ValueError(f"WORLD_SIZE ({world_size}) must be divisible by sp_size * tp_size ({size})")
    return size


def validate_tensor_parallel_config(args):
    parallelism = getattr(args, "parallelism", args)
    size = getattr(parallelism, "tp_size", 1)
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("parallelism.tp_size must be a positive integer")
    if size == 1:
        return
    inference = getattr(args, "inference", None)
    dit = getattr(args, "dit", None)
    if getattr(parallelism, "sp_size", 1) != 1:
        raise ValueError("Tensor parallel cannot be combined with sequence parallel (sp_size must be 1)")
    if getattr(inference, "rollout_type", "seg_ar") == "one_shot":
        raise ValueError("Tensor parallel supports the KV-cache rollouts (seg_ar, chunk_ar), not one_shot")
    if getattr(inference, "dit_weight_quantization", "none") not in ("none", "fp8"):
        raise ValueError("Tensor parallel supports BF16 or FP8 DiT weights, not INT8")
    if getattr(inference, "lora_adapter", None):
        raise ValueError("Tensor parallel does not yet support LoRA adapters")
    if getattr(parallelism, "dit_cpu_offload", False):
        raise ValueError("Tensor parallel does not support DiT CPU offload")
    if getattr(parallelism, "use_dynamic_ring_attention", False):
        raise ValueError("Tensor parallel does not yet support dynamic ring attention")
    if getattr(dit, "use_moe", False):
        raise ValueError("Tensor parallel currently supports dense DiT MLPs, not MoE")
