"""Frozen model residency for TE/PixelUMM methods and mutually exclusive DiT phases.

Only registered parameters and buffers move. External KV caches, activations,
outputs, and the DiT forward/camera methods keep their own lifetimes.
"""

from contextlib import contextmanager
from functools import wraps
from itertools import chain
import os
import time

import torch
from hymm.utils.inference_metrics import stage

from configs.cpu_offload import enabled_cpu_offloads


def _frozen_model_device(module):
    """Return device metadata without retaining any original tensor references."""
    if module.training or any(p.requires_grad for p in module.parameters()):
        raise ValueError("CPU offload requires frozen evaluation weights")
    if any(getattr(child, "_orig_mod", None) is not None for child in module.modules()):
        raise ValueError("CPU offload does not support compiled models")
    devices = set()
    for tensor in chain(module.parameters(), module.buffers()):
        if tensor.is_meta or hasattr(tensor, "device_mesh") or tensor.requires_grad:
            raise ValueError("CPU offload requires frozen, non-meta, unsharded parameters and buffers")
        devices.add(tensor.device)
    if len(devices) > 1:
        raise ValueError("CPU offload requires a single-device model")
    device = next(iter(devices), torch.device("cpu"))
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("CPU offload supports CPU/CUDA tensors only")
    return device


class ModelCPUOffload:
    """Keep a frozen module on CPU between complete methods or pipeline phases."""

    def __init__(self, module, device, *, stage_prefix=None):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("Model CPU offload requires a CUDA execution device")
        original = _frozen_model_device(module)
        self.stage_prefix = stage_prefix or ("pixelumm" if hasattr(module, "decode") else "te")
        if self.stage_prefix not in {"te", "pixelumm", "dit"}:
            raise ValueError("Unknown CPU-offload metric prefix")
        self.module = module
        self.depth = 0
        self.completed_calls = 0
        self.gpu_transfers = 0
        self.cpu_transfers = 0
        self.transfer_seconds = 0.0
        try:
            self.module.to(device="cpu", non_blocking=False)
        except BaseException:
            self.module.to(device=original, non_blocking=False)
            raise

    @contextmanager
    def on_device(self):
        if torch.is_grad_enabled() or self.module.training:
            raise RuntimeError("Model CPU offload supports frozen no-grad inference only")
        outermost = self.depth == 0
        if outermost:
            _frozen_model_device(self.module)
        self.depth += 1
        try:
            if outermost:
                started = time.perf_counter()
                with stage(self.stage_prefix + "_H2D"):
                    self.module.to(device=self.device, non_blocking=False)
                self.transfer_seconds += time.perf_counter() - started
                self.gpu_transfers += 1
            yield
            if outermost:
                self.completed_calls += 1
        finally:
            self.depth -= 1
            if outermost:
                try:
                    torch.cuda.synchronize(self.device)
                finally:
                    # Device only: keep INT8 payloads and FP32 quantization scales.
                    started = time.perf_counter()
                    with stage(self.stage_prefix + "_D2H"):
                        self.module.to(device="cpu", non_blocking=False)
                    self.transfer_seconds += time.perf_counter() - started
                    self.cpu_transfers += 1


def cpu_offloaded(method):
    """Cover complete TE encode/PixelUMM decode calls, including nested direct calls."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        self._last_inference_mode = torch.is_inference_mode_enabled()
        controller = getattr(self, "_model_cpu_offload", None)
        if controller is None:
            if not hasattr(self, "_fsdp_cpu_offload_device"):
                return method(self, *args, **kwargs)
            from torch.distributed._composable.fsdp import FSDPModule

            if torch.is_grad_enabled() or self.training:
                raise RuntimeError("Sharded CPU offload supports frozen no-grad inference only")
            try:
                result = method(self, *args, **kwargs)
                self._fsdp_cpu_offload_calls = getattr(self, "_fsdp_cpu_offload_calls", 0) + 1
                return result
            finally:
                # Reshard even when a final norm or another encode operation raises.
                # No extra collective: reshard only releases local gathered storage.
                for child in self.model.modules():
                    if isinstance(child, FSDPModule):
                        child.reshard()
        with controller.on_device():
            return method(self, *args, **kwargs)
    return wrapped


def install_model_cpu_offload(module, device, *, stage_prefix=None):
    if getattr(module, "_model_cpu_offload", None) is not None:
        raise ValueError("Model CPU offload is already installed")
    controller = ModelCPUOffload(module, device, stage_prefix=stage_prefix)
    module._model_cpu_offload = controller
    return controller


@contextmanager
def temporary_cpu_residency(module):
    """Unload a frozen DiT for an entire decode/postprocess phase, then restore."""
    if torch.is_grad_enabled() or module.training:
        raise RuntimeError("Decode-phase offload requires frozen no-grad inference")
    if getattr(module, "_model_cpu_offload", None) is not None:
        raise ValueError("DiT must not use a per-call CPU offload controller")
    original = _frozen_model_device(module)
    if original.type == "cpu":
        yield
        return
    torch.cuda.synchronize(original)
    stats = getattr(module, "_decode_cpu_offload_stats", None)
    if stats is None:
        stats = {"cpu_transfers": 0, "gpu_restores": 0, "completed_phases": 0,
                 "transfer_seconds": 0.0}
        module._decode_cpu_offload_stats = stats
    try:
        started = time.perf_counter()
        with stage("dit_D2H"):
            module.to(device="cpu", non_blocking=False)
        stats["transfer_seconds"] += time.perf_counter() - started
        stats["cpu_transfers"] += 1
        yield
        stats["completed_phases"] += 1
    finally:
        # Restore after decode, postprocessing, or a partial CPU transfer fails.
        started = time.perf_counter()
        with stage("dit_H2D"):
            module.to(device=original, non_blocking=False)
        stats["transfer_seconds"] += time.perf_counter() - started
        stats["gpu_restores"] += 1


def offload_dit_for_decode(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        enabled = getattr(
            getattr(self.args, "parallelism", None), "dit_cpu_offload_for_decode", False,
        )
        if not enabled:
            return method(self, *args, **kwargs)
        self.transformer._last_inference_mode = torch.is_inference_mode_enabled()
        with temporary_cpu_residency(self.transformer):
            return method(self, *args, **kwargs)
    return wrapped


def validate_model_cpu_offload(args, device):
    """Validate before checkpoint loading, including actual execution topology."""
    if not enabled_cpu_offloads(args):
        return
    if torch.device(device).type != "cuda":
        raise ValueError("P2 CPU offload requires a CUDA execution device")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
    sp_size = args.parallelism.sp_size * getattr(args.parallelism, "tp_size", 1)
    if world_size != sp_size:
        raise ValueError(
            f"CPU offload requires world_size == sp_size * tp_size; got {world_size} and {sp_size}"
        )
    if world_size > 1 and not (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    ):
        raise ValueError("Model-parallel CPU offload requires an initialized distributed process group")


def cpu_offload_snapshot(pipeline):
    """Record scalar residency evidence without retaining parameter/buffer views."""
    result = {}
    for name, module in (
        ("text_encoder", pipeline.text_encoder),
        ("pixelumm", pipeline.vae),
        ("dit", pipeline.transformer),
    ):
        if module is None:
            continue
        row = {"parameter_bytes": {}, "buffer_bytes": {}, "sharded_parameters": 0,
               "last_inference_mode": getattr(module, "_last_inference_mode", None)}
        for kind, tensors in (("parameter_bytes", module.parameters()),
                              ("buffer_bytes", module.buffers())):
            for tensor in tensors:
                if hasattr(tensor, "to_local"):
                    row["sharded_parameters"] += int(kind == "parameter_bytes")
                    tensor = tensor.to_local()
                device = str(tensor.device)
                row[kind][device] = row[kind].get(device, 0) + tensor.numel() * tensor.element_size()
            # Do not retain the final tensor while the next model changes residency.
            tensor = None
        controller = getattr(module, "_model_cpu_offload", None)
        if controller is not None:
            row.update(
                policy="denoise_phase" if name == "dit" else "whole_model",
                completed_calls=controller.completed_calls,
                gpu_transfers=controller.gpu_transfers, cpu_transfers=controller.cpu_transfers,
                transfer_seconds=controller.transfer_seconds, depth=controller.depth,
            )
        elif hasattr(module, "_fsdp_cpu_offload_device"):
            row.update(policy="fsdp_cpu_shards",
                       completed_calls=getattr(module, "_fsdp_cpu_offload_calls", 0))
        elif hasattr(module, "_decode_cpu_offload_stats"):
            row.update(policy="decode_phase", **module._decode_cpu_offload_stats)
        else:
            row["policy"] = "resident"
        result[name] = row
    return result
