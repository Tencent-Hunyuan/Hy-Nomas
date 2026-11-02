"""Opt-in inference FP8 for PixelUMM MLP and Q/O projections on SM90.

Install after loading the final BF16 checkpoint and entering eval mode. BF16
parameters remain checkpoint-compatible; nonpersistent FP8 buffers can be loaded
from an exported sidecar instead of quantized again at startup.
GEMM is an opaque custom op so the surrounding decoder remains fullgraph compilable.
No training support or silent runtime fallback: disable by not calling the installer.
"""
from collections import Counter
from pathlib import Path
import hashlib

import torch
from torch import nn

_calls = Counter()


def _quantize(x):
    import deep_gemm
    return deep_gemm.utils.per_token_cast_to_fp8(x, False)


_compiled_quantize = torch.compile(_quantize, mode="default", fullgraph=True, dynamic=False)


@torch.library.custom_op("hymm_pixelumm_fp8::linear", mutates_args=())
def _linear(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    import deep_gemm
    qx = _compiled_quantize(x)
    output = torch.empty((x.shape[0], weight.shape[0]), device=x.device, dtype=torch.bfloat16)
    # Serialized scales are FP32; Blackwell's UE8M0 conversion can produce NaNs.
    if torch.cuda.get_device_capability(x.device)[0] >= 10:
        deep_gemm.fp8_gemm_nt(qx, (weight, scale), output, disable_ue8m0_cast=True)
    else:
        deep_gemm.fp8_gemm_nt(qx, (weight, scale), output)
    _calls[(x.device.index, x.shape[0], x.shape[1], weight.shape[0])] += 1
    return output


@_linear.register_fake
def _linear_fake(x, weight, scale):
    return x.new_empty((x.shape[0], weight.shape[0]), dtype=torch.bfloat16)


class FP8Linear(nn.Module):
    def __init__(self, linear, quantized=None):
        super().__init__()
        import deep_gemm
        self.weight = linear.weight
        self.bias = linear.bias
        qw, sw = (deep_gemm.utils.per_block_cast_to_fp8(linear.weight.detach(), False)
                  if quantized is None else quantized)
        self.register_buffer("fp8_weight", qw, persistent=False)
        self.register_buffer("fp8_scale", sw, persistent=False)
        self.train(False)

    @classmethod
    def from_quantized(cls, weight, scale, expected_shape):
        """Load the shared single-file format without retaining BF16 weights."""
        from .prequantized import validate_serialized_tensors

        validate_serialized_tensors(weight, scale, expected_shape)
        module = cls.__new__(cls)
        nn.Module.__init__(module)
        module.register_buffer("fp8_weight", weight.contiguous())
        module.register_buffer("fp8_scale", scale.contiguous())
        return module.eval()

    def _apply(self, fn, recurse=True):
        # Avoid casting full FP8 buffers during dtype/device/offload moves.
        # Legacy instances also have BF16 parameters; serialized ones do not.
        buffers = {name: self._buffers.pop(name) for name in ("fp8_weight", "fp8_scale")}
        try:
            super()._apply(fn, recurse)
            for name, tensor in buffers.items():
                target = fn(torch.empty(0, device=tensor.device, dtype=tensor.dtype))
                self._buffers[name] = tensor.to(device=target.device)
        except Exception:
            self._buffers.update(buffers)
            raise
        return self

    def forward(self, x):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("PixelUMM FP8 supports eval + no_grad inference only")
        if x.device != self.fp8_weight.device:
            raise ValueError("FP8 input and weights must share a CUDA device")
        if x.dtype != torch.bfloat16:
            if not torch.is_autocast_enabled("cuda"):
                raise TypeError("PixelUMM FP8 requires BF16 or BF16 CUDA autocast")
            if torch.get_autocast_dtype("cuda") != torch.bfloat16:
                raise TypeError("PixelUMM FP8 requires BF16 CUDA autocast")
            x = x.to(torch.bfloat16)
        output = _linear(x.reshape(-1, x.shape[-1]).contiguous(), self.fp8_weight, self.fp8_scale)
        return output.reshape(*x.shape[:-1], self.fp8_weight.shape[0])


@torch.no_grad()
def enable_deepgemm_fp8(decoder, checkpoint_path=None):
    """Replace 3 MLP + Q/O linears per layer; K/V and boundary layers stay BF16."""
    from .model import DCAEVPixelUMMDecoder
    if type(decoder) is not DCAEVPixelUMMDecoder:
        raise TypeError("FP8 integration is validated for DCAEVPixelUMMDecoder only")
    if decoder.training:
        raise ValueError("Call eval() before enabling PixelUMM FP8")
    targets = []
    for layer in decoder.layers:
        targets.extend((layer.mlp, name) for name in ("gate_proj", "up_proj", "down_proj"))
        targets.extend((layer.attn, name) for name in ("q_proj", "o_proj"))
    for parent, name in targets:
        linear = getattr(parent, name)
        if not isinstance(linear, nn.Linear) or linear.bias is not None:
            raise ValueError("Expected an unmodified bias-free PixelUMM linear")
        w = linear.weight
        if not w.is_cuda or w.dtype != torch.bfloat16 or any(d % 128 for d in w.shape):
            raise ValueError("FP8 weights must be CUDA BF16 with dimensions divisible by 128")
        if torch.cuda.get_device_capability(w.device) != (9, 0):
            raise ValueError("This FP8 integration requires SM90")
    names = {id(module): name for name, module in decoder.named_modules()}
    saved = None
    if checkpoint_path is not None:
        from safetensors import safe_open
        checkpoint_path = Path(checkpoint_path)
        saved = safe_open(str(checkpoint_path), framework="pt", device=str(next(decoder.parameters()).device))
        expected = {names[id(getattr(parent, name))] + suffix
                    for parent, name in targets for suffix in (".fp8_weight", ".fp8_scale")}
        if (saved.metadata() or {}).get("format") != "pixelumm_deepgemm_fp8_v1" or set(saved.keys()) != expected:
            raise ValueError("Incompatible or incomplete PixelUMM FP8 checkpoint")
    for parent, name in targets:
        linear = getattr(parent, name)
        quantized = None
        if saved is not None:
            key = names[id(linear)]
            qw, sw = (saved.get_tensor(key + suffix) for suffix in (".fp8_weight", ".fp8_scale"))
            if (qw.dtype != torch.float8_e4m3fn or qw.shape != linear.weight.shape
                    or sw.dtype != torch.float32
                    or tuple(sw.shape) != tuple(d // 128 for d in linear.weight.shape)):
                raise ValueError("Invalid FP8 weight/scale: " + key)
            quantized = (qw, sw)
        setattr(parent, name, FP8Linear(linear, quantized))
    return len(targets)


def execution_stats():
    return [{"device":d,"M":m,"K":k,"N":n,"calls":count}
            for (d,m,k,n),count in sorted(_calls.items())]


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_fp8_checkpoint(vae_path):
    """Only use a sidecar tied to this exact base checkpoint; never silently mismatch."""
    from safetensors import safe_open
    path = Path(vae_path).parent / "pixelumm_fp8.safetensors"
    if not path.is_file():
        return None
    with safe_open(str(path), framework="pt", device="cpu") as saved:
        metadata = saved.metadata() or {}
    if metadata.get("source_sha256") != checkpoint_sha256(vae_path):
        raise ValueError("FP8 checkpoint does not match the selected base weights")
    return path
