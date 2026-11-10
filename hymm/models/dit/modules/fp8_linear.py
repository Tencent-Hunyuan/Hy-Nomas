"""Fine-grained inference W8A8: offline E4M3 weights, dynamic token/group activations.

Scales are rounded upward to powers of two for native Blackwell UE8M0 MMA.
Only large attention/MLP projections are selected; conditioning stays BF16.
"""
import re
import os

import torch
from torch import nn
import triton
import triton.language as tl

FORMAT = 'nomas-dit-fp8-block128-v1'
SELECT = re.compile(r'^double_blocks\.\d+\.(img|txt)_(attn_[qkv]|attn_proj|mlp\.fc[12])$')


def quantize_weight(weight):
    if weight.ndim != 2 or any(d % 128 for d in weight.shape):
        raise ValueError('FP8 weights require N and K divisible by 128')
    if not torch.isfinite(weight).all():
        raise ValueError('Nonfinite FP8 source weight')
    n, k = weight.shape
    blocks = weight.float().reshape(n // 128, 128, k // 128, 128)
    scale = torch.exp2(torch.ceil(torch.log2(blocks.abs().amax((1, 3)).clamp_min(1e-4) / 448)))
    q = (blocks / scale[:, None, :, None]).to(torch.float8_e4m3fn).reshape(n, k)
    return q.contiguous(), scale.contiguous()


@triton.jit
def _quantize_activation(X, Q, S, K: tl.constexpr, GROUPS: tl.constexpr):
    row = tl.program_id(0)
    group = tl.program_id(1)
    offsets = row * K + group * 128 + tl.arange(0, 128)
    x = tl.load(X + offsets).to(tl.float32)
    scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(x), 0), 1.e-4) / 448.)))
    tl.store(Q + offsets, x / scale)
    tl.store(S + row * GROUPS + group, scale)


def quantize_activation(x):
    if x.ndim != 2 or x.shape[1] % 128 or not x.is_cuda or x.dtype != torch.bfloat16:
        raise ValueError('Expected CUDA BF16 [tokens, channels], channels divisible by 128')
    x = x.contiguous()
    m, k = x.shape
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty((m, k // 128), device=x.device, dtype=torch.float32)
    if m:
        _quantize_activation[(m, k // 128)](x, q, scales, k, k // 128, num_warps=4)
    return q, scales


@triton.jit
def _quantize_packed(X, Q, S, M: tl.constexpr, K: tl.constexpr, STRIDE: tl.constexpr, ROWS: tl.constexpr = 4, BIAS=None, HAS_BIAS: tl.constexpr = False, GELU: tl.constexpr = False):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    groups = tl.program_id(1) * 4 + tl.arange(0, 4)
    offsets = rows[:, None, None] * K + groups[None, :, None] * 128 + tl.arange(0, 128)[None, None, :]
    mask = (rows[:, None, None] < M) & (groups[None, :, None] < K // 128)
    x = tl.load(X + offsets, mask, 0).to(tl.float32)
    if GELU:
        # Keep the original BF16 bias-add and GELU rounding before FP8 quantization.
        if HAS_BIAS:
            bias_offsets = groups[:, None] * 128 + tl.arange(0, 128)[None, :]
            bias = tl.load(BIAS + bias_offsets, bias_offsets < K, 0).to(tl.float32)
            x = (x + bias[None, :, :]).to(tl.bfloat16).to(tl.float32)
        t = 0.7978845608028654 * (x + 0.044715 * x * x * x)
        # Native approximate tanh avoids the much slower libdevice implementation.
        t = tl.inline_asm_elementwise("tanh.approx.f32 $0, $1;", constraints="=f,f",
                                     args=[t], dtype=tl.float32, is_pure=True, pack=1)
        x = (0.5 * x * (1.0 + t)).to(tl.bfloat16).to(tl.float32)
    scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(x), 2), 1.e-4) / 448.)))
    tl.store(Q + offsets, x / scale[:, :, None], mask)
    exponent = (scale.to(tl.int32, bitcast=True) >> 23) & 255
    exponent = tl.where(groups[None, :] < K // 128, exponent, 0)
    packed = tl.sum(exponent << (tl.arange(0, 4)[None, :] * 8), 1)
    tl.store(S + rows + tl.program_id(1) * STRIDE, packed, rows < M)


def quantize_activation_packed(x, bias=None, gelu=False):
    if x.ndim != 2 or x.shape[1] % 128 or not x.is_cuda or x.dtype != torch.bfloat16:
        raise ValueError('Expected CUDA BF16 [tokens, channels], channels divisible by 128')
    if bias is not None and (bias.shape != (x.shape[1],) or bias.dtype != torch.bfloat16 or bias.device != x.device):
        raise ValueError('GELU bias must be BF16 [channels] on the input device')
    x = x.contiguous()
    m, k = x.shape
    stride = triton.cdiv(m, 4) * 4
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty_strided((m, triton.cdiv(k, 512)), (1, stride), device=x.device, dtype=torch.int32)
    if m:
        rows = 16 if m >= 4096 else 4
        _quantize_packed[(triton.cdiv(m, rows), triton.cdiv(k, 512))](
            x, q, scales, m, k, stride, ROWS=rows, BIAS=bias, HAS_BIAS=bias is not None, GELU=gelu, num_warps=4)
    return q, scales


@torch.library.custom_op("nomas_fp8::linear", mutates_args=())
def fp8_linear_op(x: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    import deep_gemm
    q, s = quantize_activation_packed(x)
    out = torch.empty((x.shape[0], weight.shape[0]), device=x.device, dtype=torch.bfloat16)
    if x.shape[0]:
        deep_gemm.fp8_gemm_nt((q, s), (weight, scales), out, recipe=(1, 1, 128))
    return out


@fp8_linear_op.register_fake
def _fp8_linear_fake(x, weight, scales):
    return x.new_empty((x.shape[0], weight.shape[0]))


@torch.library.custom_op("nomas_fp8::gelu_linear", mutates_args=())
def fp8_gelu_linear_op(x: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor,
                       input_bias: torch.Tensor | None) -> torch.Tensor:
    import deep_gemm
    q, s = quantize_activation_packed(x, bias=input_bias, gelu=True)
    out = torch.empty((x.shape[0], weight.shape[0]), device=x.device, dtype=torch.bfloat16)
    if x.shape[0]:
        deep_gemm.fp8_gemm_nt((q, s), (weight, scales), out, recipe=(1, 1, 128))
    return out


@fp8_gelu_linear_op.register_fake
def _fp8_gelu_fake(x, weight, scales, input_bias):
    return x.new_empty((x.shape[0], weight.shape[0]))


class FP8Linear(nn.Module):
    is_fp8_linear = True
    def __init__(self, original):
        super().__init__()
        self.in_features, self.out_features = original.in_features, original.out_features
        self.fused_gelu_quant = os.environ.get("HY_DIT_FP8_FUSED_GELU", "0") == "1"
        if self.in_features % 128 or self.out_features % 128:
            raise ValueError('FP8Linear requires 128-aligned dimensions')
        self.register_buffer('weight_q', torch.empty((self.out_features, self.in_features),
                             device='meta', dtype=torch.float8_e4m3fn))
        self.register_buffer('weight_scale', torch.empty((self.out_features // 128, self.in_features // 128),
                             device='meta', dtype=torch.float32))
        self.bias = original.bias
        self.register_buffer("_packed_scale", None, persistent=False)
        self.register_buffer("_qkv_weight", None, persistent=False)
        self.register_buffer("_qkv_scale", None, persistent=False)

    def forward_no_bias(self, x):
        if x.dtype != torch.bfloat16 and torch.is_autocast_enabled('cuda'):
            x = x.to(torch.bfloat16)
        shape = x.shape
        x = x.reshape(-1, self.in_features)
        if self._packed_scale is None:
            self.prepare_for_inference()
        out = fp8_linear_op(x, self.weight_q, self._packed_scale)
        return out.reshape(*shape[:-1], self.out_features)

    def forward(self, x):
        out = self.forward_no_bias(x)
        return out if self.bias is None else out + self.bias

    def forward_gelu(self, x, input_bias):
        if self._packed_scale is None:
            self.prepare_for_inference()
        shape = x.shape
        out = fp8_gelu_linear_op(x.reshape(-1, self.in_features), self.weight_q, self._packed_scale, input_bias)
        out = out.reshape(*shape[:-1], self.out_features)
        return out if self.bias is None else out + self.bias

    def forward_qkv(self, x, k, v):
        if x.dtype != torch.bfloat16 and torch.is_autocast_enabled('cuda'):
            x = x.to(torch.bfloat16)
        if self._qkv_weight is None:
            return self(x), k(x), v(x)
        shape = x.shape
        out = fp8_linear_op(x.reshape(-1, self.in_features), self._qkv_weight, self._qkv_scale)
        outputs = out.reshape(*shape[:-1], self.out_features * 3).chunk(3, dim=-1)
        return tuple(y if layer.bias is None else y + layer.bias for y, layer in zip(outputs, (self, k, v)))

    def prepare_for_inference(self):
        import deep_gemm
        if torch.cuda.get_device_capability(self.weight_q.device)[0] != 10:
            raise RuntimeError('Packed FP8 DiT requires Blackwell SM100/SM103')
        expanded = self.weight_scale.repeat_interleave(128, dim=0)
        self._packed_scale = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(expanded)

    def shard_rows(self, rank, size):
        n = self.out_features // size
        if self.out_features % size or n % 128:
            raise ValueError('FP8 TP output shard must be divisible by 128')
        self.weight_q = self.weight_q[rank*n:(rank+1)*n].contiguous()
        self.weight_scale = self.weight_scale[rank*n//128:(rank+1)*n//128].contiguous()
        if self.bias is not None:
            self.bias = nn.Parameter(self.bias[rank*n:(rank+1)*n].contiguous(), requires_grad=False)
        self._qkv_weight = self._qkv_scale = None
        self.out_features = n
        self._packed_scale = None


def prepare_fp8_qkv(block):
    """Fuse already-quantized projections; share the merged weight storage with Q/K/V."""
    for stream in ('img', 'txt'):
        layers = [getattr(block, f'{stream}_attn_{kind}') for kind in 'qkv']
        if not all(isinstance(layer, FP8Linear) for layer in layers):
            continue
        q, k, v = layers
        weight = torch.cat([layer.weight_q for layer in layers], dim=0)
        merged_scale = torch.cat([layer._packed_scale for layer in layers], dim=0)
        scale = torch.empty_strided(merged_scale.shape, (1, merged_scale.shape[0]),
                                    device=merged_scale.device, dtype=torch.int32)
        scale.copy_(merged_scale)
        q._qkv_weight, q._qkv_scale = weight, scale
        for layer, part in zip(layers, weight.chunk(3, dim=0)):
            layer.weight_q = part


def verify_fp8_model(model):
    """Check that every selected projection of a loaded model holds E4M3 weights; returns their count."""
    count = 0
    for name, module in model.named_modules():
        if not SELECT.fullmatch(name):
            continue
        if not isinstance(module, FP8Linear):
            raise RuntimeError(f"FP8 DiT projection is not quantized: {name} is {type(module).__name__}")
        if module.weight_q.dtype != torch.float8_e4m3fn or module.weight_q.is_meta:
            raise RuntimeError(f"FP8 DiT projection holds no E4M3 weight: {name} ({module.weight_q.dtype})")
        if module.weight_scale.dtype != torch.float32 or module.weight_scale.is_meta:
            raise RuntimeError(f"FP8 DiT projection holds no weight scale: {name}")
        count += 1
    if not count:
        raise RuntimeError("FP8 DiT has no quantized projection")
    return count


def load_fp8_model(model_factory, checkpoint, device):
    """Validate every shard and tensor before loading an inference-only model."""
    import json
    from pathlib import Path
    from safetensors.torch import load_file
    from .int8_stored_bf16_linear import _sha256, _restore_derived_buffers

    root = Path(checkpoint).resolve()
    manifest = json.loads((root / 'fp8_manifest.json').read_text())
    if manifest.get('format') != FORMAT or not manifest.get('layers') or not manifest.get('shards'):
        raise ValueError('Invalid FP8 checkpoint manifest')
    with torch.device('meta'):
        model = model_factory()
    selected = {name for name, module in model.named_modules()
                if isinstance(module, nn.Linear) and SELECT.fullmatch(name)}
    if selected != set(manifest['layers']):
        raise ValueError('FP8 manifest must cover every selected DiT projection')
    for name, shape in manifest['layers'].items():
        original = model.get_submodule(name)
        if not SELECT.fullmatch(name) or not isinstance(original, nn.Linear) or list(original.weight.shape) != shape:
            raise ValueError(f'FP8 layer mismatch: {name}')
        parent, _, leaf = name.rpartition('.')
        setattr(model.get_submodule(parent), leaf, FP8Linear(original))
    expected = model.state_dict()
    seen, files = set(), set()
    for shard in manifest['shards']:
        name = shard['file']
        path = (root / name).resolve()
        if Path(name).name != name or not path.is_relative_to(root) or name in files:
            raise ValueError('Invalid FP8 shard path')
        files.add(name)
        if path.stat().st_size != shard['bytes'] or _sha256(path) != shard['sha256']:
            raise ValueError(f'FP8 shard checksum mismatch: {name}')
        tensors = load_file(str(path))
        for key, tensor in tensors.items():
            if key not in expected or key in seen or tensor.shape != expected[key].shape:
                raise ValueError(f'FP8 checkpoint key/shape mismatch: {key}')
            dtype = expected[key].dtype
            ordinary_bf16 = (not key.endswith(('.weight_q', '.weight_scale'))
                             and dtype.is_floating_point and tensor.dtype == torch.bfloat16)
            if tensor.dtype != dtype and not ordinary_bf16:
                raise ValueError(f'FP8 checkpoint dtype mismatch: {key}')
            if not torch.isfinite(tensor.float()).all():
                raise ValueError(f'Nonfinite tensor: {key}')
            if key.endswith('.weight_scale'):
                if not torch.all(tensor > 0) or torch.any(tensor.view(torch.int32) & 0x7fffff):
                    raise ValueError(f'Expected positive power-of-two scale: {key}')
        seen.update(tensors)
        model.load_state_dict(tensors, strict=False, assign=True)
    if seen != set(expected):
        raise ValueError(f'Missing FP8 tensors: {sorted(set(expected)-seen)[:5]}')
    _restore_derived_buffers(model)
    model.eval().requires_grad_(False).to(device=device)
    if torch.device(device).type == 'cuda':
        for module in model.modules():
            if isinstance(module, FP8Linear):
                module.prepare_for_inference()
        for block in model.double_blocks:
            prepare_fp8_qkv(block)
    return model
