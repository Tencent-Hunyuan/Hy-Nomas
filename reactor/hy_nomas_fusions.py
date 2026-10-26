"""PixelUMM split QK RMSNorm and T/H/W RoPE with native FP32 arithmetic."""

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_kernel(
    Q,
    K,
    V,
    Y,
    TOTAL: tl.constexpr,
    B: tl.constexpr,
    L: tl.constexpr,
    HL: tl.constexpr,
    D: tl.constexpr,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    QS2: tl.constexpr,
    QS3: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    KS3: tl.constexpr,
    VS0: tl.constexpr,
    VS1: tl.constexpr,
    VS2: tl.constexpr,
    VS3: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    d = i % D
    slot = i // D
    head = slot % HL
    batch = (slot // HL) % B
    row = (slot // (HL * B)) % L
    destination = slot // (HL * B * L)
    global_head = destination * HL + head
    q = tl.load(
        Q + batch * QS0 + row * QS1 + global_head * QS2 + d * QS3, i < TOTAL, other=0
    )
    k = tl.load(
        K + batch * KS0 + row * KS1 + global_head * KS2 + d * KS3, i < TOTAL, other=0
    )
    v = tl.load(
        V + batch * VS0 + row * VS1 + global_head * VS2 + d * VS3, i < TOTAL, other=0
    )
    base = slot * (3 * D) + d
    tl.store(Y + base, q, i < TOTAL)
    tl.store(Y + base + D, k, i < TOTAL)
    tl.store(Y + base + 2 * D, v, i < TOTAL)


def pack_qkv(q, k, v, world):
    batch, length, heads, dim = q.shape
    out = torch.empty(
        (world, length, batch, heads // world, 3 * dim), dtype=q.dtype, device=q.device
    )
    total = q.numel()
    _pack_kernel[(triton.cdiv(total, 1024),)](
        q,
        k,
        v,
        out,
        total,
        batch,
        length,
        heads // world,
        dim,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        1024,
        num_warps=4,
    )
    return out


@triton.jit
def _qk_kernel(
    X,
    WT,
    WH,
    COS,
    SIN,
    Y,
    N: tl.constexpr,
    H: tl.constexpr,
    TOTAL: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    XS2: tl.constexpr,
    XS3: tl.constexpr,
    CS: tl.constexpr,
    SS: tl.constexpr,
    EPS: tl.constexpr,
    ROWS: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    d = tl.arange(0, 128)
    token = (row // H) % N
    batch = row // (N * H)
    head = row % H
    x = tl.load(
        X
        + batch[:, None] * XS0
        + token[:, None] * XS1
        + head[:, None] * XS2
        + d[None, :] * XS3,
        row[:, None] < TOTAL,
        other=0,
    ).to(tl.float32)
    square = x * x
    lo = tl.sum(tl.where(d[None, :] < 64, square, 0), 1) / 64
    hi = tl.sum(tl.where(d[None, :] >= 64, square, 0), 1) / 64
    variance = tl.where(d[None, :] < 64, lo[:, None], hi[:, None])
    w = tl.where(d < 64, tl.load(WT + d % 64), tl.load(WH + d % 64)).to(tl.float32)
    normalized = (x * tl.rsqrt(variance + EPS)) * w[None, :]
    base = tl.where(d < 64, 0, tl.where(d < 96, 64, 96))
    width = tl.where(d < 64, 64, 32)
    partner = base + (d - base + width // 2) % width
    rotated = tl.gather(normalized, tl.broadcast_to(partner[None, :], (ROWS, 128)), 1)
    rotated = tl.where((d - base)[None, :] < width[None, :] // 2, -rotated, rotated)
    cos = tl.load(COS + token[:, None] * CS + d[None, :], row[:, None] < TOTAL, other=0)
    sin = tl.load(SIN + token[:, None] * SS + d[None, :], row[:, None] < TOTAL, other=0)
    out = normalized * cos + rotated * sin
    tl.store(Y + row[:, None] * 128 + d[None, :], out, row[:, None] < TOTAL)


@torch.library.custom_op("hy_nomas_fused::pixelumm_qk", mutates_args=())
def qk_prepare(
    x: torch.Tensor,
    wt: torch.Tensor,
    wh: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if x.shape[-1] != 128 or x.dtype != torch.bfloat16:
        raise ValueError("PixelUMM QK fusion requires BF16 128-dimensional heads")
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    total = x.shape[0] * x.shape[1] * x.shape[2]
    _qk_kernel[(triton.cdiv(total, 4),)](
        x,
        wt,
        wh,
        cos,
        sin,
        out,
        x.shape[1],
        x.shape[2],
        total,
        *x.stride(),
        cos.stride(1),
        sin.stride(1),
        eps,
        4,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


@qk_prepare.register_fake
def _fake(x, wt, wh, cos, sin, eps):
    return x.new_empty(x.shape)


def install_qk_fusion(decoder):
    from types import MethodType

    from hymm.models.vae.pixelumm_decoder.model import GenAttention

    if decoder.arch.head_dim != 128 or decoder.arch.rope_sections != (64, 32, 32):
        raise ValueError("PixelUMM QK fusion requires the released rotary geometry")

    def prepare(self, x, cos, sin, norm_t, norm_hw, dtype):
        if x.dtype == torch.bfloat16 and dtype == torch.bfloat16 and x.is_cuda:
            return qk_prepare(x, norm_t.weight, norm_hw.weight, cos, sin, norm_t.eps)
        return GenAttention._prepare_qk(self, x, cos, sin, norm_t, norm_hw, dtype)

    for layer in decoder.layers:
        layer.attn._prepare_qk = MethodType(prepare, layer.attn)
