"""Optional inference-only torch.compile boundaries around Nomas cache attention.

The functions take parameter tensors explicitly so all 54 DiT blocks can share
one compiled graph per input shape. Attention, cache updates, and distributed
communication remain outside these graphs.
"""

from functools import lru_cache

import torch
import torch.nn.functional as F


def _modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _rms_norm(x, weight, eps):
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    return y.to(x.dtype) * weight


def _rotate_half(x):
    real, imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    return torch.stack((-imag, real), dim=-1).flatten(3)


def pre_attention(
    img, txt, vec, vec_txt, cos, sin,
    img_mod_w, img_mod_b, txt_mod_w, txt_mod_b,
    img_q_w, img_q_b, img_k_w, img_k_b, img_v_w, img_v_b,
    txt_q_w, txt_q_b, txt_k_w, txt_k_b, txt_v_w, txt_v_b,
    img_q_norm_w, img_k_norm_w, txt_q_norm_w, txt_k_norm_w,
    heads, qk_eps,
):
    img_parts = F.linear(F.silu(vec), img_mod_w, img_mod_b).chunk(6, dim=-1)
    txt_parts = F.linear(F.silu(vec_txt), txt_mod_w, txt_mod_b).chunk(6, dim=-1)

    img_hidden = _modulate(
        F.layer_norm(img, (img.shape[-1],), eps=1e-6),
        img_parts[0], img_parts[1],
    )
    head_dim = img.shape[-1] // heads
    img_q = F.linear(img_hidden, img_q_w, img_q_b).reshape(img.shape[0], -1, heads, head_dim)
    img_k = F.linear(img_hidden, img_k_w, img_k_b).reshape(img.shape[0], -1, heads, head_dim)
    img_v = F.linear(img_hidden, img_v_w, img_v_b).reshape(img.shape[0], -1, heads, head_dim)
    img_q = _rms_norm(img_q, img_q_norm_w, qk_eps).to(img_v)
    img_k = _rms_norm(img_k, img_k_norm_w, qk_eps).to(img_v)
    cos = cos.view(1, img_q.shape[1], 1, head_dim).to(img_q.device)
    sin = sin.view(1, img_q.shape[1], 1, head_dim).to(img_q.device)
    img_q = (img_q.float() * cos + _rotate_half(img_q) * sin).type_as(img_v)
    img_k = (img_k.float() * cos + _rotate_half(img_k) * sin).type_as(img_v)

    txt_hidden = _modulate(
        F.layer_norm(txt, (txt.shape[-1],), eps=1e-6),
        txt_parts[0], txt_parts[1],
    )
    txt_q = F.linear(txt_hidden, txt_q_w, txt_q_b).reshape(txt.shape[0], -1, heads, head_dim)
    txt_k = F.linear(txt_hidden, txt_k_w, txt_k_b).reshape(txt.shape[0], -1, heads, head_dim)
    txt_v = F.linear(txt_hidden, txt_v_w, txt_v_b).reshape(txt.shape[0], -1, heads, head_dim)
    txt_q = _rms_norm(txt_q, txt_q_norm_w, qk_eps).to(txt_v)
    txt_k = _rms_norm(txt_k, txt_k_norm_w, qk_eps).to(txt_v)

    return (
        img_q, img_k, img_v, txt_q, txt_k, txt_v,
        img_parts[2], img_parts[3], img_parts[4], img_parts[5],
        txt_parts[2], txt_parts[3], txt_parts[4], txt_parts[5],
    )


def post_attention(
    img, txt, img_attn, txt_attn,
    img_gate1, img_shift2, img_scale2, img_gate2,
    txt_gate1, txt_shift2, txt_scale2, txt_gate2,
    img_o_w, img_o_b, txt_o_w, txt_o_b,
    img_fc1_w, img_fc1_b, img_fc2_w, img_fc2_b,
    txt_fc1_w, txt_fc1_b, txt_fc2_w, txt_fc2_b,
):
    img = img + F.linear(img_attn, img_o_w, img_o_b) * img_gate1.unsqueeze(1)
    img_hidden = _modulate(
        F.layer_norm(img, (img.shape[-1],), eps=1e-6),
        img_shift2, img_scale2,
    )
    img_hidden = F.linear(img_hidden, img_fc1_w, img_fc1_b)
    img_hidden = F.gelu(img_hidden, approximate="tanh")
    img_hidden = F.linear(img_hidden, img_fc2_w, img_fc2_b)
    img = img + img_hidden * img_gate2.unsqueeze(1)

    txt = txt + F.linear(txt_attn, txt_o_w, txt_o_b) * txt_gate1.unsqueeze(1)
    txt_hidden = _modulate(
        F.layer_norm(txt, (txt.shape[-1],), eps=1e-6),
        txt_shift2, txt_scale2,
    )
    txt_hidden = F.linear(txt_hidden, txt_fc1_w, txt_fc1_b)
    txt_hidden = F.gelu(txt_hidden, approximate="tanh")
    txt_hidden = F.linear(txt_hidden, txt_fc2_w, txt_fc2_b)
    txt = txt + txt_hidden * txt_gate2.unsqueeze(1)
    return img, txt


@lru_cache(maxsize=1)
def compiled_pre_attention():
    return torch.compile(pre_attention, fullgraph=True, options={"triton.cudagraphs": False})


@lru_cache(maxsize=1)
def compiled_post_attention():
    return torch.compile(post_attention, fullgraph=True, options={"triton.cudagraphs": False})


def tp_modulation(vec, img_weight, img_bias, txt_weight, txt_bias):
    activated = F.silu(vec)
    return F.linear(activated, img_weight, img_bias), F.linear(activated, txt_weight, txt_bias)


def tp_local_norm(img, txt):
    return (
        F.layer_norm(img, (img.shape[-1],), eps=1e-6),
        F.layer_norm(txt, (txt.shape[-1],), eps=1e-6),
    )


def tp_qkv(
    img, txt, img_shift, img_scale, txt_shift, txt_scale, cos, sin,
    iqw, iqb, ikw, ikb, ivw, ivb, tqw, tqb, tkw, tkb, tvw, tvb,
    iqnorm, iknorm, tqnorm, tknorm, heads, eps,
):
    head_dim = iqw.shape[0] // heads
    img = _modulate(img, img_shift, img_scale)
    txt = _modulate(txt, txt_shift, txt_scale)
    iq = F.linear(img, iqw, iqb).reshape(img.shape[0], -1, heads, head_dim)
    ik = F.linear(img, ikw, ikb).reshape(img.shape[0], -1, heads, head_dim)
    iv = F.linear(img, ivw, ivb).reshape(img.shape[0], -1, heads, head_dim)
    iq = _rms_norm(iq, iqnorm, eps).to(iv)
    ik = _rms_norm(ik, iknorm, eps).to(iv)
    cos = cos.view(1, img.shape[1], 1, head_dim)
    sin = sin.view(1, img.shape[1], 1, head_dim)
    iq = (iq.float() * cos + _rotate_half(iq) * sin).to(iv)
    ik = (ik.float() * cos + _rotate_half(ik) * sin).to(iv)
    tq = F.linear(txt, tqw, tqb).reshape(txt.shape[0], -1, heads, head_dim)
    tk = F.linear(txt, tkw, tkb).reshape(txt.shape[0], -1, heads, head_dim)
    tv = F.linear(txt, tvw, tvb).reshape(txt.shape[0], -1, heads, head_dim)
    return iq, ik, iv, _rms_norm(tq, tqnorm, eps).to(tv), _rms_norm(tk, tknorm, eps).to(tv), tv


def tp_attention_partial(img, txt, img_weight, txt_weight):
    # Row projection bias is added only after the distributed sum.
    return F.linear(img, img_weight), F.linear(txt, txt_weight)


def tp_residual_norm(img, txt, img_partial, txt_partial, img_bias, txt_bias, img_gate, txt_gate):
    if img_bias is not None:
        img_partial = img_partial + img_bias
    if txt_bias is not None:
        txt_partial = txt_partial + txt_bias
    img = img + img_partial * img_gate.unsqueeze(1)
    txt = txt + txt_partial * txt_gate.unsqueeze(1)
    img_norm, txt_norm = tp_local_norm(img, txt)
    return img, txt, img_norm, txt_norm


def tp_mlp_partial(
    img, txt, img_shift, img_scale, txt_shift, txt_scale,
    img_fc1, img_bias1, img_fc2, txt_fc1, txt_bias1, txt_fc2,
):
    img = _modulate(img, img_shift, img_scale)
    txt = _modulate(txt, txt_shift, txt_scale)
    img = F.gelu(F.linear(img, img_fc1, img_bias1), approximate="tanh")
    txt = F.gelu(F.linear(txt, txt_fc1, txt_bias1), approximate="tanh")
    return F.linear(img, img_fc2), F.linear(txt, txt_fc2)


def tp_residual(img, txt, img_partial, txt_partial, img_bias, txt_bias, img_gate, txt_gate):
    if img_bias is not None:
        img_partial = img_partial + img_bias
    if txt_bias is not None:
        txt_partial = txt_partial + txt_bias
    return img + img_partial * img_gate.unsqueeze(1), txt + txt_partial * txt_gate.unsqueeze(1)


@lru_cache(maxsize=None)
def compiled_tp_region(name):
    regions = dict(
        modulation=tp_modulation, local_norm=tp_local_norm, qkv=tp_qkv,
        attention_partial=tp_attention_partial, residual_norm=tp_residual_norm,
        mlp_partial=tp_mlp_partial, residual=tp_residual,
    )
    return torch.compile(
        regions[name], fullgraph=True,
        options={"triton.cudagraphs": False, "emulate_precision_casts": True},
    )


def tp_pre_attention(block, img, txt, vec, vec_txt, freqs):
    """Compile local math; keep TP token gathers outside compiled regions."""
    from hymm.distributed.tensor_parallel import gather_sequence
    if vec is not vec_txt:
        # Normal cache inference uses the same timestep for both streams.
        img_mod = block.img_mod(vec)
        txt_mod = block.txt_mod(vec_txt)
    else:
        img_mod, txt_mod = compiled_tp_region("modulation")(
            vec, block.img_mod.linear.weight, block.img_mod.linear.bias,
            block.txt_mod.linear.weight, block.txt_mod.linear.bias,
        )
    img_parts, txt_parts = img_mod.chunk(6, -1), txt_mod.chunk(6, -1)
    img_norm, txt_norm = compiled_tp_region("local_norm")(img, txt)
    group = block.tensor_parallel_group
    img_norm = gather_sequence(img_norm, group, block.img_norm1.sequence_length)
    txt_norm = gather_sequence(txt_norm, group, block.txt_norm1.sequence_length)
    qkv = compiled_tp_region("qkv")(
        img_norm, txt_norm, img_parts[0], img_parts[1], txt_parts[0], txt_parts[1],
        freqs[0], freqs[1],
        block.img_attn_q.weight, block.img_attn_q.bias,
        block.img_attn_k.weight, block.img_attn_k.bias,
        block.img_attn_v.weight, block.img_attn_v.bias,
        block.txt_attn_q.weight, block.txt_attn_q.bias,
        block.txt_attn_k.weight, block.txt_attn_k.bias,
        block.txt_attn_v.weight, block.txt_attn_v.bias,
        block.img_attn_q_norm.weight, block.img_attn_k_norm.weight,
        block.txt_attn_q_norm.weight, block.txt_attn_k_norm.weight,
        block.heads_num, block.img_attn_q_norm.eps,
    )
    return (*qkv, *img_parts[2:], *txt_parts[2:])


def tp_post_attention(
    block, img, txt, img_attn, txt_attn,
    img_gate1, img_shift2, img_scale2, img_gate2,
    txt_gate1, txt_shift2, txt_scale2, txt_gate2,
):
    """Retain both attention and MLP reduce-scatter sums and post-sum biases."""
    from hymm.distributed.tensor_parallel import gather_sequence, reduce_scatter_sequence
    group = block.tensor_parallel_group
    img_partial, txt_partial = compiled_tp_region("attention_partial")(
        img_attn, txt_attn, block.img_attn_proj.weight, block.txt_attn_proj.weight,
    )
    img_partial = reduce_scatter_sequence(img_partial, group)
    txt_partial = reduce_scatter_sequence(txt_partial, group)
    img, txt, img_norm, txt_norm = compiled_tp_region("residual_norm")(
        img, txt, img_partial, txt_partial, block.img_attn_proj.bias, block.txt_attn_proj.bias,
        img_gate1, txt_gate1,
    )
    img_norm = gather_sequence(img_norm, group, block.img_norm2.sequence_length)
    txt_norm = gather_sequence(txt_norm, group, block.txt_norm2.sequence_length)
    img_partial, txt_partial = compiled_tp_region("mlp_partial")(
        img_norm, txt_norm, img_shift2, img_scale2, txt_shift2, txt_scale2,
        block.img_mlp.fc1.weight, block.img_mlp.fc1.bias, block.img_mlp.fc2.weight,
        block.txt_mlp.fc1.weight, block.txt_mlp.fc1.bias, block.txt_mlp.fc2.weight,
    )
    img_partial = reduce_scatter_sequence(img_partial, group)
    txt_partial = reduce_scatter_sequence(txt_partial, group)
    return compiled_tp_region("residual")(
        img, txt, img_partial, txt_partial, block.img_mlp.fc2.bias, block.txt_mlp.fc2.bias,
        img_gate2, txt_gate2,
    )


def compilation_snapshot():
    """Copy scalar Dynamo timers without resetting process-wide compiler state."""
    from torch._dynamo.utils import compilation_time_metrics
    return {
        name: {"calls": len(values), "seconds": float(sum(values))}
        for name, values in compilation_time_metrics.items()
    }


def compilation_delta(before):
    after = compilation_snapshot()
    regions = {}
    for name, current in after.items():
        previous = before.get(name, {"calls": 0, "seconds": 0.})
        calls = current["calls"] - previous["calls"]
        if calls > 0:
            regions[name] = {
                "calls": calls,
                "seconds": max(0., current["seconds"] - previous["seconds"]),
            }
    return {
        "torch_version": torch.__version__,
        "compiler_invoked": bool(regions),
        "timed_regions": regions,
        "note": "Dynamo timed regions overlap; never sum them. Sample time includes compilation.",
    }
