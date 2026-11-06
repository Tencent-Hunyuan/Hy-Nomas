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
