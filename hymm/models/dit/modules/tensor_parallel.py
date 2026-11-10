"""Inference-only tensor parallelism for the Nomas DiT and its Qwen text encoder.

DiT dual-stream blocks keep the residual stream of both streams sharded along the sequence between
blocks. Inside a block:

    all-gather(seq) -> column-parallel Q/K/V (local heads) -> attention with a head-sharded
    KV cache -> all-to-all (heads to sequence) -> output projection, residual, modulation and MLP
    on the sequence shard with whole weights.

Everything after the attention acts on each token alone, so it needs no communication: per rank it
costs the same as the column/row-parallel layers would, for the memory of unsharded weights.

The first block shards its replicated input and the last block gathers its output, so the
transformer outside the blocks is unchanged. Camera action modules, which run between blocks, see
the whole image sequence: it is gathered before and sharded after each of them. The text encoder
uses plain Megatron TP with all-reduce through DTensor.
"""
import types
import os
from functools import lru_cache

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .nomas_compile_prepost import _modulate, _rms_norm, _rotate_half, _qkv, _mlp


def _use_cudagraphs(fp8):
    override = os.environ.get("HY_DIT_CUDAGRAPHS")
    return override == "1" if override is not None else fp8 and os.environ.get("HY_DIT_FP8_CUDAGRAPHS", "0") == "1"


def _linear(x, weight, bias):
    return weight(x) if isinstance(weight, nn.Module) else F.linear(x, weight, bias)


def _operand(module):
    return module if hasattr(module, 'weight_q') else module.weight


def _shard_rows(linear, rank, size):
    out = linear.out_features // size
    linear.weight = nn.Parameter(linear.weight[rank * out:(rank + 1) * out].contiguous(),
                                 requires_grad=False)
    if linear.bias is not None:
        linear.bias = nn.Parameter(linear.bias[rank * out:(rank + 1) * out].contiguous(),
                                   requires_grad=False)
    linear.out_features = out


class _SeqComm:
    """Sequence all-gather / all-to-all for [1, L, C] tensors; pads L to a multiple of size."""

    def __init__(self, group, rank, size):
        self.group, self.rank, self.size = group, rank, size

    def shard(self, x):
        length = x.shape[1]
        padded = -(-length // self.size) * self.size
        if padded != length:
            x = F.pad(x, (0, 0, 0, padded - length))
        part = padded // self.size
        return x[:, self.rank * part:(self.rank + 1) * part].contiguous(), length

    def gather(self, x, length):
        out = x.new_empty((x.shape[0] * self.size, *x.shape[1:]))  # [size, L/size, C] when B=1
        dist.all_gather_into_tensor(out, x.contiguous(), group=self.group)
        out = out.reshape(1, -1, x.shape[-1])
        return out[:, :length] if out.shape[1] != length else out

    def heads_to_sequence(self, x):
        """[1, L, C / size] of the local heads to [1, L / size, C] of the local tokens."""
        length = x.shape[1]
        padded = -(-length // self.size) * self.size
        if padded != length:
            x = F.pad(x, (0, 0, 0, padded - length))
        part = padded // self.size
        sent = x.reshape(self.size, part, x.shape[-1]).contiguous()
        received = sent.new_empty(sent.shape)
        dist.all_to_all_single(received, sent, group=self.group)
        # Row r holds rank r's heads, and the heads were sharded in rank order.
        return received.transpose(0, 1).reshape(1, part, -1)


def _pre_attention(
    img, txt, vec, vec_txt, cos, sin,
    img_mod_w, img_mod_b, txt_mod_w, txt_mod_b,
    img_q_w, img_q_b, img_k_w, img_k_b, img_v_w, img_v_b,
    txt_q_w, txt_q_b, txt_k_w, txt_k_b, txt_v_w, txt_v_b,
    img_q_norm_w, img_k_norm_w, txt_q_norm_w, txt_k_norm_w,
    heads, qk_eps,
):
    """nomas_compile_prepost.pre_attention with Q/K/V holding only the local heads."""
    img_parts = _linear(F.silu(vec), img_mod_w, img_mod_b).chunk(6, dim=-1)
    txt_parts = _linear(F.silu(vec_txt), txt_mod_w, txt_mod_b).chunk(6, dim=-1)
    head_dim = (img_q_w.out_features if isinstance(img_q_w, nn.Module) else img_q_w.shape[0]) // heads

    def project(x, parts, q_w, q_b, k_w, k_b, v_w, v_b, q_norm_w, k_norm_w):
        hidden = _modulate(F.layer_norm(x, (x.shape[-1],), eps=1e-6), parts[0], parts[1])
        q, k, v = (y.reshape(x.shape[0], -1, heads, head_dim)
                   for y in _qkv(hidden, q_w, q_b, k_w, k_b, v_w, v_b))
        return _rms_norm(q, q_norm_w, qk_eps).to(v), _rms_norm(k, k_norm_w, qk_eps).to(v), v

    img_q, img_k, img_v = project(img, img_parts, img_q_w, img_q_b, img_k_w, img_k_b,
                                  img_v_w, img_v_b, img_q_norm_w, img_k_norm_w)
    cos = cos.view(1, img_q.shape[1], 1, head_dim).to(img_q.device)
    sin = sin.view(1, img_q.shape[1], 1, head_dim).to(img_q.device)
    img_q = (img_q.float() * cos + _rotate_half(img_q) * sin).type_as(img_v)
    img_k = (img_k.float() * cos + _rotate_half(img_k) * sin).type_as(img_v)
    txt_q, txt_k, txt_v = project(txt, txt_parts, txt_q_w, txt_q_b, txt_k_w, txt_k_b,
                                  txt_v_w, txt_v_b, txt_q_norm_w, txt_k_norm_w)
    return (
        img_q, img_k, img_v, txt_q, txt_k, txt_v,
        img_parts[2], img_parts[3], img_parts[4], img_parts[5],
        txt_parts[2], txt_parts[3], txt_parts[4], txt_parts[5],
    )


def _post_attention(x, attn, o_w, o_b, gate1, shift2, scale2, fc1_w, fc1_b, fc2_w, fc2_b, gate2):
    """Output projection, residual, modulation and MLP of the local tokens."""
    x = x + _linear(attn, o_w, o_b) * gate1.unsqueeze(1)
    hidden = _modulate(F.layer_norm(x, (x.shape[-1],), eps=1e-6), shift2, scale2)
    mlp = _mlp(hidden, fc1_w, fc1_b, fc2_w, fc2_b)
    return x + mlp * gate2.unsqueeze(1)


@lru_cache(maxsize=None)
def _maybe_compiled(name, enabled, fp8=False):
    fn = {"pre": _pre_attention, "post": _post_attention}[name]
    if not enabled:
        return fn
    return torch.compile(fn, fullgraph=True, options={"triton.cudagraphs": _use_cudagraphs(fp8)})


def _tp_forward_with_cache(self, img, txt, vec, vec_txt=None, freqs_cis=None, text_mask=None,
                           kv_cache=None, cache_valid_len=0, update_cache=False, attn_mode=None,
                           cache_txt=True, cache_mask=None, intra_seg_txt_offsets=None,
                           intra_seg_vid_windows=None, intra_seg_tokens_per_frame=0,
                           sparse_context=None, attention_metadata=None, sp_metadata=None):
    from hymm.models.dit.nomas_cpt import causal_attention_with_cache_mp

    comm, tp = self._tp_comm, self._tp_state
    if img.shape[0] != 1 or txt.shape[0] != 1:
        raise ValueError("DiT tensor parallelism supports batch 1 (CFG=1) only")
    if not isinstance(freqs_cis, tuple):
        raise ValueError("DiT tensor parallelism requires tuple cos/sin RoPE")
    vec_txt = vec if vec_txt is None else vec_txt
    compile_enabled = bool(getattr(self, "compile_cache_prepost", False))

    # Residual streams: replicated into the first block, sequence-sharded afterwards.
    if self._tp_first:
        tp["img_len"], tp["txt_len"] = img.shape[1], txt.shape[1]
        img_full, txt_full = img, txt
        img, _ = comm.shard(img)
        txt, _ = comm.shard(txt)
    else:
        img_full = comm.gather(img, tp["img_len"])
        txt_full = comm.gather(txt, tp["txt_len"])

    pre = _maybe_compiled("pre", compile_enabled, getattr(self.img_attn_q, "is_fp8_linear", False))
    (img_q, img_k, img_v, txt_q, txt_k, txt_v,
     img_gate1, img_shift2, img_scale2, img_gate2,
     txt_gate1, txt_shift2, txt_scale2, txt_gate2) = pre(
        img_full, txt_full, vec, vec_txt, freqs_cis[0], freqs_cis[1],
        self.img_mod.linear.weight, self.img_mod.linear.bias,
        self.txt_mod.linear.weight, self.txt_mod.linear.bias,
        _operand(self.img_attn_q), self.img_attn_q.bias, _operand(self.img_attn_k), self.img_attn_k.bias,
        _operand(self.img_attn_v), self.img_attn_v.bias,
        _operand(self.txt_attn_q), self.txt_attn_q.bias, _operand(self.txt_attn_k), self.txt_attn_k.bias,
        _operand(self.txt_attn_v), self.txt_attn_v.bias,
        self.img_attn_q_norm.weight, self.img_attn_k_norm.weight,
        self.txt_attn_q_norm.weight, self.txt_attn_k_norm.weight,
        self.heads_num, self.img_attn_q_norm.eps,
    )
    del img_full, txt_full
    img_attn, txt_attn = causal_attention_with_cache_mp(
        (img_q, txt_q), (img_k, txt_k), (img_v, txt_v),
        kv_cache=kv_cache, cache_valid_len=cache_valid_len,
        attn_mode=attn_mode or self.attn_mode, text_mask=text_mask,
        update_cache=update_cache, cache_txt=cache_txt, cache_mask=cache_mask,
        intra_seg_txt_offsets=intra_seg_txt_offsets, intra_seg_vid_windows=intra_seg_vid_windows,
        tokens_per_frame=intra_seg_tokens_per_frame,
        seg_ar_attention_backend=self.seg_ar_attention_backend,
        # Routes are per head and each rank holds its heads and their cache, so it routes alone.
        sparse_context=sparse_context,
        attention_metadata=attention_metadata,
    )
    del img_q, img_k, img_v, txt_q, txt_k, txt_v

    post = _maybe_compiled("post", compile_enabled, getattr(self.img_attn_proj, "is_fp8_linear", False))
    streams = []
    for x, attn, proj, block_mlp, gate1, shift2, scale2, gate2 in (
        (img, img_attn, self.img_attn_proj, self.img_mlp,
         img_gate1, img_shift2, img_scale2, img_gate2),
        (txt, txt_attn, self.txt_attn_proj, self.txt_mlp,
         txt_gate1, txt_shift2, txt_scale2, txt_gate2),
    ):
        attn = comm.heads_to_sequence(attn.reshape(1, attn.shape[1], -1))
        streams.append(post(x, attn, _operand(proj), proj.bias, gate1, shift2, scale2,
                            _operand(block_mlp.fc1), block_mlp.fc1.bias,
                            _operand(block_mlp.fc2), block_mlp.fc2.bias, gate2))
    img, txt = streams

    if self._tp_last:
        img = comm.gather(img, tp["img_len"])
        txt = comm.gather(txt, tp["txt_len"])
    return img, txt


def _camera_on_gathered_sequence(module, comm, last_block):
    """Run a camera action module on the whole sequence; the last block has already gathered it."""
    forward = module.forward

    def wrapped(hidden_states, tt, th, tw, *args, **kwargs):
        if last_block:
            return forward(hidden_states, tt, th, tw, *args, **kwargs)
        whole = comm.gather(hidden_states, tt * th * tw)
        return comm.shard(forward(whole, tt, th, tw, *args, **kwargs))[0]

    module.forward = wrapped


def apply_dit_tensor_parallel(transformer, group=None):
    """Shard every double block in place and install the sequence-parallel TP forward."""
    group = group or dist.group.WORLD
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    blocks = list(transformer.double_blocks)
    if transformer.heads_num % size:
        raise ValueError(f"heads_num={transformer.heads_num} is not divisible by tp={size}")
    if getattr(transformer, "single_blocks", None):
        raise ValueError("DiT tensor parallelism covers double-stream blocks only")
    transformer.fp8_cudagraphs = _use_cudagraphs(any(getattr(b.img_attn_q, "is_fp8_linear", False) for b in blocks))
    comm, state = _SeqComm(group, rank, size), {}
    for index, block in enumerate(blocks):
        for name in ("img_attn_q", "img_attn_k", "img_attn_v", "txt_attn_q", "txt_attn_k", "txt_attn_v"):
            linear = getattr(block, name)
            from .fp8_linear import FP8Linear
            if isinstance(linear, FP8Linear):
                linear.shard_rows(rank, size)
                linear.prepare_for_inference()
            elif type(linear) is nn.Linear:
                _shard_rows(linear, rank, size)
            else:
                raise ValueError(f"Unsupported TP projection: {type(linear)}")
        from .fp8_linear import prepare_fp8_qkv
        prepare_fp8_qkv(block)
        block.heads_num //= size
        block._tp_comm, block._tp_state = comm, state
        block._tp_first, block._tp_last = index == 0, index == len(blocks) - 1
        block.forward_with_cache = types.MethodType(_tp_forward_with_cache, block)
    for block, module in zip(blocks, getattr(transformer, "action_modules", None) or ()):
        _camera_on_gathered_sequence(module, comm, block._tp_last)
    return rank, size


def apply_text_encoder_tensor_parallel(text_encoder, mesh):
    """Megatron TP (column q/k/v/gate/up, row o/down + all-reduce) on a Qwen2-style decoder."""
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

    model = getattr(text_encoder, "model", text_encoder)
    size = mesh.size()
    plan = {
        "self_attn.q_proj": ColwiseParallel(), "self_attn.k_proj": ColwiseParallel(),
        "self_attn.v_proj": ColwiseParallel(), "self_attn.o_proj": RowwiseParallel(),
        "mlp.gate_proj": ColwiseParallel(), "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(),
    }
    for layer in model.layers:
        attention = layer.self_attn
        config = attention.config
        if config.num_attention_heads % size or config.num_key_value_heads % size:
            raise ValueError("text encoder heads are not divisible by the TP size")
        parallelize_module(layer, mesh, plan)
        # HF attention reshapes with -1 heads; keep the explicit counts consistent with the shard.
        for name in ("num_heads", "num_key_value_heads"):
            if hasattr(attention, name):
                setattr(attention, name, getattr(attention, name) // size)
    return model
