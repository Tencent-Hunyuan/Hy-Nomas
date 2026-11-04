"""Inference-only tensor parallelism for the Nomas DiT and its Qwen text encoder.

DiT dual-stream blocks use the Megatron sequence-parallel layout. Between blocks the
residual stream of both streams is sharded along the sequence. Inside a block:

    all-gather(seq) -> column-parallel Q/K/V (local heads) -> attention with a head-sharded
    KV cache -> row-parallel output projection -> reduce-scatter(seq) -> residual/modulation on
    the shard -> all-gather(seq) -> column-parallel fc1 -> GELU -> row-parallel fc2 ->
    reduce-scatter(seq) -> residual.

The first block shards its replicated input and the last block gathers its output, so the
transformer outside the blocks is unchanged. Camera action modules, which run between blocks, see
the whole image sequence: it is gathered before and sharded after each of them. The text encoder
uses plain Megatron TP with all-reduce through DTensor.
"""
import types
from functools import lru_cache

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .nomas_compile_prepost import _modulate, _rms_norm, _rotate_half


def _shard_rows(linear, rank, size):
    out = linear.out_features // size
    linear.weight = nn.Parameter(linear.weight[rank * out:(rank + 1) * out].contiguous(),
                                 requires_grad=False)
    if linear.bias is not None:
        linear.bias = nn.Parameter(linear.bias[rank * out:(rank + 1) * out].contiguous(),
                                   requires_grad=False)
    linear.out_features = out


def _shard_cols(linear, rank, size):
    # The bias stays whole and is added once after the reduce-scatter.
    columns = linear.in_features // size
    linear.weight = nn.Parameter(linear.weight[:, rank * columns:(rank + 1) * columns].contiguous(),
                                 requires_grad=False)
    linear.in_features = columns


class _SeqComm:
    """Sequence all-gather / reduce-scatter for [1, L, C] tensors; pads L to a multiple of size."""

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

    def reduce_scatter(self, x):
        length = x.shape[1]
        padded = -(-length // self.size) * self.size
        if padded != length:
            x = F.pad(x, (0, 0, 0, padded - length))
        out = x.new_empty((1, padded // self.size, x.shape[-1]))
        dist.reduce_scatter_tensor(out, x.contiguous(), op=dist.ReduceOp.SUM, group=self.group)
        return out


def _pre_attention(
    img, txt, vec, vec_txt, cos, sin,
    img_mod_w, img_mod_b, txt_mod_w, txt_mod_b,
    img_q_w, img_q_b, img_k_w, img_k_b, img_v_w, img_v_b,
    txt_q_w, txt_q_b, txt_k_w, txt_k_b, txt_v_w, txt_v_b,
    img_q_norm_w, img_k_norm_w, txt_q_norm_w, txt_k_norm_w,
    heads, qk_eps,
):
    """nomas_compile_prepost.pre_attention with Q/K/V holding only the local heads."""
    img_parts = F.linear(F.silu(vec), img_mod_w, img_mod_b).chunk(6, dim=-1)
    txt_parts = F.linear(F.silu(vec_txt), txt_mod_w, txt_mod_b).chunk(6, dim=-1)
    head_dim = img_q_w.shape[0] // heads

    def project(x, parts, q_w, q_b, k_w, k_b, v_w, v_b, q_norm_w, k_norm_w):
        hidden = _modulate(F.layer_norm(x, (x.shape[-1],), eps=1e-6), parts[0], parts[1])
        q, k, v = (F.linear(hidden, w, b).reshape(x.shape[0], -1, heads, head_dim)
                   for w, b in ((q_w, q_b), (k_w, k_b), (v_w, v_b)))
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


def _attn_out_and_norm2(x, attn_reduced, o_bias, gate1, shift2, scale2):
    x = x + (attn_reduced + o_bias) * gate1.unsqueeze(1)
    hidden = _modulate(F.layer_norm(x, (x.shape[-1],), eps=1e-6), shift2, scale2)
    return x, hidden


def _mlp_partial(hidden, fc1_w, fc1_b, fc2_w):
    return F.linear(F.gelu(F.linear(hidden, fc1_w, fc1_b), approximate="tanh"), fc2_w)


def _mlp_residual(x, mlp_reduced, fc2_b, gate2):
    return x + (mlp_reduced + fc2_b) * gate2.unsqueeze(1)


@lru_cache(maxsize=None)
def _maybe_compiled(name, enabled):
    fn = {"pre": _pre_attention, "norm2": _attn_out_and_norm2, "mlp": _mlp_partial,
          "residual": _mlp_residual}[name]
    if not enabled:
        return fn
    return torch.compile(fn, fullgraph=True, options={"triton.cudagraphs": False})


def _tp_forward_with_cache(self, img, txt, vec, vec_txt=None, freqs_cis=None, text_mask=None,
                           kv_cache=None, cache_valid_len=0, update_cache=False, attn_mode=None,
                           cache_txt=True, cache_mask=None, intra_seg_txt_offsets=None,
                           intra_seg_vid_windows=None, intra_seg_tokens_per_frame=0,
                           sparse_context=None):
    from hymm.models.dit.nomas_cpt import causal_attention_with_cache_mp

    comm, tp = self._tp_comm, self._tp_state
    if img.shape[0] != 1 or txt.shape[0] != 1:
        raise ValueError("DiT tensor parallelism supports batch 1 (CFG=1) only")
    if sparse_context is not None:
        raise ValueError("DiT tensor parallelism does not support sparse attention")
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

    pre = _maybe_compiled("pre", compile_enabled)
    (img_q, img_k, img_v, txt_q, txt_k, txt_v,
     img_gate1, img_shift2, img_scale2, img_gate2,
     txt_gate1, txt_shift2, txt_scale2, txt_gate2) = pre(
        img_full, txt_full, vec, vec_txt, freqs_cis[0], freqs_cis[1],
        self.img_mod.linear.weight, self.img_mod.linear.bias,
        self.txt_mod.linear.weight, self.txt_mod.linear.bias,
        self.img_attn_q.weight, self.img_attn_q.bias, self.img_attn_k.weight, self.img_attn_k.bias,
        self.img_attn_v.weight, self.img_attn_v.bias,
        self.txt_attn_q.weight, self.txt_attn_q.bias, self.txt_attn_k.weight, self.txt_attn_k.bias,
        self.txt_attn_v.weight, self.txt_attn_v.bias,
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
        seg_ar_attention_backend=self.seg_ar_attention_backend, sparse_context=None,
    )
    del img_q, img_k, img_v, txt_q, txt_k, txt_v

    norm2 = _maybe_compiled("norm2", compile_enabled)
    mlp = _maybe_compiled("mlp", compile_enabled)
    residual = _maybe_compiled("residual", compile_enabled)
    streams = []
    for x, attn, length, proj, block_mlp, gate1, shift2, scale2, gate2 in (
        (img, img_attn, tp["img_len"], self.img_attn_proj, self.img_mlp,
         img_gate1, img_shift2, img_scale2, img_gate2),
        (txt, txt_attn, tp["txt_len"], self.txt_attn_proj, self.txt_mlp,
         txt_gate1, txt_shift2, txt_scale2, txt_gate2),
    ):
        attn = attn.reshape(1, attn.shape[1], -1)
        reduced = comm.reduce_scatter(F.linear(attn, proj.weight))
        x, hidden = norm2(x, reduced, proj.bias, gate1, shift2, scale2)
        hidden = comm.gather(hidden, length)
        reduced = comm.reduce_scatter(mlp(hidden, block_mlp.fc1.weight, block_mlp.fc1.bias,
                                          block_mlp.fc2.weight))
        streams.append(residual(x, reduced, block_mlp.fc2.bias, gate2))
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
    comm, state = _SeqComm(group, rank, size), {}
    for index, block in enumerate(blocks):
        for name in ("img_attn_q", "img_attn_k", "img_attn_v", "txt_attn_q", "txt_attn_k", "txt_attn_v"):
            linear = getattr(block, name)
            if type(linear) is not nn.Linear:
                raise ValueError(f"tensor parallelism needs nn.Linear for {name}, got {type(linear)}")
            _shard_rows(linear, rank, size)
        for mlp in (block.img_mlp, block.txt_mlp):
            _shard_rows(mlp.fc1, rank, size)
            _shard_cols(mlp.fc2, rank, size)
        _shard_cols(block.img_attn_proj, rank, size)
        _shard_cols(block.txt_attn_proj, rank, size)
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
