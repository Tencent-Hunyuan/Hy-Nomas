"""DiT tensor parallelism with sequence-sharded residual activations.

LayerNorm consumes local tokens and all-gathers normalized tokens for column
projections. Attention/MLP row projections reduce-scatter partial products back
to local tokens, then add bias once. Image and text streams use independent
sequence partitions; Q/K/V keep the full token sequence and shard only heads.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist


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
    if getattr(inference, "dit_weight_quantization", "none") != "none":
        raise ValueError("Tensor parallel currently requires unquantized DiT weights")
    if getattr(inference, "lora_adapter", None):
        raise ValueError("Tensor parallel does not yet support LoRA adapters")
    if getattr(inference, "torch_compile", False):
        raise ValueError("Tensor parallel requires inference.torch_compile=false; fused projections bypass TP reductions")
    if getattr(parallelism, "use_dynamic_ring_attention", False):
        raise ValueError("Tensor parallel does not yet support dynamic ring attention")
    if getattr(dit, "use_moe", False):
        raise ValueError("Tensor parallel currently supports dense DiT MLPs, not MoE")


def sequence_partition(length, size, rank):
    """Balanced contiguous shards, including empty shards for short text."""
    width, remainder = divmod(length, size)
    return rank * width + min(rank, remainder), width + int(rank < remainder)


def _sequence_sizes(value, group, sequence_length=None):
    size, rank = _group_info(group)
    if sequence_length is None:
        local = torch.tensor([value.shape[1]], device=value.device, dtype=torch.int64)
        lengths = [torch.empty_like(local) for _ in range(size)]
        dist.all_gather(lengths, local, group=group)
        sizes = tuple(int(length.item()) for length in lengths)
    else:
        sizes = tuple(sequence_partition(sequence_length, size, r)[1] for r in range(size))
    if value.shape[1] != sizes[rank]:
        raise ValueError("Local sequence length does not match the TP partition")
    return sizes


def _gather_sequence(value, group, sizes):
    """Gather ragged token shards; padding never enters attention or the cache."""
    size = len(sizes)
    maximum = max(sizes)
    if maximum == 0:
        return value.clone()
    local = value.movedim(1, 0).contiguous()
    if local.shape[0] < maximum:
        padded = local.new_zeros((maximum, *local.shape[1:]))
        padded[:local.shape[0]].copy_(local)
        local = padded
    output = local.new_empty((size * maximum, *local.shape[1:]))
    dist.all_gather_into_tensor(output, local, group=group)
    if len(set(sizes)) == 1:
        return output.movedim(0, 1).contiguous()
    chunks = output.split(maximum, dim=0)
    return torch.cat([chunk[:length] for chunk, length in zip(chunks, sizes)], dim=0).movedim(0, 1).contiguous()


def _reduce_scatter_sequence(value, group, sizes):
    maximum = max(sizes)
    rank = dist.get_rank(group)
    if maximum == 0:
        return value[:, :0].clone()
    moved = value.movedim(1, 0).contiguous()
    if len(set(sizes)) == 1:
        packed = moved
    else:
        # reduce_scatter_tensor requires equal transfer sizes, not equal real
        # sequence lengths. Pad each destination shard independently.
        padded = []
        for chunk in moved.split(sizes, dim=0):
            item = moved.new_zeros((maximum, *moved.shape[1:]))
            item[:chunk.shape[0]].copy_(chunk)
            padded.append(item)
        packed = torch.cat(padded, dim=0)
    output = moved.new_empty((maximum, *moved.shape[1:]))
    dist.reduce_scatter_tensor(output, packed, op=dist.ReduceOp.SUM, group=group)
    return output[:sizes[rank]].movedim(0, 1).contiguous()


class _GatherSequence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group, sequence_length):
        ctx.group = group
        ctx.sizes = _sequence_sizes(value, group, sequence_length)
        return _gather_sequence(value, group, ctx.sizes)

    @staticmethod
    def backward(ctx, grad):
        return _reduce_scatter_sequence(grad, ctx.group, ctx.sizes), None, None


class _ReduceScatterSequence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group):
        ctx.group = group
        size = dist.get_world_size(group)
        ctx.sizes = tuple(sequence_partition(value.shape[1], size, r)[1] for r in range(size))
        return _reduce_scatter_sequence(value, group, ctx.sizes)

    @staticmethod
    def backward(ctx, grad):
        return _gather_sequence(grad, ctx.group, ctx.sizes), None


class _ScatterSequence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group):
        size, rank = _group_info(group)
        ctx.group = group
        ctx.sizes = tuple(sequence_partition(value.shape[1], size, r)[1] for r in range(size))
        start, length = sequence_partition(value.shape[1], size, rank)
        return value.narrow(1, start, length).clone(memory_format=torch.contiguous_format)

    @staticmethod
    def backward(ctx, grad):
        return _gather_sequence(grad, ctx.group, ctx.sizes), None


def scatter_sequence(value, group):
    return _ScatterSequence.apply(value, group)


def gather_sequence(value, group, sequence_length=None):
    return _GatherSequence.apply(value, group, sequence_length)


def reduce_scatter_sequence(value, group):
    return _ReduceScatterSequence.apply(value, group)


class SequenceParallelLayerNorm(nn.Module):
    """Normalize local [B,S/TP,H] tokens, then gather to [B,S,H]."""
    def __init__(self, norm, group):
        super().__init__()
        if not isinstance(norm, nn.LayerNorm):
            raise ValueError("Sequence-parallel norm requires nn.LayerNorm")
        _group_info(group)
        self.normalized_shape = norm.normalized_shape
        self.eps = norm.eps
        self.weight = norm.weight
        self.bias = norm.bias
        self.group = group
        # Model entry sets this once per forward to avoid metadata collectives
        # at every norm. Standalone blocks can discover ragged shard lengths.
        self.sequence_length = None
        self.train(norm.training)

    def forward(self, value):
        normalized = F.layer_norm(value, self.normalized_shape, self.weight, self.bias, self.eps)
        return gather_sequence(normalized, self.group, self.sequence_length)


def shard_transformer_streams(model, img, txt):
    """Enter the TP residual layout once, before the first DiT block."""
    if getattr(model, "tensor_parallel_size", 1) == 1:
        return img, txt
    for block in model.double_blocks:
        for stream, value in (("img", img), ("txt", txt)):
            for suffix in ("norm1", "norm2"):
                getattr(block, f"{stream}_{suffix}").sequence_length = value.shape[1]
    model.final_layer.norm_final.sequence_length = img.shape[1]
    group = model.tensor_parallel_group
    return scatter_sequence(img, group), scatter_sequence(txt, group)


def local_modulation_gate(block, gate):
    """Only gates multiply local residuals; shifts/scales follow gathered norms."""
    if gate.dim() == 3 and getattr(block, "tensor_parallel_size", 1) > 1:
        return scatter_sequence(gate, block.tensor_parallel_group)
    return gate


def _group_info(group):
    if not dist.is_initialized() or group is None:
        raise ValueError("Tensor parallel requires an initialized TP process group")
    return dist.get_world_size(group), dist.get_rank(group)


def _slice_parts(value, dim, parts, size, rank):
    if sum(parts) != value.shape[dim] or any(part % size for part in parts):
        raise ValueError(f"Projection dimensions {parts} must each be divisible by tp_size={size}")
    chunks = value.split(parts, dim=dim)
    return torch.cat([chunk.narrow(dim, rank * (part // size), part // size)
                      for chunk, part in zip(chunks, parts)], dim=dim).contiguous()


class ColumnParallelLinear(nn.Module):
    """Replicated input -> a contiguous local slice of output features."""
    def __init__(self, linear, group):
        super().__init__()
        if type(linear) is not nn.Linear:
            raise ValueError("Column parallel conversion requires an ordinary nn.Linear")
        self.tp_size, self.tp_rank = _group_info(group)
        self.group = group
        self.in_features = linear.in_features
        self.out_features = linear.out_features // self.tp_size
        self.weight = nn.Parameter(_slice_parts(linear.weight.detach(), 0,
                                  (linear.out_features,), self.tp_size, self.tp_rank),
                                  requires_grad=linear.weight.requires_grad)
        self.bias = None if linear.bias is None else nn.Parameter(
            _slice_parts(linear.bias.detach(), 0, (linear.out_features,), self.tp_size, self.tp_rank),
            requires_grad=linear.bias.requires_grad)
        self.train(linear.training)

    def forward(self, value):
        # The preceding norm's all-gather owns the reduce-scatter adjoint.
        # A second copy-to-TP all-reduce here would multiply input gradients.
        return F.linear(value, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """Local input features -> sequence-sharded output after reduce-scatter SUM.

    input_parts describes concatenated branches (e.g. attention + MLP in a
    single-stream block). Each branch must be sliced independently rather than
    treating the concatenation as one contiguous shard.
    """
    def __init__(self, linear, group, input_parts=None):
        super().__init__()
        if type(linear) is not nn.Linear:
            raise ValueError("Row parallel conversion requires an ordinary nn.Linear")
        self.tp_size, self.tp_rank = _group_info(group)
        self.group = group
        self.in_features = linear.in_features // self.tp_size
        self.out_features = linear.out_features
        parts = tuple(input_parts or (linear.in_features,))
        self.weight = nn.Parameter(_slice_parts(linear.weight.detach(), 1, parts,
                                  self.tp_size, self.tp_rank),
                                  requires_grad=linear.weight.requires_grad)
        self.bias = None if linear.bias is None else nn.Parameter(
            linear.bias.detach().clone(), requires_grad=linear.bias.requires_grad)
        self.train(linear.training)

    def forward(self, value):
        output = reduce_scatter_sequence(F.linear(value, self.weight), self.group)
        return output if self.bias is None else output + self.bias


def parallelize_transformer(model, group, sp_size=1):
    """Shard dense double/single-stream blocks after loading the full checkpoint.

    Global model hidden_size/heads_num stay unchanged for patch embeddings,
    RoPE and camera modules. Each block uses local heads; KV storage is allocated
    separately with heads_num / (SP * TP). Ancillary modules remain replicated.
    """
    size, rank = _group_info(group)
    if size == 1:
        return model
    if getattr(model, "tensor_parallel_size", 1) != 1:
        raise ValueError("Transformer is already tensor parallel")
    blocks = list(model.double_blocks) + list(model.single_blocks)
    # Validate the complete plan before mutating any parameters.
    plans = []
    norms = []
    for block in blocks:
        if block.heads_num % (size * sp_size):
            raise ValueError(f"Attention heads ({block.heads_num}) must be divisible by sp_size * tp_size ({sp_size * size})")
        if block.moe_config is not None:
            raise ValueError("Tensor parallel currently supports dense DiT MLPs, not MoE")
        if getattr(block, "compile_cache_prepost", False):
            raise ValueError("Compiled cache projections bypass tensor-parallel reductions")
        if hasattr(block, "img_attn_q"):
            for stream in ("img", "txt"):
                norms.extend((block, f"{stream}_norm{i}") for i in (1, 2))
                for projection in ("q", "k", "v"):
                    plans.append((block, f"{stream}_attn_{projection}", "column", None))
                plans.append((block, f"{stream}_attn_proj", "row", None))
                mlp = getattr(block, f"{stream}_mlp")
                if not isinstance(mlp.norm, nn.Identity):
                    raise ValueError("Tensor-parallel MLP requires no intermediate normalization")
                plans.extend([(mlp, "fc1", "column", None), (mlp, "fc2", "row", None)])
        else:
            norms.append((block, "pre_norm"))
            for name in ("linear1_q", "linear1_k", "linear1_v", "linear1_mlp"):
                plans.append((block, name, "column", None))
            plans.append((block.linear2, "fc", "row", (block.hidden_size, block.mlp_hidden_dim)))
    for parent, name, kind, parts in plans:
        linear = getattr(parent, name)
        if type(linear) is not nn.Linear:
            raise ValueError(f"{name} must be an ordinary nn.Linear before TP conversion")
        dimensions = parts or (linear.out_features if kind == "column" else linear.in_features,)
        if any(dim % size for dim in dimensions):
            raise ValueError(f"{name} dimensions {dimensions} are not divisible by tp_size={size}")
    if hasattr(model, "final_layer"):
        norms.append((model.final_layer, "norm_final"))
    for parent, name in norms:
        if not isinstance(getattr(parent, name), nn.LayerNorm):
            raise ValueError(f"{name} must be nn.LayerNorm before TP conversion")
    with torch.no_grad():
        for parent, name in norms:
            setattr(parent, name, SequenceParallelLayerNorm(getattr(parent, name), group))
        for parent, name, kind, parts in plans:
            linear = getattr(parent, name)
            replacement = (ColumnParallelLinear(linear, group) if kind == "column"
                           else RowParallelLinear(linear, group, parts))
            setattr(parent, name, replacement)
        for block in blocks:
            block.tensor_parallel_size = size
            block.tensor_parallel_group = group
            block.heads_num //= size
            if hasattr(block, "mlp_hidden_dim"):
                block.mlp_hidden_dim //= size
    for module in getattr(model, "action_modules", []):
        module.tensor_parallel_size = size
        module.tensor_parallel_rank = rank
    model.tensor_parallel_size = size
    model.tensor_parallel_rank = rank
    model.tensor_parallel_group = group
    return model
