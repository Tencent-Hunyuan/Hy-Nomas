"""CPU/Gloo numerical checks for dense DiT tensor parallelism.

Standalone: torchrun --standalone --nproc_per_node=8 -m \
    tests.distributed.test_tensor_parallel --tp-size 8
"""
from __future__ import annotations

import argparse
import copy
import datetime
from collections import Counter
from contextlib import contextmanager
from unittest.mock import patch
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.distributed as dist

from hymm.distributed.parallel_states import (
    ParallelDims, nccl_info, initialize_sequence_parallel_state_v2,
    get_model_parallel_info, get_sequence_parallel_state,
    get_tensor_parallel_state, set_sequence_parallel_state,
)
from hymm.distributed.tensor_parallel import (
    ColumnParallelLinear, RowParallelLinear, parallelize_transformer,
    validate_parallel_sizes, validate_tensor_parallel_config,
    SequenceParallelLayerNorm, scatter_sequence, gather_sequence,
    sequence_partition, reduce_scatter_sequence,
)


@pytest.mark.parametrize("sp,tp,world", [(0, 1, 8), (1, 0, 8), (1, 3, 8), (True, 1, 8), (1, 1.5, 8)])
def test_invalid_topology(sp, tp, world):
    with pytest.raises(ValueError):
        validate_parallel_sizes(sp, tp, world)


def test_topology_and_incompatible_options():
    assert validate_parallel_sizes(1, 8, 8) == 8
    assert validate_parallel_sizes(2, 2, 8) == 4
    for key, value in [("torch_compile", True), ("lora_adapter", "/adapter"),
                       ("dit_weight_quantization", "w8_bf16")]:
        args = SimpleNamespace(parallelism=SimpleNamespace(tp_size=8),
                               inference=SimpleNamespace(**{key: value}))
        with pytest.raises(ValueError):
            validate_tensor_parallel_config(args)
    with pytest.raises(ValueError, match="dynamic ring"):
        validate_tensor_parallel_config(SimpleNamespace(
            parallelism=SimpleNamespace(tp_size=8, use_dynamic_ring_attention=True)))
    validate_tensor_parallel_config(SimpleNamespace(parallelism=SimpleNamespace(tp_size=1)))


@contextmanager
def communication_counts():
    """Observe real collective calls; forbid replacing RS with all-reduce."""
    calls = Counter()
    originals = {name: getattr(dist, name) for name in
                 ("all_reduce", "all_gather_into_tensor", "reduce_scatter_tensor")}
    def counted(name):
        def wrapper(*args, **kwargs):
            calls[name] += 1
            return originals[name](*args, **kwargs)
        return wrapper
    with patch.object(dist, "all_reduce", counted("all_reduce")), \
         patch.object(dist, "all_gather_into_tensor", counted("all_gather_into_tensor")), \
         patch.object(dist, "reduce_scatter_tensor", counted("reduce_scatter_tensor")):
        yield calls


def token_shard(value, size, rank):
    # Independent oracle for the balanced contiguous sequence partition.
    return torch.tensor_split(value, size, dim=1)[rank]


def check_linear_gradients(group):
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    for length in (0, 3, 2 * size, 2 * size + 1):
        torch.manual_seed(31)
        norm = nn.LayerNorm(16, dtype=torch.float64)
        fc1, fc2 = nn.Linear(16, 32, dtype=torch.float64), nn.Linear(32, 16, dtype=torch.float64)
        local_norm = SequenceParallelLayerNorm(copy.deepcopy(norm), group)
        local_norm.sequence_length = length
        column, row = ColumnParallelLinear(fc1, group), RowParallelLinear(fc2, group)
        value = torch.randn(2, length, 16, dtype=torch.float64, requires_grad=True)
        local_value = token_shard(value, size, rank).detach().clone().requires_grad_(True)
        expected = fc2(torch.nn.functional.gelu(fc1(norm(value))))
        with communication_counts() as calls:
            normalized = local_norm(local_value)
            hidden = column(normalized)
            actual = row(torch.nn.functional.gelu(hidden))
        assert normalized.shape == value.shape
        assert hidden.shape == (2, length, 32 // size)
        assert actual.shape == local_value.shape
        assert calls["all_reduce"] == 0
        assert calls["all_gather_into_tensor"] == int(length > 0)
        assert calls["reduce_scatter_tensor"] == int(length > 0)
        torch.testing.assert_close(actual, token_shard(expected, size, rank), rtol=1e-10, atol=1e-10)
        expected.square().sum().backward()
        actual.square().sum().backward()
        torch.testing.assert_close(local_value.grad, token_shard(value.grad, size, rank), rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(column.weight.grad, fc1.weight.grad.chunk(size, 0)[rank])
        torch.testing.assert_close(row.weight.grad, fc2.weight.grad.chunk(size, 1)[rank])
        # Replicated affine parameters see disjoint token losses; training
        # would aggregate these gradients separately, outside forward.
        for local_grad, reference_grad in ((row.bias.grad, fc2.bias.grad),
                                           (local_norm.weight.grad, norm.weight.grad),
                                           (local_norm.bias.grad, norm.bias.grad)):
            dist.all_reduce(local_grad, group=group)
            torch.testing.assert_close(local_grad, reference_grad, rtol=1e-10, atol=1e-10)
        with torch.inference_mode():
            shard = scatter_sequence(value.detach(), group)
            gathered = gather_sequence(shard, group, length)
            torch.testing.assert_close(gathered, value.detach())
            # Local residual storage must not retain a view of the full tensor.
            assert shard.untyped_storage().nbytes() == shard.numel() * shard.element_size()
        if length:
            bf16_norm = SequenceParallelLayerNorm(copy.deepcopy(norm).to(torch.bfloat16), group)
            bf16_norm.sequence_length = length
            bf16_col = ColumnParallelLinear(copy.deepcopy(fc1).to(torch.bfloat16), group)
            bf16_row = RowParallelLinear(copy.deepcopy(fc2).to(torch.bfloat16), group)
            with torch.inference_mode():
                got = bf16_row(torch.nn.functional.gelu(bf16_col(bf16_norm(local_value.detach().bfloat16()))))
                assert got.dtype == torch.bfloat16
                torch.testing.assert_close(got.float(), token_shard(expected.detach().float(), size, rank),
                                           rtol=0.06, atol=0.02)


def check_fused_single_stream(group):
    from hymm.models.dit.nomas_cpt import MMSingleStreamBlock
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    torch.manual_seed(44)
    block = MMSingleStreamBlock(64, 8, mlp_width_ratio=2, attn_mode="torch", dtype=torch.float32).eval()
    container = nn.Module()
    container.double_blocks = nn.ModuleList()
    container.single_blocks = nn.ModuleList([copy.deepcopy(block)])
    parallelize_transformer(container, group)
    local = container.single_blocks[0]
    x = torch.randn(2, 5, 64)
    attn = torch.randn(2, 5, 64)
    with torch.no_grad(), communication_counts() as calls:
        expected = block.linear2(attn, block.mlp_act(block.linear1_mlp(block.pre_norm(x))))
        normalized = local.pre_norm(token_shard(x, size, rank))
        actual = local.linear2(attn.chunk(size, dim=-1)[rank], local.mlp_act(local.linear1_mlp(normalized)))
        torch.testing.assert_close(actual, token_shard(expected, size, rank), rtol=2e-5, atol=2e-6)
        assert calls["all_reduce"] == 0 and calls["reduce_scatter_tensor"] == 1
    assert local.heads_num == 8 // size
    assert local.linear2.fc.weight.shape == (64, 192 // size)


def small_transformer():
    from hymm.models.dit.nomas_cpt import CPTHYNomasDiffusionTransformer
    model = CPTHYNomasDiffusionTransformer(
        patch_size=[1, 1, 1], in_channels=4, out_channels=4,
        hidden_size=64, heads_num=8, mlp_width_ratio=2,
        mm_double_blocks_depth=2, mm_single_blocks_depth=0,
        rope_dim_list=[2, 2, 4], text_states_dim=12,
        text_projection="linear", dtype=torch.float32, attn_mode="torch",
    ).eval().requires_grad_(False)
    # Default modulation and final weights are zero: randomize them so this
    # comparison actually exercises attention/MLP and cross-rank reductions.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "mod" in name or "final_layer" in name:
                param.normal_(0, 0.08)
    return model


def record_layouts(model):
    layouts, handles = {}, []
    for name, module in model.named_modules():
        if isinstance(module, (SequenceParallelLayerNorm, ColumnParallelLinear, RowParallelLinear)):
            def capture(module, inputs, output, name=name):
                layouts[name] = (tuple(inputs[0].shape), tuple(output.shape))
            handles.append(module.register_forward_hook(capture))
    return layouts, handles


def assert_layouts(layouts, img_length, txt_length, size, rank, blocks=2):
    for i in range(blocks):
        for stream, full in (("img", img_length), ("txt", txt_length)):
            local = full // size + int(rank < full % size)
            prefix = f"double_blocks.{i}.{stream}_"
            for name in ("norm1", "norm2"):
                assert layouts[prefix + name] == ((1, local, 64), (1, full, 64))
            for name in ("q", "k", "v"):
                assert layouts[prefix + "attn_" + name] == ((1, full, 64), (1, full, 64 // size))
            assert layouts[prefix + "attn_proj"] == ((1, full, 64 // size), (1, local, 64))
            assert layouts[prefix + "mlp.fc1"] == ((1, full, 64), (1, full, 128 // size))
            assert layouts[prefix + "mlp.fc2"] == ((1, full, 128 // size), (1, local, 64))
    local_img = img_length // size + int(rank < img_length % size)
    assert layouts["final_layer.norm_final"] == ((1, local_img, 64), (1, img_length, 64))


def check_transformer_cache(group):
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    torch.manual_seed(80 + dist.get_rank() // (nccl_info.sp_size * nccl_info.tp_size))
    reference = small_transformer()
    actual = copy.deepcopy(reference)
    parallelize_transformer(actual, group, nccl_info.sp_size)
    assert actual.hidden_size == reference.hidden_size and actual.heads_num == reference.heads_num
    assert actual.double_blocks[0].img_attn_q.weight.shape == (64 // size, 64)
    assert actual.double_blocks[0].img_mlp.fc1.weight.shape == (128 // size, 64)
    layouts, handles = record_layouts(actual)

    def caches(heads):
        return [{"k": torch.zeros(1, 64, heads, 8), "v": torch.zeros(1, 64, heads, 8)} for _ in range(2)]
    reference_cache = caches(8)
    local_heads = 8 // (size * nccl_info.sp_size)
    actual_cache = caches(local_heads)
    ref_masks = [torch.zeros(1, 64, dtype=torch.bool) for _ in range(2)]
    actual_masks = [mask.clone() for mask in ref_masks]
    valid = 0
    frame_offset = 0
    # Different, ragged shapes ensure norm metadata is refreshed, including
    # image/text lengths smaller than TP and cached action-tail bookkeeping.
    for segment, (frames, width, text_length) in enumerate(((3, 3, 3), (1, 3, 5))):
        latent = torch.randn(1, 4, frames, 1, width)
        text = torch.randn(1, text_length, 12)
        mask = torch.ones(1, text_length, dtype=torch.bool)
        mask[:, -1] = False
        kwargs = dict(hidden_states=latent, timestep=torch.tensor([350.]),
                      text_states=text, text_mask=mask, cache_valid_len=valid,
                      global_frame_counter=frame_offset, attn_mode_override="torch",
                      guidance=torch.tensor([1.]), current_action_start=1)
        for update in (False, True):
            set_sequence_parallel_state(False)
            with torch.inference_mode():
                expected, ref_added = reference.forward_with_cache(
                    **kwargs, kv_cache_list=reference_cache, cache_mask_list=ref_masks,
                    update_cache=update)
            set_sequence_parallel_state(nccl_info.sp_size > 1)
            with torch.inference_mode(), communication_counts() as calls:
                got, added = actual.forward_with_cache(
                    **kwargs, kv_cache_list=actual_cache, cache_mask_list=actual_masks,
                    update_cache=update)
            torch.testing.assert_close(got, expected, rtol=5e-5, atol=3e-6)
            assert added == ref_added == (frames * width + text_length if update else 0)
            assert calls["all_reduce"] == 0
            assert calls["reduce_scatter_tensor"] == 8  # 4 row projections x 2 blocks
            assert calls["all_gather_into_tensor"] >= 9  # 4 norms x 2 + final norm
            sp_img_length = torch.chunk(latent.flatten(2), nccl_info.sp_size, dim=2)[nccl_info.rank_within_spgroup].shape[2]
            assert_layouts(layouts, sp_img_length, text_length, size, rank)
        valid += added
        frame_offset += frames
        start = rank * (8 // size) + nccl_info.rank_within_spgroup * local_heads
        for local, full, local_mask, full_mask in zip(actual_cache, reference_cache, actual_masks, ref_masks):
            for key in ("k", "v"):
                torch.testing.assert_close(local[key][:, :valid],
                                           full[key][:, :valid, start:start + local_heads],
                                           rtol=5e-5, atol=3e-6)
            torch.testing.assert_close(local_mask, full_mask)
            assert local_mask._action_tail_ranges == full_mask._action_tail_ranges
    for handle in handles:
        handle.remove()
    from hymm.pipeline.causal_ar import CPTCausalInferencePipeline
    pipeline = CPTCausalInferencePipeline.__new__(CPTCausalInferencePipeline)
    nn.Module.__init__(pipeline)
    pipeline.hidden_size, pipeline.heads_num, pipeline.num_double_blocks = 64, 8, 2
    pipeline.args = SimpleNamespace(inference=SimpleNamespace(), vae=SimpleNamespace())
    pipeline.int8_kv_storage = False
    pipeline._initialize_kv_cache(1, 64, torch.float32, torch.device("cpu"))
    assert pipeline.kv_cache[0]["k"].shape == (1, 64, local_heads, 8)



def cpu_flash_attention(qkv, mask, **kwargs):
    """Test adapter for the CUDA flash kernel; all surrounding code is real."""
    q, k, v = qkv.unbind(2)
    result = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        attn_mask=mask[:, None, None, :], dropout_p=0.)
    return result.transpose(1, 2)


def check_noncache_forward(group):
    from hymm.models.dit.modules import attention
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    torch.manual_seed(191)
    reference = small_transformer()
    actual = copy.deepcopy(reference)
    parallelize_transformer(actual, group, nccl_info.sp_size)
    layouts, handles = record_layouts(actual)
    for per_token in (False, True):
        latent = torch.randn(1, 4, 3, 1, 3)
        text = torch.randn(1, 2, 3, 12)
        mask = torch.ones(1, 2, 3, dtype=torch.bool)
        mask[:, :, -1] = False
        kwargs = dict(hidden_states=latent, text_states=text,
                      timestep=torch.tensor([[200., 400., 500.]]) if per_token else torch.tensor([200.]),
                      encoder_attention_mask=mask, guidance=torch.tensor([1.]),
                      extra_kwargs=dict(prompt_video_boundaries=[0, 1, 3], chunk_size=0,
                                        attn_mode_override="flash3"))
        with patch.object(attention, "flash_attn_no_pad_v3", cpu_flash_attention):
            set_sequence_parallel_state(False)
            with torch.inference_mode():
                expected = reference(**kwargs)[0]
            set_sequence_parallel_state(nccl_info.sp_size > 1)
            with torch.inference_mode(), communication_counts() as calls:
                got = actual(**kwargs)[0]
            torch.testing.assert_close(got, expected, rtol=5e-5, atol=3e-6)
            assert calls["all_reduce"] == 0 and calls["reduce_scatter_tensor"] == 8
            img_length = torch.chunk(latent.flatten(2), nccl_info.sp_size, dim=2)[nccl_info.rank_within_spgroup].shape[2]
            assert_layouts(layouts, img_length, 6, size, rank)
    for handle in handles:
        handle.remove()


def check_single_stream_forward(group):
    if nccl_info.sp_size > 1:
        return  # The current public model uses only double-stream blocks.
    from hymm.models.dit.nomas_cpt import MMSingleStreamBlock
    from hymm.models.dit.modules import attention
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    torch.manual_seed(205)
    reference = MMSingleStreamBlock(64, 8, mlp_width_ratio=2, attn_mode="flash3").eval()
    with torch.no_grad():
        reference.modulation.linear.weight.normal_(0, 0.03)
        reference.modulation.linear.bias.normal_(0, 0.03)
    model = nn.Module()
    model.double_blocks = nn.ModuleList()
    model.single_blocks = nn.ModuleList([copy.deepcopy(reference)])
    parallelize_transformer(model, group)
    value, vec = torch.randn(1, 11, 64), torch.randn(1, 64)
    kwargs = dict(vec=vec, txt_len=3, freqs_cis=(torch.ones(8, 8), torch.zeros(8, 8)),
                  text_mask=torch.tensor([[True, True, False]]))
    with patch.object(attention, "flash_attn_no_pad_v3", cpu_flash_attention), torch.inference_mode():
        expected = reference(value, **kwargs)
        with communication_counts() as calls:
            got = model.single_blocks[0](scatter_sequence(value, group), **kwargs)
        torch.testing.assert_close(got, token_shard(expected, size, rank), rtol=3e-5, atol=3e-6)
        assert calls["all_reduce"] == 0 and calls["reduce_scatter_tensor"] == 1


def cpu_camera_flash(q, k, v, **kwargs):
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=0.).transpose(1, 2)


def cpu_range_attention(self, *, q, k, v, q_ranges, k_ranges, **kwargs):
    output = torch.zeros_like(q)
    for (start, end), (ks, ke) in zip(q_ranges.tolist(), k_ranges.tolist()):
        if start == end:
            continue
        output[start:end] = cpu_camera_flash(
            q[None, start:end], k[None, ks:ke], v[None, ks:ke])[0]
    return output, None


def check_camera_offsets(group):
    from hymm.models.camera import action_module
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    for chunked in (False, True):
        torch.manual_seed(251)
        reference = action_module.ActionModule(
            keyboard_dim_in=8, hidden_size=16, img_hidden_size=64,
            keyboard_hidden_dim=32, heads_num=4, rope_dim_list=[2, 2, 4],
            per_chunk_camera_attn=chunked, camera_chunk_size=1,
            seg_ar_attention_backend="flex",
        ).eval()
        actual = copy.deepcopy(reference)
        actual.tensor_parallel_size, actual.tensor_parallel_rank = size, rank
        for frames, width in ((3, 3), (1, 1)):
            total = frames * width
            value, keyboard = torch.randn(1, total, 64), torch.randn(1, frames, 8)
            if nccl_info.sp_size > 1:
                # Match torch.chunk but allow empty trailing test shards.
                step = (total + nccl_info.sp_size - 1) // nccl_info.sp_size
                start = min(total, nccl_info.rank_within_spgroup * step)
                sp_value = value[:, start:min(total, start + step)]
            else:
                start, sp_value = 0, value
            local = token_shard(sp_value, size, rank)
            expected_start = start + sum(part.shape[1] for part in torch.tensor_split(sp_value, size, dim=1)[:rank])
            kwargs = dict(tt=frames, th=1, tw=width, keyboard=keyboard, rope_time_offset=2,
                          has_camera_mask=torch.tensor([True]))
            with patch.object(action_module.ActionModule, "_range_kernel", cpu_range_attention), torch.inference_mode():
                set_sequence_parallel_state(False)
                expected = reference(value, **kwargs)
                set_sequence_parallel_state(nccl_info.sp_size > 1)
                assert actual._sp_local_window(total, local.shape[1]) == (expected_start, expected_start + local.shape[1])
                got = actual(local, **kwargs)
                torch.testing.assert_close(got, expected[:, expected_start:expected_start + local.shape[1]],
                                           rtol=4e-5, atol=3e-6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--sp-size", type=int, default=1)
    opts = parser.parse_args()
    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=120))
    try:
        validate_parallel_sizes(opts.sp_size, opts.tp_size, dist.get_world_size())
        dims = ParallelDims(1, -1, opts.sp_size, opts.tp_size, 1, dist.get_world_size())
        mesh = dims.build_mesh("cpu")
        initialize_sequence_parallel_state_v2(dims, mesh)
        group, group_size, local_rank = get_model_parallel_info()
        assert group_size == opts.sp_size * opts.tp_size
        assert local_rank == dist.get_rank() % group_size
        assert dist.get_process_group_ranks(group) == list(range(
            dist.get_rank() // group_size * group_size,
            (dist.get_rank() // group_size + 1) * group_size))
        assert get_tensor_parallel_state() == (opts.tp_size > 1)
        check_linear_gradients(nccl_info.tp_group)
        check_fused_single_stream(nccl_info.tp_group)
        check_transformer_cache(nccl_info.tp_group)
        check_noncache_forward(nccl_info.tp_group)
        check_single_stream_forward(nccl_info.tp_group)
        check_camera_offsets(nccl_info.tp_group)
        # Reinitialization resets stale TP/SP state even on a DP-only mesh.
        initialize_sequence_parallel_state_v2(ParallelDims(1, dist.get_world_size(), 1, 1, 1, dist.get_world_size()), None)
        assert not get_sequence_parallel_state() and not get_tensor_parallel_state()
        assert get_model_parallel_info() == (None, 1, 0)
        if dist.get_rank() == 0:
            print(f"PASS CPU/Gloo TP={opts.tp_size} SP={opts.sp_size}: sequence LayerNorm/AG, head-sharded QKV, row RS, ragged DiT/cache, token modulation, camera offsets, DP isolation", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
