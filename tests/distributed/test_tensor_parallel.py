"""CPU/Gloo numerical checks for DiT tensor parallelism.

The pytest entry point launches the multi-process check below on CPU. Standalone:
    torchrun --standalone --nproc_per_node=2 -m tests.distributed.test_tensor_parallel --tp-size 2
"""
from __future__ import annotations

import argparse
import copy
import datetime
import os
import subprocess
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
import torch.distributed as dist

from hymm.distributed.parallel_states import (
    ParallelDims, nccl_info, initialize_sequence_parallel_state_v2,
    get_model_parallel_info, get_sequence_parallel_state,
    get_tensor_parallel_state, set_sequence_parallel_state,
)
from hymm.distributed.tensor_parallel import validate_parallel_sizes, validate_tensor_parallel_config


@pytest.mark.parametrize("sp,tp,world", [(0, 1, 8), (1, 0, 8), (1, 3, 8), (True, 1, 8), (1, 1.5, 8)])
def test_invalid_topology(sp, tp, world):
    with pytest.raises(ValueError):
        validate_parallel_sizes(sp, tp, world)


def test_topology_and_incompatible_options():
    assert validate_parallel_sizes(1, 8, 8) == 8
    assert validate_parallel_sizes(2, 2, 8) == 4
    for parallelism, inference in [
        (dict(sp_size=2), {}),
        ({}, dict(rollout_type="one_shot")),
        ({}, dict(dit_weight_quantization="int8")),
        ({}, dict(lora_adapter="/adapter")),
        (dict(dit_cpu_offload=True), {}),
        (dict(use_dynamic_ring_attention=True), {}),
    ]:
        args = SimpleNamespace(parallelism=SimpleNamespace(tp_size=8, **parallelism),
                               inference=SimpleNamespace(**inference))
        with pytest.raises(ValueError):
            validate_tensor_parallel_config(args)
    validate_tensor_parallel_config(SimpleNamespace(
        parallelism=SimpleNamespace(tp_size=8, sp_size=1),
        inference=SimpleNamespace(rollout_type="seg_ar", dit_weight_quantization="none")))
    validate_tensor_parallel_config(SimpleNamespace(parallelism=SimpleNamespace(tp_size=1, sp_size=4)))


@pytest.mark.parametrize("tp", [2, 4])
def test_tensor_parallel_matches_single_process_on_cpu(tp):
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(root))
    result = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={tp}",
         "-m", "tests.distributed.test_tensor_parallel", "--tp-size", str(tp)],
        cwd=root, env=env, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    assert "PASS" in result.stdout


@contextmanager
def communication_counts():
    """Count real collectives: one all-gather and one all-to-all per stream and block."""
    calls = Counter()
    names = ("all_reduce", "all_gather_into_tensor", "reduce_scatter_tensor", "all_to_all_single")
    originals = {name: getattr(dist, name) for name in names}

    def counted(name):
        def wrapper(*args, **kwargs):
            calls[name] += 1
            return originals[name](*args, **kwargs)
        return wrapper
    patches = [patch.object(dist, name, counted(name)) for name in names]
    for item in patches:
        item.start()
    try:
        yield calls
    finally:
        for item in patches:
            item.stop()


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
    # comparison actually exercises attention/MLP and cross-rank exchanges.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "mod" in name or "final_layer" in name:
                param.normal_(0, 0.08)
    return model


def check_transformer_cache(group):
    from hymm.models.dit.modules.tensor_parallel import apply_dit_tensor_parallel
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    torch.manual_seed(80)
    reference = small_transformer()
    actual = copy.deepcopy(reference)
    assert apply_dit_tensor_parallel(actual, group) == (rank, size)
    assert actual.hidden_size == reference.hidden_size and actual.heads_num == reference.heads_num
    assert actual.double_blocks[0].img_attn_q.weight.shape == (64 // size, 64)
    # Output projection and MLP keep whole weights and run on the local tokens.
    assert actual.double_blocks[0].img_mlp.fc1.weight.shape == (128, 64)

    def caches(heads):
        return [{"k": torch.zeros(1, 64, heads, 8), "v": torch.zeros(1, 64, heads, 8)} for _ in range(2)]
    reference_cache = caches(8)
    local_heads = 8 // size
    actual_cache = caches(local_heads)
    ref_masks = [torch.zeros(1, 64, dtype=torch.bool) for _ in range(2)]
    actual_masks = [mask.clone() for mask in ref_masks]
    valid = 0
    frame_offset = 0
    # Ragged shapes: image/text lengths that are not multiples of TP, and shorter than TP.
    for frames, width, text_length in ((3, 3, 3), (1, 3, 5), (1, 1, 1)):
        latent = torch.randn(1, 4, frames, 1, width)
        text = torch.randn(1, text_length, 12)
        mask = torch.ones(1, text_length, dtype=torch.bool)
        if text_length > 1:
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
                with communication_counts() as calls:
                    got, added = actual.forward_with_cache(
                        **kwargs, kv_cache_list=actual_cache, cache_mask_list=actual_masks,
                        update_cache=update)
            torch.testing.assert_close(got, expected, rtol=5e-5, atol=3e-6)
            assert added == ref_added == (frames * width + text_length if update else 0)
            assert calls["all_reduce"] == calls["reduce_scatter_tensor"] == 0
            assert calls["all_to_all_single"] == 4  # 2 streams x 2 blocks
            # 2 streams x (2 blocks - first block already replicated) + final gather of 2 streams
            assert calls["all_gather_into_tensor"] == 4
        valid += added
        frame_offset += frames
        start = rank * local_heads
        for local, full, local_mask, full_mask in zip(actual_cache, reference_cache, actual_masks, ref_masks):
            for key in ("k", "v"):
                torch.testing.assert_close(local[key][:, :valid],
                                           full[key][:, :valid, start:start + local_heads],
                                           rtol=5e-5, atol=3e-6)
            torch.testing.assert_close(local_mask, full_mask)

    from hymm.pipeline.causal_ar import CPTCausalInferencePipeline
    pipeline = CPTCausalInferencePipeline.__new__(CPTCausalInferencePipeline)
    nn.Module.__init__(pipeline)
    pipeline.hidden_size, pipeline.heads_num, pipeline.num_double_blocks = 64, 8, 2
    pipeline.args = SimpleNamespace(inference=SimpleNamespace(), vae=SimpleNamespace())
    pipeline.int8_kv_storage = False
    pipeline._initialize_kv_cache(1, 64, torch.float32, torch.device("cpu"))
    assert pipeline.kv_cache[0]["k"].shape == (1, 64, local_heads, 8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-size", type=int, default=2)
    opts = parser.parse_args()
    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=120))
    try:
        validate_parallel_sizes(1, opts.tp_size, dist.get_world_size())
        dims = ParallelDims(1, -1, 1, opts.tp_size, 1, dist.get_world_size())
        initialize_sequence_parallel_state_v2(dims, dims.build_mesh("cpu"))
        group, group_size, local_rank = get_model_parallel_info()
        assert group_size == opts.tp_size
        assert local_rank == dist.get_rank() % group_size
        assert get_tensor_parallel_state() == (opts.tp_size > 1)
        check_transformer_cache(nccl_info.tp_group)
        # Reinitialization resets stale TP state even on a DP-only mesh.
        initialize_sequence_parallel_state_v2(
            ParallelDims(1, dist.get_world_size(), 1, 1, 1, dist.get_world_size()), None)
        assert not get_sequence_parallel_state() and not get_tensor_parallel_state()
        assert get_model_parallel_info() == (None, 1, 0)
        if dist.get_rank() == 0:
            print(f"PASS CPU/Gloo TP={opts.tp_size}: head-sharded QKV/cache, all-to-all, "
                  "ragged DiT/cache, DP isolation", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
