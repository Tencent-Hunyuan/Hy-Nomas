"""CPU checks for per-request streaming and cross-request cache lifetime."""

import ast
import inspect
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from hymm.models.vae.pixelumm_decoder.autoencoder import PixelUMMDCAELatentDecoder


ROOT = Path(__file__).resolve().parents[2]


def _load_cpu_pipeline():
    # Execute production class/method definitions without importing CUDA-only DiT
    # dependencies. These tests exercise only decode setup and cache lifecycle.
    import math
    import os
    import time
    from typing import Callable, List, Optional
    from loguru import logger
    from omegaconf import ListConfig
    from hymm.distributed.parallel_states import nccl_info, get_model_parallel_info
    from hymm.utils.inference_metrics import stage

    source = (ROOT / "hymm/pipeline/causal_ar.py").read_text()
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    namespace = dict(torch=torch, math=math, os=os, time=time, logger=logger,
                     Callable=Callable, List=List, Optional=Optional, ListConfig=ListConfig,
                     BasePipeline=nn.Module, nccl_info=nccl_info, stage=stage,
                     get_model_parallel_info=get_model_parallel_info)
    exec(compile(tree, "causal_ar.py", "exec"), namespace)
    return SimpleNamespace(**namespace)


causal_ar = _load_cpu_pipeline()
CPTCausalInferencePipeline = causal_ar.CPTCausalInferencePipeline
_resolve_pixelumm_stable_prefix_streaming = causal_ar._resolve_pixelumm_stable_prefix_streaming


@pytest.mark.parametrize(
    "vae_name, segment_decode, requested, expected",
    [
        ("pixelumm", False, False, False),
        ("pixelumm", True, True, True),
        ("dcae-v-f32t4c32", False, False, False),
        ("dcae-v-f32t4c32", True, False, False),
    ],
)
def test_streaming_policy_is_independent_of_kv_preservation(
    vae_name, segment_decode, requested, expected,
):
    assert list(inspect.signature(_resolve_pixelumm_stable_prefix_streaming).parameters) == [
        "requested", "segment_vae_decode", "vae_name",
    ]
    assert _resolve_pixelumm_stable_prefix_streaming(
        requested=requested, segment_vae_decode=segment_decode, vae_name=vae_name,
    ) is expected


@pytest.mark.parametrize(
    "vae_name, segment_decode, requested, message",
    [
        ("pixelumm", True, False, r"model\.vae\.stable_prefix_streaming=true"),
        ("pixelumm", False, True, r"inference\.seg_ar_segment_vae_decode=true"),
        ("dcae-v-f32t4c32", True, True, r"model\.vae\.name=pixelumm"),
    ],
)
def test_invalid_streaming_combinations_fail(
    vae_name, segment_decode, requested, message,
):
    with pytest.raises(ValueError, match=message):
        _resolve_pixelumm_stable_prefix_streaming(
            requested=requested, segment_vae_decode=segment_decode,
            vae_name=vae_name,
        )


def _process_ast():
    tree = ast.parse((ROOT / "hymm/pipeline/causal_ar.py").read_text())
    pipeline = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                    and node.name == "CPTCausalInferencePipeline")
    return next(node for node in pipeline.body if isinstance(node, ast.FunctionDef)
                and node.name == "_process_single_prompt")


def _load_setup():
    # Execute the production setup before noise/DiT work, without GPU inference.
    process = deepcopy(_process_ast())
    process.decorator_list = []
    seed_index = next(i for i, node in enumerate(process.body)
                      if isinstance(node, ast.If)
                      and ast.unparse(node.test) == "seed is not None")
    process.body = process.body[:seed_index] + [
        ast.Return(value=ast.Name(id="pixelumm_stable_streaming", ctx=ast.Load())),
    ]
    namespace = dict(vars(causal_ar))
    exec(compile(ast.fix_missing_locations(
        ast.Module(body=[process], type_ignores=[]),
    ), str(ROOT / "hymm/pipeline/causal_ar.py"), "exec"), namespace)
    return namespace["_process_single_prompt"]


def _load_lifetime_branch():
    # Keep the real retention/freeing statements; omit unrelated KV audit work.
    process = _process_ast()
    branch = deepcopy(next(
        node for node in process.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "preserve_kv_cache"
        and any(isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr == "_reset_interactive_pixelumm_decode_state"
                for statement in node.orelse for child in ast.walk(statement))
    ))
    free_index = next(i for i, node in enumerate(branch.orelse)
                      if isinstance(node, ast.Assign)
                      and ast.unparse(node.targets[0]) == "self.kv_cache")
    branch.orelse = branch.orelse[free_index:]
    namespace = {"logger": Mock()}
    function = ast.parse("def finish(self, preserve_kv_cache, cached_action_tails, "
                         "cached_seg_video_ranges, session_segment_offset, "
                         "num_segments):\n    pass").body[0]
    function.body = [branch]
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=[function], type_ignores=[],
    )), "<cache-lifetime>", "exec"), namespace)
    return namespace["finish"]


class TemporalContextDecoder(nn.Module):
    local_attention_tile = 0

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, latent, xt):
        cond = latent[:, :3] + latent[:, :3].mean(dim=2, keepdim=True)
        cond = cond.repeat_interleave(4, 2).repeat_interleave(32, 3).repeat_interleave(32, 4)
        return self.scale * xt + cond


def _pipeline():
    pipeline = CPTCausalInferencePipeline.__new__(CPTCausalInferencePipeline)
    nn.Module.__init__(pipeline)
    pipeline.args = SimpleNamespace(
        inference=SimpleNamespace(
            rollout_type="seg_ar", seg_ar_segment_vae_decode=True,
        ),
        vae=SimpleNamespace(name="pixelumm", stable_prefix_streaming=True),
    )
    pipeline.transformer = nn.Module()
    pipeline.transformer.sparse_optim_counts = {}
    pipeline.transformer.sparse_optim = {}
    pipeline.device = torch.device("cpu")
    pipeline.vae = PixelUMMDCAELatentDecoder(
        TemporalContextDecoder(), temporal_window_latents=4,
        temporal_overlap_latents=2, decoder_tile_pixels=0,
    )
    pipeline._interactive_pixelumm_decode_session = None
    pipeline._last_pixelumm_streaming_metadata = None
    pipeline.kv_cache = [{"k": torch.zeros(1), "v": torch.zeros(1)}]
    pipeline.cache_mask_list = []
    pipeline.cache_valid_len = 1
    pipeline.segment_token_lengths = [1]
    pipeline.global_frame_counter = 0
    pipeline._interactive_noise_bank = torch.zeros(1)
    pipeline._interactive_noise_generator = torch.Generator()
    pipeline._interactive_noise_seed = 42
    pipeline._interactive_segment_noise_rng_states = []
    return pipeline


@pytest.mark.parametrize("kv_quantization", ["none", "int8"])
def test_flex_with_swa_can_initialize_pixelumm_segment_decode(monkeypatch, kv_quantization):
    from apps.seg_ar_demo_config import load_demo_config
    from hymm.utils.range_attention_config import validate_range_attention_config

    for name in ("HY_DIT_COMPILE_PREPOST", "HY_AR_KV_INT8_STORAGE"):
        monkeypatch.delenv(name, raising=False)
    config = load_demo_config([
        "+experiment=inference",
        "model.dit.seg_ar_attention_backend=flex",
        "parallelism.sp_size=1",
        "model.vae.parallel_mode=none",
        f"inference.kv_cache_quantization={kv_quantization}",
        "inference.sliding_window.enabled=true",
        "inference.seg_ar_segment_vae_decode=true",
        "model.vae.stable_prefix_streaming=true",
    ])
    validate_range_attention_config(config)
    pipeline = _pipeline()
    pipeline.args = config
    assert _load_setup()(
        pipeline, prompts=["first", "second"], num_frames=32, height=32, width=32,
        guidance_scale=1., camera_guidance_scale=1., num_inference_steps=1,
        flow_shift=5., seed=None,
    )
    assert pipeline._interactive_pixelumm_decode_session is not None


@pytest.mark.parametrize("preserve", [False, True])
def test_pipeline_streams_and_retains_state_only_when_requested(preserve):
    pipeline = _pipeline()
    setup = _load_setup()
    finish = _load_lifetime_branch()
    setup_args = dict(
        prompts=["first", "second"], num_frames=32, height=32, width=32,
        guidance_scale=1., camera_guidance_scale=1., num_inference_steps=1,
        flow_shift=5., seed=None, preserve_kv_cache=preserve,
    )
    assert setup(pipeline, **setup_args)
    session = pipeline._interactive_pixelumm_decode_session
    latent = torch.linspace(-1, 1, 3 * 8).reshape(1, 3, 8, 1, 1)
    with torch.inference_mode():
        first = session.push_preview(latent[:, :, :4], pixel_frames=16)
        second = session.push_preview(latent[:, :, 4:], pixel_frames=16)
        assembled = torch.cat(
            [first.video[:, :, :second.pixel_start], second.video], dim=2,
        )
        reference = pipeline.vae.decode(latent, pixel_frames=32).sample
    assert torch.equal(assembled, reference)
    assert second.replace_pixel_frames > 0
    cache = pipeline.kv_cache
    finish(pipeline, preserve, [], [], 0, 2)
    if preserve:
        assert pipeline.kv_cache is cache
        assert pipeline._interactive_pixelumm_decode_session is session
        assert setup(pipeline, **setup_args)
        assert pipeline._interactive_pixelumm_decode_session is session
    else:
        assert pipeline.kv_cache is None
        assert pipeline.cache_valid_len == 0
        assert pipeline._interactive_pixelumm_decode_session is None
        assert pipeline._interactive_noise_bank is None
        assert setup(pipeline, **setup_args)
        new_session = pipeline._interactive_pixelumm_decode_session
        assert new_session is not session
        assert new_session.available_pixel_frames == 0


def test_pipeline_rejects_pixelumm_selector_without_pixelumm_decoder():
    pipeline = _pipeline()
    pipeline.vae = nn.Identity()
    with pytest.raises(RuntimeError, match="did not construct a PixelUMM"):
        _load_setup()(
            pipeline, prompts=["first"], num_frames=16, height=32, width=32,
            guidance_scale=1., camera_guidance_scale=1., num_inference_steps=1,
            flow_shift=5., seed=None,
        )


def test_low_memory_preset_reuses_offloaded_streaming_across_requests(monkeypatch):
    from apps.seg_ar_demo_config import load_demo_config
    from hymm.utils.model_cpu_offload import install_model_cpu_offload
    from hymm.utils.range_attention_config import validate_range_attention_config

    for name in ("HY_DIT_COMPILE_PREPOST", "HY_AR_KV_INT8_STORAGE"):
        monkeypatch.delenv(name, raising=False)
    config = load_demo_config([
        "+experiment=single_gpu_low_mem_inference",
        "inference.seg_ar_segment_vae_decode=true",
        "model.vae.stable_prefix_streaming=true",
    ])
    validate_range_attention_config(config)
    assert config.vae.cpu_offload
    pipeline = _pipeline()
    pipeline.args = config
    controller = install_model_cpu_offload(pipeline.vae.eval().requires_grad_(False), "cuda:0")
    controller.device = torch.device("cpu")
    monkeypatch.setattr(torch.cuda, "synchronize", Mock())
    setup = _load_setup()
    finish = _load_lifetime_branch()
    setup_args = dict(
        prompts=["segment"], num_frames=16, height=32, width=32,
        guidance_scale=1., camera_guidance_scale=1., num_inference_steps=1,
        flow_shift=5., seed=None, preserve_kv_cache=True,
    )
    latent = torch.linspace(-1, 1, 24).reshape(1, 3, 8, 1, 1)
    with torch.inference_mode():
        assert setup(pipeline, **setup_args)
        session = pipeline._interactive_pixelumm_decode_session
        first = session.push_preview(latent[:, :, :4], pixel_frames=16)
        finish(pipeline, True, [], [], 0, 1)
        assert setup(pipeline, **setup_args)
        assert pipeline._interactive_pixelumm_decode_session is session
        second = session.push_preview(latent[:, :, 4:], pixel_frames=16)
        assembled = torch.cat([first.video[:, :, :second.pixel_start], second.video], dim=2)
        reference = pipeline.vae.decode(latent, pixel_frames=32).sample
    assert torch.equal(assembled, reference)
    assert second.replace_pixel_frames > 0
    assert controller.gpu_transfers == controller.cpu_transfers == 3
    assert controller.depth == 0
    finish(pipeline, False, [], [], 0, 2)
    assert pipeline._interactive_pixelumm_decode_session is None


def test_offline_inference_never_implicitly_preserves_kv():
    tree = ast.parse((ROOT / "hymm/inference.py").read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)
             and isinstance(node.func.value, ast.Name)
             and node.func.value.id == "pipeline" and node.func.attr == "inference"]
    assert calls
    assert all(keyword.arg != "preserve_kv_cache"
               for call in calls for keyword in call.keywords)
