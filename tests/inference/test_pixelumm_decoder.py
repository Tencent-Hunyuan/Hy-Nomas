"""CPU tests for the inference-only PixelUMM latent decoder backend."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from hymm.models.vae.pixelumm_decoder.autoencoder import PixelUMMDCAELatentDecoder
from hymm.models.vae.pixelumm_decoder.model import DCAEVPixelUMMDecoder, GenExpertArch

TINY = GenExpertArch(
    hidden_size=64, intermediate_size=128, num_layers=2, num_heads=4, num_kv_heads=2, head_dim=16
)
# 3 latent frames would be shorter than one window; 7 gives windows [0, 2, 3] at window 4 / overlap 2.
LATENT_SHAPE = (1, 32, 7, 3, 5)
TILING = dict(
    temporal_window_latents=4,
    temporal_overlap_latents=2,
    decoder_tile_pixels=64,
    decoder_tile_overlap_pixels=32,
)


# nospatile geometry: whole-frame calls + sliding-tile attention. A [*,3,5] latent is a 6 x 10
# token plane; 2-token (32px) tiles give 3 x 5 tiles, so the FlexAttention path is taken.
NOSPATILE = dict(
    temporal_window_latents=4,
    temporal_overlap_latents=2,
    decoder_tile_pixels=0,
    decoder_tile_overlap_pixels=0,
)
LOCAL_TILE_TOKENS = 2


def _tiny_decoder(local_attention_tile: int = 0) -> DCAEVPixelUMMDecoder:
    torch.manual_seed(1234)
    decoder = DCAEVPixelUMMDecoder(TINY, local_attention_tile=local_attention_tile)
    with torch.no_grad():
        for parameter in decoder.parameters():
            parameter.normal_(0.0, 0.05)
    return decoder.eval()


def _latent() -> torch.Tensor:
    return torch.randn(LATENT_SHAPE, generator=torch.Generator().manual_seed(7))


class _LocalDecoder(nn.Module):
    """Position-independent pointwise decoder: tiled and untiled decodes must agree."""

    local_attention_tile = 0  # the real decoder always exposes this; 0 = full attention

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, latent: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
        cond = latent[:, :3].repeat_interleave(4, 2).repeat_interleave(32, 3).repeat_interleave(32, 4)
        return self.scale * xt + cond


def test_checkpoint_roundtrip_reads_decoder_tensors_only():
    decoder = _tiny_decoder()
    state = {f"decoder.{k}": v.clone() for k, v in decoder.state_dict().items()}
    state["encoder.stages.0.weight"] = torch.zeros(3)
    from safetensors.torch import save_file

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "diffusion_pytorch_model.safetensors")
        save_file(state, path)
        loaded = PixelUMMDCAELatentDecoder.from_checkpoint(
            path, dtype=torch.float32, device="cpu", arch=TINY, num_steps=1
        )
    expected = decoder.state_dict()
    actual = loaded.decoder.state_dict()
    assert set(actual) == set(expected)
    assert all(torch.equal(actual[key], expected[key]) for key in expected)
    assert not any(p.requires_grad for p in loaded.parameters())
    latent = _latent()[:, :, :2, :2, :2]
    xt = torch.randn(DCAEVPixelUMMDecoder.pixel_shape(latent))
    with torch.no_grad():
        # Equal weights; CPU kernels may differ by an ULP on mmap-backed buffers.
        torch.testing.assert_close(loaded.decoder(latent, xt), decoder(latent, xt), rtol=0, atol=1e-6)


def test_single_window_single_tile_is_one_decoder_call_on_seeded_noise():
    wrapper = PixelUMMDCAELatentDecoder(_tiny_decoder(), num_steps=1, seed=3).eval()
    latent = _latent()[:, :, :3, :2, :2]
    with torch.no_grad():
        video = wrapper.decode(latent, return_dict=False, pixel_frames=10)[0]
        reference = wrapper.decoder(latent, wrapper._make_noise(latent, 0, 3))
    assert torch.equal(video, reference[:, :, :10])


def test_noise_is_consistent_across_overlapping_slices():
    wrapper = PixelUMMDCAELatentDecoder(_LocalDecoder(), seed=5)
    latent = _latent()
    full = wrapper._make_noise(latent, 0, 7)
    assert torch.equal(wrapper._make_noise(latent, 2, 6), full[:, :, 8:24])


@pytest.mark.parametrize("num_steps", [1, 3])
def test_tiled_windowed_decode_matches_untiled_reference(num_steps):
    tiled = PixelUMMDCAELatentDecoder(_LocalDecoder(), num_steps=num_steps, seed=11, **TILING).eval()
    latent = _latent()
    with torch.no_grad():
        video = tiled.decode(latent, return_dict=False)[0]
        # Source sample_distilled, untiled.
        x = tiled._make_noise(latent, 0, latent.shape[2])
        steps = tiled.denoising_step_list[:num_steps]
        for t_cur, t_next in zip(steps, (*steps[1:], 0.0)):
            x0 = tiled.decoder(latent, x).float()
            x = x0 + (t_next / t_cur) * (x - x0)
    assert video.shape == x.shape
    torch.testing.assert_close(video, x, rtol=1e-5, atol=1e-5)


def test_invalid_schedule_and_steps_are_rejected():
    with pytest.raises(ValueError):
        PixelUMMDCAELatentDecoder(_LocalDecoder(), num_steps=5)
    with pytest.raises(ValueError):
        PixelUMMDCAELatentDecoder(_LocalDecoder(), denoising_step_list=[0.5, 0.9])


def _expected_local_mask(grid, tile):
    """Independent statement of the sliding-tile rule, in the original THW token order."""
    frames, height, width = grid
    tiles_h, tiles_w = -(-height // tile), -(-width // tile)
    t, h, w = torch.meshgrid(*(torch.arange(n) for n in grid), indexing="ij")
    th, tw = (h // tile).flatten(), (w // tile).flatten()
    ch, cw = th.clamp(1, max(tiles_h - 2, 1)), tw.clamp(1, max(tiles_w - 2, 1))
    return ((ch[:, None] - th[None]).abs() <= 1) & ((cw[:, None] - tw[None]).abs() <= 1)


def test_local_attention_layout_and_flex_match_dense_masked_attention():
    from hymm.models.vae.pixelumm_decoder.model import _flex_attention, local_attention_layout

    grid = (3, 6, 10)
    perm, inverse, block_mask = local_attention_layout(grid, LOCAL_TILE_TOKENS, "cpu")
    assert torch.equal(perm[inverse], torch.arange(perm.numel()))
    expected = _expected_local_mask(grid, LOCAL_TILE_TOKENS)
    q_idx, kv_idx = torch.arange(perm.numel())[:, None], torch.arange(perm.numel())[None]
    permuted = block_mask.mask_mod(0, 0, q_idx, kv_idx)
    assert torch.equal(permuted[inverse][:, inverse], expected)
    assert not expected.all()  # the local path really restricts attention

    generator = torch.Generator().manual_seed(0)
    q = torch.randn(1, 4, perm.numel(), 16, generator=generator)
    k = torch.randn(1, 2, perm.numel(), 16, generator=generator)
    v = torch.randn(1, 2, perm.numel(), 16, generator=generator)
    out = _flex_attention(q[:, :, perm], k[:, :, perm], v[:, :, perm], block_mask)[:, :, inverse]
    reference = torch.nn.functional.scaled_dot_product_attention(
        q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1), attn_mask=expected
    )
    torch.testing.assert_close(out, reference, rtol=1e-5, atol=1e-5)
    # <= 3 x 3 tiles: plain full attention, no layout.
    assert local_attention_layout((3, 6, 6), LOCAL_TILE_TOKENS, "cpu") is None


def test_local_attention_changes_the_forward_and_loads_from_pixels():
    full, local = _tiny_decoder(), _tiny_decoder(LOCAL_TILE_TOKENS)
    latent = _latent()[:, :, :2]
    xt = torch.randn(DCAEVPixelUMMDecoder.pixel_shape(latent), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        assert not torch.allclose(full(latent, xt), local(latent, xt))
    from safetensors.torch import save_file

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "diffusion_pytorch_model.safetensors")
        save_file({f"decoder.{k}": v.clone() for k, v in local.state_dict().items()}, path)
        loaded = PixelUMMDCAELatentDecoder.from_checkpoint(
            path, dtype=torch.float32, device="cpu", arch=TINY,
            local_attention_tile_pixels=LOCAL_TILE_TOKENS * 16,
        )
        with pytest.raises(ValueError):
            PixelUMMDCAELatentDecoder.from_checkpoint(path, arch=TINY, local_attention_tile_pixels=24)
    assert loaded.decoder.local_attention_tile == LOCAL_TILE_TOKENS
    with torch.no_grad():
        torch.testing.assert_close(loaded.decoder(latent, xt), local(latent, xt), rtol=0, atol=1e-6)


@pytest.mark.parametrize("vae_name", [None, "dcae-v-f32t4c32"])
def test_loader_returns_none_for_other_vaes(vae_name):
    from hymm.models.vae.pixelumm_decoder import maybe_load_pixelumm_decoder

    args = SimpleNamespace(vae=SimpleNamespace(name=vae_name))
    assert maybe_load_pixelumm_decoder(args, device="cpu", logger=None) is None


class _WindowContextDecoder(nn.Module):
    """Output depends on the whole input window (mean over time), like attention would."""

    local_attention_tile = 0

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, latent: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
        context = latent[:, :3].mean(dim=2, keepdim=True)
        cond = (latent[:, :3] + context).repeat_interleave(4, 2).repeat_interleave(32, 3).repeat_interleave(32, 4)
        return self.scale * xt + cond


def _streaming_wrapper(**kwargs) -> PixelUMMDCAELatentDecoder:
    return PixelUMMDCAELatentDecoder(_WindowContextDecoder(), num_steps=1, seed=13, **NOSPATILE, **kwargs).eval()


def _load_streaming_benchmark():
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "scripts/inference/benchmark_pixelumm_streaming_decode.py"
    spec = importlib.util.spec_from_file_location("benchmark_pixelumm_streaming_decode", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "pixel_boundaries",
    [
        # Real n86 sample (402 latent frames): every boundary is a multiple of 4.
        [0, 408, 624, 816, 1032, 1224, 1440, 1608],
        # Boundaries with different modulo-4 residues: the last push absorbs the global tail.
        [0, 10, 17, 33],
    ],
)
def test_streaming_benchmark_stream_decode_matches_full_decode(pixel_boundaries):
    benchmark = _load_streaming_benchmark()
    wrapper = _streaming_wrapper()
    num_frames = pixel_boundaries[-1]
    latent_frames = (num_frames - 1) // 4 + 1
    latent = torch.randn((1, 32, latent_frames, 1, 1), generator=torch.Generator().manual_seed(3))
    with torch.inference_mode():
        reference = wrapper.decode(latent, return_dict=False, pixel_frames=num_frames)[0]
        streamed, pushes = benchmark.stream_decode(wrapper, latent, pixel_boundaries)
    assert len(pushes) == len(pixel_boundaries) - 1
    assert streamed.shape == reference.shape
    metrics = benchmark.compare(reference, streamed)
    assert metrics["max_abs"] <= 1e-5, metrics


def test_streaming_session_rejects_multistep():
    wrapper = PixelUMMDCAELatentDecoder(_WindowContextDecoder(), num_steps=2, **NOSPATILE)
    with pytest.raises(ValueError, match="num_steps=1"):
        wrapper.create_streaming_decode_session()


def test_streaming_preview_matches_full_decode_for_arbitrary_segments():
    wrapper = _streaming_wrapper()
    latent_lengths = [3, 2, 4]
    pixel_lengths = [10, 7, 15]
    latent = torch.randn((1, 32, sum(latent_lengths), 2, 2), generator=torch.Generator().manual_seed(21))
    with torch.inference_mode():
        session = wrapper.create_streaming_decode_session(to_cpu=True)
        assembled = None
        latent_end = 0
        raw_start = 0
        visible_ranges = []
        for latent_length, pixel_length in zip(latent_lengths, pixel_lengths):
            update = session.push_preview(latent[:, :, latent_end : latent_end + latent_length], pixel_frames=pixel_length)
            latent_end += latent_length
            assembled = (
                update.video
                if assembled is None
                else torch.cat([assembled[:, :, : assembled.shape[2] - update.replace_pixel_frames], update.video], dim=2)
            )
            visible_ranges.append((raw_start, raw_start + pixel_length))
            raw_start += latent_length * 4
            # Cumulative full-clip decode of the same latents is the reference at every push.
            raw_reference = wrapper.decode(latent[:, :, :latent_end], return_dict=False)[0]
            visible_reference = torch.cat([raw_reference[:, :, a:b] for a, b in visible_ranges], dim=2)
            assert assembled.shape[2] == sum(pixel_lengths[: len(visible_ranges)])
            torch.testing.assert_close(assembled, visible_reference, rtol=0, atol=1e-6)
        session.finish()
        torch.testing.assert_close(session.full_video(), visible_reference, rtol=0, atol=1e-6)
    # finish() publishes the withheld provisional tail.
    assert session.stable_pixel_frames == session.available_pixel_frames == sum(pixel_lengths)


def _streaming_worker(rank, world_size, rendezvous, result_path):
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size
    )
    try:
        torch.set_num_threads(1)
        latent = torch.randn((1, 32, 7, 2, 2), generator=torch.Generator().manual_seed(5))
        reference = _streaming_wrapper()
        parallel = _streaming_wrapper()
        parallel.configure_distributed_decode("window", process_group=dist.group.WORLD, output_rank=0)
        with torch.inference_mode():
            raw = reference.decode(latent, return_dict=False)[0]
            session = parallel.create_streaming_decode_session(to_cpu=True)
            first = session.push_preview(latent[:, :, :4], pixel_frames=16)
            second = session.push_preview(latent[:, :, 4:], pixel_frames=12)
        if rank == 0:
            assembled = torch.cat(
                [first.video[:, :, : first.video.shape[2] - second.replace_pixel_frames], second.video], dim=2
            )
            result = {
                "max_abs": float((assembled - raw).abs().max()),
                "shape_equal": list(assembled.shape) == list(raw.shape),
            }
        else:
            result = {"numel": first.video.numel() + second.video.numel()}
        gathered = [None] * world_size
        dist.all_gather_object(gathered, result)
        if rank == 0:
            Path(result_path).write_text(json.dumps(gathered))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_streaming_window_parallel_matches_single_process_full_decode():
    with tempfile.TemporaryDirectory() as directory:
        result_path = os.path.join(directory, "results.json")
        mp.start_processes(
            _streaming_worker,
            args=(2, os.path.join(directory, "rendezvous"), result_path),
            nprocs=2,
            join=True,
            start_method="spawn",
        )
        results = json.loads(Path(result_path).read_text())
    assert results[0]["shape_equal"]
    assert results[0]["max_abs"] <= 1e-6
    assert results[1]["numel"] == 0


def _worker(rank, world_size, geometry, rendezvous, result_path):
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size
    )
    try:
        torch.set_num_threads(1)
        tiling, local_tile = (TILING, 0) if geometry == "tiled" else (NOSPATILE, LOCAL_TILE_TOKENS)
        results = {}
        latent = _latent()
        for num_steps in (1, 2):
            reference = PixelUMMDCAELatentDecoder(
                _tiny_decoder(local_tile), num_steps=num_steps, seed=9, **tiling
            ).eval()
            with torch.no_grad():
                expected = reference.decode(latent, return_dict=False, pixel_frames=25)[0]
            parallel = PixelUMMDCAELatentDecoder(
                _tiny_decoder(local_tile), num_steps=num_steps, seed=9, **tiling
            ).eval()
            parallel.configure_distributed_decode("window", process_group=dist.group.WORLD, output_rank=0)
            # Non-output ranks hold a different latent: the output rank's copy must win.
            local = latent if rank == 0 else torch.zeros_like(latent)
            with torch.no_grad():
                video = parallel.decode(local, return_dict=False, pixel_frames=25)[0]
            results[str(num_steps)] = (
                {"equal": torch.equal(video, expected), "shape": list(video.shape)}
                if rank == 0
                else {"numel": video.numel()}
            )
        gathered = [None] * world_size
        dist.all_gather_object(gathered, results)
        if rank == 0:
            Path(result_path).write_text(json.dumps(gathered))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
@pytest.mark.parametrize(
    "geometry,world_size",
    # tiled: 3 windows x (2 x 4) tiles = 24 calls, world 5 leaves an idle slot.
    # nospatile: 3 whole-frame window calls, world 2 leaves an idle slot.
    [("tiled", 2), ("tiled", 5), ("nospatile", 2)],
)
def test_window_parallel_matches_single_process_bitwise(geometry, world_size):
    with tempfile.TemporaryDirectory() as directory:
        result_path = os.path.join(directory, "results.json")
        mp.start_processes(
            _worker,
            args=(world_size, geometry, os.path.join(directory, "rendezvous"), result_path),
            nprocs=world_size,
            join=True,
            start_method="spawn",
        )
        results = json.loads(Path(result_path).read_text())
    for num_steps in ("1", "2"):
        assert results[0][num_steps]["equal"], num_steps
        assert results[0][num_steps]["shape"] == [1, 3, 25, 96, 160]
        for rank_result in results[1:]:
            assert rank_result[num_steps]["numel"] == 0


def test_streaming_tail_fix_pads_only_unfinished_pushes():
    """Tail fix: provisional tails sit on the stride grid and are noise-filled to a full window.

    Only a push that a later push will correct (``final_segment = False``) is padded; the push that
    ends the video is not, so the finished stream still equals the full-clip decode exactly.
    """
    latent = torch.randn((1, 32, 27, 1, 1), generator=torch.Generator().manual_seed(11))
    window = NOSPATILE["temporal_window_latents"]
    pushes = [9, 9, 9]
    wrapper = _streaming_wrapper(tail_hold_latents=1)
    assert wrapper.hold_back_pixel_frames == 4
    assert _streaming_wrapper(tail_hold_latents=0).hold_back_pixel_frames == 0

    seen = []
    decode = wrapper.decoder.forward
    wrapper.decoder.forward = lambda lat, xt: (seen.append(int(lat.shape[2])), decode(lat, xt))[1]
    with torch.inference_mode():
        reference = wrapper.decode(latent, return_dict=False)[0]
        session = wrapper.create_streaming_decode_session(to_cpu=True, retain_history=True)
        start = 0
        for index, length in enumerate(pushes):
            session.final_segment = index == len(pushes) - 1
            session.push_preview(latent[:, :, start : start + length], pixel_frames=length * 4)
            start += length
        session.finish()
        streamed = session.full_video()

    assert set(seen) == {window}, "every decoder call keeps the trained window length"
    assert torch.equal(streamed, reference), "the finished stream must equal the full-clip decode"


def test_streaming_rollback_replaces_last_push_without_changing_prefix():
    wrapper = _streaming_wrapper()
    latent = torch.randn((1, 32, 10, 1, 1), generator=torch.Generator().manual_seed(18))
    session = wrapper.create_streaming_decode_session(to_cpu=True)
    with torch.inference_mode():
        first = session.push_preview(latent[:, :, :4], pixel_frames=16)
        session.push_preview(latent[:, :, 4:7], pixel_frames=12)
        session.rollback_last_push()
        assert session.available_latent_frames == 4
        assert session.available_pixel_frames == 16
        replacement = session.push_preview(latent[:, :, 7:], pixel_frames=12)
        assembled = torch.cat([first.video[:, :, :replacement.pixel_start], replacement.video], dim=2)
        retained = torch.cat([latent[:, :, :4], latent[:, :, 7:]], dim=2)
        expected = wrapper.decode(retained, return_dict=False)[0]
    torch.testing.assert_close(assembled, expected, rtol=0, atol=1e-6)


def test_decoder_inputs_are_contiguous_across_clip_lengths():
    decoder = _LocalDecoder()
    observed = []
    decoder.register_forward_pre_hook(
        lambda module, args: observed.append(tuple(x.is_contiguous() for x in args))
    )
    wrapper = PixelUMMDCAELatentDecoder(decoder, **TILING).eval()
    with torch.inference_mode():
        for frames in (7, 9):
            wrapper.decode(torch.zeros(1, 32, frames, 3, 5), return_dict=False)
    assert len(observed) > 1
    assert all(latent_contiguous and noise_contiguous for latent_contiguous, noise_contiguous in observed)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_streaming_preview_tail_reuse_is_bitwise_equal(dtype):
    # Demo-shaped pushes: 10-latent stable pushes, then a preview of the rest of the segment.
    segments = [36, 27, 27, 18, 18, 27, 18, 18, 18, 18, 27, 18, 13, 31, 9]
    latent = torch.randn((1, 32, sum(segments), 1, 1), generator=torch.Generator().manual_seed(5))
    wrapper = _streaming_wrapper().to(dtype)
    latent = latent.to(dtype)
    calls = {True: [], False: []}
    with torch.inference_mode():
        outputs = {}
        for reuse in (True, False):
            session = wrapper.create_streaming_decode_session(to_cpu=True)
            session.reuse_pending_tail = reuse
            original = session._decode_windows
            session._decode_windows = lambda starts, original=original, reuse=reuse: (
                calls[reuse].append(len(starts)) or original(starts))
            videos, offset = [], 0
            for length in segments:
                for start in range(0, length, 10):
                    end = min(length, start + 10)
                    chunk = latent[:, :, offset + start: offset + end]
                    if end == length:
                        update = session.push_preview(chunk, pixel_frames=chunk.shape[2] * 4)
                        videos.append((update.pixel_start, update.video))
                    else:
                        videos.append((None, session.push(chunk, pixel_frames=chunk.shape[2] * 4)))
                offset += length
            outputs[reuse] = videos
    for (start_a, a), (start_b, b) in zip(outputs[True], outputs[False]):
        assert start_a == start_b
        assert torch.equal(a, b)
    assert sum(calls[True]) < sum(calls[False])
