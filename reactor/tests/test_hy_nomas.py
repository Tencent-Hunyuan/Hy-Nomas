import asyncio
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
import yaml
from reactor_runtime import ApplicationError, CommandError, StepOutcome, UploadedFile
from reactor_runtime.interface.model.contract import ContractError, ModelContract

from hy_nomas import HyNomas
from hy_nomas_attention import decoder_ranges
from hy_nomas_camera import CAMERAS, decode_trajectory, plan_camera, plan_keyboard
from hy_nomas_model import (
    HyNomasInput,
    HyNomasModel,
    HyNomasResult,
    MissingWorld,
    RolloutComplete,
    WorldStart,
)
from hy_nomas_prompts import shape_prompt
from hy_nomas_types import HyNomasState


@pytest.mark.parametrize("prompt", ["A stone courtyard.", "[1, 2]", '{"other": true}'])
def test_camera_continuation_uses_minimal_prompt_only_for_unchanged_plain_text(prompt):
    config = SimpleNamespace(inference={}, dataset={})
    assert shape_prompt(prompt, config) == prompt
    assert (
        shape_prompt(prompt, config, camera_continuation=True) == '{"segments": [{}]}'
    )


@pytest.mark.parametrize("first", [True, False])
def test_explicit_segment_keeps_native_prompt_shaping_on_camera_continuation(
    monkeypatch, first
):
    native = Mock(return_value=("shaped native segment", 1))
    monkeypatch.setitem(
        sys.modules, "hymm.inference", SimpleNamespace(_build_whole_item_prompts=native)
    )
    item = {"segments": [{"visual_description_long": "A courtyard.", "actions": []}]}
    if first:
        item["subject_list"] = []
    config = SimpleNamespace(
        inference={"keep_keywords": ["actions"], "no_desc_long_after_first": True},
        dataset={"json_prompt_normalize_action_timestamp": True},
    )
    prompt = json.dumps(item)
    assert (
        shape_prompt(prompt, config, camera_continuation=True)
        == "shaped native segment"
    )
    args = native.call_args.kwargs
    assert args["normalize_action_timestamp"] is True
    assert args["segments"][0]["_segment_idx"] == (0 if first else 1)
    assert ("visual_description_long" in args["segments"][0]) == first
    assert json.loads(prompt) == item


def test_native_prompt_shaping_failure_propagates(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "hymm.inference",
        SimpleNamespace(_build_whole_item_prompts=Mock(return_value=("a|||b", 2))),
    )
    with pytest.raises(RuntimeError, match="segment count"):
        shape_prompt(
            '{"segments": [{}]}',
            SimpleNamespace(inference={"keep_keywords": ["actions"]}, dataset={}),
        )


def app():
    instance = HyNomas()
    instance.state = HyNomasState()
    instance.engine = Mock()
    messages = []

    async def send(message):
        messages.append(message)

    instance.send = send
    return instance, messages


async def begin(instance, **kwargs):
    values = {
        "prompt": "",
        "seed": 42,
        "segment_frames": 120,
        "camera": instance.state.camera,
        "translation_strength": 1.0,
        "rotation_strength": 2.5,
        "trajectory": None,
    }
    values.update(kwargs)
    return await instance.start(**values)


def result(index=1, world=1, complete=False):
    return HyNomasResult(
        frames=np.zeros((3, 32, 32, 3), np.uint8),
        world_id=world,
        chunk_index=index,
        generated_frames=index * 120,
        emitted_frames=index * 120 - 36,
        complete=complete,
    )


def snapshot(world=1, start=None, prompt="garden"):
    return HyNomasInput(
        world_id=world,
        start=start,
        prompt=prompt,
        c2ws=None,
        intrinsics=None,
        translation_strength=1,
        rotation_strength=2.5,
    )


def test_no_inference_before_start():
    instance, _ = app()
    with pytest.raises(ApplicationError):
        asyncio.run(instance.process_input())
    instance.engine.generate.assert_not_called()


def test_invalid_prompt_refusal_has_no_broadcast_or_state_change():
    instance, messages = app()
    with pytest.raises(CommandError) as error:
        asyncio.run(instance.set_prompt(" "))
    assert error.value.code == "invalid_prompt"
    assert messages == [] and instance.state.prompt == ""


def test_generate_uses_only_snapshot():
    instance, _ = app()
    value = snapshot(start=WorldStart(seed=4, segment_frames=120))
    instance.state = None
    instance.engine.generate.return_value = result()
    assert instance.generate(value).chunk_index == 1
    instance.engine.generate.assert_called_once_with(value)


def test_anchor_until_ack_and_two_live_updates_while_paused():
    async def check():
        instance, _ = app()
        await begin(instance, prompt="garden", seed=123, segment_frames=120)
        first = await instance.process_input()
        assert first.start.seed == 123
        assert (await instance.process_input()).start == first.start
        await instance.process_output(StepOutcome(result=result(), elapsed=1.25))
        assert (await instance.process_input()).start is None
        instance.output.flush = Mock()
        await instance.set_paused(True)
        await instance.set_prompt("rain")
        await instance.set_camera("look_right")
        await instance.set_prompt("sunlight")
        assert instance.state.paused and instance.state._world_id == 1
        with pytest.raises(ApplicationError):
            await instance.process_input()
        await instance.set_paused(False)
        value = await instance.process_input()
        assert value.prompt == "sunlight" and value.start is None
        assert value.c2ws.shape == (120, 4, 4)
        assert not np.allclose(value.c2ws[-1], value.c2ws[0])
        instance.output.flush.assert_not_called()
        instance.engine.reset.assert_not_called()

    asyncio.run(check())


def test_fresh_world_and_session_seed_isolation():
    async def check():
        instance, _ = app()
        await begin(instance, prompt="garden", seed=99, segment_frames=120)
        await instance.process_output(StepOutcome(result=result(), elapsed=1))
        await instance.reset()
        assert instance.state._world_id == 2 and instance.state._completed == 0
        assert (await instance.process_input()).start.seed == 99
        await begin(instance, prompt="forest", seed=42, segment_frames=48)
        assert (await instance.process_input()).start.seed == 42
        instance.on_session_ended()
        instance.engine.reset.assert_called_once()
        fresh = HyNomasState()
        assert fresh._start is None and fresh.paused and fresh._camera_poses is None

    asyncio.run(check())


def test_error_outcome_never_acknowledges_or_increments():
    instance, _ = app()
    asyncio.run(begin(instance, prompt="garden", seed=42, segment_frames=120))
    with pytest.raises(RuntimeError, match="native failure"):
        asyncio.run(
            instance.process_output(
                StepOutcome(error=RuntimeError("native failure"), elapsed=1)
            )
        )
    assert instance.state._completed == 0 and instance.state._applied_world_id is None


def test_real_loop_routes_model_error_through_outcome():
    async def check():
        instance, _ = app()
        await begin(instance, prompt="garden", seed=42, segment_frames=120)
        instance.engine.generate.side_effect = RuntimeError("failed compute")
        instance._on_loop_ready()
        instance._live.set()
        with pytest.raises(RuntimeError, match="failed compute"):
            await asyncio.wait_for(instance.run(), timeout=2)
        assert instance.state._completed == 0
        instance.engine.generate.assert_called_once()

    asyncio.run(check())


def test_real_loop_refuses_paused_without_calling_model():
    async def check():
        instance, _ = app()
        instance._on_loop_ready()
        instance._live.set()
        task = asyncio.create_task(instance.run())
        await asyncio.sleep(0.03)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        instance.engine.generate.assert_not_called()

    asyncio.run(check())


def test_model_owns_history_and_failure_does_not_increment():
    model = HyNomasModel()
    model._backend = Mock()
    model._backend.step.return_value = (result().frames, 120, False)
    with pytest.raises(MissingWorld):
        model.generate(snapshot())
    first = model.generate(snapshot(start=WorldStart(seed=9, segment_frames=120)))
    assert first.chunk_index == 1
    model._backend.reset.assert_called_once()
    model._backend.step.side_effect = RuntimeError("native failed")
    with pytest.raises(RuntimeError):
        model.generate(snapshot(prompt="rain"))
    assert model._chunk_index == 1 and model._generated_frames == 120
    model._backend.step.side_effect = None
    assert model.generate(snapshot(prompt="sunlight")).chunk_index == 2
    model._backend.reset.assert_called_once()
    new = model.generate(
        snapshot(world=2, start=WorldStart(seed=8, segment_frames=120))
    )
    assert new.chunk_index == 1 and new.generated_frames == 120


def test_native_horizon_is_explicit_and_reset_allows_new_world():
    async def check():
        instance, _ = app()
        await begin(instance, prompt="garden", seed=42, segment_frames=120)
        await instance.process_output(
            StepOutcome(result=result(index=24, complete=True), elapsed=1)
        )
        assert instance.state.paused and instance.state._complete
        for call in (
            instance.set_paused(False),
            instance.set_prompt("rain"),
            instance.set_camera("look_left"),
        ):
            with pytest.raises(CommandError) as error:
                await call
            assert error.value.code == "rollout_complete"
        with pytest.raises(ApplicationError):
            await instance.process_input()
        await instance.reset()
        assert not instance.state._complete and not instance.state.paused

    asyncio.run(check())


def test_model_refuses_after_completion():
    model = HyNomasModel()
    model._backend = Mock()
    model._backend.step.return_value = (result().frames, 120, True)
    model.generate(snapshot(start=WorldStart(seed=42, segment_frames=120)))
    with pytest.raises(RolloutComplete):
        model.generate(snapshot())
    assert model._backend.step.call_count == 1


@pytest.mark.parametrize(
    "mode", [x for x in CAMERAS if x not in ("text_only", "trajectory")]
)
def test_camera_planning_is_rigid_and_uses_native_identity_first_frame(mode):
    poses, intrinsics = plan_camera(mode, 120)
    assert poses.dtype == np.float32 and poses.shape == (120, 4, 4)
    assert intrinsics.shape == (120, 3, 3)
    assert np.array_equal(poses[0], np.eye(4))
    rotation = poses[:, :3, :3]
    assert np.allclose(np.linalg.det(rotation), 1, atol=1e-6)
    if mode in ("stationary", "keyboard"):
        assert np.allclose(poses, np.eye(4))
    else:
        assert not np.allclose(poses[-1], np.eye(4))


def test_text_only_disables_camera_and_invalid_motion_is_refused():
    assert plan_camera("text_only", 120) == (None, None)
    instance, messages = app()
    contract = ModelContract.of(HyNomas)
    for command in ("start", "set_camera"):
        with pytest.raises(ContractError):
            contract.validate(command, {"camera": "teleport"})
    assert not messages and instance.state.camera == "text_only"


def test_keyboard_has_one_atomic_control_path():
    contract = ModelContract.of(HyNomas)
    assert "set_keyboard" in contract.commands
    assert (
        not {
            "set_w",
            "set_s",
            "set_a",
            "set_d",
            "set_up",
            "set_down",
            "set_left",
            "set_right",
        }
        & contract.commands.keys()
    )
    instance, messages = app()
    payload = contract.validate("set_keyboard", {"world_id": 1, "w": True})
    with pytest.raises(CommandError) as caught:
        asyncio.run(instance.set_keyboard(**vars(payload)))
    assert caught.value.code == "no_world"
    assert not messages and instance.state._held_keys == ()


def npz(poses=None, intrinsics=None):
    buffer = io.BytesIO()
    default_poses, default_intrinsics = plan_camera("forward", 240)
    np.savez(
        buffer,
        c2ws=default_poses if poses is None else poses,
        intrinsics=default_intrinsics if intrinsics is None else intrinsics,
    )
    return buffer.getvalue()


@pytest.mark.parametrize("data", [b"", b"not npz", b"a" * (4 * 1024 * 1024 + 1)])
def test_camera_upload_rejects_bad_or_oversized_data(data):
    with pytest.raises(ValueError):
        decode_trajectory(data)


def test_camera_upload_roundtrip_and_nonrigid_refusal():
    poses, intrinsics = decode_trajectory(npz())
    assert poses.shape == (240, 4, 4) and intrinsics.shape == (240, 3, 3)
    poses[0, 0, 0] = 2
    with pytest.raises(ValueError, match="rigid"):
        decode_trajectory(npz(poses=poses))


def test_camera_upload_refuses_nan_and_object_arrays():
    poses, _ = plan_camera("forward", 240)
    poses[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        decode_trajectory(npz(poses=poses))
    with pytest.raises(ValueError):
        decode_trajectory(npz(poses=np.array([object()], dtype=object)))


def test_trajectory_cursor_ack_and_exhaustion_do_not_reset_world():
    async def check():
        instance, _ = app()
        poses, intrinsics = decode_trajectory(npz())
        instance.state._camera_poses = poses
        instance.state._camera_intrinsics = intrinsics
        await instance.set_camera("trajectory")
        await begin(instance, prompt="garden", seed=42, segment_frames=120)
        first = await instance.process_input()
        assert np.array_equal(first.c2ws, poses[:120])
        assert instance.state._camera_cursor == 0
        await instance.process_output(StepOutcome(result=result(), elapsed=1))
        second = await instance.process_input()
        assert np.array_equal(second.c2ws, poses[120:])
        await instance.process_output(StepOutcome(result=result(index=2), elapsed=1))
        with pytest.raises(ApplicationError):
            await instance.process_input()
        with pytest.raises(CommandError):
            await instance.set_paused(False)
        await instance.set_camera("text_only")
        await instance.set_paused(False)
        assert (await instance.process_input()).start is None
        assert instance.state._world_id == 1

    asyncio.run(check())


def test_connect_sends_current_world_only_to_joiner_without_reset():
    instance, messages = app()
    asyncio.run(begin(instance, prompt="courtyard"))
    asyncio.run(instance.process_output(StepOutcome(result=result(3), elapsed=0.5)))
    before = instance._status()
    messages.clear()
    joiner = SimpleNamespace(send=AsyncMock())
    asyncio.run(instance.on_connected(joiner))
    joiner.send.assert_awaited_once_with(before)
    assert messages == []
    assert instance._status() == before
    instance.engine.reset.assert_not_called()


def test_get_state_is_a_read_only_caller_reply():
    instance, messages = app()
    asyncio.run(begin(instance, prompt="courtyard"))
    asyncio.run(instance.process_output(StepOutcome(result=result(3), elapsed=0.5)))
    before = instance._status()
    messages.clear()
    reply = instance.get_state()
    assert reply == before and instance._status() == before
    assert messages == []
    instance.engine.generate.assert_not_called()
    instance.engine.reset.assert_not_called()


def test_segment_geometry_refusal_is_atomic():
    instance, messages = app()
    with pytest.raises(CommandError):
        asyncio.run(begin(instance, prompt="garden", seed=42, segment_frames=49))
    assert instance.state._start is None and not messages


def test_model_imports_without_runtime_or_gpu_packages():
    script = """
import sys
class BlockHeavy:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'reactor_runtime', 'torch', 'hymm'}:
            raise AssertionError(f'model import reached {fullname}')
sys.meta_path.insert(0, BlockHeavy())
import hy_nomas_model, hy_nomas_backend
assert hy_nomas_model.HyNomasModel()._world_id is None
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_new_trajectory_can_start_a_world_after_completion():
    async def check():
        instance, _ = app()
        await begin(instance, prompt="garden")
        await instance.process_output(
            StepOutcome(result=result(index=24, complete=True), elapsed=1)
        )
        upload = UploadedFile(
            name="path.npz", mime_type="application/octet-stream", data=npz()
        )
        await begin(instance, prompt="garden", camera="trajectory", trajectory=upload)
        value = await instance.process_input()
        assert value.world_id == 2 and value.start is not None
        assert value.c2ws.shape == (120, 4, 4)
        assert not instance.state._complete

    asyncio.run(check())


def test_malformed_upload_command_does_not_change_camera_or_world():
    async def check():
        instance, messages = app()
        upload = UploadedFile(
            name="broken.npz", mime_type="application/octet-stream", data=b"broken"
        )
        with pytest.raises(CommandError) as error:
            await begin(
                instance, prompt="garden", camera="trajectory", trajectory=upload
            )
        assert error.value.code == "invalid_camera_path"
        assert not messages and instance.state._start is None
        with pytest.raises(CommandError):
            await instance.set_camera_path(upload)
        assert instance.state.camera == "text_only"

    asyncio.run(check())


async def keyboard(instance, keys=(), stop=False, world_id=None, latch=True):
    return await instance.set_keyboard(
        world_id=instance.state._world_id if world_id is None else world_id,
        **{
            key: key in keys
            for key in ("w", "s", "a", "d", "up", "down", "left", "right")
        },
        stop=stop,
        latch=latch,
    )


def test_keyboard_combination_and_prompt_share_one_world_without_reset():
    async def check():
        instance, _ = app()
        await begin(instance, prompt="garden", camera="keyboard")
        await instance.process_output(StepOutcome(result=result(), elapsed=1))
        instance.output.flush = Mock()
        await instance.set_paused(True)
        await keyboard(instance, ("w", "d", "right", "up"))
        await instance.set_prompt("rainy garden")
        assert instance.state.paused
        await instance.set_paused(False)
        value = await instance.process_input()
        assert (
            value.prompt == "rainy garden"
            and value.world_id == 1
            and value.start is None
        )
        expected, _ = plan_keyboard(("w", "d", "right", "up"), 120)
        np.testing.assert_array_equal(value.c2ws, expected)
        np.testing.assert_array_equal((await instance.process_input()).c2ws, expected)
        instance.output.flush.assert_not_called()
        instance.engine.reset.assert_not_called()

    asyncio.run(check())


def test_releasing_a_consumed_chord_does_not_queue_extra_motion():
    async def check():
        instance, _ = app()
        await begin(instance, prompt="garden", camera="keyboard")
        await keyboard(instance, ("w", "right"))
        await instance.process_input()
        await keyboard(instance, ("right",), latch=False)
        await keyboard(instance, latch=False)
        assert np.allclose((await instance.process_input()).c2ws, np.eye(4))

    asyncio.run(check())


def test_keyboard_tap_survives_release_once_and_stop_cancels_pending():
    async def check():
        instance, _ = app()
        await begin(instance, prompt="garden", camera="keyboard")
        await keyboard(instance, ("w", "left"))
        await keyboard(instance)
        assert instance._status().keyboard_keys == []
        assert instance._status().queued_keyboard_keys == ["w", "left"]
        first = await instance.process_input()
        assert not np.allclose(first.c2ws[-1], np.eye(4))
        second = await instance.process_input()
        assert np.allclose(second.c2ws, np.eye(4))
        await keyboard(instance, ("s",))
        await keyboard(instance, stop=True)
        assert np.allclose((await instance.process_input()).c2ws, np.eye(4))

    asyncio.run(check())


def test_stale_keyboard_world_is_refused_and_reset_clears_held_keys():
    async def check():
        instance, messages = app()
        with pytest.raises(CommandError) as error:
            await keyboard(instance, ("w",), world_id=1)
        assert error.value.code == "no_world" and not messages
        await begin(instance, prompt="garden", camera="keyboard")
        await keyboard(instance, ("w",))
        await instance.reset()
        messages.clear()
        with pytest.raises(CommandError) as error:
            await keyboard(instance, ("w",), world_id=1)
        assert error.value.code == "world_changed" and not messages
        assert np.allclose((await instance.process_input()).c2ws, np.eye(4))
        await keyboard(instance, ("d",))
        await instance.set_camera("text_only")
        await instance.set_camera("keyboard")
        assert np.allclose((await instance.process_input()).c2ws, np.eye(4))

    asyncio.run(check())


@pytest.mark.parametrize(
    "grid,tile",
    [
        ((2, 9, 17), 4),
        ((1, 4, 17), 4),
        ((3, 17, 4), 4),
        ((2, 8, 16), 4),
        ((10, 44, 80), 8),
    ],
)
def test_decoder_rectangles_preserve_native_tile_mask(grid, tile):
    frames, height, width = grid
    rows, columns = -(-height // tile), -(-width // tile)
    # Independently enumerate native tile-major token order and its predicate.
    tokens = sorted(
        (h // tile, w // tile, t, h % tile, w % tile)
        for t in range(frames)
        for h in range(height)
        for w in range(width)
    )
    coords = np.asarray([token[:2] for token in tokens])
    qr, kr = decoder_ranges(grid, tile)
    for query in np.linspace(0, len(tokens) - 1, min(25, len(tokens)), dtype=int):
        center = np.maximum(
            1, np.minimum(coords[query], [max(rows - 2, 1), max(columns - 2, 1)])
        )
        expected = (np.abs(coords - center) <= 1).all(axis=1)
        actual = np.zeros(len(tokens), dtype=bool)
        for (q0, q1), (k0, k1) in zip(qr, kr):
            if q0 <= query < q1:
                actual[k0:k1] = True
        np.testing.assert_array_equal(actual, expected)


def test_decoder_ranges_reject_invalid_geometry():
    for grid, tile in [((0, 4, 4), 4), ((1, 4, 4), 0)]:
        with pytest.raises(ValueError):
            decoder_ranges(grid, tile)


@pytest.mark.parametrize("precision", ["bf16", "fp8", "int8_fp8"])
@pytest.mark.parametrize(
    "steps,history,sampling",
    [(4, 36, "recent"), (6, 18, "temporal_context"), (8, None, "recent")],
)
def test_resolved_performance_settings(
    tmp_path, monkeypatch, steps, history, sampling, precision
):
    from hy_nomas_assets import SOURCE_REVISION, resolve_assets

    download = Mock()
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download)
    )
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: SOURCE_REVISION)
    for name in [
        "model/DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors",
        "model/VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors",
        "model/DiT/distilled_int8/diffusion_pytorch_model_ema_0_995_int8.safetensors",
        "model/VAE/PixelUMM_FT_FP8/diffusion_pytorch_model_fp8.safetensors",
        "text_encoder/model.safetensors.index.json",
        *[f"text_encoder/model-{i:05d}-of-00005.safetensors" for i in range(1, 6)],
    ]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    config = {
        "source": str(tmp_path),
        "num_gpus": 4,
        "height": 720,
        "width": 1280,
        "denoising_steps": steps,
        "history_frames": history,
        "history_sampling": sampling,
        "precision": precision,
        "warmup_frames": 72,
        "playback_fps": 24,
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    settings, count = resolve_assets(path, tmp_path)
    assert (
        settings.denoising_steps,
        settings.history_frames,
        settings.history_sampling,
        count.num_gpus,
    ) == (steps, history, sampling, 4)
    assert settings.precision == precision
    assert ("distilled_int8" in str(settings.dit)) == (precision == "int8_fp8")
    assert ("PixelUMM_FT_FP8" in str(settings.decoder)) == (precision != "bf16")
    assert count.warmup_frames == 72 and count.playback_fps == 24
    download.assert_not_called()


@pytest.mark.parametrize(
    "override",
    [
        {"denoising_steps": 3},
        {"denoising_steps": True},
        {"history_frames": 0},
        {"history_frames": 721},
        {"history_frames": True},
        {"history_sampling": "unknown"},
        {"precision": "int4"},
        {"cpu_offload": "yes"},
        {"cpu_offload": True},
        {"cache_int8_weights": True},
        {"decoder_batch_size": 0},
        {"decoder_batch_size": True},
        {"decoder_batch_size": 5},
        {"warmup_frames": 49},
        {"warmup_frames": True},
        {"playback_fps": 30},
        {"playback_fps": True},
    ],
)
def test_bad_performance_settings_fail_before_asset_access(
    tmp_path, monkeypatch, override
):
    from hy_nomas_assets import resolve_assets

    download = Mock()
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download)
    )
    config = {"source": "/missing", "num_gpus": 4, "height": 720, "width": 1280}
    config.update(override)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        resolve_assets(path, tmp_path)
    download.assert_not_called()


@pytest.mark.parametrize("override", [False, True])
def test_completed_read_only_weights_need_no_runtime_writes(
    tmp_path, monkeypatch, override
):
    from hy_nomas_assets import SOURCE_REVISION, resolve_assets

    weights = tmp_path / "weights"
    for name in [
        "model/DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors",
        "model/VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors",
        "text_encoder/model.safetensors.index.json",
        *[f"text_encoder/model-{i:05d}-of-00005.safetensors" for i in range(1, 6)],
    ]:
        path = weights / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    download = Mock(side_effect=AssertionError("Completed weights must not download"))
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download)
    )
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: SOURCE_REVISION)
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {"source": str(tmp_path), "num_gpus": 4, "height": 720, "width": 1280}
        )
    )
    cache = tmp_path / "cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
    expected = {
        "TORCHINDUCTOR_CACHE_DIR": cache / "hy-nomas/inductor",
        "TORCH_EXTENSIONS_DIR": cache / "hy-nomas/extensions",
    }
    for name, path in expected.items():
        if override:
            path = tmp_path / "explicit" / name.lower()
            expected[name] = path
            monkeypatch.setenv(name, str(path))
        else:
            monkeypatch.delenv(name, raising=False)
    mkdir = Path.mkdir

    def reject_weight_write(path, *args, **kwargs):
        if path.is_relative_to(weights):
            raise PermissionError("read-only weights")
        return mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", reject_weight_write)
    weights.chmod(0o555)
    try:
        settings, _ = resolve_assets(config, weights)
    finally:
        weights.chmod(0o755)
    assert settings.dit.is_file()
    assert all(os.environ[name] == str(path) for name, path in expected.items())
    assert not (weights / "kernels").exists()
    download.assert_not_called()


def test_startup_warmup_discards_world_and_counters(tmp_path, monkeypatch):
    from hy_nomas_model import ModelSettings

    backend = Mock()
    backend.step.return_value = (result().frames, 72, False)
    factory = Mock(return_value=backend)
    monkeypatch.setitem(
        sys.modules, "hy_nomas_backend", SimpleNamespace(HyNomasBackend=factory)
    )
    settings = ModelSettings(
        source=tmp_path, dit=tmp_path, decoder=tmp_path, text_encoder=tmp_path
    )
    warmup = snapshot(world=0, start=WorldStart(seed=42, segment_frames=72))
    model = HyNomasModel()
    model.load(settings, warmup)
    assert backend.step.call_count == 2
    assert backend.step.call_args_list[0].args[0] is warmup
    assert backend.step.call_args_list[1].args[0].start is None
    assert model._world_id is None and model._chunk_index == 0
    assert model._generated_frames == model._emitted_frames == 0
    first = snapshot(world=1, start=WorldStart(seed=123, segment_frames=72))
    produced = model.generate(first)
    assert produced.world_id == 1 and produced.chunk_index == 1
    assert produced.generated_frames == 72
    backend.start.assert_called_with(first.start)


def test_application_owns_warmup_and_playback_policy(tmp_path, monkeypatch):
    import hy_nomas
    from hy_nomas_assets import ServingSettings
    from hy_nomas_model import ModelSettings

    class LocalApp(HyNomas):
        pass

    settings = ModelSettings(
        source=tmp_path, dit=tmp_path, decoder=tmp_path, text_encoder=tmp_path
    )
    serving = ServingSettings(num_gpus=4, warmup_frames=72, playback_fps=24)
    monkeypatch.setattr(hy_nomas, "resolve_assets", lambda *args: (settings, serving))
    monkeypatch.setattr(hy_nomas, "get_weights_path", lambda: str(tmp_path))
    runner = Mock()
    monkeypatch.setattr(hy_nomas, "DistributedRunner", runner)
    instance = LocalApp()
    instance.load(None)
    arguments = runner.call_args.kwargs
    warmup = arguments["load_kwargs"]["warmup"]
    assert arguments["world_size"] == 4
    assert warmup.world_id == 0 and warmup.start.segment_frames == 72
    assert warmup.c2ws.shape == (72, 4, 4) and warmup.intrinsics.shape == (72, 3, 3)
    assert LocalApp.fps == 24
    runner.return_value.start.assert_called_once()
    serving = ServingSettings(num_gpus=1)
    LocalApp().load(None)
    assert "fps" not in vars(LocalApp)
    assert runner.call_args.kwargs["load_kwargs"]["warmup"] is None
