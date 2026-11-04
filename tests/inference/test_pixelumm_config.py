"""Hydra composition, loader, and demo contracts for inline VAE settings."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from configs.config import _flatten_cfg
from hymm.models.vae.pixelumm_decoder.autoencoder import PixelUMMDCAELatentDecoder
from hymm.models.vae.pixelumm_decoder.loader import maybe_load_pixelumm_decoder


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _compose(*overrides):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="config", overrides=list(overrides))


def test_pixelumm_options_are_declared_in_inference_preset():
    cfg = OmegaConf.load(CONFIG_DIR / "experiment/inference.yaml").model.vae
    assert cfg.name == "pixelumm"
    assert cfg.checkpoint == "/path/to/pixelumm.safetensors"
    for key in ("defaults", "enabled", "pixelumm_decoder"):
        assert key not in cfg


@pytest.mark.parametrize("experiment", ["inference", "sft"])
def test_vae_settings_are_fully_resolved_in_experiment_presets(experiment):
    preset = OmegaConf.load(CONFIG_DIR / f"experiment/{experiment}.yaml")
    cfg = _compose(f"+experiment={experiment}")
    assert list(preset.defaults) == ["_self_"]
    for key, value in preset.model.vae.items():
        assert cfg.model.vae[key] == value


def test_inference_uses_inline_pixelumm_config_with_offline_streaming_disabled():
    cfg = _compose("+experiment=inference")
    assert "pixelumm_decoder" not in cfg
    assert cfg.model.vae.name == "pixelumm"
    assert "enabled" not in cfg.model.vae
    assert cfg.model.vae.parallel_mode == "window"
    assert cfg.inference.seg_ar_segment_vae_decode is False
    assert cfg.model.vae.stable_prefix_streaming is False
    assert cfg.model.vae.vae_precision == "bf16"
    assert cfg.model.vae.deepgemm_fp8 is False
    assert cfg.model.vae.cutlass_fp8 is False

    args = _flatten_cfg(cfg)
    assert args.vae_type == "pixelumm"
    assert args.vae == cfg.model.vae
    assert "pixelumm_decoder" not in args


def test_pixelumm_nested_cli_overrides_survive_flattening_and_demo_launcher():
    from apps.seg_ar_demo_config import launcher_overrides, load_demo_config

    overrides = [
        "model.vae.checkpoint=/tmp/custom.safetensors",
        "model.vae.parallel_mode=none",
        "model.vae.compile_decoder=true",
        "model.vae.temporal_window_latents=12",
    ]
    args = _flatten_cfg(_compose("+experiment=inference", *overrides))
    demo = load_demo_config(launcher_overrides(overrides, {}))
    assert args.vae == demo.vae
    assert args.vae.checkpoint == "/tmp/custom.safetensors"
    assert args.vae.parallel_mode == "none"
    assert args.vae.compile_decoder is True
    assert args.vae.temporal_window_latents == 12


@pytest.mark.parametrize("overrides", [[], ["+experiment=sft"]])
def test_training_and_plain_defaults_still_select_dcae(overrides):
    args = _flatten_cfg(_compose(*overrides))
    assert args.vae.name == "dcae-v-f32t4c32"
    assert args.vae_type == "dcae-v-f32t4c32"
    assert "pixelumm_decoder" not in args
    assert maybe_load_pixelumm_decoder(args, device="cpu", logger=None) is None


@pytest.mark.parametrize("vae_name", [None, "dcae-v-f32t4c32", "vae2.0"])
def test_pixelumm_loader_returns_none_for_other_vaes(vae_name):
    args = SimpleNamespace(vae=SimpleNamespace(name=vae_name))
    assert maybe_load_pixelumm_decoder(args, device="cpu", logger=None) is None


def test_pixelumm_loader_forwards_nested_options(monkeypatch):
    from hymm.distributed.parallel_states import nccl_info

    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    decoder = Mock()
    decoder.denoising_step_list = [0.999, 0.866, 0.634, 0.342]
    decoder.num_steps = 1
    factory = Mock(return_value=decoder)
    monkeypatch.setattr(PixelUMMDCAELatentDecoder, "from_checkpoint", factory)
    monkeypatch.setattr(nccl_info, "sp_size", 2)
    monkeypatch.setattr(nccl_info, "sp_group", "test-sp-group")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    args = _flatten_cfg(_compose(
        "+experiment=inference",
        "model.vae.checkpoint=/tmp/custom.safetensors",
        "model.vae.compile_decoder=true",
        "model.vae.temporal_window_latents=12",
        "model.vae.temporal_overlap_latents=6",
        "model.vae.tile_pixels=128",
        "model.vae.tile_overlap_pixels=32",
    ))
    assert maybe_load_pixelumm_decoder(
        args, device=torch.device("cpu"), logger=Mock(),
    ) is decoder
    factory.assert_called_once_with(
        "/tmp/custom.safetensors",
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        fp8_backend=None,
        local_attention_tile_pixels=128,
        denoising_step_list=[0.999, 0.866, 0.634, 0.342],
        num_steps=1,
        seed=42,
        tail_hold_latents=1,
        decode_output_cpu=False,
        temporal_window_latents=12,
        temporal_overlap_latents=6,
        decoder_tile_pixels=128,
        decoder_tile_overlap_pixels=32,
    )
    decoder.configure_distributed_decode.assert_called_once_with(
        "window", process_group="test-sp-group", output_rank=0,
    )


def test_pixelumm_parallel_loader_requires_active_sp_group(monkeypatch):
    monkeypatch.setattr(PixelUMMDCAELatentDecoder, "from_checkpoint", Mock())
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    args = _flatten_cfg(_compose("+experiment=inference"))
    with pytest.raises(RuntimeError, match="active multi-rank SP/TP model group"):
        maybe_load_pixelumm_decoder(args, device="cpu", logger=Mock())


@pytest.mark.parametrize("is_pixelumm", [False, True])
def test_demo_public_config_reports_selected_vae(is_pixelumm):
    from apps.seg_ar_web_server import GenerationService

    cfg = _flatten_cfg(_compose(
        "+experiment=inference",
        f"model.vae.name={'pixelumm' if is_pixelumm else 'dcae-v-f32t4c32'}",
        "model.vae.checkpoint=/tmp/custom.safetensors",
        "model.vae.stable_prefix_streaming=true",
    ))
    service = GenerationService.__new__(GenerationService)
    service.config = cfg
    service.rollout_type = "seg_ar"
    service.world_size = 8
    service.dry_run_source = None
    public = service.public_config()
    assert public["decoder"] == ("pixelumm" if is_pixelumm else "dcae-v-f32t4c32")
    assert public["pixelumm_checkpoint"] == ("/tmp/custom.safetensors" if is_pixelumm else None)
    assert public["pixelumm_stable_prefix_streaming"] is is_pixelumm


@pytest.mark.parametrize("is_pixelumm", [False, True])
def test_replay_checks_new_vae_selector(tmp_path, is_pixelumm):
    from apps.seg_ar_replay import replay_sessions

    cfg = _flatten_cfg(_compose(
        "+experiment=inference",
        f"model.vae.name={'pixelumm' if is_pixelumm else 'dcae-v-f32t4c32'}",
        "inference.seg_ar_segment_vae_decode=true",
        "model.vae.stable_prefix_streaming=true",
    ))
    service = SimpleNamespace(
        config=cfg, output_dir=tmp_path, public_config=lambda: {},
    )
    if is_pixelumm:
        replay_sessions(service, {"sessions": []})
        assert (tmp_path / "replay_group_000.jsonl").is_file()
    else:
        with pytest.raises(ValueError, match="requires the PixelUMM decoder"):
            replay_sessions(service, {"sessions": []})
