"""Resolve pinned source and inference-only checkpoints in the application."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

from hy_nomas_model import ModelSettings

SOURCE_REVISION = "01b7dba36b05cf532dd8985e379aecc8b80d395f"
MODEL_REVISION = "bac719a80c8d02f91c485fca804bb8221e149b70"
TEXT_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"


@dataclass(frozen=True)
class ServingSettings:
    """Application-owned device count, startup preparation and playback policy."""

    num_gpus: int
    warmup_frames: int | None = None
    playback_fps: int | None = None


def resolve_assets(
    config_path: Path | None, weights_root: Path
) -> tuple[ModelSettings, ServingSettings]:
    from huggingface_hub import snapshot_download

    config_path = config_path or Path(__file__).with_name("hy_nomas.yaml")
    config = yaml.safe_load(config_path.read_text())
    count = int(config["num_gpus"])
    if count not in (1, 2, 4, 8):
        raise ValueError("num_gpus must be 1, 2, 4, or 8.")
    steps = config.get("denoising_steps", 8)
    if type(steps) is not int or steps not in (4, 6, 8):
        raise ValueError("denoising_steps must be 4, 6, or 8.")
    history = config.get("history_frames")
    if history is not None and (type(history) is not int or not 1 <= history <= 720):
        raise ValueError("history_frames must be null or an integer from 1 to 720.")
    sampling = config.get("history_sampling", "recent")
    if sampling not in ("recent", "temporal_context"):
        raise ValueError("history_sampling must be recent or temporal_context.")
    precision = config.get("precision", "bf16")
    if precision not in ("bf16", "fp8", "int8_fp8"):
        raise ValueError("precision must be bf16, fp8, or int8_fp8.")
    offload = config.get("cpu_offload", False)
    cache_weights = config.get("cache_int8_weights", False)
    batch = config.get("decoder_batch_size", 3)
    if type(offload) is not bool or type(cache_weights) is not bool:
        raise ValueError("cpu_offload and cache_int8_weights must be booleans.")
    if type(batch) is not int or not 1 <= batch <= 4:
        raise ValueError("decoder_batch_size must be an integer from 1 to 4.")
    if offload and (count != 1 or batch != 1 or cache_weights):
        raise ValueError(
            "CPU offload requires one GPU, decoder_batch_size=1, and cache_int8_weights=false."
        )
    if cache_weights and precision != "int8_fp8":
        raise ValueError("cache_int8_weights requires precision=int8_fp8.")
    warmup = config.get("warmup_frames")
    if warmup is not None and (
        type(warmup) is not int or not 48 <= warmup <= 240 or warmup % 24
    ):
        raise ValueError("warmup_frames must be null or 48–240 in multiples of 24.")
    playback = config.get("playback_fps")
    if playback is not None and (type(playback) is not int or playback != 24):
        raise ValueError("playback_fps must be null (adaptive) or the native 24.")
    configured_source = Path(os.environ.get("HY_NOMAS_SOURCE", config["source"]))
    source = (
        configured_source
        if configured_source.is_absolute()
        else config_path.parent / configured_source
    )
    source = source.resolve()
    if (
        subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        != SOURCE_REVISION
    ):
        raise ValueError(f"Hy-Nomas source must be checked out at {SOURCE_REVISION}.")
    root = Path(weights_root)
    model_dir = root / "model"
    text_dir = root / "text_encoder"
    dit_folder = "DiT/distilled_int8" if precision == "int8_fp8" else "DiT/distilled"
    dit_file = (
        "diffusion_pytorch_model_ema_0_995_int8.safetensors"
        if precision == "int8_fp8"
        else "diffusion_pytorch_model_ema_0_995.safetensors"
    )
    decoder_folder = "VAE/PixelUMM_FT_FP8" if precision != "bf16" else "VAE/PixelUMM_FT"
    decoder_file = (
        "diffusion_pytorch_model_fp8.safetensors"
        if precision != "bf16"
        else "diffusion_pytorch_model.safetensors"
    )
    dit = model_dir / dit_folder / dit_file
    decoder = model_dir / decoder_folder / decoder_file
    if not dit.is_file() or not decoder.is_file():
        snapshot_download(
            "tencent/Hy-Nomas",
            revision=MODEL_REVISION,
            allow_patterns=[f"{dit_folder}/*", f"{decoder_folder}/*", "LICENSE"],
            local_dir=model_dir,
        )
    if not (text_dir / "model.safetensors.index.json").is_file() or not all(
        (text_dir / f"model-{index:05d}-of-00005.safetensors").is_file()
        for index in range(1, 6)
    ):
        snapshot_download(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            revision=TEXT_REVISION,
            allow_patterns=["*.json", "*.txt", "*.safetensors", "LICENSE"],
            local_dir=text_dir,
        )
    # Completed weights may be mounted read-only; workers inherit writable
    # compiler locations independently of checkpoint storage.
    kernels = (
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "hy-nomas"
    )
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(kernels / "inductor"))
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(kernels / "extensions"))
    cutlass = Path(config.get("cutlass", "/opt/cutlass"))
    cutlass = (
        cutlass if cutlass.is_absolute() else config_path.parent / cutlass
    ).resolve()
    return ModelSettings(
        source=source,
        dit=dit,
        decoder=decoder,
        text_encoder=text_dir,
        height=int(config["height"]),
        width=int(config["width"]),
        denoising_steps=steps,
        history_frames=history,
        history_sampling=sampling,
        precision=precision,
        cpu_offload=offload,
        cache_int8_weights=cache_weights,
        decoder_batch_size=batch,
        cutlass=cutlass,
    ), ServingSettings(num_gpus=count, warmup_frames=warmup, playback_fps=playback)
