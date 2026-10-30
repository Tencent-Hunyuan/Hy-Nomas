"""Shared inference configuration for the web demo and its launchers."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import parse_args
from omegaconf import DictConfig, open_dict


def load_demo_config(overrides: Sequence[str] = ()) -> DictConfig:
    overrides = list(overrides)
    if not any(
        arg.startswith(("+experiment=", "++experiment=", "experiment="))
        for arg in overrides
    ):
        overrides.insert(0, "+experiment=inference")
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *overrides]
        return parse_args(mode="inference")
    finally:
        sys.argv = original_argv


def apply_server_overrides(config: DictConfig, args: argparse.Namespace) -> None:
    """Apply only explicitly supplied server flags, after Hydra composition."""
    with open_dict(config):
        if args.dit_weight is not None:
            config.dit_weight = args.dit_weight
        if args.text_encoder_model is not None:
            config.text_encoder.model_name = args.text_encoder_model
        if args.rollout_type is not None:
            rollout = "one_shot" if args.rollout_type == "oneshot" else "seg_ar"
            config.inference.rollout_type = config.rollout_type = rollout
            if rollout == "seg_ar":
                config.inference.inference_chunk_size = config.inference_chunk_size = 0
        if args.output_dir is not None:
            config.inference.save_path = config.save_path = args.output_dir
        if args.dry_run_source is not None:
            config.inference.prompt = config.prompt = args.dry_run_source


def launcher_overrides(argv: Sequence[str], environ: dict[str, str]) -> list[str]:
    """Translate legacy environment/CLI options; command-line values win."""
    overrides = []
    experiment = environ.get("EXPERIMENT")
    if experiment:
        overrides.append(f"+experiment={experiment}")
    fields = {
        "DIT_WEIGHT": "dit_weight",
        "SAVE_PATH": "inference.save_path",
        "TRAIN_OUTPUT": "output_dir",
        "DRY_RUN_SOURCE": "inference.prompt",
        "SP_SIZE": "parallelism.sp_size",
        "TP_SIZE": "parallelism.tp_size",
    }
    string_fields = {
        "DIT_WEIGHT", "SAVE_PATH", "TRAIN_OUTPUT", "DRY_RUN_SOURCE",
    }
    for name, field in fields.items():
        value = environ.get(name)
        if value:
            if name in string_fields:
                value = json.dumps(value)
            overrides.append(f"{field}={value}")

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--single-gpu-low-mem-mode", action="store_true")
    parser.add_argument("--dit-weight")
    parser.add_argument("--sp-size", type=int)
    parser.add_argument("--tp-size", type=int)
    parser.add_argument("--save-path", "--output-dir", dest="save_path")
    parser.add_argument("--text-encoder-model")
    parser.add_argument("--dry-run-source")
    parser.add_argument("--rollout-type", choices=("seg_ar", "oneshot", "one_shot"))
    args, hydra_args = parser.parse_known_args(argv)
    flag_fields = {
        "dit_weight": "dit_weight",
        "sp_size": "parallelism.sp_size",
        "tp_size": "parallelism.tp_size",
        "save_path": "inference.save_path",
        "text_encoder_model": "model.text_encoder.model_name",
        "dry_run_source": "inference.prompt",
        "rollout_type": "inference.rollout_type",
    }
    for name, field in flag_fields.items():
        value = getattr(args, name)
        if value is not None:
            if name == "rollout_type" and value == "oneshot":
                value = "one_shot"
            if isinstance(value, str):
                value = json.dumps(value)
            overrides.append(f"{field}={value}")
    overrides.extend(hydra_args)
    if args.single_gpu_low_mem_mode:
        # Resolve the mode once; child launchers receive the selected Hydra preset.
        overrides.append("+experiment=single_gpu_low_mem_inference")
    # Hydra rejects repeated config-group selection. Keep the last value for
    # each key so environment defaults can be overridden by launch arguments.
    latest: dict[str, str] = {}
    for override in overrides:
        key = override.partition("=")[0].lstrip("+")
        latest[key] = override
    return list(latest.values())


def main() -> None:
    overrides = launcher_overrides(sys.argv[1:], dict(os.environ))
    config = load_demo_config(overrides)
    experiment = next(
        (arg.partition("=")[2] for arg in overrides
         if arg.partition("=")[0].lstrip("+") == "experiment"),
        "inference",
    )
    values = {
        "DEMO_CONFIG_FILE": f"configs/experiment/{experiment}.yaml",
        "DIT_WEIGHT": config.dit_weight or "",
        "SAVE_PATH": config.inference.save_path,
        "SP_SIZE": config.parallelism.sp_size,
        "TP_SIZE": config.parallelism.tp_size,
        "DRY_RUN_SOURCE": config.inference.prompt or "",
    }
    for name, value in values.items():
        print(f"export {name}={shlex.quote(str(value))}")
    print("DEMO_INFERENCE_ARGS=(" + " ".join(map(shlex.quote, overrides)) + ")")


if __name__ == "__main__":
    main()
