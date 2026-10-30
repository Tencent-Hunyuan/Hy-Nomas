"""HTTP server for synchronous Hy-Nomas video generation.

The server supports two startup-selectable rollout modes:

``seg_ar``
    Keep one ``CPTCausalInferencePipeline`` resident. The model prompt comes
    from the webpage's prompt field.

``oneshot``
    Keep one ``MultiPromptBidirectionalPipeline`` resident. The webpage's
    ``subject`` value must be an absolute text-file path; the backend reads that
    file and uses its contents (including optional ``|||`` separators) as the
    model prompt. Webpage-generated JSON/text is not used as model text.

Rank 0 runs a FastAPI/uvicorn server; the remaining sequence-parallel ranks wait
in a broadcast loop and participate in every generation request.

Example (single GPU)::

    python apps/seg_ar_web_server.py \
        +experiment=inference \
        --rollout-type seg_ar \
        --dit-weight /path/to/model.safetensors \
        --host 0.0.0.0 \
        --port 8000

Example (8-way sequence parallelism)::

    torchrun --standalone --nproc_per_node=8 \
        apps/seg_ar_web_server.py \
        +experiment=inference \
        --rollout-type oneshot \
        --dit-weight /path/to/model.safetensors \
        --host 0.0.0.0 \
        --port 8000

Hydra overrides may be appended directly, for example::

    inference.video_size='[352,640]' inference.infer_steps_video=20

API:

    GET  /health
    GET  /v1/config
    POST /v1/generate
    POST /v1/shutdown
    GET  /outputs/<generated-file>

``POST /v1/generate`` blocks until inference and MP4 encoding finish. Requests
are serialized because one sequence-parallel model replica can execute only
one collective generation at a time.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import random
import re
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import torch
import torch.distributed as dist
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger
from omegaconf import DictConfig, ListConfig, open_dict
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("PYTHONPATH", str(PROJECT_ROOT))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PROMPT_SEPARATOR = "|||"
GENERATE_COMMAND = "generate"
IDLE_COMMAND = "idle"
ROLLBACK_COMMAND = "rollback"
RELEASE_SESSION_COMMAND = "release_session"
SHUTDOWN_COMMAND = "shutdown"
from apps.seg_ar_demo_config import apply_server_overrides, load_demo_config
from apps.seg_ar_job_store import JobStore
from apps.camera_video_overlay import burn_video_overlays, video_overlays_enabled
from hymm.models.dit.modules.uniform_history_cache import (
    UnsupportedStreamingRollbackError, validate_swa_streaming_rollback,
)


def _classify_inference_error(exc: BaseException) -> str:
    message = str(exc).lower()
    if isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in message:
        return "OUT_OF_MEMORY"
    if any(
        marker in message
        for marker in (
            "shape mismatch",
            "size mismatch",
            "shape",
            "must match",
        )
    ):
        return "SHAPE_MISMATCH"
    if any(marker in message for marker in ("cuda", "nccl", "cublas", "cudnn")):
        return "CUDA_ERROR"
    return "INFERENCE_ERROR"


def _parse_bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _publish_local_file(
    source: Path,
    destination: Path,
    *,
    attempts: int = 3,
) -> None:
    """Copy a complete local file to the public output path with retries.

    COS FUSE can intermittently return ``EIO`` while renaming a fully copied
    ``.uploading`` file. The URL is not returned to the browser until this
    function completes, so writing the final unique path directly is both safe
    for this service and avoids the unreliable remote rename operation.
    """
    expected_size = source.stat().st_size
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Optional[OSError] = None
    for attempt in range(max(1, attempts)):
        try:
            with source.open("rb") as reader, destination.open("wb") as writer:
                shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
                writer.flush()
            actual_size = destination.stat().st_size
            if actual_size != expected_size:
                raise OSError(
                    5,
                    f"published file size mismatch ({actual_size} != {expected_size})",
                    str(destination),
                )
            return
        except OSError as exc:
            last_error = exc
            destination.unlink(missing_ok=True)
            if attempt + 1 < attempts:
                logger.warning(
                    f"Output publish attempt {attempt + 1}/{attempts} failed "
                    f"for {destination}: {exc}; retrying"
                )
                time.sleep(0.25 * (attempt + 1))
    assert last_error is not None
    raise last_error


def _publish_output_bytes(data: bytes, destination: Path) -> None:
    local_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="nomas_publish_",
            suffix=destination.suffix,
            delete=False,
        ) as handle:
            local_path = Path(handle.name)
            handle.write(data)
        _publish_local_file(local_path, destination)
    finally:
        if local_path is not None:
            local_path.unlink(missing_ok=True)


class GenerateRequest(BaseModel):
    """One synchronous generation request."""

    prompt: str = ""
    subject: str = ""
    negative_prompt: Optional[str] = None
    height: Optional[int] = Field(default=None, ge=16)
    width: Optional[int] = Field(default=None, ge=16)
    num_frames: Optional[int] = Field(default=None, ge=1)
    seed: Optional[int] = None
    guidance_scale: Optional[float] = Field(default=None, ge=0.0)
    num_inference_steps: Optional[int] = Field(default=None, ge=1)
    flow_shift: Optional[float] = Field(default=None, gt=0.0)
    solver_type: Optional[Literal["euler"]] = None
    segment_boundaries: Optional[List[int]] = None
    camera_trajectory: Optional[str] = None
    trajectory_translation_scale: Optional[float] = Field(
        default=None, gt=0.0, allow_inf_nan=False,
    )
    trajectory_rotation_scale: Optional[float] = Field(
        default=None, gt=0.0, allow_inf_nan=False,
    )
    segment_has_camera: Optional[List[bool]] = None
    user_prompts: Optional[List[str]] = None
    stable_prefix_carry_frames: int = Field(default=0, ge=0)
    stable_prefix_carry_total_frames: int = Field(default=0, ge=0)
    carry_camera_trajectory: Optional[str] = None
    carry_user_prompt: str = ""
    carry_has_camera: bool = False
    cfg_schedule: Optional[str] = None
    cfg_skip_steps: Optional[int] = Field(default=None, ge=0)
    session_id: Optional[str] = Field(default=None, max_length=128)
    preserve_kv_cache: bool = False
    reset_session: bool = False
    trusted_segment_boundaries: bool = False


class GenerationService:
    """Own one resident pipeline and coordinate requests across all ranks."""

    def __init__(
        self,
        config: DictConfig,
        device: torch.device,
        output_dir: Path,
        rank: int,
        world_size: int,
        world_mesh: Any,
        command_group: Any,
        rollout_type: str,
        dry_run_source: Optional[Path] = None,
    ) -> None:
        self.config = config
        self.device = device
        self.output_dir = output_dir
        self.rank = rank
        self.world_size = world_size
        self.world_mesh = world_mesh
        self.command_group = command_group
        self.rollout_type = rollout_type
        self.dry_run_source = dry_run_source
        self._lock = threading.Lock()
        self._dry_run_lock = threading.Lock()
        self._dry_run_jobs: Dict[str, Dict[str, Any]] = {}
        self._dry_run_entries: Optional[List[Any]] = None
        self._progress_callbacks: Dict[
            str, Callable[[int, int, str, torch.Tensor], None]
        ] = {}
        self._shutdown_lock = threading.Lock()
        self._shutdown_started = False
        self._request_count = 0
        self._active_session_id: Optional[str] = None
        self._kv_cpu_offload_enabled = bool(
            rank == 0 and _parse_bool_env("NOMAS_KV_CPU_OFFLOAD", False)
        )
        self._kv_cpu_offload_min_tokens = max(
            0,
            int(os.environ.get("NOMAS_KV_CPU_OFFLOAD_MIN_TOKENS", "0")),
        )
        self._kv_cache_offloaded = False

        self.output_dir.mkdir(parents=True, exist_ok=True)

        started = time.perf_counter()
        if rollout_type == "seg_ar":
            from hymm.pipeline.causal_ar import CPTCausalInferencePipeline

            logger.info("Loading CPTCausalInferencePipeline (seg_ar)...")
            self.pipeline = CPTCausalInferencePipeline(
                args=config,
                device=device,
                dit_model_path=config.dit_weight,
                world_mesh=world_mesh,
            )
        else:
            from hymm.pipeline.bidirectional import MultiPromptBidirectionalPipeline

            logger.info(
                "Loading MultiPromptBidirectionalPipeline (oneshot file prompt)..."
            )
            self.pipeline = MultiPromptBidirectionalPipeline(
                args=config,
                device=device,
                dit_model_path=config.dit_weight,
                world_mesh=world_mesh,
            )
        if hasattr(self.pipeline, "prewarm_interactive_kv_cache"):
            self.pipeline.prewarm_interactive_kv_cache()
        logger.info(
            f"{type(self.pipeline).__name__} ready in "
            f"{time.perf_counter() - started:.1f}s"
        )

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    @property
    def shutdown_started(self) -> bool:
        return self._shutdown_started

    def public_config(self) -> Dict[str, Any]:
        """Return the model/runtime fields consumed by the lightweight gateway."""
        height, width = self._video_size(self.config)
        vae_config = getattr(self.config, "vae", None)
        is_pixelumm = getattr(vae_config, "name", None) == "pixelumm"
        pixelumm_stable_prefix = bool(
            is_pixelumm
            and bool(
                getattr(
                    vae_config,
                    "stable_prefix_streaming",
                    False,
                )
            )
        )
        return {
            "rollout_type": self.rollout_type,
            "prompt_source": (
                "subject_file" if self.rollout_type == "oneshot" else "prompt"
            ),
            "height": height,
            "width": width,
            "num_frames": (
                None
                if self.rollout_type == "oneshot"
                else int(self.config.inference.num_frames)
            ),
            "num_frames_source": (
                "prompt_timestamps"
                if self.rollout_type == "oneshot"
                else "request_or_config"
            ),
            "fps": int(self.config.inference.video_fps),
            "guidance_scale": float(self.config.inference.cfg_scale_video),
            "trajectory_translation_scale": float(
                getattr(self.config.inference, "trajectory_translation_scale", None)
                if getattr(self.config.inference, "trajectory_translation_scale", None) is not None
                else 1.0
            ),
            "trajectory_rotation_scale": float(
                getattr(self.config.inference, "trajectory_rotation_scale", None)
                if getattr(self.config.inference, "trajectory_rotation_scale", None) is not None
                else 1.0
            ),
            "num_inference_steps": int(self.config.inference.infer_steps_video),
            "flow_shift": float(self.config.inference.infer_flow_shift_video),
            "sp_size": int(getattr(self.config.parallelism, "sp_size", 1)),
            "tp_size": int(getattr(self.config.parallelism, "tp_size", 1)),
            "model_name": str(getattr(self.config, "name", "") or ""),
            "dit_weight": str(self.config.dit_weight),
            "decoder": (
                "pixelumm"
                if is_pixelumm
                else str(getattr(self.config, "vae_type", "vae"))
            ),
            "pixelumm_checkpoint": (
                str(getattr(vae_config, "checkpoint", "") or "")
                if is_pixelumm
                else None
            ),
            "pixelumm_stable_prefix_streaming": pixelumm_stable_prefix,
            "dry_run_enabled": bool(
                self.rollout_type == "seg_ar"
                and self.dry_run_source
                and self.dry_run_source.is_file()
            ),
            "dry_run_source": (
                str(self.dry_run_source) if self.dry_run_source else None
            ),
        }

    @staticmethod
    def _video_size(config: DictConfig) -> Tuple[int, int]:
        value = config.inference.video_size
        if isinstance(value, (list, tuple, ListConfig)):
            height = int(value[0])
            width = int(value[1] if len(value) > 1 else value[0])
        else:
            height = width = int(value)
        return height, width

    @staticmethod
    def _read_prompt_file(subject: str) -> Tuple[str, str]:
        raw_path = subject.strip()
        if not raw_path:
            raise ValueError(
                "oneshot mode requires subject to contain an absolute .txt file path"
            )
        prompt_path = Path(raw_path).expanduser()
        if not prompt_path.is_absolute():
            raise ValueError("oneshot subject path must be absolute")
        prompt_path = prompt_path.resolve()
        if not prompt_path.is_file():
            raise ValueError(f"oneshot prompt file does not exist: {prompt_path}")
        if prompt_path.suffix.lower() != ".txt":
            raise ValueError(f"oneshot prompt file must end with .txt: {prompt_path}")
        max_bytes = 2 * 1024 * 1024
        file_size = prompt_path.stat().st_size
        if file_size > max_bytes:
            raise ValueError(
                f"oneshot prompt file is too large ({file_size} bytes; max {max_bytes})"
            )
        try:
            prompt = prompt_path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"oneshot prompt file is not valid UTF-8: {prompt_path}"
            ) from exc
        if not prompt:
            raise ValueError(f"oneshot prompt file is empty: {prompt_path}")
        return prompt, str(prompt_path)

    @staticmethod
    def _timestamp_seconds(value: Any) -> float:
        if value is None:
            raise ValueError("timestamp is missing")
        if isinstance(value, (int, float)):
            seconds = float(value)
        else:
            raw = str(value).strip()
            parts = raw.split(":")
            if len(parts) == 3:
                hours, minutes, seconds_raw = parts
                seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds_raw)
            elif len(parts) == 2:
                minutes, seconds_raw = parts
                seconds = int(minutes) * 60 + float(seconds_raw)
            else:
                seconds = float(raw)
        if seconds < 0:
            raise ValueError(f"timestamp must be non-negative, got {value!r}")
        return seconds

    @classmethod
    def _oneshot_timing_from_prompts(
        cls,
        prompts: List[str],
        fps: int,
    ) -> Tuple[int, Optional[List[int]], List[float]]:
        """Derive output length from per-prompt JSON timestamps.

        Each ``|||`` prompt is treated as one sequential video segment. Its
        duration is taken from the largest local end timestamp found in:

        1. ``segments[*].contained_actions[*].end_time``
        2. ``segments[*].end_time``

        Prompt-file timestamps are segment-local in the Stage-7 sidecars, so
        durations are accumulated across prompts. Pixel-frame boundaries use
        ``int(total_seconds * fps)`` and the total frame count is the final
        boundary. This mirrors ``hymm/inference.py`` event-length behavior while
        handling sidecars whose later prompt timestamps restart from zero.
        """
        if fps <= 0:
            raise ValueError(f"video_fps must be positive, got {fps}")

        durations: List[float] = []
        frame_lengths: List[int] = []
        for prompt_index, prompt in enumerate(prompts):
            try:
                item = json.loads(prompt)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "oneshot automatic frame selection requires every '|||' "
                    f"prompt to be JSON; segment {prompt_index} is invalid"
                ) from exc
            if not isinstance(item, dict):
                raise ValueError(
                    f"oneshot prompt segment {prompt_index} must be a JSON object"
                )
            segments = item.get("segments")
            if not isinstance(segments, list) or not segments:
                raise ValueError(
                    f"oneshot prompt segment {prompt_index} has no segments list"
                )

            segment_starts: List[float] = []
            segment_ends: List[float] = []
            action_starts: List[float] = []
            action_ends: List[float] = []
            for segment in segments:
                if not isinstance(segment, dict):
                    continue
                if segment.get("start_time") is not None:
                    segment_starts.append(cls._timestamp_seconds(segment["start_time"]))
                if segment.get("end_time") is not None:
                    segment_ends.append(cls._timestamp_seconds(segment["end_time"]))
                actions = segment.get("contained_actions")
                if isinstance(actions, dict):
                    actions = [actions]
                if isinstance(actions, list):
                    for action in actions:
                        if not isinstance(action, dict):
                            continue
                        if action.get("start_time") is not None:
                            action_starts.append(
                                cls._timestamp_seconds(action["start_time"])
                            )
                        if action.get("end_time") is not None:
                            action_ends.append(
                                cls._timestamp_seconds(action["end_time"])
                            )

            # Stage-9 keeps segment timestamps on the global video timeline,
            # while json_prompt_normalize_action_timestamp=true makes action
            # timestamps local to their segment. Never combine those two
            # coordinate systems when deriving this request's frame count.
            if segment_ends:
                local_start = min(segment_starts) if segment_starts else 0.0
                local_end = max(segment_ends)
            elif action_ends:
                local_start = min(action_starts) if action_starts else 0.0
                local_end = max(action_ends)
            else:
                raise ValueError(
                    f"oneshot prompt segment {prompt_index} has no usable end_time; "
                    "pass num_frames explicitly or add timestamps to the prompt file"
                )
            duration = local_end - local_start
            if duration <= 0:
                raise ValueError(
                    f"oneshot prompt segment {prompt_index} has non-positive "
                    f"duration ({duration}s)"
                )
            durations.append(duration)
            start_frame = int(local_start * fps)
            end_frame = int(local_end * fps)
            frame_lengths.append(max(1, end_frame - start_frame))

        boundaries = [0]
        for frame_length in frame_lengths:
            boundaries.append(boundaries[-1] + frame_length)

        num_frames = boundaries[-1]
        return (
            num_frames,
            boundaries if len(prompts) > 1 else None,
            durations,
        )

    @classmethod
    def _seg_ar_timing_from_prompts(
        cls,
        prompts: List[str],
        fps: int,
    ) -> Tuple[int, List[int], List[float]]:
        """Recompute seg_ar timing from the submitted structured prompts."""
        num_frames, boundaries, durations = cls._oneshot_timing_from_prompts(
            prompts,
            fps,
        )
        return num_frames, boundaries or [0, num_frames], durations

    def _shape_json_prompts_for_text_encoder(
        self,
        prompts: List[str],
    ) -> List[str]:
        """Run the standard whole-item JSON shaping before text encoding.

        The original JSON remains available for duration validation, metadata,
        PE history, and rollback. Only the strings passed to the text encoder
        are transformed, using the same builder as
        ``hymm.inference._load_from_json``.
        """
        keep_keywords = getattr(self.config.inference, "keep_keywords", None)
        if not keep_keywords:
            keep_keywords = getattr(
                getattr(self.config, "dataset", object()),
                "json_prompt_keep_keys",
                None,
            )
        if not keep_keywords:
            return prompts
        from hymm.inference import _build_whole_item_prompts

        shaped: List[str] = []
        for index, prompt in enumerate(prompts):
            try:
                item = json.loads(prompt)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "configured keep_keywords require JSON prompts; "
                    f"segment {index} is invalid"
                ) from exc
            if not isinstance(item, dict):
                raise ValueError(f"prompt segment {index} must be a JSON object")
            segments = item.get("segments")
            if not isinstance(segments, list) or len(segments) != 1:
                raise ValueError("each seg_ar prompt must contain exactly one segment")
            segment = json.loads(json.dumps(segments[0], ensure_ascii=False))
            # Interactive follow-ups omit global context. Mark them as a later
            # event so no_desc_long_after_first applies exactly as it does when
            # the standard loader processes a complete multi-event item.
            is_first_segment = isinstance(item.get("subject_list"), list)
            segment["_segment_idx"] = 0 if is_first_segment else 1
            if not is_first_segment and bool(
                getattr(
                    self.config.inference,
                    "no_desc_long_after_first",
                    False,
                )
            ):
                segment.pop("visual_description_long", None)
            shaped_prompt, count = _build_whole_item_prompts(
                item={**item, "segments": [segment]},
                segments=[segment],
                keep_keywords=(list(keep_keywords) if keep_keywords else None),
                resolve_subject_ids=bool(
                    getattr(
                        self.config.inference,
                        "resolve_subject_ids",
                        False,
                    )
                ),
                global_only_in_first_segment=bool(
                    getattr(
                        self.config.inference,
                        "global_only_in_first_segment",
                        False,
                    )
                ),
                no_desc_long_after_first=bool(
                    getattr(
                        self.config.inference,
                        "no_desc_long_after_first",
                        False,
                    )
                ),
                normalize_action_timestamp=bool(
                    getattr(
                        getattr(self.config, "dataset", object()),
                        "json_prompt_normalize_action_timestamp",
                        False,
                    )
                ),
            )
            if count != 1 or PROMPT_SEPARATOR in shaped_prompt:
                raise RuntimeError("standard prompt shaping changed segment count")
            shaped.append(shaped_prompt)
        return shaped

    def build_params(self, request: GenerateRequest) -> Dict[str, Any]:
        height_default, width_default = self._video_size(self.config)
        prompt_file = None
        if self.rollout_type == "oneshot":
            prompt, prompt_file = self._read_prompt_file(request.subject)
        else:
            prompt = request.prompt.strip()
            if not prompt:
                raise ValueError("seg_ar mode requires a non-empty prompt")
        prompts = [
            item.strip() for item in prompt.split(PROMPT_SEPARATOR) if item.strip()
        ]
        if not prompts:
            raise ValueError("prompt is empty after splitting on '|||'")

        boundaries = request.segment_boundaries
        boundaries_from_request = boundaries is not None
        if boundaries is not None:
            boundaries = [int(value) for value in boundaries]
            if len(boundaries) != len(prompts) + 1:
                raise ValueError(
                    "segment_boundaries must contain exactly len(prompts) + 1 "
                    f"values; got {len(boundaries)} boundaries for {len(prompts)} prompts"
                )
            if boundaries[0] != 0:
                raise ValueError("segment_boundaries must start at 0")
            if any(right <= left for left, right in zip(boundaries, boundaries[1:])):
                raise ValueError("segment_boundaries must be strictly increasing")

        seed = (
            int(request.seed)
            if request.seed is not None
            else int(getattr(self.config.inference, "seed", 42))
        )
        if seed < 0:
            seed = int(torch.randint(0, 2**31, (1,)).item())

        fps = int(getattr(self.config.inference, "video_fps", 24))
        timing_source = "request" if request.num_frames is not None else "config"
        segment_durations_seconds: Optional[List[float]] = None
        if (
            self.rollout_type == "seg_ar"
            and boundaries_from_request
            and not request.trusted_segment_boundaries
        ):
            if request.num_frames is None:
                raise ValueError(
                    "seg_ar requests with segment_boundaries must also provide "
                    "num_frames"
                )
            try:
                (
                    expected_num_frames,
                    expected_boundaries,
                    segment_durations_seconds,
                ) = self._seg_ar_timing_from_prompts(prompts, fps)
            except ValueError as exc:
                raise ValueError(f"seg_ar timestamp validation failed: {exc}") from exc
            if boundaries != expected_boundaries:
                raise ValueError(
                    "seg_ar segment_boundaries do not match prompt timestamps "
                    f"at {fps} FPS: received {boundaries}, "
                    f"expected {expected_boundaries}"
                )
            if int(request.num_frames) != expected_num_frames:
                raise ValueError(
                    "seg_ar num_frames does not match prompt timestamps "
                    f"at {fps} FPS: received {request.num_frames}, "
                    f"expected {expected_num_frames}"
                )
            timing_source = "validated_prompt_timestamps"
        elif (
            self.rollout_type == "seg_ar"
            and boundaries_from_request
            and request.trusted_segment_boundaries
        ):
            segment_durations_seconds = [
                (right - left) / fps for left, right in zip(boundaries, boundaries[1:])
            ]
            timing_source = "standard_loader_boundaries"
        if self.rollout_type == "oneshot" and request.num_frames is None:
            (
                num_frames,
                inferred_boundaries,
                segment_durations_seconds,
            ) = self._oneshot_timing_from_prompts(prompts, fps)
            timing_source = "prompt_timestamps"
            if boundaries is None:
                boundaries = inferred_boundaries
        else:
            num_frames = (
                int(request.num_frames)
                if request.num_frames is not None
                else int(getattr(self.config.inference, "num_frames", 1))
            )
        if boundaries is not None:
            if self.rollout_type == "seg_ar" and request.num_frames is None:
                num_frames = boundaries[-1]
            elif self.rollout_type == "seg_ar" and boundaries[-1] != num_frames:
                raise ValueError(
                    "the final segment boundary must equal num_frames; "
                    f"got {boundaries[-1]} and {num_frames}"
                )
            elif (
                self.rollout_type == "oneshot"
                and boundaries_from_request
                and boundaries[-1] != num_frames
            ):
                raise ValueError(
                    "oneshot segment_boundaries must end at the selected "
                    f"num_frames; got {boundaries[-1]} and {num_frames}"
                )

        session_id = (request.session_id or "").strip()
        if session_id and not re.fullmatch(r"[A-Za-z0-9._:-]+", session_id):
            raise ValueError(
                "session_id may only contain letters, digits, '.', '_', ':', '-'"
            )

        # Match standard inference: request-specific trajectory overrides the
        # experiment's global camera trajectory.
        effective_camera_trajectory = (
            request.camera_trajectory
            if request.camera_trajectory is not None
            else getattr(self.config.inference, "camera_trajectory", None)
        )
        # Camera gating is derived from the effective trajectory. Prompt-only
        # requests keep camera attention disabled.
        has_camera = bool((effective_camera_trajectory or "").strip())
        requested_camera_flags = request.segment_has_camera or []
        segment_has_camera = (
            [bool(value) for value in requested_camera_flags]
            if len(requested_camera_flags) == len(prompts)
            else [has_camera] * len(prompts)
        )
        user_prompts = [str(value).strip() for value in (request.user_prompts or [])]
        if len(user_prompts) != len(prompts):
            user_prompts = [""] * len(prompts)

        model_prompts = (
            self._shape_json_prompts_for_text_encoder(prompts)
            if self.rollout_type == "seg_ar"
            else prompts
        )
        solver_type = request.solver_type or str(
            getattr(self.config.inference, "solver_type", "euler")
        )
        if solver_type != "euler":
            raise ValueError(
                f"Unsupported inference solver: {solver_type!r}; expected 'euler'"
            )

        self._request_count += 1
        return {
            "command": GENERATE_COMMAND,
            "request_id": f"gen_{self._request_count:06d}_{uuid.uuid4().hex[:8]}",
            "prompt": prompt,
            "prompt_file": prompt_file,
            "subject": request.subject.strip(),
            "prompts": model_prompts,
            "negative_prompt": (
                request.negative_prompt.strip()
                if request.negative_prompt is not None
                else str(getattr(self.config.inference, "neg_prompt", "") or "")
            ),
            "height": int(request.height or height_default),
            "width": int(request.width or width_default),
            "num_frames": num_frames,
            "timing_source": timing_source,
            "segment_durations_seconds": segment_durations_seconds,
            "video_fps": fps,
            "seed": seed,
            "guidance_scale": float(
                request.guidance_scale
                if request.guidance_scale is not None
                else self.config.inference.cfg_scale_video
            ),
            "num_inference_steps": int(
                request.num_inference_steps
                if request.num_inference_steps is not None
                else self.config.inference.infer_steps_video
            ),
            "flow_shift": float(
                request.flow_shift
                if request.flow_shift is not None
                else self.config.inference.infer_flow_shift_video
            ),
            "solver_type": solver_type,
            "segment_boundaries": boundaries,
            "segment_boundaries_from_request": boundaries_from_request,
            "camera_trajectory": effective_camera_trajectory,
            "trajectory_translation_scale": (
                request.trajectory_translation_scale
                if request.trajectory_translation_scale is not None
                else getattr(self.config.inference, "trajectory_translation_scale", None)
            ),
            "trajectory_rotation_scale": (
                request.trajectory_rotation_scale
                if request.trajectory_rotation_scale is not None
                else getattr(self.config.inference, "trajectory_rotation_scale", None)
            ),
            "segment_has_camera": segment_has_camera,
            "user_prompts": user_prompts,
            "stable_prefix_carry_frames": int(request.stable_prefix_carry_frames),
            "stable_prefix_carry_total_frames": int(
                request.stable_prefix_carry_total_frames
            ),
            "carry_camera_trajectory": request.carry_camera_trajectory,
            "carry_user_prompt": request.carry_user_prompt.strip(),
            "carry_has_camera": bool(request.carry_has_camera),
            "cfg_schedule": (
                request.cfg_schedule
                if request.cfg_schedule is not None
                else getattr(self.config.inference, "cfg_schedule", None)
            ),
            "cfg_skip_steps": int(
                request.cfg_skip_steps
                if request.cfg_skip_steps is not None
                else getattr(self.config.inference, "cfg_skip_steps", 0)
            ),
            "use_dynamic_cfg_scale": bool(
                getattr(self.config.inference, "use_dynamic_cfg_scale", False)
            ),
            "cfg_scale_min": float(
                getattr(self.config.inference, "cfg_scale_min", 1.0)
            ),
            "cfg_scale_max": float(
                getattr(self.config.inference, "cfg_scale_max", 8.0)
            ),
            "cfg_scale_max_step": int(
                getattr(self.config.inference, "cfg_scale_max_step", 16)
            ),
            "camera_guidance_scale": float(
                getattr(self.config.inference, "camera_cfg_scale", 1.0)
            ),
            "session_id": session_id or None,
            "reset_session": bool(request.reset_session),
            # A session_id identifies/reset state but must not implicitly turn
            # on persistent AR streaming. Direct structured-JSON requests from
            # the webpage include a session_id too, yet they are standalone
            # renders and must finalize all requested frames. The gateway marks
            # only true PE-backed interactive continuations explicitly.
            "preserve_kv_cache": bool(request.preserve_kv_cache),
        }

    def generate(
        self,
        request: GenerateRequest,
        progress_callback: Optional[
            Callable[[int, int, str, torch.Tensor], None]
        ] = None,
    ) -> Dict[str, Any]:
        """Serialize one rank-0 HTTP request into one collective model call."""
        if self.rank != 0:
            raise RuntimeError("HTTP generation may only be called on rank 0")

        params = self.build_params(request)
        with self._lock:
            if progress_callback is not None:
                self._progress_callbacks[params["request_id"]] = progress_callback
            try:
                return self._collective_generate(params)
            finally:
                self._progress_callbacks.pop(params["request_id"], None)

    def _broadcast_command(self, command: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if self.world_size <= 1:
            if command is None:
                raise RuntimeError("rank-0 command is required in single-process mode")
            return command
        objects = [command]
        # Worker ranks spend most of their lifetime blocked waiting for a
        # command. Using the default NCCL group for this wait leaves a
        # long-lived GPU collective resident on every device. With two SP8
        # replicas sharing the same 8 GPUs through CUDA MPS, those waiting
        # kernels can starve the other replica's first denoise collective.
        # Keep model/SP collectives on NCCL, but transport the tiny Python
        # command object over a CPU/Gloo group.
        dist.broadcast_object_list(
            objects,
            src=0,
            group=self.command_group,
        )
        received = objects[0]
        if not isinstance(received, dict):
            raise RuntimeError(f"invalid distributed command: {type(received)!r}")
        return received

    def _reset_pipeline_state(self) -> None:
        if self.pipeline is None:
            return
        if hasattr(self.pipeline, "_finish_interactive_decode_phase"):
            self.pipeline._finish_interactive_decode_phase()
        if hasattr(self.pipeline, "kv_cache"):
            kv_cache = self.pipeline.kv_cache
            if (
                kv_cache is not None
                and hasattr(kv_cache, "reset")
                and hasattr(kv_cache, "cache_masks")
            ):
                kv_cache.reset()
            else:
                self.pipeline.kv_cache = None
        if hasattr(self.pipeline, "cache_mask_list"):
            kv_cache = getattr(self.pipeline, "kv_cache", None)
            self.pipeline.cache_mask_list = (
                kv_cache.cache_masks
                if kv_cache is not None and hasattr(kv_cache, "cache_masks")
                else None
            )
        if hasattr(self.pipeline, "cache_valid_len"):
            self.pipeline.cache_valid_len = 0
        if hasattr(self.pipeline, "global_frame_counter"):
            self.pipeline.global_frame_counter = 0
        if hasattr(self.pipeline, "segment_token_lengths"):
            self.pipeline.segment_token_lengths.clear()
        if hasattr(self.pipeline, "_interactive_cached_action_tails"):
            self.pipeline._interactive_cached_action_tails = []
        if hasattr(self.pipeline, "_interactive_cached_seg_video_ranges"):
            self.pipeline._interactive_cached_seg_video_ranges = []
        if hasattr(self.pipeline, "_interactive_segment_count"):
            self.pipeline._interactive_segment_count = 0
        if hasattr(self.pipeline, "_interactive_segment_cache_tokens"):
            self.pipeline._interactive_segment_cache_tokens = []
        if hasattr(self.pipeline, "_interactive_segment_latent_lengths"):
            self.pipeline._interactive_segment_latent_lengths = []
        if hasattr(self.pipeline, "_reset_interactive_noise_state"):
            self.pipeline._reset_interactive_noise_state()
        if hasattr(self.pipeline, "_reset_interactive_pixelumm_decode_state"):
            self.pipeline._reset_interactive_pixelumm_decode_state()
        self._active_session_id = None
        self._kv_cache_offloaded = False

    def _move_interactive_kv_cache(
        self,
        device: torch.device | str,
    ) -> int:
        if self.pipeline is None or getattr(self.pipeline, "kv_cache", None) is None:
            self._kv_cache_offloaded = False
            return 0
        if hasattr(self.pipeline.kv_cache, "cache_masks"):
            self._kv_cache_offloaded = False
            return 0
        target = torch.device(device)
        moved_bytes = 0
        for cache in self.pipeline.kv_cache:
            for key in ("k", "v", "k_scale", "v_scale"):
                if key not in cache:
                    continue
                tensor = cache[key]
                if tensor.device == target:
                    continue
                moved_bytes += tensor.numel() * tensor.element_size()
                cache[key] = tensor.to(
                    device=target,
                    non_blocking=False,
                )
        masks = getattr(self.pipeline, "cache_mask_list", None)
        if masks is not None:
            for index, tensor in enumerate(masks):
                if tensor.device == target:
                    continue
                moved_bytes += tensor.numel() * tensor.element_size()
                moved_mask = tensor.to(
                    device=target,
                    non_blocking=False,
                )
                if hasattr(tensor, "_action_tail_ranges"):
                    moved_mask._action_tail_ranges = list(tensor._action_tail_ranges)
                masks[index] = moved_mask
        self._kv_cache_offloaded = target.type == "cpu"
        return moved_bytes

    def _maybe_restore_interactive_kv_cache(self) -> None:
        if not self._kv_cache_offloaded:
            return
        started = time.perf_counter()
        moved = self._move_interactive_kv_cache(self.device)
        logger.info(
            "Restored interactive KV cache to GPU: "
            f"{moved / (1024**3):.2f} GiB in "
            f"{time.perf_counter() - started:.3f}s"
        )

    def _maybe_offload_interactive_kv_cache(self) -> None:
        if (
            not self._kv_cpu_offload_enabled
            or self._active_session_id is None
            or getattr(self.pipeline, "kv_cache", None) is None
            or int(getattr(self.pipeline, "cache_valid_len", 0))
            < self._kv_cpu_offload_min_tokens
        ):
            return
        started = time.perf_counter()
        moved = self._move_interactive_kv_cache("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(
            "Offloaded idle interactive KV cache to CPU: "
            f"{moved / (1024**3):.2f} GiB in "
            f"{time.perf_counter() - started:.3f}s"
        )

    def rollback_session(self, session_id: str) -> Dict[str, Any]:
        """Collectively remove the latest interactive segment from all ranks."""
        if self.rank != 0:
            raise RuntimeError("session rollback may only be initiated on rank 0")
        with self._lock:
            if (
                not session_id
                or self._active_session_id != session_id
                or getattr(self.pipeline, "kv_cache", None) is None
            ):
                raise ValueError(
                    "the requested interactive session is not active on this worker"
                )
            # Reject unsupported policies before waking the other SP ranks.
            validate_swa_streaming_rollback(self.pipeline.kv_cache)
            command = {
                "command": ROLLBACK_COMMAND,
                "session_id": session_id,
            }
            params = self._broadcast_command(command)
            result = self._rollback_session_from_received(params)
            if result["remaining_segments"] == 0:
                self._reset_pipeline_state()
            return result

    def release_session(self, session_id: str) -> Dict[str, Any]:
        """Collectively release one resident interactive session."""
        if self.rank != 0:
            raise RuntimeError("session release may only be initiated on rank 0")
        with self._lock:
            if not session_id or self._active_session_id != session_id:
                return {
                    "session_id": session_id,
                    "released": False,
                }
            command = {
                "command": RELEASE_SESSION_COMMAND,
                "session_id": session_id,
            }
            params = self._broadcast_command(command)
            return self._release_session_from_received(params)

    def _release_session_from_received(
        self,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        session_id = str(params.get("session_id") or "")
        released = bool(session_id and self._active_session_id == session_id)
        if released:
            self._reset_pipeline_state()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return {
            "session_id": session_id,
            "released": released,
        }

    def _rollback_session_from_received(
        self,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        session_id = str(params.get("session_id") or "")
        if (
            not session_id
            or self._active_session_id != session_id
            or getattr(self.pipeline, "kv_cache", None) is None
        ):
            raise ValueError(
                "the requested interactive session is not active on this worker"
            )
        result = self.pipeline.rollback_last_interactive_segment()
        if result["remaining_segments"] == 0:
            self._reset_pipeline_state()
        return result

    def _prepare_session_state(self, params: Dict[str, Any]) -> None:
        preserve = bool(params.get("preserve_kv_cache"))
        session_id = params.get("session_id")
        reset = bool(params.get("reset_session"))
        continuing = (
            preserve
            and not reset
            and bool(session_id)
            and self._active_session_id == session_id
            and getattr(self.pipeline, "kv_cache", None) is not None
        )
        params["continue_kv_cache"] = continuing
        logger.info(
            "Interactive AR session state: "
            f"session_id={session_id!r}, preserve={preserve}, reset={reset}, "
            f"continue={continuing}, active_session={self._active_session_id!r}, "
            f"cache_present={getattr(self.pipeline, 'kv_cache', None) is not None}, "
            f"cache_valid_len={getattr(self.pipeline, 'cache_valid_len', 0)}, "
            f"global_frames={getattr(self.pipeline, 'global_frame_counter', 0)}"
        )
        if not continuing:
            self._reset_pipeline_state()
        if preserve and session_id:
            self._active_session_id = str(session_id)

    def _collective_generate(
        self, command: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        params = self._broadcast_command(command)
        if params.get("command") == SHUTDOWN_COMMAND:
            return None
        if params.get("command") != GENERATE_COMMAND:
            raise RuntimeError(f"unsupported command: {params.get('command')!r}")

        request_id = params["request_id"]
        logger.info(
            f"[{request_id}] {self.rollout_type} generation: "
            f"prompts={len(params['prompts'])}, "
            f"frames={params['num_frames']}, "
            f"timing_source={params['timing_source']}, "
            f"size={params['height']}x{params['width']}"
        )

        with open_dict(self.config):
            internal_rollout = (
                "one_shot" if self.rollout_type == "oneshot" else "seg_ar"
            )
            self.config.inference.rollout_type = internal_rollout
            self.config.rollout_type = internal_rollout
            self.config.inference.solver_type = params["solver_type"]

        self._prepare_session_state(params)
        self._maybe_restore_interactive_kv_cache()
        latent_path: Optional[Path] = None
        latent_output_dir = os.environ.get(
            "NOMAS_INFER_SAVE_LATENTS_DIR",
            "",
        ).strip()
        previous_latent_path = os.environ.get("HY_INFER_SAVE_FINAL_LATENTS_PATH")
        if latent_output_dir:
            worker_slot = os.environ.get("NOMAS_WORKER_SLOT", "0").strip() or "0"
            latent_path = (
                Path(latent_output_dir) / f"slot_{worker_slot}" / f"{request_id}.npy"
            )
            os.environ["HY_INFER_SAVE_FINAL_LATENTS_PATH"] = str(latent_path)
        started = time.perf_counter()
        try:
            output = self._run_pipeline(params)
        except Exception:
            self._reset_pipeline_state()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
        finally:
            if latent_output_dir:
                if previous_latent_path is None:
                    os.environ.pop("HY_INFER_SAVE_FINAL_LATENTS_PATH", None)
                else:
                    os.environ["HY_INFER_SAVE_FINAL_LATENTS_PATH"] = (
                        previous_latent_path
                    )

        generation_seconds = time.perf_counter() - started
        if self.rank != 0:
            del output
            return None

        result = self._save_output(output, params, generation_seconds)
        pixelumm_streaming = getattr(
            self.pipeline,
            "_last_pixelumm_streaming_metadata",
            None,
        )
        if isinstance(pixelumm_streaming, dict):
            result.update(
                {
                    "pixelumm_stable_prefix": True,
                    **pixelumm_streaming,
                }
            )
        if latent_path is not None and latent_path.is_file():
            digest = hashlib.sha256()
            with latent_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            result.update(
                {
                    "latent_path": str(latent_path),
                    "latent_sha256": digest.hexdigest(),
                }
            )
        del output
        self._maybe_offload_interactive_kv_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    def _run_pipeline(self, params: Dict[str, Any]) -> torch.Tensor:
        common = {
            "prompts": params["prompts"],
            "num_frames": params["num_frames"],
            "height": params["height"],
            "width": params["width"],
            "guidance_scale": params["guidance_scale"],
            "num_inference_steps": params["num_inference_steps"],
            "flow_shift": params["flow_shift"],
            "seed": params["seed"],
            "negative_prompt": params["negative_prompt"],
            "camera_trajectory": params["camera_trajectory"],
            "trajectory_translation_scale": params.get("trajectory_translation_scale"),
            "trajectory_rotation_scale": params.get("trajectory_rotation_scale"),
            "segment_has_camera": params["segment_has_camera"],
            "cfg_schedule": params["cfg_schedule"],
            "cfg_skip_steps": params["cfg_skip_steps"],
        }
        if self.rollout_type == "oneshot":
            return self.pipeline.inference(
                **common,
                camera_guidance_scale=params["camera_guidance_scale"],
                prompt_video_boundaries=params["segment_boundaries"],
                use_dynamic_cfg_scale=params["use_dynamic_cfg_scale"],
                cfg_scale_min=params["cfg_scale_min"],
                cfg_scale_max=params["cfg_scale_max"],
                cfg_scale_max_step=params["cfg_scale_max_step"],
            )
        return self.pipeline.inference(
            **common,
            camera_guidance_scale=params["camera_guidance_scale"],
            segment_frame_boundaries=params["segment_boundaries"],
            segment_callback=self._progress_callbacks.get(params["request_id"]),
            preserve_kv_cache=bool(params.get("preserve_kv_cache")),
        )

    def _resolve_output_url(self, relative_url: str) -> Path:
        clean = relative_url.split("?", 1)[0].lstrip("/")
        if clean.startswith("outputs/"):
            clean = clean[len("outputs/") :]
        path = (self.output_dir / clean).resolve()
        if self.output_dir not in path.parents or not path.is_file():
            raise FileNotFoundError(f"session video does not exist: {relative_url}")
        return path

    def _concat_session_video(
        self,
        previous_url: str,
        current_url: str,
        *,
        session_id: str,
        request_id: str,
        replace_tail_frames: int = 0,
        fps: int = 24,
        previous_frame_count: Optional[int] = None,
    ) -> str:
        import imageio_ffmpeg

        previous_path = self._resolve_output_url(previous_url)
        current_path = self._resolve_output_url(current_url)
        safe_session = re.sub(r"[^A-Za-z0-9_-]+", "_", session_id)
        safe_request = re.sub(r"[^A-Za-z0-9_-]+", "_", request_id)
        filename = f"session_{safe_session}_{safe_request}.mp4"
        output_path = self.output_dir / filename
        local_output: Optional[Path] = None
        list_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                prefix="nomas_concat_",
                suffix=".txt",
                delete=False,
                encoding="utf-8",
            ) as handle:
                list_path = Path(handle.name)
                for path in (previous_path, current_path):
                    escaped = str(path).replace("'", "'\\''")
                    handle.write(f"file '{escaped}'\n")
            with tempfile.NamedTemporaryFile(
                prefix="nomas_session_",
                suffix=".mp4",
                delete=False,
            ) as handle:
                local_output = Path(handle.name)
            if replace_tail_frames > 0:
                previous_frames = (
                    int(previous_frame_count)
                    if previous_frame_count is not None
                    else self._video_frame_count(previous_path)
                )
                keep_frames = max(
                    previous_frames - int(replace_tail_frames),
                    0,
                )
                if keep_frames <= 0:
                    shutil.copyfile(current_path, local_output)
                else:
                    subprocess.run(
                        [
                            imageio_ffmpeg.get_ffmpeg_exe(),
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-i",
                            str(previous_path),
                            "-i",
                            str(current_path),
                            "-filter_complex",
                            (
                                f"[0:v]trim=end_frame={keep_frames},"
                                "setpts=PTS-STARTPTS[old];"
                                "[1:v]setpts=PTS-STARTPTS[new];"
                                "[old][new]concat=n=2:v=1:a=0[out]"
                            ),
                            "-map",
                            "[out]",
                            "-an",
                            "-r",
                            str(int(fps)),
                            "-c:v",
                            "libx264",
                            "-preset",
                            "veryfast",
                            "-crf",
                            "18",
                            "-pix_fmt",
                            "yuv420p",
                            "-movflags",
                            "+faststart",
                            "-y",
                            str(local_output),
                        ],
                        check=True,
                    )
            else:
                subprocess.run(
                    [
                        imageio_ffmpeg.get_ffmpeg_exe(),
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-f",
                        "concat",
                        "-safe",
                        "0",
                        "-i",
                        str(list_path),
                        "-c",
                        "copy",
                        "-movflags",
                        "+faststart",
                        "-y",
                        str(local_output),
                    ],
                    check=True,
                )
            _publish_local_file(local_output, output_path)
        finally:
            if local_output is not None:
                local_output.unlink(missing_ok=True)
            if list_path is not None:
                list_path.unlink(missing_ok=True)
        return f"outputs/{filename}"

    @staticmethod
    def _video_frame_count(path: Path) -> int:
        """Return the exact decoded frame count of a local video."""
        import imageio.v2 as imageio

        reader = imageio.get_reader(str(path))
        try:
            count = reader.count_frames()
        finally:
            reader.close()
        return int(count)

    @staticmethod
    def _split_fragmented_mp4(data: bytes) -> Tuple[bytes, bytes]:
        """Split one fragmented MP4 into its init and media byte ranges."""
        offset = 0
        first_moof: Optional[int] = None
        while offset + 8 <= len(data):
            size = int.from_bytes(data[offset : offset + 4], "big")
            box_type = data[offset + 4 : offset + 8]
            header_size = 8
            if size == 1:
                if offset + 16 > len(data):
                    break
                size = int.from_bytes(data[offset + 8 : offset + 16], "big")
                header_size = 16
            elif size == 0:
                size = len(data) - offset
            if size < header_size or offset + size > len(data):
                break
            if box_type == b"moof":
                first_moof = offset
                break
            offset += size
        if first_moof is None:
            raise RuntimeError("ffmpeg output is not a fragmented MP4 (missing moof)")
        init = data[:first_moof]
        media = data[first_moof:]
        if b"moov" not in init or not media[4:8] == b"moof":
            raise RuntimeError("invalid fragmented MP4 init/media split")
        return init, media

    @staticmethod
    def _mse_mime_type(init: bytes) -> str:
        """Read the AVC profile/compatibility/level triplet from ``avcC``."""
        marker = init.find(b"avcC")
        if marker >= 0 and marker + 8 <= len(init):
            profile, compatibility, level = init[marker + 5 : marker + 8]
            return (
                f'video/mp4; codecs="avc1.{profile:02X}{compatibility:02X}{level:02X}"'
            )
        return 'video/mp4; codecs="avc1.64001F"'

    @staticmethod
    def _prepare_mse_media(media: bytes, sequence: int) -> bytes:
        """Keep media boxes only and assign monotonic fragment sequence IDs."""
        output = bytearray()
        offset = 0
        fragment_index = 0
        while offset + 8 <= len(media):
            size = int.from_bytes(media[offset : offset + 4], "big")
            box_type = media[offset + 4 : offset + 8]
            header_size = 8
            if size == 1:
                if offset + 16 > len(media):
                    raise RuntimeError("truncated extended-size MP4 box")
                size = int.from_bytes(media[offset + 8 : offset + 16], "big")
                header_size = 16
            elif size == 0:
                size = len(media) - offset
            if size < header_size or offset + size > len(media):
                raise RuntimeError("truncated fragmented MP4 media box")
            box = bytearray(media[offset : offset + size])
            if box_type == b"moof":
                mfhd = box.find(b"mfhd")
                if mfhd < 0 or mfhd + 12 > len(box):
                    raise RuntimeError("fragmented MP4 moof is missing mfhd")
                sequence_number = sequence * 1000 + fragment_index + 1
                box[mfhd + 8 : mfhd + 12] = sequence_number.to_bytes(4, "big")
                fragment_index += 1
                output.extend(box)
            elif box_type == b"mdat":
                output.extend(box)
            # Drop the random-access trailer (mfra/tfra/mfro). It is useful for
            # standalone files but not part of an appendable MSE media segment.
            offset += size
        if fragment_index == 0:
            raise RuntimeError("fragmented MP4 media contains no moof")
        return bytes(output)

    def _write_mse_fragment(
        self,
        current_url: str,
        *,
        session_id: str,
        sequence: int,
    ) -> Dict[str, Any]:
        """Remux a generated segment into MSE init + appendable fMP4 media."""
        import imageio_ffmpeg

        current_path = self._resolve_output_url(current_url)
        safe_session = re.sub(r"[^A-Za-z0-9_-]+", "_", session_id)
        stream_dir = self.output_dir / "stream" / safe_session
        stream_dir.mkdir(parents=True, exist_ok=True)
        init_path = stream_dir / "init.mp4"
        media_path = stream_dir / f"segment_{sequence:06d}.m4s"
        local_fragmented: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix="nomas_mse_",
                suffix=".mp4",
                delete=False,
            ) as handle:
                local_fragmented = Path(handle.name)
            subprocess.run(
                [
                    imageio_ffmpeg.get_ffmpeg_exe(),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(current_path),
                    "-map",
                    "0:v:0",
                    "-an",
                    "-c",
                    "copy",
                    "-movflags",
                    "+frag_keyframe+empty_moov+default_base_moof",
                    "-f",
                    "mp4",
                    "-y",
                    str(local_fragmented),
                ],
                check=True,
            )
            init_bytes, media_bytes = self._split_fragmented_mp4(
                local_fragmented.read_bytes()
            )
            media_bytes = self._prepare_mse_media(media_bytes, sequence)
            if sequence == 0 or not init_path.is_file():
                _publish_output_bytes(init_bytes, init_path)
            _publish_output_bytes(media_bytes, media_path)
        finally:
            if local_fragmented is not None:
                local_fragmented.unlink(missing_ok=True)
        prefix = f"outputs/stream/{safe_session}"
        return {
            "stream_init_url": f"{prefix}/init.mp4",
            "stream_segment_url": f"{prefix}/{media_path.name}",
            "stream_sequence": sequence,
            "stream_mime_type": self._mse_mime_type(init_bytes),
        }

    @staticmethod
    def _caption_for_prompt(prompt: str) -> str:
        """Return a readable caption for the dry-run chat panel."""
        try:
            item = json.loads(prompt)
        except json.JSONDecodeError:
            return prompt
        if not isinstance(item, dict):
            return prompt
        segments = item.get("segments")
        if not isinstance(segments, list) or not segments:
            return prompt
        segment = segments[0]
        if not isinstance(segment, dict):
            return prompt
        parts: List[str] = []
        for key in ("visual_description_long", "description_change"):
            value = segment.get(key)
            if value:
                parts.append(str(value))
        actions = segment.get("contained_actions")
        if isinstance(actions, dict):
            actions = [actions]
        if isinstance(actions, list):
            for action in actions:
                if isinstance(action, dict) and action.get("action_description"):
                    parts.append(str(action["action_description"]))
        return "\n\n".join(parts) or prompt

    def _load_random_dry_run_request(
        self,
    ) -> Tuple[GenerateRequest, Dict[str, Any]]:
        if self.rollout_type != "seg_ar":
            raise ValueError("dry run requires the server to use --rollout-type seg_ar")
        if self.dry_run_source is None or not self.dry_run_source.is_file():
            raise ValueError(f"dry-run source does not exist: {self.dry_run_source}")

        from hymm.inference import _load_from_json

        if self._dry_run_entries is None:
            self._dry_run_entries = _load_from_json(
                self.dry_run_source,
                max_segments=getattr(self.config.inference, "max_segments", None),
                keep_keywords=getattr(self.config.inference, "keep_keywords", None),
                fps=int(self.config.inference.video_fps),
                desc_field_first=str(
                    getattr(self.config.inference, "desc_field_first", "") or ""
                ),
                desc_field_rest=str(
                    getattr(self.config.inference, "desc_field_rest", "") or ""
                ),
                resolve_subject_ids=bool(
                    getattr(self.config.inference, "resolve_subject_ids", False)
                ),
                global_only_in_first_segment=bool(
                    getattr(
                        self.config.inference,
                        "global_only_in_first_segment",
                        False,
                    )
                ),
                no_desc_long_after_first=bool(
                    getattr(
                        self.config.inference,
                        "no_desc_long_after_first",
                        False,
                    )
                ),
                normalize_action_timestamp=bool(
                    getattr(
                        getattr(self.config, "dataset", object()),
                        "json_prompt_normalize_action_timestamp",
                        False,
                    )
                ),
            )
        entries = self._dry_run_entries
        if not entries:
            raise ValueError(
                f"dry-run source has no usable samples: {self.dry_run_source}"
            )

        prompt, boundaries, filename, entry_meta = random.choice(entries)
        prompts = [
            item.strip() for item in prompt.split(PROMPT_SEPARATOR) if item.strip()
        ]
        if boundaries is None:
            raise ValueError("standard JSON loader returned no segment boundaries")
        num_frames = int(boundaries[-1])
        request = GenerateRequest(
            prompt=prompt,
            num_frames=num_frames,
            segment_boundaries=boundaries,
            trusted_segment_boundaries=True,
            camera_trajectory=(
                entry_meta.get("camera_trajectory")
                if isinstance(entry_meta, dict)
                else None
            ),
        )
        return request, {
            "sample_id": filename or "random_sample",
            "source": str(self.dry_run_source),
            "captions": [self._caption_for_prompt(item) for item in prompts],
            "total_segments": len(prompts),
            "num_frames": num_frames,
        }

    def _write_dry_run_preview(
        self,
        job_id: str,
        segment_index: int,
        total_segments: int,
        video: torch.Tensor,
    ) -> str:
        import imageio.v2 as imageio

        preview_dir = self.output_dir / "dryrun"
        preview_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{job_id}_segment_{segment_index + 1:03d}.mp4"
        output_path = preview_dir / filename
        video_np = (
            video[0].permute(1, 2, 3, 0).mul(255).clamp(0, 255).to(torch.uint8).numpy()
        )
        # Encode and close the MP4 on local storage first. Only then copy the
        # complete file to COS; the URL is published after the copy completes.
        local_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f"{job_id}_segment_",
                suffix=".mp4",
                delete=False,
            ) as handle:
                local_path = Path(handle.name)
            imageio.mimwrite(
                str(local_path),
                video_np,
                fps=int(self.config.inference.video_fps),
                codec="libx264",
                pixelformat="yuv420p",
                output_params=["-movflags", "+faststart"],
            )
            _publish_local_file(local_path, output_path)
        finally:
            if local_path is not None:
                local_path.unlink(missing_ok=True)
        logger.info(
            f"[{job_id}] published dry-run preview "
            f"{segment_index + 1}/{total_segments}: {output_path}"
        )
        return f"outputs/dryrun/{filename}"

    def start_dry_run(self) -> Dict[str, Any]:
        request, sample = self._load_random_dry_run_request()
        job_id = f"dryrun_{uuid.uuid4().hex[:12]}"
        job = {
            "id": job_id,
            "status": "queued",
            **sample,
            "completed_segments": 0,
            "segment_updates": [],
            "video_url": None,
            "error": None,
        }
        with self._dry_run_lock:
            self._dry_run_jobs[job_id] = job

        previous_segment_end_frame = 0

        def progress(
            segment_index: int,
            total_segments: int,
            _prompt: str,
            video: torch.Tensor,
        ) -> None:
            nonlocal previous_segment_end_frame
            preview_url = self._write_dry_run_preview(
                job_id,
                segment_index,
                total_segments,
                video,
            )
            fps = int(self.config.inference.video_fps)
            segment_end_frame = int(video.shape[2])
            update = {
                "segment_index": segment_index,
                "caption": sample["captions"][segment_index],
                "video_url": preview_url,
                "start_frame": previous_segment_end_frame,
                "end_frame": segment_end_frame,
                "playback_start_seconds": previous_segment_end_frame / fps,
                "playback_end_seconds": segment_end_frame / fps,
            }
            previous_segment_end_frame = segment_end_frame
            with self._dry_run_lock:
                current = self._dry_run_jobs[job_id]
                current["status"] = "running"
                current["completed_segments"] = segment_index + 1
                current["segment_updates"].append(update)
                current["video_url"] = preview_url

        def run() -> None:
            with self._dry_run_lock:
                self._dry_run_jobs[job_id]["status"] = "running"
            try:
                result = self.generate(request, progress_callback=progress)
                with self._dry_run_lock:
                    current = self._dry_run_jobs[job_id]
                    current["status"] = "succeeded"
                    current["result"] = result
                    current["video_url"] = result["video_url"]
            except Exception as exc:  # noqa: BLE001
                logger.exception(f"[{job_id}] dry run failed")
                with self._dry_run_lock:
                    current = self._dry_run_jobs[job_id]
                    current["status"] = "failed"
                    current["error"] = str(exc)

        threading.Thread(
            target=run,
            name=job_id,
            daemon=True,
        ).start()
        return dict(job)

    def get_dry_run(self, job_id: str) -> Dict[str, Any]:
        with self._dry_run_lock:
            job = self._dry_run_jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return {
                **job,
                "segment_updates": list(job["segment_updates"]),
            }

    def execute_queue_job(
        self,
        store: JobStore,
        job: Dict[str, Any],
    ) -> None:
        """Execute one durable gateway job on rank 0."""
        job_id = str(job["id"])
        kind = str(job["kind"])
        try:
            if kind == "dryrun":
                request, sample = self._load_random_dry_run_request()
                store.patch(
                    job_id,
                    status="running",
                    **sample,
                    completed_segments=0,
                    segment_updates=[],
                    video_url=None,
                    error=None,
                )
                previous_segment_end_frame = 0

                def progress(
                    segment_index: int,
                    total_segments: int,
                    _prompt: str,
                    video: torch.Tensor,
                ) -> None:
                    nonlocal previous_segment_end_frame
                    preview_url = self._write_dry_run_preview(
                        job_id,
                        segment_index,
                        total_segments,
                        video,
                    )
                    fps = int(self.config.inference.video_fps)
                    segment_end_frame = int(video.shape[2])
                    store.append_segment(
                        job_id,
                        {
                            "segment_index": segment_index,
                            "caption": sample["captions"][segment_index],
                            "video_url": preview_url,
                            "start_frame": previous_segment_end_frame,
                            "end_frame": segment_end_frame,
                            "playback_start_seconds": (
                                previous_segment_end_frame / fps
                            ),
                            "playback_end_seconds": segment_end_frame / fps,
                        },
                    )
                    previous_segment_end_frame = segment_end_frame

                result = self.generate(request, progress_callback=progress)
                store.patch(
                    job_id,
                    status="succeeded",
                    result=result,
                    video_url=result["video_url"],
                    error=None,
                )
                return

            if kind == "generate":
                payload = dict(job.get("payload") or {})
                pe_session = payload.pop("_pe_session", None)
                pe_request_id = payload.pop("_pe_request_id", None)
                if isinstance(pe_session, dict):
                    payload["session_id"] = pe_session.get("session_id")
                    payload["preserve_kv_cache"] = True
                request = GenerateRequest.model_validate(payload)
                result = self.generate(request)
                if isinstance(pe_session, dict):
                    current_segment_url = str(result["video_url"])
                    previous_result = pe_session.get("last_result")
                    replace_tail_frames = min(
                        max(
                            0,
                            int(result.get("replace_pixel_frames", 0) or 0),
                        ),
                        int(request.stable_prefix_carry_frames),
                    )
                    previous_tail_is_published = bool(
                        isinstance(previous_result, dict)
                        and "provisional_pixel_frames" in previous_result
                    )
                    published_replace_frames = (
                        replace_tail_frames if previous_tail_is_published else 0
                    )
                    previous_frames = (
                        int(previous_result.get("frames", 0))
                        if isinstance(previous_result, dict)
                        else 0
                    )
                    visible_new_frames = max(
                        int(result["frames"]) - published_replace_frames,
                        0,
                    )
                    previous_video_url = pe_session.get("video_url")
                    stream_sequence = (
                        int(previous_result.get("stream_sequence", -1)) + 1
                        if isinstance(previous_result, dict)
                        else 0
                    )
                    result.update(
                        self._write_mse_fragment(
                            current_segment_url,
                            session_id=str(pe_session["session_id"]),
                            sequence=stream_sequence,
                        )
                    )
                    if published_replace_frames > 0:
                        result["stream_replace_start_seconds"] = (
                            previous_frames - published_replace_frames
                        ) / int(result["fps"])
                    if previous_video_url:
                        result["video_url"] = self._concat_session_video(
                            str(previous_video_url),
                            str(result["video_url"]),
                            session_id=str(pe_session["session_id"]),
                            request_id=str(result["id"]),
                            replace_tail_frames=published_replace_frames,
                            fps=int(result["fps"]),
                            previous_frame_count=previous_frames,
                        )
                    result["playback_start_seconds"] = (
                        previous_frames
                        + (replace_tail_frames if not previous_tail_is_published else 0)
                    ) / int(result["fps"])
                    result["segment_frames"] = visible_new_frames
                    result["frames"] = previous_frames + visible_new_frames
                    result["duration_seconds"] = result["frames"] / int(result["fps"])
                    result_history = list(pe_session.get("result_history") or [])
                    if not result_history and isinstance(previous_result, dict):
                        # Migrate sessions created before result_history existed.
                        result_history.append(dict(previous_result))
                    result_history.append(dict(result))
                    pe_session["result_history"] = result_history
                    pe_session["video_url"] = result.get("video_url")
                    pe_session["last_result"] = result
                    store.put_session(
                        str(pe_session["session_id"]),
                        pe_session,
                    )
                    result = {
                        **result,
                        "prompt_enhanced": True,
                        "pe_request_id": pe_request_id,
                    }
                store.patch(
                    job_id,
                    status="succeeded",
                    result=result,
                    video_url=result["video_url"],
                    error=None,
                )
                return

            if kind == "rollback":
                session_id = str(
                    (job.get("payload") or {}).get("session_id") or ""
                ).strip()
                session = store.get_session(session_id)
                if not session:
                    raise ValueError("interactive session does not exist")

                result_history = list(session.get("result_history") or [])
                if not result_history:
                    last_result = session.get("last_result")
                    if isinstance(last_result, dict):
                        result_history = [last_result]
                if not result_history:
                    raise ValueError("interactive session has no segment to roll back")

                cache_result = self.rollback_session(session_id)
                result_history.pop()
                messages = list(session.get("messages") or [])
                if len(messages) >= 2:
                    del messages[-2:]
                dit_prompts = list(session.get("dit_prompts") or [])
                if dit_prompts:
                    dit_prompts.pop()
                frame_lengths = list(session.get("segment_frame_lengths") or [])
                if frame_lengths:
                    frame_lengths.pop()

                if result_history:
                    previous_result = dict(result_history[-1])
                    previous_prompt = json.loads(dit_prompts[-1])
                    previous_segment = previous_prompt["segments"][-1]
                    session.update(
                        {
                            "previous_end": previous_segment["end_time"],
                            "messages": messages,
                            "dit_prompts": dit_prompts,
                            "segment_frame_lengths": frame_lengths,
                            "video_url": previous_result.get("video_url"),
                            "last_result": previous_result,
                            "result_history": result_history,
                            "branch_revision": (
                                int(session.get("branch_revision", 0) or 0) + 1
                            ),
                            "rollback_pending": True,
                            "updated_at": time.time(),
                        }
                    )
                    store.put_session(session_id, session)
                    playback = {
                        "video_url": previous_result.get("video_url"),
                        "frames": int(previous_result.get("frames", 0)),
                        "fps": int(previous_result.get("fps", 24)),
                        "stream_init_url": previous_result.get("stream_init_url"),
                        "stream_segment_url": previous_result.get("stream_segment_url"),
                        "stream_sequence": previous_result.get("stream_sequence"),
                        "stream_mime_type": previous_result.get("stream_mime_type"),
                    }
                else:
                    store.delete_session(session_id)
                    playback = {
                        "video_url": None,
                        "frames": 0,
                        "fps": int(self.config.inference.video_fps),
                        "stream_init_url": None,
                        "stream_segment_url": None,
                        "stream_sequence": None,
                        "stream_mime_type": None,
                    }
                result = {
                    **cache_result,
                    **playback,
                    "session_id": session_id,
                }
                store.patch(
                    job_id,
                    status="succeeded",
                    result=result,
                    **result,
                    error=None,
                )
                return

            if kind == "session_release":
                session_id = str(
                    (job.get("payload") or {}).get("session_id") or ""
                ).strip()
                if not session_id:
                    raise ValueError("session_id is required")
                result = self.release_session(session_id)
                store.release_session(
                    session_id,
                    worker_id=str(job.get("worker_id") or "") or None,
                )
                store.patch(
                    job_id,
                    status="succeeded",
                    result=result,
                    **result,
                    error=None,
                )
                return

            raise ValueError(f"unsupported queued job kind: {kind!r}")
        except UnsupportedStreamingRollbackError as exc:
            # No state was changed: retain KV/RNG/decoder state and the worker
            # assignment so rejecting an undo does not end the user's world.
            store.patch(
                job_id, status="failed", error=str(exc),
                error_code="UNSUPPORTED_STREAMING_ROLLBACK", recoverable=True,
            )
            logger.warning(f"[{job_id}] {exc}")
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"[{job_id}] queued {kind} job failed")
            error_code = _classify_inference_error(exc)
            session_id = JobStore._job_session_id(job)
            session_note = ""
            if session_id:
                store.release_session(
                    session_id,
                    worker_id=str(job.get("worker_id") or "") or None,
                )
                session_note = (
                    " The interactive session was invalidated so a later "
                    "request cannot silently continue from incomplete state."
                )
            error_message = f"{exc}{session_note}"
            store.patch(
                job_id,
                status="failed",
                error=error_message,
                error_code=error_code,
                recoverable=True,
            )
            store.record_error(
                code=error_code,
                message=error_message,
                job_id=job_id,
                worker_id=str(job.get("worker_id") or "") or None,
                recoverable=True,
            )
            try:
                self._reset_pipeline_state()
            except Exception:
                logger.exception(
                    f"[{job_id}] failed to reset request state after error"
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def queue_worker_loop(
        self,
        store: JobStore,
        *,
        poll_interval: float,
        heartbeat_interval: float,
        collective_keepalive_interval: float,
        max_concurrent_jobs: int,
        min_free_gpu_memory_bytes: int,
        admission_settle_seconds: float,
        session_idle_timeout: float,
        long_session_frame_threshold: int,
        max_long_sessions: int,
        exclusive_long_session: bool,
    ) -> None:
        """Claim jobs serially inside one replica; replicas run concurrently."""
        if self.rank != 0:
            raise RuntimeError("queue_worker_loop is rank-0 only")

        worker_id = (
            os.environ.get("NOMAS_WORKER_ID", "").strip()
            or f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        )
        worker_slot = (
            os.environ.get("NOMAS_WORKER_SLOT", "").strip()
            or None
        )
        recovered = store.reset_worker(worker_id)
        if any(recovered.values()):
            logger.warning(f"Recovered stale worker state: {recovered}")

        stop_event = threading.Event()

        def worker_resources() -> Dict[str, Any]:
            if not torch.cuda.is_available():
                return {
                    "slot": worker_slot,
                    "device": str(self.device),
                }
            device_index = self.device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
            return {
                "slot": worker_slot,
                "device": str(self.device),
                "cuda_visible_devices": os.environ.get(
                    "CUDA_VISIBLE_DEVICES",
                    "",
                ),
                "gpu_memory_free_bytes": int(free_bytes),
                "gpu_memory_total_bytes": int(total_bytes),
                "gpu_memory_allocated_bytes": int(
                    torch.cuda.memory_allocated(device_index)
                ),
                "gpu_memory_reserved_bytes": int(
                    torch.cuda.memory_reserved(device_index)
                ),
            }

        def heartbeat() -> None:
            while not stop_event.is_set():
                store.heartbeat(
                    worker_id,
                    "busy" if self.busy else "ready",
                    self.public_config(),
                    active_session_id=self._active_session_id,
                    resources=worker_resources(),
                )
                stop_event.wait(heartbeat_interval)

        heartbeat_thread = threading.Thread(
            target=heartbeat,
            name="gpu-worker-heartbeat",
            daemon=True,
        )
        heartbeat_thread.start()
        logger.info(f"GPU queue worker ready: id={worker_id}, db={store.path}")
        last_collective_at = time.monotonic()
        last_session_activity_at = time.monotonic()
        try:
            while not self.shutdown_started:
                resources = worker_resources()
                # Optional fallback for plain GPU caches. The default Hy-Nomas
                # deployment instead uses PhaseSwappedKVCache, which parks
                # exact BF16 KV on CPU around the PixelUMM decode phase.
                self._maybe_offload_interactive_kv_cache()
                job = store.claim_next(
                    worker_id,
                    active_session_id=self._active_session_id,
                    max_concurrent_jobs=max_concurrent_jobs,
                    available_free_bytes=resources.get("gpu_memory_free_bytes"),
                    min_free_bytes=min_free_gpu_memory_bytes,
                    admission_settle_seconds=admission_settle_seconds,
                    long_session_frame_threshold=(long_session_frame_threshold),
                    max_long_sessions=max_long_sessions,
                    exclusive_long_session=exclusive_long_session,
                )
                if job is None:
                    if (
                        self._active_session_id
                        and session_idle_timeout > 0
                        and time.monotonic() - last_session_activity_at
                        >= session_idle_timeout
                    ):
                        expired_session_id = self._active_session_id
                        self.release_session(expired_session_id)
                        store.release_session(
                            expired_session_id,
                            worker_id=worker_id,
                        )
                        last_session_activity_at = time.monotonic()
                        logger.info(
                            "Released idle interactive session "
                            f"{expired_session_id!r} after "
                            f"{session_idle_timeout:g}s"
                        )
                        continue
                    if (
                        self.world_size > 1
                        and time.monotonic() - last_collective_at
                        >= collective_keepalive_interval
                    ):
                        self._broadcast_command({"command": IDLE_COMMAND})
                        last_collective_at = time.monotonic()
                        logger.info(
                            "Sent distributed idle keepalive on the CPU command group"
                        )
                    time.sleep(poll_interval)
                    continue
                logger.info(f"[{job['id']}] claimed queued job kind={job['kind']}")
                self.execute_queue_job(store, job)
                if self._active_session_id:
                    last_session_activity_at = time.monotonic()
                last_collective_at = time.monotonic()
        finally:
            stop_event.set()
            heartbeat_thread.join(timeout=max(heartbeat_interval * 2, 1.0))
            store.heartbeat(
                worker_id,
                "offline",
                self.public_config(),
                active_session_id=self._active_session_id,
                resources=worker_resources(),
            )

    def _save_output(
        self,
        output: torch.Tensor,
        params: Dict[str, Any],
        generation_seconds: float,
    ) -> Dict[str, Any]:
        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise RuntimeError(
                "imageio is required to encode MP4 output. Install the project "
                "dependencies before starting this server."
            ) from exc

        video_tensor = output[0]
        video_np = (
            video_tensor.permute(1, 2, 3, 0).cpu().numpy()
            if video_tensor.dtype == torch.uint8
            else video_tensor.permute(1, 2, 3, 0)
            .mul(255)
            .clamp(0, 255)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
        fps = int(self.config.inference.video_fps)
        carry_frames = min(
            max(0, int(params.get("stable_prefix_carry_frames") or 0)),
            len(video_np),
        )
        burn_overlays = video_overlays_enabled()
        if burn_overlays and carry_frames:
            carry_total_frames = max(
                carry_frames,
                int(params.get("stable_prefix_carry_total_frames") or carry_frames),
            )
            video_np[:carry_frames] = burn_video_overlays(
                video_np[:carry_frames],
                params.get("carry_camera_trajectory"),
                fps,
                user_prompts=[params.get("carry_user_prompt") or ""],
                segment_boundaries=[0, carry_frames],
                segment_has_camera=[bool(params.get("carry_has_camera"))],
                timeline_frame_offset=max(
                    0,
                    carry_total_frames - carry_frames,
                ),
                timeline_frame_count=carry_total_frames,
            )
        if burn_overlays and carry_frames < len(video_np):
            video_np[carry_frames:] = burn_video_overlays(
                video_np[carry_frames:],
                params.get("camera_trajectory"),
                fps,
                user_prompts=params.get("user_prompts"),
                segment_boundaries=params.get("segment_boundaries"),
                segment_has_camera=params.get("segment_has_camera"),
                timeline_frame_offset=0,
                timeline_frame_count=int(params["num_frames"]),
            )

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_id = re.sub(r"[^A-Za-z0-9_-]+", "_", params["request_id"])
        filename = f"{safe_id}_{timestamp}_seed{params['seed']}.mp4"
        video_path = self.output_dir / filename

        encode_started = time.perf_counter()
        local_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f"{safe_id}_",
                suffix=".mp4",
                delete=False,
            ) as handle:
                local_path = Path(handle.name)
            imageio.mimwrite(
                str(local_path),
                video_np,
                fps=fps,
                codec="libx264",
                pixelformat="yuv420p",
                output_params=["-movflags", "+faststart"],
            )
            _publish_local_file(local_path, video_path)
        finally:
            if local_path is not None:
                local_path.unlink(missing_ok=True)
        encode_seconds = time.perf_counter() - encode_started

        frame_count = int(video_np.shape[0])
        metadata = {
            **{key: value for key, value in params.items() if key != "command"},
            "rollout_type": self.rollout_type,
            "fps": fps,
            "frame_count": frame_count,
            "duration_seconds": frame_count / fps,
            "generation_seconds": generation_seconds,
            "encoding_seconds": encode_seconds,
            "video_filename": filename,
        }
        metadata_path = video_path.with_suffix(".json")
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logger.info(
            f"[{params['request_id']}] complete: generation={generation_seconds:.2f}s, "
            f"encoding={encode_seconds:.2f}s, output={video_path}"
        )
        return {
            "id": params["request_id"],
            "status": "succeeded",
            "rollout_type": self.rollout_type,
            # Relative URLs preserve the deployment's reverse-proxy subpath.
            "video_url": f"outputs/{filename}",
            "metadata_url": f"outputs/{metadata_path.name}",
            "seed": params["seed"],
            "width": int(video_np.shape[2]),
            "height": int(video_np.shape[1]),
            "frames": frame_count,
            "fps": fps,
            "duration_seconds": frame_count / fps,
            "generation_seconds": generation_seconds,
            "encoding_seconds": encode_seconds,
            "timing_source": params["timing_source"],
            "segment_durations_seconds": params["segment_durations_seconds"],
        }

    def worker_loop(self) -> None:
        """Wait for rank-0 HTTP requests and join every collective inference."""
        if self.rank == 0:
            raise RuntimeError("rank 0 must run the HTTP server, not worker_loop")
        logger.info(f"[rank {self.rank}] entering {self.rollout_type} request loop")
        try:
            while True:
                try:
                    command = self._broadcast_command(None)
                    if command.get("command") == SHUTDOWN_COMMAND:
                        logger.info(f"[rank {self.rank}] shutdown received")
                        break
                    elif command.get("command") == IDLE_COMMAND:
                        continue
                    elif command.get("command") == ROLLBACK_COMMAND:
                        self._rollback_session_from_received(command)
                    elif command.get("command") == RELEASE_SESSION_COMMAND:
                        self._release_session_from_received(command)
                    else:
                        self._collective_generate_from_received(command)
                except Exception:
                    logger.exception(f"[rank {self.rank}] generation failed")
                    self._reset_pipeline_state()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        finally:
            self._release_pipeline()

    def _collective_generate_from_received(
        self, params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Execute a command that a worker has already received.

        ``_collective_generate`` normally performs the broadcast itself. Worker
        ranks receive first so they can inspect the shutdown command without
        causing a second broadcast.
        """
        if params.get("command") != GENERATE_COMMAND:
            raise RuntimeError(f"unsupported command: {params.get('command')!r}")

        request_id = params["request_id"]
        logger.info(f"[{request_id}] worker rank joining generation")
        with open_dict(self.config):
            internal_rollout = (
                "one_shot" if self.rollout_type == "oneshot" else "seg_ar"
            )
            self.config.inference.rollout_type = internal_rollout
            self.config.rollout_type = internal_rollout
            self.config.inference.solver_type = params["solver_type"]
        self._prepare_session_state(params)
        output = self._run_pipeline(params)
        del output
        return None

    def _release_pipeline(self) -> None:
        """Release request-local state and the resident model on this rank."""
        pipeline = getattr(self, "pipeline", None)
        if pipeline is not None:
            try:
                self._reset_pipeline_state()
            except Exception:
                logger.exception(f"[rank {self.rank}] failed to reset pipeline state")
            self.pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def shutdown(self) -> None:
        """Gracefully stop workers and release the resident pipeline.

        The operation is idempotent. On rank 0 it waits for any active request
        to finish, broadcasts one shutdown command to every worker rank, and
        only then frees model state.
        """
        with self._shutdown_lock:
            if self._shutdown_started:
                return
            self._shutdown_started = True

        if self.rank == 0:
            logger.info("Graceful shutdown requested; waiting for active request")
            with self._lock:
                if self.world_size > 1:
                    try:
                        self._broadcast_command({"command": SHUTDOWN_COMMAND})
                        logger.info("Shutdown command delivered to worker ranks")
                    except Exception:
                        logger.exception("failed to broadcast worker shutdown")

        self._release_pipeline()
        logger.info(f"[rank {self.rank}] model resources released")


def parse_server_args() -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="seg_ar/oneshot HTTP inference server",
        epilog=(
            "Defaults to configs/experiment/inference.yaml. "
            "Select another experiment with the same Hydra syntax as "
            "hymm/inference.py: +experiment=<path>. "
            "Other unknown arguments are forwarded as Hydra overrides."
        ),
    )
    parser.add_argument("--dit-weight", help="Override DiT checkpoint path")
    parser.add_argument(
        "--replay-manifest",
        help="Headless recorded-parameter replay; no HTTP server or prompt enhancer.",
    )
    parser.add_argument(
        "--rollout-type",
        choices=("seg_ar", "oneshot"),
        help=(
            "seg_ar reads request.prompt; oneshot reads the text file path "
            "provided in request.subject."
        ),
    )
    parser.add_argument(
        "--text-encoder-model",
        help="Hugging Face model ID or local model/tokenizer directory",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--output-dir",
        help="Override inference.save_path from the experiment config.",
    )
    parser.add_argument(
        "--frontend-dir",
        default=str(PROJECT_ROOT / "apps" / "frontend" / "dist"),
        help="Built frontend directory to serve at /. Empty disables it.",
    )
    parser.add_argument(
        "--cors-origin",
        action="append",
        default=[],
        help="Allowed browser origin; repeatable. Defaults to '*'.",
    )
    parser.add_argument(
        "--shutdown-token",
        default=os.environ.get("NOMAS_SHUTDOWN_TOKEN", ""),
        help=(
            "Enable POST /v1/shutdown with this bearer token. "
            "Defaults to $NOMAS_SHUTDOWN_TOKEN; empty disables the endpoint."
        ),
    )
    parser.add_argument(
        "--dry-run-source",
        help="Override inference.prompt for the optional POST /v1/dry-run.",
    )
    parser.add_argument(
        "--worker-only",
        action="store_true",
        help="Run only the durable GPU queue worker; do not bind an HTTP port.",
    )
    parser.add_argument(
        "--job-db",
        default="/tmp/nomas_seg_ar_jobs.json",
        help="SQLite queue shared with apps/seg_ar_gateway.py.",
    )
    parser.add_argument("--worker-poll-interval", type=float, default=0.5)
    parser.add_argument("--worker-heartbeat-interval", type=float, default=2.0)
    parser.add_argument(
        "--worker-collective-keepalive-interval",
        type=float,
        default=300.0,
        help=(
            "Seconds between no-op distributed broadcasts while the queue is "
            "idle. Must remain below the process-group NCCL timeout."
        ),
    )
    parser.add_argument(
        "--worker-max-concurrent-jobs",
        type=int,
        default=int(
            os.environ.get("NOMAS_MAX_CONCURRENT_JOBS", "1")
        ),
        help=(
            "Global running-job cap shared by all worker replicas using the "
            "same job store. The backward-compatible default is 1."
        ),
    )
    parser.add_argument(
        "--worker-min-free-gpu-memory-gib",
        type=float,
        default=float(
            os.environ.get("NOMAS_MIN_FREE_GPU_MEMORY_GIB", "0")
        ),
        help=(
            "Do not admit a new job on this replica when rank 0 has less than "
            "this much free GPU memory. Zero disables the guard."
        ),
    )
    parser.add_argument(
        "--worker-admission-settle-seconds",
        type=float,
        default=float(
            os.environ.get("NOMAS_ADMISSION_SETTLE_SECONDS", "2")
        ),
        help=(
            "Delay another replica briefly after a job starts so allocator "
            "growth is visible before the next admission decision."
        ),
    )
    parser.add_argument(
        "--worker-session-idle-timeout",
        type=float,
        default=float(
            os.environ.get("NOMAS_SESSION_IDLE_TIMEOUT", "0")
        ),
        help=(
            "Release a sticky interactive session after this many idle "
            "seconds. Zero preserves the original indefinite session lifetime."
        ),
    )
    parser.add_argument(
        "--worker-long-session-threshold-seconds",
        type=float,
        default=float(os.environ.get("NOMAS_LONG_SESSION_THRESHOLD_SECONDS", "0")),
        help=(
            "Treat a sticky session at or above this cumulative duration as "
            "long. Zero disables the long-session admission guard."
        ),
    )
    parser.add_argument(
        "--worker-max-long-sessions",
        type=int,
        default=int(os.environ.get("NOMAS_MAX_LONG_SESSIONS", "0")),
        help=(
            "Maximum resident long sessions across replicas. Zero disables the guard."
        ),
    )
    parser.add_argument(
        "--worker-exclusive-long-session",
        action=argparse.BooleanOptionalAction,
        default=(
            os.environ.get(
                "NOMAS_EXCLUSIVE_LONG_SESSION",
                "0",
            )
            .strip()
            .lower()
            in {"1", "true", "yes", "on"}
        ),
        help=(
            "When true, one resident long session reserves all shared GPUs. "
            "Enable for replicas that overlap the same physical devices; "
            "disable for disjoint GPU groups."
        ),
    )
    args, hydra_args = parser.parse_known_args()
    return args, hydra_args


def initialize_runtime(
    server_args: argparse.Namespace,
    hydra_args: List[str],
) -> Tuple[DictConfig, torch.device, int, int, Any, Any]:
    config = load_demo_config(hydra_args)
    apply_server_overrides(config, server_args)
    from hymm.distributed.parallel_states import (
        ParallelDims,
        initialize_sequence_parallel_state_v2,
    )
    from hymm.inference import (
        _ensure_dit_weight_safetensors,
        _validate_inference_config,
    )

    _validate_inference_config(config)

    rollout = str(config.inference.rollout_type)
    if rollout not in {"seg_ar", "one_shot"}:
        raise ValueError(f"Online demo requires seg_ar or one_shot, got {rollout!r}")
    server_args.rollout_type = "oneshot" if rollout == "one_shot" else "seg_ar"
    server_args.output_dir = str(config.inference.save_path or config.output_dir)
    server_args.dry_run_source = config.inference.prompt

    if not config.dit_weight:
        raise ValueError("dit_weight is required")
    if not config.text_encoder.model_name:
        raise ValueError(
            "Set model.text_encoder.model_name to a model ID or local directory"
        )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    sp_size = getattr(config.parallelism, "sp_size", 1)
    tp_size = getattr(config.parallelism, "tp_size", 1)
    from hymm.distributed.tensor_parallel import validate_parallel_sizes
    model_parallel_size = validate_parallel_sizes(sp_size, tp_size, world_size)

    if os.environ.get("NOMAS_BIND_LOCAL_NUMA", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    } and hasattr(os, "sched_setaffinity"):
        worker_slot = int(os.environ.get("NOMAS_WORKER_SLOT", "0") or 0)
        gpus_per_numa = max(1, world_size // 2)
        numa_node = min(local_rank // gpus_per_numa, 1)
        rank_in_numa = local_rank % gpus_per_numa
        replicas = max(
            1,
            int(os.environ.get("NOMAS_WORKER_REPLICAS", "1") or 1),
        )
        physical_cores_per_numa = 96
        logical_sibling_offset = 192
        groups_per_numa = gpus_per_numa * replicas
        cores_per_group = max(
            1,
            physical_cores_per_numa // groups_per_numa,
        )
        group = (worker_slot % replicas) * gpus_per_numa + rank_in_numa
        physical_start = numa_node * physical_cores_per_numa + group * cores_per_group
        physical_end = min(
            (numa_node + 1) * physical_cores_per_numa,
            physical_start + cores_per_group,
        )
        cpu_affinity = set(range(physical_start, physical_end))
        cpu_affinity.update(
            cpu + logical_sibling_offset for cpu in range(physical_start, physical_end)
        )
        if cpu_affinity:
            os.sched_setaffinity(0, cpu_affinity)

    if world_size != model_parallel_size:
        raise ValueError(
            "This server hosts exactly one model-parallel replica, so "
            f"WORLD_SIZE ({world_size}) must equal sp_size * tp_size ({model_parallel_size})."
        )

    world_mesh = None
    command_group = None
    if world_size > 1:
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=-1,
            sp=sp_size,
            tp=tp_size,
            pp=1,
            world_size=world_size,
        )
        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(seconds=3600),
        )
        torch.cuda.set_device(local_rank)
        command_group = dist.new_group(
            ranks=list(range(world_size)),
            backend="gloo",
            timeout=datetime.timedelta(seconds=3600),
        )
        world_mesh = parallel_dims.build_mesh(device_type="cuda")
        initialize_sequence_parallel_state_v2(
            parallel_dims,
            world_mesh,
            bool(
                getattr(
                    config.parallelism,
                    "use_dynamic_ring_attention",
                    False,
                )
            ),
        )
    elif torch.cuda.is_available():
        torch.cuda.set_device(0)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    if config.inference.dit_weight_quantization == "none":
        config.dit_weight = _ensure_dit_weight_safetensors(
            str(config.dit_weight),
            rank=rank,
            world_size=world_size,
        )
    else:
        # W8 checkpoints carry INT8 payloads and scales, in shards or one file.
        # Keep their path intact for the quantized loader, as offline inference does.
        from hymm.models.dit.modules.int8_stored_bf16_linear import resolve_w8_bf16_checkpoint
        resolve_w8_bf16_checkpoint(config.inference, config.dit_weight)
    return config, device, rank, world_size, world_mesh, command_group


def create_app(
    service: GenerationService,
    frontend_dir: Optional[Path],
    cors_origins: List[str],
    shutdown_token: str,
    request_server_shutdown: Callable[[], None],
):
    @asynccontextmanager
    async def lifespan(_: Any):
        yield
        service.shutdown()

    app = FastAPI(
        title="Hy-Nomas seg_ar inference server",
        version="1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "busy": service.busy,
            "shutdown_started": service.shutdown_started,
            "rollout_type": service.rollout_type,
            "prompt_source": (
                "subject_file" if service.rollout_type == "oneshot" else "prompt"
            ),
            "rank": service.rank,
            "world_size": service.world_size,
        }

    @app.get("/v1/config")
    def get_config() -> Dict[str, Any]:
        return service.public_config()

    @app.post("/v1/generate")
    async def generate(request: GenerateRequest) -> Dict[str, Any]:
        # Keep the collective model call on uvicorn's main thread. Declaring
        # this route as a normal ``def`` would move it into Starlette's worker
        # thread pool, which is undesirable for CUDA/NCCL collectives.
        try:
            result = service.generate(request)
            if result is None:
                raise RuntimeError("rank 0 received no generation result")
            return result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except torch.cuda.OutOfMemoryError as exc:
            raise HTTPException(
                status_code=507,
                detail="GPU out of memory; reduce resolution, frames, or steps",
            ) from exc
        except Exception as exc:
            logger.exception("generation request failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/v1/dry-run", status_code=202)
    async def start_dry_run() -> Dict[str, Any]:
        try:
            return service.start_dry_run()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("failed to start dry run")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/v1/dry-run/{job_id}")
    async def get_dry_run(job_id: str) -> Dict[str, Any]:
        try:
            return service.get_dry_run(job_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="dry-run job not found"
            ) from exc

    @app.post("/v1/shutdown", status_code=202)
    async def shutdown(
        background_tasks: BackgroundTasks,
        authorization: Optional[str] = Header(default=None),
    ) -> Dict[str, str]:
        """Ask uvicorn to exit after returning this response.

        Uvicorn's lifespan shutdown then calls ``service.shutdown()``, which
        broadcasts SHUTDOWN to the nonzero ranks before destroying the process
        group. The endpoint is disabled unless a token was configured.
        """
        if not shutdown_token:
            raise HTTPException(status_code=404, detail="shutdown endpoint disabled")
        scheme, _, supplied = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not secrets.compare_digest(
            supplied,
            shutdown_token,
        ):
            raise HTTPException(status_code=401, detail="invalid shutdown token")
        if service.shutdown_started:
            return {"status": "already_stopping"}

        background_tasks.add_task(request_server_shutdown)
        return {"status": "stopping"}

    app.mount(
        "/outputs",
        StaticFiles(directory=str(service.output_dir)),
        name="outputs",
    )
    if frontend_dir is not None and frontend_dir.is_dir():
        app.mount(
            "/",
            StaticFiles(directory=str(frontend_dir), html=True),
            name="frontend",
        )
        logger.info(f"Serving frontend from {frontend_dir}")
    else:
        logger.info("Frontend directory not found; API-only mode")
    return app


def configure_logging(rank: int) -> None:
    logger.configure(extra={"rank": rank})
    logger.remove()
    level = "INFO" if rank == 0 else "WARNING"
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:^8} | "
            "[Rank {extra[rank]}]: {message}"
        ),
    )


def main() -> None:
    server_args, hydra_args = parse_server_args()
    replay_manifest = None
    if server_args.replay_manifest:
        from apps.seg_ar_replay import read_json, validate_params

        replay_manifest = read_json(server_args.replay_manifest)
        if replay_manifest.get("schema") != "seg_ar_recorded_params_replay_v1":
            raise ValueError("Unsupported demo replay manifest")
        if server_args.worker_only or server_args.rollout_type not in {None, "seg_ar"}:
            raise ValueError("Replay requires seg_ar and cannot use --worker-only")
        for session in replay_manifest["sessions"]:
            for index, turn in enumerate(session["turns"]):
                validate_params(turn["params"], session["session_id"], index == 0)
    (
        config,
        device,
        rank,
        world_size,
        world_mesh,
        command_group,
    ) = initialize_runtime(
        server_args,
        hydra_args,
    )
    if replay_manifest is not None and server_args.rollout_type != "seg_ar":
        raise ValueError("Replay requires seg_ar")
    configure_logging(rank)

    output_dir = Path(server_args.output_dir).expanduser().resolve()
    service = GenerationService(
        config=config,
        device=device,
        output_dir=output_dir,
        rank=rank,
        world_size=world_size,
        world_mesh=world_mesh,
        command_group=command_group,
        rollout_type=server_args.rollout_type,
        dry_run_source=(
            Path(server_args.dry_run_source).expanduser().resolve()
            if server_args.dry_run_source
            else None
        ),
    )

    try:
        if replay_manifest is not None and rank == 0:
            from apps.seg_ar_replay import replay_sessions
            from omegaconf import OmegaConf

            # Full resolved config is retained for future reproducibility audits.
            config_path = output_dir / (
                f"replay_config_group_{int(os.environ.get('GLOBAL_NODE_RANK', '0')):03d}.yaml"
            )
            with config_path.open("x", encoding="utf-8") as handle:
                handle.write(OmegaConf.to_yaml(config, resolve=True))
            replay_sessions(
                service,
                replay_manifest,
                group_index=int(os.environ.get("GLOBAL_NODE_RANK", "0")),
                group_count=int(os.environ.get("GLOBAL_NUM_NODES", "1")),
            )
        elif rank == 0 and server_args.worker_only:
            service.queue_worker_loop(
                JobStore(server_args.job_db),
                poll_interval=max(server_args.worker_poll_interval, 0.05),
                heartbeat_interval=max(
                    server_args.worker_heartbeat_interval,
                    0.25,
                ),
                collective_keepalive_interval=max(
                    server_args.worker_collective_keepalive_interval,
                    1.0,
                ),
                max_concurrent_jobs=max(
                    server_args.worker_max_concurrent_jobs,
                    1,
                ),
                min_free_gpu_memory_bytes=max(
                    int(server_args.worker_min_free_gpu_memory_gib * (1024**3)),
                    0,
                ),
                admission_settle_seconds=max(
                    server_args.worker_admission_settle_seconds,
                    0.0,
                ),
                session_idle_timeout=max(
                    server_args.worker_session_idle_timeout,
                    0.0,
                ),
                long_session_frame_threshold=max(
                    int(
                        server_args.worker_long_session_threshold_seconds
                        * int(config.inference.video_fps)
                    ),
                    0,
                ),
                max_long_sessions=max(
                    server_args.worker_max_long_sessions,
                    0,
                ),
                exclusive_long_session=bool(server_args.worker_exclusive_long_session),
            )
        elif rank == 0:
            try:
                import uvicorn
            except ImportError as exc:
                raise RuntimeError(
                    "uvicorn is missing. Install the project web dependencies."
                ) from exc

            frontend_dir = (
                Path(server_args.frontend_dir).expanduser().resolve()
                if server_args.frontend_dir
                else None
            )
            shutdown_callback_holder: Dict[str, Callable[[], None]] = {
                "callback": lambda: None,
            }
            app = create_app(
                service,
                frontend_dir=frontend_dir,
                cors_origins=server_args.cors_origin,
                shutdown_token=server_args.shutdown_token,
                request_server_shutdown=lambda: shutdown_callback_holder["callback"](),
            )
            logger.info(
                f"Starting {server_args.rollout_type} web server on "
                f"http://{server_args.host}:{server_args.port}"
            )
            uvicorn_config = uvicorn.Config(
                app=app,
                host=server_args.host,
                port=server_args.port,
                workers=1,
                log_level="info",
            )
            uvicorn_server = uvicorn.Server(uvicorn_config)
            # Let the shutdown route finish its HTTP response before asking
            # uvicorn to enter the lifespan shutdown phase.
            shutdown_callback_holder["callback"] = lambda: setattr(
                uvicorn_server,
                "should_exit",
                True,
            )
            uvicorn_server.run()
        else:
            service.worker_loop()
    finally:
        if "service" in locals():
            service.shutdown()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
