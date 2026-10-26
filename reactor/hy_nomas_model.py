"""Runtime-independent Hy-Nomas model and CPU step contracts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

MAX_WORLD_FRAMES = 2880
"""Released interactive noise-bank horizon at the native 24 FPS timebase."""

if TYPE_CHECKING:
    from hy_nomas_backend import HyNomasBackend


@dataclass(frozen=True)
class ModelSettings:
    """Resolved native checkout, checkpoint paths and output canvas."""

    source: Path
    dit: Path
    decoder: Path
    text_encoder: Path
    height: int = 720
    width: int = 1280
    denoising_steps: int = 8
    history_frames: int | None = None
    history_sampling: str = "recent"
    precision: str = "bf16"
    cpu_offload: bool = False
    cache_int8_weights: bool = False
    decoder_batch_size: int = 3
    cutlass: Path = Path("/opt/cutlass")


@dataclass(frozen=True)
class WorldStart:
    """Immutable seed and segment length, sent until the new world is acknowledged."""

    seed: int
    segment_frames: int


@dataclass(frozen=True)
class HyNomasInput:
    """One native segment; camera arrays are float32 CPU [T,4,4]/[T,3,3]."""

    world_id: int
    """Identity chosen by the caller; a changed identity requests a fresh world."""
    start: WorldStart | None
    """Required for an unseen world; omitted after a successful acknowledgment."""
    prompt: str
    """Text or native structured JSON for this segment; may change within a world."""
    c2ws: np.ndarray | None
    """CPU float32 [T,4,4] camera-to-world poses, or None for text-only input."""
    intrinsics: np.ndarray | None
    """CPU float32 [T,3,3] normalized calibration, paired with c2ws."""
    translation_strength: float
    rotation_strength: float


@dataclass(frozen=True)
class HyNomasResult:
    """New stable RGB uint8 CPU [T,H,W,3] and completed-world facts."""

    frames: np.ndarray
    """New stable CPU uint8 [T,H,W,3] pixels; only rank zero returns video."""
    world_id: int
    """Identity now held by the model."""
    chunk_index: int
    """One-based count of successful generation calls in this world."""
    generated_frames: int
    """Cumulative pixel timeline, including the decoder's pending overlap."""
    emitted_frames: int
    """Cumulative stable pixels returned on rank zero."""
    complete: bool
    """True after the released 120-second interactive noise bank is consumed."""


class MissingWorld(ValueError):
    """The requested world has no initial seed and segment geometry."""


class RolloutComplete(ValueError):
    """The native interactive noise horizon has already been emitted."""


class HyNomasModel:
    """Own native history behind load/generate/reset, independent of serving.

    An unseen world_id needs WorldStart or raises MissingWorld. Successful calls
    acknowledge that identity and advance a one-based chunk_index. A completed
    world raises RolloutComplete until a new identity or reset. Backend failures
    propagate; the caller must end that failed rollout rather than retry it.
    """

    def __init__(self) -> None:
        self.rank = 0
        self.world_size = 1
        self._backend: HyNomasBackend | None = None
        self._world_id: int | None = None
        self._chunk_index = 0
        self._generated_frames = 0
        self._emitted_frames = 0
        self._complete = False

    def load(self, settings: ModelSettings, warmup: HyNomasInput | None = None) -> None:
        """Load ready assets in the worker; rank/world_size are injected by its owner."""
        from hy_nomas_backend import HyNomasBackend

        self._backend = HyNomasBackend(
            settings, rank=self.rank, world_size=self.world_size
        )
        if warmup is not None:
            self.generate(warmup)
            self.generate(replace(warmup, start=None))
            self.reset()

    def generate(self, input: HyNomasInput) -> HyNomasResult:
        """Generate one segment, retaining native visual/text/noise/decoder state."""
        if input.world_id != self._world_id:
            if input.start is None:
                raise MissingWorld("A fresh world requires its starting settings.")
            self.reset()
            self._backend.start(input.start)
            self._world_id = input.world_id
        if self._complete:
            raise RolloutComplete(
                "Start a fresh world after the native 120-second horizon."
            )
        frames, generated_frames, complete = self._backend.step(input)
        self._chunk_index += 1
        self._generated_frames += generated_frames
        self._emitted_frames += len(frames)
        self._complete = complete
        return HyNomasResult(
            frames=frames,
            world_id=input.world_id,
            chunk_index=self._chunk_index,
            generated_frames=self._generated_frames,
            emitted_frames=self._emitted_frames,
            complete=complete,
        )

    def reset(self) -> None:
        """Forget the world and its native caches while retaining loaded weights."""
        if self._backend is not None:
            self._backend.reset()
        self._world_id = None
        self._chunk_index = 0
        self._generated_frames = 0
        self._emitted_frames = 0
        self._complete = False
