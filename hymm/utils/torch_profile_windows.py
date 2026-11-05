"""Opt-in ``torch.profiler`` windows around whole seg_ar segments.

Profiling every denoise step of a long multi-segment rollout produces traces
that are too large to open, so this records only the segments named in
``HY_PROFILE_SEGMENTS`` while the rest of the run executes untouched.

The hook wraps ``transformer.forward_with_cache`` instead of reaching into the
pipeline loop, which keeps it independent of the segment/chunk bookkeeping in
``hymm/pipeline/causal_ar.py``. A segment is bounded by its cache write
(``update_cache=True``), so counting those calls yields the current segment
index. A window therefore spans that segment's denoise steps, its cache write,
and - because the VAE decode runs after the cache write but before the next
segment's first forward - that segment's VAE decode as well.

Everything is disabled unless ``HY_PROFILE_DIR`` is set, so importing or
installing this on a normal run costs one wrapper call per forward.

Environment variables
---------------------
``HY_PROFILE_DIR``       output directory; unset disables the hook entirely.
``HY_PROFILE_SEGMENTS``  comma-separated 0-based segment indices (default ``0``).
``HY_PROFILE_RANKS``     comma-separated global ranks to trace (default ``0``).
``HY_PROFILE_SHAPES``    record operator input shapes (default ``1``).
``HY_PROFILE_STACK``     record Python stacks; large traces (default ``0``).
``HY_PROFILE_MEMORY``    record allocator events; large traces (default ``0``).
``HY_PROFILE_TABLE_ROWS``rows kept in the key_averages tables (default ``80``).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from loguru import logger


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def _index_set(name: str, default: str) -> set[int]:
    raw = os.environ.get(name, default)
    return {int(piece) for piece in raw.replace(" ", "").split(",") if piece != ""}


class SegmentWindowProfiler:
    """Start/stop a profiler so each traced window holds exactly one segment."""

    def __init__(self, out_dir, segments, *, rank=0, record_shapes=True,
                 with_stack=False, profile_memory=False, table_rows=80):
        self.out_dir = Path(out_dir)
        self.segments = set(segments)
        self.rank = int(rank)
        self.table_rows = int(table_rows)
        self._profiler_kwargs = dict(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=record_shapes,
            with_stack=with_stack,
            profile_memory=profile_memory,
            with_flops=True,
        )
        self.segment_index = 0
        self.forward_calls = 0
        self._profiler = None
        self._active_segment = None
        self.windows = []

    # -- window control -----------------------------------------------------
    def _start(self, segment):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._profiler = torch.profiler.profile(**self._profiler_kwargs)
        self._profiler.start()
        self._active_segment = segment
        logger.info(f"[torch-profiler] window OPEN for segment {segment} (rank {self.rank})")

    def _stop(self):
        if self._profiler is None:
            return
        segment = self._active_segment
        profiler, self._profiler, self._active_segment = self._profiler, None, None
        profiler.stop()
        stem = f"segment{segment:02d}.rank{self.rank}"
        trace_path = self.out_dir / f"{stem}.chrome_trace.json.gz"
        try:
            profiler.export_chrome_trace(str(trace_path))
        except Exception as error:  # a failed export must not kill the rollout
            logger.warning(f"[torch-profiler] chrome trace export failed: {error!r}")
            trace_path = None
        tables = {}
        for label, sort_key in (("cuda_self", "self_cuda_time_total"),
                                ("cpu_self", "self_cpu_time_total")):
            try:
                averages = profiler.key_averages()
                text = averages.table(sort_by=sort_key, row_limit=self.table_rows)
            except Exception as error:
                logger.warning(f"[torch-profiler] {label} table failed: {error!r}")
                continue
            path = self.out_dir / f"{stem}.{label}.txt"
            path.write_text(text)
            tables[label] = str(path)
        self.windows.append(dict(segment=segment, rank=self.rank,
                                 trace=str(trace_path) if trace_path else None,
                                 tables=tables))
        logger.info(f"[torch-profiler] window CLOSED for segment {segment} -> {trace_path}")

    # -- hook ---------------------------------------------------------------
    def before_forward(self):
        """Open or close a window so the profiler matches the current segment."""
        segment = self.segment_index
        if segment in self.segments:
            if self._active_segment != segment:
                self._stop()
                self._start(segment)
        elif self._profiler is not None:
            self._stop()

    def after_forward(self, update_cache):
        self.forward_calls += 1
        if update_cache:
            # The cache write closes the segment; its VAE decode still runs
            # before the next forward, so the window stays open until then.
            self.segment_index += 1

    def finalize(self):
        """Close the trailing window (last segment has no following forward)."""
        self._stop()
        if self.windows:
            summary = self.out_dir / f"windows.rank{self.rank}.json"
            import json

            summary.write_text(json.dumps(
                dict(rank=self.rank, forward_calls=self.forward_calls,
                     segments_seen=self.segment_index, windows=self.windows),
                indent=2) + "\n")


def install(pipeline, rank):
    """Wrap ``pipeline.transformer.forward_with_cache``; return the profiler or None."""
    out_dir = os.environ.get("HY_PROFILE_DIR", "").strip()
    if not out_dir:
        return None
    if rank not in _index_set("HY_PROFILE_RANKS", "0"):
        return None
    transformer = getattr(pipeline, "transformer", None)
    original = getattr(transformer, "forward_with_cache", None)
    if original is None:
        logger.warning("[torch-profiler] transformer has no forward_with_cache; hook skipped")
        return None

    profiler = SegmentWindowProfiler(
        out_dir,
        _index_set("HY_PROFILE_SEGMENTS", "0"),
        rank=rank,
        record_shapes=_flag("HY_PROFILE_SHAPES", "1"),
        with_stack=_flag("HY_PROFILE_STACK", "0"),
        profile_memory=_flag("HY_PROFILE_MEMORY", "0"),
        table_rows=int(os.environ.get("HY_PROFILE_TABLE_ROWS", "80")),
    )

    def wrapped(*args, **kwargs):
        profiler.before_forward()
        try:
            return original(*args, **kwargs)
        finally:
            profiler.after_forward(bool(kwargs.get("update_cache", False)))

    transformer.forward_with_cache = wrapped
    logger.info(
        f"[torch-profiler] hook installed on rank {rank}: segments={sorted(profiler.segments)}, "
        f"shapes={profiler._profiler_kwargs['record_shapes']}, "
        f"stack={profiler._profiler_kwargs['with_stack']}, "
        f"memory={profiler._profiler_kwargs['profile_memory']} -> {out_dir}"
    )
    return profiler
