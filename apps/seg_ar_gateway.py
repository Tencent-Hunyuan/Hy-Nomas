"""Lightweight, always-on frontend/API process for the Hy-Nomas GPU job queue."""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import html
import inspect
import json
import logging
import math
import os
import re
import secrets
import shutil
import sys
import tempfile
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator
from urllib.parse import unquote, urlsplit

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from apps.camera_video_overlay import burn_shared_overlay_video
from apps.durable_file import replace_or_overwrite
from apps.kimi_k3_prompt_enhancer import (
    ALLOWED_PERSPECTIVES,
    DEFAULT_PROMPT_ENHANCER_MODEL,
    DEFAULT_PROMPT_ENHANCER_REASONING_EFFORT,
    PROMPT_ENHANCER_OPTIONS,
    PROMPT_ENHANCER_REASONING_EFFORTS,
    PROMPT_ENHANCER_SEED_MAX,
    PROMPT_ENHANCER_SEED_MIN,
    OnlinePromptEnhancer,
    PromptEnhancementError,
    build_local_fallback_prompt,
    build_minimal_camera_prompt,
    normalize_prompt_enhancer_model,
    normalize_prompt_enhancer_reasoning_effort,
    normalize_prompt_enhancer_seed,
    persist_pe_failure,
    persist_pe_turn,
)
from apps.seg_ar_attempt_archive import UserAttemptArchive
from apps.seg_ar_job_store import JobStore
from apps.seg_ar_resource_monitor import ResourceMonitor
from apps.seg_ar_demo_config import load_demo_config

logger = logging.getLogger("seg_ar_gateway")

_SHARE_OVERLAY_MAX_BYTES = 2_000_000
_SHARE_OVERLAY_MAX_TURNS = 128
_SHARE_CAMERA_MAX_FRAMES = 12_000
_SHARE_CAMERA_MAX_KEY_EVENTS = 24_000
_DOWNLOAD_OVERLAY_RENDERER_VERSION = 3
NUM_INFERENCE_STEPS_OPTIONS = (4, 6, 8)
DEFAULT_NUM_INFERENCE_STEPS = 8
_GALLERY_LOCK_TIMEOUT_SECONDS = 0.75
_GALLERY_LOCK_RETRY_SECONDS = 0.05
_GALLERY_RETRYABLE_ERRNOS = {
    errno.EAGAIN,
    errno.EBUSY,
    errno.EIO,
    getattr(errno, "ESTALE", 116),
    errno.EWOULDBLOCK,
}
_SHARE_CAMERA_KEYS = {
    "KeyW",
    "KeyA",
    "KeyS",
    "KeyD",
    "ArrowUp",
    "ArrowDown",
    "ArrowLeft",
    "ArrowRight",
}


class GalleryStorageBusyError(RuntimeError):
    """Raised when the shared Gallery index cannot be locked promptly."""


_GENERIC_GALLERY_TITLES = {"", "saved world"}
_GALLERY_TITLE_PREFIX_RE = re.compile(
    r"^(?:接下来|最后|首先|随后|然后)?(?:约)?\d+(?:\.\d+)?秒[，,:：\s]*"
)


def _gallery_title_snippet(value: object, *, limit: int = 48) -> str:
    """Turn a subject or direction prompt into a compact Gallery card title."""

    text = " ".join(str(value or "").split()).strip()
    text = _GALLERY_TITLE_PREFIX_RE.sub("", text)
    for marker in (
        "。镜头继续",
        "；镜头继续",
        ". The camera",
        "。使用第三人称",
    ):
        text = text.split(marker, 1)[0].strip()
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip("，,。.;；:： ") + "…"


def _gallery_item_title(raw_item: dict[str, Any]) -> str:
    explicit = " ".join(str(raw_item.get("title") or "").split()).strip()
    if explicit.casefold() not in _GENERIC_GALLERY_TITLES:
        return explicit[:96]
    for candidate in (
        raw_item.get("subject"),
        raw_item.get("direction"),
        raw_item.get("attributes"),
    ):
        title = _gallery_title_snippet(candidate)
        if title:
            return title
    return "Untitled world"


def _gallery_prompt_timeline(value: object) -> list[dict[str, Any]]:
    """Normalize prompt-only timeline metadata used by Gallery playback."""

    raw_turns = value.get("turns") if isinstance(value, dict) else value
    if not isinstance(raw_turns, list):
        return []
    turns: list[dict[str, Any]] = []
    for index, raw_turn in enumerate(raw_turns[:_SHARE_OVERLAY_MAX_TURNS]):
        if not isinstance(raw_turn, dict):
            continue
        prompt = str(raw_turn.get("prompt") or "").strip()[:8_000]
        try:
            start = float(raw_turn.get("start_seconds"))
            end = float(raw_turn.get("end_seconds"))
        except (TypeError, ValueError):
            continue
        if (
            not prompt
            or not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end <= start
        ):
            continue
        try:
            number = max(1, int(raw_turn.get("number") or index + 1))
        except (TypeError, ValueError):
            number = index + 1
        turns.append(
            {
                "number": number,
                "prompt": prompt,
                "start_seconds": start,
                "end_seconds": end,
            }
        )
    return turns


def _gallery_paths(gallery_dir: Path) -> tuple[Path, Path]:
    return (
        gallery_dir.parent / f".{gallery_dir.name}_items.json",
        gallery_dir.parent / f".{gallery_dir.name}_items.lock",
    )


def _read_gallery_index(
    index_path: Path,
    *,
    strict: bool,
) -> list[dict[str, Any]]:
    try:
        value = json.loads(index_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, json.JSONDecodeError):
        if strict:
            raise
        return []
    if not isinstance(value, list):
        if strict:
            raise ValueError("Gallery index must contain a JSON list")
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _write_gallery_index(
    gallery_dir: Path,
    index_path: Path,
    items: list[dict[str, Any]],
) -> None:
    with tempfile.NamedTemporaryFile(
        dir=str(gallery_dir),
        prefix=".nomas-gallery-items.",
        suffix=".json.tmp",
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as handle:
        json.dump(items, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary_index = Path(handle.name)
    try:
        replace_or_overwrite(temporary_index, index_path)
    finally:
        temporary_index.unlink(missing_ok=True)


@contextmanager
def _gallery_index_lock(gallery_dir: Path) -> Iterator[Path]:
    """Take the cross-process Gallery lock without an unbounded FUSE wait."""

    index_path, lock_path = _gallery_paths(gallery_dir)
    gallery_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + _GALLERY_LOCK_TIMEOUT_SECONDS
    lock = None
    while lock is None:
        candidate = None
        try:
            candidate = lock_path.open("a+", encoding="utf-8")
            fcntl.flock(
                candidate.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
            lock = candidate
        except OSError as exc:
            if candidate is not None:
                candidate.close()
            if (
                exc.errno not in _GALLERY_RETRYABLE_ERRNOS
                or time.monotonic() >= deadline
            ):
                raise GalleryStorageBusyError(
                    "Gallery index is busy; retry shortly"
                ) from exc
            time.sleep(_GALLERY_LOCK_RETRY_SECONDS)
    try:
        yield index_path
    finally:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        except OSError:
            logger.warning("Could not unlock Gallery index %s", lock_path)
        lock.close()


def _mutate_gallery_index(
    gallery_dir: Path,
    process_lock: threading.RLock,
    mutate: Callable[
        [list[dict[str, Any]]],
        tuple[list[dict[str, Any]], Any, bool],
    ],
) -> Any:
    with process_lock, _gallery_index_lock(gallery_dir) as index_path:
        try:
            items = _read_gallery_index(index_path, strict=True)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise GalleryStorageBusyError(
                "Gallery index could not be read safely; retry shortly"
            ) from exc
        updated_items, result, changed = mutate(items)
        if changed:
            _write_gallery_index(gallery_dir, index_path, updated_items)
        return result


def _gallery_public_item(raw_item: dict[str, Any]) -> dict[str, Any]:
    relative_video = str(raw_item.get("relative_video") or "")
    return {
        "id": str(raw_item.get("id") or ""),
        "title": _gallery_item_title(raw_item),
        "video": f"gallery/{relative_video}",
        "duration": str(raw_item.get("duration") or "—"),
        "perspective": str(raw_item.get("perspective") or "third-person"),
        "subject": str(raw_item.get("subject") or ""),
        "attributes": str(raw_item.get("attributes") or ""),
        "direction": str(raw_item.get("direction") or ""),
        "prompt_timeline": _gallery_prompt_timeline(raw_item.get("prompt_timeline")),
        "created_at": raw_item.get("created_at"),
    }


def read_gallery_items(gallery_dir: Path | None) -> list[dict[str, Any]]:
    if gallery_dir is None:
        return []
    index_path, _ = _gallery_paths(gallery_dir)
    try:
        value = _read_gallery_index(index_path, strict=False)
    except OSError as exc:
        logger.warning("Could not read Gallery index %s: %s", index_path, exc)
        return []
    items = []
    gallery_root = gallery_dir.resolve()
    for raw_item in value:
        if raw_item.get("hidden") is True:
            continue
        relative_video = str(raw_item.get("relative_video") or "")
        candidate = (gallery_root / relative_video).resolve()
        try:
            valid = (
                bool(relative_video)
                and candidate.is_relative_to(gallery_root)
                and candidate.suffix.lower() == ".mp4"
                and candidate.is_file()
                and candidate.stat().st_size > 0
            )
        except OSError:
            valid = False
        if valid:
            items.append(_gallery_public_item(raw_item))
    return sorted(
        items,
        key=lambda item: float(item.get("created_at") or 0),
        reverse=True,
    )


def read_hidden_gallery_item_ids(gallery_dir: Path | None) -> list[str]:
    if gallery_dir is None:
        return []
    index_path, _ = _gallery_paths(gallery_dir)
    value = _read_gallery_index(index_path, strict=False)
    return sorted(
        {
            str(item.get("id") or "").strip()
            for item in value
            if item.get("hidden") is True and str(item.get("id") or "").strip()
        }
    )


def _copy_gallery_video(source: Path, destination: Path) -> None:
    source_size = source.stat().st_size
    if source_size <= 0:
        raise ValueError("generated video is empty")
    try:
        destination_valid = (
            destination.is_file() and destination.stat().st_size == source_size
        )
    except OSError:
        destination_valid = False
    if destination_valid:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with source.open("rb") as source_handle, destination.open(
            "wb"
        ) as destination_handle:
            shutil.copyfileobj(
                source_handle,
                destination_handle,
                length=8 * 1024 * 1024,
            )
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
        actual_size = destination.stat().st_size
        if actual_size != source_size:
            raise OSError(
                "Gallery video copy was incomplete: "
                f"expected={source_size}, actual={actual_size}"
            )
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def add_gallery_item_from_source(
    gallery_dir: Path,
    source: Path,
    payload: dict[str, Any],
    *,
    process_lock: threading.RLock,
) -> dict[str, Any]:
    source = source.expanduser().resolve()
    if source.suffix.lower() != ".mp4" or not source.is_file():
        raise FileNotFoundError(source)

    perspective = str(payload.get("perspective") or "").strip()
    if perspective not in {*ALLOWED_PERSPECTIVES, "third-person"}:
        raise ValueError(
            "perspective must be first-person, second-person, "
            "third-person follow, or third-person observer"
        )
    subject = str(payload.get("subject") or "").strip()[:400]
    attributes = str(payload.get("attributes") or "").strip()[:1200]
    direction = str(payload.get("direction") or "").strip()[:1200]
    prompt_timeline = _gallery_prompt_timeline(payload.get("prompt_timeline"))
    title = _gallery_item_title(
        {
            "title": payload.get("title"),
            "subject": subject,
            "direction": direction,
            "attributes": attributes,
        }
    )
    try:
        duration_seconds = max(
            0.0,
            float(payload.get("duration_seconds") or 0),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("duration_seconds must be a number") from exc

    source_stat = source.stat()
    fingerprint = hashlib.sha256(
        (f"{source}:{source_stat.st_size}:{source_stat.st_mtime_ns}").encode("utf-8")
    ).hexdigest()[:20]
    item_id = f"saved-{fingerprint}"
    relative_video = f"favorites/{item_id}.mp4"
    destination = gallery_dir / relative_video
    stored = {
        "id": item_id,
        "title": title,
        "relative_video": relative_video,
        "duration": (
            f"{max(1, int(round(duration_seconds)))}s" if duration_seconds > 0 else "—"
        ),
        "perspective": perspective,
        "subject": subject,
        "attributes": attributes,
        "direction": direction,
        "prompt_timeline": prompt_timeline,
        "created_at": time.time(),
    }

    gallery_dir.mkdir(parents=True, exist_ok=True)

    def add(
        items: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], tuple[dict[str, Any], bool], bool]:
        existing = next(
            (item for item in items if str(item.get("id") or "") == item_id),
            None,
        )
        _copy_gallery_video(source, destination)
        if existing is not None:
            restored = {
                **stored,
                "created_at": existing.get("created_at") or stored["created_at"],
            }
            if existing.get("hidden") is not True and existing == restored:
                return items, (dict(existing), False), False
            updated = [
                restored if str(item.get("id") or "") == item_id else item
                for item in items
            ]
            return updated, (restored, existing.get("hidden") is True), True
        return [*items, stored], (stored, True), True

    saved, added = _mutate_gallery_index(
        gallery_dir,
        process_lock,
        add,
    )
    return {
        "item": _gallery_public_item(saved),
        "added": added,
    }


def remove_gallery_item(
    gallery_dir: Path,
    item_id: str,
    *,
    process_lock: threading.RLock,
    delete_media: bool = False,
) -> dict[str, Any]:
    clean_item_id = str(item_id or "").strip()
    if not clean_item_id:
        raise ValueError("gallery item id is required")

    def remove(
        items: list[dict[str, Any]],
    ) -> tuple[
        list[dict[str, Any]],
        tuple[dict[str, Any] | None, bool],
        bool,
    ]:
        removed = next(
            (item for item in items if str(item.get("id") or "") == clean_item_id),
            None,
        )
        if removed is not None and removed.get("hidden") is True:
            return items, (None, False), False
        hidden = (
            {**removed, "hidden": True}
            if removed is not None
            else {
                "id": clean_item_id,
                "hidden": True,
                "created_at": time.time(),
            }
        )
        updated = [
            hidden if str(item.get("id") or "") == clean_item_id else item
            for item in items
        ]
        if removed is None:
            updated.append(hidden)
        return updated, (dict(removed) if removed is not None else None, True), True

    removed, was_removed = _mutate_gallery_index(
        gallery_dir,
        process_lock,
        remove,
    )
    media_deleted = False
    if was_removed and delete_media and removed is not None:
        gallery_root = gallery_dir.resolve()
        relative_video = str(removed.get("relative_video") or "")
        candidate = (gallery_root / relative_video).resolve()
        if (
            relative_video
            and candidate.is_relative_to(gallery_root)
            and candidate.suffix.lower() == ".mp4"
        ):
            candidate.unlink(missing_ok=True)
            media_deleted = not candidate.exists()
    return {
        "removed": was_removed,
        "item": _gallery_public_item(removed) if removed is not None else None,
        "media_deleted": media_deleted,
    }


def normalize_num_inference_steps(
    value: object, default: int = DEFAULT_NUM_INFERENCE_STEPS
) -> int:
    if value is None:
        return default
    normalized = (
        value if isinstance(value, int) and not isinstance(value, bool) else None
    )
    options = sorted(set((*NUM_INFERENCE_STEPS_OPTIONS, default)))
    if normalized not in options:
        allowed = ", ".join(str(option) for option in options)
        raise ValueError(f"num_inference_steps must be one of {allowed}")
    return normalized


def _share_float(value: object, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def _normalize_share_overlay(value: object) -> Dict[str, Any]:
    if value is None:
        return {"version": 1, "turns": []}
    if not isinstance(value, dict):
        raise ValueError("overlay must be an object")
    try:
        serialized = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("overlay must be JSON serializable") from exc
    if len(serialized.encode("utf-8")) > _SHARE_OVERLAY_MAX_BYTES:
        raise ValueError("overlay metadata is too large")

    raw_turns = value.get("turns")
    if not isinstance(raw_turns, list):
        raise ValueError("overlay.turns must be an array")
    if len(raw_turns) > _SHARE_OVERLAY_MAX_TURNS:
        raise ValueError("overlay contains too many turns")

    turns: list[Dict[str, Any]] = []
    for index, raw_turn in enumerate(raw_turns):
        if not isinstance(raw_turn, dict):
            raise ValueError(f"overlay.turns[{index}] must be an object")
        prompt = str(raw_turn.get("prompt") or "").strip()
        if len(prompt) > 8_000:
            raise ValueError(f"overlay.turns[{index}].prompt is too long")
        start = _share_float(
            raw_turn.get("start_seconds"),
            f"overlay.turns[{index}].start_seconds",
        )
        end = _share_float(
            raw_turn.get("end_seconds"),
            f"overlay.turns[{index}].end_seconds",
        )
        if start < 0 or end <= start:
            raise ValueError(f"overlay.turns[{index}] must have 0 <= start < end")
        try:
            number = max(1, int(raw_turn.get("number") or index + 1))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"overlay.turns[{index}].number must be an integer"
            ) from exc

        turn: Dict[str, Any] = {
            "number": number,
            "prompt": prompt,
            "start_seconds": start,
            "end_seconds": end,
        }
        raw_camera = raw_turn.get("camera")
        if raw_camera is not None:
            if not isinstance(raw_camera, dict):
                raise ValueError(f"overlay.turns[{index}].camera must be an object")
            raw_frames = raw_camera.get("frames")
            if not isinstance(raw_frames, list) or not raw_frames:
                raise ValueError(
                    f"overlay.turns[{index}].camera.frames must be non-empty"
                )
            if len(raw_frames) > _SHARE_CAMERA_MAX_FRAMES:
                raise ValueError("camera overlay contains too many frames")
            frames: list[Dict[str, Any]] = []
            for frame_index, raw_frame in enumerate(raw_frames):
                if not isinstance(raw_frame, dict):
                    raise ValueError(f"camera frame {frame_index} must be an object")
                position = raw_frame.get("position")
                quaternion = raw_frame.get("quaternion")
                if not isinstance(position, dict) or not isinstance(
                    quaternion,
                    dict,
                ):
                    raise ValueError(
                        f"camera frame {frame_index} needs position/quaternion"
                    )
                frames.append(
                    {
                        "frame": int(raw_frame.get("frame") or frame_index),
                        "t": max(
                            0.0,
                            _share_float(
                                raw_frame.get("t", 0),
                                f"camera.frames[{frame_index}].t",
                            ),
                        ),
                        "position": {
                            axis: _share_float(
                                position.get(axis),
                                f"camera.frames[{frame_index}].position.{axis}",
                            )
                            for axis in ("x", "y", "z")
                        },
                        "quaternion": {
                            axis: _share_float(
                                quaternion.get(axis),
                                f"camera.frames[{frame_index}].quaternion.{axis}",
                            )
                            for axis in ("x", "y", "z", "w")
                        },
                        "fov": _share_float(
                            raw_frame.get("fov", 55),
                            f"camera.frames[{frame_index}].fov",
                        ),
                    }
                )
            raw_events = raw_camera.get("key_events") or []
            if not isinstance(raw_events, list):
                raise ValueError("camera.key_events must be an array")
            if len(raw_events) > _SHARE_CAMERA_MAX_KEY_EVENTS:
                raise ValueError("camera overlay contains too many key events")
            key_events: list[Dict[str, Any]] = []
            for event_index, raw_event in enumerate(raw_events):
                if not isinstance(raw_event, dict):
                    raise ValueError(
                        f"camera key event {event_index} must be an object"
                    )
                code = str(raw_event.get("code") or "")
                if code not in _SHARE_CAMERA_KEYS:
                    raise ValueError(f"unsupported camera key event code: {code!r}")
                key_events.append(
                    {
                        "t": max(
                            0.0,
                            _share_float(
                                raw_event.get("t", 0),
                                f"camera.key_events[{event_index}].t",
                            ),
                        ),
                        "code": code,
                        "pressed": bool(raw_event.get("pressed")),
                    }
                )
            key_events.sort(key=lambda event: event["t"])
            camera: Dict[str, Any] = {
                "name": str(raw_camera.get("name") or "Camera path")[:160],
                "frames": frames,
                "key_events": key_events,
            }
            if raw_camera.get("duration_ms") is not None:
                camera["duration_ms"] = max(
                    0.0,
                    _share_float(
                        raw_camera.get("duration_ms"),
                        "camera.duration_ms",
                    ),
                )
            if raw_camera.get("sample_rate") is not None:
                camera["sample_rate"] = max(
                    0.001,
                    _share_float(
                        raw_camera.get("sample_rate"),
                        "camera.sample_rate",
                    ),
                )
            turn["camera"] = camera
        turns.append(turn)

    turns.sort(key=lambda turn: (turn["start_seconds"], turn["number"]))
    result: Dict[str, Any] = {"version": 1, "turns": turns}
    theme = str(value.get("theme") or "").strip().lower()
    if theme in {"dark", "light"}:
        result["theme"] = theme
    return result


_SHARED_VIDEO_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta name="theme-color" content="#11130f" />
  <title>Shared World · Hy-Nomas</title>
  <style>
    @font-face {
      font-family: Tencent Sans;
      src: url('../fonts/Tencent-W7CN.woff2') format('woff2');
      font-display: swap;
    }
    * { box-sizing: border-box; }
    html, body { min-height: 100%; margin: 0; }
    body {
      display: grid;
      place-items: center;
      padding: clamp(20px, 4vw, 56px);
      color: #f1eee6;
      background:
        radial-gradient(circle at 75% 12%, rgba(167, 207, 116, .13), transparent 32%),
        #11130f;
      font-family: Tencent Sans, system-ui, sans-serif;
    }
    main { width: min(1180px, 100%); }
    header {
      display: flex;
      margin-bottom: 20px;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
    }
    .brand { display: flex; align-items: center; gap: 11px; font-weight: 700; }
    .brand img { width: 34px; height: 34px; }
    header > span:last-child {
      color: #a7cf74;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: .08em;
      text-transform: uppercase;
    }
    .card {
      padding: clamp(10px, 1.4vw, 16px);
      border: 1px solid #353b30;
      border-radius: 24px;
      background: #1b1e18;
      box-shadow: 0 28px 90px rgba(0, 0, 0, .38);
    }
    .player { position: relative; overflow: hidden; border-radius: 16px; background: #080a07; }
    video {
      display: block;
      width: 100%;
      max-height: calc(100vh - 160px);
      background: #080a07;
    }
    .overlay {
      position: absolute;
      top: clamp(12px, 1.5vw, 22px);
      right: clamp(12px, 1.5vw, 22px);
      width: min(390px, calc(100% - 24px));
      min-height: 154px;
      padding: 12px 18px 17px;
      color: #f0ede5;
      border: 1px solid rgba(203, 218, 186, .38);
      border-radius: 18px;
      background: linear-gradient(135deg, rgba(35,39,31,.97), rgba(23,26,21,.94));
      box-shadow: 0 12px 32px rgba(0,0,0,.34);
      backdrop-filter: blur(14px);
      transition: opacity 160ms ease, transform 160ms ease;
    }
    .overlay[hidden], .camera-body[hidden], .prompt-body[hidden], .reveal[hidden] { display: none; }
    .overlay-head { display: flex; min-height: 30px; align-items: center; justify-content: space-between; gap: 12px; }
    .overlay-title { display: flex; min-width: 0; align-items: center; gap: 9px; }
    .overlay-icon {
      display: grid; width: 27px; height: 27px; flex: none; place-items: center;
      color: #192010; border-radius: 50%; background: #b9e47f; font-size: 15px;
    }
    .overlay-title div { display: grid; gap: 2px; }
    .overlay-title small { color: #b6dc84; font-size: 10px; font-weight: 700; letter-spacing: .13em; }
    .overlay-title strong { color: #bdbab1; font-size: 11px; }
    .overlay-tools { display: flex; align-items: center; gap: 8px; }
    .state { color: #aaa79e; font-size: 9px; font-weight: 700; letter-spacing: .1em; }
    .state.live { color: #b9e47f; }
    button {
      border: 1px solid #4a5144; color: #d9d6cd; background: rgba(20,23,18,.8);
      cursor: pointer; font: inherit;
    }
    .hide { display: grid; width: 29px; height: 29px; padding: 0; place-items: center; border-radius: 50%; }
    .prompt-body {
      display: -webkit-box; margin: 16px 0 4px; overflow: hidden;
      color: #f0ede5; font-size: clamp(17px, 1.35vw, 20px); line-height: 1.34;
      -webkit-box-orient: vertical; -webkit-line-clamp: 3;
    }
    .camera-body { margin-top: 10px; overflow: hidden; border: 1px solid #3d4438; border-radius: 9px; background: #0c1110; }
    canvas { display: block; width: 100%; height: 132px; }
    .camera-footer { display: flex; min-height: 32px; padding: 7px 10px; align-items: center; justify-content: space-between; gap: 12px; color: #aaa79e; border-top: 1px solid #343a31; font-size: 11px; }
    .camera-footer span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .camera-footer strong { color: #d9efbc; }
    .keys { display: flex; padding: 8px 10px 10px; justify-content: space-between; gap: 12px; border-top: 1px solid #343a31; }
    .key-group { display: grid; grid-template-columns: repeat(3, 24px); grid-template-rows: repeat(2, 24px); gap: 3px; }
    kbd { display: grid; place-items: center; border: 1px solid #586154; border-radius: 4px; color: #aaa79e; background: #242a22; font: 600 12px/1 Tencent Sans, sans-serif; }
    kbd:nth-child(1) { grid-column: 2; }
    kbd:nth-child(2) { grid-column: 1; grid-row: 2; }
    kbd:nth-child(3) { grid-column: 2; grid-row: 2; }
    kbd:nth-child(4) { grid-column: 3; grid-row: 2; }
    kbd.active { color: #18200f; border-color: #b9e47f; background: #b9e47f; box-shadow: 0 0 9px rgba(185,228,127,.46); }
    .track { position: absolute; right: 18px; bottom: 10px; left: 18px; height: 3px; overflow: hidden; border-radius: 999px; background: rgba(221,231,208,.13); }
    .track i { display: block; width: 0; height: 100%; border-radius: inherit; background: #b9e47f; box-shadow: 0 0 5px rgba(185,228,127,.42); }
    .reveal {
      position: absolute; top: clamp(12px,1.5vw,22px); right: clamp(12px,1.5vw,22px);
      min-height: 38px; padding: 0 14px; border-radius: 999px; color: #e1ded5;
    }
    .caption { margin: 14px 4px 2px; color: #aaa79e; font-size: 13px; }
    @media (max-width: 720px) {
      .overlay { min-height: 136px; padding-inline: 14px; }
      .prompt-body { font-size: 15px; }
      canvas { height: 108px; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <span class="brand">
        <img src="../hunyuan-logo.png" alt="" />
        Hy-Nomas
      </span>
      <span>Shared video</span>
    </header>
    <section class="card">
      <div class="player">
        <video id="video" controls autoplay muted playsinline src="../public/__SHARE_ID__.mp4"></video>
        <aside id="overlay" class="overlay" hidden>
          <div class="overlay-head">
            <div class="overlay-title">
              <span class="overlay-icon">↗</span>
              <div>
                <small id="overlay-kind">REAL USER PROMPT</small>
                <strong id="overlay-turn">Turn 01</strong>
              </div>
            </div>
            <div class="overlay-tools">
              <span id="overlay-state" class="state">PLAYBACK</span>
              <button id="hide" class="hide" type="button" aria-label="Hide overlay" title="Hide overlay">◉</button>
            </div>
          </div>
          <p id="prompt-body" class="prompt-body"></p>
          <div id="camera-body" class="camera-body" hidden>
            <canvas id="camera-canvas"></canvas>
            <div class="camera-footer"><span id="camera-name"></span><strong id="camera-progress">0%</strong></div>
            <div class="keys">
              <div class="key-group">
                <kbd data-code="KeyW">W</kbd><kbd data-code="KeyA">A</kbd><kbd data-code="KeyS">S</kbd><kbd data-code="KeyD">D</kbd>
              </div>
              <div class="key-group">
                <kbd data-code="ArrowUp">↑</kbd><kbd data-code="ArrowLeft">←</kbd><kbd data-code="ArrowDown">↓</kbd><kbd data-code="ArrowRight">→</kbd>
              </div>
            </div>
          </div>
          <div class="track"><i id="progress"></i></div>
        </aside>
        <button id="reveal" class="reveal" type="button" hidden>Show overlay</button>
      </div>
      <p class="caption">This is a read-only public video with synchronized HTML overlays.</p>
    </section>
  </main>
  <script>
    (() => {
      const video = document.getElementById('video');
      const overlay = document.getElementById('overlay');
      const reveal = document.getElementById('reveal');
      const hide = document.getElementById('hide');
      const kind = document.getElementById('overlay-kind');
      const turnLabel = document.getElementById('overlay-turn');
      const state = document.getElementById('overlay-state');
      const promptBody = document.getElementById('prompt-body');
      const cameraBody = document.getElementById('camera-body');
      const cameraName = document.getElementById('camera-name');
      const cameraProgress = document.getElementById('camera-progress');
      const progress = document.getElementById('progress');
      const canvas = document.getElementById('camera-canvas');
      const keys = Array.from(document.querySelectorAll('kbd[data-code]'));
      let turns = [];
      let hidden = false;

      const activeTurn = (seconds) => {
        let selected = turns[0];
        for (const turn of turns) {
          if (seconds >= turn.start_seconds) selected = turn;
          else break;
        }
        return selected;
      };
      const activeKeys = (events, milliseconds) => {
        const result = {};
        for (const event of events || []) {
          if (event.t > milliseconds) break;
          result[event.code] = Boolean(event.pressed);
        }
        return result;
      };
      const drawCamera = (camera, ratio) => {
        const frames = camera.frames || [];
        if (!frames.length) return;
        const rect = canvas.getBoundingClientRect();
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const width = Math.max(1, Math.round(rect.width * dpr));
        const height = Math.max(1, Math.round(rect.height * dpr));
        if (canvas.width !== width || canvas.height !== height) {
          canvas.width = width;
          canvas.height = height;
        }
        const ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        const w = width / dpr;
        const h = height / dpr;
        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = '#0c1110';
        ctx.fillRect(0, 0, w, h);
        ctx.strokeStyle = 'rgba(185,228,127,.08)';
        ctx.lineWidth = 1;
        for (let x = 0; x < w; x += 28) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
        for (let y = 0; y < h; y += 28) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
        const xs = frames.map((frame) => Number(frame.position.x));
        let ys = frames.map((frame) => Number(frame.position.z));
        const zSpan = Math.max(...ys) - Math.min(...ys);
        if (zSpan < 1e-6) ys = frames.map((frame) => Number(frame.position.y));
        const minX = Math.min(...xs); const maxX = Math.max(...xs);
        const minY = Math.min(...ys); const maxY = Math.max(...ys);
        const spanX = Math.max(maxX - minX, .001);
        const spanY = Math.max(maxY - minY, .001);
        const pad = 18;
        const point = (index) => ({
          x: pad + ((xs[index] - minX) / spanX) * (w - pad * 2),
          y: h - pad - ((ys[index] - minY) / spanY) * (h - pad * 2),
        });
        const markerIndex = Math.min(frames.length - 1, Math.max(0, Math.round(ratio * (frames.length - 1))));
        const strokePath = (end, color, lineWidth) => {
          ctx.beginPath();
          frames.slice(0, end + 1).forEach((_, index) => {
            const p = point(index);
            if (index === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
          });
          ctx.strokeStyle = color; ctx.lineWidth = lineWidth; ctx.lineJoin = 'round'; ctx.lineCap = 'round'; ctx.stroke();
        };
        strokePath(frames.length - 1, 'rgba(201,212,195,.34)', 2);
        strokePath(markerIndex, '#b9e47f', 3);
        const marker = point(markerIndex);
        ctx.beginPath(); ctx.arc(marker.x, marker.y, 5, 0, Math.PI * 2);
        ctx.fillStyle = '#f5f2ea'; ctx.fill();
        ctx.strokeStyle = '#18200f'; ctx.lineWidth = 2; ctx.stroke();
      };
      const render = () => {
        if (!turns.length) {
          overlay.hidden = true;
          reveal.hidden = true;
          requestAnimationFrame(render);
          return;
        }
        const turn = activeTurn(video.currentTime);
        const duration = Math.max(.001, turn.end_seconds - turn.start_seconds);
        const ratio = Math.max(0, Math.min(1, (video.currentTime - turn.start_seconds) / duration));
        overlay.hidden = hidden;
        reveal.hidden = !hidden;
        state.textContent = video.paused ? 'PLAYBACK' : 'LIVE';
        state.classList.toggle('live', !video.paused);
        turnLabel.textContent = `Turn ${String(turn.number).padStart(2, '0')}`;
        progress.style.width = `${Math.round(ratio * 100)}%`;
        if (turn.camera) {
          kind.textContent = 'CAMERA PATH';
          promptBody.hidden = true;
          cameraBody.hidden = false;
          cameraName.textContent = turn.camera.name || 'Camera path';
          cameraProgress.textContent = `${Math.round(ratio * 100)}%`;
          const durationMs = turn.camera.duration_ms || duration * 1000;
          const keyState = activeKeys(turn.camera.key_events, ratio * durationMs);
          keys.forEach((key) => key.classList.toggle('active', Boolean(keyState[key.dataset.code])));
          drawCamera(turn.camera, ratio);
        } else {
          kind.textContent = 'REAL USER PROMPT';
          promptBody.hidden = false;
          cameraBody.hidden = true;
          promptBody.textContent = turn.prompt || '';
        }
        requestAnimationFrame(render);
      };
      hide.addEventListener('click', () => { hidden = true; });
      reveal.addEventListener('click', () => { hidden = false; });
      fetch('../public/__SHARE_ID__.json', { cache: 'no-store' })
        .then((response) => response.ok ? response.json() : Promise.reject(new Error(`HTTP ${response.status}`)))
        .then((metadata) => {
          turns = (metadata.overlay?.turns || []).slice().sort(
            (a, b) => a.start_seconds - b.start_seconds || a.number - b.number,
          );
        })
        .catch((error) => console.warn('Could not load shared overlay metadata', error))
        .finally(() => requestAnimationFrame(render));
    })();
  </script>
</body>
</html>"""


def _shared_video_page_html(share_id: str) -> str:
    return _SHARED_VIDEO_HTML.replace(
        "__SHARE_ID__",
        html.escape(share_id, quote=True),
    )


def create_gateway_app(
    store: JobStore,
    *,
    frontend_dir: Path | None,
    output_dir: Path,
    gallery_dir: Path | None = None,
    cors_origins: list[str] | None = None,
    prompt_enhancer: OnlinePromptEnhancer | None = None,
    inference_config: Dict[str, Any] | None = None,
    resource_monitor: ResourceMonitor | None = None,
) -> FastAPI:
    output_dir.mkdir(parents=True, exist_ok=True)
    if gallery_dir is not None:
        gallery_dir.mkdir(parents=True, exist_ok=True)
    share_index_path = output_dir.parent / f".{output_dir.name}_video_shares.json"
    share_lock = threading.RLock()
    download_lock = threading.RLock()
    gallery_write_lock = threading.RLock()
    pe_export_dir = output_dir / "pe_exports"
    attempt_archive = UserAttemptArchive(output_dir)
    attempt_archive_stop = threading.Event()
    attempt_archive_versions: dict[str, float] = {}
    attempt_archive_versions_lock = threading.Lock()

    def archive_generation_record(
        record: Dict[str, Any],
        *,
        payload_overrides: Dict[str, Any] | None = None,
    ) -> None:
        job_id = str(record.get("id") or "")
        updated_at = float(record.get("updated_at") or 0.0)
        if not job_id:
            return
        if payload_overrides:
            payload = record.get("payload")
            payload = dict(payload) if isinstance(payload, dict) else {}
            payload.update(payload_overrides)
            record = {**record, "payload": payload}
        with attempt_archive_versions_lock:
            if (
                not payload_overrides
                and attempt_archive_versions.get(job_id) == updated_at
            ):
                return
        attempt_archive.write(record)
        with attempt_archive_versions_lock:
            attempt_archive_versions[job_id] = updated_at

    def archive_generation_job(
        job_id: str,
        *,
        payload_overrides: Dict[str, Any] | None = None,
    ) -> None:
        try:
            record = store.generation_attempt_record(job_id)
            if record is not None:
                archive_generation_record(
                    record,
                    payload_overrides=payload_overrides,
                )
        except Exception:
            logger.exception("Could not archive generation attempt %s", job_id)

    def mirror_generation_attempts() -> None:
        # Backfill earlier attempts after a Gateway hot restart. Records made
        # before Dispatcher identity forwarding are grouped as unknown users.
        while not attempt_archive_stop.is_set():
            with attempt_archive_versions_lock:
                known_updates = dict(attempt_archive_versions)
            for record in store.changed_generation_attempt_records(known_updates):
                job_id = str(record.get("id") or "")
                try:
                    archive_generation_record(record)
                except Exception:
                    logger.exception(
                        "Could not mirror generation attempt %s",
                        job_id,
                    )
            attempt_archive_stop.wait(5)

    def load_shares() -> dict[str, dict[str, Any]]:
        if not share_index_path.is_file():
            return {}
        try:
            value = json.loads(share_index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def save_shares(shares: dict[str, dict[str, Any]]) -> None:
        temporary = share_index_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(shares, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(share_index_path)

    def resolve_shared_video(share_id: str) -> Path:
        with share_lock:
            record = load_shares().get(share_id)
        if not isinstance(record, dict):
            raise HTTPException(status_code=404, detail="shared video not found")
        relative_path = str(record.get("relative_path") or "")
        candidate = (output_dir / relative_path).resolve()
        output_root = output_dir.resolve()
        if (
            not candidate.is_relative_to(output_root)
            or candidate.suffix.lower() != ".mp4"
            or not candidate.is_file()
        ):
            raise HTTPException(status_code=404, detail="shared video not found")
        return candidate

    def load_gallery_items() -> list[dict[str, Any]]:
        return read_gallery_items(gallery_dir)

    def add_gallery_item(payload: Dict[str, Any]) -> dict[str, Any]:
        if gallery_dir is None:
            raise HTTPException(
                status_code=503,
                detail="Gallery storage is not configured",
            )
        video_url = str(payload.get("video_url") or "").strip()
        clean_path = unquote(urlsplit(video_url).path).lstrip("/")
        if not clean_path.startswith("outputs/"):
            raise HTTPException(
                status_code=400,
                detail="video_url must reference a generated output",
            )
        source = (output_dir / clean_path[len("outputs/") :]).resolve()
        output_root = output_dir.resolve()
        if (
            not source.is_relative_to(output_root)
            or source.suffix.lower() != ".mp4"
            or not source.is_file()
        ):
            raise HTTPException(status_code=404, detail="generated video not found")
        try:
            return add_gallery_item_from_source(
                gallery_dir,
                source,
                payload,
                process_lock=gallery_write_lock,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc
        except GalleryStorageBusyError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    def hide_gallery_item(
        item_id: str,
        *,
        delete_media: bool = False,
    ) -> dict[str, Any]:
        if gallery_dir is None:
            raise HTTPException(
                status_code=503,
                detail="Gallery storage is not configured",
            )
        try:
            return remove_gallery_item(
                gallery_dir,
                item_id,
                process_lock=gallery_write_lock,
                delete_media=delete_media,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except GalleryStorageBusyError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    def export_pe_session(session_id: str) -> Path:
        clean_session_id = str(session_id or "").strip()
        if not clean_session_id:
            raise HTTPException(status_code=400, detail="session_id is required")
        records = [
            record
            for record in store.session_job_records(clean_session_id)
            if record.get("kind") == "generate"
        ]
        if not records:
            raise HTTPException(
                status_code=404,
                detail="No generation records found for this session",
            )

        transcript: Dict[str, Any] | None = None
        transcript_path: Path | None = None
        for record in reversed(records):
            state = record.get("state")
            if not isinstance(state, dict):
                continue
            raw_path = str(state.get("pe_transcript_path") or "").strip()
            if not raw_path:
                continue
            candidate = Path(raw_path).expanduser().resolve()
            try:
                loaded = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(loaded, dict):
                transcript = loaded
                transcript_path = candidate
                break

        prompt_profile = (
            getattr(prompt_enhancer, "prompt_profile", None)
            if prompt_enhancer is not None
            else None
        )
        system_message = (
            transcript.get("system_message") if isinstance(transcript, dict) else None
        )
        if not isinstance(system_message, dict) and prompt_profile is not None:
            system_message = {
                "role": "system",
                "content": str(prompt_profile.system_prompt),
            }
        prompt_sources = (
            transcript.get("prompt_sources") if isinstance(transcript, dict) else None
        )
        if not isinstance(prompt_sources, list) and prompt_profile is not None:
            prompt_sources = [
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in prompt_profile.source_paths
            ]
        session_system_prompt_sha256 = (
            hashlib.sha256(
                str(system_message.get("content") or "").encode()
            ).hexdigest()
            if isinstance(system_message, dict)
            else None
        )
        current_system_message = (
            {
                "role": "system",
                "content": str(prompt_profile.system_prompt),
            }
            if prompt_profile is not None
            else None
        )
        current_prompt_sources = (
            [
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in prompt_profile.source_paths
            ]
            if prompt_profile is not None
            else []
        )
        current_system_prompt_sha256 = (
            hashlib.sha256(
                str(current_system_message.get("content") or "").encode()
            ).hexdigest()
            if isinstance(current_system_message, dict)
            else None
        )
        session_prompt_profile_sha256 = (
            str(transcript.get("prompt_profile_sha256") or "").strip()
            if isinstance(transcript, dict)
            else ""
        ) or None
        if transcript is None and prompt_profile is not None:
            session_prompt_profile_sha256 = prompt_profile.sha256
        prompt_matches_current_service = bool(
            session_system_prompt_sha256
            and current_system_prompt_sha256
            and session_system_prompt_sha256 == current_system_prompt_sha256
        )
        transcript_turns = (
            transcript.get("turns") if isinstance(transcript, dict) else None
        )
        if not isinstance(transcript_turns, list):
            transcript_turns = []

        turns: list[Dict[str, Any]] = []
        for index, record in enumerate(records, start=1):
            payload = record.get("payload")
            state = record.get("state")
            payload = payload if isinstance(payload, dict) else {}
            state = state if isinstance(state, dict) else {}
            raw_camera = payload.get("camera_trajectory")
            camera: Any = raw_camera
            if isinstance(raw_camera, str):
                try:
                    camera = json.loads(raw_camera)
                except json.JSONDecodeError:
                    camera = raw_camera
            camera_summary: Dict[str, Any] | None = None
            if isinstance(camera, dict):
                frames = camera.get("frames")
                key_events = camera.get(
                    "keyEvents",
                    camera.get("key_events"),
                )
                camera_summary = {
                    key: camera.get(key)
                    for key in (
                        "name",
                        "recordedAt",
                        "sampleRate",
                        "durationMs",
                        "totalDistance",
                    )
                    if camera.get(key) is not None
                }
                camera_summary.update({
                    key: payload[key]
                    for key in ("trajectory_translation_scale", "trajectory_rotation_scale")
                    if payload.get(key) is not None
                })
                camera_summary.update(
                    {
                        "frame_count": (len(frames) if isinstance(frames, list) else 0),
                        "key_event_count": (
                            len(key_events) if isinstance(key_events, list) else 0
                        ),
                    }
                )
            result = state.get("result")
            transcript_turn = next(
                (
                    turn
                    for turn in transcript_turns
                    if isinstance(turn, dict)
                    and int(turn.get("turn_number") or 0) == index
                ),
                None,
            )
            turns.append(
                {
                    "turn_number": index,
                    "job_id": record.get("id"),
                    "status": record.get("status"),
                    "created_at": record.get("created_at"),
                    "started_at": record.get("started_at"),
                    "finished_at": record.get("finished_at"),
                    "original_user_prompts": payload.get("user_prompts") or [],
                    "carry_user_prompt": payload.get("carry_user_prompt") or "",
                    "prompt_enhancement_mode": state.get("prompt_enhancement_mode"),
                    "pe_request_id": state.get("pe_request_id"),
                    "pe_prompt": state.get("pe_prompt") or "",
                    "pe_reasoning": state.get("pe_reasoning") or "",
                    "pe_transcript_path": state.get("pe_transcript_path"),
                    "turn_prompt": (
                        transcript_turn.get("turn_prompt")
                        if isinstance(transcript_turn, dict)
                        else (
                            prompt_profile.turn_prompt(first=index == 1)
                            if prompt_profile is not None
                            else None
                        )
                    ),
                    "turn_prompt_path": (
                        transcript_turn.get("turn_prompt_path")
                        if isinstance(transcript_turn, dict)
                        else (
                            str(prompt_profile.turn_prompt_path(first=index == 1))
                            if (
                                prompt_profile is not None
                                and prompt_profile.turn_prompt_path(first=index == 1)
                                is not None
                            )
                            else None
                        )
                    ),
                    "final_model_prompt": payload.get("prompt") or "",
                    "segment_boundaries": payload.get("segment_boundaries"),
                    "segment_has_camera": payload.get("segment_has_camera"),
                    "camera_path_summary": camera_summary,
                    "camera_trajectory": camera,
                    "result": result if isinstance(result, dict) else None,
                }
            )

        export = {
            "schema_version": 1,
            "exported_at": time.time(),
            "session_id": clean_session_id,
            "prompt_profile": (
                transcript.get("prompt_profile")
                if isinstance(transcript, dict)
                else getattr(prompt_profile, "name", None)
            ),
            # Compatibility fields below describe the exact historical prompt
            # used by this session. They must not silently change when the
            # service deploys a newer prompt profile.
            "prompt_profile_sha256": session_prompt_profile_sha256,
            "system_prompt_sha256": session_system_prompt_sha256,
            "system_message": system_message,
            "prompt_sources": prompt_sources or [],
            "prompt_matches_current_service": (prompt_matches_current_service),
            "session_prompt_snapshot": {
                "scope": "exact_prompt_used_by_this_session",
                "prompt_profile": (
                    transcript.get("prompt_profile")
                    if isinstance(transcript, dict)
                    else None
                ),
                "prompt_profile_sha256": session_prompt_profile_sha256,
                "system_prompt_sha256": session_system_prompt_sha256,
                "system_message": system_message,
                "prompt_sources": prompt_sources or [],
            },
            "current_service_prompt": {
                "scope": "prompt_loaded_by_service_at_export_time",
                "prompt_profile": getattr(prompt_profile, "name", None),
                "prompt_profile_sha256": (
                    getattr(prompt_profile, "sha256", None)
                    if prompt_profile is not None
                    else None
                ),
                "system_prompt_sha256": current_system_prompt_sha256,
                "system_message": current_system_message,
                "prompt_sources": current_prompt_sources,
            },
            "source_transcript_path": (
                str(transcript_path) if transcript_path is not None else None
            ),
            "turn_count": len(turns),
            "turns": turns,
        }
        pe_export_dir.mkdir(parents=True, exist_ok=True)
        safe_session_id = "".join(
            character if character.isalnum() or character in {"-", "_"} else "_"
            for character in clean_session_id
        )[:128]
        destination = pe_export_dir / (f"session_{safe_session_id}_pe_prompts.json")
        with tempfile.NamedTemporaryFile(
            dir=str(pe_export_dir),
            prefix=f".session-{safe_session_id}.",
            suffix=".json.tmp",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(export, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            replace_or_overwrite(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    @asynccontextmanager
    async def lifespan(_: Any):
        attempt_archive_stop.clear()
        attempt_archive_thread = threading.Thread(
            target=mirror_generation_attempts,
            name="user-attempt-archive",
            daemon=True,
        )
        attempt_archive_thread.start()
        if resource_monitor is not None:
            resource_monitor.start()
        yield
        attempt_archive_stop.set()
        attempt_archive_thread.join(timeout=2)
        if resource_monitor is not None:
            resource_monitor.stop()

    app = FastAPI(
        title="Hy-Nomas inference gateway",
        version="2.1",
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
        worker = store.worker()
        scheduler = store.scheduler_snapshot()
        value = {
            "status": "ok",
            "architecture": "gateway_worker",
            "worker_online": worker["online"],
            "worker_status": worker["status"],
            "busy": scheduler["running_jobs"] > 0,
            "worker_count": scheduler["worker_count"],
            "online_worker_count": scheduler["online_worker_count"],
            "busy_worker_count": scheduler["busy_worker_count"],
            "queued_jobs": scheduler["queued_jobs"],
            "running_jobs": scheduler["running_jobs"],
            "reserved_session_count": scheduler["reserved_session_count"],
            "available": bool(
                worker["online"]
                and scheduler["queued_jobs"] == 0
                and scheduler["running_jobs"] == 0
                and scheduler["enhancing_jobs"] == 0
                and scheduler["reserved_session_count"] == 0
            ),
        }
        return value

    @app.get("/v1/config")
    def config(
        x_nomas_client_id: str | None = Header(
            default=None,
            alias="X-Nomas-Client-Id",
        ),
    ) -> Dict[str, Any]:
        worker = store.worker()
        scheduler = store.scheduler_snapshot()
        client_id = x_nomas_client_id
        if resource_monitor is not None and client_id:
            resource_monitor.touch(client_id)
        resources = (
            resource_monitor.snapshot(
                running_jobs=scheduler["running_jobs"],
            )
            if resource_monitor is not None
            else {
                "available": False,
                "online_users": 0,
                "gpus": [],
            }
        )
        worker_config = {**(inference_config or {}), **dict(worker.get("config") or {})}
        default_steps = int(worker_config.get("num_inference_steps") or DEFAULT_NUM_INFERENCE_STEPS)
        value = {
            "rollout_type": "seg_ar",
            "prompt_source": "prompt",
            "num_frames": None,
            "num_frames_source": "request_or_config",
            "fps": 24,
            **worker_config,
            "num_inference_steps_options": sorted(set((*NUM_INFERENCE_STEPS_OPTIONS, default_steps))),
            "default_num_inference_steps": default_steps,
        }
        value.update(
            {
                "architecture": "gateway_worker",
                "worker_online": worker["online"],
                "worker_status": worker["status"],
                "dry_run_enabled": bool(worker["online"] and worker_config.get("dry_run_enabled")),
                "prompt_enhancement_enabled": bool(
                    prompt_enhancer is not None and prompt_enhancer.configured
                ),
                "prompt_enhancement_profile": (
                    getattr(
                        getattr(prompt_enhancer, "prompt_profile", None),
                        "name",
                        None,
                    )
                    if prompt_enhancer is not None
                    else None
                ),
                "prompt_enhancement_profile_sha256": (
                    getattr(
                        getattr(prompt_enhancer, "prompt_profile", None),
                        "sha256",
                        None,
                    )
                    if prompt_enhancer is not None
                    else None
                ),
                "prompt_enhancement_system_sha256": (
                    hashlib.sha256(
                        str(
                            getattr(
                                getattr(
                                    prompt_enhancer,
                                    "prompt_profile",
                                    None,
                                ),
                                "system_prompt",
                                "",
                            )
                        ).encode()
                    ).hexdigest()
                    if prompt_enhancer is not None
                    and getattr(
                        getattr(prompt_enhancer, "prompt_profile", None),
                        "system_prompt",
                        None,
                    )
                    else None
                ),
                "prompt_enhancer_options": (
                    prompt_enhancer.model_options
                    if isinstance(prompt_enhancer, OnlinePromptEnhancer)
                    else list(PROMPT_ENHANCER_OPTIONS)
                ),
                "default_prompt_enhancer_model": (
                    prompt_enhancer.default_model
                    if isinstance(prompt_enhancer, OnlinePromptEnhancer)
                    else DEFAULT_PROMPT_ENHANCER_MODEL
                ),
                "prompt_enhancer_reasoning_effort_options": list(
                    PROMPT_ENHANCER_REASONING_EFFORTS
                ),
                "default_prompt_enhancer_reasoning_effort": (
                    DEFAULT_PROMPT_ENHANCER_REASONING_EFFORT
                ),
                "prompt_enhancer_seed_min": PROMPT_ENHANCER_SEED_MIN,
                "prompt_enhancer_seed_max": PROMPT_ENHANCER_SEED_MAX,
                "prompt_fallback_enabled": True,
                "gallery_items": load_gallery_items(),
                "gallery_hidden_ids": read_hidden_gallery_item_ids(gallery_dir),
                "worker_count": scheduler["worker_count"],
                "online_worker_count": scheduler["online_worker_count"],
                "busy_worker_count": scheduler["busy_worker_count"],
                "queued_jobs": scheduler["queued_jobs"],
                "running_jobs": scheduler["running_jobs"],
                "reserved_session_count": scheduler["reserved_session_count"],
                "concurrency_enabled": (scheduler["online_worker_count"] > 1),
                "online_users": (resources["online_users"]),
                "resources": resources,
                "recent_errors": store.recent_errors(limit=8),
            }
        )
        return value

    @app.post("/v1/shares", status_code=201)
    def create_video_share(payload: Dict[str, Any]) -> Dict[str, Any]:
        video_url = str(payload.get("video_url") or "").strip()
        clean_path = unquote(urlsplit(video_url).path).lstrip("/")
        if not clean_path.startswith("outputs/"):
            raise HTTPException(
                status_code=400,
                detail="video_url must reference a generated output",
            )
        relative_path = clean_path[len("outputs/") :]
        source = (output_dir / relative_path).resolve()
        output_root = output_dir.resolve()
        if (
            not source.is_relative_to(output_root)
            or source.suffix.lower() != ".mp4"
            or not source.is_file()
        ):
            raise HTTPException(status_code=404, detail="generated video not found")

        share_id = secrets.token_urlsafe(18)
        try:
            overlay = _normalize_share_overlay(payload.get("overlay"))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        with share_lock:
            shares = load_shares()
            shares[share_id] = {
                "relative_path": source.relative_to(output_root).as_posix(),
                "created_at": time.time(),
                "overlay": overlay,
            }
            save_shares(shares)
        return {
            "id": share_id,
            "share_path": f"./share/{share_id}",
        }

    @app.get("/v1/gallery")
    def gallery_items() -> Dict[str, Any]:
        items = load_gallery_items()
        return {
            "items": items,
            "count": len(items),
            "hidden_ids": read_hidden_gallery_item_ids(gallery_dir),
        }

    @app.get("/v1/sessions/{session_id}/pe-export")
    def download_pe_session_export(session_id: str) -> FileResponse:
        export_path = export_pe_session(session_id)
        return FileResponse(
            export_path,
            media_type="application/json",
            filename=export_path.name,
        )

    @app.post("/v1/gallery", status_code=201)
    def save_to_gallery(payload: Dict[str, Any]) -> Dict[str, Any]:
        return add_gallery_item(payload)

    @app.delete("/v1/gallery/{item_id}")
    def remove_from_gallery(
        item_id: str,
        delete_media: bool = False,
    ) -> Dict[str, Any]:
        return hide_gallery_item(item_id, delete_media=delete_media)

    @app.post("/v1/downloads", status_code=201)
    def create_video_download(payload: Dict[str, Any]) -> Dict[str, Any]:
        video_url = str(payload.get("video_url") or "").strip()
        clean_path = unquote(urlsplit(video_url).path).lstrip("/")
        if not clean_path.startswith("outputs/"):
            raise HTTPException(
                status_code=400,
                detail="video_url must reference a generated output",
            )
        relative_path = clean_path[len("outputs/") :]
        source = (output_dir / relative_path).resolve()
        output_root = output_dir.resolve()
        if (
            not source.is_relative_to(output_root)
            or source.suffix.lower() != ".mp4"
            or not source.is_file()
        ):
            raise HTTPException(status_code=404, detail="generated video not found")

        include_overlay = payload.get("include_overlay", True) is not False
        if not include_overlay:
            return {
                "download_path": f"./outputs/{source.relative_to(output_root).as_posix()}",
                "filename": source.name,
                "overlay": False,
            }
        try:
            overlay = _normalize_share_overlay(payload.get("overlay"))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        fingerprint = hashlib.sha256(
            (
                f"v{_DOWNLOAD_OVERLAY_RENDERER_VERSION}:"
                f"{source}:{source.stat().st_size}:{source.stat().st_mtime_ns}:"
                + json.dumps(
                    overlay,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ).encode("utf-8")
        ).hexdigest()[:16]
        download_dir = output_dir / "downloads"
        destination = download_dir / (f"{source.stem}-with-overlay-{fingerprint}.mp4")
        with download_lock:
            if not destination.is_file():
                download_dir.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    prefix=f"nomas-overlay-{fingerprint}-",
                    suffix=".mp4",
                    delete=False,
                ) as handle:
                    temporary = Path(handle.name)
                try:
                    burn_shared_overlay_video(source, temporary, overlay)
                    shutil.copyfile(temporary, destination)
                except Exception as exc:  # noqa: BLE001
                    destination.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=500,
                        detail=f"Could not render overlay video: {exc}",
                    ) from exc
                finally:
                    temporary.unlink(missing_ok=True)
        return {
            "download_path": (
                f"./outputs/{destination.relative_to(output_root).as_posix()}"
            ),
            "filename": destination.name,
            "overlay": True,
        }

    @app.get("/share/{share_id}", response_class=HTMLResponse)
    def shared_video_page(share_id: str) -> HTMLResponse:
        resolve_shared_video(share_id)
        return HTMLResponse(
            content=_shared_video_page_html(share_id),
            headers={
                "Cache-Control": "no-store",
                "X-Robots-Tag": "noindex, nofollow",
            },
        )

    @app.api_route("/public/{share_id}.mp4", methods=["GET", "HEAD"])
    def shared_video_file(share_id: str) -> FileResponse:
        return FileResponse(
            resolve_shared_video(share_id),
            media_type="video/mp4",
            filename=None,
            headers={"Cache-Control": "private, max-age=3600"},
        )

    @app.get("/public/{share_id}.json")
    def shared_video_metadata(share_id: str) -> JSONResponse:
        resolve_shared_video(share_id)
        with share_lock:
            record = load_shares().get(share_id)
        if not isinstance(record, dict):
            raise HTTPException(status_code=404, detail="shared video not found")
        return JSONResponse(
            {
                "id": share_id,
                "created_at": record.get("created_at"),
                "overlay": _normalize_share_overlay(record.get("overlay")),
            },
            headers={"Cache-Control": "private, max-age=300"},
        )

    @app.post("/v1/dry-run", status_code=202)
    def dry_run() -> Dict[str, Any]:
        return store.submit("dryrun")

    @app.get("/v1/dry-run/{job_id}")
    def dry_run_status(job_id: str) -> Dict[str, Any]:
        job = store.get(job_id)
        if job is None or job.get("kind") != "dryrun":
            raise HTTPException(status_code=404, detail="dry-run job not found")
        return job

    def prepare_generation(job_id: str, submitted: Dict[str, Any]) -> None:
        payload = dict(submitted)
        pe_input = payload.pop("pe_input", None)
        previous_session: Dict[str, Any] | None = None
        enhancement_mode = "unknown"
        stream_state: Dict[str, Any] = {
            "reasoning": "",
            "content": "",
        }
        try:
            if pe_input is not None:
                if not isinstance(pe_input, dict):
                    raise PromptEnhancementError("pe_input must be an object")
                model_override = pe_input.pop("prompt_enhancer_model", None)
                default_model = (
                    prompt_enhancer.default_model or DEFAULT_PROMPT_ENHANCER_MODEL
                    if isinstance(prompt_enhancer, OnlinePromptEnhancer)
                    else DEFAULT_PROMPT_ENHANCER_MODEL
                )
                requested_enhancer_model = normalize_prompt_enhancer_model(
                    model_override if model_override is not None else default_model
                )
                if (
                    isinstance(prompt_enhancer, OnlinePromptEnhancer)
                    and not pe_input.get("camera_only_minimal")
                    and (model_override is not None or prompt_enhancer.configured)
                ):
                    prompt_enhancer.resolve_model(requested_enhancer_model)
                requested_reasoning_effort = normalize_prompt_enhancer_reasoning_effort(
                    pe_input.get(
                        "prompt_enhancer_reasoning_effort",
                        DEFAULT_PROMPT_ENHANCER_REASONING_EFFORT,
                    )
                )
                pe_input["prompt_enhancer_reasoning_effort"] = (
                    requested_reasoning_effort
                )
                requested_seed = normalize_prompt_enhancer_seed(
                    pe_input.get("prompt_enhancer_seed")
                )
                if requested_seed is None:
                    pe_input.pop("prompt_enhancer_seed", None)
                else:
                    pe_input["prompt_enhancer_seed"] = requested_seed
                store.patch(
                    job_id,
                    prompt_enhancement_mode=requested_enhancer_model,
                    prompt_enhancer_reasoning_effort=(requested_reasoning_effort),
                    prompt_enhancer_seed=requested_seed,
                    prompt_enhancer_seed_source=(
                        str(pe_input.get("prompt_enhancer_seed_source") or "").strip()
                        or None
                    ),
                )
                session_id = str(pe_input.get("session_id") or "").strip()
                if not session_id:
                    raise PromptEnhancementError("pe_input.session_id is required")
                raw_user_prompt = str(
                    pe_input.get("action")
                    or pe_input.get("environment_description")
                    or ""
                ).strip()
                stream_lock = threading.Lock()
                stream_state = {
                    "reasoning": "",
                    "content": "",
                    "content_started": False,
                    "last_publish": 0.0,
                }

                def publish_stream(kind: str, delta: str) -> None:
                    with stream_lock:
                        if kind == "reset":
                            stream_state.update(
                                {
                                    "reasoning": "",
                                    "content": "",
                                    "content_started": False,
                                    "last_publish": 0.0,
                                }
                            )
                            store.patch(
                                job_id,
                                pe_reasoning="",
                                pe_content_preview="",
                                pe_content_started=False,
                            )
                            return
                        if kind == "reasoning":
                            stream_state["reasoning"] += delta
                        elif kind == "content":
                            stream_state["content"] += delta
                            if not stream_state["content_started"]:
                                stream_state["content_started"] = True
                                store.patch(
                                    job_id,
                                    pe_reasoning="",
                                    pe_content_preview=str(stream_state["content"]),
                                    pe_content_started=True,
                                )
                                stream_state["last_publish"] = time.monotonic()
                                return
                        now = time.monotonic()
                        if now - stream_state["last_publish"] < 0.12:
                            return
                        stream_state["last_publish"] = now
                        if kind == "content":
                            store.patch(
                                job_id,
                                pe_reasoning="",
                                pe_content_preview=str(stream_state["content"]),
                                pe_content_started=True,
                            )
                            return
                        # The UI renders a single compact line, so retaining a
                        # bounded tail avoids growing the queue JSON indefinitely.
                        reasoning = str(stream_state["reasoning"])[-1200:]
                        store.patch(
                            job_id,
                            pe_reasoning=reasoning,
                            pe_content_started=False,
                        )

                previous_session = store.get_session(session_id)
                if previous_session is None and payload.get("reset_session") is False:
                    raise PromptEnhancementError(
                        "WORLD_SESSION_EXPIRED: This World session is no "
                        "longer available after the service restart. "
                        "Create a new World to continue."
                    )
                previous_result = (
                    (previous_session or {}).get("last_result")
                    if isinstance(previous_session, dict)
                    else None
                )
                carry_frames = (
                    int(
                        previous_result.get(
                            "provisional_pixel_frames",
                            previous_result.get(
                                "withheld_pixel_frames",
                                0,
                            ),
                        )
                        or 0
                    )
                    if isinstance(previous_result, dict)
                    else 0
                )
                previous_overlay = (
                    dict((previous_session or {}).get("last_overlay") or {})
                    if isinstance(previous_session, dict)
                    else {}
                )
                camera_controlled = bool(pe_input.get("camera_controlled"))
                enhancer_configured = bool(
                    prompt_enhancer is not None
                    and (
                        prompt_enhancer.configured_for(requested_enhancer_model)
                        if hasattr(prompt_enhancer, "configured_for")
                        else prompt_enhancer.configured
                    )
                )
                if camera_controlled:
                    # Pass the trajectory object rather than an opaque marker.
                    # The PE layer detaches it and removes fields that do not
                    # contribute to camera-motion interpretation; the original
                    # payload remains authoritative for DiT conditioning.
                    camera_trajectory = payload.get("camera_trajectory")
                    if camera_trajectory is None:
                        raise PromptEnhancementError(
                            "camera-controlled generation requires camera_trajectory"
                        )
                    if isinstance(camera_trajectory, str):
                        try:
                            camera_control = json.loads(camera_trajectory)
                        except json.JSONDecodeError as exc:
                            raise PromptEnhancementError(
                                "camera_trajectory must be valid JSON"
                            ) from exc
                    else:
                        camera_control = camera_trajectory
                    if not isinstance(camera_control, dict):
                        raise PromptEnhancementError(
                            "camera_trajectory must decode to an object"
                        )
                    pe_input["camera_control"] = camera_control
                    # Keep the legacy duration mirror synchronized with the
                    # authoritative trajectory object so PE, DiT and PixelUMM derive
                    # one identical frame count.
                    if camera_control.get("durationMs") is not None:
                        pe_input["camera_duration_ms"] = camera_control["durationMs"]
                if bool(pe_input.get("camera_only_minimal")) or (
                    camera_controlled
                    and (prompt_enhancer is None or not enhancer_configured)
                ):
                    enhancement_mode = "camera-minimal"
                    enhanced = build_minimal_camera_prompt(
                        pe_input,
                        previous_session,
                    )
                elif prompt_enhancer is not None and enhancer_configured:
                    enhancement_mode = requested_enhancer_model
                    enhance_parameters = inspect.signature(
                        prompt_enhancer.enhance
                    ).parameters
                    enhance_kwargs: Dict[str, Any] = {}
                    if "stream_callback" in enhance_parameters:
                        enhance_kwargs["stream_callback"] = publish_stream
                    if "enhancer_model" in enhance_parameters:
                        enhance_kwargs["enhancer_model"] = requested_enhancer_model
                    if "reasoning_effort" in enhance_parameters:
                        enhance_kwargs["reasoning_effort"] = requested_reasoning_effort
                    if "seed" in enhance_parameters:
                        enhance_kwargs["seed"] = requested_seed
                    enhanced = prompt_enhancer.enhance(
                        pe_input,
                        previous_session,
                        **enhance_kwargs,
                    )
                else:
                    enhancement_mode = "local-fallback"
                    enhanced = build_local_fallback_prompt(
                        pe_input,
                        previous_session,
                    )
                if isinstance(prompt_enhancer, OnlinePromptEnhancer):
                    try:
                        persist_pe_turn(
                            prompt_enhancer.transcript_store,
                            pe_input,
                            previous_session,
                            enhanced,
                            enhancement_mode=enhancement_mode,
                        )
                    except Exception as transcript_exc:  # noqa: BLE001
                        # Prompt auditing is important but must not turn an
                        # otherwise valid K3 response into a failed generation
                        # when COS/FUSE has a transient metadata-write error.
                        logger.exception(
                            "Could not persist PE transcript for %s",
                            session_id,
                        )
                        store.record_error(
                            code="PE_TRANSCRIPT_WRITE_FAILED",
                            message=str(transcript_exc),
                            job_id=job_id,
                            recoverable=True,
                        )
                payload.update(
                    {
                        "prompt": enhanced.prompt,
                        "num_frames": enhanced.num_frames,
                        "segment_boundaries": enhanced.segment_boundaries,
                        "user_prompts": (
                            [raw_user_prompt]
                            * max(1, len(enhanced.segment_boundaries) - 1)
                        ),
                        "stable_prefix_carry_frames": carry_frames,
                        "stable_prefix_carry_total_frames": int(
                            previous_overlay.get("segment_frames") or carry_frames
                        ),
                        "carry_camera_trajectory": previous_overlay.get(
                            "camera_trajectory"
                        ),
                        "carry_user_prompt": str(
                            previous_overlay.get("user_prompt") or ""
                        ),
                        "carry_has_camera": bool(previous_overlay.get("has_camera")),
                        "_pe_session": enhanced.session,
                        "_pe_request_id": enhanced.request_id,
                    }
                )
                enhanced.session["last_overlay"] = {
                    "camera_trajectory": payload.get("camera_trajectory"),
                    "user_prompt": raw_user_prompt,
                    "has_camera": bool(
                        str(payload.get("camera_trajectory") or "").strip()
                    ),
                    "segment_frames": int(enhanced.num_frames),
                }
                pretty_prompt = "\n\n".join(
                    json.dumps(
                        json.loads(part),
                        ensure_ascii=False,
                        indent=2,
                    )
                    for part in enhanced.prompt.split("|||")
                )
                store.prepare(
                    job_id,
                    payload,
                    pe_prompt=pretty_prompt,
                    pe_reasoning="",
                    pe_content_preview=pretty_prompt,
                    pe_content_started=True,
                    pe_request_id=enhanced.request_id,
                    pe_transcript_path=enhanced.session.get("pe_transcript_path"),
                    prompt_enhancement_mode=enhancement_mode,
                    prompt_enhancer_reasoning_effort=(requested_reasoning_effort),
                    prompt_enhancer_seed=requested_seed,
                )
            else:
                store.prepare(job_id, payload)
            archive_generation_job(job_id)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            transcript_path = None
            if isinstance(pe_input, dict) and isinstance(
                prompt_enhancer, OnlinePromptEnhancer
            ):
                try:
                    transcript_path = persist_pe_failure(
                        prompt_enhancer.transcript_store,
                        pe_input,
                        previous_session,
                        exc,
                        enhancement_mode=enhancement_mode,
                        streamed_thinking=str(stream_state.get("reasoning") or ""),
                        streamed_response=str(stream_state.get("content") or ""),
                    )
                except Exception as transcript_exc:  # noqa: BLE001
                    store.record_error(
                        code="PE_TRANSCRIPT_WRITE_FAILED",
                        message=str(transcript_exc),
                        job_id=job_id,
                        recoverable=True,
                    )
            store.patch(
                job_id,
                status="failed",
                error=error,
                pe_transcript_path=(
                    str(transcript_path) if transcript_path is not None else None
                ),
                error_code=(
                    "SESSION_EXPIRED"
                    if error.startswith("WORLD_SESSION_EXPIRED:")
                    else None
                ),
            )
            archive_generation_job(job_id)

    @app.post("/v1/generate", status_code=202)
    def generate(
        payload: Dict[str, Any],
        background_tasks: BackgroundTasks,
        request: Request,
    ) -> Dict[str, Any]:
        submitted = dict(payload)
        try:
            effective_num_inference_steps = normalize_num_inference_steps(
                submitted.get("num_inference_steps"),
                default=int(
                    (store.worker().get("config") or {}).get("num_inference_steps")
                    or (inference_config or {}).get("num_inference_steps")
                    or DEFAULT_NUM_INFERENCE_STEPS
                ),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        submitted["num_inference_steps"] = effective_num_inference_steps
        submitted_pe_input = submitted.get("pe_input")
        submitted_reasoning_effort: str | None = None
        submitted_prompt_enhancer_seed: int | None = None
        submitted_prompt_enhancer_seed_source: str | None = None
        if isinstance(submitted_pe_input, dict):
            normalized_pe_input = dict(submitted_pe_input)
            try:
                submitted_reasoning_effort = normalize_prompt_enhancer_reasoning_effort(
                    normalized_pe_input.get(
                        "prompt_enhancer_reasoning_effort",
                        DEFAULT_PROMPT_ENHANCER_REASONING_EFFORT,
                    )
                )
                normalized_pe_input["prompt_enhancer_reasoning_effort"] = (
                    submitted_reasoning_effort
                )
                submitted_prompt_enhancer_seed = normalize_prompt_enhancer_seed(
                    normalized_pe_input.get("prompt_enhancer_seed")
                )
                if submitted_prompt_enhancer_seed is None:
                    submitted_prompt_enhancer_seed = (
                        PROMPT_ENHANCER_SEED_MIN
                        + secrets.randbelow(
                            PROMPT_ENHANCER_SEED_MAX - PROMPT_ENHANCER_SEED_MIN + 1
                        )
                    )
                    submitted_prompt_enhancer_seed_source = "generated"
                else:
                    submitted_prompt_enhancer_seed_source = "provided"
                normalized_pe_input["prompt_enhancer_seed"] = (
                    submitted_prompt_enhancer_seed
                )
                normalized_pe_input["prompt_enhancer_seed_source"] = (
                    submitted_prompt_enhancer_seed_source
                )
            except PromptEnhancementError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            submitted["pe_input"] = normalized_pe_input
        requested_session_id = str(
            (
                submitted.get("pe_input")
                if isinstance(submitted.get("pe_input"), dict)
                else {}
            ).get("session_id")
            or submitted.get("session_id")
            or ""
        ).strip()
        if submitted.get("reset_session") is True:
            scheduler = store.scheduler_snapshot()
            resident_session_ids = {
                str(session_id).strip()
                for session_id in scheduler.get(
                    "session_workers",
                    {},
                )
                if str(session_id).strip()
            }
            resident_session_ids.update(
                str(worker.get("active_session_id") or "").strip()
                for worker in scheduler.get("workers", [])
                if str(worker.get("active_session_id") or "").strip()
            )
            for active_session_id in sorted(resident_session_ids):
                if active_session_id == requested_session_id:
                    continue
                release_job = store.submit(
                    "session_release",
                    {"session_id": active_session_id},
                )
                logger.info(
                    "Queued resident session release before reset generation: "
                    "release_job=%s active_session=%s new_session=%s",
                    release_job["id"],
                    active_session_id,
                    requested_session_id,
                )
        job = store.submit("generate", submitted, status="enhancing")
        job = store.patch(
            job["id"],
            user_id=(
                unquote(str(request.headers.get("x-nomas-user-id") or "").strip())
                or "_unknown_user"
            ),
            dispatch_host_id=(
                str(request.headers.get("x-nomas-host-id") or "").strip()
                or output_dir.name
            ),
            num_inference_steps=effective_num_inference_steps,
            **(
                {"prompt_enhancer_reasoning_effort": (submitted_reasoning_effort)}
                if submitted_reasoning_effort is not None
                else {}
            ),
            **(
                {"prompt_enhancer_seed": (submitted_prompt_enhancer_seed)}
                if submitted_prompt_enhancer_seed is not None
                else {}
            ),
            **(
                {"prompt_enhancer_seed_source": (submitted_prompt_enhancer_seed_source)}
                if submitted_prompt_enhancer_seed_source is not None
                else {}
            ),
        )
        archive_generation_job(
            job["id"],
            payload_overrides={
                "num_inference_steps": effective_num_inference_steps,
            },
        )
        background_tasks.add_task(prepare_generation, job["id"], submitted)
        return job

    @app.get("/v1/generate/{job_id}")
    def generation_status(job_id: str) -> Dict[str, Any]:
        job = store.get(job_id)
        if job is None or job.get("kind") != "generate":
            raise HTTPException(status_code=404, detail="generation job not found")
        archive_generation_job(job_id)
        return job

    @app.get("/v1/admin/status")
    def admin_status() -> Dict[str, Any]:
        scheduler = store.scheduler_snapshot()
        return {
            **scheduler,
            "resources": (
                resource_monitor.snapshot(
                    running_jobs=scheduler["running_jobs"],
                )
                if resource_monitor is not None
                else {
                    "available": False,
                    "online_users": 0,
                    "gpus": [],
                }
            ),
            "jobs": store.list_jobs(limit=100),
            "recent_errors": store.recent_errors(limit=25),
        }

    @app.post("/v1/presence")
    def presence(payload: Dict[str, Any]) -> Dict[str, Any]:
        if resource_monitor is None:
            return {"online_users": 0}
        client_id = str(payload.get("client_id") or "").strip()
        if not client_id:
            raise HTTPException(status_code=400, detail="client_id is required")
        return {"online_users": resource_monitor.touch(client_id)}

    @app.delete("/v1/presence/{client_id}")
    def leave(client_id: str) -> Dict[str, Any]:
        if resource_monitor is None:
            return {"online_users": 0}
        return {"online_users": resource_monitor.remove(client_id)}

    @app.post("/v1/admin/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> Dict[str, Any]:
        try:
            return store.cancel(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/v1/sessions/{session_id}/status")
    def session_status(session_id: str) -> Dict[str, Any]:
        # Reading the reservation must not renew the worker's idle timeout.
        return {
            "session_id": session_id,
            "status": "active" if store.session_worker(session_id) else "expired",
        }

    @app.post("/v1/sessions/{session_id}/rollback", status_code=202)
    def rollback_session(session_id: str) -> Dict[str, Any]:
        clean_session_id = session_id.strip()
        if not clean_session_id:
            raise HTTPException(status_code=400, detail="session_id is required")
        job = store.submit(
            "rollback",
            {"session_id": clean_session_id},
        )
        return store.patch(job["id"], session_id=clean_session_id)

    @app.get("/v1/sessions/{session_id}/rollback/{job_id}")
    def rollback_status(session_id: str, job_id: str) -> Dict[str, Any]:
        job = store.get(job_id)
        if job is None or job.get("kind") != "rollback":
            raise HTTPException(status_code=404, detail="rollback job not found")
        if job.get("session_id") not in (None, session_id):
            raise HTTPException(status_code=404, detail="rollback job not found")
        result = job.get("result")
        if isinstance(result, dict) and result.get("session_id") not in (
            None,
            session_id,
        ):
            raise HTTPException(status_code=404, detail="rollback job not found")
        return job

    @app.post("/v1/sessions/{session_id}/release", status_code=202)
    def release_session(session_id: str) -> Dict[str, Any]:
        clean_session_id = session_id.strip()
        if not clean_session_id:
            raise HTTPException(status_code=400, detail="session_id is required")
        job = store.submit(
            "session_release",
            {"session_id": clean_session_id},
        )
        return store.patch(job["id"], session_id=clean_session_id)

    @app.get("/v1/sessions/{session_id}/release/{job_id}")
    def release_session_status(
        session_id: str,
        job_id: str,
    ) -> Dict[str, Any]:
        job = store.get(job_id)
        if job is None or job.get("kind") != "session_release":
            raise HTTPException(
                status_code=404,
                detail="session release job not found",
            )
        if job.get("session_id") not in (None, session_id):
            raise HTTPException(
                status_code=404,
                detail="session release job not found",
            )
        return job

    app.mount("/outputs", StaticFiles(directory=str(output_dir)), name="outputs")
    if gallery_dir is not None and gallery_dir.is_dir():
        app.mount(
            "/gallery",
            StaticFiles(directory=str(gallery_dir)),
            name="gallery",
        )
    if frontend_dir is not None and frontend_dir.is_dir():
        app.mount(
            "/",
            StaticFiles(directory=str(frontend_dir), html=True),
            name="frontend",
        )
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hy-Nomas lightweight web gateway")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--job-db", default="/tmp/nomas_seg_ar_jobs.json")
    parser.add_argument(
        "--output-dir",
        help="Override inference.save_path from the inference config.",
    )
    parser.add_argument(
        "--frontend-dir",
        default=str(PROJECT_ROOT / "apps" / "frontend" / "dist"),
    )
    parser.add_argument(
        "--gallery-dir",
        default="",
        help=(
            "Optional gallery media directory mounted at /gallery. Keeping it "
            "outside frontend/dist makes it survive frontend rebuilds."
        ),
    )
    parser.add_argument("--cors-origin", action="append", default=[])
    args, args.inference_overrides = parser.parse_known_args()
    return args


def main() -> None:
    import uvicorn

    args = parse_args()
    store = JobStore(args.job_db)
    config = load_demo_config(args.inference_overrides)
    args.output_dir = args.output_dir or config.inference.save_path or config.output_dir
    video_size = config.inference.video_size
    from omegaconf import ListConfig

    if isinstance(video_size, (list, tuple, ListConfig)):
        height, width = int(video_size[0]), int(video_size[-1])
    else:
        height = width = int(video_size)
    rollout = "oneshot" if config.inference.rollout_type == "one_shot" else "seg_ar"
    inference_config = {
        "rollout_type": rollout,
        "prompt_source": "subject_file" if rollout == "oneshot" else "prompt",
        "model_name": str(config.name or ""),
        "dit_weight": str(config.dit_weight or ""),
        "height": height,
        "width": width,
        "fps": int(config.inference.video_fps),
        "num_inference_steps": int(config.inference.infer_steps_video),
        "guidance_scale": float(config.inference.cfg_scale_video),
        "flow_shift": float(config.inference.infer_flow_shift_video),
        "sp_size": int(config.parallelism.sp_size),
        "tp_size": int(getattr(config.parallelism, "tp_size", 1)),
        "num_frames": (
            None if rollout == "oneshot" else int(config.inference.num_frames)
        ),
        "num_frames_source": (
            "prompt_timestamps" if rollout == "oneshot" else "request_or_config"
        ),
    }
    resource_monitor = ResourceMonitor()
    app = create_gateway_app(
        store,
        frontend_dir=(
            Path(args.frontend_dir).expanduser().resolve()
            if args.frontend_dir
            else None
        ),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        gallery_dir=(
            Path(args.gallery_dir).expanduser().resolve() if args.gallery_dir else None
        ),
        cors_origins=args.cors_origin,
        prompt_enhancer=OnlinePromptEnhancer(
            output_dir=Path(args.output_dir).expanduser().resolve(),
        ),
        inference_config=inference_config,
        resource_monitor=resource_monitor,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
