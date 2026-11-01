"""Online prompt enhancement for converting UI actions into DiT segment JSON.

It calls an existing SGLang Chat Completions-compatible endpoint directly.
It does not start or manage the inference server.

The prompt profile uses the self-contained Markdown prompts under
``apps/prompts``.
"""

from __future__ import annotations

import base64
import copy
import fcntl
import hashlib
import inspect
import io
import json
import math
import os
import random
import re
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit

import requests
from PIL import Image

from apps.durable_file import replace_or_overwrite

from apps.prompts.online_user_video_rewriter_system_prompt import (
    BASE_SYSTEM_PROMPT_PATH as SYSTEM_PROMPT_PATH,
    SYSTEM_PROMPT,
)

DEFAULT_API_URL = "http://127.0.0.1:30000/v1/chat/completions"
PROMPT_ENHANCER_KIMI_K3 = "kimi-k3"
DEFAULT_PROMPT_ENHANCER_MODEL = "Nomas-PE"
PROMPT_ENHANCER_REASONING_EFFORTS = tuple(
    value.strip()
    for value in os.environ.get(
        "PE_REASONING_EFFORTS", "none,low,medium,high,xhigh"
    ).split(",")
    if value.strip()
)
DEFAULT_PROMPT_ENHANCER_REASONING_EFFORT = os.environ.get(
    "PE_DEFAULT_REASONING_EFFORT", "none"
)
PROMPT_ENHANCER_SEED_MIN = 0
PROMPT_ENHANCER_SEED_MAX = 2**31 - 1
PROMPT_ENHANCER_OPTIONS = (
    {
        "id": DEFAULT_PROMPT_ENHANCER_MODEL,
        "label": DEFAULT_PROMPT_ENHANCER_MODEL,
        "default": True,
        "reasoning_effort_options": list(PROMPT_ENHANCER_REASONING_EFFORTS),
        "default_reasoning_effort": (DEFAULT_PROMPT_ENHANCER_REASONING_EFFORT),
    },
)
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
FIRST_TURN_PROMPT_PATH = PROMPT_DIR / "online_user_video_rewriter_first_turn.md"
INTERMEDIATE_TURN_PROMPT_PATH = (
    PROMPT_DIR / "online_user_video_rewriter_intermediate_turn.md"
)
ALLOWED_PERSPECTIVES = {
    "first-person",
    "second-person",
    "third-person follow",
    "third-person observer",
}
USER_PERSPECTIVES = {
    *ALLOWED_PERSPECTIVES,
    # Backward compatibility for clients from before the four-way selector.
    "third-person",
}
ALLOWED_STYLES = {
    "real-world footage",
    "animation/anime",
    "CG",
    "video game",
    "software/web/app/computer screen recording",
    "slideshow",
}
TIME_RE = re.compile(r"^(\d+):([0-5]\d)\.(\d{3})$")


def _read_prompt(path: Path, label: str) -> str:
    prompt = path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"{label} prompt is empty: {path}")
    return prompt


FIRST_TURN_PROMPT = _read_prompt(FIRST_TURN_PROMPT_PATH, "first-turn")
INTERMEDIATE_TURN_PROMPT = _read_prompt(
    INTERMEDIATE_TURN_PROMPT_PATH,
    "intermediate-turn",
)


@dataclass(frozen=True)
class PromptProfile:
    """Prompt and validation sources selected for one enhancer instance."""

    name: str
    system_prompt: str
    system_prompt_path: Path
    first_turn_prompt: Optional[str]
    first_turn_prompt_path: Optional[Path]
    intermediate_turn_prompt: Optional[str]
    intermediate_turn_prompt_path: Optional[Path]

    @property
    def source_paths(self) -> tuple[Path, ...]:
        return tuple(
            path
            for path in (
                self.system_prompt_path,
                self.first_turn_prompt_path,
                self.intermediate_turn_prompt_path,
            )
            if path is not None
        )

    def turn_prompt(self, *, first: bool) -> Optional[str]:
        return self.first_turn_prompt if first else self.intermediate_turn_prompt

    def turn_prompt_path(self, *, first: bool) -> Optional[Path]:
        return (
            self.first_turn_prompt_path if first else self.intermediate_turn_prompt_path
        )

    @property
    def sha256(self) -> str:
        parts = [
            self.system_prompt,
            self.first_turn_prompt or "",
            self.intermediate_turn_prompt or "",
        ]
        return hashlib.sha256("\0".join(parts).encode()).hexdigest()


DEFAULT_PROMPT_PROFILE = PromptProfile(
    name="markdown",
    system_prompt=SYSTEM_PROMPT,
    system_prompt_path=SYSTEM_PROMPT_PATH,
    first_turn_prompt=FIRST_TURN_PROMPT,
    first_turn_prompt_path=FIRST_TURN_PROMPT_PATH,
    intermediate_turn_prompt=INTERMEDIATE_TURN_PROMPT,
    intermediate_turn_prompt_path=INTERMEDIATE_TURN_PROMPT_PATH,
)


class PromptEnhancementError(RuntimeError):
    """Prompt-enhancement failure with an optional durable audit snapshot."""

    pe_audit: Optional[Dict[str, Any]] = None


def normalize_prompt_enhancer_model(value: object) -> str:
    """Preserve the endpoint's exact model ID; availability is checked separately."""
    model = DEFAULT_PROMPT_ENHANCER_MODEL if value is None else value
    if not isinstance(model, str) or not model.strip():
        raise PromptEnhancementError(
            f"prompt enhancer model must be a nonempty string; got {value!r}"
        )
    return model.strip()


def normalize_prompt_enhancer_reasoning_effort(value: object) -> str:
    """Return a shared per-request PE reasoning effort or reject the request."""
    effort = (
        DEFAULT_PROMPT_ENHANCER_REASONING_EFFORT
        if value is None
        else str(value).strip().lower()
    )
    if effort not in PROMPT_ENHANCER_REASONING_EFFORTS:
        supported = ", ".join(PROMPT_ENHANCER_REASONING_EFFORTS)
        raise PromptEnhancementError(
            "prompt_enhancer_reasoning_effort must be one of "
            f"{supported}; got {value!r}"
        )
    return effort


def normalize_prompt_enhancer_seed(value: object) -> Optional[int]:
    """Return an optional reproducibility seed for the PE endpoint."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        raise PromptEnhancementError(
            "prompt_enhancer_seed must be an integer between "
            f"{PROMPT_ENHANCER_SEED_MIN} and {PROMPT_ENHANCER_SEED_MAX}"
        )
    try:
        seed = int(value)
    except (TypeError, ValueError) as exc:
        raise PromptEnhancementError(
            "prompt_enhancer_seed must be an integer between "
            f"{PROMPT_ENHANCER_SEED_MIN} and {PROMPT_ENHANCER_SEED_MAX}"
        ) from exc
    if isinstance(value, float) and not value.is_integer():
        raise PromptEnhancementError(
            "prompt_enhancer_seed must be an integer between "
            f"{PROMPT_ENHANCER_SEED_MIN} and {PROMPT_ENHANCER_SEED_MAX}"
        )
    if not PROMPT_ENHANCER_SEED_MIN <= seed <= PROMPT_ENHANCER_SEED_MAX:
        raise PromptEnhancementError(
            "prompt_enhancer_seed must be an integer between "
            f"{PROMPT_ENHANCER_SEED_MIN} and {PROMPT_ENHANCER_SEED_MAX}; "
            f"got {value!r}"
        )
    return seed


def _first_nonempty_text(*values: object) -> str:
    """Return the first meaningful string, ignoring serialized null values."""
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() not in {"none", "null", "undefined"}:
            return text
    return ""


@dataclass
class EnhancedPrompt:
    prompt: str
    num_frames: int
    segment_boundaries: List[int]
    session: Dict[str, Any]
    raw_response: str
    request_id: Optional[str]
    user_message: Optional[Dict[str, Any]] = None
    assistant_thinking: str = ""
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    audit: Dict[str, Any] = field(default_factory=dict)


def _json_clone(value: Any) -> Any:
    """Detach a JSON-compatible value without dropping inline image payloads."""
    return json.loads(json.dumps(value, ensure_ascii=False))


def _turn_user_message(
    instructions: str,
    user_text: str,
    *,
    frames: Optional[List[tuple]] = None,
    previous_segment_start_ms: int = 0,
    segment_index: Optional[int] = None,
    frame_index_start: int = 0,
) -> Dict[str, Any]:
    """Build the exact user message sent to Kimi for the current turn."""
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": instructions},
        {
            "type": "text",
            "text": f"Current user input JSON:\n{user_text}",
        },
    ]
    if frames is not None:
        _append_segment_frame_evidence(
            content,
            frames,
            previous_segment_start_ms=previous_segment_start_ms,
            segment_index=segment_index,
            frame_index_start=frame_index_start,
        )
    return {"role": "user", "content": content}


def _append_segment_frame_evidence(
    content: List[Dict[str, Any]],
    frames: List[tuple],
    *,
    previous_segment_start_ms: int,
    segment_index: Optional[int],
    frame_index_start: int,
) -> None:
    """Append one completed segment's labeled sampled frames to a message."""
    segment_label = (
        f"Segment {segment_index}"
        if segment_index is not None
        else "the previous completed segment"
    )
    content.append(
        {
            "type": "text",
            "text": (
                f"The following {len(frames)} frames are chronological "
                f"visual evidence from completed {segment_label}. They "
                "belong to history, not to the current requested segment."
            ),
        }
    )
    for segment_frame_index, sample in enumerate(frames, start=1):
        if len(sample) == 3:
            image_url, global_sec, relative_sec = sample
            global_ms = max(0, int(round(global_sec * 1000)))
        else:
            image_url, relative_sec = sample
            global_ms = previous_segment_start_ms + max(
                0,
                int(round(relative_sec * 1000)),
            )
        relative_ms = max(0, int(round(relative_sec * 1000)))
        global_frame_index = frame_index_start + segment_frame_index
        segment_metadata = (
            f" | segment={segment_index}"
            f" | segment_frame={segment_frame_index}/{len(frames)}"
            if segment_index is not None
            else f"/{len(frames)}"
        )
        content.extend(
            [
                {
                    "type": "text",
                    "text": (
                        f"Frame {global_frame_index}{segment_metadata} | "
                        f"video_timestamp={_format_time(global_ms)}"
                        " | segment_relative_time="
                        f"{_format_time(relative_ms)}"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
            ]
        )


class PETranscriptStore:
    """Append complete PE turns to one durable JSON transcript per session."""

    def __init__(
        self,
        directory: Path,
        *,
        prompt_profile: PromptProfile = DEFAULT_PROMPT_PROFILE,
    ) -> None:
        self.directory = directory.expanduser().resolve()
        self.prompt_profile = prompt_profile
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_session_id(session_id: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", session_id).strip("._")
        digest = hashlib.sha256(session_id.encode()).hexdigest()[:12]
        return f"{safe[:80] or 'session'}-{digest}"

    def path_for(self, session_id: str) -> Path:
        return self.directory / f"{self._safe_session_id(session_id)}.json"

    def _prompt_sources(self) -> List[Dict[str, str]]:
        return [
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in self.prompt_profile.source_paths
        ]

    def append_turn(
        self,
        session_id: str,
        turn: Dict[str, Any],
    ) -> Path:
        path = self.path_for(session_id)
        lock_path = path.with_suffix(".json.lock")
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                transcript: Dict[str, Any]
                if path.is_file():
                    try:
                        loaded = json.loads(path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        loaded = None
                    transcript = loaded if isinstance(loaded, dict) else {}
                else:
                    transcript = {}
                transcript["schema_version"] = 2
                transcript["session_id"] = session_id
                transcript["system_message"] = {
                    "role": "system",
                    "content": self.prompt_profile.system_prompt,
                }
                transcript["prompt_profile"] = self.prompt_profile.name
                transcript["prompt_profile_sha256"] = self.prompt_profile.sha256
                transcript["system_prompt_sha256"] = hashlib.sha256(
                    self.prompt_profile.system_prompt.encode()
                ).hexdigest()
                transcript["prompt_sources"] = self._prompt_sources()
                turns = transcript.setdefault("turns", [])
                if not isinstance(turns, list):
                    turns = []
                    transcript["turns"] = turns
                stored_turn = _json_clone(turn)
                turn_id = stored_turn.get("turn_id")
                existing_index = next(
                    (
                        index
                        for index, existing in enumerate(turns)
                        if isinstance(existing, dict)
                        and turn_id
                        and existing.get("turn_id") == turn_id
                    ),
                    None,
                )
                if existing_index is None:
                    stored_turn["record_index"] = len(turns) + 1
                    turns.append(stored_turn)
                else:
                    stored_turn["record_index"] = turns[existing_index].get(
                        "record_index",
                        existing_index + 1,
                    )
                    turns[existing_index] = stored_turn
                transcript["updated_at"] = time.time()

                fd, raw_tmp = tempfile.mkstemp(
                    prefix=f".{path.name}.",
                    suffix=".tmp",
                    dir=str(path.parent),
                )
                tmp_path = Path(raw_tmp)
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as handle:
                        json.dump(
                            transcript,
                            handle,
                            ensure_ascii=False,
                            indent=2,
                        )
                        handle.write("\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    replace_or_overwrite(tmp_path, path)
                finally:
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except OSError:
                        # A successful fallback overwrite may leave a
                        # transient COS/FUSE handle. Do not fail the write.
                        pass
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        return path

    def latest_incomplete_turn(
        self,
        session_id: str,
        *,
        turn_number: int,
        branch_revision: int,
    ) -> Optional[Dict[str, Any]]:
        """Return the newest non-finalized audit record for this logical turn."""
        path = self.path_for(session_id)
        lock_path = path.with_suffix(".json.lock")
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
            try:
                if not path.is_file():
                    return None
                try:
                    transcript = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    return None
                turns = (
                    transcript.get("turns") if isinstance(transcript, dict) else None
                )
                if not isinstance(turns, list):
                    return None
                for turn in reversed(turns):
                    if (
                        isinstance(turn, dict)
                        and int(turn.get("turn_number") or 0) == turn_number
                        and int(turn.get("branch_revision") or 0) == branch_revision
                        and turn.get("status") not in {"succeeded", "failed"}
                    ):
                        return _json_clone(turn)
                return None
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _turn_audit_metadata(
    transcript_store: PETranscriptStore,
    request: Dict[str, Any],
    previous_session: Optional[Dict[str, Any]],
    *,
    enhancement_mode: str,
) -> Dict[str, Any]:
    first = not previous_session
    prompt_profile = transcript_store.prompt_profile
    turn_prompt = prompt_profile.turn_prompt(first=first)
    turn_prompt_path = prompt_profile.turn_prompt_path(first=first)
    return {
        "turn_id": uuid.uuid4().hex,
        "turn_number": len(list((previous_session or {}).get("dit_prompts") or [])) + 1,
        "branch_revision": int((previous_session or {}).get("branch_revision", 0) or 0),
        "created_at": time.time(),
        "enhancement_mode": enhancement_mode,
        "prompt_enhancer_reasoning_effort": (
            normalize_prompt_enhancer_reasoning_effort(
                request.get("prompt_enhancer_reasoning_effort")
            )
        ),
        "prompt_enhancer_seed": normalize_prompt_enhancer_seed(
            request.get("prompt_enhancer_seed")
        ),
        "prompt_enhancer_seed_source": (
            str(request.get("prompt_enhancer_seed_source") or "").strip() or None
        ),
        "mode": (
            "camera-control"
            if bool(request.get("camera_controlled"))
            else "text-prompt"
        ),
        "phase": "first-turn" if first else "subsequent-turn",
        "prompt_profile": prompt_profile.name,
        "turn_prompt": turn_prompt,
        "turn_prompt_path": (
            str(turn_prompt_path) if turn_prompt_path is not None else None
        ),
        "turn_prompt_sha256": (
            hashlib.sha256(turn_prompt.encode()).hexdigest()
            if turn_prompt is not None
            else None
        ),
        "input_request": _json_clone(request),
    }


def _enhancer_request_snapshot(
    client: Any,
    messages: List[Dict[str, Any]],
    *,
    reasoning_effort: str,
    seed: Optional[int],
) -> Dict[str, Any]:
    """Capture the complete logical request passed to an enhancer client."""
    request_payload = getattr(client, "request_payload", None)
    if callable(request_payload):
        request_parameters = inspect.signature(request_payload).parameters
        request_kwargs: Dict[str, Any] = {}
        if "reasoning_effort" in request_parameters:
            request_kwargs["reasoning_effort"] = reasoning_effort
        if "seed" in request_parameters:
            request_kwargs["seed"] = seed
        snapshot = request_payload(messages, **request_kwargs)
    else:
        snapshot = {
            "messages": _json_clone(messages),
            "reasoning_effort": reasoning_effort,
            "max_tokens": 4096,
            "stream": True,
        }
    snapshot["reasoning_effort"] = reasoning_effort
    if seed is not None:
        snapshot["seed"] = seed
    for key in ("api_url", "model", "timeout"):
        value = getattr(client, key, None)
        if value is not None:
            snapshot[key] = value
    return snapshot


def _audit_turn_metadata(
    transcript_store: PETranscriptStore,
    request: Dict[str, Any],
    previous_session: Optional[Dict[str, Any]],
    audit: Dict[str, Any],
    *,
    enhancement_mode: str,
) -> Dict[str, Any]:
    """Reuse the in-flight turn identity when finalizing an audit record."""
    metadata = _turn_audit_metadata(
        transcript_store,
        request,
        previous_session,
        enhancement_mode=enhancement_mode,
    )
    for key in tuple(metadata):
        if key in audit:
            metadata[key] = _json_clone(audit[key])
    return metadata


def persist_pe_turn(
    transcript_store: PETranscriptStore,
    request: Dict[str, Any],
    previous_session: Optional[Dict[str, Any]],
    enhanced: EnhancedPrompt,
    *,
    enhancement_mode: str,
) -> Path:
    """Persist a successful PE turn, including every exact K3 request."""
    user_message = enhanced.user_message or {
        "role": "user",
        "content": json.dumps(request, ensure_ascii=False, separators=(",", ":")),
    }
    try:
        parsed_response: Any = _extract_json(enhanced.raw_response)
    except PromptEnhancementError:
        try:
            parsed_response = json.loads(enhanced.prompt)
        except (TypeError, json.JSONDecodeError):
            parsed_response = None
    audit = _json_clone(enhanced.audit or {})
    attempts = audit.get("attempts")
    if not isinstance(attempts, list):
        attempts = _json_clone(enhanced.attempts)
    final_request = audit.get("final_k3_request")
    path = transcript_store.append_turn(
        str(enhanced.session["session_id"]),
        {
            **_audit_turn_metadata(
                transcript_store,
                request,
                previous_session,
                audit,
                enhancement_mode=enhancement_mode,
            ),
            "status": "succeeded",
            "user_message": _json_clone(user_message),
            "final_k3_request_attempt": len(attempts) or None,
            "assistant_thinking": enhanced.assistant_thinking,
            "assistant_response": enhanced.raw_response,
            "assistant_parsed_response": parsed_response,
            "request_id": enhanced.request_id,
            "attempts": attempts,
            **(
                {"final_k3_request": _json_clone(final_request)}
                if isinstance(final_request, dict)
                else {}
            ),
        },
    )
    enhanced.session["pe_transcript_path"] = str(path)
    enhanced.session["pe_transcript_turns"] = (
        len(list((previous_session or {}).get("dit_prompts") or [])) + 1
    )
    return path


def persist_pe_failure(
    transcript_store: PETranscriptStore,
    request: Dict[str, Any],
    previous_session: Optional[Dict[str, Any]],
    error: Exception,
    *,
    enhancement_mode: str,
    streamed_thinking: str = "",
    streamed_response: str = "",
) -> Path:
    """Persist a failed PE turn without mutating the last valid session."""
    session_id = str(
        request.get("session_id")
        or (previous_session or {}).get("session_id")
        or "unknown-session"
    )
    audit = getattr(error, "pe_audit", None)
    audit = _json_clone(audit) if isinstance(audit, dict) else {}
    metadata = _turn_audit_metadata(
        transcript_store,
        request,
        previous_session,
        enhancement_mode=enhancement_mode,
    )
    if not audit:
        audit = (
            transcript_store.latest_incomplete_turn(
                session_id,
                turn_number=int(metadata["turn_number"]),
                branch_revision=int(metadata["branch_revision"]),
            )
            or {}
        )
    attempts = audit.get("attempts")
    if not isinstance(attempts, list):
        attempts = []
    final_request = audit.get("final_k3_request")
    final_attempt = attempts[-1] if attempts else {}
    assistant_thinking = str(
        final_attempt.get("assistant_thinking") or streamed_thinking or ""
    )
    assistant_response = str(
        final_attempt.get("assistant_response") or streamed_response or ""
    )
    return transcript_store.append_turn(
        session_id,
        {
            **_audit_turn_metadata(
                transcript_store,
                request,
                previous_session,
                audit,
                enhancement_mode=enhancement_mode,
            ),
            "status": "failed",
            "user_message": audit.get("user_message"),
            "final_k3_request_attempt": len(attempts) or None,
            "assistant_thinking": assistant_thinking,
            "assistant_response": assistant_response,
            "assistant_parsed_response": audit.get("assistant_parsed_response"),
            "request_id": final_attempt.get("request_id"),
            "attempts": attempts,
            **(
                {"final_k3_request": _json_clone(final_request)}
                if isinstance(final_request, dict)
                else {}
            ),
            "error": str(error),
            "error_type": type(error).__name__,
        },
    )


def _requested_duration_frames(
    request: Dict[str, Any],
    fps: int,
    default_seconds: float,
) -> int:
    camera_control = request.get("camera_control")
    raw_duration_ms = (
        camera_control.get("durationMs")
        if isinstance(camera_control, dict)
        and camera_control.get("durationMs") is not None
        else request.get("camera_duration_ms")
    )
    if raw_duration_ms is None:
        duration_frames = max(1, int(round(default_seconds * fps)))
    else:
        try:
            duration_ms = float(raw_duration_ms)
        except (TypeError, ValueError) as exc:
            raise PromptEnhancementError("camera_duration_ms must be numeric") from exc
        if not math.isfinite(duration_ms) or duration_ms <= 0:
            raise PromptEnhancementError("camera_duration_ms must be positive")
        duration_frames = max(1, int(round(duration_ms * fps / 1000.0)))
    return duration_frames


def _infer_fallback_perspective(value: str, description: str) -> str:
    """Resolve only the legacy grouped viewpoint to a concrete viewpoint."""
    if value == "third-person":
        lowered = description.lower()
        if any(
            marker in lowered
            for marker in (
                "fixed camera",
                "static camera",
                "surveillance",
                "observer",
                "wide shot",
            )
        ):
            return "third-person observer"
        return "third-person follow"
    return value


def _infer_fallback_style(description: str) -> str:
    """Choose a conservative style when K3 is unavailable."""
    lowered = description.lower()
    if any(
        marker in lowered
        for marker in (
            "anime",
            "animated",
            "animation",
            "cartoon",
            "illustrated",
        )
    ):
        return "animation/anime"
    if any(marker in lowered for marker in ("video game", "gameplay", "game engine")):
        return "video game"
    if any(marker in lowered for marker in ("cgi", "3d render", "3d-render")):
        return "CG"
    if any(
        marker in lowered
        for marker in ("screen recording", "website", "software ui", "app ui")
    ):
        return "software/web/app/computer screen recording"
    if any(marker in lowered for marker in ("slideshow", "presentation slides")):
        return "slideshow"
    return "real-world footage"


def _perspective_matches_request(
    requested: str,
    resolved: Any,
    *,
    allow_inference: bool,
) -> bool:
    if not isinstance(resolved, str) or resolved not in ALLOWED_PERSPECTIVES:
        return False
    if requested in ALLOWED_PERSPECTIVES:
        return resolved == requested
    if allow_inference and requested == "third-person":
        return resolved in {"third-person follow", "third-person observer"}
    return False


def _retime_segment(
    segment: Dict[str, Any],
    *,
    start_ms: int,
    duration_frames: int,
    fps: int,
) -> Dict[str, Any]:
    """Retarget a segment and its caption timestamps to an exact frame count."""
    result = json.loads(json.dumps(segment, ensure_ascii=False))
    # Round upward so parsing the millisecond timestamp with int(seconds * fps)
    # deterministically recovers duration_frames.
    duration_ms = int(math.ceil(duration_frames * 1000.0 / fps))
    old_start = _parse_time(result.get("start_time"))
    old_end = _parse_time(result.get("end_time"))
    old_duration = (
        old_end - old_start
        if old_start is not None and old_end is not None and old_end > old_start
        else duration_ms
    )
    end_ms = start_ms + duration_ms
    result["start_time"] = _format_time(start_ms)
    result["end_time"] = _format_time(end_ms)

    actions = result.get("contained_actions")
    if isinstance(actions, list):
        for action in actions:
            if not isinstance(action, dict):
                continue
            action_start = _parse_time(action.get("start_time"))
            action_end = _parse_time(action.get("end_time"))
            if action_start is None or action_end is None:
                action["start_time"] = _format_time(start_ms)
                action["end_time"] = _format_time(end_ms)
                continue
            source_start = old_start or 0
            scaled_start = start_ms + round(
                max(0, action_start - source_start) * duration_ms / old_duration
            )
            scaled_end = start_ms + round(
                max(0, action_end - source_start) * duration_ms / old_duration
            )
            scaled_start = min(max(start_ms, scaled_start), end_ms - 1)
            scaled_end = min(max(scaled_start + 1, scaled_end), end_ms)
            action["start_time"] = _format_time(scaled_start)
            action["end_time"] = _format_time(scaled_end)
    return result


def build_minimal_camera_prompt(
    request: Dict[str, Any],
    previous_session: Optional[Dict[str, Any]],
    *,
    fps: int = 24,
    duration_seconds: float = 5.0,
) -> EnhancedPrompt:
    """Build a camera-only segment with no semantic text conditioning.

    The raw prompt retains segment timestamps so the worker can validate the
    requested frame count and rollback can recover the previous segment end.
    The standard Stage-9 keep-key shaping removes those segment timestamps
    before text encoding, leaving exactly ``{"segments": [{}]}``.
    """
    session_id = str(request.get("session_id") or "").strip()
    if not session_id:
        raise PromptEnhancementError("session_id is required")
    if fps <= 0 or duration_seconds <= 0:
        raise PromptEnhancementError("fps and duration_seconds must be positive")

    duration_frames = _requested_duration_frames(
        request,
        fps,
        duration_seconds,
    )
    duration_ms = int(math.ceil(duration_frames * 1000.0 / fps))
    start_ms = (
        0
        if not previous_session
        else (_parse_time((previous_session or {}).get("previous_end")) or 0)
    )
    end_ms = start_ms + duration_ms
    prompt = json.dumps(
        {
            "segments": [
                {
                    "start_time": _format_time(start_ms),
                    "end_time": _format_time(end_ms),
                }
            ]
        },
        ensure_ascii=False,
    )

    prior = previous_session or {}
    prior_lengths = list(prior.get("segment_frame_lengths") or [])
    requested_perspective = str(
        request.get("perspective")
        or prior.get("requested_perspective")
        or prior.get("perspective")
        or ""
    )
    session = {
        **dict(prior),
        "session_id": session_id,
        "requested_perspective": requested_perspective,
        "perspective": str(prior.get("perspective") or requested_perspective),
        "style": str(request.get("style") or prior.get("style") or ""),
        "subject": str(
            request.get("subject_description")
            or request.get("subject")
            or prior.get("subject")
            or ""
        ),
        "environment_description": str(
            request.get("environment_description")
            or prior.get("environment_description")
            or ""
        ),
        "previous_end": _format_time(end_ms),
        "messages": list(prior.get("messages") or []),
        "dit_prompts": [
            *list(prior.get("dit_prompts") or []),
            prompt,
        ],
        "segment_frame_lengths": [*prior_lengths, duration_frames],
        "reference_hidden_frames": int(prior.get("reference_hidden_frames", 0)),
        "reference_history_url": prior.get("reference_history_url"),
        "video_url": prior.get("video_url"),
        "last_result": prior.get("last_result"),
        "result_history": list(prior.get("result_history") or []),
        "branch_revision": int(prior.get("branch_revision", 0) or 0),
        "rollback_pending": False,
        "minimal_camera_prompt": True,
        "updated_at": time.time(),
    }
    return EnhancedPrompt(
        prompt=prompt,
        num_frames=duration_frames,
        segment_boundaries=[0, duration_frames],
        session=session,
        raw_response=prompt,
        request_id=None,
    )


def build_local_fallback_prompt(
    request: Dict[str, Any],
    previous_session: Optional[Dict[str, Any]],
    *,
    fps: int = 24,
    duration_seconds: float = 5.0,
) -> EnhancedPrompt:
    """Build a valid segment prompt when online enhancement is unavailable."""
    first = not previous_session
    uses_inferred_initial_contract = first and (
        "subject_description" in request or "environment_description" in request
    )
    requested_perspective = _first_nonempty_text(
        request.get("perspective"),
        (previous_session or {}).get("requested_perspective"),
        (previous_session or {}).get("perspective"),
    )
    subject = str(
        request.get("subject_description")
        or request.get("subject")
        or (previous_session or {}).get("subject")
        or ""
    ).strip()
    environment = str(
        request.get("environment_description")
        or (previous_session or {}).get("environment_description")
        or ""
    ).strip()
    action = str(request.get("action") or "").strip()
    if uses_inferred_initial_contract:
        if requested_perspective not in USER_PERSPECTIVES:
            raise PromptEnhancementError(
                f"unsupported user perspective: {requested_perspective!r}"
            )
        perspective = _infer_fallback_perspective(
            requested_perspective,
            f"{subject} {environment}",
        )
        style = _infer_fallback_style(f"{subject} {environment}")
        action = environment
    else:
        perspective = _first_nonempty_text(
            request.get("perspective"),
            (previous_session or {}).get("perspective"),
            (previous_session or {}).get("requested_perspective"),
        )
        style = _first_nonempty_text(
            request.get("style"),
            (previous_session or {}).get("style"),
        )
    if perspective not in ALLOWED_PERSPECTIVES:
        raise PromptEnhancementError(f"unsupported perspective: {perspective!r}")
    if style not in ALLOWED_STYLES:
        raise PromptEnhancementError(f"unsupported style: {style!r}")
    if not subject or not action:
        raise PromptEnhancementError(
            "subject description and environment description must be non-empty"
            if uses_inferred_initial_contract
            else "subject and action must be non-empty"
        )
    if fps <= 0 or duration_seconds <= 0:
        raise PromptEnhancementError("fps and duration_seconds must be positive")

    session_id = str(request.get("session_id") or uuid.uuid4())
    start_ms = (
        0 if first else (_parse_time((previous_session or {}).get("previous_end")) or 0)
    )
    duration_frames = _requested_duration_frames(
        request,
        fps,
        duration_seconds,
    )
    duration_ms = int(math.ceil(duration_frames * 1000.0 / fps))
    end_ms = start_ms + duration_ms
    start_time = _format_time(start_ms)
    end_time = _format_time(end_ms)
    local_end_time = _format_time(duration_ms)
    wrapped_action = f"<Action1><|Subject1|> {action}</Action1>"

    segment = {
        "start_time": start_time,
        "end_time": end_time,
        "visual_description_long": (
            (f"{subject} Visual style: {style}. Perspective: {perspective}. {action}")
            if first
            else None
        ),
        "description_change": (
            None
            if first
            else (
                "The same world, subject identity, viewpoint, lighting, and "
                f"environment continue. The next change is: {action}"
            )
        ),
        "contained_actions": [
            {
                "start_time": "00:00.000",
                "end_time": local_end_time,
                "action_description": wrapped_action,
            }
        ],
    }
    if first:
        dit_item = {
            "perspective_view": perspective,
            "Style": style,
            "subject_list": [
                {
                    "subject_id": "<|Subject1|>",
                    "display_name": subject[:120],
                    "subject_attribute": subject,
                }
            ],
            "segments": [segment],
        }
    else:
        dit_item = {"segments": [segment]}

    prompt = json.dumps(dit_item, ensure_ascii=False)
    prior_lengths = list((previous_session or {}).get("segment_frame_lengths") or [])
    session = {
        "session_id": session_id,
        "requested_perspective": requested_perspective,
        "perspective": perspective,
        "style": style,
        "subject": subject,
        "environment_description": (
            environment
            or str((previous_session or {}).get("environment_description") or "")
        ),
        "previous_end": end_time,
        "messages": list((previous_session or {}).get("messages") or []),
        "dit_prompts": [
            *list((previous_session or {}).get("dit_prompts") or []),
            prompt,
        ],
        "segment_frame_lengths": [*prior_lengths, duration_frames],
        "reference_hidden_frames": int((previous_session or {}).get("reference_hidden_frames", 0)),
        "reference_history_url": (previous_session or {}).get("reference_history_url"),
        "video_url": (previous_session or {}).get("video_url"),
        "last_result": (previous_session or {}).get("last_result"),
        "result_history": list((previous_session or {}).get("result_history") or []),
        "branch_revision": int((previous_session or {}).get("branch_revision", 0) or 0),
        "rollback_pending": False,
        "local_fallback": True,
        "updated_at": time.time(),
    }
    return EnhancedPrompt(
        prompt=prompt,
        num_frames=duration_frames,
        segment_boundaries=[0, duration_frames],
        session=session,
        raw_response=prompt,
        request_id=None,
    )


def _parse_time(value: Any) -> Optional[int]:
    if not isinstance(value, str):
        return None
    match = TIME_RE.fullmatch(value)
    if match is None:
        return None
    minutes, seconds, milliseconds = (int(part) for part in match.groups())
    return (minutes * 60 + seconds) * 1000 + milliseconds


def _format_time(milliseconds: int) -> str:
    milliseconds = max(0, int(milliseconds))
    minutes, remainder = divmod(milliseconds, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{minutes:02d}:{seconds:02d}.{millis:03d}"


def _session_subject_ids(session: Optional[Dict[str, Any]]) -> tuple[str, ...]:
    if not session:
        return ()
    stored = session.get("subject_ids")
    if isinstance(stored, list):
        ids = tuple(value for value in stored if isinstance(value, str) and value)
        if ids:
            return ids
    prompts = session.get("dit_prompts")
    if isinstance(prompts, list) and prompts:
        try:
            first_prompt = json.loads(prompts[0])
        except (TypeError, json.JSONDecodeError):
            return ()
        subjects = first_prompt.get("subject_list")
        if isinstance(subjects, list):
            return tuple(
                subject["subject_id"]
                for subject in subjects
                if isinstance(subject, dict)
                and isinstance(subject.get("subject_id"), str)
            )
    return ()


def _parse_message_json(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, str):
        return None
    clean = re.sub(
        r"<think>.*?</think>",
        "",
        value,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()
    candidates = [clean]
    start, end = clean.find("{"), clean.rfind("}")
    if start >= 0 and end > start:
        candidates.append(clean[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _parse_user_message_json(value: Any) -> Optional[Dict[str, Any]]:
    """Extract the current-input JSON from text-only or multimodal user content."""
    if isinstance(value, str):
        return _parse_message_json(value)
    if not isinstance(value, list):
        return None
    prefix = "Current user input JSON:\n"
    fallback_texts = []
    for item in value:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not isinstance(text, str):
            continue
        if text.startswith(prefix):
            parsed = _parse_message_json(text[len(prefix) :])
            if parsed is not None:
                return parsed
        fallback_texts.append(text)
    for text in fallback_texts:
        parsed = _parse_message_json(text)
        if parsed is not None:
            return parsed
    return None


def _normalize_subject_input(value: Any) -> Any:
    if isinstance(value, dict):
        description = value.get("description")
        if isinstance(description, str) and description.strip():
            return description.strip()
        return value
    if isinstance(value, list):
        return [_normalize_subject_input(item) for item in value]
    return value


def _normalize_camera_control(value: Any) -> Dict[str, Any]:
    """Reduce camera control to the duration and key events sent to K3."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise PromptEnhancementError(
                "camera_control must be a valid JSON object"
            ) from exc
    if not isinstance(value, dict):
        raise PromptEnhancementError("camera_control must be an object")

    key_events = value.get("keyEvents")
    if not isinstance(key_events, list):
        raise PromptEnhancementError("camera_control.keyEvents must be a list")
    duration_ms = value.get("durationMs")
    try:
        duration_ms = float(duration_ms)
    except (TypeError, ValueError) as exc:
        raise PromptEnhancementError(
            "camera_control.durationMs must be numeric"
        ) from exc
    if not math.isfinite(duration_ms) or duration_ms <= 0:
        raise PromptEnhancementError("camera_control.durationMs must be positive")

    # K3 reasons only from explicit input intent. The original request retains
    # the complete camera data independently for DiT conditioning.
    return {
        "durationMs": value.get("durationMs"),
        "keyEvents": json.loads(json.dumps(key_events, ensure_ascii=False)),
    }


def _normalize_history_camera_control(value: Any) -> Dict[str, Any]:
    """Normalize historical camera turns to the current PE protocol."""
    if isinstance(value, dict):
        return _normalize_camera_control(value)
    if isinstance(value, str) and value.strip() != "<camera-control-stream>":
        return _normalize_camera_control(value)
    raise PromptEnhancementError(
        "legacy camera-control placeholder cannot be replayed because its "
        "key events are unavailable; start a new session"
    )


def _normalize_history_messages(
    history: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Migrate persisted pre-sync turns to the current PE chat protocol."""
    normalized: List[Dict[str, Any]] = []
    user_turn = 0
    for message in history:
        current = json.loads(json.dumps(message, ensure_ascii=False))
        role = current.get("role")
        content = current.get("content")
        parsed = (
            _parse_user_message_json(content)
            if role == "user"
            else _parse_message_json(content)
        )
        if role == "user":
            if parsed is not None:
                if user_turn == 0 and isinstance(parsed.get("initial"), dict):
                    initial = parsed["initial"]
                    clean_user = {
                        "perspective_view": parsed.get("perspective_view"),
                        "Style": parsed.get("Style"),
                        "initial": {
                            "subject": _normalize_subject_input(initial.get("subject")),
                            "caption": initial.get("caption"),
                        },
                    }
                    if (
                        "camera_control" in parsed
                        or initial.get("control_mode") == "camera"
                    ):
                        clean_user["camera_control"] = (
                            _normalize_history_camera_control(
                                parsed.get("camera_control")
                            )
                        )
                elif "camera_control" in parsed or (
                    parsed.get("control_mode") == "camera"
                    and not parsed.get("subsequent_text_prompt")
                    and not parsed.get("caption")
                ):
                    clean_user = {
                        "camera_control": (
                            _normalize_history_camera_control(
                                parsed.get("camera_control")
                            )
                        )
                    }
                elif isinstance(
                    parsed.get("subsequent_text_prompt"),
                    str,
                ):
                    clean_user = {
                        "subsequent_text_prompt": parsed["subsequent_text_prompt"]
                    }
                elif isinstance(parsed.get("caption"), str):
                    # Migrate text turns persisted before the subsequent-turn
                    # wire key was restored from ``caption``.
                    clean_user = {"subsequent_text_prompt": parsed["caption"]}
                else:
                    clean_user = parsed
                current["content"] = json.dumps(
                    clean_user,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            user_turn += 1
        elif role == "assistant" and parsed is not None:
            if "segment" in parsed and "segments" not in parsed:
                parsed["segments"] = [parsed.pop("segment")]
            current["content"] = json.dumps(
                parsed,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        normalized.append(current)
    return normalized


def _normalize_action_timestamps(segment: Dict[str, Any]) -> Dict[str, Any]:
    """Keep the segment on the global timeline and localize only its actions.

    This mirrors ``dataset.json_prompt_normalize_action_timestamp=true`` used
    by the Stage-9 model: segment start/end timestamps remain absolute across
    the AR sequence, while contained-action timestamps are relative to the
    current segment start.
    """
    start = _parse_time(segment["start_time"])
    end = _parse_time(segment["end_time"])
    if start is None or end is None:
        raise PromptEnhancementError(
            "cannot normalize actions for invalid segment timestamps"
        )
    result = json.loads(json.dumps(segment, ensure_ascii=False))
    for action in result.get("contained_actions") or []:
        action_start = _parse_time(action.get("start_time"))
        action_end = _parse_time(action.get("end_time"))
        if action_start is not None:
            action["start_time"] = _format_time(action_start - start)
        if action_end is not None:
            action["end_time"] = _format_time(action_end - start)
    return result


def _extract_json(text: str) -> Dict[str, Any]:
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean)
        clean = re.sub(r"\s*```$", "", clean)
    candidates = [clean]
    start, end = clean.find("{"), clean.rfind("}")
    if start >= 0 and end > start:
        candidates.append(clean[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise PromptEnhancementError("Kimi-K3 did not return a JSON object")


def _validate_segment(
    segment: Any,
    *,
    first: bool,
    previous_end: Optional[str],
    camera_controlled: bool = False,
) -> List[str]:
    errors: List[str] = []
    if not isinstance(segment, dict):
        return ["segment must be an object"]
    required = {
        "start_time",
        "end_time",
        "visual_description_long",
        "description_change",
        "contained_actions",
    }
    if set(segment) != required:
        errors.append(f"segment keys must be exactly {sorted(required)}")
    start = _parse_time(segment.get("start_time"))
    end = _parse_time(segment.get("end_time"))
    if start is None or end is None or end <= start:
        errors.append("segment timestamps must be valid and strictly increasing")
    if first:
        if segment.get("start_time") != "00:00.000":
            errors.append("first segment must start at 00:00.000")
        if not isinstance(segment.get("visual_description_long"), str):
            errors.append("first segment requires visual_description_long")
        if segment.get("description_change") is not None:
            errors.append("first segment description_change must be null")
    else:
        if segment.get("start_time") != previous_end:
            errors.append("follow-up start_time must equal previous end_time")
        if segment.get("visual_description_long") is not None:
            errors.append("follow-up visual_description_long must be null")
        description_change = segment.get("description_change")
        if not isinstance(description_change, str):
            errors.append("follow-up requires description_change")
        elif not description_change.startswith("Compared to the previous segment, "):
            errors.append(
                "follow-up description_change must start with "
                "'Compared to the previous segment, '"
            )
    actions = segment.get("contained_actions")
    if not isinstance(actions, list):
        errors.append("contained_actions must be a list")
    else:
        if camera_controlled and actions:
            errors.append("camera-controlled segment must not contain actions")
        prior_end = start
        for index, action in enumerate(actions, 1):
            if not isinstance(action, dict):
                errors.append(f"action {index} must be an object")
                continue
            if set(action) != {"start_time", "end_time", "action_description"}:
                errors.append(f"action {index} has invalid keys")
            action_start = _parse_time(action.get("start_time"))
            action_end = _parse_time(action.get("end_time"))
            if (
                action_start is None
                or action_end is None
                or start is None
                or end is None
                or action_start < start
                or action_end > end
                or action_end <= action_start
                or (prior_end is not None and action_start < prior_end)
            ):
                errors.append(f"action {index} has invalid timeline")
            prior_end = action_end
            description = action.get("action_description")
            if (
                not isinstance(description, str)
                or f"<Action{index}>" not in description
            ):
                errors.append(f"action {index} requires matching Action wrapper")
    return errors


def validate_response(
    parsed: Dict[str, Any],
    *,
    first: bool,
    perspective: str,
    style: Optional[str],
    previous_end: Optional[str],
    expected_subject_ids: Optional[tuple[str, ...]] = None,
    camera_controlled: bool = False,
    allow_perspective_inference: bool = False,
    allow_style_inference: bool = False,
) -> List[str]:
    errors: List[str] = []
    if first:
        required = {"perspective_view", "Style", "subject_list", "segments"}
        if set(parsed) != required:
            errors.append(f"first response keys must be exactly {sorted(required)}")
        if not _perspective_matches_request(
            perspective,
            parsed.get("perspective_view"),
            allow_inference=allow_perspective_inference,
        ):
            errors.append("perspective_view must match the selected user perspective")
        resolved_style = parsed.get("Style")
        if allow_style_inference:
            if resolved_style not in ALLOWED_STYLES:
                errors.append("Style must be one of the supported styles")
        elif resolved_style != style:
            errors.append("Style must exactly match the UI selection")
        subjects = parsed.get("subject_list")
        if not isinstance(subjects, list) or not subjects:
            errors.append("subject_list must contain at least one subject")
        else:
            subject_ids = [
                subject.get("subject_id") if isinstance(subject, dict) else None
                for subject in subjects
            ]
            sequential = [
                f"<|Subject{index}|>" for index in range(1, len(subjects) + 1)
            ]
            if subject_ids != sequential:
                errors.append("subject IDs must be sequential from <|Subject1|>")
    elif set(parsed) != {"segments"}:
        errors.append('follow-up response must contain exactly "segments"')

    segments = parsed.get("segments")
    if not isinstance(segments, list) or len(segments) != 1:
        errors.append("segments must contain exactly one segment")
        segment = None
    else:
        segment = segments[0]
    errors.extend(
        _validate_segment(
            segment,
            first=first,
            previous_end=previous_end,
            camera_controlled=camera_controlled,
        )
    )
    if not first and expected_subject_ids:
        actions = segment.get("contained_actions") if isinstance(segment, dict) else []
        for action in actions if isinstance(actions, list) else []:
            if not isinstance(action, dict):
                continue
            description = action.get("action_description")
            if not isinstance(description, str):
                continue
            used_ids = {
                f"<|Subject{number}|>"
                for number in re.findall(r"<\|Subject(\d+)\|>", description)
            }
            unknown_ids = sorted(used_ids - set(expected_subject_ids))
            if unknown_ids:
                errors.append(
                    f"action contains undefined Subject token(s) {unknown_ids}"
                )
    return errors


class KimiK3Client:
    """SGLang OpenAI-compatible Chat Completions client."""

    def __init__(
        self,
        *,
        api_url: str = DEFAULT_API_URL,
        model: str = DEFAULT_PROMPT_ENHANCER_MODEL,
        timeout: int = 300,
        retries: int = 3,
        reasoning_effort: str = DEFAULT_PROMPT_ENHANCER_REASONING_EFFORT,
        api_key: Optional[str] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        parsed = urlsplit(api_url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("PE_API_URL must be an HTTP(S) API address")
        path = parsed.path.rstrip("/")
        if path.endswith("/chat/completions"):
            path = path[: -len("/chat/completions")]
        if not path.endswith("/v1"):
            path += "/v1"
        self.api_url = urlunsplit(parsed._replace(path=path + "/chat/completions", fragment=""))
        self.models_url = urlunsplit(parsed._replace(path=path + "/models", fragment=""))
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.retries = retries
        try:
            self.reasoning_effort = normalize_prompt_enhancer_reasoning_effort(
                reasoning_effort
            )
        except PromptEnhancementError as exc:
            raise ValueError(str(exc)) from exc
        self.session = session or requests.Session()
        self.session.trust_env = False
        self._model_lock = threading.Lock()
        self._model_ids: List[str] = []
        self._models_refresh_at = 0.0
        self.models_error: Optional[str] = None

    @property
    def display_name(self) -> str:
        return "Kimi-K3" if self.model == PROMPT_ENHANCER_KIMI_K3 else self.model

    def available_models(self) -> List[str]:
        """Cache discovery so frontend config polling does not hammer the server."""
        with self._model_lock:
            if time.monotonic() < self._models_refresh_at:
                return list(self._model_ids)
            try:
                headers = {"Accept": "application/json"}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                response = self.session.get(
                    self.models_url, headers=headers, timeout=(3, 5),
                )
                response.raise_for_status()
                data = response.json()
                entries = data.get("data") if isinstance(data, dict) else None
                if not isinstance(entries, list):
                    raise ValueError("model discovery response must contain a data list")
                ids = [
                    item["id"]
                    for item in entries
                    if isinstance(item, dict)
                    and isinstance(item.get("id"), str)
                    and item["id"].strip()
                    and item["id"] == item["id"].strip()
                ]
                self._model_ids = list(dict.fromkeys(ids))
                if not self._model_ids:
                    raise ValueError("API advertises no available models")
                self.models_error = None
                self._models_refresh_at = time.monotonic() + 60
            except (requests.RequestException, ValueError) as exc:
                self._model_ids = []
                self.models_error = str(exc)
                self._models_refresh_at = time.monotonic() + 5
            return list(self._model_ids)

    def for_model(self, model: str) -> KimiK3Client:
        # Keep the selected model request-local, including the durable audit.
        client = copy.copy(self)
        client.model = model
        return client

    @property
    def configured(self) -> bool:
        return bool(self.api_url and self.model)

    def request_payload(
        self,
        messages: List[Dict[str, Any]],
        *,
        reasoning_effort: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        effective_effort = normalize_prompt_enhancer_reasoning_effort(
            self.reasoning_effort if reasoning_effort is None else reasoning_effort
        )
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 4096,
            "stream": True,
        }
        if effective_effort == "none":
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        else:
            payload["reasoning_effort"] = effective_effort
        normalized_seed = normalize_prompt_enhancer_seed(seed)
        if normalized_seed is not None:
            payload["seed"] = normalized_seed
        return payload

    def complete(
        self,
        messages: List[Dict[str, Any]],
        *,
        stream_callback: Optional[Callable[[str, str], None]] = None,
        reasoning_effort: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> tuple[str, Optional[str]]:
        headers = self.request_headers()
        payload = self.request_payload(
            messages,
            reasoning_effort=reasoning_effort,
            seed=seed,
        )
        last_error: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                response = self.session.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=(20, self.timeout + 30),
                    stream=bool(payload.get("stream", False)),
                )
                if response.status_code == 200:
                    content_type = response.headers.get("Content-Type", "").lower()
                    if "text/event-stream" not in content_type:
                        data = response.json()
                        choices = data.get("choices") or []
                        message = choices[0].get("message") or {} if choices else {}
                        reasoning = message.get("reasoning_content")
                        if (
                            isinstance(reasoning, str)
                            and reasoning
                            and stream_callback is not None
                        ):
                            stream_callback("reasoning", reasoning)
                        content = message.get("content")
                        if isinstance(content, str) and content.strip():
                            return (
                                content.strip(),
                                response.headers.get("X-Request-Id") or data.get("id"),
                            )
                        raise PromptEnhancementError(f"{self.display_name} returned empty content")
                    content_parts: List[str] = []
                    response_id = response.headers.get("X-Request-Id")
                    for raw_line in response.iter_lines(decode_unicode=True):
                        if not raw_line:
                            continue
                        line = (
                            raw_line
                            if isinstance(raw_line, str)
                            else raw_line.decode("utf-8", errors="replace")
                        )
                        if not line.startswith("data:"):
                            continue
                        event = line[5:].strip()
                        if not event or event == "[DONE]":
                            continue
                        chunk = json.loads(event)
                        response_id = response_id or chunk.get("id")
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        reasoning = delta.get("reasoning_content")
                        if isinstance(reasoning, str) and reasoning:
                            if stream_callback is not None:
                                stream_callback("reasoning", reasoning)
                        content = delta.get("content")
                        if isinstance(content, str) and content:
                            content_parts.append(content)
                            if stream_callback is not None:
                                stream_callback("content", content)
                    content = "".join(content_parts).strip()
                    if content:
                        return content, response_id
                    raise PromptEnhancementError(f"{self.display_name} returned empty content")
                last_error = PromptEnhancementError(
                    self.response_error(response)
                )
                if response.status_code not in {408, 409, 429, 500, 502, 503, 504}:
                    raise last_error
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
            if attempt < self.retries:
                time.sleep(min(8.0, 2**attempt) + random.random())
        raise PromptEnhancementError(f"{self.display_name} request failed: {last_error}")

    def request_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def response_error(self, response) -> str:
        return f"{self.display_name} HTTP {response.status_code}: {response.text[:500]}"


class K3APIClient(KimiK3Client):
    """K3 API using the same app/API-key credentials as image captioning."""

    def __init__(self, *, session: Optional[requests.Session] = None) -> None:
        api_url = os.environ.get("K3_API_URL", "").strip()
        super().__init__(
            api_url=api_url or DEFAULT_API_URL,
            model=os.environ.get("K3_MODEL", PROMPT_ENHANCER_KIMI_K3),
            api_key=os.environ.get("K3_API_KEY") or os.environ.get("AIHUB_API_KEY", ""),
            timeout=int(os.environ.get("K3_API_TIMEOUT", "300")),
            retries=int(os.environ.get("K3_API_RETRIES", "3")),
            session=session,
        )
        self._endpoint_configured = bool(api_url)
        self.provider = os.environ.get("K3_API_PROVIDER", "moonshot" if self.api_key else "ali")
        self.app_id = os.environ.get("APP_ID", "")
        self.app_key = os.environ.get("APP_KEY", "")

    @property
    def configured(self) -> bool:
        def valid(value):
            return bool(value and value.isascii() and not any(c.isspace() for c in value))
        return bool(self._endpoint_configured and (
            valid(self.api_key) if self.api_key else valid(self.app_id) and valid(self.app_key)
        ))

    def available_models(self) -> List[str]:
        # This authenticated service has no SGLang model-discovery endpoint.
        return [self.model] if self.configured else []

    def request_headers(self) -> Dict[str, str]:
        if not self.configured:
            raise PromptEnhancementError("K3 API endpoint or credentials are not configured")
        if self.api_key:
            auth = f"Bearer {self.api_key}?provider={self.provider}"
        else:
            request_id = uuid.uuid4().hex
            auth = (
                f"Bearer {self.app_id}:{self.app_key}"
                f"?provider={self.provider}&model={self.model}&timeout={self.timeout}"
                f"&max_retry=2&request_id={request_id}&cache_task_id=pe-{request_id}"
            )
        return {"Authorization": auth, "Content-Type": "application/json"}

    def request_payload(self, messages, *, reasoning_effort=None, seed=None):
        payload = super().request_payload(messages, reasoning_effort=reasoning_effort, seed=seed)
        payload.pop("chat_template_kwargs", None)
        payload["reasoning_effort"] = normalize_prompt_enhancer_reasoning_effort(
            self.reasoning_effort if reasoning_effort is None else reasoning_effort
        )
        if self.api_key:
            payload.pop("max_tokens")
            payload.update(
                max_completion_tokens=int(os.environ.get("K3_MAX_COMPLETION_TOKENS", "16384")),
                response_format={"type": "json_object"}, temperature=1.0, top_p=0.95, stream=False,
            )
        return payload

    def response_error(self, response) -> str:
        # The internal service can echo authentication details in error bodies.
        return f"Kimi-K3 API HTTP {response.status_code}"

    def complete(self, messages, *, stream_callback=None, reasoning_effort=None, seed=None):
        try:
            return super().complete(messages, stream_callback=stream_callback,
                                    reasoning_effort=reasoning_effort, seed=seed)
        except PromptEnhancementError as exc:
            message = str(exc)
            for secret in (self.api_key, self.app_id, self.app_key):
                if secret:
                    message = message.replace(secret, "[REDACTED]")
            raise PromptEnhancementError(message) from None


class OnlinePromptEnhancer:
    def __init__(
        self,
        *,
        output_dir: Path,
        client: Optional[Any] = None,
        fps: int = 24,
        max_condition_frames: int = 40,
        condition_short_side: int = 480,
        transcript_dir: Optional[Path] = None,
        k3_client: Optional[K3APIClient] = None,
    ) -> None:
        self.output_dir = output_dir.resolve()
        self.client = client or KimiK3Client(
            api_url=os.environ.get("PE_API_URL", DEFAULT_API_URL),
            model=os.environ.get("PE_MODEL", DEFAULT_PROMPT_ENHANCER_MODEL),
            timeout=int(os.environ.get("PE_API_TIMEOUT", "300")),
            retries=int(os.environ.get("PE_API_RETRIES", "3")),
            api_key=os.environ.get("PE_API_KEY"),
        )
        self.k3_client = k3_client if k3_client is not None else K3APIClient()
        self.fps = fps
        self.max_condition_frames = max_condition_frames
        self.condition_short_side = condition_short_side
        self.prompt_profile = DEFAULT_PROMPT_PROFILE
        self.prompt_profile_sha256 = self.prompt_profile.sha256
        configured_transcript_dir = os.environ.get("PE_TRANSCRIPT_DIR")
        self.transcript_store = PETranscriptStore(
            transcript_dir
            or (
                Path(configured_transcript_dir)
                if configured_transcript_dir
                else self.output_dir / "pe_transcripts"
            ),
            prompt_profile=self.prompt_profile,
        )

    def _resize_condition_frame(self, image: Image.Image) -> Image.Image:
        """Downscale a K3 conditioning frame by its short side.

        Preserve the source aspect ratio and avoid upscaling already-small
        frames. For the 1280x704 demo output this produces approximately
        873x480 rather than constraining the long side to 768.
        """
        width, height = image.size
        short_side = min(width, height)
        target = int(self.condition_short_side)
        if target <= 0 or short_side <= target:
            return image
        scale = target / short_side
        resized = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        return image.resize(resized, Image.Resampling.LANCZOS)

    @property
    def configured(self) -> bool:
        return bool(self.model_options and (
            getattr(self.client, "configured", False) or self.k3_client.configured
        ))

    @property
    def model_options(self) -> List[Dict[str, Any]]:
        ids = (
            self.client.available_models()
            if isinstance(self.client, KimiK3Client)
            else [DEFAULT_PROMPT_ENHANCER_MODEL]
        )
        if self.k3_client.configured:
            ids = list(dict.fromkeys([*ids, self.k3_client.model]))
        preferred = getattr(self.client, "model", DEFAULT_PROMPT_ENHANCER_MODEL)
        default = preferred if preferred in ids else next(iter(ids), "")
        return [
            {
                "id": model,
                "label": "Kimi-K3" if model == PROMPT_ENHANCER_KIMI_K3 else model,
                "default": model == default,
                "reasoning_effort_options": list(PROMPT_ENHANCER_REASONING_EFFORTS),
                "default_reasoning_effort": DEFAULT_PROMPT_ENHANCER_REASONING_EFFORT,
            }
            for model in ids
        ]

    @property
    def default_model(self) -> str:
        return next(
            (option["id"] for option in self.model_options if option["default"]),
            "",
        )

    def resolve_model(self, enhancer_model: object = None) -> str:
        options = self.model_options
        default = next(
            (option["id"] for option in options if option["default"]),
            "",
        )
        if not options:
            raise PromptEnhancementError(
                "prompt enhancer API has no available models; check /v1/models"
            )
        model = normalize_prompt_enhancer_model(
            enhancer_model if enhancer_model is not None else default
        )
        if model not in {option["id"] for option in options}:
            raise PromptEnhancementError(
                f"prompt enhancer model {model!r} is not advertised by the API; "
                f"available models: {[option['id'] for option in options]}"
            )
        return model

    def configured_for(self, enhancer_model: object) -> bool:
        try:
            client = self.client_for(enhancer_model)
        except PromptEnhancementError:
            return False
        return bool(getattr(client, "configured", False))

    def client_for(self, enhancer_model: object) -> Any:
        model = self.resolve_model(enhancer_model)
        if self.k3_client.configured and model == self.k3_client.model:
            return self.k3_client
        return (
            self.client.for_model(model)
            if isinstance(self.client, KimiK3Client)
            else self.client
        )

    def _sample_previous_video(
        self,
        relative_url: Optional[str],
        segment_frame_lengths: Optional[List[int]] = None,
    ) -> List[tuple[str, float]]:
        """Sample only the latest completed segment (legacy helper)."""
        groups = self._sample_previous_video_history(
            relative_url,
            segment_frame_lengths,
        )
        return [(sample[0], sample[-1]) for sample in (groups[-1] if groups else [])]

    @staticmethod
    def _video_frame_count(reader: Any, source_fps: float) -> int:
        metadata = reader.get_meta_data()
        raw_frame_count = metadata.get("nframes")
        frame_count = (
            int(raw_frame_count)
            if isinstance(raw_frame_count, (int, float))
            and math.isfinite(float(raw_frame_count))
            else 0
        )
        if frame_count <= 0 or frame_count > 10**8:
            duration = float(metadata.get("duration") or 0)
            frame_count = max(1, int(duration * source_fps))
        return frame_count

    def _sample_previous_video_history(
        self,
        relative_url: Optional[str],
        segment_frame_lengths: Optional[List[int]] = None,
    ) -> List[List[tuple[str, float, float]]]:
        """Sample every completed segment from the latest cumulative video.

        Each returned inner list contains
        ``(image_url, global_video_sec, segment_relative_sec)`` tuples for one
        generated segment. Sampling the latest cumulative video refreshes older
        PixelUMM provisional tails after cross-segment replacement.
        """
        if not relative_url:
            return []
        lengths = [
            int(value) for value in (segment_frame_lengths or []) if int(value) > 0
        ]
        if not lengths:
            return []
        clean = relative_url.split("?", 1)[0].lstrip("/")
        if clean.startswith("outputs/"):
            clean = clean[len("outputs/") :]
        path = (self.output_dir / clean).resolve()
        if self.output_dir not in path.parents or not path.is_file():
            return []
        import imageio.v2 as imageio

        reader = imageio.get_reader(str(path))
        try:
            metadata = reader.get_meta_data()
            source_fps = float(metadata.get("fps") or self.fps)
            frame_count = self._video_frame_count(reader, source_fps)

            groups: List[List[tuple[str, float, float]]] = []
            range_start = 0
            for segment_index, requested_frames in enumerate(lengths, start=1):
                range_end = min(range_start + requested_frames, frame_count)
                if range_end <= range_start:
                    raise PromptEnhancementError(
                        "previous video has no decodable frames for stored "
                        f"Segment {segment_index}"
                    )
                segment_frames = range_end - range_start
                sample_count = min(
                    self.max_condition_frames,
                    segment_frames,
                    max(
                        1,
                        int(round(segment_frames * 2.0 / source_fps)),
                    ),
                )
                if sample_count == 1:
                    indices = [range_start]
                else:
                    indices = [
                        range_start
                        + int(sample_index * (segment_frames - 1) / (sample_count - 1))
                        for sample_index in range(sample_count)
                    ]
                indices = list(dict.fromkeys(indices))
                samples: List[tuple[str, float, float]] = []
                for index in indices:
                    frame = reader.get_data(index)
                    image = Image.fromarray(frame).convert("RGB")
                    image = self._resize_condition_frame(image)
                    buffer = io.BytesIO()
                    image.save(
                        buffer,
                        format="JPEG",
                        quality=85,
                        optimize=True,
                    )
                    samples.append(
                        (
                            "data:image/jpeg;base64,"
                            + base64.b64encode(buffer.getvalue()).decode("ascii"),
                            index / source_fps,
                            max(0.0, (index - range_start) / source_fps),
                        )
                    )
                groups.append(samples)
                range_start += requested_frames
            return groups
        finally:
            reader.close()

    def _segment_start_times_ms(
        self,
        previous_session: Dict[str, Any],
        segment_frame_lengths: List[int],
    ) -> List[int]:
        """Resolve starts from saved prompts, falling back to frame counts."""
        prompts = list(previous_session.get("dit_prompts") or [])
        starts: List[int] = []
        fallback_start = 0
        for segment_index, frame_length in enumerate(segment_frame_lengths):
            start = None
            end = None
            if segment_index < len(prompts):
                try:
                    item = json.loads(str(prompts[segment_index]).split("|||")[-1])
                    segments = item.get("segments")
                    if (
                        isinstance(segments, list)
                        and segments
                        and isinstance(segments[-1], dict)
                    ):
                        start = _parse_time(segments[-1].get("start_time"))
                        end = _parse_time(segments[-1].get("end_time"))
                except (TypeError, ValueError, json.JSONDecodeError):
                    pass
            if start is None:
                start = fallback_start
            starts.append(start)
            fallback_start = (
                end
                if end is not None and end > start
                else start + int(math.ceil(frame_length * 1000.0 / self.fps))
            )
        return starts

    def _previous_segment_start_ms(
        self,
        previous_session: Dict[str, Any],
    ) -> int:
        """Resolve the exact absolute start of the latest completed segment."""
        lengths = [
            int(value) for value in previous_session.get("segment_frame_lengths") or []
        ]
        starts = self._segment_start_times_ms(previous_session, lengths)
        return starts[-1] if starts else 0

    def _attach_previous_video_history(
        self,
        history: List[Dict[str, Any]],
        frame_groups: List[List[tuple]],
        segment_start_times_ms: List[int],
    ) -> tuple[List[Dict[str, Any]], int, int]:
        """Attach each completed segment once before the following user turn."""
        rebuilt: List[Dict[str, Any]] = []
        user_turn = 0
        attached_segments = 0
        frame_index_start = 0
        for message in history:
            if message.get("role") != "user":
                rebuilt.append(message)
                continue
            parsed_user = _parse_user_message_json(message.get("content"))
            user_text = (
                json.dumps(
                    parsed_user,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                if parsed_user is not None
                else str(message.get("content") or "")
            )
            instructions = self.prompt_profile.turn_prompt(first=user_turn == 0)
            preceding_segment = user_turn - 1
            frames = (
                frame_groups[preceding_segment]
                if 0 <= preceding_segment < len(frame_groups) - 1
                else None
            )
            if instructions is not None:
                rebuilt.append(
                    _turn_user_message(
                        instructions,
                        user_text,
                        frames=frames,
                        previous_segment_start_ms=(
                            segment_start_times_ms[preceding_segment]
                            if frames is not None
                            else 0
                        ),
                        segment_index=(
                            preceding_segment + 1 if frames is not None else None
                        ),
                        frame_index_start=frame_index_start,
                    )
                )
            elif frames is not None:
                current["content"] = self._previous_video_content(
                    user_text,
                    frames,
                    segment_start_times_ms[preceding_segment],
                    segment_index=preceding_segment + 1,
                    frame_index_start=frame_index_start,
                )
                rebuilt.append(current)
            else:
                rebuilt.append(message)
            if frames is not None:
                frame_index_start += len(frames)
                attached_segments += 1
            user_turn += 1
        return rebuilt, attached_segments, frame_index_start

    @staticmethod
    def _previous_video_content(
        user_text: str,
        frames: List[tuple],
        previous_segment_start_ms: int,
        *,
        segment_index: Optional[int] = None,
        frame_index_start: int = 0,
    ) -> List[Dict[str, Any]]:
        """Build the standard timestamp-interleaved K3 multi-image content."""
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": user_text},
            {
                "type": "text",
                "text": (
                    f"The following {len(frames)} frames are ordered "
                    "chronologically and belong only to the previous "
                    "completed segment."
                ),
            },
        ]
        for index, sample in enumerate(frames, start=1):
            if len(sample) == 3:
                image_url, global_sec, relative_sec = sample
                global_ms = max(0, int(round(global_sec * 1000)))
            else:
                image_url, relative_sec = sample
                global_ms = previous_segment_start_ms + max(
                    0,
                    int(round(relative_sec * 1000)),
                )
            relative_ms = max(0, int(round(relative_sec * 1000)))
            global_frame_index = frame_index_start + index
            segment_metadata = (
                f" | segment={segment_index} | segment_frame={index}/{len(frames)}"
                if segment_index is not None
                else f"/{len(frames)}"
            )
            content.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            f"Frame {global_frame_index}{segment_metadata} | "
                            "video_timestamp="
                            f"{_format_time(global_ms)}"
                            " | segment_relative_time="
                            f"{_format_time(relative_ms)}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ]
            )
        return content

    def enhance(
        self,
        request: Dict[str, Any],
        previous_session: Optional[Dict[str, Any]],
        *,
        enhancer_model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        seed: Optional[int] = None,
        stream_callback: Optional[Callable[[str, str], None]] = None,
    ) -> EnhancedPrompt:
        enhancer_model = self.resolve_model(enhancer_model)
        reasoning_effort = normalize_prompt_enhancer_reasoning_effort(
            reasoning_effort
            if reasoning_effort is not None
            else request.get("prompt_enhancer_reasoning_effort")
        )
        seed = normalize_prompt_enhancer_seed(
            seed if seed is not None else request.get("prompt_enhancer_seed")
        )
        request = {
            **request,
            "prompt_enhancer_reasoning_effort": reasoning_effort,
            **({"prompt_enhancer_seed": seed} if seed is not None else {}),
        }
        client = self.client_for(enhancer_model)
        enhancer_display_name = getattr(client, "display_name", "Prompt enhancer")
        prior = previous_session or {}
        first = not previous_session
        uses_inferred_initial_contract = first and (
            "subject_description" in request or "environment_description" in request
        )
        perspective = _first_nonempty_text(
            *(
                (
                    request.get("perspective"),
                    prior.get("requested_perspective"),
                    prior.get("perspective"),
                )
                if first
                else (
                    request.get("perspective"),
                    prior.get("perspective"),
                    prior.get("requested_perspective"),
                )
            )
        )
        style = _first_nonempty_text(
            request.get("style"),
            prior.get("style"),
        )
        subject = str(
            request.get("subject_description")
            or request.get("subject")
            or prior.get("subject")
            or ""
        ).strip()
        environment = str(
            request.get("environment_description")
            or prior.get("environment_description")
            or ""
        ).strip()
        action = str(request.get("action") or "").strip()
        camera_controlled = bool(request.get("camera_controlled"))
        camera_control = (
            _normalize_camera_control(request.get("camera_control"))
            if camera_controlled
            else None
        )
        if camera_control is not None:
            # Quantize the user duration once, before PE sees the request, to
            # the exact video-frame timeline that DiT generates and PixelUMM decodes.
            # This is only sub-frame normalization; unlike the former
            # stable-prefix behavior it never rounds up to a 40/80-frame block.
            camera_duration_frames = _requested_duration_frames(
                {"camera_control": camera_control},
                self.fps,
                5.0,
            )
            camera_duration_ms = int(
                math.ceil(camera_duration_frames * 1000.0 / self.fps)
            )
            camera_control["durationMs"] = camera_duration_ms
            request = {
                **request,
                "camera_control": camera_control,
                "camera_duration_ms": camera_duration_ms,
            }
        if uses_inferred_initial_contract:
            if perspective not in USER_PERSPECTIVES:
                raise PromptEnhancementError(
                    f"unsupported user perspective: {perspective!r}"
                )
        elif perspective not in ALLOWED_PERSPECTIVES:
            raise PromptEnhancementError(f"unsupported perspective: {perspective!r}")
        if not uses_inferred_initial_contract and style not in ALLOWED_STYLES:
            raise PromptEnhancementError(f"unsupported style: {style!r}")

        if not subject:
            raise PromptEnhancementError("subject description must be non-empty")
        if uses_inferred_initial_contract:
            if not environment:
                raise PromptEnhancementError(
                    "environment description must be non-empty"
                )
        elif not camera_controlled and not action:
            raise PromptEnhancementError("action must be non-empty")
        history = _normalize_history_messages(
            list((previous_session or {}).get("messages") or [])
        )
        previous_end = (previous_session or {}).get("previous_end")
        branch_revision = int((previous_session or {}).get("branch_revision", 0) or 0)
        previous_subject_ids = _session_subject_ids(previous_session)
        session_id = str(request.get("session_id") or uuid.uuid4())
        previous_frame_index_start = 0
        if first:
            if uses_inferred_initial_contract:
                user_object = {
                    "perspective_view": perspective,
                    "subject_description": subject,
                    "environment_description": environment,
                }
            else:
                initial_caption = action or (
                    "Establish a coherent initial environment for the "
                    "controlled subject while preserving a neutral subject "
                    "state."
                )
                user_object = {
                    "perspective_view": perspective,
                    "Style": style,
                    "initial": {
                        "subject": subject,
                        "caption": initial_caption,
                    },
                }
            if camera_controlled:
                user_object["camera_control"] = camera_control
            user_text = json.dumps(
                user_object, ensure_ascii=False, separators=(",", ":")
            )
        else:
            segment_frame_lengths = [
                int(value)
                for value in (
                    (previous_session or {}).get("segment_frame_lengths") or []
                )
            ]
            reference_history_url = (previous_session or {}).get('reference_history_url')
            if reference_history_url:
                frame_groups = self._sample_previous_video_history(reference_history_url, segment_frame_lengths[:1])
                later_groups = self._sample_previous_video_history((previous_session or {}).get('video_url'), segment_frame_lengths[1:])
                visible_reference_seconds = segment_frame_lengths[0] / self.fps
                frame_groups += [[(url, absolute + visible_reference_seconds, relative)
                    for url, absolute, relative in group] for group in later_groups]
            else:
                frame_groups = self._sample_previous_video_history(
                    (previous_session or {}).get("video_url"), segment_frame_lengths)
            # The public reference clip omits its warm-up prefix. Sampling uses
            # visible frame counts, while PE timestamps retain model time.
            hidden_seconds = int((previous_session or {}).get("reference_hidden_frames", 0)) / self.fps
            if hidden_seconds:
                frame_groups = [
                    [(url, absolute + hidden_seconds,
                      relative + (hidden_seconds if index == 0 else 0))
                     for url, absolute, relative in group]
                    for index, group in enumerate(frame_groups)
                ]
            if len(frame_groups) != len(segment_frame_lengths) or any(
                not group for group in frame_groups
            ):
                raise PromptEnhancementError(
                    "all previous segment video frames are unavailable for "
                    "prompt-enhancer visual conditioning"
                )
            segment_start_times_ms = self._segment_start_times_ms(
                previous_session or {},
                segment_frame_lengths,
            )
            (
                history,
                first_current_frame_group,
                previous_frame_index_start,
            ) = self._attach_previous_video_history(
                history,
                frame_groups,
                segment_start_times_ms,
            )
            user_object: Dict[str, Any] = (
                {"camera_control": camera_control}
                if camera_controlled
                else {"subsequent_text_prompt": action}
            )
            user_text = json.dumps(
                user_object, ensure_ascii=False, separators=(",", ":")
            )

        turn_prompt = self.prompt_profile.turn_prompt(first=first)
        if turn_prompt is not None:
            user_message = _turn_user_message(
                turn_prompt,
                user_text,
            )
            if not first:
                current_content = user_message["content"]
                for segment_offset, frames in enumerate(
                    frame_groups[first_current_frame_group:],
                    start=first_current_frame_group,
                ):
                    _append_segment_frame_evidence(
                        current_content,
                        frames,
                        previous_segment_start_ms=(
                            segment_start_times_ms[segment_offset]
                        ),
                        segment_index=segment_offset + 1,
                        frame_index_start=previous_frame_index_start,
                    )
                    previous_frame_index_start += len(frames)
        else:
            content: Any = user_text
            if not first:
                content = [{"type": "text", "text": user_text}]
                for segment_offset, frames in enumerate(
                    frame_groups[first_current_frame_group:],
                    start=first_current_frame_group,
                ):
                    _append_segment_frame_evidence(
                        content,
                        frames,
                        previous_segment_start_ms=(
                            segment_start_times_ms[segment_offset]
                        ),
                        segment_index=segment_offset + 1,
                        frame_index_start=previous_frame_index_start,
                    )
                    previous_frame_index_start += len(frames)
            user_message = {"role": "user", "content": content}
        content = user_message["content"]
        base_messages = [
            {
                "role": "system",
                "content": self.prompt_profile.system_prompt,
            },
            *history,
        ]
        audit_metadata = _turn_audit_metadata(
            self.transcript_store,
            request,
            previous_session,
            enhancement_mode=enhancer_model,
        )
        attempts = []
        audit = {
            **audit_metadata,
            "status": "prepared",
            "user_message": _json_clone(user_message),
            "attempts": attempts,
        }

        raw_response = ""
        request_id: Optional[str] = None
        parsed: Dict[str, Any] = {}
        errors: List[str] = []
        for attempt in range(3):
            attempt_reasoning_parts: List[str] = []
            attempt_content_parts: List[str] = []

            def capture_stream(kind: str, delta: str) -> None:
                if kind == "reasoning":
                    attempt_reasoning_parts.append(delta)
                elif kind == "content":
                    attempt_content_parts.append(delta)
                if stream_callback is not None:
                    stream_callback(kind, delta)

            if stream_callback is not None:
                stream_callback("reset", "")
            retry = ""
            if errors:
                retry = (
                    "\n\nSTRICT RETRY: Return corrected raw JSON only. Fix: "
                    + "; ".join(errors)
                )
            attempt_message = dict(user_message)
            if retry:
                if isinstance(content, list):
                    attempt_message["content"] = [
                        *content,
                        {"type": "text", "text": retry},
                    ]
                else:
                    attempt_message["content"] = str(content) + retry
            complete_kwargs: Dict[str, Any] = {}
            if "reasoning_effort" in inspect.signature(client.complete).parameters:
                complete_kwargs["reasoning_effort"] = reasoning_effort
            if "seed" in inspect.signature(client.complete).parameters:
                complete_kwargs["seed"] = seed
            if "stream_callback" in inspect.signature(client.complete).parameters:
                complete_kwargs["stream_callback"] = capture_stream
            attempt_record: Dict[str, Any] = {
                "attempt": len(attempts) + 1,
                "request_attempt": attempt + 1,
                "enhancer_model": enhancer_model,
                "reasoning_effort": reasoning_effort,
                "seed": seed,
                "retry_instruction": retry.strip() or None,
                "assistant_thinking": "",
                "assistant_response": "",
                "request_id": None,
                "validation_errors": [],
            }
            complete_messages = [*base_messages, attempt_message]
            attempt_record["k3_request"] = _enhancer_request_snapshot(
                client,
                complete_messages,
                reasoning_effort=reasoning_effort,
                seed=seed,
            )
            attempts.append(attempt_record)
            audit.update(
                {
                    "status": "requesting",
                    "attempts": attempts,
                    "final_k3_request": attempt_record["k3_request"],
                }
            )
            audit_path = self.transcript_store.append_turn(
                session_id,
                audit,
            )
            audit["transcript_path"] = str(audit_path)
            try:
                raw_response, request_id = client.complete(
                    complete_messages,
                    **complete_kwargs,
                )
                attempt_record.update(
                    {
                        "assistant_thinking": "".join(attempt_reasoning_parts),
                        "assistant_response": raw_response,
                        "streamed_assistant_response": "".join(attempt_content_parts),
                        "request_id": request_id,
                    }
                )
                parsed = _extract_json(raw_response)
            except Exception as exc:
                attempt_record.update(
                    {
                        "assistant_thinking": (
                            "".join(attempt_reasoning_parts)
                            or str(attempt_record.get("assistant_thinking") or "")
                        ),
                        "assistant_response": (
                            "".join(attempt_content_parts)
                            or str(attempt_record.get("assistant_response") or "")
                        ),
                        "error": str(exc),
                    }
                )
                audit.update(
                    {
                        "status": "failed",
                        "assistant_thinking": attempt_record["assistant_thinking"],
                        "assistant_response": attempt_record["assistant_response"],
                        "request_id": attempt_record.get("request_id"),
                        "attempts": attempts,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                        "completed_at": time.time(),
                    }
                )
                self.transcript_store.append_turn(session_id, audit)
                wrapped = (
                    exc
                    if isinstance(exc, PromptEnhancementError)
                    else PromptEnhancementError(str(exc))
                )
                wrapped.pe_audit = _json_clone(audit)
                raise wrapped from exc
            expected_subject_ids = (
                previous_subject_ids if not first and previous_subject_ids else None
            )
            errors = validate_response(
                parsed,
                first=first,
                perspective=perspective,
                style=style or None,
                previous_end=previous_end,
                expected_subject_ids=expected_subject_ids,
                camera_controlled=camera_controlled,
                allow_perspective_inference=uses_inferred_initial_contract,
                allow_style_inference=uses_inferred_initial_contract,
            )
            attempt_record["validation_errors"] = list(errors)
            audit.update(
                {
                    "status": "retrying" if errors else "k3_succeeded",
                    "assistant_thinking": attempt_record["assistant_thinking"],
                    "assistant_response": attempt_record["assistant_response"],
                    "assistant_parsed_response": _json_clone(parsed),
                    "request_id": request_id,
                    "attempts": attempts,
                }
            )
            self.transcript_store.append_turn(session_id, audit)
            if not errors:
                break
        if errors:
            error = PromptEnhancementError(
                f"{enhancer_display_name} output failed validation: "
                + "; ".join(errors)
            )
            audit.update(
                {
                    "status": "failed",
                    "error": str(error),
                    "error_type": type(error).__name__,
                    "completed_at": time.time(),
                }
            )
            self.transcript_store.append_turn(session_id, audit)
            error.pe_audit = _json_clone(audit)
            raise error

        subject_ids = (
            tuple(
                subject["subject_id"]
                for subject in parsed["subject_list"]
                if isinstance(subject, dict)
                and isinstance(subject.get("subject_id"), str)
            )
            if first
            else previous_subject_ids
        )
        resolved_perspective = (
            str(parsed["perspective_view"])
            if first
            else _first_nonempty_text(
                prior.get("perspective"),
                perspective,
            )
        )
        resolved_style = (
            str(parsed["Style"])
            if first
            else _first_nonempty_text(
                prior.get("style"),
                style,
            )
        )
        segment = parsed["segments"][0]
        requested_duration_frames: Optional[int] = None
        if request.get("camera_duration_ms") is not None:
            requested_duration_frames = _requested_duration_frames(
                request,
                self.fps,
                5.0,
            )
            segment_start = 0 if first else (_parse_time(previous_end) or 0)
            segment = _retime_segment(
                segment,
                start_ms=segment_start,
                duration_frames=requested_duration_frames,
                fps=self.fps,
            )
            parsed["segments"][0] = segment
        start_ms = _parse_time(segment["start_time"])
        end_ms = _parse_time(segment["end_time"])
        assert start_ms is not None and end_ms is not None
        duration_frames = (
            requested_duration_frames
            if requested_duration_frames is not None
            else max(
                1,
                (end_ms * self.fps) // 1000 - (start_ms * self.fps) // 1000,
            )
        )
        if requested_duration_frames is None:
            # Store the same frame-quantized duration that DiT generates and PixelUMM
            # decodes. This removes sub-frame timestamp drift without imposing
            # any stable-prefix stride/minimum extension.
            segment = _retime_segment(
                segment,
                start_ms=start_ms,
                duration_frames=duration_frames,
                fps=self.fps,
            )
            parsed["segments"][0] = segment
        # Keep Kimi's absolute timeline here. The worker applies the same
        # json_prompt_normalize_action_timestamp + keep-key shaping as
        # hymm/inference.py immediately before text encoding.
        prompt_segment = json.loads(json.dumps(segment, ensure_ascii=False))
        if first:
            dit_item = {
                "perspective_view": parsed["perspective_view"],
                "Style": parsed["Style"],
                "subject_list": parsed["subject_list"],
                "segments": [prompt_segment],
            }
        else:
            dit_item = {"segments": [prompt_segment]}
        current_prompt = json.dumps(dit_item, ensure_ascii=False)
        prior_lengths = list(
            (previous_session or {}).get("segment_frame_lengths") or []
        )
        segment_frame_lengths = [*prior_lengths, duration_frames]
        # Persist the complete multimodal turn so the session itself contains
        # every sampled historical segment. Before the next K3 request,
        # _normalize_history_messages strips these images back to the user JSON
        # and _attach_previous_video_history refreshes them from the newest
        # cumulative video (including any PixelUMM tail replacement).
        persisted_user = json.loads(json.dumps(user_message, ensure_ascii=False))
        persisted_assistant = {
            "role": "assistant",
            "content": json.dumps(parsed, ensure_ascii=False, separators=(",", ":")),
        }
        assistant_thinking = str(attempts[-1].get("assistant_thinking") or "")
        if not assistant_thinking:
            embedded_thinking = re.findall(
                r"<think>(.*?)</think>",
                raw_response,
                flags=re.DOTALL | re.IGNORECASE,
            )
            assistant_thinking = "\n\n".join(
                part.strip() for part in embedded_thinking if part.strip()
            )
        session = {
            "session_id": session_id,
            "requested_perspective": (
                perspective
                if first
                else _first_nonempty_text(
                    prior.get("requested_perspective"),
                    perspective,
                )
            ),
            "perspective": resolved_perspective,
            "style": resolved_style,
            "subject": subject,
            "environment_description": (
                environment or str(prior.get("environment_description") or "")
            ),
            "subject_ids": list(subject_ids),
            "previous_end": segment["end_time"],
            "messages": [*history, persisted_user, persisted_assistant],
            "dit_prompts": [
                *list((previous_session or {}).get("dit_prompts") or []),
                current_prompt,
            ],
            "segment_frame_lengths": segment_frame_lengths,
            "reference_hidden_frames": int((previous_session or {}).get("reference_hidden_frames", 0)),
            "reference_history_url": (previous_session or {}).get("reference_history_url"),
            "video_url": (previous_session or {}).get("video_url"),
            "last_result": (previous_session or {}).get("last_result"),
            "result_history": list(
                (previous_session or {}).get("result_history") or []
            ),
            "branch_revision": branch_revision,
            "rollback_pending": False,
            "prompt_profile": self.prompt_profile.name,
            "prompt_enhancer_reasoning_effort": reasoning_effort,
            "prompt_enhancer_seed": seed,
            "prompt_enhancer_seed_source": (
                str(request.get("prompt_enhancer_seed_source") or "").strip() or None
            ),
            "updated_at": time.time(),
        }
        return EnhancedPrompt(
            prompt=current_prompt,
            num_frames=duration_frames,
            segment_boundaries=[0, duration_frames],
            session=session,
            raw_response=raw_response,
            request_id=request_id,
            user_message=user_message,
            assistant_thinking=assistant_thinking,
            attempts=attempts,
            audit=audit,
        )
