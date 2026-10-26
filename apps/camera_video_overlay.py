"""Burn prompt captions, camera controls, and path maps into generated video."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class CameraPose:
    t: float
    position: np.ndarray
    quaternion: np.ndarray


@dataclass(frozen=True)
class CameraKeyEvent:
    t: float
    code: str
    pressed: bool


@dataclass(frozen=True)
class CameraOverlay:
    name: str
    duration_ms: float
    poses: tuple[CameraPose, ...]
    key_events: tuple[CameraKeyEvent, ...]
    visual_poses: tuple[CameraPose, ...] = ()


@dataclass(frozen=True)
class OverlaySegment:
    start: int
    end: int
    has_camera: bool
    user_prompt: str


def video_overlays_enabled() -> bool:
    value = os.environ.get("NOMAS_BURN_VIDEO_OVERLAYS")
    return value is not None and value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def parse_camera_overlay(value: str | None, fps: float) -> CameraOverlay | None:
    """Parse the browser camera-trajectory JSON used for model conditioning."""
    if not value or not value.strip():
        return None
    try:
        payload = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("frames"), list):
        return None

    sample_rate = _positive_float(payload.get("sampleRate"), fps)
    poses: list[CameraPose] = []
    for index, raw in enumerate(payload["frames"]):
        if not isinstance(raw, dict):
            continue
        position = raw.get("position")
        quaternion = raw.get("quaternion")
        if not isinstance(position, dict) or not isinstance(quaternion, dict):
            continue
        try:
            position_array = np.array(
                [position["x"], position["y"], position["z"]],
                dtype=np.float64,
            )
            quaternion_array = np.array(
                [
                    quaternion["x"],
                    quaternion["y"],
                    quaternion["z"],
                    quaternion["w"],
                ],
                dtype=np.float64,
            )
            timestamp = float(raw.get("t", index * 1000.0 / sample_rate))
        except (KeyError, TypeError, ValueError):
            continue
        norm = float(np.linalg.norm(quaternion_array))
        if (
            not np.isfinite(position_array).all()
            or not np.isfinite(quaternion_array).all()
            or not math.isfinite(timestamp)
            or norm < 1e-8
        ):
            continue
        poses.append(
            CameraPose(
                t=timestamp,
                position=position_array,
                quaternion=quaternion_array / norm,
            )
        )
    if not poses:
        return None

    poses.sort(key=lambda pose: pose.t)
    inferred_duration = max(
        1000.0 / sample_rate,
        poses[-1].t - poses[0].t + 1000.0 / sample_rate,
    )
    duration_ms = _positive_float(payload.get("durationMs"), inferred_duration)
    supported_keys = {
        "KeyW",
        "KeyA",
        "KeyS",
        "KeyD",
        "ArrowUp",
        "ArrowDown",
        "ArrowLeft",
        "ArrowRight",
    }
    key_events: list[CameraKeyEvent] = []
    raw_key_events = payload.get("keyEvents")
    if isinstance(raw_key_events, list):
        for raw in raw_key_events:
            if not isinstance(raw, dict):
                continue
            try:
                timestamp = float(raw["t"])
                code = str(raw["code"])
            except (KeyError, TypeError, ValueError):
                continue
            if (
                not math.isfinite(timestamp)
                or timestamp < 0
                or code not in supported_keys
            ):
                continue
            key_events.append(
                CameraKeyEvent(
                    t=timestamp,
                    code=code,
                    pressed=raw.get("pressed") is True,
                )
            )
    key_events.sort(key=lambda event: event.t)
    return CameraOverlay(
        name=str(payload.get("name") or "Camera path"),
        duration_ms=duration_ms,
        poses=tuple(poses),
        key_events=tuple(key_events),
        visual_poses=_visual_camera_poses(
            poses, _positive_float(payload.get("visualization_rotation_scale"), 1.0)
        ),
    )


def _quaternion_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.concatenate((
        a[3] * b[:3] + b[3] * a[:3] + np.cross(a[:3], b[:3]),
        [a[3] * b[3] - np.dot(a[:3], b[:3])],
    ))


def _visual_camera_poses(
    poses: Sequence[CameraPose], user_scale: float,
) -> tuple[CameraPose, ...]:
    """Apply 0.75 times the user rotation gain for visualization only."""
    if not poses:
        return ()
    gain = .75 * user_scale
    orientation = poses[0].quaternion.copy()
    result = [replace(poses[0], quaternion=orientation.copy())]
    for previous, current in zip(poses, poses[1:]):
        inverse = previous.quaternion * np.array([-1, -1, -1, 1])
        delta = _quaternion_product(inverse, current.quaternion)
        delta /= max(1e-10, np.linalg.norm(delta))
        if delta[3] < 0:
            delta = -delta
        sine = float(np.linalg.norm(delta[:3]))
        if sine > 1e-10:
            angle = math.atan2(sine, float(delta[3])) * gain
            step = np.concatenate((delta[:3] * math.sin(angle) / sine, [math.cos(angle)]))
            orientation = _quaternion_product(orientation, step)
            orientation /= max(1e-10, np.linalg.norm(orientation))
        result.append(replace(current, quaternion=orientation.copy()))
    return tuple(result)


def burn_camera_overlay(
    frames: np.ndarray,
    camera_trajectory: str | None,
    fps: float,
) -> np.ndarray:
    """Return RGB uint8 frames with camera telemetry permanently composited."""
    overlay = parse_camera_overlay(camera_trajectory, fps)
    if overlay is None or frames.ndim != 4 or frames.shape[-1] != 3:
        return frames

    output = np.ascontiguousarray(frames.copy())
    frame_count = int(output.shape[0])
    if frame_count == 0:
        return output

    for frame_index in range(frame_count):
        progress = frame_index / max(frame_count - 1, 1)
        _draw_overlay_frame(output[frame_index], overlay, progress)
    return output


def burn_video_overlays(
    frames: np.ndarray,
    camera_trajectory: str | None,
    fps: float,
    *,
    user_prompts: Sequence[str] | None = None,
    segment_boundaries: Sequence[int] | None = None,
    segment_has_camera: Sequence[bool] | None = None,
    timeline_frame_offset: int = 0,
    timeline_frame_count: int | None = None,
) -> np.ndarray:
    """Burn camera or original-user-text HUDs onto their matching segments."""
    if frames.ndim != 4 or frames.shape[-1] != 3 or not len(frames):
        return frames

    overlay = parse_camera_overlay(camera_trajectory, fps)
    segments = _overlay_segments(
        len(frames),
        user_prompts,
        segment_boundaries,
        segment_has_camera,
        overlay is not None,
    )
    if not segments:
        return burn_camera_overlay(frames, camera_trajectory, fps)

    output = np.ascontiguousarray(frames.copy())
    for segment in segments:
        length = max(1, segment.end - segment.start)
        current_kind = _segment_kind(segment)
        for frame_index in range(segment.start, segment.end):
            local_index = frame_index - segment.start
            if timeline_frame_count is not None and len(segments) == 1:
                progress = (timeline_frame_offset + local_index) / max(
                    int(timeline_frame_count) - 1, 1
                )
                progress = min(1.0, max(0.0, progress))
            else:
                progress = local_index / max(length - 1, 1)
            if current_kind == "camera" and overlay is not None:
                _draw_overlay_frame(output[frame_index], overlay, progress)
            elif current_kind == "text":
                _draw_text_overlay_frame(
                    output[frame_index],
                    segment.user_prompt,
                )
    return output


def burn_shared_overlay_video(
    source: str | Path,
    destination: str | Path,
    overlay: dict[str, Any],
) -> dict[str, Any]:
    """Render the share-page prompt/camera timeline into a downloadable MP4.

    Frames are processed one at a time so exporting a long cumulative World
    does not require holding the complete video in host memory.
    """
    import imageio.v2 as imageio

    source_path = Path(source)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    raw_turns = overlay.get("turns") if isinstance(overlay, dict) else None
    turns = [dict(turn) for turn in (raw_turns or []) if isinstance(turn, dict)]
    turns.sort(
        key=lambda turn: (
            float(turn.get("start_seconds") or 0.0),
            int(turn.get("number") or 0),
        )
    )
    reader = imageio.get_reader(str(source_path))
    writer = None
    frame_count = 0
    try:
        metadata = reader.get_meta_data()
        fps = _positive_float(metadata.get("fps"), 24.0)
        writer = imageio.get_writer(
            str(destination_path),
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=None,
            ffmpeg_log_level="error",
            output_params=[
                "-preset",
                "ultrafast",
                "-crf",
                "21",
                "-movflags",
                "+faststart",
            ],
        )
        parsed_cameras: dict[int, CameraOverlay | None] = {}
        active_turn_index = 0
        for frame_index, frame in enumerate(reader):
            image = np.ascontiguousarray(frame[..., :3])
            timestamp = frame_index / fps
            while active_turn_index + 1 < len(turns) and timestamp >= float(
                turns[active_turn_index + 1].get("start_seconds") or 0.0
            ):
                active_turn_index += 1
            turn = turns[active_turn_index] if turns else None
            if turn is not None:
                start = float(turn.get("start_seconds") or 0.0)
                end = float(turn.get("end_seconds") or start)
                if start <= timestamp <= end:
                    progress = min(
                        1.0,
                        max(0.0, (timestamp - start) / max(end - start, 1 / fps)),
                    )
                    camera_value = turn.get("camera")
                    if isinstance(camera_value, dict):
                        key = id(turn)
                        if key not in parsed_cameras:
                            parsed_cameras[key] = parse_camera_overlay(
                                json.dumps(
                                    {
                                        **camera_value,
                                        "durationMs": camera_value.get("duration_ms"),
                                        "sampleRate": camera_value.get("sample_rate"),
                                        "keyEvents": camera_value.get("key_events"),
                                    },
                                    ensure_ascii=False,
                                ),
                                fps,
                            )
                        camera = parsed_cameras[key]
                        if camera is not None:
                            _draw_overlay_frame(image, camera, progress)
                    else:
                        prompt = str(turn.get("prompt") or "").strip()
                        if prompt:
                            _draw_text_overlay_frame(image, prompt)
            _draw_demo_timeline(image, turns, timestamp)
            writer.append_data(image)
            frame_count += 1
    finally:
        reader.close()
        if writer is not None:
            writer.close()

    return {
        "frames": frame_count,
        "fps": fps,
        "duration_seconds": frame_count / fps if fps > 0 else 0.0,
    }


def _segment_kind(segment: OverlaySegment) -> str:
    if segment.has_camera:
        return "camera"
    if segment.user_prompt:
        return "text"
    return "none"


def _overlay_segments(
    frame_count: int,
    user_prompts: Sequence[str] | None,
    boundaries: Sequence[int] | None,
    camera_flags: Sequence[bool] | None,
    has_camera_overlay: bool,
) -> list[OverlaySegment]:
    prompts = [str(value).strip() for value in (user_prompts or [])]
    flags = [bool(value) for value in (camera_flags or [])]
    if (
        not boundaries
        or len(boundaries) < 2
        or int(boundaries[0]) != 0
        or any(
            int(right) <= int(left) for left, right in zip(boundaries, boundaries[1:])
        )
    ):
        if prompts or has_camera_overlay:
            return [
                OverlaySegment(
                    0,
                    frame_count,
                    bool(has_camera_overlay),
                    prompts[0] if prompts else "",
                )
            ]
        return []

    source_end = int(boundaries[-1])
    if source_end <= 0:
        return []
    segment_count = len(boundaries) - 1
    segments: list[OverlaySegment] = []
    for index in range(segment_count):
        start = int(round(int(boundaries[index]) * frame_count / source_end))
        end = int(round(int(boundaries[index + 1]) * frame_count / source_end))
        start = max(0, min(frame_count, start))
        end = max(start + 1, min(frame_count, end))
        segments.append(
            OverlaySegment(
                start=start,
                end=end,
                has_camera=(flags[index] if index < len(flags) else has_camera_overlay),
                user_prompt=prompts[index] if index < len(prompts) else "",
            )
        )
    if segments:
        segments[-1] = OverlaySegment(
            segments[-1].start,
            frame_count,
            segments[-1].has_camera,
            segments[-1].user_prompt,
        )
    return segments


def _font_candidates() -> list[str]:
    configured = os.environ.get("NOMAS_OVERLAY_FONT")
    return [
        value
        for value in (
            configured,
            str(
                Path(__file__).resolve().parent
                / "frontend"
                / "public"
                / "fonts"
                / "Tencent-W7CN.woff2"
            ),
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        )
        if value
    ]


@lru_cache(maxsize=32)
def _load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = _font_candidates()
    if bold:
        candidates = [
            value.replace("Regular", "Bold").replace(".ttf", "-Bold.ttf")
            for value in candidates
        ] + candidates
    for candidate in candidates:
        if Path(candidate).is_file():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _text_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    text_font: ImageFont.ImageFont,
) -> int:
    box = draw.textbbox((0, 0), text, font=text_font)
    return int(box[2] - box[0])


def _wrap_prompt(
    draw: ImageDraw.ImageDraw,
    prompt: str,
    text_font: ImageFont.ImageFont,
    max_width: int,
    max_lines: int = 4,
) -> list[str]:
    words = prompt.replace("\r", "").split()
    if not words:
        return []
    # Languages without spaces still wrap naturally at character boundaries.
    tokens = words if len(words) > 1 else list(words[0])
    separator = " " if len(words) > 1 else ""
    lines: list[str] = []
    current = ""
    for token in tokens:
        candidate = f"{current}{separator if current else ''}{token}"
        if _text_width(draw, candidate, text_font) <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = token
        if len(lines) == max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    consumed = separator.join(lines)
    if len(consumed) < len(prompt.strip()) and lines:
        ellipsis = "…"
        while (
            lines[-1]
            and _text_width(
                draw,
                lines[-1] + ellipsis,
                text_font,
            )
            > max_width
        ):
            lines[-1] = lines[-1][:-1].rstrip()
        lines[-1] += ellipsis
    return lines


def _draw_text_overlay_frame(
    image: np.ndarray,
    prompt: str,
) -> None:
    # Intentionally independent of the UI theme: white captions over video.
    _composite_rgba_array(
        image, _cached_demo_caption(image.shape[1], image.shape[0], prompt)
    )


@lru_cache(maxsize=32)
def _cached_demo_caption(width: int, height: int, prompt: str) -> np.ndarray:
    scale = width / 1280.0
    layer = Image.new("RGBA", (width, height))
    draw = ImageDraw.Draw(layer)
    font = _load_font(max(1, round(26 * scale)))
    badge_font = _load_font(max(1, round(13 * scale)), bold=True)
    left = round(width * 0.051)
    text_left = left + round(65 * scale)
    lines = _wrap_prompt(draw, prompt, font, round(width * 0.944) - text_left, 3)
    line_height = max(1, round(34 * scale))
    top = round(height * 0.92) - line_height * max(1, len(lines))
    draw.rounded_rectangle(
        (left, top + 3 * scale, left + 54 * scale, top + 28 * scale),
        radius=max(1, round(13 * scale)), fill="white",
    )
    draw.text(
        (left + 27 * scale, top + 15 * scale), "TEXT",
        font=badge_font, anchor="mm", fill=(17, 17, 17),
    )
    for index, line in enumerate(lines):
        draw.text(
            (text_left, top + index * line_height), line, font=font,
            fill="white", stroke_width=max(1, round(scale)),
            stroke_fill=(0, 0, 0, 90),
        )
    return np.asarray(layer)


def _draw_demo_timeline(
    image: np.ndarray, turns: Sequence[dict[str, Any]], timestamp: float,
) -> None:
    """Bottom segmented timeline; no top-left title or turn counter."""
    timed = [
        turn for turn in turns
        if float(turn.get("end_seconds") or 0) > float(turn.get("start_seconds") or 0)
    ]
    if not timed:
        return
    height, width = image.shape[:2]
    scale = width / 1280.0
    layer = Image.new("RGBA", (width, height))
    draw = ImageDraw.Draw(layer)
    font = _load_font(max(1, round(11 * scale)))
    left, right = width * .051, width * .944
    gap = max(1, 4 * scale)
    available = max(1, right - left - gap * (len(timed) - 1))
    durations = [
        float(turn["end_seconds"]) - float(turn.get("start_seconds") or 0)
        for turn in timed
    ]
    total = sum(durations)
    top = height * .966 - max(1, 5 * scale)
    thickness = max(1, round(5 * scale))
    for index, (turn, duration) in enumerate(zip(timed, durations)):
        end = left + available * duration / total
        start_time = float(turn.get("start_seconds") or 0)
        progress = min(1.0, max(0.0, (timestamp - start_time) / duration))
        kind = "CAMERA" if turn.get("camera") else "START" if index == 0 else "TEXT"
        label = f"T{turn.get('number', index + 1)} · {kind}"
        if _text_width(draw, label, font) <= end - left:
            draw.text((left, top - 19 * scale), label, font=font, fill=(255, 255, 255, 180))
        draw.rounded_rectangle(
            (left, top, end, top + thickness), radius=thickness / 2,
            fill=(255, 255, 255, 90),
        )
        current = left + (end - left) * progress
        if progress > 0:
            draw.rounded_rectangle(
                (left, top, current, top + thickness), radius=thickness / 2,
                fill="white",
            )
        if start_time <= timestamp < start_time + duration:
            radius = max(1, 4 * scale)
            y = top + thickness / 2
            draw.ellipse((current - radius, y - radius, current + radius, y + radius), fill="white")
        left = end + gap
    _composite_rgba(image, layer)


def _positive_float(value: Any, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return parsed if math.isfinite(parsed) and parsed > 0 else float(fallback)


def _pose_index_at_progress(
    poses: Sequence[CameraPose],
    duration_ms: float,
    progress: float,
) -> int:
    target = poses[0].t + min(1.0, max(0.0, progress)) * duration_ms
    for index, pose in enumerate(poses):
        if pose.t >= target:
            return index
    return len(poses) - 1


def _quaternion_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _actions_at_time(
    key_events: Sequence[CameraKeyEvent],
    timestamp_ms: float,
) -> dict[str, bool]:
    actions = {
        "forward": False,
        "backward": False,
        "left": False,
        "right": False,
        "look_up": False,
        "look_down": False,
        "look_left": False,
        "look_right": False,
    }
    action_by_code = {
        "KeyW": "forward",
        "KeyA": "left",
        "KeyS": "backward",
        "KeyD": "right",
        "ArrowUp": "look_up",
        "ArrowDown": "look_down",
        "ArrowLeft": "look_left",
        "ArrowRight": "look_right",
    }
    for event in key_events:
        if event.t > timestamp_ms:
            break
        actions[action_by_code[event.code]] = event.pressed
    return actions


def _composite_rgba(image: np.ndarray, layer: Image.Image) -> None:
    _composite_rgba_array(image, np.asarray(layer.convert("RGBA")))


def _composite_rgba_array(image: np.ndarray, rgba: np.ndarray) -> None:
    foreground = rgba[:, :, :3].astype(np.float32)
    mask = rgba[:, :, 3:4].astype(np.float32) / 255.0
    image[:] = np.clip(
        image.astype(np.float32) * (1.0 - mask) + foreground * mask,
        0,
        255,
    ).astype(np.uint8)


def _trajectory_projector(
    positions: np.ndarray,
    left: int,
    top: int,
    width: int,
    height: int,
):
    """Match the Three.js preview's perspective camera and look-at framing."""
    minimum = positions.min(axis=0)
    maximum = positions.max(axis=0)
    center = (minimum + maximum) / 2.0
    size = max(float(np.linalg.norm(maximum - minimum)), 1.0)
    eye = center + np.array([0.0, size * 0.9, size * 1.45], dtype=np.float64)
    forward = center - eye
    forward /= max(float(np.linalg.norm(forward)), 1e-8)
    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right /= max(float(np.linalg.norm(right)), 1e-8)
    camera_up = np.cross(right, forward)
    camera_up /= max(float(np.linalg.norm(camera_up)), 1e-8)
    focal = 0.5 * height / math.tan(math.radians(38.0) / 2.0)

    def project(points: np.ndarray) -> np.ndarray:
        points = np.atleast_2d(points).astype(np.float64)
        relative = points - eye
        depth = relative @ forward
        depth = np.maximum(depth, 1e-6)
        x = left + width / 2.0 + focal * (relative @ right) / depth
        y = top + height / 2.0 - focal * (relative @ camera_up) / depth
        return np.rint(np.column_stack((x, y))).astype(np.int32)

    return project, minimum, size


def _draw_overlay_frame(
    image: np.ndarray,
    overlay: CameraOverlay,
    progress: float,
) -> None:
    actions = _actions_at_time(overlay.key_events, progress * overlay.duration_ms)
    _draw_demo_camera_map(image, overlay, progress)
    _composite_rgba_array(
        image,
        _cached_demo_controls(
            image.shape[1], image.shape[0],
            tuple(bool(actions.get(key)) and progress < 1 for key in (
                "forward", "left", "backward", "right",
                "look_up", "look_left", "look_down", "look_right",
            )),
        ),
    )


def _interpolated_camera_pose(overlay: CameraOverlay, progress: float) -> CameraPose:
    """Interpolate position and shortest-arc orientation at playback time."""
    poses = overlay.poses
    target = poses[0].t + min(1.0, max(0.0, progress)) * overlay.duration_ms
    right = _pose_index_at_progress(poses, overlay.duration_ms, progress)
    a, b = poses[max(0, right - 1)], poses[right]
    alpha = min(1.0, max(0.0, (target - a.t) / max(1.0, b.t - a.t)))
    qa, qb = a.quaternion, b.quaternion
    dot = float(np.dot(qa, qb))
    if dot < 0:
        qb, dot = -qb, -dot
    if dot > .9995:
        quaternion = qa + alpha * (qb - qa)
    else:
        angle = math.acos(min(1.0, max(-1.0, dot)))
        quaternion = (
            math.sin((1 - alpha) * angle) * qa + math.sin(alpha * angle) * qb
        ) / math.sin(angle)
    quaternion /= max(1e-8, float(np.linalg.norm(quaternion)))
    return CameraPose(target, a.position + alpha * (b.position - a.position), quaternion)


def _draw_demo_camera_map(
    image: np.ndarray, overlay: CameraOverlay, progress: float,
) -> None:
    """Translucent white map matching the browser's SVG projection and marker."""
    overlay = replace(
        overlay,
        poses=overlay.visual_poses or _visual_camera_poses(overlay.poses, 1.0),
    )
    height, width = image.shape[:2]
    scale = width / 1280.0
    map_width = max(1, round(width * .24))
    plot_height = max(1, round(map_width / 2))
    header_height = max(1, round(30 * scale))
    map_height = plot_height + 2 * header_height
    left, top = round(width * .938) - map_width, round(height * .054)
    layer = Image.new("RGBA", (width, height))
    draw = ImageDraw.Draw(layer)
    draw.rounded_rectangle(
        (left, top, left + map_width, top + map_height),
        radius=max(1, round(10 * scale)), fill=(23, 23, 23, 64),
        outline=(255, 255, 255, 128), width=max(1, round(scale)),
    )
    font = _load_font(max(1, round(11 * scale)))
    bold = _load_font(max(1, round(11 * scale)), bold=True)
    pad = max(1, round(13 * scale))
    draw.text((left + pad, top + header_height / 2), "CAMERA PATH",
              font=bold, anchor="lm", fill="white")
    for y in (top + header_height, top + header_height + plot_height):
        draw.line((left, y, left + map_width, y), fill=(255, 255, 255, 51))
    name = _wrap_prompt(draw, overlay.name, font, max(1, map_width - 2 * pad - 40 * scale), 1)
    footer_y = top + header_height + plot_height + header_height / 2
    draw.text((left + pad, footer_y), name[0] if name else "",
              font=font, anchor="lm", fill="white")
    draw.text(
        (left + map_width - pad, footer_y),
        f"{round(min(1.0, max(0.0, progress)) * 100)}%",
        font=bold, anchor="rm", fill="white",
    )

    # A separate plot layer clips the perspective grid to the map viewport.
    plot = Image.new("RGBA", (map_width, plot_height))
    plot_draw = ImageDraw.Draw(plot)
    positions = np.stack([pose.position for pose in overlay.poses])
    project, minimum, size = _trajectory_projector(positions, 0, 0, map_width, plot_height)
    center = (minimum + positions.max(axis=0)) / 2
    half = size * 1.1
    y = minimum[1] - size * .08
    for offset in np.linspace(-half, half, 9):
        for points in (
            [[center[0] - half, y, center[2] + offset], [center[0] + half, y, center[2] + offset]],
            [[center[0] + offset, y, center[2] - half], [center[0] + offset, y, center[2] + half]],
        ):
            plot_draw.line([tuple(point) for point in project(points)], fill=(255, 255, 255, 38))
    if len(positions) > 1:
        plot_draw.line([tuple(point) for point in project(positions)], fill=(255, 255, 255, 85))
    pose = _interpolated_camera_pose(overlay, progress)
    index = _pose_index_at_progress(overlay.poses, overlay.duration_ms, progress)
    travelled = np.concatenate((positions[:index], pose.position[None]), axis=0)
    if len(travelled) > 1:
        plot_draw.line([tuple(point) for point in project(travelled)],
                       fill="white", width=max(1, round(1.8 * scale)))
    local_marker = np.array([
        [0, 0, 0], [-.065, -.045, -.13], [.065, -.045, -.13],
        [0, 0, 0], [.065, .045, -.13], [-.065, .045, -.13],
        [0, 0, 0], [-.065, -.045, -.13], [-.065, .045, -.13],
        [.065, .045, -.13], [.065, -.045, -.13],
    ]) * size
    marker = local_marker @ _quaternion_rotation_matrix(pose.quaternion).T + pose.position
    plot_draw.line([tuple(point) for point in project(marker)],
                   fill="white", width=max(1, round(1.3 * scale)))
    radius = max(1, 2.5 * scale)
    for x, y in project([positions[0], positions[-1], pose.position]):
        plot_draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="white")
    layer.alpha_composite(plot, (left, top + header_height))
    _composite_rgba(image, layer)


@lru_cache(maxsize=64)
def _cached_demo_controls(
    width: int, height: int, active: tuple[bool, ...],
) -> np.ndarray:
    """Reference-style MOVE/LOOK key groups, both at the bottom right."""
    scale = width / 1280.0
    key, gap, group_gap = 51 * scale, 7 * scale, 45 * scale
    group_width = key * 3 + gap * 2
    left = width * .938 - group_width * 2 - group_gap
    bottom = height * .898
    top = bottom - 27 * scale - key * 2 - gap
    layer = Image.new("RGBA", (width, height))
    draw = ImageDraw.Draw(layer)
    font = _load_font(max(1, round(23 * scale)), bold=True)
    caption_font = _load_font(max(1, round(13 * scale)), bold=True)
    for group, caption in enumerate(("MOVE", "LOOK")):
        origin = left + group * (group_width + group_gap)
        labels = ("W", "A", "S", "D") if group == 0 else ("↑", "←", "↓", "→")
        for index, label in enumerate(labels):
            column, row = ((1, 0), (0, 1), (1, 1), (2, 1))[index]
            x, y = origin + column * (key + gap), top + row * (key + gap)
            enabled = active[group * 4 + index]
            draw.rounded_rectangle(
                (x, y, x + key, y + key), radius=max(1, round(10 * scale)),
                fill=(255, 255, 255, 255 if enabled else 38),
                outline=(255, 255, 255, 255 if enabled else 128),
                width=max(1, round(scale)),
            )
            draw.text(
                (x + key / 2, y + key / 2), label,
                font=font, anchor="mm",
                fill=(17, 17, 17, 255) if enabled else (255, 255, 255, 230),
            )
        draw.text(
            (origin + group_width / 2, bottom - 7 * scale), caption,
            font=caption_font, anchor="mm", fill="white",
        )
    return np.asarray(layer)
