"""Plan CPU camera paths in Hy-Nomas's OpenCV coordinate convention.

Movement and look defaults follow the upstream keyboard trajectory generator:
2.8 units/second, 25 degrees/second, 24 FPS, and identity at frame zero.
"""

from __future__ import annotations

import io
import math
import zipfile

import numpy as np

from hy_nomas_model import MAX_WORLD_FRAMES

CAMERAS = [
    "text_only",
    "keyboard",
    "stationary",
    "forward",
    "backward",
    "strafe_left",
    "strafe_right",
    "look_left",
    "look_right",
    "look_up",
    "look_down",
    "trajectory",
]

KEYS = ("w", "s", "a", "d", "up", "down", "left", "right")


def plan_camera(mode: str, frames: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    if mode == "text_only":
        return None, None
    if mode not in CAMERAS or mode == "trajectory":
        raise ValueError("Select a camera motion or upload a trajectory.")
    keys = {
        "keyboard": (),
        "stationary": (),
        "forward": ("w",),
        "backward": ("s",),
        "strafe_left": ("a",),
        "strafe_right": ("d",),
        "look_left": ("left",),
        "look_right": ("right",),
        "look_up": ("up",),
        "look_down": ("down",),
    }[mode]
    return plan_keyboard(keys, frames)


def plan_keyboard(keys: tuple[str, ...], frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert held native keys to one relative segment, including combinations."""
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], frames, axis=0)
    position = np.zeros(3)
    pitch = yaw = 0.0
    for index in range(1, frames):
        yaw += (25 / 24) * (("right" in keys) - ("left" in keys))
        pitch += (25 / 24) * (("up" in keys) - ("down" in keys))
        pitch = min(89.0, max(-89.0, pitch))
        p, y = math.radians(pitch), math.radians(yaw)
        cp, sp, cy, sy = math.cos(p), math.sin(p), math.cos(y), math.sin(y)
        forward = np.array([sy * cp, -sp, cy * cp])
        right = np.array([cy, 0.0, -sy])
        position += (2.8 / 24) * (
            forward * (("w" in keys) - ("s" in keys))
            + right * (("d" in keys) - ("a" in keys))
        )
        # Ry @ Rx, matching the released keyboard/mouse path generator.
        poses[index, :3, :3] = [
            [cy, sy * sp, sy * cp],
            [0, cp, -sp],
            [-sy, cy * sp, cy * cp],
        ]
        poses[index, :3, 3] = position
    calibration = np.array(
        [[2500 / 640, 0, 0.5], [0, 2500 / 352, 0.5], [0, 0, 1]], dtype=np.float32
    )
    return poses, np.repeat(calibration[None], frames, axis=0)


def decode_trajectory(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Accept a bounded, rigid camera-to-world NPZ without pickle/object arrays."""
    if len(data) > 4 * 1024 * 1024:
        raise ValueError("Camera NPZ must be at most 4 MiB.")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if sum(info.file_size for info in archive.infolist()) > 4 * 1024 * 1024:
                raise ValueError("Expanded camera NPZ must be at most 4 MiB.")
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            poses = np.asarray(archive["c2ws"], dtype=np.float32)
            intrinsics = np.asarray(archive["intrinsics"], dtype=np.float32)
    except (KeyError, OSError, zipfile.BadZipFile, ValueError) as exc:
        raise ValueError(
            "Use an NPZ containing numeric c2ws and intrinsics arrays."
        ) from exc
    if (
        poses.ndim != 3
        or poses.shape[1:] != (4, 4)
        or not 48 <= len(poses) <= MAX_WORLD_FRAMES
    ):
        raise ValueError("c2ws must have shape [T,4,4], with 48–2880 frames at 24 FPS.")
    if intrinsics.shape == (3, 3):
        intrinsics = np.repeat(intrinsics[None], len(poses), axis=0)
    if intrinsics.shape == (1, 3, 3):
        intrinsics = np.repeat(intrinsics, len(poses), axis=0)
    if intrinsics.shape != (len(poses), 3, 3):
        raise ValueError("intrinsics must have shape [T,3,3], [1,3,3], or [3,3].")
    if not np.isfinite(poses).all() or not np.isfinite(intrinsics).all():
        raise ValueError("Camera matrices must be finite.")
    rotation = poses[:, :3, :3]
    if (
        not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-4)
        or not np.allclose(np.swapaxes(rotation, 1, 2) @ rotation, np.eye(3), atol=2e-3)
        or not np.allclose(np.linalg.det(rotation), 1, atol=2e-3)
    ):
        raise ValueError(
            "c2ws must contain rigid, right-handed camera-to-world transforms."
        )
    if (intrinsics[:, [0, 1], [0, 1]] <= 0).any() or not np.allclose(
        intrinsics[:, 2], [0, 0, 1]
    ):
        raise ValueError(
            "Camera calibration requires positive focal lengths and a [0,0,1] last row."
        )
    return np.ascontiguousarray(poses), np.ascontiguousarray(intrinsics)
