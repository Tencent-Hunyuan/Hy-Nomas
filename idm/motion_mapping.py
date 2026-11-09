"""Frozen, label-independent adapter from IDM bucket probabilities to CameraBench.

These are action-head proxies, not a reconstruction of camera poses. In
particular, translation stays camera-local rather than in the initial frame.
"""
from __future__ import annotations

import numpy as np

ACTION_GROUPS = {
    "Move_In": (0,), "Move_Left": (1,), "Move_Out": (2,),
    "Move_Right": (3,), "Move_Up": (13,), "Move_Down": (14,),
    "Pan_Right": (4, 5, 11), "Pan_Left": (7, 8, 9),
    "Tilt_Up": (5, 6, 7), "Tilt_Down": (9, 10, 11),
    "Roll_Clockwise": (16,), "Roll_Counterclockwise": (17,),
}
CLASSES = tuple(sorted((*ACTION_GROUPS, "Zoom_In", "Zoom_Out", "Static")))
ADAPTER_VERSION = "action_prob_direction_max_temporal_max_v1"


def validate_probabilities(values, width, name):
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != width or not len(arr):
        raise ValueError(f"{name}: expected nonempty [N,{width}], got {arr.shape}")
    if not np.isfinite(arr).all() or arr.min() < 0 or arr.max() > 1:
        raise ValueError(f"{name}: invalid probability values")
    return arr


def bucket_scores(motion_prob, zoom_prob, direction_prob8=None):
    motion = validate_probabilities(motion_prob, 18, "motion_prob")
    zoom = validate_probabilities(zoom_prob, 2, "zoom_prob")
    if len(motion) != len(zoom):
        raise ValueError("motion/zoom bucket counts disagree")
    # A diagonal action supports both its horizontal pan and vertical tilt.
    # Max is a fixed fuzzy OR; no thresholds or test-set calibration are used.
    result = {name: motion[:, idx].max(axis=1) for name, idx in ACTION_GROUPS.items()}
    result.update(Zoom_In=zoom[:, 0], Zoom_Out=zoom[:, 1])
    if direction_prob8 is not None:
        direction = validate_probabilities(direction_prob8, 8, "direction_prob8")
        if len(direction) != len(motion):
            raise ValueError("direction/action bucket counts disagree")
        names = ("Move_In", "Move_Left", "Move_Out", "Move_Right",
                 "Pan_Right", "Tilt_Up", "Pan_Left", "Tilt_Down")
        result.update({name: direction[:, i] for i, name in enumerate(names)})
    # Official Static asks for no motion OR shaking. Special keys are unrelated.
    moving_or_shaking = np.column_stack((*result.values(), motion[:, 15]))
    result["Static"] = 1 - moving_or_shaking.max(axis=1)
    return result


def clip_camera_motion(motion_prob, zoom_prob, direction_prob8=None):
    scores = bucket_scores(motion_prob, zoom_prob, direction_prob8)
    # Motion may occur at any time. Static requires absence throughout the clip.
    return {name: float(arr.min() if name == "Static" else arr.max())
            for name, arr in scores.items()}
