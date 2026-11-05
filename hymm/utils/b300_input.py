"""Convert one benchmark annotation entry to B300 pipeline arguments.

The annotation parser is shared with hymm.inference; no model is instantiated.
Already prepared pipeline inputs do not import the inference stack.
"""
from copy import deepcopy
from pathlib import Path
import sys


def _select_entry(payload, entry_index):
    if isinstance(payload, list):
        if not payload:
            raise ValueError("Input JSON list is empty")
        if entry_index is None:
            if len(payload) != 1:
                raise ValueError(
                    f"Input contains {len(payload)} entries; select one with "
                    "--entry-index N (or B300_ENTRY_INDEX=N in the shell launcher)"
                )
            entry_index = 0
        if not 0 <= entry_index < len(payload):
            raise ValueError(f"--entry-index must be between 0 and {len(payload) - 1}")
        payload = payload[entry_index]
    elif entry_index not in (None, 0):
        raise ValueError("--entry-index only applies to a JSON list")
    if not isinstance(payload, dict):
        raise ValueError("Expected a pipeline input object or an annotation object with segments")
    return deepcopy(payload)


def _annotation_config(overrides):
    from configs.config import parse_args
    original_argv = sys.argv
    try:
        sys.argv = ["b300-input", "+experiment=inference", *overrides]
        return parse_args(mode="inference")
    finally:
        sys.argv = original_argv


def prepare_b300_input(payload, input_path, *, entry_index=None, overrides=(),
                       input_base_dir=None):
    """Select one entry and preserve the common launcher's prompt shaping."""
    item = _select_entry(payload, entry_index)
    input_path = Path(input_path)
    source_path = (
        Path(input_base_dir).resolve() / input_path.name
        if input_base_dir is not None else input_path.resolve()
    )
    if "prompts" in item:
        # Make file references independent of torchrun's cwd.
        trajectory = item.get("camera_trajectory")
        if isinstance(trajectory, str) and not Path(trajectory).is_absolute():
            candidate = source_path.parent / trajectory
            if candidate.is_file():
                item["camera_trajectory"] = str(candidate.resolve())
        return item

    segments = item.get("segments", item.get("events"))
    if not isinstance(segments, list) or not segments or not all(
            isinstance(segment, dict) for segment in segments):
        raise ValueError("Expected pipeline input JSON with prompts, or an annotation with nonempty segments/events")

    args = _annotation_config(overrides)
    inf = args.inference
    if inf.rollout_type != "seg_ar":
        raise ValueError("B300 annotation conversion requires inference.rollout_type=seg_ar")
    if inf.video_fps != 24:
        raise ValueError("The B300 entry point writes 24-fps video; inference.video_fps must be 24")

    # A missing sidecar must not silently remove camera conditioning.
    camera_spec = item.get("camera_trajectory_spec") or {}
    if camera_spec.get("ground_truth_npz"):
        camera_path = (source_path.parent / camera_spec["ground_truth_npz"]).resolve()
        if not camera_path.is_file():
            raise ValueError(f"Camera trajectory file does not exist: {camera_path}")

    from hymm.inference import _load_from_json, PROMPT_SEPARATOR
    keep_keywords = inf.keep_keywords or getattr(args.dataset, "json_prompt_keep_keys", None)
    entries = _load_from_json(
        source_path,
        max_segments=None,
        keep_keywords=keep_keywords,
        fps=inf.video_fps,
        desc_field_first=inf.desc_field_first,
        desc_field_rest=inf.desc_field_rest,
        resolve_subject_ids=inf.resolve_subject_ids,
        global_only_in_first_segment=inf.global_only_in_first_segment,
        no_desc_long_after_first=inf.no_desc_long_after_first,
        normalize_action_timestamp=getattr(args.dataset, "json_prompt_normalize_action_timestamp", False),
        items=[item],
    )
    if len(entries) != 1:
        raise ValueError("The selected annotation must produce exactly one inference request")
    text, timestamp_boundaries, _, meta = entries[0]
    prompts = [prompt.strip() for prompt in text.split(PROMPT_SEPARATOR)]
    configured_boundaries = inf.segment_frame_boundaries
    boundaries = list(configured_boundaries) if configured_boundaries is not None else timestamp_boundaries
    if boundaries is None:
        raise ValueError(
            "Annotation segments need start_time/end_time, or explicit "
            "inference.segment_frame_boundaries; frame boundaries are not guessed"
        )
    boundaries = list(boundaries)
    if inf.video_length == "frame":
        frames = inf.num_frames
        if frames is None:
            raise ValueError("inference.num_frames is required for video_length=frame")
        if boundaries[-1] != frames and boundaries[-1] > 0:
            scale = frames / boundaries[-1]
            boundaries = [round(bound * scale) for bound in boundaries]
            boundaries[0], boundaries[-1] = 0, frames
    elif inf.video_length == "event":
        frames = boundaries[-1]
    else:
        raise ValueError("B300 annotation conversion requires video_length=event or frame")

    size = inf.video_size
    if isinstance(size, int):
        height = width = size
    else:
        size = list(size)
        if len(size) == 1:
            height = width = size[0]
        elif len(size) == 2:
            height, width = size
        else:
            raise ValueError("inference.video_size must contain height and width")

    result = dict(
        prompts=prompts,
        segment_frame_boundaries=boundaries,
        num_frames=frames,
        height=height,
        width=width,
        guidance_scale=inf.cfg_scale_video,
        camera_guidance_scale=inf.camera_cfg_scale,
        num_inference_steps=inf.infer_steps_video,
        seed=inf.seed if inf.seed is not None else 42,
    )
    trajectory = (meta or {}).get("camera_trajectory") or inf.camera_trajectory
    if trajectory is not None:
        result["camera_trajectory"] = trajectory
    for key in ("cfg_schedule", "cfg_skip_steps", "trajectory_translation_scale",
                "trajectory_rotation_scale"):
        value = getattr(inf, key, None)
        if value is not None:
            result[key] = value
    segments_meta = (meta or {}).get("segments_meta")
    if segments_meta and len(segments_meta) == len(prompts):
        result["segment_has_camera"] = [
            segment.get("segment_control") != "prompt" for segment in segments_meta
        ]
    return result

