"""Apply the released web entrypoint's shaping to native segment JSON."""

from __future__ import annotations

import json


def shape_prompt(prompt: str, config, *, camera_continuation: bool = False) -> str:
    """Shape explicit JSON; continue unchanged camera text with native empty JSON.

    The released gateway's camera-minimal mode encodes ``{"segments": [{}]}``
    after its scene is established. New text is encoded at its next boundary.
    Explicit structured input remains a caller-owned per-segment instruction.
    """
    plain = '{"segments": [{}]}' if camera_continuation else prompt
    try:
        item = json.loads(prompt)
    except json.JSONDecodeError:
        return plain
    if not isinstance(item, dict):
        return plain
    segments = item.get("segments")
    if (
        not isinstance(segments, list)
        or len(segments) != 1
        or not isinstance(segments[0], dict)
    ):
        return plain
    keep = config.inference.get("keep_keywords") or config.dataset.get(
        "json_prompt_keep_keys"
    )
    if not keep:
        return prompt
    from hymm.inference import _build_whole_item_prompts

    # The released web frontend marks first turns with a subject_list. Later
    # turns carry description_change/actions and omit that global context.
    segment = dict(segments[0])
    first = isinstance(item.get("subject_list"), list)
    segment["_segment_idx"] = 0 if first else 1
    no_long = bool(config.inference.get("no_desc_long_after_first", False))
    if not first and no_long:
        segment.pop("visual_description_long", None)
    shaped, count = _build_whole_item_prompts(
        item={**item, "segments": [segment]},
        segments=[segment],
        keep_keywords=list(keep),
        resolve_subject_ids=bool(config.inference.get("resolve_subject_ids", False)),
        global_only_in_first_segment=bool(
            config.inference.get("global_only_in_first_segment", False)
        ),
        no_desc_long_after_first=no_long,
        normalize_action_timestamp=bool(
            config.dataset.get("json_prompt_normalize_action_timestamp", False)
        ),
    )
    if count != 1 or "|||" in shaped:
        raise RuntimeError("Native prompt shaping changed the segment count.")
    return shaped
