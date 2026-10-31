"""
Unified Inference Pipeline

Supports three rollout modes via --rollout-type:

  one_shot:  Non-AR bidirectional inference (MultiPromptBidirectionalPipeline).
             All segments are denoised jointly with full bidirectional attention.

  seg_ar:    Segment-level autoregressive inference (CPTCausalInferencePipeline).
             Each segment is denoised sequentially with KV cache.

  chunk_ar:  Chunk-level autoregressive inference (CPTCausalInferencePipeline).
             Each segment is split into chunks of inference_chunk_size latent
             frames. Requires inference_chunk_size > 0.

Usage:
    bash scripts/inference/inference_from_config.sh configs/experiment/inference.yaml \\
        dit_weight=/path/to/distilled_model.safetensors \\
        model.vae.checkpoint=/path/to/vae_checkpoint \\
        inference.prompt=/path/to/prompts.json \\
        inference.save_path=./outputs/distill
"""

import ast
import os
import sys
import datetime
import json
import hashlib
import tempfile
import shutil
import re
import yaml
import torch
import torch.distributed as dist
from loguru import logger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hymm.pipeline.bidirectional import MultiPromptBidirectionalPipeline
from hymm.pipeline.causal_ar import CPTCausalInferencePipeline
from hymm.distributed.parallel_states import (
    nccl_info,
    ParallelDims,
    initialize_sequence_parallel_state_v2,
)
from configs.config import parse_args as _hydra_parse_args
from hymm.utils.video_output import video_to_uint8_numpy
from hymm.utils.inference_metrics import InferenceMetrics, stage, tensor_storage_bytes
from hymm.utils.inference_mode import offline_sample_mode


# ============================================================================
# Constants
# ============================================================================

PROMPT_SEPARATOR = "|||"

_CAMERA_TEXT_BLOCK_RE = re.compile(
    r"\s*<Camera(\d+)>.*?</Camera\1>\s*",
    flags=re.DOTALL,
)

JSON_DESC_FIELDS = [
    "visual_description_long",
    "visual_description_mid",
    "visual_description_short",
    "label",
]

PromptEntry = Tuple[str, Optional[List[int]], Optional[str], Optional[Dict[str, Any]]]
# (prompt_string, frame_boundaries, filename, meta)
#   meta: per-entry metadata captured BEFORE the suite JSON is stripped /
#         serialised to the text encoder. Always-on for JSON suites so
#         downstream code (filename routing, per-segment control, eval) can
#         read bench-specific fields after they have been removed from the
#         text-encoder input.  Currently understood keys:
#           - "camera_trajectory": absolute path to a .npz / generator name
#                                  to use for THIS entry only (camera bench
#                                  format, see tools_tmp/camera_control_v*).
#           - "segments_meta": list aligned with events[*], each a dict with
#                            the per-event bench keys (segment_control,
#                            joint_caption_constraint, segment_index).
#           - top-level bench keys captured from the suite item (see
#             ``_BENCH_META_TOP_LEVEL_KEYS`` below): test_id, perspective,
#             scene_type, camera_trajectory_spec, joint_control_spec, ...

# --------------------------------------------------------------------------
# Bench-specific keys that exist in benchmark suite JSONs but NOT in the
# training ``caption_stage_2`` top-level. To keep train/eval distribution
# close, the JSON whole-item loader captures these into ``entry_meta`` and
# then strips them from the dict before serialisation (the strip step is
# driven by ``inference.keep_keywords`` in the experiment YAML; this
# capture is unconditional so even if the user keeps a key in the JSON, the
# captured copy stays available for downstream consumers).
# --------------------------------------------------------------------------
_BENCH_META_TOP_LEVEL_KEYS: Tuple[str, ...] = (
    # identifiers / labels
    "test_id",
    "test_type",
    "test_focus",
    "dimension",
    "benchmark_kind",
    # trajectory taxonomy
    "trajectory_family",
    "trajectory_class",
    "trajectory_variant",
    # duration / perspective / scene
    "duration_bucket",
    "total_duration",
    "perspective",
    "perspective_compatibility",
    "scene_type",
    "scene_category",
    "content_domain",
    # game / scene-specific bench metadata (absent from training caption_stage_2)
    "game_genre",
    "obstacle_meta",
    # bench-specific control structures
    "camera_trajectory_spec",
    "joint_control_spec",
    "evaluation_criteria",
)

# Per-event bench keys (joint bench; absent in camera / multiprompt benches).
_BENCH_META_SEGMENT_KEYS: Tuple[str, ...] = (
    "segment_index",
    "segment_control",
    "joint_caption_constraint",
)

# Always-stripped keys for the JSON-whole-item branch. These are bench-only
# (never present in training ``caption_stage_2``) so the strip is unconditional
# — the matching values are still available via ``entry_meta`` and
# ``entry_meta["segments_meta"]`` for any downstream code that needs them.
_BENCH_STRIP_KEYS: Tuple[str, ...] = (
    _BENCH_META_TOP_LEVEL_KEYS + _BENCH_META_SEGMENT_KEYS
)

MAX_BASENAME_CHARS = 100


def _canonicalize_stage2_keys(item: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a caption item onto the canonical ``segments`` schema.

    Canonical keys are ``segments`` / ``subject_list``. Legacy benchmark JSONs
    that use ``events`` / ``subjects_lifecycle`` are renamed on load (shallow
    copy) so all downstream inference code — and the serialized text-encoder
    prompt — uses ``segments`` / ``subject_list``. Mirrors
    ``WM1MTrainingDataset._canonicalize_stage2_keys``. Canonical keys win if
    both are present.
    """
    if not isinstance(item, dict):
        return item
    if "segments" in item and "subject_list" in item:
        return item
    out = dict(item)
    if "segments" not in out and "events" in out:
        out["segments"] = out.pop("events")
    if "subject_list" not in out and "subjects_lifecycle" in out:
        out["subject_list"] = out.pop("subjects_lifecycle")
    return out


def _restore_full_prompt_for_txt(
    entry: str, entry_meta: Optional[Dict[str, Any]]
) -> str:
    """Re-inject the bench-only keys (captured into ``entry_meta`` before
    stripping) back into the prompt string that gets written to the sidecar
    ``.txt``.

    This is purely for the on-disk record / visualisation: the text encoder
    already consumed the stripped ``entry`` earlier, so restoring fields here
    does NOT change what the model saw. We keep the ``|||`` per-segment layout
    (so the txt still maps 1:1 to the encoded segments) and only merge the
    captured metadata back in:
      - per-event keys (``segment_control`` etc.) -> that segment's events[0]
      - top-level keys (``joint_control_spec`` etc.) -> segment 0's object

    Falls back to the raw ``entry`` whenever there is nothing to restore or the
    segments are not JSON (e.g. plain-text desc_field mode).
    """
    if not entry_meta:
        return entry
    segs = entry.split(PROMPT_SEPARATOR)
    try:
        objs = [json.loads(s) for s in segs]
    except Exception:
        return entry  # plain-text prompts (non-JSON benches) -> leave as-is

    segments_meta = entry_meta.get("segments_meta") or []
    top_meta = {k: entry_meta[k] for k in _BENCH_META_TOP_LEVEL_KEYS if k in entry_meta}
    if not segments_meta and not top_meta:
        return entry

    for i, obj in enumerate(objs):
        if not isinstance(obj, dict):
            return entry  # unexpected shape -> bail out safely
        if i == 0 and top_meta:
            obj = {**top_meta, **obj}
        evs = obj.get("segments")
        if (
            segments_meta
            and i < len(segments_meta)
            and isinstance(evs, list)
            and evs
            and isinstance(evs[0], dict)
        ):
            em = segments_meta[i] or {}
            if em:
                evs[0] = {**em, **evs[0]}
        objs[i] = obj
    return PROMPT_SEPARATOR.join(json.dumps(o, ensure_ascii=False) for o in objs)


_MODEL_WEIGHT_NAME = "diffusion_pytorch_model.safetensors"


def _ensure_dit_weight_safetensors(
    dit_weight: str, *, rank: int, world_size: int
) -> str:
    """Resolve a safetensors file or a synchronous checkpoint directory."""
    if not dit_weight:
        return dit_weight
    path = Path(dit_weight).expanduser().resolve()
    if path.is_dir():
        path = path / _MODEL_WEIGHT_NAME
    if path.suffix != ".safetensors":
        raise ValueError(f"Expected a .safetensors DiT checkpoint: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"DiT checkpoint not found: {path}")

    from safetensors import safe_open

    try:
        with safe_open(str(path), framework="pt", device="cpu") as file:
            if not file.keys():
                raise ValueError("checkpoint contains no tensors")
    except Exception as exc:
        raise ValueError(f"Invalid safetensors checkpoint: {path}") from exc
    if rank == 0:
        logger.info(f"Using DiT weights: {path}")
    return str(path)


# ============================================================================
# Shared utilities
# ============================================================================


def sanitize_basename(s: str, max_chars: int = MAX_BASENAME_CHARS) -> str:
    """Sanitize a string for use as a filename component."""
    if not s or not isinstance(s, str):
        return "scene"
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^\w\-]", "", s)
    if not s:
        return "scene"
    return s[:max_chars].rstrip("_-")


def _shorten_trajectory_class(traj_class: str) -> str:
    """Convert a (possibly composite) ``trajectory_class`` into a concise,
    filename-safe token.

    Single-trajectory cases pass through unchanged; composite cases are
    parsed to keep their segment list visible in the filename:

      - ``"forward"``                                              -> ``"forward"``
      - ``"pan_left"``                                             -> ``"pan_left"``
      - ``"composite:static -> pan_right -> static"``              -> ``"static-pan_right-static"``
      - ``"composite:tilt_up -> pan_right"``                       -> ``"tilt_up-pan_right"``
      - ``"composite:forward + pan_right (simultaneous, ...)"``    -> ``"sim-forward-pan_right"``
      - ``"composite:pan_left -> pan_right -> pan_left -> pan_right (2 oscillations)"``
                                                                   -> ``"pan_left-pan_right-pan_left-pan_right"``

    Sequential composites (``->``) join segments with ``-``.
    Simultaneous composites (``+``) get a ``sim-`` prefix and also join with ``-``
    (``+`` itself would be stripped by ``sanitize_basename``).

    Parenthesized human-readable annotations (e.g. "(simultaneous, ...)")
    are dropped; intra-segment whitespace is collapsed to ``_``.
    """
    if ":" not in traj_class:
        return traj_class
    rhs = traj_class.split(":", 1)[1]
    rhs = re.sub(r"\([^)]*\)", "", rhs).strip()
    if not rhs:
        return "composite"
    if "+" in rhs:
        parts = [p.strip() for p in rhs.split("+") if p.strip()]
        prefix = "sim-"
    elif "->" in rhs:
        parts = [p.strip() for p in rhs.split("->") if p.strip()]
        prefix = ""
    else:
        return re.sub(r"\s+", "_", rhs) or "composite"
    parts = [re.sub(r"\s+", "_", p) for p in parts if p]
    if not parts:
        return "composite"
    return prefix + "-".join(parts)


def _camera_bench_filename(test_id: str, entry_meta: Dict[str, Any]) -> str:
    """Build a self-describing filename for a camera control bench case.

    Schema: ``{test_id}__{perspective}__{scene_short}__{traj_short}__{seconds}s``

    Examples:
        CC-A01-S-001__pov__indoor-static__forward__8s
        CC-B03-M-048__pov__game-fps__forward__16s
        CC-C01-M-040__tpf__indoor-static__static-pan_right-static__16s
        CC-C03-M-042__pov__outdoor-action__sim-truck_left-pan_right__16s

    All fields come from ``entry_meta`` (populated from the camera bench
    suite.json).  ``trajectory_class`` like ``"composite:foo -> bar"`` is
    expanded by :func:`_shorten_trajectory_class` so the segment list stays
    visible in the filename.  Falls back to ``"?"`` for any missing field,
    but the overall format stays parsable.
    """
    perspective = entry_meta.get("perspective") or "?"
    scene_type = entry_meta.get("scene_type") or "?"
    scene_short = scene_type.replace("scn_", "").replace("_", "-")

    traj_class = entry_meta.get("trajectory_class") or "?"
    traj_short = _shorten_trajectory_class(traj_class)
    traj_short = sanitize_basename(traj_short, max_chars=40)

    # Parse "00:08.000" → 8s. Fall back to duration_bucket if parse fails.
    total_duration = entry_meta.get("total_duration")
    seconds_part = entry_meta.get("duration_bucket") or "?"  # "S"/"M"/"L"/"XL"
    if total_duration:
        try:
            secs = int(round(_parse_timestamp(total_duration)))
            seconds_part = f"{secs}s"
        except Exception:
            pass

    return f"{test_id}__{perspective}__{scene_short}__{traj_short}__{seconds_part}"


# ============================================================================
# JSON prompt loading utilities
# ============================================================================


def _estimate_prompt_entry_duration_seconds(
    prompt_entry: PromptEntry,
    fps: float,
) -> float:
    """Estimate generated duration for cross-group load balancing.

    Frame boundaries take priority because they reflect prompt shaping such as
    max-segment truncation. Benchmark ``total_duration`` is the fallback.
    """
    if len(prompt_entry) == 4:
        _entry, frame_boundaries, _filename, meta = prompt_entry
    else:
        _entry, frame_boundaries, _filename = prompt_entry
        meta = None
    if frame_boundaries and len(frame_boundaries) >= 2 and float(fps) > 0:
        duration = (float(frame_boundaries[-1]) - float(frame_boundaries[0])) / float(
            fps
        )
        if duration > 0:
            return duration

    meta = meta or {}
    total_duration = meta.get("total_duration")
    if total_duration is not None:
        try:
            duration = _parse_timestamp(total_duration)
            if duration > 0:
                return duration
        except (TypeError, ValueError):
            pass
    return 1.0


def _duration_balanced_prompt_shards(
    indexed_entries: List[Tuple[int, PromptEntry]],
    num_shards: int,
    fps: float,
) -> Tuple[List[List[Tuple[int, PromptEntry]]], List[float]]:
    """Deterministic longest-video-first greedy balancing."""
    if num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {num_shards}")

    shards: List[List[Tuple[int, PromptEntry]]] = [[] for _ in range(num_shards)]
    loads = [0.0] * num_shards
    weighted = [
        (
            _estimate_prompt_entry_duration_seconds(entry, fps),
            orig_idx,
            entry,
        )
        for orig_idx, entry in indexed_entries
    ]
    weighted.sort(key=lambda item: (-item[0], item[1]))

    for duration, orig_idx, entry in weighted:
        shard_id = min(range(num_shards), key=lambda sid: (loads[sid], sid))
        shards[shard_id].append((orig_idx, entry))
        loads[shard_id] += duration

    # Stable processing/output order within every assigned shard.
    for shard in shards:
        shard.sort(key=lambda item: item[0])
    return shards, loads


def _parse_timestamp(ts) -> float:
    """
    Parse a timestamp to seconds.

    Supported formats:
      - float / int e.g. 2.0 -> 2.0
      - "MM:SS.mmm"  e.g. "00:02.000" -> 2.0
      - "HH:MM:SS.mmm" e.g. "00:00:02.000" -> 2.0
      - "SS.mmm" e.g. "2.000" -> 2.0
      - plain number e.g. "2" -> 2.0
    """
    if ts is None:
        return 0.0
    if isinstance(ts, (int, float)):
        return float(ts)
    ts = ts.strip()
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    else:
        return float(ts)


def _remove_keys_from_dict(d: dict, keys_to_remove: List[str]) -> dict:
    """
    Return a deep copy of *d* with the specified keys removed (blacklist).

    Supports arbitrary nesting depth via dot-separated paths:
      - ``"field"``          -- remove at all levels (global).
      - ``"parent.child"``   -- remove *child* only inside ``d["parent"]``.
      - ``"a.b.c"``         -- remove *c* inside ``a`` -> ``b``.

    Used internally to strip bench-only keys (``_BENCH_STRIP_KEYS``); user-
    supplied filtering now goes through :func:`_keep_keys_in_dict` instead.
    """
    import copy

    tree: dict = {}
    global_keys: set = set()
    for key in keys_to_remove:
        parts = key.split(".")
        if len(parts) == 1:
            global_keys.add(parts[0])
        else:
            node = tree
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node.setdefault(None, set()).add(parts[-1])
    tree[None] = global_keys

    def _apply(obj: dict, t: dict, inherited_globals: set) -> dict:
        local_keys = t.get(None, set())
        remove_here = local_keys | inherited_globals
        result = {}
        for k, v in obj.items():
            if k in remove_here:
                continue
            sub_tree = t.get(k)
            if isinstance(v, list):
                child_tree = {kk: vv for kk, vv in t.items() if kk is not None}
                if sub_tree:
                    child_tree.update(sub_tree)
                v = [
                    _apply(item, child_tree, inherited_globals)
                    if isinstance(item, dict)
                    else item
                    for item in v
                ]
            result[k] = v
        return result

    return _apply(copy.deepcopy(d), tree, global_keys)


def _keep_keys_in_dict(d: dict, keys_to_keep: List[str]) -> dict:
    """
    Return a deep copy of *d* that retains ONLY the specified keys (whitelist).

    Supports arbitrary nesting depth via dot-separated paths:
      - ``"field"``          -- keep at all levels (global).
      - ``"parent.child"``   -- keep *child* only inside ``d["parent"]`` (and
                                ``parent`` itself is implicitly kept).
      - ``"a.b.c"``         -- keep *c* inside ``a`` -> ``b`` (and the
                                intermediate parents are implicitly kept).

    Semantics:
      * Keys not listed are dropped.
      * If a listed key has no nested rules, its full sub-structure is kept.
      * If a listed key has nested rules, recursion descends and filters
        accordingly (dicts and list-of-dicts).
      * Single-token (global) keys also propagate to deeper levels, so e.g.
        listing ``"events"`` matches an ``events`` field at any depth.
    """
    import copy

    # Legacy alias: items are canonicalized to segments/subject_list before
    # filtering, so old configs listing the legacy top-level names still match.
    _KEEP_KEY_LEGACY_ALIAS = {
        "events": "segments",
        "subjects_lifecycle": "subject_list",
    }

    tree: dict = {}
    if keys_to_keep:
        for key in keys_to_keep:
            parts = key.split(".")
            if parts:
                parts[0] = _KEEP_KEY_LEGACY_ALIAS.get(parts[0], parts[0])
            if len(parts) == 1:
                tree.setdefault(None, set()).add(parts[0])
            else:
                node = tree
                for part in parts[:-1]:
                    node.setdefault(None, set()).add(part)
                    node = node.setdefault(part, {})
                node.setdefault(None, set()).add(parts[-1])

    def _apply(obj: dict, t: dict, inherited_globals: set) -> dict:
        local_keys = t.get(None, set()) if t else set()
        keep_here = local_keys | inherited_globals
        result = {}
        for k, v in obj.items():
            if k not in keep_here:
                continue
            sub_tree = t.get(k) if t else None
            if sub_tree:
                if isinstance(v, dict):
                    v = _apply(v, sub_tree, inherited_globals)
                elif isinstance(v, list):
                    v = [
                        _apply(item, sub_tree, inherited_globals)
                        if isinstance(item, dict)
                        else item
                        for item in v
                    ]
            result[k] = v
        return result

    inherited_globals = tree.get(None, set()) if tree else set()
    return _apply(copy.deepcopy(d), tree, inherited_globals)


def _resolve_subject_ids(segment: dict, subject_map: dict) -> dict:
    """Replace subject-id references with display names in contained_actions."""
    import copy

    seg = copy.deepcopy(segment)
    actions = seg.get("contained_actions", [])
    if not actions:
        return seg

    for action in actions:
        if not isinstance(action, dict):
            continue
        for key, val in action.items():
            if isinstance(val, str):
                for sid, dname in subject_map.items():
                    val = val.replace(sid, dname)
                action[key] = val
    return seg


def _format_relative_timestamp(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    return f"{minutes:02d}:{rem:06.3f}"


def _normalize_contained_action_timestamps(obj: Any) -> Any:
    """Make contained action timestamps relative to their segment start_time."""
    segments = obj.get("segments") if isinstance(obj, dict) else None
    if not isinstance(segments, list):
        return obj
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        actions = seg.get("contained_actions")
        if not isinstance(actions, list):
            continue
        segment_start = _parse_timestamp(seg.get("start_time", 0.0))
        normalized = []
        for action in actions:
            if isinstance(action, dict):
                action_out = dict(action)
                for key in ("start_time", "end_time"):
                    if key in action_out:
                        rel_t = _parse_timestamp(action_out[key]) - segment_start
                        action_out[key] = _format_relative_timestamp(rel_t)
            else:
                action_out = action
            normalized.append(action_out)
        seg["contained_actions"] = normalized
    return obj


def _strip_camera_text_blocks_for_camera_segment(seg: dict) -> dict:
    """Remove matching CameraN text blocks only from camera-controlled segments."""
    if not isinstance(seg, dict) or seg.get("segment_control") != "camera":
        return seg

    def _clean(value: Any) -> Any:
        if isinstance(value, str):
            if "<Camera" not in value:
                return value
            cleaned = _CAMERA_TEXT_BLOCK_RE.sub(" ", value)
            return re.sub(r"\s+", " ", cleaned).strip()
        if isinstance(value, dict):
            return {key: _clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_clean(item) for item in value]
        return value

    return _clean(seg)


def _build_whole_item_prompts(
    item: Dict[str, Any],
    segments: List[dict],
    keep_keywords: Optional[List[str]],
    resolve_subject_ids: bool,
    global_only_in_first_segment: bool,
    no_desc_long_after_first: bool,
    normalize_action_timestamp: bool,
) -> Tuple[str, int]:
    """Serialize per-segment JSON prompts using the training-side shaping rules."""
    if keep_keywords:
        cleaned = _keep_keys_in_dict(item, list(keep_keywords))
    else:
        import copy as _copy_mod

        cleaned = _copy_mod.deepcopy(item)
    cleaned = _remove_keys_from_dict(cleaned, list(_BENCH_STRIP_KEYS))

    context = {k: v for k, v in cleaned.items() if k != "segments"}
    subject_map: Dict[str, str] = {}
    if resolve_subject_ids:
        for subj in item.get("subject_list", []):
            sid = subj.get("subject_id") if isinstance(subj, dict) else None
            dname = subj.get("display_name") if isinstance(subj, dict) else None
            if sid and dname:
                subject_map[sid] = dname

    per_segment_prompts: List[str] = []
    for i, seg in enumerate(segments):
        seg = _strip_camera_text_blocks_for_camera_segment(seg)
        seg_processed = (
            _resolve_subject_ids(seg, subject_map) if subject_map else dict(seg)
        )
        seg_idx = seg_processed.get("_segment_idx", i)
        seg_processed.pop("_segment_idx", None)

        if no_desc_long_after_first:
            if i > 0:
                seg_processed.pop("visual_description_long", None)
            if "description_change" in seg_processed and seg_idx == 0:
                seg_processed.pop("description_change", None)

        if "contained_actions" in seg_processed:
            actions = seg_processed.pop("contained_actions")
            seg_processed["contained_actions"] = actions

        if global_only_in_first_segment and i > 0:
            segment_obj = {"segments": [seg_processed]}
        else:
            segment_obj = {**context, "segments": [seg_processed]}
        if normalize_action_timestamp:
            # Normalize action start/end times to the event-local timeline while
            # the event start_time is still present. ``contained_actions`` stays a
            # plain list (no <|ActionN|> dict rewrite).
            segment_obj = _normalize_contained_action_timestamps(segment_obj)
        if keep_keywords:
            segment_obj = _keep_keys_in_dict(segment_obj, list(keep_keywords))

        per_segment_prompts.append(json.dumps(segment_obj, ensure_ascii=False))

    return PROMPT_SEPARATOR.join(per_segment_prompts), len(per_segment_prompts)


def _fallback_desc(segment: dict) -> str:
    """Try fallback description fields in priority order."""
    for field in JSON_DESC_FIELDS:
        val = segment.get(field)
        if val:
            return val
    return ""


def _parse_dict_string(s: str) -> Optional[dict]:
    """
    Try to parse a string as a dictionary.

    Attempts JSON first, then falls back to ``ast.literal_eval``.
    Returns ``None`` if the string is not a valid dict representation.
    """
    s = s.strip()
    if not s:
        return None

    # Fast-path: JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: Python literal (handles single-quoted dicts, etc.)
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            return obj
    except (ValueError, SyntaxError):
        pass

    return None


def _load_single_arrow(
    arrow_path: Path,
    arrow_column: str,
    arrow_filename_column: Optional[str] = None,
    arrow_caption_key: Optional[str] = None,
    keep_keywords: Optional[List[str]] = None,
    max_entries: Optional[int] = None,
    global_offset: int = 0,
) -> List[PromptEntry]:
    """
    Load prompt entries from a single .arrow file (PyArrow IPC format).

    When the column value is a **string representation of a dictionary**,
    the function automatically parses it and:
      - If *arrow_caption_key* is given, extracts that key as the prompt.
      - Otherwise, re-serialises the (optionally filtered) dict as JSON.

    Args:
        arrow_path: Path to the .arrow file.
        arrow_column: Column name whose content is used as the text prompt.
        arrow_filename_column: Optional column name for output filenames.
        arrow_caption_key: Optional key to extract from a dict-valued column.
        keep_keywords: Optional list of keys to retain in parsed dicts
            (whitelist; same dot-path syntax as JSON prompt ``keep_keywords``).
        max_entries: Maximum number of rows to load (None = all).
        global_offset: Index offset for default filename numbering (used when
            loading from a directory with multiple files).

    Returns:
        List of PromptEntry tuples: (prompt_string, frame_boundaries, filename).
    """
    import pyarrow as pa
    import pyarrow.ipc as ipc

    logger.info(f"Loading prompt entries from Arrow file: {arrow_path}")

    with pa.memory_map(str(arrow_path), "r") as source:
        reader = ipc.open_file(source)
        table = reader.read_all()

    logger.info(f"  Arrow table: {table.num_rows} rows, {table.num_columns} columns")
    logger.info(f"  Available columns: {table.column_names}")

    if arrow_column not in table.column_names:
        raise ValueError(
            f"Column '{arrow_column}' not found in {arrow_path}. "
            f"Available columns: {table.column_names}"
        )

    use_filename_col = arrow_filename_column
    if use_filename_col and use_filename_col not in table.column_names:
        logger.warning(
            f"Filename column '{use_filename_col}' not found in {arrow_path}. "
            f"Available: {table.column_names}. Will fall back to index-based naming."
        )
        use_filename_col = None

    num_rows = table.num_rows
    if max_entries is not None:
        num_rows = min(num_rows, max_entries)
        logger.info(
            f"  Limiting to first {num_rows} entries (max_entries={max_entries})"
        )

    prompt_col = table.column(arrow_column)
    filename_col = table.column(use_filename_col) if use_filename_col else None

    if arrow_caption_key:
        logger.info(f"  arrow_caption_key={arrow_caption_key} (will extract from dict)")
    if keep_keywords:
        logger.info(
            f"  keep_keywords={keep_keywords} (will retain only these dict keys)"
        )

    entries: List[PromptEntry] = []
    dict_parse_count = 0
    for i in range(num_rows):
        raw_value = prompt_col[i].as_py()
        prompt_text = str(raw_value).strip() if raw_value is not None else ""
        if not prompt_text:
            logger.warning(f"  Row {i}: empty prompt, skipping.")
            continue

        # Try to parse as dict
        parsed_dict = _parse_dict_string(prompt_text)
        if parsed_dict is not None:
            dict_parse_count += 1
            # Apply keep_keywords filtering (whitelist)
            if keep_keywords:
                parsed_dict = _keep_keys_in_dict(parsed_dict, keep_keywords)

            if arrow_caption_key:
                # Extract a specific key from the dict
                value = parsed_dict.get(arrow_caption_key)
                if value is None:
                    logger.warning(
                        f"  Row {i}: key '{arrow_caption_key}' not found in dict. "
                        f"Available keys: {list(parsed_dict.keys())}. Skipping."
                    )
                    continue
                prompt_text = str(value).strip()
            else:
                # Re-serialise the (filtered) dict as JSON string
                prompt_text = json.dumps(parsed_dict, ensure_ascii=False)

        if not prompt_text:
            logger.warning(f"  Row {i}: empty prompt after parsing, skipping.")
            continue

        if filename_col is not None:
            fname = str(filename_col[i].as_py()).strip()
            fname = (
                sanitize_basename(fname) if fname else f"arrow_{global_offset + i:06d}"
            )
        else:
            fname = f"arrow_{global_offset + i:06d}"

        entries.append((prompt_text, None, fname, None))

    if dict_parse_count > 0:
        logger.info(
            f"  Parsed {dict_parse_count}/{num_rows} rows as dict-valued strings"
        )
    logger.info(f"Loaded {len(entries)} prompt entries from {arrow_path.name}")
    return entries


def _load_from_arrow(
    arrow_path: Path,
    arrow_column: str,
    arrow_filename_column: Optional[str] = None,
    arrow_caption_key: Optional[str] = None,
    keep_keywords: Optional[List[str]] = None,
    max_entries: Optional[int] = None,
) -> List[PromptEntry]:
    """
    Load prompt entries from .arrow source(s).

    Supports:
      - A single .arrow file.
      - A directory containing one or more .arrow files (sorted by name).

    Args:
        arrow_path: Path to a .arrow file **or** a directory of .arrow files.
        arrow_column: Column name whose content is used as the text prompt.
        arrow_filename_column: Optional column name for output filenames.
        arrow_caption_key: Optional key to extract from dict-valued columns.
        keep_keywords: Optional list of keys to retain in parsed dicts (whitelist).
        max_entries: Maximum total rows to load across all files (None = all).

    Returns:
        List of PromptEntry tuples: (prompt_string, frame_boundaries, filename).
    """
    logger.info(f"Arrow source: {arrow_path}")
    logger.info(f"  prompt column: {arrow_column}")
    if arrow_filename_column:
        logger.info(f"  filename column: {arrow_filename_column}")
    if arrow_caption_key:
        logger.info(f"  caption key: {arrow_caption_key}")

    # Collect the list of .arrow files to process
    if arrow_path.is_dir():
        arrow_files = sorted(arrow_path.glob("*.arrow"))
        if not arrow_files:
            raise FileNotFoundError(f"No .arrow files found in directory: {arrow_path}")
        logger.info(f"  Found {len(arrow_files)} .arrow files in directory")
    else:
        arrow_files = [arrow_path]

    all_entries: List[PromptEntry] = []
    remaining = max_entries  # None means unlimited

    for fpath in arrow_files:
        per_file_limit = remaining  # may be None
        file_entries = _load_single_arrow(
            fpath,
            arrow_column=arrow_column,
            arrow_filename_column=arrow_filename_column,
            arrow_caption_key=arrow_caption_key,
            keep_keywords=keep_keywords,
            max_entries=per_file_limit,
            global_offset=len(all_entries),
        )
        all_entries.extend(file_entries)

        if remaining is not None:
            remaining -= len(file_entries)
            if remaining <= 0:
                logger.info(f"  Reached max_entries={max_entries}, stopping.")
                break

    logger.info(
        f"Total: {len(all_entries)} prompt entries from {len(arrow_files)} arrow file(s)"
    )
    return all_entries


def _load_from_directory(
    dir_path: Path,
    desc_field_first: str = "",
    desc_field_rest: str = "",
    max_segments: Optional[int] = None,
    keep_keywords: Optional[List[str]] = None,
    fps: int = 24,
    arrow_column: Optional[str] = None,
    arrow_filename_column: Optional[str] = None,
    arrow_caption_key: Optional[str] = None,
    resolve_subject_ids: bool = False,
    global_only_in_first_segment: bool = False,
    no_desc_long_after_first: bool = False,
    normalize_action_timestamp: bool = False,
) -> List[PromptEntry]:
    """
    Load prompt entries from a directory containing .json and/or .arrow files.

    All .json and .arrow files in the directory are discovered and loaded in
    sorted order.  The function transparently delegates to ``_load_from_json``
    for each JSON file and ``_load_from_arrow`` for the collection of Arrow
    files (which already supports directory-level batching internally).

    Args:
        dir_path: Path to the directory.
        desc_field_first / desc_field_rest: Passed through to JSON loader.
        max_segments: Maximum total entries (applied across all files).
        keep_keywords: Passed through to JSON loader (whitelist).
        fps: Passed through to JSON loader.
        arrow_column: Column name for Arrow files.
        arrow_filename_column: Optional filename column for Arrow files.

    Returns:
        Combined list of PromptEntry tuples from all discovered files.
    """
    json_files = sorted(dir_path.glob("*.json"))
    arrow_files = sorted(dir_path.glob("*.arrow"))

    if not json_files and not arrow_files:
        raise FileNotFoundError(
            f"No .json or .arrow files found in directory: {dir_path}"
        )

    logger.info(
        f"Directory prompt source: {dir_path}  "
        f"({len(json_files)} .json, {len(arrow_files)} .arrow)"
    )

    all_entries: List[PromptEntry] = []
    remaining = max_segments  # None = unlimited

    # --- JSON files ---
    for jf in json_files:
        entries = _load_from_json(
            jf,
            max_segments=remaining,
            keep_keywords=keep_keywords,
            fps=fps,
            desc_field_first=desc_field_first,
            desc_field_rest=desc_field_rest,
            resolve_subject_ids=resolve_subject_ids,
            global_only_in_first_segment=global_only_in_first_segment,
            no_desc_long_after_first=no_desc_long_after_first,
            normalize_action_timestamp=normalize_action_timestamp,
        )
        all_entries.extend(entries)
        if remaining is not None:
            remaining -= len(entries)
            if remaining <= 0:
                logger.info(
                    f"  Reached max_segments={max_segments} after JSON files, stopping."
                )
                break

    # --- Arrow files ---
    if arrow_files and (remaining is None or remaining > 0):
        if not arrow_column:
            raise ValueError(
                f"Directory '{dir_path}' contains .arrow files but 'arrow_column' is not set. "
                "Please specify which column to read via inference.arrow_column in your config "
                "or +inference.arrow_column=<column_name> on the CLI."
            )
        # _load_from_arrow already handles a directory, but here we already
        # have the individual files.  We call _load_single_arrow per file to
        # honour the remaining budget accurately.
        for af in arrow_files:
            file_entries = _load_single_arrow(
                af,
                arrow_column=arrow_column,
                arrow_filename_column=arrow_filename_column,
                arrow_caption_key=arrow_caption_key,
                keep_keywords=keep_keywords,
                max_entries=remaining,
                global_offset=len(all_entries),
            )
            all_entries.extend(file_entries)
            if remaining is not None:
                remaining -= len(file_entries)
                if remaining <= 0:
                    logger.info(
                        f"  Reached max_segments={max_segments} after Arrow files, stopping."
                    )
                    break

    logger.info(
        f"Total: {len(all_entries)} prompt entries from directory "
        f"({len(json_files)} .json + {len(arrow_files)} .arrow)"
    )
    return all_entries


def _load_prompt_entries(
    prompt_arg: str,
    desc_field_first: str = "",
    desc_field_rest: str = "",
    max_segments: Optional[int] = None,
    keep_keywords: Optional[List[str]] = None,
    fps: int = 24,
    arrow_column: Optional[str] = None,
    arrow_filename_column: Optional[str] = None,
    arrow_caption_key: Optional[str] = None,
    resolve_subject_ids: bool = False,
    global_only_in_first_segment: bool = False,
    no_desc_long_after_first: bool = False,
    normalize_action_timestamp: bool = False,
) -> List[PromptEntry]:
    """
    Load prompt entries from a JSON file, an Arrow file, a directory, a text file, or use as-is.

    Returns:
        List of PromptEntry tuples: (prompt_string, frame_boundaries, filename).
    """
    try:
        prompt_path = Path(prompt_arg)
        is_file = prompt_path.is_file()
    except OSError:
        is_file = False
        prompt_path = None

    # Also check if it's a directory
    try:
        is_dir = prompt_path.is_dir() if prompt_path else False
    except OSError:
        is_dir = False

    if is_file and prompt_path.suffix.lower() == ".json":
        return _load_from_json(
            prompt_path,
            max_segments,
            keep_keywords,
            fps=fps,
            desc_field_first=desc_field_first,
            desc_field_rest=desc_field_rest,
            resolve_subject_ids=resolve_subject_ids,
            global_only_in_first_segment=global_only_in_first_segment,
            no_desc_long_after_first=no_desc_long_after_first,
            normalize_action_timestamp=normalize_action_timestamp,
        )
    elif is_file and prompt_path.suffix.lower() == ".arrow":
        if not arrow_column:
            raise ValueError(
                "An .arrow file was provided as prompt, but 'arrow_column' is not set. "
                "Please specify which column to read via inference.arrow_column in your config "
                "or +inference.arrow_column=<column_name> on the CLI."
            )
        return _load_from_arrow(
            prompt_path,
            arrow_column=arrow_column,
            arrow_filename_column=arrow_filename_column,
            arrow_caption_key=arrow_caption_key,
            keep_keywords=keep_keywords,
            max_entries=max_segments,
        )
    elif is_dir:
        return _load_from_directory(
            prompt_path,
            desc_field_first=desc_field_first,
            desc_field_rest=desc_field_rest,
            max_segments=max_segments,
            keep_keywords=keep_keywords,
            fps=fps,
            arrow_column=arrow_column,
            arrow_filename_column=arrow_filename_column,
            arrow_caption_key=arrow_caption_key,
            global_only_in_first_segment=global_only_in_first_segment,
            resolve_subject_ids=resolve_subject_ids,
            no_desc_long_after_first=no_desc_long_after_first,
            normalize_action_timestamp=normalize_action_timestamp,
        )
    elif is_file:
        logger.info(f"Loading prompt entries from text file: {prompt_path}")
        with open(prompt_path, "r", encoding="utf-8") as f:
            entries = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(entries)} prompt entries from text file")
        return [(e, None, None, None) for e in entries]
    else:
        entry = prompt_arg
        return [(entry, None, None, None)]


def _load_from_json(
    json_path: Path,
    max_segments: Optional[int],
    keep_keywords: Optional[List[str]],
    fps: int = 24,
    desc_field_first: str = "",
    desc_field_rest: str = "",
    resolve_subject_ids: bool = False,
    global_only_in_first_segment: bool = False,
    no_desc_long_after_first: bool = False,
    normalize_action_timestamp: bool = False,
) -> List[PromptEntry]:
    """
    Load prompt entries from a JSON file.

    Two modes:
    1. Default (desc_field_first/rest set): per-event extraction with ||| joining.
    2. Whole-item mode (both empty): each event serialised as JSON string.
    """
    logger.info(f"Loading prompt entries from JSON: {json_path}")
    logger.info(
        f"  desc_field_first={desc_field_first}, desc_field_rest={desc_field_rest}, max_segments={max_segments}, fps={fps}"
    )
    if keep_keywords:
        logger.info(
            f"  keep_keywords={keep_keywords}  (whole-item serialisation mode, whitelist)"
        )

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Mixed suites may contain both camera-controlled and text-only samples.
    # Resolve each entry independently; one text-only entry must not disable
    # the camera trajectories on every other sample in the input file.
    camera_case_count = sum(
        1 for item in data
        if isinstance(item, dict)
        and isinstance(item.get("camera_trajectory_spec"), dict)
        and item["camera_trajectory_spec"].get("ground_truth_npz")
    )
    if camera_case_count:
        logger.info(
            f"  Detected {camera_case_count}/{len(data)} entries with per-case "
            "camera_trajectory NPZ overrides."
        )

    entries: List[PromptEntry] = []
    for idx, item in enumerate(data):
        # Canonicalize legacy events/subjects_lifecycle -> segments/subject_list
        # so all downstream reads (and the serialized prompt) use one schema.
        item = _canonicalize_stage2_keys(item)
        test_id = item.get("test_id", f"{idx:04d}")
        test_type = item.get("test_type", "unknown")
        # Keep every sidecar aligned under one stable, identifier-only stem:
        #   <org_raw_id>.mp4 / <org_raw_id>.json / <org_raw_id>.txt
        # Some legacy benchmark suites store the original raw id in
        # ``test_id`` instead, so retain that as a compatibility fallback.
        raw_output_id = item.get("org_raw_id") or test_id
        output_filename = sanitize_basename(
            str(raw_output_id), max_chars=MAX_BASENAME_CHARS
        )

        # ---- Capture bench-specific metadata BEFORE any stripping ----
        # entry_meta is always populated for JSON suites so downstream code
        # (filename routing, per-segment control logic, evaluation hooks)
        # can read these fields even after they are removed from the JSON
        # dict that ends up at the text encoder.
        entry_meta: Dict[str, Any] = {
            k: item.get(k) for k in _BENCH_META_TOP_LEVEL_KEYS if k in item
        }

        # Per-event bench keys (joint bench: segment_control etc.).
        raw_segments_for_meta = item.get("segments", []) or []
        segments_meta_list: List[Dict[str, Any]] = []
        any_segment_meta = False
        for seg in raw_segments_for_meta:
            if isinstance(seg, dict):
                em = {k: seg.get(k) for k in _BENCH_META_SEGMENT_KEYS if k in seg}
                if em:
                    any_segment_meta = True
                segments_meta_list.append(em)
            else:
                segments_meta_list.append({})
        if any_segment_meta:
            entry_meta["segments_meta"] = segments_meta_list

        # Per-case camera trajectory NPZ override (camera control bench only).
        # Resolve the relative npz path against the directory the suite.json
        # sits in.
        spec = item.get("camera_trajectory_spec")
        if isinstance(spec, dict) and spec.get("ground_truth_npz"):
            npz_rel = str(spec["ground_truth_npz"])
            npz_path = (json_path.parent / npz_rel).resolve()
            if not npz_path.is_file():
                if (spec.get("params") or {}).get(
                    "format"
                ) == "demo_camera_precomputed_v1":
                    raise FileNotFoundError(
                        f"Precomputed demo camera NPZ missing: {npz_path}"
                    )
                logger.warning(
                    f"[{test_id}] camera_trajectory NPZ does not exist: "
                    f"{npz_path} — falling back to global camera_trajectory."
                )
            else:
                entry_meta["camera_trajectory"] = str(npz_path)

        if not desc_field_first and not desc_field_rest:
            raw_segments = item.get("segments", [])
            if max_segments is not None and len(raw_segments) > max_segments:
                raw_segments = raw_segments[:max_segments]

            frame_boundaries: List[int] = []
            has_ts = True
            for seg in raw_segments:
                st = seg.get("start_time")
                et = seg.get("end_time")
                if st is not None and et is not None:
                    start_sec = _parse_timestamp(st)
                    end_sec = _parse_timestamp(et)
                    if len(frame_boundaries) == 0:
                        frame_boundaries.append(int(start_sec * fps))
                    frame_boundaries.append(int(end_sec * fps))
                else:
                    has_ts = False

            boundaries = (
                frame_boundaries if (has_ts and len(frame_boundaries) >= 2) else None
            )
            camera_spec = item.get("camera_trajectory_spec") or {}
            precomputed_params = camera_spec.get("params") or {}
            if precomputed_params.get("format") == "demo_camera_precomputed_v1":
                if max_segments is not None:
                    raise ValueError(
                        "Precomputed demo camera requires all original segments; "
                        "disable max_segments"
                    )
                if int(camera_spec.get("fps", 0)) != fps:
                    raise ValueError(
                        "Precomputed demo camera FPS differs from inference FPS"
                    )
                explicit = precomputed_params["segment_boundaries_pixel"]
                if (
                    len(explicit) != len(raw_segments) + 1
                    or explicit[0] != 0
                    or any(a >= b for a, b in zip(explicit[:-1], explicit[1:]))
                ):
                    raise ValueError("Invalid precomputed camera pixel boundaries")
                boundaries = [int(value) for value in explicit]
                logger.info(
                    f"[{test_id}] precomputed demo camera: using explicit per-turn "
                    f"latent-aligned pixel boundaries {boundaries}; "
                    f"right_padding_frames={precomputed_params.get('right_padding_frames')}"
                )

            if raw_segments:
                entry_str, num_prompt_segments = _build_whole_item_prompts(
                    item=item,
                    segments=raw_segments,
                    keep_keywords=keep_keywords,
                    resolve_subject_ids=resolve_subject_ids,
                    global_only_in_first_segment=global_only_in_first_segment,
                    no_desc_long_after_first=no_desc_long_after_first,
                    normalize_action_timestamp=normalize_action_timestamp,
                )
            else:
                if keep_keywords:
                    cleaned = _keep_keys_in_dict(item, list(keep_keywords))
                else:
                    import copy as _copy_mod

                    cleaned = _copy_mod.deepcopy(item)
                cleaned = _remove_keys_from_dict(cleaned, list(_BENCH_STRIP_KEYS))
                entry_str = json.dumps(cleaned, ensure_ascii=False)
                num_prompt_segments = 1

            if boundaries is not None and len(boundaries) != num_prompt_segments + 1:
                logger.warning(
                    f"[{test_id}] {test_type}: boundary/prompt mismatch after "
                    f"prompt shaping: len(boundaries)={len(boundaries)} "
                    f"num_prompts={num_prompt_segments}; falling back to even split."
                )
                boundaries = None

            bench_filename = output_filename

            entries.append((entry_str, boundaries, bench_filename, entry_meta))
            logger.info(
                f"[{test_id}] {test_type}: whole-item mode -> {num_prompt_segments} prompts "
                f"(stripped: bench-keys, kept: {keep_keywords or 'all'}, "
                f"resolve_subject_ids={resolve_subject_ids}), "
                f"frame_boundaries={boundaries}, filename={bench_filename}"
                + (
                    f", camera_npz={entry_meta['camera_trajectory']}"
                    if entry_meta.get("camera_trajectory")
                    else ""
                )
            )
            continue

        segments = item.get("segments", [])

        if not segments:
            logger.warning(f"Entry {test_id} ({test_type}) has no segments, skipping.")
            continue

        if max_segments is not None and len(segments) > max_segments:
            logger.info(
                f"Entry {test_id} ({test_type}): truncating from "
                f"{len(segments)} to {max_segments} segments"
            )
            segments = segments[:max_segments]

        prompts = []
        frame_boundaries = []
        has_timestamps = True

        for seg_idx, seg in enumerate(segments):
            seg = _strip_camera_text_blocks_for_camera_segment(seg)
            if seg_idx == 0:
                _field = desc_field_first
            else:
                _field = desc_field_rest
                if not _field:  # desc_field_rest is empty -> skip remaining events
                    break
            prompt_text = seg.get(_field) or _fallback_desc(seg) or ""
            prompt_text = prompt_text.strip()

            if prompt_text:
                prompts.append(prompt_text)

                start_t = seg.get("start_time")
                end_t = seg.get("end_time")
                if start_t is not None and end_t is not None:
                    start_sec = _parse_timestamp(start_t)
                    end_sec = _parse_timestamp(end_t)
                    if len(frame_boundaries) == 0:
                        frame_boundaries.append(int(start_sec * fps))
                    frame_boundaries.append(int(end_sec * fps))
                else:
                    has_timestamps = False

        if not prompts:
            logger.warning(
                f"Entry {test_id} ({test_type}) produced no prompts, skipping."
            )
            continue

        entry_str = PROMPT_SEPARATOR.join(prompts)

        boundaries = None
        if has_timestamps and len(frame_boundaries) == len(prompts) + 1:
            boundaries = frame_boundaries
            logger.info(
                f"[{test_id}] {test_type}: frame_boundaries={boundaries} "
                f"(fps={fps}, formula: seconds * {fps})"
            )
        else:
            if not has_timestamps:
                logger.info(
                    f"[{test_id}] {test_type}: missing timestamps in some events, "
                    f"falling back to even split."
                )

        bench_filename = output_filename

        entries.append((entry_str, boundaries, bench_filename, entry_meta))
        logger.info(
            f"[{test_id}] {test_type}: {len(prompts)} events -> 1 multi-prompt entry, "
            f"filename={bench_filename}"
            + (
                f", camera_npz={entry_meta['camera_trajectory']}"
                if entry_meta.get("camera_trajectory")
                else ""
            )
        )

    logger.info(f"Loaded {len(entries)} prompt entries from JSON")
    return entries


# ============================================================================
# Argument parsing (unified)
# ============================================================================


def parse_args():
    """Load configuration via Hydra Compose API.

        All inference-specific fields (rollout_type, max_cache_segments, etc.) are
        defined in configs/config.yaml with sensible defaults. Experiment presets
        and per-run overrides are passed via CLI:

    torchrun ... hymm/inference.py +experiment=inference rollout_type=seg_ar
    """
    return _hydra_parse_args(mode="inference")


# ============================================================================
# Config validation
# ============================================================================


def _get_required_inference_params() -> list:
    """Read the 'inference' section keys from configs/config.yaml.

    This avoids hardcoding the required parameter list — any new key added to
    config.yaml's ``inference:`` block is automatically enforced.

    Nested dict keys are excluded here. The selected runtime option groups
    (sliding_window and sparse_attention) are validated by the configuration
    contract in config.py; other nested groups such as monitor are not promoted.
    """
    config_yaml_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs",
        "config.yaml",
    )
    with open(config_yaml_path, "r") as f:
        raw = yaml.safe_load(f)

    inference_section = raw.get("inference", {})
    if not inference_section:
        raise RuntimeError(
            f"No 'inference' section found in {config_yaml_path}. "
            f"Please add an 'inference:' block with default values."
        )
    # Scalar requirements only; runtime option groups have their own validator.
    return [k for k, v in inference_section.items() if not isinstance(v, dict)]


def _validate_inference_config(cfg) -> None:
    """Validate that all required inference parameters are defined in the config.

    Raises ``KeyError`` with a descriptive message listing every missing field
    so that the user can fix them all in one go.
    """
    if not hasattr(cfg, "inference"):
        raise KeyError(
            "Config has no 'inference' section. "
            "Make sure configs/config.yaml defines an 'inference:' block."
        )

    from hymm.distributed.tensor_parallel import validate_tensor_parallel_config
    validate_tensor_parallel_config(cfg)
    required_params = _get_required_inference_params()
    missing = [
        param
        for param in required_params
        if not hasattr(cfg.inference, param) or param not in cfg.inference
    ]

    if missing:
        raise KeyError(
            f"The following inference parameter(s) are missing from config.yaml "
            f"(or experiment overrides): {missing}. "
            f"Please add them to the 'inference:' section in configs/config.yaml "
            f"or provide them via CLI overrides (e.g. +inference.{missing[0]}=<value>)."
        )


def _auto_save_path(args, default: str, logger) -> str:
    """Derive ``save_path`` from ``dit_weight`` and ``prompt`` when not set explicitly.

    Convention:
        <experiment_root>/validation/<suite_name>/iter_<step>

    where:
        experiment_root = parent of the ``checkpoints/`` dir inside dit_weight
        suite_name      = stem of the prompt JSON file
        step            = the number after ``global_step-`` in dit_weight path

    Falls back to ``default`` if the paths don't match the expected pattern.
    """
    dit_weight = getattr(args, "dit_weight", None) or ""
    prompt = getattr(args.inference, "prompt", None) or ""

    # Extract step number: .../global_step-27000/...
    m = re.search(r"global_step-(\d+)", dit_weight)
    if not m:
        logger.info(
            f"  save_path: auto-derive failed (no global_step-N in dit_weight={dit_weight!r}), "
            f"using default={default}"
        )
        return default
    step = m.group(1)

    # Experiment root: everything before /checkpoints/
    ckpt_idx = dit_weight.find("/checkpoints/")
    if ckpt_idx < 0:
        logger.info(
            f"  save_path: auto-derive failed (no /checkpoints/ in dit_weight={dit_weight!r}), "
            f"using default={default}"
        )
        return default
    experiment_root = dit_weight[:ckpt_idx]

    # Suite name from prompt JSON
    suite_name = Path(prompt).stem if prompt else "unknown"

    derived = f"{experiment_root}/validation/{suite_name}/iter_{step}"
    logger.info(f"  save_path: auto-derived from dit_weight + prompt -> {derived}")
    return derived


# ============================================================================
# Unified main
# ============================================================================


def main():
    args = parse_args()
    _validate_inference_config(args)

    if isinstance(args.inference.video_size, int):
        args.inference.video_size = [
            args.inference.video_size,
            args.inference.video_size,
        ]
    elif (
        isinstance(args.inference.video_size, list)
        and len(args.inference.video_size) == 1
    ):
        args.inference.video_size = [
            args.inference.video_size[0],
            args.inference.video_size[0],
        ]

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    sp_size = (
        getattr(args.parallelism, "sp_size", 1)
        if hasattr(args, "parallelism")
        else getattr(args, "sp_size", 1)
    )
    tp_size = (
        getattr(args.parallelism, "tp_size", 1)
        if hasattr(args, "parallelism") else getattr(args, "tp_size", 1)
    )
    from hymm.distributed.tensor_parallel import validate_parallel_sizes
    from hymm.distributed.parallel_states import get_model_parallel_info
    model_parallel_size = validate_parallel_sizes(sp_size, tp_size, world_size)
    world_mesh = None

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
        world_mesh = parallel_dims.build_mesh(device_type="cuda")
        initialize_sequence_parallel_state_v2(
            parallel_dims,
            world_mesh,
            getattr(args.parallelism, "use_dynamic_ring_attention", False)
            if hasattr(args, "parallelism")
            else getattr(args, "use_dynamic_ring_attention", False),
        )

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    if world_size > 1:
        sp_group_id = rank // model_parallel_size
        num_sp_groups = world_size // model_parallel_size
        rank_within_sp = rank % model_parallel_size
    else:
        sp_group_id = 0
        num_sp_groups = 1
        rank_within_sp = 0

    # Group scheduling and output ownership use the entire SP x TP replica.
    # Retain the historical SP variable names used by shard metadata.
    # Reduce log verbosity for non-primary ranks within each model group.
    # Only rank 0 of each SP group prints INFO logs; others only print WARNING+.
    # Use logger.configure() to set extra globally so all modules (e.g. bidirectional.py)
    # that import logger from loguru can also access {extra[rank]} in the format string.
    logger.configure(extra={"rank": rank})
    logger.remove()
    if rank_within_sp != 0:
        logger.add(
            sys.stderr,
            level="WARNING",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:^8} | [Rank {extra[rank]}]: {message}",
        )
    else:
        logger.add(
            sys.stderr,
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:^8} | [Rank {extra[rank]}]: {message}",
        )

    if world_size > 1:
        logger.info(
            f"Distributed: rank={rank}/{world_size}, sp_size={nccl_info.sp_size}, tp_size={tp_size}"
        )

    # Cross-node data sharding for pdsh-based independent node launches.
    # When each node runs NNODES=1 independently (sp_size == world_size),
    # num_sp_groups == 1 within each node, so no intra-node DP sharding happens.
    # Use GLOBAL_NODE_RANK / GLOBAL_NUM_NODES to shard data across nodes.
    global_num_nodes = int(os.environ.get("GLOBAL_NUM_NODES", 1))
    global_node_rank = int(os.environ.get("GLOBAL_NODE_RANK", 0))
    if global_num_nodes > 1:
        # Each node's local SP groups get a globally unique ID
        sp_group_id = global_node_rank * num_sp_groups + sp_group_id
        num_sp_groups = global_num_nodes * num_sp_groups
        logger.info(
            f"Cross-node sharding: global_node_rank={global_node_rank}/{global_num_nodes}, "
            f"effective sp_group_id={sp_group_id}/{num_sp_groups}"
        )

    # --- Ensure dit_weight safetensors exists (auto-convert from shards if needed) ---
    if args.inference.dit_weight_quantization == "none":
        args.dit_weight = _ensure_dit_weight_safetensors(
            args.dit_weight, rank=rank, world_size=world_size
        )
    else:
        from hymm.models.dit.modules.int8_stored_bf16_linear import resolve_w8_bf16_checkpoint
        resolve_w8_bf16_checkpoint(args.inference, args.dit_weight)

    # --- Select pipeline based on rollout_type ---
    rollout_type = args.inference.rollout_type
    if rollout_type not in ("one_shot", "seg_ar", "chunk_ar"):
        raise ValueError(f"Unsupported inference rollout_type: {rollout_type!r}")
    if (
        bool(
            getattr(
                args.inference,
                "seg_ar_segment_vae_decode",
                False,
            )
        )
        and rollout_type != "seg_ar"
    ):
        raise ValueError(
            "inference.seg_ar_segment_vae_decode=true requires "
            f"rollout_type='seg_ar', got {rollout_type!r}."
        )
    profile_memory = bool(getattr(args.inference, "profile_memory", False))
    metrics_id = hashlib.sha256(str(args.inference.save_path).encode()).hexdigest()[:16]
    metrics_root = Path(os.environ.get("INFERENCE_METRICS_DIR", "/tmp/nomas-inference-metrics")) / metrics_id
    metrics_root.mkdir(parents=True, exist_ok=True)
    sample_sequence = 0
    with InferenceMetrics(device, detailed=profile_memory,
                          jsonl=metrics_root / f"model_load.rank{rank}.jsonl") as load_metrics:
        if rollout_type == "one_shot":
            pipeline = MultiPromptBidirectionalPipeline(
                args=args,
                device=device,
                dit_model_path=args.dit_weight,
                world_mesh=world_mesh,
            )
        else:
            pipeline = CPTCausalInferencePipeline(
                args=args,
                device=device,
                dit_model_path=args.dit_weight,
                world_mesh=world_mesh,
            )

    Path(args.inference.save_path).mkdir(parents=True, exist_ok=True)
    Path(args.inference.save_path, f"model_load.rank{rank}.json").write_text(
        json.dumps(load_metrics.summary(), indent=2) + "\n"
    )
    raw_prompt = args.inference.prompt or "A beautiful sunset over the ocean"
    video_fps = args.inference.video_fps
    skip_first_turn = bool(getattr(args.inference, "skip_first_turn", False))
    if skip_first_turn:
        if (
            rollout_type != "seg_ar"
            or args.inference.desc_field_first
            or args.inference.desc_field_rest
            or args.inference.video_length != "event"
        ):
            raise ValueError(
                "skip_first_turn requires seg_ar, event length and whole-item JSON"
            )
        logger.warning(
            "skip_first_turn=true: generate ALL turns with full KV history; "
            "omit turn 1 only from the saved video and overlay. "
            "Use a fresh output directory to avoid skip_existing reusing full-session videos."
        )

    # Validate: frame mode does not support JSON whole-item prompt format
    video_length_mode = args.inference.video_length  # "event" or "frame"
    _df_first = args.inference.desc_field_first
    _df_rest = args.inference.desc_field_rest
    assert not (video_length_mode == "frame" and not _df_first and not _df_rest), (
        "video_length='frame' is incompatible with JSON whole-item prompt mode "
        "(desc_field_first and desc_field_rest are both empty). "
        "In frame mode, please set desc_field_first (and optionally desc_field_rest) "
        "to extract text prompts from events, e.g. desc_field_first='visual_description_long'."
    )
    _max_segments = args.inference.max_segments
    # Validate: frame mode with desc_field_first set and max_segments > 1 requires desc_field_rest
    if video_length_mode == "frame" and _df_first:
        if _max_segments is None or _max_segments > 1:
            assert _df_rest, (
                "video_length='frame' with max_segments>1 (or null) requires desc_field_rest to be set "
                "when desc_field_first is not empty. "
                f"Got desc_field_first='{_df_first}', desc_field_rest='{_df_rest}', max_segments={_max_segments}. "
                "In frame mode with multiple events, each event after the first needs a text prompt, "
                "so desc_field_rest must also be specified, "
                "e.g. desc_field_rest='visual_description_long'."
            )
    if video_length_mode == "event" and _df_first:
        if _max_segments is not None and _max_segments == 1:
            # Single-event mode: use the first event's timestamps for video length.
            # desc_field_rest must be empty (only the first event is used).
            assert not _df_rest, (
                "video_length='event' with max_segments=1 requires desc_field_rest to be empty, "
                f"because only the first event is used. Got desc_field_rest='{_df_rest}'. "
                "Set desc_field_rest='' to use single-event mode."
            )
        else:
            # Multi-event mode: each event needs a text prompt.
            # desc_field_rest must be set.
            assert _df_rest, (
                "video_length='event' with max_segments>1 (or null) requires desc_field_rest to be set "
                "when desc_field_first is not empty. "
                f"Got desc_field_first='{_df_first}', desc_field_rest='{_df_rest}', max_segments={_max_segments}. "
                "In event mode, each event needs a text prompt, so desc_field_rest must also be specified, "
                "e.g. desc_field_rest='description_change' or desc_field_rest='visual_description_long'."
            )

    # Mirror the training-side keep list when inference doesn't override it.
    # Training uses ``dataset.json_prompt_keep_keys`` (whitelist of fields
    # to retain in the serialised JSON prompt). Falling back here keeps
    # the JSON the text encoder sees in the same distribution as during
    # training. Bench-specific keys are stripped automatically by
    # ``_load_from_json`` regardless of this list.
    _inf_rk = getattr(args.inference, "keep_keywords", None)
    if not _inf_rk:
        _inf_rk = getattr(
            getattr(args, "dataset", object()), "json_prompt_keep_keys", None
        )
    _normalize_action_timestamp = bool(
        getattr(
            getattr(args, "dataset", object()),
            "json_prompt_normalize_action_timestamp",
            False,
        )
    )

    prompt_entries = _load_prompt_entries(
        raw_prompt,
        desc_field_first=args.inference.desc_field_first,
        desc_field_rest=args.inference.desc_field_rest,
        max_segments=args.inference.max_segments,
        keep_keywords=_inf_rk,
        fps=video_fps,
        arrow_column=args.inference.arrow_column,
        arrow_filename_column=args.inference.arrow_filename_column,
        arrow_caption_key=args.inference.arrow_caption_key,
        resolve_subject_ids=getattr(args.inference, "resolve_subject_ids", False),
        global_only_in_first_segment=getattr(
            args.inference, "global_only_in_first_segment", False
        ),
        no_desc_long_after_first=getattr(
            args.inference, "no_desc_long_after_first", False
        ),
        normalize_action_timestamp=_normalize_action_timestamp,
    )

    # Limit the number of inference iterations (prompt entries) if max_iters is set
    max_iters = args.inference.max_iters
    if max_iters is not None and max_iters > 0:
        if len(prompt_entries) > max_iters:
            logger.info(
                f"max_iters={max_iters}: truncating from {len(prompt_entries)} to {max_iters} entries"
            )
            prompt_entries = prompt_entries[:max_iters]

    seed = args.inference.seed if args.inference.seed is not None else 42
    negative_prompt = args.inference.neg_prompt or ""

    default_save = (
        "./outputs/one_shot.mp4"
        if rollout_type == "one_shot"
        else "./outputs/cpt_ar.mp4"
    )
    explicit_save = getattr(args.inference, "save_path", None)
    if explicit_save:
        base_save_path = explicit_save
    else:
        # Auto-derive: <experiment_root>/validation/<suite_name>/iter_<step>
        # from dit_weight path (.../checkpoints/global_step-XXXX/model.safetensors)
        # and prompt path (.../suite_name.json).
        base_save_path = _auto_save_path(args, default_save, logger)

    logger.info(f"  save_path: {base_save_path}")

    skip_existing = args.inference.skip_existing

    # ---- Path computation helper ----
    # Mirrors the inline naming logic below; used for both pre-filtering already-
    # generated entries (resume balancing) and as the source of truth in the loop.
    # IMPORTANT: total_orig_entries and orig_num_sp_groups must reflect the values
    # *before* any pre-filtering, so file names match what was previously written.
    def _compute_output_paths(
        prompt_entry,
        global_entry_idx,
        total_orig_entries,
        orig_num_sp_groups,
    ):
        if len(prompt_entry) == 4:
            entry, _fb, bench_filename, _meta = prompt_entry
        else:
            entry, _fb, bench_filename = prompt_entry
        prompts_local = [p.strip() for p in entry.split(PROMPT_SEPARATOR)]
        base_path = Path(base_save_path)
        is_dir_target = base_path.is_dir() or base_path.suffix.lower() not in {
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".webm",
        }
        if bench_filename:
            video_basename = f"{bench_filename}.mp4"
        else:
            first_prompt = prompts_local[0]
            safe_name = "".join(
                c if c.isalnum() or c in "_-" else "_" for c in first_prompt
            )[:80]
            if total_orig_entries == 1 and orig_num_sp_groups == 1:
                video_basename = f"{safe_name}.mp4"
            else:
                video_basename = f"{global_entry_idx:04d}_{safe_name}.mp4"
        max_bytes = 250
        stem = video_basename.rsplit(".", 1)[0]
        ext = ".mp4"
        encoded = stem.encode("utf-8")
        if len(encoded) + len(ext) > max_bytes:
            truncated = encoded[: max_bytes - len(ext)].decode("utf-8", errors="ignore")
            video_basename = truncated + ext
        if is_dir_target:
            save_dir = base_path
            output_path_local = str(save_dir / video_basename)
        else:
            if bench_filename:
                save_dir = base_path.parent
                output_path_local = str(save_dir / video_basename)
            elif total_orig_entries == 1 and orig_num_sp_groups == 1:
                save_dir = base_path.parent
                output_path_local = str(save_dir / video_basename)
            else:
                save_dir = base_path.parent
                output_path_local = str(save_dir / video_basename)
        prompt_txt_path_local = os.path.splitext(output_path_local)[0] + ".txt"
        return output_path_local, prompt_txt_path_local, save_dir, is_dir_target

    # ---- Deterministic sharding across SP groups (race-free) ----
    # IMPORTANT: shard the FULL entry list first, then apply skip_existing
    # *within* each SP group's slice. Doing it in the opposite order
    # (filter unfinished globally -> shard) is racy across nodes:
    # different nodes scan the filesystem at different times, so they see
    # different `unfinished` lengths and compute different slices. Some
    # entries can then end up unowned by any node (silently missing videos),
    # while others get duplicated. By making the slice depend only on the
    # original entry list (which is identical on all nodes), every entry has
    # exactly one owner regardless of how many videos already exist.
    total_orig_entries = len(prompt_entries)
    orig_num_sp_groups = num_sp_groups  # capture before any sharding
    indexed_entries = list(enumerate(prompt_entries))  # [(orig_idx, entry), ...]

    if num_sp_groups > 1:
        total_full = len(indexed_entries)
        shard_strategy = (
            str(
                getattr(
                    args.inference,
                    "prompt_shard_strategy",
                    "round_robin",
                )
            )
            .strip()
            .lower()
        )
        if shard_strategy == "duration_balanced":
            all_shards, shard_loads = _duration_balanced_prompt_shards(
                indexed_entries,
                num_shards=num_sp_groups,
                fps=video_fps,
            )
            indexed_entries = all_shards[sp_group_id]
            logger.info(
                f"SP group {sp_group_id}/{num_sp_groups}: assigned "
                f"{len(indexed_entries)} entries by duration-balanced LPT "
                f"(estimated video time={shard_loads[sp_group_id]:.3f}s, "
                f"all shard times={[round(v, 3) for v in shard_loads]}, "
                f"{total_full} total entries; deterministic shard)"
            )
        elif shard_strategy == "round_robin":
            indexed_entries = indexed_entries[sp_group_id::num_sp_groups]
            shard_time = sum(
                _estimate_prompt_entry_duration_seconds(entry, video_fps)
                for _orig_idx, entry in indexed_entries
            )
            logger.info(
                f"SP group {sp_group_id}/{num_sp_groups}: assigned "
                f"{len(indexed_entries)} entries "
                f"(round-robin stride={num_sp_groups} starting at "
                f"{sp_group_id} out of {total_full} total; "
                f"estimated video time={shard_time:.3f}s; deterministic shard)"
            )
        else:
            raise ValueError(
                "inference.prompt_shard_strategy must be "
                f"'duration_balanced' or 'round_robin', got {shard_strategy!r}"
            )

    # ---- Per-shard resume filter (skip already-generated entries) ----
    # Done AFTER sharding so each node only filters its own slice. The slice
    # boundaries do not depend on filesystem state, so ownership is stable.
    if skip_existing and len(indexed_entries) > 0:
        unfinished = []
        already_done = 0
        for orig_idx, prompt_entry in indexed_entries:
            output_path_pre, prompt_txt_pre, _save_dir_pre, _is_dir_pre = (
                _compute_output_paths(
                    prompt_entry,
                    orig_idx,
                    total_orig_entries,
                    orig_num_sp_groups,
                )
            )
            if os.path.isfile(output_path_pre) and os.path.isfile(prompt_txt_pre):
                already_done += 1
            else:
                unfinished.append((orig_idx, prompt_entry))
        if already_done > 0 or num_sp_groups > 1:
            logger.info(
                f"[Resume filter, SP group {sp_group_id}/{num_sp_groups}] "
                f"shard size: {len(indexed_entries)}, "
                f"already generated (mp4+txt exist): {already_done}, "
                f"remaining for this shard: {len(unfinished)}"
            )
        indexed_entries = unfinished

    skipped_count = 0

    for entry_idx, (global_entry_idx, prompt_entry) in enumerate(indexed_entries):
        if args.dit.seg_ar_attention_backend == "flex":
            from hymm.models.dit.modules.attention.range_attention import (
                clear_range_cache, reset_range_stats,
            )
            clear_range_cache()
            reset_range_stats()
        # PromptEntry can be a 3-tuple (legacy) or 4-tuple (with per-entry meta).
        # Unpack defensively so older callers / formats still work.
        if len(prompt_entry) == 4:
            entry, frame_boundaries, bench_filename, entry_meta = prompt_entry
        else:
            entry, frame_boundaries, bench_filename = prompt_entry
            entry_meta = None
        prompts = [p.strip() for p in entry.split(PROMPT_SEPARATOR)]
        logger.info(
            f"=== Entry {entry_idx + 1}/{len(indexed_entries)} (global_idx={global_entry_idx}) ==="
        )
        logger.info(f"Parsed {len(prompts)} prompts:")
        for idx, p in enumerate(prompts):
            logger.info(f"  [{idx}] {p}")

        # --- Pre-compute output path (use orig totals so file names stay stable) ---
        output_path, prompt_txt_path_for_skip, save_dir, is_dir_target = (
            _compute_output_paths(
                prompt_entry,
                global_entry_idx,
                total_orig_entries,
                orig_num_sp_groups,
            )
        )
        # Ensure save_dir exists (helper does not create it to keep it pure).
        save_dir.mkdir(parents=True, exist_ok=True)

        # --- Dump parsed per-segment prompts for inspection (rank 0 only) ---
        # Writes the exact per-segment JSON strings fed to the text encoder.
        if int(os.environ.get("RANK", "0")) == 0:
            try:
                _seg_dump = {
                    "skip_first_turn": skip_first_turn,
                    "global_idx": global_entry_idx,
                    "bench_filename": bench_filename,
                    "frame_boundaries": frame_boundaries,
                    "num_segments": len(prompts),
                    "segments": [
                        {"idx": _i, "prompt": _p} for _i, _p in enumerate(prompts)
                    ],
                }
                _seg_path = Path(output_path).with_suffix(".json")
                with open(_seg_path, "w", encoding="utf-8") as _f:
                    json.dump(_seg_dump, _f, ensure_ascii=False, indent=2)
                logger.info(f"  Per-segment prompts dumped to: {_seg_path}")
            except Exception as _e:
                logger.warning(f"  Failed to dump per-segment prompts: {_e}")

        # --- Skip only if both the video (.mp4) and prompt (.txt) files exist ---
        # Race-condition guard: another SP group may have written this file
        # between pre-filtering and now (rare, but harmless to re-check).
        if (
            skip_existing
            and os.path.isfile(output_path)
            and os.path.isfile(prompt_txt_path_for_skip)
        ):
            logger.info(f"  Skipping (both mp4 and txt exist): {output_path}")
            skipped_count += 1
            continue

        # Determine boundaries
        if rollout_type == "one_shot":
            cli_boundaries = getattr(args, "prompt_video_boundaries", None)
        else:
            cli_boundaries = args.inference.segment_frame_boundaries

        if cli_boundaries is not None:
            entry_boundaries = cli_boundaries
            logger.info(f"  Using CLI boundaries: {entry_boundaries}")
        elif frame_boundaries is not None:
            entry_boundaries = frame_boundaries
            logger.info(f"  Using JSON pixel frame boundaries: {frame_boundaries}")
        else:
            entry_boundaries = None
            logger.info(f"  No boundary info; will use even split.")

        video_length_mode = args.inference.video_length  # "event" or "frame"
        if video_length_mode == "frame":
            # Use the configured num_frames directly
            entry_num_frames = args.inference.num_frames
            logger.info(
                f"  video_length='frame': using configured num_frames={entry_num_frames}"
            )
            # Rescale JSON boundaries to fit the fixed num_frames
            if (
                entry_boundaries is not None
                and entry_boundaries[-1] != entry_num_frames
                and entry_boundaries[-1] > 0
            ):
                orig_boundaries = entry_boundaries
                scale = entry_num_frames / entry_boundaries[-1]
                entry_boundaries = [int(round(b * scale)) for b in entry_boundaries]
                entry_boundaries[0] = 0
                entry_boundaries[-1] = entry_num_frames
                # Ensure monotonically increasing
                for i in range(1, len(entry_boundaries)):
                    if entry_boundaries[i] <= entry_boundaries[i - 1]:
                        entry_boundaries[i] = entry_boundaries[i - 1] + 1
                if entry_boundaries[-1] != entry_num_frames:
                    entry_boundaries[-1] = entry_num_frames
                logger.info(
                    f"  Rescaled boundaries to fit num_frames={entry_num_frames}: "
                    f"{orig_boundaries} -> {entry_boundaries}"
                )
        elif frame_boundaries is not None and cli_boundaries is None:
            # video_length='event': derive length from event timestamps
            entry_num_frames = frame_boundaries[-1]
            logger.info(
                f"  video_length='event': overriding num_frames: {args.inference.num_frames} -> {entry_num_frames}"
            )
        else:
            entry_num_frames = args.inference.num_frames

        precomputed_spec = (entry_meta or {}).get("camera_trajectory_spec") or {}
        if (precomputed_spec.get("params") or {}).get(
            "format"
        ) == "demo_camera_precomputed_v1":
            expected_bounds = precomputed_spec["params"]["segment_boundaries_pixel"]
            if (
                rollout_type != "seg_ar"
                or list(entry_boundaries or []) != expected_bounds
                or entry_num_frames != expected_bounds[-1]
            ):
                raise ValueError(
                    "Precomputed demo camera requires seg_ar and original explicit "
                    "turn boundaries; do not override video length or boundaries"
                )

        saved_video_frame_offset = 0
        if skip_first_turn:
            from hymm.models.camera.geometry.video_output_trim import first_turn_offset

            saved_video_frame_offset = first_turn_offset(
                entry_boundaries, entry_num_frames
            )

        # Reset KV cache for AR modes
        if rollout_type != "one_shot":
            pipeline.kv_cache = None
            pipeline.cache_valid_len = 0
            pipeline.global_frame_counter = 0
            if bool(
                getattr(
                    args.vae,
                    "stable_prefix_streaming",
                    False,
                )
            ):
                pipeline._reset_interactive_pixelumm_decode_state()

        # Per-entry camera_trajectory override (camera control bench format).
        # Falls back to the global args.inference.camera_trajectory.
        global_camera_trajectory = getattr(args.inference, "camera_trajectory", None)
        entry_camera_trajectory = (
            entry_meta.get("camera_trajectory") if entry_meta else None
        )
        effective_camera_trajectory = (
            entry_camera_trajectory or global_camera_trajectory
        )
        if entry_camera_trajectory:
            logger.info(
                f"  Using per-entry camera_trajectory: {entry_camera_trajectory} "
                f"(overriding global {global_camera_trajectory!r})"
            )

        # ---- Per-segment text drop & camera-off (joint bench: segments_meta) ----
        # Triggers (only fire when entry_meta carries the relevant keys; benches
        # without segments_meta keep the legacy "all-on" behaviour):
        #   joint_caption_constraint == "empty"  →  drop text on this segment
        #                                           (becomes text-uncond)
        #   segment_control          == "prompt" →  has_camera = False on this
        #                                           segment (camera attention
        #                                           output gated to 0 for that
        #                                           segment's tokens)
        segment_text_drops = None
        segment_has_camera = None
        overlay_segment_has_camera = None
        segments_meta = entry_meta.get("segments_meta") if entry_meta else None
        if segments_meta and len(segments_meta) == len(prompts):
            text_drops_list = [
                em.get("joint_caption_constraint") == "empty" for em in segments_meta
            ]
            has_camera_list = [
                em.get("segment_control") != "prompt" for em in segments_meta
            ]
            overlay_segment_has_camera = has_camera_list
            # Only pass through when at least one segment differs from the
            # default (camera on, text on). Keeps logs / kwargs clean for the
            # camera_control_v2.1 / multiprompt benches that don't carry these
            # fields, and for joint cases that happen to be all "camera"
            # control / no empty captions.
            if any(text_drops_list):
                segment_text_drops = text_drops_list
            if not all(has_camera_list):
                segment_has_camera = has_camera_list
            if segment_text_drops is not None or segment_has_camera is not None:
                logger.info(
                    f"  segments_meta present: "
                    f"segment_text_drops={segment_text_drops}, "
                    f"segment_has_camera={segment_has_camera}"
                )

        sample_metrics_path = metrics_root / f"{Path(output_path).stem}.rank{rank}.jsonl"
        inference_mode_enabled = bool(args.inference.inference_mode)
        compile_before = None
        if args.inference.torch_compile:
            from hymm.models.dit.modules.nomas_compile_prepost import compilation_snapshot, compilation_delta
            compile_before = compilation_snapshot()
        with offline_sample_mode(inference_mode_enabled), InferenceMetrics(
            device, detailed=profile_memory, jsonl=sample_metrics_path
        ) as sample_metrics:
            # Persist the final DiT latents alongside each generated video.
            if getattr(args.inference, "save_latents", False):
                if rollout_type == "one_shot":
                    raise ValueError("inference.save_latents is only supported for seg_ar / chunk_ar")
                os.environ["HY_INFER_SAVE_FINAL_LATENTS_PATH"] = os.path.splitext(output_path)[0] + ".npy"

            # Run inference
            if rollout_type == "one_shot":
                output = pipeline.inference(
                    prompts=prompts,
                    num_frames=entry_num_frames,
                    height=args.inference.video_size[0],
                    width=args.inference.video_size[1],
                    guidance_scale=args.inference.cfg_scale_video,
                    camera_guidance_scale=args.inference.camera_cfg_scale,
                    num_inference_steps=args.inference.infer_steps_video,
                    flow_shift=args.inference.infer_flow_shift_video,
                    seed=seed,
                    negative_prompt=negative_prompt,
                    prompt_video_boundaries=entry_boundaries,
                    use_dynamic_cfg_scale=args.inference.use_dynamic_cfg_scale,
                    cfg_scale_min=args.inference.cfg_scale_min,
                    cfg_scale_max=args.inference.cfg_scale_max,
                    cfg_scale_max_step=args.inference.cfg_scale_max_step,
                    cfg_schedule=getattr(args.inference, "cfg_schedule", None),
                    cfg_skip_steps=getattr(args.inference, "cfg_skip_steps", 0),
                    camera_trajectory=effective_camera_trajectory,
                    segment_text_drops=segment_text_drops,
                    segment_has_camera=segment_has_camera,
                )
            else:
                output = pipeline.inference(
                    prompts=prompts,
                    num_frames=entry_num_frames,
                    height=args.inference.video_size[0],
                    width=args.inference.video_size[1],
                    guidance_scale=args.inference.cfg_scale_video,
                    camera_guidance_scale=args.inference.camera_cfg_scale,
                    num_inference_steps=args.inference.infer_steps_video,
                    flow_shift=args.inference.infer_flow_shift_video,
                    seed=seed,
                    negative_prompt=negative_prompt,
                    segment_frame_boundaries=entry_boundaries,
                    camera_trajectory=effective_camera_trajectory,
                    segment_has_camera=segment_has_camera,
                    cfg_schedule=getattr(args.inference, "cfg_schedule", None),
                    cfg_skip_steps=getattr(args.inference, "cfg_skip_steps", 0),
                )

            with stage("output_to_cpu"):
                video_np = video_to_uint8_numpy(
                    output[0], chunk_frames=args.inference.output_chunk_frames,
                ) if rank_within_sp == 0 else None
                del output
        rank_metrics = dict(sample_metrics.summary(), rank=rank,
                            execution_mode="inference_mode" if inference_mode_enabled else "no_grad",
                            model_storage_bytes={
                                "dit": tensor_storage_bytes(pipeline.transformer),
                                "text_encoder": tensor_storage_bytes(pipeline.text_encoder),
                                "vae": tensor_storage_bytes(pipeline.vae),
                            })
        if compile_before is not None:
            rank_metrics["compilation"] = compilation_delta(compile_before)
        if model_parallel_size > 1:
            model_group, _, _ = get_model_parallel_info()
            all_rank_metrics = [None] * model_parallel_size
            dist.all_gather_object(all_rank_metrics, rank_metrics, group=model_group)
        else:
            all_rank_metrics = [rank_metrics]
        Path(output_path).with_suffix(f".metrics.rank{rank}.json").write_text(
            json.dumps(rank_metrics, indent=2) + "\n"
        )
        if sample_metrics_path.exists():
            shutil.copyfile(sample_metrics_path, Path(output_path).with_suffix(f".stages.rank{rank}.jsonl"))

        # Record actual selected-kernel calls before discarding sample metadata.
        if args.dit.seg_ar_attention_backend == "flex":
            from hymm.models.dit.modules.attention.range_attention import range_stats, clear_range_cache
            backend_audit = range_stats()
            backend_audit.update(requested_backend="flex", rank=rank)
            audit_path = Path(output_path).with_suffix(f".attention.rank{rank}.json")
            audit_path.write_text(json.dumps(backend_audit, indent=2) + "\n")
            clear_range_cache()

        kv_audit = getattr(pipeline, "_last_kv_optimization_audit", None)
        if kv_audit and (kv_audit["int8_kv"] or kv_audit.get("qvg_kv", False) or kv_audit["swa"]):
            Path(output_path).with_suffix(f".kv.rank{rank}.json").write_text(
                json.dumps(dict(kv_audit, rank=rank), indent=2) + "\n"
            )

        from configs.cpu_offload import enabled_cpu_offloads
        if enabled_cpu_offloads(args):
            from hymm.utils.model_cpu_offload import cpu_offload_snapshot
            residency = cpu_offload_snapshot(pipeline)
            residency.update(rank=rank, sp_size=sp_size, tp_size=tp_size)
            Path(output_path).with_suffix(f".offload.rank{rank}.json").write_text(
                json.dumps(residency, indent=2) + "\n"
            )

        if rank_within_sp == 0:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            generated_video_frames = len(video_np)
            if saved_video_frame_offset:
                if generated_video_frames != entry_num_frames:
                    raise ValueError(
                        "Cannot trim turn 1: decoded frame count differs from generation timeline"
                    )
                video_np = video_np[saved_video_frame_offset:]
                logger.info(
                    f"Output-only first-turn trim: {generated_video_frames} -> "
                    f"{len(video_np)} frames; removed {saved_video_frame_offset}; "
                    "all turns were generated with unchanged KV history."
                )

            sample_performance = {
                "video": Path(output_path).name, "global_idx": global_entry_idx,
                "prompt_sha256": hashlib.sha256(entry.encode()).hexdigest(),
                "sample_sequence": sample_sequence,
                "seed": seed, "num_frames": len(video_np), "video_fps": args.inference.video_fps,
                "height": int(video_np.shape[1]), "width": int(video_np.shape[2]),
                "total_seconds": max(row["total_seconds"] for row in all_rank_metrics),
                "peak_allocated_gib": max(row["peak_allocated_gib"] for row in all_rank_metrics),
                "peak_reserved_gib": max(row["peak_reserved_gib"] for row in all_rank_metrics),
                "timings": rank_metrics["timings"], "ranks": all_rank_metrics,
                "measurement_scope": "pipeline inference through CPU handoff; excludes model loading and MP4/overlay encoding",
                "execution_mode": "inference_mode" if inference_mode_enabled else "no_grad",
                "output_chunk_frames": args.inference.output_chunk_frames,
                "metrics_note": "Nested stages overlap; do not sum their peaks or times. Device-used peaks sample boundaries only.",
            }
            if compile_before is not None:
                sample_performance["compiler_invoked"] = any(
                    row["compilation"]["compiler_invoked"] for row in all_rank_metrics
                )
            sample_performance["generated_fps"] = len(video_np) / sample_performance["total_seconds"]
            sample_performance["duration_seconds"] = len(video_np) / args.inference.video_fps
            sample_sequence += 1

            import imageio
            import tempfile

            # Write to local /tmp first, then copy to shared storage to avoid I/O errors
            tmp_fd, tmp_video_path = tempfile.mkstemp(suffix=".mp4", dir="/tmp")
            os.close(tmp_fd)
            try:
                imageio.mimwrite(
                    tmp_video_path,
                    video_np,
                    fps=args.inference.video_fps,
                    codec="libx264",
                    pixelformat="yuv420p",
                    # Put the MP4 index before media for progressive web playback.
                    output_params=["-movflags", "+faststart"],
                )
                # Do not copy local metadata to object-backed output mounts.
                shutil.copyfile(tmp_video_path, output_path)
            finally:
                if os.path.exists(tmp_video_path):
                    os.remove(tmp_video_path)

            logger.info(f"Video saved to: {output_path}")
            logger.info(
                f"  Frames: {video_np.shape[0]}, Duration: {video_np.shape[0] / args.inference.video_fps:.2f}s"
            )
            if skip_first_turn:
                trim_path = Path(output_path).with_suffix(".timeline.json")
                trim_path.write_text(
                    json.dumps(
                        {
                            "skip_first_turn": True,
                            "mode": "output_only",
                            "generated_frames": generated_video_frames,
                            "saved_frames": len(video_np),
                            "saved_video_frame_offset": saved_video_frame_offset,
                            "generation_frame_boundaries": entry_boundaries,
                            "saved_frame_boundaries": [
                                b - saved_video_frame_offset
                                for b in entry_boundaries[1:]
                            ],
                            "first_turn_generated": True,
                        },
                        indent=2,
                    )
                    + "\n"
                )

            # Optional: overlay WASD + arrow-key indicators on the rendered
            # video for camera-bench cases.
            #
            # Default behavior:
            #   - When the suite supplies a per-entry NPZ (i.e. this is a
            #     camera control bench), overlay is ON automatically and the
            #     overlaid mp4 **replaces** the plain mp4 in-place (saving
            #     disk space). Set `inference.overlay_keys_keep_original:
            #     true` to keep both files (`<base>.mp4` + `<base>_keys.mp4`).
            #   - When only a global `inference.camera_trajectory` is set
            #     (no per-entry NPZ), overlay stays OFF unless explicitly
            #     enabled by `inference.overlay_keys: true` in the YAML.
            # Set `inference.overlay_keys: false` to force-disable.
            #
            # The NPZ produced by `tools_tmp/build_camera_bench_keyboard.py`
            # carries ground-truth `gt_keys`/`gt_mouse`, so the overlay
            # reflects exactly what the user asked the model to follow;
            # otherwise the keys are inferred from c2w deltas (4T4R).
            overlay_keys_setting = getattr(args.inference, "overlay_keys", None)
            if overlay_keys_setting is None:
                # Auto: enable iff this entry came with a per-case NPZ.
                overlay_keys_enabled = bool(entry_camera_trajectory)
            else:
                overlay_keys_enabled = bool(overlay_keys_setting)
            overlay_keep_original = bool(
                getattr(args.inference, "overlay_keys_keep_original", False)
            )
            overlay_npz = effective_camera_trajectory if overlay_keys_enabled else None
            if overlay_keys_enabled and overlay_npz and os.path.isfile(overlay_npz):
                try:
                    from hymm.models.camera.geometry.overlay_camera_keys_on_video import (
                        overlay_one_video,
                    )

                    base, ext = os.path.splitext(output_path)
                    if overlay_keep_original:
                        # Side-by-side: original keeps `output_path`, overlay
                        # is written to `<base><suffix><ext>`.
                        overlay_suffix = getattr(
                            args.inference, "overlay_keys_suffix", "_keys"
                        )
                        overlay_output_path = f"{base}{overlay_suffix}{ext}"
                    else:
                        # In-place replacement: overlay overwrites the plain mp4.
                        overlay_output_path = output_path

                    tmp_fd, tmp_overlay_path = tempfile.mkstemp(suffix=ext, dir="/tmp")
                    os.close(tmp_fd)
                    try:
                        stats = overlay_one_video(
                            video_path=output_path,
                            npz_path=overlay_npz,
                            output_path=tmp_overlay_path,
                            smooth_window=getattr(
                                args.inference, "overlay_keys_smooth_window", 5
                            ),
                            key_size=getattr(args.inference, "overlay_keys_size", 36),
                            h264_quality=getattr(
                                args.inference, "overlay_keys_quality", 8
                            ),
                            abs_threshold=getattr(
                                args.inference, "overlay_keys_abs_threshold", None
                            ),
                            segment_frame_boundaries=entry_boundaries,
                            segment_has_camera=overlay_segment_has_camera,
                            video_frame_offset=saved_video_frame_offset,
                        )
                        # Copy content only: object-backed mounts may reject
                        # chmod/utime even when writing file contents succeeds.
                        shutil.copyfile(tmp_overlay_path, overlay_output_path)
                        if overlay_keep_original:
                            logger.info(
                                f"  Overlay saved to: {overlay_output_path}  "
                                f"(src={stats['key_source']}, "
                                f"n_video={stats['n_video']}, n_traj={stats['n_traj']})"
                            )
                        else:
                            logger.info(
                                f"  Overlay applied in-place: {output_path}  "
                                f"(src={stats['key_source']}, "
                                f"n_video={stats['n_video']}, n_traj={stats['n_traj']})"
                            )
                    finally:
                        if os.path.exists(tmp_overlay_path):
                            os.remove(tmp_overlay_path)
                except Exception as overlay_err:
                    # Never let overlay failures break inference.
                    logger.warning(
                        f"  Overlay generation failed (non-fatal): {overlay_err}"
                    )
            elif overlay_keys_enabled and not overlay_npz:
                logger.info(
                    "  Overlay skipped: no camera_trajectory NPZ available for this entry."
                )
            elif (
                overlay_keys_enabled and overlay_npz and not os.path.isfile(overlay_npz)
            ):
                logger.warning(f"  Overlay skipped: NPZ does not exist: {overlay_npz}")

            # Save prompt text alongside the video
            if args.inference.save_prompt_txt:
                prompt_txt_path = os.path.splitext(output_path)[0] + ".txt"
                tmp_fd, tmp_txt_path = tempfile.mkstemp(suffix=".txt", dir="/tmp")
                os.close(tmp_fd)
                try:
                    with open(tmp_txt_path, "w", encoding="utf-8") as pf:
                        # Write the EXACT string that was fed to the text
                        # encoder — no bench-key restoration, no re-formatting.
                        # This guarantees the on-disk .txt is a faithful record
                        # of the model's textual input (per ``|||`` segment).
                        pf.write(entry)
                    shutil.copyfile(tmp_txt_path, prompt_txt_path)
                finally:
                    if os.path.exists(tmp_txt_path):
                        os.remove(tmp_txt_path)
                logger.info(f"  Prompt saved to: {prompt_txt_path}")
            sample_performance["video_bytes"] = Path(output_path).stat().st_size
            Path(output_path).with_suffix(".performance.json").write_text(
                json.dumps(sample_performance, indent=2) + "\n"
            )
        del video_np

    if skipped_count > 0:
        logger.info(
            f"Skipped {skipped_count}/{len(indexed_entries)} entries (already generated)"
        )

    # ------------------------------------------------------------------
    # Per-shard completion marker (cross-node barrier for the monitor).
    # ------------------------------------------------------------------
    # Without this, the launcher has no way to
    # tell a "still running" node from a "finished its slice" node. The
    # monitor would otherwise count *.mp4 files against the global expected
    # count (e.g. 150) — but each SP group only owns a fraction (~150/N), so
    # `mp4_count >= 150` is never satisfied locally and either:
    #   * the monitor keeps re-launching inference for the same step on this
    #     node, or
    #   * a fast-finishing node advances to the next step and starts writing
    #     into iter_<K+1>/ while a slow node is still finishing iter_<K>/.
    # By dropping a `.shard_<sp_group_id>.done` file in the iter_<step>
    # output dir, the monitor can wait until ALL shards are done before
    # advancing — and when checking "is this step inferred", it counts done
    # files (cheap, deterministic) rather than mp4s (subject to per-shard
    # accounting).
    if rank_within_sp == 0:
        try:
            save_path_str = str(args.inference.save_path)
            done_marker = os.path.join(
                save_path_str, f".shard_{sp_group_id:04d}_of_{num_sp_groups:04d}.done"
            )
            os.makedirs(save_path_str, exist_ok=True)
            with open(done_marker, "w") as f:
                f.write(
                    f"sp_group_id={sp_group_id}\n"
                    f"num_sp_groups={num_sp_groups}\n"
                    f"entries_processed={len(indexed_entries)}\n"
                    f"skipped={skipped_count}\n"
                    f"rollout_type={rollout_type}\n"
                    f"timestamp={datetime.datetime.now().isoformat()}\n"
                )
            logger.info(f"Wrote shard-done marker: {done_marker}")
        except Exception as e:
            # Never let marker-writing failures break inference.
            logger.warning(f"Failed to write shard-done marker (non-fatal): {e}")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

    logger.info(
        f"Inference completed! (rollout_type={rollout_type}, {len(indexed_entries)} entries on this SP group)"
    )


if __name__ == "__main__":
    main()
