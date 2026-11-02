"""Temporal frame sampling with a CPU or opt-in GPU KV archive.

The GPU working set contains at most ``past_frames`` video frames plus one
text block per represented segment. Already-rotated keys keep their original
positions. INT8 mode quantizes each completed segment once and archives its
payload and FP32 scales together; sampling never requantizes old entries. The
full archive lets future grids reselect frames absent from the GPU working set.
"""

from dataclasses import dataclass
import math
import os
from numbers import Real

from hymm.utils.inference_metrics import profiled


ARCHIVED_HISTORY_SAMPLINGS = ("uniform", "recent_mix_uniform", "temporal_context")
HISTORY_SAMPLINGS = ("recent", *ARCHIVED_HISTORY_SAMPLINGS)
_GPU_REQUEST_CACHES = {}


def release_gpu_request_history():
    """Release payloads only after a caller has ended a nonpersistent request."""
    for cache in _GPU_REQUEST_CACHES.values():
        cache.pop("swa_archive", None)
        for key in ("k", "v", "k_scale", "v_scale"):
            cache.pop(key, None)
    _GPU_REQUEST_CACHES.clear()


class UnsupportedStreamingRollbackError(ValueError):
    """A rollback policy rejection that leaves the live session usable."""


def validate_swa_streaming_rollback(caches):
    """Check the active cache policy before mutating or broadcasting a rollback."""
    # Ordinary caches may inherit sampling=recent in config while SWA is off.
    # Actual recent SWA caches have swa_capacity but may omit swa_sampling.
    if any("swa_capacity" in cache and cache.get("swa_sampling", "recent") == "recent"
           for cache in caches or ()):
        raise UnsupportedStreamingRollbackError("recent模式SWA不支持streaming回退")


def _validate_frame_budget(total_frames, past_frames):
    if isinstance(total_frames, bool) or not isinstance(total_frames, int) or total_frames < 0:
        raise ValueError("total_frames must be a nonnegative integer")
    if isinstance(past_frames, bool) or not isinstance(past_frames, int) or past_frames < 1:
        raise ValueError("past_frames must be a positive integer")


def uniform_frame_indices(total_frames, past_frames):
    """Sorted, endpoint-inclusive integer grid over the complete history."""
    _validate_frame_budget(total_frames, past_frames)
    count = min(total_frames, past_frames)
    if count == 0:
        return ()
    if count == 1:
        return (total_frames - 1,)
    return tuple(i * (total_frames - 1) // (count - 1) for i in range(count))


def recent_mix_uniform_frame_indices(total_frames, past_frames):
    """Keep K//2 contiguous recent frames and uniformly sample the older prefix.

    Histories shorter than the video-frame budget are kept completely. Returned
    indices are zero-based, chronological and never include a frame twice.
    """
    _validate_frame_budget(total_frames, past_frames)
    if total_frames <= past_frames:
        return tuple(range(total_frames))
    recent = past_frames // 2
    older_end = total_frames - recent
    historical = uniform_frame_indices(older_end, past_frames - recent)
    return historical + tuple(range(older_end, total_frames))


def temporal_context_frame_indices(total_frames, past_frames, gamma=2.0):
    """Power-spaced distances from the newest frame, returned oldest first.

    N=min(past_frames, T), u=i/max(N-1, 1), and
    d_i=i+rint((T-N)*u**gamma). Original frame positions are T-1-d_i.
    For finite gamma>0 the rounded term is nondecreasing, so d grows by
    at least one at every step. Mirroring and reversing preserves uniqueness.
    gamma>1 biases selection toward recent frames; gamma=1 is evenly spaced.
    """
    _validate_frame_budget(total_frames, past_frames)
    if (isinstance(gamma, bool) or not isinstance(gamma, Real)
            or not math.isfinite(gamma) or gamma <= 0):
        raise ValueError("temporal_context gamma must be a finite positive number")
    count = min(total_frames, past_frames)
    if count == 0:
        return ()
    if count == 1:
        return (total_frames - 1,)
    if count == total_frames:
        return tuple(range(total_frames))

    import numpy as np

    i = np.arange(count)
    u = i / (count - 1)
    slack = total_frames - count
    rounded = np.rint(slack * u**gamma)
    # Keep the mathematical bounds/endpoints exact even when T exceeds float64
    # integer precision. Clipping is monotone and does not remove/repeat indices.
    offsets = (0, *(min(slack, max(0, int(value))) for value in rounded[1:-1]), slack)
    # Add/mirror as Python integers to avoid int64 overflow in large histories.
    distances = tuple(index + offset for index, offset in enumerate(offsets))
    return tuple(total_frames - 1 - distance for distance in reversed(distances))


def history_frame_indices(total_frames, past_frames, sampling="uniform", *, gamma=2.0):
    """Select and check original frame positions before rebuilding history KV."""
    if sampling == "uniform":
        frame_ids = uniform_frame_indices(total_frames, past_frames)
    elif sampling == "recent_mix_uniform":
        frame_ids = recent_mix_uniform_frame_indices(total_frames, past_frames)
    elif sampling == "temporal_context":
        frame_ids = temporal_context_frame_indices(total_frames, past_frames, gamma)
    else:
        raise ValueError(f"Unknown archived history sampling: {sampling!r}")

    expected = min(total_frames, past_frames)
    if len(frame_ids) != expected:
        raise RuntimeError(
            f"History sampling {sampling!r} selected {len(frame_ids)} frames; expected {expected}"
        )
    if any(type(frame) is not int or not 0 <= frame < total_frames for frame in frame_ids):
        raise RuntimeError(f"History sampling {sampling!r} produced invalid frame indices: {frame_ids}")
    if len(set(frame_ids)) != len(frame_ids):
        raise RuntimeError(f"History sampling {sampling!r} selected duplicate frames: {frame_ids}")
    if any(left >= right for left, right in zip(frame_ids, frame_ids[1:])):
        raise RuntimeError(f"History sampling {sampling!r} must return chronological frame indices")
    return frame_ids


@dataclass(frozen=True)
class ArchivedSegment:
    segment_index: int
    frame_start: int
    frame_count: int
    image_tokens: int
    text_tokens: int
    action_start: int
    k: object
    v: object
    mask: object
    k_scale: object = None
    v_scale: object = None
    qvg_payload: object = None


@dataclass(frozen=True)
class HistoryRollback:
    records: list[ArchivedSegment]
    frame_ids: tuple[int, ...]
    tokens_per_frame: int
    total_frames: int
    valid_tokens: int
    archive_bytes: int
    removed_tokens: int


def prepare_history_rollback(cache, mask, *, segment_index, latent_frames, total_frames):
    """Validate one streaming layer without mutating tensors or its archive."""
    import torch

    if (cache.get('storage_format') == 'qvg'
            or cache.get('swa_sampling') not in ARCHIVED_HISTORY_SAMPLINGS):
        raise ValueError("Streaming history rollback requires floating-point or INT8 archived SWA")
    records = cache.get('swa_archive', [])
    if (len(records) != segment_index + 1 or not records
            or records[-1].frame_count != latent_frames
            or cache.get('swa_total_frames') != total_frames):
        raise RuntimeError("Streaming SWA archive does not match the segment to roll back")
    keys = ('k', 'v', 'k_scale', 'v_scale') if cache['k'].dtype == torch.int8 else ('k', 'v')
    stored = cache['k']
    if (mask.shape != stored.shape[:2] or mask.dtype != torch.bool
            or mask.device != stored.device):
        raise RuntimeError("Streaming SWA mask does not match KV storage")
    for key in keys:
        tensor = cache[key]
        expected_shape = (*stored.shape[:-1], 1) if key.endswith('_scale') else stored.shape
        expected_dtype = torch.float32 if key.endswith('_scale') else stored.dtype
        if (tensor.shape != expected_shape or tensor.dtype != expected_dtype
                or tensor.device != stored.device):
            raise RuntimeError("Streaming SWA KV/scale storage is inconsistent")
    cursor = 0
    archive_bytes = 0
    tokens_per_frame = records[-1].image_tokens // latent_frames
    for index, record in enumerate(records):
        length = record.image_tokens + record.text_tokens
        if (record.segment_index != index or record.frame_start != cursor
                or record.frame_count <= 0 or tokens_per_frame <= 0
                or record.image_tokens != record.frame_count * tokens_per_frame
                or record.text_tokens < 0 or not -1 <= record.action_start <= record.text_tokens):
            raise RuntimeError("Streaming SWA archive frame geometry is inconsistent")
        if (record.mask.shape != (stored.shape[0], length)
                or record.mask.dtype != torch.bool or record.mask.device.type != 'cpu'):
            raise RuntimeError("Streaming SWA archived mask is inconsistent")
        record_bytes = record.mask.numel() * record.mask.element_size()
        for key in keys:
            tensor = getattr(record, key)
            if (not isinstance(tensor, torch.Tensor)
                    or tensor.shape != (stored.shape[0], length, *cache[key].shape[2:])
                    or tensor.dtype != cache[key].dtype or tensor.device.type != 'cpu'):
                raise RuntimeError("Streaming SWA archived KV/scale storage is inconsistent")
            record_bytes += tensor.numel() * tensor.element_size()
        if index < segment_index:
            archive_bytes += record_bytes
        cursor += record.frame_count
    if cursor != total_frames:
        raise RuntimeError("Streaming SWA archive does not cover its frame timeline")
    remaining = records[:-1]
    frame_ids = history_frame_indices(
        total_frames - latent_frames, cache['swa_past_frames'], cache['swa_sampling'],
        gamma=cache.get('swa_gamma', 2.0),
    )
    valid_tokens = len(frame_ids) * tokens_per_frame + sum(
        record.text_tokens for record in remaining
        if any(record.frame_start <= frame < record.frame_start + record.frame_count for frame in frame_ids)
    )
    if valid_tokens > min(cache['swa_capacity'], stored.shape[1]):
        raise RuntimeError("Streaming SWA rollback exceeds the existing history capacity")
    return HistoryRollback(
        remaining, frame_ids, tokens_per_frame, total_frames - latent_frames,
        valid_tokens, archive_bytes, records[-1].image_tokens + records[-1].text_tokens,
    )


@profiled("kv_rebuild")
def apply_history_rollback(cache, mask, rollback):
    """Restore prevalidated streaming history without re-encoding old KV."""
    _rebuild_history(cache, mask, rollback.records, rollback.frame_ids, rollback.tokens_per_frame)
    cache['swa_archive'] = rollback.records
    cache['swa_total_frames'] = rollback.total_frames
    cache['swa_archive_bytes'] = rollback.archive_bytes
    cache.pop('swa_segment_index', None)
    cache.pop('swa_action_start', None)


@profiled("kv_rebuild")
def commit_uniform_history(cache, mask, current_k, current_v, current_mask,
                           image_tokens, text_tokens, tokens_per_frame):
    """Archive the completed segment, then rebuild the GPU history exactly.

    Called only after attention has consumed the previous working set and
    the complete current segment. All CUDA/CPU copies complete before return.
    Layout tuples are (segment, video_start, video_end, text_start, text_end,
    action_start), in the packed working set's token coordinates.
    """
    import torch

    if cache.get("storage_format") == "qvg":
        from .qvg_kv_cache import commit_qvg_history

        return commit_qvg_history(
            cache, mask, current_k, current_v, current_mask,
            image_tokens, text_tokens, tokens_per_frame,
        )
    if tokens_per_frame <= 0 or image_tokens <= 0 or image_tokens % tokens_per_frame:
        raise ValueError("Uniform history requires complete, nonempty video frames")
    if text_tokens < 0 or current_k.shape[1] != image_tokens + text_tokens:
        raise ValueError("Uniform history video/text token counts disagree")
    if current_k.shape != current_v.shape or current_mask.shape != current_k.shape[:2]:
        raise ValueError("Uniform history K/V/mask shapes disagree")
    stored_k, stored_v = cache['k'], cache['v']
    if (stored_k.ndim != 4 or stored_k.shape != stored_v.shape
            or stored_k.dtype != stored_v.dtype or stored_k.device != stored_v.device):
        raise ValueError("Uniform history storage K/V shapes, dtypes and devices must match")
    if (current_k.ndim != 4 or current_k.shape[0] != stored_k.shape[0]
            or current_k.shape[2:] != stored_k.shape[2:]
            or current_k.device != stored_k.device or current_v.device != stored_v.device
            or mask.shape != stored_k.shape[:2] or mask.device != stored_k.device
            or mask.dtype != torch.bool):
        raise ValueError("Uniform history input/storage shapes or devices disagree")
    int8_storage = stored_k.dtype == torch.int8
    storage_keys = ('k', 'v', 'k_scale', 'v_scale') if int8_storage else ('k', 'v')
    if int8_storage:
        if current_k.dtype != torch.bfloat16 or current_v.dtype != torch.bfloat16:
            raise ValueError("Uniform INT8 history requires BF16 current K/V")
        for key in ('k_scale', 'v_scale'):
            scale = cache.get(key)
            if (not isinstance(scale, torch.Tensor) or scale.dtype != torch.float32
                    or scale.shape != (*stored_k.shape[:-1], 1)
                    or scale.device != stored_k.device):
                raise ValueError("Uniform INT8 history requires matching FP32 K/V scales")
    elif stored_k.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("Uniform history requires floating-point or INT8 KV storage")
    elif not (stored_k.dtype == current_k.dtype == current_v.dtype):
        raise ValueError("Uniform history must preserve the KV dtype")
    past_frames = cache['swa_past_frames']
    history = cache.get('swa_archive', [])
    frame_start = cache.get('swa_total_frames', 0)
    frame_count = image_tokens // tokens_per_frame
    frame_ids = history_frame_indices(
        frame_start + frame_count, past_frames, cache.get('swa_sampling', 'uniform'),
        gamma=cache.get('swa_gamma', 2.0),
    )
    segment_index = cache['swa_segment_index']
    if history and segment_index <= history[-1].segment_index:
        raise ValueError("Uniform history requires one chronological write per segment")
    action_start = int(cache.get('swa_action_start', -1))
    if not -1 <= action_start <= text_tokens:
        raise ValueError("Uniform history action offset is outside the text block")
    if int8_storage:
        from .int8_kv_cache import make_int8_kv_block, quantize_kv_pair_into

        # Quantize on the current device before archiving, after current-segment
        # attention. Keep the original payload/scales for every later selection.
        segment = make_int8_kv_block(
            current_k.shape[0], current_k.shape[1], current_k.shape[2], current_k.shape[3],
            device=current_k.device,
        )
        quantize_kv_pair_into(segment, current_k.detach(), current_v.detach(), 0)
    else:
        segment = {'k': current_k.detach(), 'v': current_v.detach()}
    from hymm.utils.inference_metrics import stage
    from .kv_cache_storage import storage_bytes

    fast_history = os.environ.get("HY_KV_FAST_HISTORY", "0") == "1"
    gpu_archive = os.environ.get("HY_KV_ARCHIVE_DEVICE", "cpu") == "cuda"
    if gpu_archive and not current_k.is_cuda:
        raise ValueError("GPU KV archive requires CUDA current K/V")
    if gpu_archive:
        _GPU_REQUEST_CACHES[id(cache)] = cache
        with stage("kv_gpu_archive"):
            # Own storage: incoming K/V may alias a reusable attention buffer.
            archived = {key: segment[key].clone(memory_format=torch.contiguous_format)
                        for key in storage_keys}
    else:
        with stage("kv_d2h"):
            if fast_history and current_k.is_cuda:
                archived = {}
                for key in storage_keys:
                    source = segment[key]
                    target = torch.empty(source.shape, dtype=source.dtype,
                                         device="cpu", pin_memory=True)
                    target.copy_(source, non_blocking=True)
                    archived[key] = target
                torch.cuda.current_stream(current_k.device).synchronize()
            else:
                archived = {
                    key: segment[key].to(device='cpu', copy=True, non_blocking=False).contiguous()
                    for key in storage_keys
                }
    record = ArchivedSegment(
        segment_index, frame_start, frame_count, image_tokens, text_tokens,
        action_start,
        mask=current_mask.detach().to(
            device=current_k.device if gpu_archive else 'cpu',
            dtype=torch.bool, copy=True, non_blocking=False,
        ).contiguous(),
        **archived,
    )
    records = [*history, record]
    if (fast_history or gpu_archive) and current_k.is_cuda:
        _rebuild_history_fast(cache, mask, records, frame_ids, tokens_per_frame,
                              segment_index, segment, current_mask)
    else:
        _rebuild_history(cache, mask, records, frame_ids, tokens_per_frame)
    del segment
    cache['swa_archive'] = records
    cache['swa_total_frames'] = frame_start + frame_count
    cache['swa_archive_bytes'] = cache.get('swa_archive_bytes', 0) + sum(
        tensor.numel() * tensor.element_size()
        for tensor in (*archived.values(), record.mask)
    )
    stats = cache.setdefault("kv_codec_stats", {})
    stats["encode_calls"] = stats.get("encode_calls", 0) + int(int8_storage)
    stats["d2h_bytes"] = stats.get("d2h_bytes", 0) + (0 if gpu_archive else storage_bytes(archived))
    cache["swa_archive_device"] = "cuda" if gpu_archive else "cpu"


def _rebuild_history(cache, mask, records, frame_ids, tokens_per_frame):
    """Share the exact history packing order between commits and rollbacks."""
    import torch
    from hymm.utils.inference_metrics import stage

    storage_keys = ('k', 'v', 'k_scale', 'v_scale') if cache['k'].dtype == torch.int8 else ('k', 'v')
    packed = {key: [] for key in storage_keys}
    packed_mask, layout = [], []
    cursor = 0
    for old in records:
        selected = [f - old.frame_start for f in frame_ids
                    if old.frame_start <= f < old.frame_start + old.frame_count]
        if not selected:
            continue
        indices = (torch.tensor(selected, dtype=torch.long)[:, None] * tokens_per_frame
                   + torch.arange(tokens_per_frame)[None, :]).flatten()
        if old.text_tokens:
            indices = torch.cat([indices, torch.arange(old.image_tokens,
                                                       old.image_tokens + old.text_tokens)])
        video_end = cursor + len(selected) * tokens_per_frame
        text_end = video_end + old.text_tokens
        layout.append((old.segment_index, cursor, video_end, video_end,
                       text_end, old.action_start))
        indices = indices.to(old.k.device)
        with stage("kv_gpu_pack" if old.k.is_cuda else "kv_cpu_pack"):
            for key in storage_keys:
                packed[key].append(getattr(old, key).index_select(1, indices))
            packed_mask.append(old.mask.index_select(1, indices))
        cursor = text_end
    if cursor > cache['swa_capacity'] or cursor > mask.shape[1]:
        raise ValueError(f"Uniform history needs {cursor} tokens; GPU capacity is {cache['swa_capacity']}")
    gpu_archive = bool(records) and records[0].k.is_cuda
    with stage("kv_gpu_pack" if gpu_archive else "kv_h2d"):
        if cursor:
            for key in storage_keys:
                cache[key][:, :cursor].copy_(torch.cat(packed[key], dim=1), non_blocking=False)
    if cursor:
        mask[:, :cursor].copy_(torch.cat(packed_mask, dim=1), non_blocking=False)
    mask[:, cursor:] = False
    mask._action_tail_ranges = []
    cache['swa_frame_ids'] = frame_ids
    cache['swa_uniform_layout'] = tuple(layout)
    cache['swa_valid_len'] = cursor
    stats = cache.setdefault("kv_codec_stats", {})
    stats["h2d_bytes"] = stats.get("h2d_bytes", 0) + (0 if gpu_archive else sum(
        tensor.numel() * tensor.element_size() for tensors in packed.values() for tensor in tensors
    ))


def _rebuild_history_fast(cache, mask, records, frame_ids, tokens_per_frame,
                          current_segment, current_payload, current_mask):
    """Pack directly into GPU history; current-segment selection never leaves CUDA."""
    import torch
    from hymm.utils.inference_metrics import stage

    storage_keys = ('k', 'v', 'k_scale', 'v_scale') if cache['k'].dtype == torch.int8 else ('k', 'v')
    cursor, layout, transferred = 0, [], 0
    pending = []
    for old in records:
        selected = [f - old.frame_start for f in frame_ids
                    if old.frame_start <= f < old.frame_start + old.frame_count]
        if not selected:
            continue
        indices = (torch.tensor(selected, dtype=torch.long)[:, None] * tokens_per_frame
                   + torch.arange(tokens_per_frame)[None, :]).flatten()
        if old.text_tokens:
            indices = torch.cat([indices, torch.arange(old.image_tokens,
                                                       old.image_tokens + old.text_tokens)])
        video_end = cursor + len(selected) * tokens_per_frame
        text_end = video_end + old.text_tokens
        if text_end > cache['swa_capacity'] or text_end > mask.shape[1]:
            raise ValueError("Selected history exceeds GPU working-set capacity")
        layout.append((old.segment_index, cursor, video_end, video_end,
                       text_end, old.action_start))
        if old.segment_index == current_segment or old.k.is_cuda:
            with stage("kv_gpu_pack"):
                gpu_indices = indices.to(cache['k'].device)
                payload = (current_payload if old.segment_index == current_segment else
                           {key: getattr(old, key) for key in storage_keys})
                source_mask = current_mask if old.segment_index == current_segment else old.mask
                for key in storage_keys:
                    torch.index_select(payload[key], 1, gpu_indices,
                                       out=cache[key][:, cursor:text_end])
                torch.index_select(source_mask, 1, gpu_indices,
                                   out=mask[:, cursor:text_end])
        else:
            with stage("kv_cpu_pack"):
                gathered = {}
                for key in storage_keys:
                    source = getattr(old, key)
                    target = torch.empty((source.shape[0], len(indices), *source.shape[2:]),
                                         dtype=source.dtype, device="cpu", pin_memory=True)
                    torch.index_select(source, 1, indices, out=target)
                    gathered[key] = target
                selected_mask = old.mask.index_select(1, indices).pin_memory()
            with stage("kv_h2d"):
                for key, source in gathered.items():
                    cache[key][:, cursor:text_end].copy_(source, non_blocking=True)
                    transferred += source.numel() * source.element_size()
                mask[:, cursor:text_end].copy_(selected_mask, non_blocking=True)
            pending.append((gathered, selected_mask))
        cursor = text_end
    mask[:, cursor:] = False
    # Keep pinned sources alive and readable until the queued copies have completed.
    if pending:
        torch.cuda.current_stream(cache['k'].device).synchronize()
    mask._action_tail_ranges = []
    cache['swa_frame_ids'] = frame_ids
    cache['swa_uniform_layout'] = tuple(layout)
    cache['swa_valid_len'] = cursor
    stats = cache.setdefault("kv_codec_stats", {})
    stats["h2d_bytes"] = stats.get("h2d_bytes", 0) + transferred


def selected_action_tails(layout, mask):
    """Rebuild tail coordinates after noncontiguous frame selection."""
    return [
        (segment, text_start + action_start, text_end,
         mask[:, text_start + action_start:text_end].bool().clone())
        for segment, _, _, text_start, text_end, action_start in layout
        if action_start >= 0 and text_start + action_start < text_end
    ]
