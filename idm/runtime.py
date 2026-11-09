"""Hy-Nomas IDM model loading and video inference."""
from contextlib import nullcontext
import json
import math
from pathlib import Path

import numpy as np
import torch

from .motion_mapping import clip_camera_motion
from .postprocessing import DEFAULT_SHAKE_THRESHOLD, shake_decisions, validate_shake_threshold
from .postprocessing import DEFAULT_ROTATION_THRESHOLD, rotation_decisions, validate_rotation_threshold
from .vpt import VPTIDMWrapper, _make_vpt_policy, enable_bf16_compat

ROOT = Path(__file__).resolve().parents[1]


def load_model(weights=ROOT / 'weights/idm.pt', device='cuda:0'):
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; pass --device cpu explicitly.')
    checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
    if checkpoint.get('format') != 'hy-nomas-idm-v1':
        raise ValueError('Expected the exported Hy-Nomas IDM inference checkpoint.')
    architecture = checkpoint['architecture']
    policy = _make_vpt_policy(architecture['net'], architecture['pi_head'])
    model = VPTIDMWrapper(policy, n_label_dims=18, temporal_pool=6,
                          with_zoom_head=True, with_velocity_head=True,
                          with_motion_score_head=True)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.to(device).eval()
    enable_bf16_compat(model)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    metadata = dict(checkpoint['metadata'])
    metadata.pop('protocol', None)
    metadata['supported_sampling'] = ['windows24fps', 'uniform144']
    metadata['velocity_units'] = 'model-native regression values; not calibrated physical speed'
    return model, metadata


def read_video(path):
    import decord
    decord.bridge.set_bridge('native')
    reader = decord.VideoReader(str(path), width=128, height=96, num_threads=2)
    count = len(reader)
    fps = float(reader.get_avg_fps())
    if count < 1 or not np.isfinite(fps) or fps <= 0:
        raise ValueError(f'Invalid video length/FPS: {path}')
    indices = np.rint(np.linspace(0, count - 1, 144)).astype(np.int64)
    pixels = reader.get_batch(indices.tolist()).asnumpy()
    return pixels, {'source_frames': count, 'source_fps': fps,
                    'duration_seconds': count / fps,
                    'source_frame_indices': indices.tolist()}


@torch.inference_mode()
def predict_frames(model, pixels, *, precision='bf16'):
    pixels = np.asarray(pixels)
    if pixels.shape != (144, 96, 128, 3) or pixels.dtype != np.uint8:
        raise ValueError('Expected RGB uint8 [144,96,128,3].')
    if precision not in ('bf16', 'fp32'):
        raise ValueError('precision must be bf16 or fp32')
    device = next(model.parameters()).device
    if precision == 'bf16' and device.type != 'cuda':
        raise ValueError('Use --precision fp32 with --device cpu.')
    frames = torch.from_numpy(pixels.copy()).permute(0, 3, 1, 2).unsqueeze(0).to(device)
    context = torch.autocast('cuda', dtype=torch.bfloat16) if precision == 'bf16' else nullcontext()
    with context:
        h = model.pool(frames)
        # Apply sigmoid in the autocast dtype before converting probabilities to float32.
        motion = model.head(h).sigmoid().float()[0].cpu().numpy()
        zoom = model.zoom_head(h).sigmoid().float()[0].cpu().numpy()
        velocity = model.velocity_head(h).float()[0].cpu().numpy()
        motion_score = model.motion_score_head(h.mean(1)).float().item()
    if not all(np.isfinite(x).all() for x in (motion, zoom, velocity, motion_score)):
        raise RuntimeError('Non-finite model output')
    return {'motion_prob': motion.tolist(), 'zoom_prob': zoom.tolist(),
            'velocity6': velocity.tolist(), 'motion_score': motion_score,
            'shake_prob': motion[:, 15].tolist(),
            'clip_camera_motion': clip_camera_motion(motion, zoom)}


def window_plan(frame_count, fps):
    """Nonoverlapping six-second windows sampled at 24 FPS, with a padded tail."""
    if frame_count < 1 or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid source frame count or FPS')
    duration = frame_count / fps
    for start in range(0, math.ceil(duration), 6):
        valid_duration = min(6., duration - start)
        valid_bins = math.ceil(valid_duration * 4 - 1e-8)
        if valid_bins <= 0:
            continue
        indices = np.clip(np.rint((start + np.arange(144) / 24) * fps),
                          0, frame_count - 1).astype(np.int64)
        yield {'start_seconds': float(start), 'valid_seconds': valid_duration,
               'valid_bins': valid_bins, 'source_frame_indices': indices.tolist()}


def infer_video(model, path, *, precision='bf16', sampling='windows',
                shake_threshold=DEFAULT_SHAKE_THRESHOLD,
                rotation_threshold=DEFAULT_ROTATION_THRESHOLD):
    shake_threshold = validate_shake_threshold(shake_threshold)
    rotation_threshold = validate_rotation_threshold(rotation_threshold)
    if sampling == 'uniform':
        pixels, info = read_video(path)
        outputs = predict_frames(model, pixels, precision=precision)
        return {'video': str(path), 'protocol': 'uniform144',
                'precision': precision, 'sampling': info,
                **outputs, **shake_decisions(outputs['shake_prob'], shake_threshold),
                **rotation_decisions([row[4:12] for row in outputs['motion_prob']], rotation_threshold)}
    if sampling != 'windows':
        raise ValueError('sampling must be windows or uniform')
    import decord
    decord.bridge.set_bridge('native')
    reader = decord.VideoReader(str(path), width=128, height=96, num_threads=2)
    count, fps = len(reader), float(reader.get_avg_fps())
    plans = list(window_plan(count, fps))
    motion, zoom, velocity, scores, durations, starts = [], [], [], [], [], []
    for plan in plans:
        pixels = reader.get_batch(plan['source_frame_indices']).asnumpy()
        outputs = predict_frames(model, pixels, precision=precision)
        n = plan['valid_bins']
        motion.extend(outputs['motion_prob'][:n])
        zoom.extend(outputs['zoom_prob'][:n])
        velocity.extend(outputs['velocity6'][:n])
        scores.append(outputs['motion_score'])
        durations.append(plan['valid_seconds'])
        starts.extend((plan['start_seconds'] + np.arange(n) / 4).tolist())
    return {'video': str(path), 'protocol': 'windows24fps', 'precision': precision,
            'sampling': {'source_frames': count, 'source_fps': fps,
                         'duration_seconds': count / fps, 'sample_fps': 24,
                         'window_seconds': 6, 'bucket_seconds': .25,
                         'bucket_start_seconds': starts, 'windows': plans},
            'motion_prob': motion, 'zoom_prob': zoom, 'velocity6': velocity,
            'motion_score': float(np.average(scores, weights=durations)),
            'window_motion_scores': scores, 'shake_prob': [row[15] for row in motion],
            **shake_decisions([row[15] for row in motion], shake_threshold),
            **rotation_decisions([row[4:12] for row in motion], rotation_threshold),
            'clip_camera_motion': clip_camera_motion(motion, zoom)}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)
