"""Frozen clip-level classifiers and blending for uniform video predictions."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .motion_mapping import CLASSES, bucket_scores, clip_camera_motion

DEFAULT_CLIP_HEAD = Path(__file__).resolve().parents[1] / 'weights/clip_head.json'
MOTION_CLASSES = tuple(name for name in CLASSES if name != 'Static')
VELOCITY_AXES = {
    'Move_In': (2, 1), 'Move_Out': (2, -1), 'Move_Left': (0, -1), 'Move_Right': (0, 1),
    'Move_Up': (1, -1), 'Move_Down': (1, 1), 'Pan_Right': (3, 1), 'Pan_Left': (3, -1),
    'Tilt_Up': (4, -1), 'Tilt_Down': (4, 1),
    'Roll_Clockwise': (5, 1), 'Roll_Counterclockwise': (5, -1),
}


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def three_bucket_minimum(values):
    result = values.copy()
    result[1:-1] = np.minimum(np.minimum(values[:-2], values[1:-1]), values[2:])
    result[0], result[-1] = result[1], result[-2]
    return result


def clip_features(sample):
    """Return the fixed float32 [15,13] features used to train the clip head."""
    if sample.get('protocol') != 'uniform144':
        raise ValueError('The clip-level head requires --sampling uniform')
    motion = np.asarray(sample['motion_prob'], dtype=np.float32)
    zoom = np.asarray(sample['zoom_prob'], dtype=np.float32)
    velocity = np.asarray(sample['velocity6'], dtype=np.float32)
    subject = float(sample['motion_score'])
    if motion.shape != (24, 18) or zoom.shape != (24, 2) or velocity.shape != (24, 6):
        raise ValueError('The clip-level head requires 24 uniform prediction bins')
    if not np.isfinite(velocity).all() or not np.isfinite(subject):
        raise ValueError('Velocity and subject motion quality must be finite')
    probabilities = bucket_scores(motion, zoom)
    motion_maxima = np.asarray([probabilities[name].max() for name in MOTION_CLASSES])
    features = []
    for name in CLASSES:
        values = probabilities[name]
        ordered = np.sort(values)
        row = [values.max(), ordered[-3:].mean(), ordered[-5:].mean(), values.mean(),
               (values > .3).mean(), (values > .5).mean(), three_bucket_minimum(values).max()]
        if name in VELOCITY_AXES:
            axis, sign = VELOCITY_AXES[name]
            smoothed = three_bucket_minimum(velocity[:, axis] * sign)
            transformed = np.sign(smoothed) * np.log1p(np.abs(smoothed))
            row.extend([transformed.max(), transformed.mean()])
        else:
            row.extend([0., 0.])
        row.append(float((motion_maxima > values.max()).sum()))
        if name == 'Static':
            moving = np.column_stack([probabilities[c] for c in MOTION_CLASSES])
            row.extend([subject, float(motion[:, 15].max()), float((moving > .3).any(0).sum())])
        else:
            row.extend([0., 0., 0.])
        features.append(row)
    return np.asarray(features, dtype=np.float32)


class ClipHead:
    """15 standardized logistic classifiers blended with the base predictions."""

    def __init__(self, weights=DEFAULT_CLIP_HEAD):
        weights = Path(weights)
        parameters = json.loads(weights.read_text())
        if (parameters.get('format') != 'hy-nomas-idm-clip-head-v1'
                or parameters.get('feature_version') != 'camera-motion-statistics-13-v1'
                or parameters.get('sampling') != 'uniform144'
                or parameters.get('buckets') != 24
                or parameters.get('classes') != list(CLASSES)):
            raise ValueError('Incompatible clip-level head format or feature definition')
        self.alpha = float(parameters['blend_alpha'])
        if not np.isfinite(self.alpha) or not 0 <= self.alpha <= 1:
            raise ValueError('Invalid clip-head blend weight')
        self.mean, self.scale, self.weight = [], [], []
        for name in CLASSES:
            values = parameters['per_class'][name]
            mean = np.asarray(values['mean'], dtype=np.float64)
            scale = np.asarray(values['scale'], dtype=np.float64)
            weight = np.asarray(values['weight'], dtype=np.float64)
            bias = float(values['bias'])
            if any(v.shape != (13,) or not np.isfinite(v).all() for v in (mean, scale, weight)):
                raise ValueError(f'{name}: invalid clip-head parameters')
            if np.any(scale <= 0) or not np.isfinite(bias):
                raise ValueError(f'{name}: invalid normalization or bias')
            self.mean.append(mean)
            self.scale.append(scale)
            self.weight.append(np.r_[weight, bias])
        self.mean, self.scale, self.weight = map(np.asarray, (self.mean, self.scale, self.weight))
        self.backbone_sha256 = parameters['backbone_sha256']
        self.metadata = {'format': parameters['format'], 'sha256': file_sha256(weights),
                         'backbone_sha256': self.backbone_sha256,
                         'sampling': 'uniform144', 'blend_alpha': self.alpha,
                         'feature_dim': 13, 'trainable_parameters': 210}

    def verify_backbone(self, weights):
        if file_sha256(weights) != self.backbone_sha256:
            raise ValueError('The clip-level head does not match these backbone weights')

    def apply_many(self, samples):
        if not samples:
            raise ValueError('No predictions to process')
        features = np.asarray([clip_features(sample) for sample in samples], dtype=np.float64)
        head = np.empty((len(samples), len(CLASSES)), dtype=np.float64)
        for i, name in enumerate(CLASSES):
            normalized = (features[:, i] - self.mean[i]) / self.scale[i]
            design = np.column_stack([normalized, np.ones(len(samples))])
            logits = np.clip(design @ self.weight[i], -30, 30)
            head[:, i] = 1 / (1 + np.exp(-logits))
        results = []
        for index, sample in enumerate(samples):
            # Rebuild the base values so repeated application never blends twice.
            base = clip_camera_motion(sample['motion_prob'], sample['zoom_prob'])
            fused = {name: float((1 - self.alpha) * base[name] + self.alpha * head[index, i])
                     for i, name in enumerate(CLASSES)}
            results.append({**sample, 'clip_camera_motion_base': base,
                            'clip_camera_motion_head': dict(zip(CLASSES, head[index].tolist())),
                            'clip_camera_motion': fused, 'clip_head': dict(self.metadata)})
        return results

    def apply(self, sample):
        return self.apply_many([sample])[0]
