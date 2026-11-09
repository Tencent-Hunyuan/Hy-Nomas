#!/usr/bin/env python3
"""Score uniform144 predictions against the 15 official CameraBench JSONL splits.

Predictions are the JSON produced by infer.py --sampling uniform. Labels use
the official image/label fields; only annotated video/class pairs are scored.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from idm.motion_mapping import CLASSES


def average_precision(labels, scores):
    """Non-interpolated AP, grouping equal scores at the same threshold."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != labels.shape or not len(labels):
        raise ValueError('Expected matching, nonempty label and score arrays')
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError('Labels must be binary and scores must be finite')
    positives = int(labels.sum())
    if positives == 0:
        raise ValueError('AP requires at least one positive label')
    order = np.argsort(-scores, kind='stable')
    ranked = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(ranked)), len(ranked) - 1]
    true_positives = np.cumsum(labels[order], dtype=np.float64)[ends]
    precision = true_positives / (ends + 1)
    return float(np.sum(np.diff(np.r_[0., true_positives]) / positives * precision))


def evaluate(predictions, label_dir):
    by_video = {}
    for sample in predictions['samples']:
        name = Path(sample['video']).name
        if name in by_video:
            raise ValueError(f'Duplicate video filename: {name}')
        if sample.get('protocol') != 'uniform144':
            raise ValueError(f'{name}: regenerate predictions with --sampling uniform')
        values = sample['clip_camera_motion']
        scores = np.asarray([values[k] for k in CLASSES], dtype=float)
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError(f'{name}: invalid camera-motion probabilities')
        by_video[name] = values
    per_class, video_names, hashes = {}, set(), {}
    for name in CLASSES:
        path = Path(label_dir) / (name + '.jsonl')
        payload = path.read_bytes()
        hashes[path.name] = hashlib.sha256(payload).hexdigest()
        rows = [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
        labels, scores, seen = [], [], set()
        for row in rows:
            video, label = row['image'], row['label'].strip().lower()
            if label not in ('yes', 'no') or video in seen:
                raise ValueError(f'{name}: invalid or duplicate label for {video}')
            if video not in by_video:
                raise ValueError(f'Missing prediction: {video}')
            seen.add(video)
            video_names.add(video)
            labels.append(int(label == 'yes'))
            scores.append(by_video[video][name])
        per_class[name] = {'AP': average_precision(labels, scores),
                           'known': len(labels), 'positive': sum(labels)}
    return {'evaluation_unit': 'clip', 'prediction_field': 'clip_camera_motion',
            'sampling': 'uniform144', 'videos': len(video_names), 'classes': len(CLASSES),
            'known_labels': sum(v['known'] for v in per_class.values()),
            'macro_mAP': float(np.mean([v['AP'] for v in per_class.values()])),
            'per_class': per_class, 'label_sha256': hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True, help='JSON from infer.py --sampling uniform')
    parser.add_argument('--labels', type=Path, required=True, help='Official camerabench/data/binary_classification directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protected = {args.predictions.resolve(), *(p.resolve() for p in args.labels.glob('*.jsonl'))}
    if args.output.resolve() in protected:
        parser.error('--output must not overwrite predictions or labels')
    report = evaluate(json.loads(args.predictions.read_text()), args.labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(args.output)
    print(f"CameraBench mAP: {100 * report['macro_mAP']:.4f}% ({report['videos']} videos, {report['classes']} classes)")


if __name__ == '__main__':
    main()
