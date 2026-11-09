#!/usr/bin/env python3
"""Evaluate IDM camera trajectories on ScanNet, TUM-dynamic, and Sintel.

Install requirements.txt and use the data prepared by MonST3R:
https://github.com/Junyi42/monst3r/blob/main/data/evaluation_script.md
Expected files below --data-root:
  scannetv2/<sequence>/color_90/*.jpg, pose_90.txt
  tum/<sequence>/rgb_90/*.png, groundtruth_90.txt
  sintel/training/final/<sequence>/*.png, camdata_left/<sequence>/*.cam
ScanNet uses nominal 30 FPS at stride 3; TUM RGB filenames preserve timestamps;
Sintel uses 24 FPS. Optional rgb_timestamps.txt files are supported.
Ground truth is used only for evaluation. Outputs include predictions, poses,
per-sequence metrics, and dataset summaries. The default velocity_only method
reproduces the camera-pose table in the README.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
from idm.trajectory import classified_velocity, integrate_trajectory

SEQUENCES = {
    'scannet': [f'scene{i:04d}_00' for i in range(707, 807)],
    'tum': ['rgbd_dataset_freiburg3_' + activity + '_' + motion
            for activity in ['sitting', 'walking']
            for motion in ['static', 'xyz', 'halfsphere', 'rpy']],
    'sintel': ['alley_2', 'ambush_4', 'ambush_5', 'ambush_6', 'cave_2', 'cave_4',
               'market_2', 'market_5', 'market_6', 'shaman_3', 'sleeping_1',
               'sleeping_2', 'temple_2', 'temple_3'],
}
METRICS = ('ate_rmse_m', 'rpe_translation_rmse_m', 'rpe_rotation_rmse_deg')


class InvalidGroundTruth(ValueError):
    """A selected sequence has non-finite ground-truth poses."""


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def dataset_folder(data_root, dataset):
    """Accept both upstream layouts and equivalent preselected RGB exports."""
    data_root = Path(data_root)
    if dataset == 'scannet':
        folder = data_root / 'scannetv2'
        return folder if folder.is_dir() else data_root / 'scannet'
    if dataset == 'sintel':
        folder = data_root / 'sintel' / 'training'
        return folder if folder.is_dir() else data_root / 'sintel'
    return data_root / 'tum'


def selected_inputs(data_root, dataset, name):
    folder = dataset_folder(data_root, dataset)
    if dataset == 'sintel':
        paths = sorted((folder / 'final' / name).glob('*.png'))
        times = np.arange(len(paths), dtype=float) / 24
        timing = 'Sintel native 24 FPS'
    else:
        folder = folder / name
        paths = sorted((folder / ('color_90' if dataset == 'scannet' else 'rgb_90')).glob(
            '*.jpg' if dataset == 'scannet' else '*.png'))
        if len(paths) != 90:
            raise ValueError(f'{dataset}/{name}: expected 90 preselected RGB frames, got {len(paths)}')
        stamp_file = folder / 'rgb_timestamps.txt'
        if stamp_file.is_file():
            times = np.loadtxt(stamp_file, ndmin=1)
        elif dataset == 'tum':
            # MonST3R preserves each original RGB timestamp in its PNG filename.
            times = np.asarray([float(path.stem) for path in paths])
        else:
            times = np.arange(90, dtype=float) * 3 / 30
        timing = 'Actual selected RGB timestamps'
        if dataset == 'scannet' and (not stamp_file.is_file()
                or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
            times = np.arange(90, dtype=float) * 3 / 30
            timing = 'ScanNet nominal 30 FPS, source stride 3; sensor timestamps unavailable'
    if len(paths) < 2 or times.shape != (len(paths),):
        raise ValueError(f'{dataset}/{name}: missing frames or mismatched timestamps')
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError(f'{dataset}/{name}: timestamps must be finite and increasing')
    return paths, times - times[0], timing


def load_groundtruth(data_root, dataset, name):
    from scipy.spatial.transform import Rotation
    folder = dataset_folder(data_root, dataset)
    if dataset == 'scannet':
        path = folder / name / 'pose_90.txt'
        poses = np.loadtxt(path).reshape(-1, 4, 4)
        hashes = {path.name: digest(path)}
    elif dataset == 'tum':
        path = folder / name / 'groundtruth_90.txt'
        rows = np.loadtxt(path, ndmin=2)
        if rows.shape[1] != 8:
            raise ValueError('Expected TUM timestamp, xyz, quaternion xyzw')
        if not np.isfinite(rows).all():
            raise InvalidGroundTruth('Non-finite ground-truth poses')
        poses = np.tile(np.eye(4), (len(rows), 1, 1))
        poses[:, :3, :3] = Rotation.from_quat(rows[:, 4:8]).as_matrix()
        poses[:, :3, 3] = rows[:, 1:4]
        hashes = {path.name: digest(path)}
    else:
        paths = sorted((folder / 'camdata_left' / name).glob('*.cam'))
        images = sorted((folder / 'final' / name).glob('*.png'))
        if not paths or [p.stem for p in paths] != [p.stem for p in images]:
            raise ValueError(f'{name}: Sintel image and camera frame IDs differ')
        poses, hashes = [], {}
        for path in paths:
            with path.open('rb') as stream:
                tag = np.fromfile(stream, dtype='<f4', count=1)
                if len(tag) != 1 or tag[0] != 202021.25:
                    raise ValueError(f'{path.name}: invalid Sintel camera header')
                np.fromfile(stream, dtype='<f8', count=9)  # Intrinsics are unused.
                extrinsic = np.fromfile(stream, dtype='<f8', count=12).reshape(3, 4)
            transform = np.eye(4)
            transform[:3] = extrinsic
            poses.append(np.linalg.inv(transform))
            hashes[path.name] = digest(path)
        poses = np.asarray(poses)
        poses[:, :3, 3] -= poses[:, :3, 3].mean(axis=0)
    if not np.isfinite(poses).all():
        bad = np.flatnonzero(~np.isfinite(poses).all(axis=(1, 2))).tolist()
        raise InvalidGroundTruth(f'Non-finite ground-truth poses at selected indices {bad}')
    # Match the upstream quaternion conversion of slightly rounded rotations.
    poses[:, :3, :3] = Rotation.from_matrix(poses[:, :3, :3]).as_matrix()
    return poses, hashes


def predict_selected(model, paths, times, predict_frames, precision='bf16'):
    duration = float(times[-1] + np.median(np.diff(times)))
    motion, velocity, starts, plans = [], [], [], []
    for start in range(0, math.ceil(duration), 6):
        valid = min(6., duration - start)
        bins = math.ceil(valid * 4 - 1e-8)
        if bins <= 0:
            continue
        targets = start + np.arange(144) / 24
        right = np.clip(np.searchsorted(times, targets), 0, len(times) - 1)
        left = np.maximum(right - 1, 0)
        indices = np.where(abs(times[left] - targets) <= abs(times[right] - targets), left, right)
        frames = {}
        for i in np.unique(indices):
            frame = cv2.imread(str(paths[i]), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError('Failed to decode ' + paths[i].name)
            frames[i] = cv2.cvtColor(cv2.resize(frame, (128, 96), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        output = predict_frames(model, np.stack([frames[i] for i in indices]), precision=precision)
        motion.extend(output['motion_prob'][:bins])
        velocity.extend(output['velocity6'][:bins])
        starts.extend((start + np.arange(bins) / 4).tolist())
        plans.append({'start_seconds': start, 'valid_seconds': valid,
                      'bins': bins, 'selected_rgb_indices': indices.tolist()})
    return {'motion_prob': motion, 'velocity6': velocity,
            'bucket_start_seconds': starts, 'windows': plans, 'duration_seconds': duration}


def evaluate_pair(poses, groundtruth):
    """Global Sim(3); ATE/RPE RMSE with RPE delta=1 selected frame."""
    from evo.core import metrics
    from evo.core.trajectory import PoseTrajectory3D
    if poses.shape != groundtruth.shape or len(poses) < 2:
        raise ValueError('Predicted and ground-truth trajectories must match')
    times = np.arange(len(groundtruth), dtype=np.float64)
    reference = PoseTrajectory3D(poses_se3=groundtruth, timestamps=times)
    estimate = PoseTrajectory3D(poses_se3=poses, timestamps=times)
    aligned = deepcopy(estimate)
    _, _, scale = aligned.align(reference, correct_scale=True)
    definitions = [
        (METRICS[0], metrics.APE(metrics.PoseRelation.translation_part)),
        (METRICS[1], metrics.RPE(metrics.PoseRelation.translation_part,
             delta=1, delta_unit=metrics.Unit.frames, rel_delta_tol=.01, all_pairs=True)),
        (METRICS[2], metrics.RPE(metrics.PoseRelation.rotation_angle_deg,
             delta=1, delta_unit=metrics.Unit.frames, rel_delta_tol=.01, all_pairs=True)),
    ]
    result = {}
    for name, metric in definitions:
        metric.process_data((reference, aligned))
        result[name] = float(metric.get_statistic(metrics.StatisticsType.rmse))
    result.update(sim3_scale=float(scale), evaluated_poses=len(groundtruth))
    return result, aligned.poses_se3


def summarize(rows, methods):
    summary = {}
    for method in methods:
        valid = [r['methods'][method] for r in rows if r['methods'][method]['status'] == 'evaluated']
        failed = [r['sequence'] for r in rows if r['methods'][method]['status'] != 'evaluated']
        mean = {k: float(np.mean([r[k] for r in valid])) if valid else None for k in METRICS}
        summary[method] = {'eligible_sequences': len(rows), 'successful_sequences': len(valid),
                           'failed_sequences': failed, 'successful_subset_mean': mean,
                           'full_eligible_set_mean': mean if valid and not failed else None}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', type=Path, required=True, help='Parent of scannetv2/, tum/, sintel/')
    parser.add_argument('--output', type=Path, default=Path('outputs/pose'))
    parser.add_argument('--datasets', nargs='+', choices=list(SEQUENCES), default=list(SEQUENCES))
    parser.add_argument('--sequences', nargs='+', help='Optional subset; requires exactly one dataset')
    parser.add_argument('--methods', nargs='+', choices=['velocity_only', 'classification_velocity'], default=['velocity_only'])
    parser.add_argument('--weights', type=Path, default=PACKAGE_ROOT / 'weights/idm.pt')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--precision', choices=['bf16', 'fp32'], default='bf16')
    parser.add_argument('--metrics-only', action='store_true', help='Reuse verified predictions without loading the model')
    parser.add_argument('--predictions-root', type=Path, help='Existing dataset/sequence/predictions.json directory for --metrics-only')
    args = parser.parse_args()
    if args.sequences and (len(args.datasets) != 1 or not set(args.sequences) <= set(SEQUENCES[args.datasets[0]])):
        parser.error('--sequences must belong to one selected benchmark dataset')
    if args.predictions_root and not args.metrics_only:
        parser.error('--predictions-root requires --metrics-only')
    base_identity = {'checkpoint_sha256': digest(args.weights),
                     'runtime_sha256': digest(PACKAGE_ROOT / 'idm/runtime.py'),
                     'trajectory_sha256': digest(PACKAGE_ROOT / 'idm/trajectory.py'),
                     'precision': args.precision}
    identity = {**base_identity, 'script_sha256': digest(__file__), 'device': args.device,
                'versions': {k: importlib.metadata.version(k) for k in
                             ['torch', 'numpy', 'opencv-python-headless', 'scipy', 'evo']}}
    model = None
    for dataset in args.datasets:
        names = args.sequences or SEQUENCES[dataset]
        rows, excluded = [], []
        for name in names:
            paths, times, timing = selected_inputs(args.data_root, dataset, name)
            outdir = args.output / dataset / name
            input_identity = {**base_identity, 'rgb_sha256': {p.name: digest(p) for p in paths},
                              'timestamps_seconds': times.tolist()}
            cache_identity = {**identity, **input_identity}
            cache_path = (args.predictions_root or args.output) / dataset / name / 'predictions.json'
            cached = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
            if args.metrics_only:
                if any(cached.get('identity', {}).get(k) != v for k, v in input_identity.items()):
                    raise ValueError(f'{dataset}/{name}: saved predictions do not match weights, inputs, or runtime')
                prediction = cached['prediction']
            elif cached.get('identity') == cache_identity:
                prediction = cached['prediction']
            else:
                from idm.runtime import load_model, predict_frames
                if model is None:
                    model, _ = load_model(args.weights, args.device)
                prediction = predict_selected(model, paths, times, predict_frames, args.precision)
                write_json(cache_path, {'identity': cache_identity, 'prediction': prediction})
            try:
                groundtruth, hashes = load_groundtruth(args.data_root, dataset, name)
            except InvalidGroundTruth as error:
                excluded.append({'sequence': name, 'error': str(error)})
                print(dataset, name, 'INVALID GROUND TRUTH', str(error), flush=True)
                continue
            if len(groundtruth) != len(paths):
                raise ValueError(f'{dataset}/{name}: RGB and ground-truth frame counts differ')
            row = {'sequence': name, 'frames': len(paths), 'timing': timing,
                   'groundtruth_sha256': hashes, 'methods': {}}
            native = np.asarray(prediction['velocity6'])
            arrays = {'groundtruth': groundtruth, 'timestamps_seconds': times}
            for method in args.methods:
                velocity = native if method == 'velocity_only' else classified_velocity(prediction['motion_prob'], native)
                poses = integrate_trajectory(velocity, prediction['bucket_start_seconds'], times)
                arrays[method] = poses
                try:
                    score, aligned = evaluate_pair(poses, groundtruth)
                    row['methods'][method] = {'status': 'evaluated', **score}
                    arrays[method + '_aligned'] = np.asarray(aligned)
                except Exception as error:
                    row['methods'][method] = {'status': 'failed', 'error': str(error)}
            outdir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(outdir / 'trajectories.npz', **arrays)
            write_json(outdir / 'metrics.json', row)
            rows.append(row)
            print(dataset, name, json.dumps(row['methods']), flush=True)
        report = {'dataset': dataset, 'expected_sequences': names,
                  'benchmark_sequences': SEQUENCES[dataset], 'is_subset_run': names != SEQUENCES[dataset],
                  'identity': identity, 'predictions_reused': args.metrics_only,
                  'excluded_groundtruth': excluded, 'sequences': rows,
                  'protocol': {'source': 'https://github.com/cambrian-mllm/cambrian-p#pose-estimation-performance',
                               'alignment': 'Global Sim(3) per sequence and method',
                               'rpe': 'delta=1 selected frame, all_pairs=True; translation meters, rotation degrees',
                               'aggregation': 'Arithmetic mean of per-sequence RMSE; full mean is null if a method fails',
                               'model_input': 'Selected RGB frames only; nearest timestamp at 24 FPS; six-second windows',
                               'groundtruth_used_for_prediction_or_tuning': False},
                  'summary': summarize(rows, args.methods)}
        write_json(args.output / dataset / 'report.json', report)
        print(dataset, 'SUMMARY', json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()
