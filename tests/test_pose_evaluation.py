import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.evaluate_pose import METRICS, evaluate_pair, selected_inputs, summarize


class PoseBenchmark(unittest.TestCase):
    def test_upstream_layout_without_custom_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scan = root / 'scannetv2/scene0707_00/color_90'
            tum = root / 'tum/rgbd_dataset_freiburg3_sitting_xyz/rgb_90'
            sintel = root / 'sintel/training/final/alley_2'
            for folder in (scan, tum, sintel):
                folder.mkdir(parents=True)
            for i in range(90):
                (scan / f'frame_{i:04d}.jpg').touch()
                (tum / f'{1000 + i * .1:.6f}.png').touch()
            for i in range(1, 4):
                (sintel / f'frame_{i:04d}.png').touch()
            for dataset, sequence, expected in [
                ('scannet', 'scene0707_00', np.arange(90) * .1),
                ('tum', 'rgbd_dataset_freiburg3_sitting_xyz', np.arange(90) * .1),
                ('sintel', 'alley_2', np.arange(3) / 24),
            ]:
                paths, times, _ = selected_inputs(root, dataset, sequence)
                self.assertEqual(len(paths), len(expected))
                np.testing.assert_allclose(times, expected, atol=1e-12)

    @unittest.skipUnless(importlib.util.find_spec('evo') and importlib.util.find_spec('scipy'),
                         'Install requirements-eval.txt for metric tests')
    def test_global_similarity_is_removed_without_mutating_predictions(self):
        from scipy.spatial.transform import Rotation
        time = np.linspace(0, 3, 40)
        gt = np.tile(np.eye(4), (len(time), 1, 1))
        gt[:, :3, 3] = np.column_stack((time, np.sin(time), np.cos(2 * time)))
        gt[:, :3, :3] = Rotation.from_rotvec(np.column_stack((time * 0, time / 4, time * 0))).as_matrix()
        rotation = Rotation.from_euler('xyz', [.2, -.3, .4]).as_matrix()
        prediction = gt.copy()
        prediction[:, :3, :3] = rotation @ gt[:, :3, :3]
        prediction[:, :3, 3] = 2.5 * (gt[:, :3, 3] @ rotation.T) + [3, -4, 2]
        before = prediction.copy()
        result, aligned = evaluate_pair(prediction, gt)
        self.assertLess(result[METRICS[0]], 1e-12)
        self.assertLess(result[METRICS[1]], 1e-12)
        self.assertLess(result[METRICS[2]], 1e-5)
        np.testing.assert_allclose(aligned, gt, atol=1e-12)
        np.testing.assert_array_equal(prediction, before)

    def test_alignment_failures_prevent_full_benchmark_mean(self):
        rows = [
            {'sequence': 'valid', 'methods': {'velocity_only':
                {'status': 'evaluated', **dict.fromkeys(METRICS, .1)}}},
            {'sequence': 'degenerate', 'methods': {'velocity_only':
                {'status': 'failed', 'error': 'Degenerate covariance rank'}}},
        ]
        result = summarize(rows, ['velocity_only'])['velocity_only']
        self.assertIsNone(result['full_eligible_set_mean'])
        self.assertEqual(result['successful_sequences'], 1)
        self.assertEqual(result['failed_sequences'], ['degenerate'])


if __name__ == '__main__':
    unittest.main()
