from copy import deepcopy
import tempfile
from pathlib import Path
import unittest

import numpy as np

from idm.clip_head import ClipHead, clip_features
from idm.motion_mapping import CLASSES


def sample():
    motion = np.zeros((24, 18), dtype=np.float32)
    motion[:, 0], motion[:, 15] = .8, .1
    velocity = np.zeros((24, 6), dtype=np.float32)
    velocity[:, 2] = 2
    return {'protocol': 'uniform144', 'motion_prob': motion.tolist(),
            'zoom_prob': np.zeros((24, 2)).tolist(), 'velocity6': velocity.tolist(),
            'motion_score': .4}


class ClipHeadInference(unittest.TestCase):
    def test_constant_motion_features(self):
        features = clip_features(sample())
        expected = [.8, .8, .8, .8, 1, 1, .8, np.log(3), np.log(3), 0, 0, 0, 0]
        np.testing.assert_allclose(features[CLASSES.index('Move_In')], expected, atol=1e-7)
        np.testing.assert_allclose(features[CLASSES.index('Static'), -3:], [.4, .1, 1], atol=1e-7)

    def test_blend_preserves_native_outputs_and_is_idempotent(self):
        head = ClipHead()
        original = sample()
        before = deepcopy(original)
        output = head.apply(original)
        self.assertEqual(original, before)
        for name in ['motion_prob', 'zoom_prob', 'velocity6', 'motion_score']:
            self.assertEqual(output[name], original[name])
        for name in CLASSES:
            expected = .7 * output['clip_camera_motion_base'][name] + .3 * output['clip_camera_motion_head'][name]
            self.assertAlmostEqual(output['clip_camera_motion'][name], expected, places=14)
        self.assertEqual(head.apply(output), output)
        batch = head.apply_many([original, original])
        for result in batch:
            np.testing.assert_allclose(list(result['clip_camera_motion'].values()),
                                       list(output['clip_camera_motion'].values()), atol=1e-12, rtol=0)

    def test_rejects_incompatible_sampling_and_invalid_velocity(self):
        for key, value in [('protocol', 'windows24fps'), ('velocity6', [[0] * 6]),
                           ('motion_score', float('nan'))]:
            invalid = sample()
            invalid[key] = value
            with self.assertRaises(ValueError):
                clip_features(invalid)

    def test_rejects_other_backbone_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / 'different.pt'
            weights.write_bytes(b'different weights')
            with self.assertRaises(ValueError):
                ClipHead().verify_backbone(weights)


if __name__ == '__main__':
    unittest.main()
