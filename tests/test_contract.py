import unittest
import numpy as np

from idm.motion_mapping import bucket_scores, clip_camera_motion


class MappingContract(unittest.TestCase):
    def test_diagonal_supports_pan_and_tilt(self):
        motion = np.zeros((24, 18), np.float32)
        motion[10, 5] = .8
        scores = clip_camera_motion(motion, np.zeros((24, 2)))
        self.assertAlmostEqual(scores['Pan_Right'], .8)
        self.assertAlmostEqual(scores['Tilt_Up'], .8)
        self.assertEqual(scores['Pan_Left'], 0)
        self.assertAlmostEqual(scores['Static'], .2)

    def test_static_requires_absence_throughout(self):
        motion = np.zeros((24, 18), np.float32)
        motion[:, 12] = 1  # Special key is not the CameraBench Static score.
        zoom = np.zeros((24, 2), np.float32)
        self.assertEqual(clip_camera_motion(motion, zoom)['Static'], 1)
        motion[1, 15] = .9
        self.assertAlmostEqual(clip_camera_motion(motion, zoom)['Static'], .1)
        zoom[5, 0] = 1
        self.assertEqual(clip_camera_motion(motion, zoom)['Static'], 0)

    def test_unknown_shapes_and_nonfinite_fail(self):
        with self.assertRaises(ValueError):
            bucket_scores(np.zeros((24, 17)), np.zeros((24, 2)))
        with self.assertRaises(ValueError):
            bucket_scores(np.full((24, 18), np.nan), np.zeros((24, 2)))


if __name__ == '__main__':
    unittest.main()
