import unittest
from unittest.mock import patch

import numpy as np

from idm.hud import bucket_for_frame, draw_hud
from idm.postprocessing import rotation_decisions


class HudContract(unittest.TestCase):
    def test_alignment_uses_source_frame_indices(self):
        sampling = {'source_frame_indices': (np.arange(144) * 10).tolist()}
        self.assertEqual(bucket_for_frame(0, sampling), 0)
        self.assertEqual(bucket_for_frame(54, sampling), 0)
        self.assertEqual(bucket_for_frame(55, sampling), 1)
        self.assertEqual(bucket_for_frame(1430, sampling), 23)

    def test_repeated_indices_stay_in_range(self):
        sampling = {'source_frame_indices': np.rint(np.linspace(0, 3, 144)).tolist()}
        indices = [bucket_for_frame(i, sampling) for i in range(4)]
        self.assertEqual(indices, sorted(indices))
        self.assertTrue(all(0 <= index < 24 for index in indices))

    def test_overlay_does_not_change_predictions_or_source(self):
        frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
        sample = {'sampling': {'source_frame_indices': list(range(144)), 'source_fps': 24},
                  'motion_prob': np.zeros((24, 18)).tolist(),
                  'zoom_prob': np.zeros((24, 2)).tolist(),
                  'velocity6': np.zeros((24, 6)).tolist()}
        result = draw_hud(frame, sample, 0)
        self.assertEqual(result.shape, frame.shape)
        self.assertFalse(np.array_equal(result, frame))
        self.assertTrue((frame == 120).all())
        self.assertEqual(sample['motion_prob'], np.zeros((24, 18)).tolist())

    def test_rotation_is_exclusive_and_agrees_with_json(self):
        import cv2
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cases = [([.49] * 8, .5, -1), ([.8] * 8, .5, 0),
                 ([.8, .9, .7, .6, .8, .7, .6, .7], .5, 1),
                 ([.8] * 8, .9, -1)]
        for direction in range(8):
            probabilities = [.1] * 8
            probabilities[direction] = .5
            cases.append((probabilities, .5, direction))
        for probabilities, threshold, expected in cases:
            with self.subTest(probabilities=probabilities, threshold=threshold):
                motion = np.zeros((24, 18))
                motion[:, 4:12] = probabilities
                sample = {'sampling': {'source_frame_indices': list(range(144)), 'source_fps': 24},
                          'motion_prob': motion.tolist(), 'zoom_prob': np.zeros((24, 2)).tolist(),
                          'velocity6': np.zeros((24, 6)).tolist(),
                          **rotation_decisions(motion[:, 4:12], threshold)}
                self.assertEqual(sample['rotation_sector'], [expected] * 24)
                with patch('cv2.ellipse', wraps=cv2.ellipse) as ellipse:
                    # Other keys may use a different threshold; rotation uses its own.
                    draw_hud(frame, sample, 0, threshold=.95)
                sectors = ellipse.call_args_list
                active = [i for i, call in enumerate(sectors) if call.args[6] == (166, 230, 75)]
                self.assertEqual(active, [] if expected < 0 else [expected])
                self.assertEqual(len(sectors), 8)
                for i, call in enumerate(sectors):
                    self.assertEqual(call.args[5] - call.args[4], 20 if i % 2 == 0 else 70)
                    self.assertEqual((call.args[5] + call.args[4]) / 2, -i * 45)


if __name__ == '__main__':
    unittest.main()
