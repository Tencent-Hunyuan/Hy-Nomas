import unittest

from idm.runtime import window_plan
from idm.hud import bucket_for_frame


class WindowSampling(unittest.TestCase):
    def test_six_seconds_is_one_native_window(self):
        plans = list(window_plan(144, 24))
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0]['source_frame_indices'], list(range(144)))
        self.assertEqual(plans[0]['valid_bins'], 24)

    def test_short_tail_is_padded_and_only_valid_bins_returned(self):
        plans = list(window_plan(204, 24))
        self.assertEqual(len(plans), 2)
        self.assertEqual(plans[1]['source_frame_indices'][:60], list(range(144, 204)))
        self.assertEqual(plans[1]['source_frame_indices'][60:], [203] * 84)
        self.assertEqual(plans[1]['valid_bins'], 10)
        self.assertEqual(sum(p['valid_bins'] for p in plans), 34)

    def test_long_clip_has_quarter_second_buckets(self):
        plans = list(window_plan(3507, 48))
        self.assertEqual(sum(p['valid_bins'] for p in plans), 293)
        self.assertEqual(plans[0]['source_frame_indices'], list(range(0, 288, 2)))
        self.assertEqual(plans[1]['source_frame_indices'][0], 288)
        sampling = {'source_fps': 48, 'bucket_start_seconds': [i / 4 for i in range(293)]}
        self.assertEqual(bucket_for_frame(11, sampling), 0)
        self.assertEqual(bucket_for_frame(12, sampling), 1)
        self.assertEqual(bucket_for_frame(288, sampling), 24)
        self.assertEqual(bucket_for_frame(3506, sampling), 292)


if __name__ == '__main__':
    unittest.main()
