import unittest

from tools.evaluate_camerabench import average_precision


class CameraBenchAveragePrecision(unittest.TestCase):
    def test_tied_scores_share_a_threshold(self):
        self.assertEqual(average_precision([1, 0], [.5, .5]), .5)
        self.assertEqual(average_precision([0, 1], [.5, .5]), .5)
        self.assertAlmostEqual(average_precision([1, 0, 1], [.9, .8, .7]), 5 / 6)

    def test_invalid_labels_and_missing_positives_fail(self):
        for labels, scores in [([1, -1], [.9, .1]), ([0, 0], [.9, .1]),
                               ([1, 0], [float('nan'), .1]), ([], [])]:
            with self.assertRaises(ValueError):
                average_precision(labels, scores)


if __name__ == '__main__':
    unittest.main()
