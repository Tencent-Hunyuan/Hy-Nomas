import unittest
import numpy as np

from idm.trajectory import classified_velocity, integrate_trajectory, se3_increment


class TrajectoryGeometry(unittest.TestCase):
    def test_translation_and_quarter_turn_conventions(self):
        np.testing.assert_allclose(se3_increment([1, 2, 3, 0, 0, 0], 0.5)[:3, 3], [.5, 1, 1.5])
        forward = np.asarray([0, 0, 1])
        right = np.asarray([1, 0, 0])
        np.testing.assert_allclose(se3_increment([0, 0, 0, 90, 0, 0], 1)[:3, :3] @ forward, right, atol=1e-12)
        np.testing.assert_allclose(se3_increment([0, 0, 0, 0, 90, 0], 1)[:3, :3] @ forward, [0, 1, 0], atol=1e-12)
        np.testing.assert_allclose(se3_increment([0, 0, 0, 0, 0, 90], 1)[:3, :3] @ right, [0, 1, 0], atol=1e-12)

    def test_curved_body_motion_and_time_partition_invariance(self):
        velocity = [0, 0, 1, 90, 0, 0]
        full = se3_increment(velocity, 1)
        np.testing.assert_allclose(full[:3, 3], [2 / np.pi, 0, 2 / np.pi], atol=1e-12)
        np.testing.assert_allclose(se3_increment(velocity, .3) @ se3_increment(velocity, .7), full, atol=1e-12)
        np.testing.assert_allclose(full[:3, :3].T @ full[:3, :3], np.eye(3), atol=1e-12)

    def test_classification_gates_and_exclusive_directions(self):
        motion = np.zeros((8, 18))
        for i in range(8): motion[i, 4 + i] = .8
        motion[:, 0], motion[:, 2] = .8, .7
        motion[:, 13], motion[:, 14] = .7, .9
        motion[:, 16], motion[:, 17] = .6, .8
        velocity = classified_velocity(motion, -np.ones((8, 6)))
        np.testing.assert_array_equal(velocity[:, :3], np.tile([0, 1, 1], (8, 1)))
        np.testing.assert_array_equal(velocity[:, 3:5], [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)])
        np.testing.assert_array_equal(velocity[:, 5], -np.ones(8))
        np.testing.assert_array_equal(classified_velocity(np.zeros((1, 18)), np.ones((1, 6))), np.zeros((1, 6)))

    def test_bucket_boundaries_and_invalid_inputs(self):
        velocity = [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]]
        poses = integrate_trajectory(velocity, [0, .25], [0, .1, .4, .75])
        np.testing.assert_allclose(poses[:, :3, 3], [[0, 0, 0], [.1, 0, 0], [.25, .15, 0], [.25, .5, 0]], atol=1e-12)
        with self.assertRaises(ValueError): integrate_trajectory(velocity, [0, .25], [0, .1, .1])
        with self.assertRaises(ValueError): se3_increment([0] * 6, float('nan'))
        with self.assertRaises(ValueError): classified_velocity(np.zeros((1, 18)), np.ones((1, 6)), threshold=float('nan'))


if __name__ == '__main__':
    unittest.main()
