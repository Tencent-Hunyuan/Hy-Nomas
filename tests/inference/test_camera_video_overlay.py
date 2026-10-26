import json
import unittest

import numpy as np

from apps.camera_video_overlay import (
    CameraPose,
    _actions_at_time,
    _cached_demo_controls,
    _draw_demo_timeline,
    _draw_text_overlay_frame,
    _visual_camera_poses,
    burn_camera_overlay,
    burn_video_overlays,
)


class DemoOverlayTests(unittest.TestCase):
    def test_public_renderers_preserve_source_and_match_camera_output(self):
        source = np.full((2, 176, 320, 3), 70, dtype=np.uint8)
        trajectory = json.dumps({
            "frames": [{
                "t": 0,
                "position": {"x": 0, "y": 0, "z": 0},
                "quaternion": {"x": 0, "y": 0, "z": 0, "w": 1},
            }],
            "durationMs": 1000,
            "keyEvents": [{"t": 0, "code": "KeyW", "pressed": True}],
            "visualization_rotation_scale": 2,
        })
        camera = burn_camera_overlay(source, trajectory, 24)
        segmented = burn_video_overlays(
            source, trajectory, 24,
            segment_boundaries=[0, 2], segment_has_camera=[True],
        )
        np.testing.assert_array_equal(camera, segmented)
        self.assertTrue(np.all(source == 70))
        self.assertTrue(np.all(camera[:, :100, :160] == 70))
        self.assertTrue(np.any(camera[:, :65, 220:] != 70))
        self.assertTrue(np.any(camera[0, 120:] != camera[1, 120:]))

    def test_visual_rotation_gain_does_not_change_source_or_timing(self):
        angle = np.deg2rad(40) / 2
        poses = (
            CameraPose(0, np.zeros(3), np.array([0., 0., 0., 1.])),
            CameraPose(1000, np.ones(3), np.array([0., np.sin(angle), 0., np.cos(angle)])),
        )
        result = _visual_camera_poses(poses, 2)
        self.assertAlmostEqual(np.rad2deg(2 * np.arccos(result[1].quaternion[3])), 60)
        self.assertAlmostEqual(np.rad2deg(2 * np.arccos(poses[1].quaternion[3])), 40)
        self.assertEqual(result[1].t, 1000)
        np.testing.assert_array_equal(result[1].position, poses[1].position)

    def test_caption_leaves_top_of_video_untouched(self):
        image = np.full((704, 1280, 3), 70, dtype=np.uint8)
        _draw_text_overlay_frame(image, "Explore the moon base")
        self.assertTrue(np.all(image[:500] == 70))
        self.assertTrue(np.any(image[550:] != 70))

    def test_camera_keys_are_bottom_right_and_active_keys_are_white(self):
        idle = _cached_demo_controls(1280, 704, (False,) * 8)
        active = _cached_demo_controls(1280, 704, (True,) + (False,) * 7)
        self.assertFalse(active[:490, :, 3].any())
        self.assertFalse(active[:, :800, 3].any())
        self.assertEqual(tuple(active[510, 890]), (255, 255, 255, 255))
        self.assertEqual(int(idle[510, 890, 3]), 38)

    def test_timeline_only_changes_bottom_strip(self):
        image = np.zeros((704, 1280, 3), dtype=np.uint8)
        turns = [
            {"number": 1, "start_seconds": 0, "end_seconds": 3},
            {"number": 2, "start_seconds": 3, "end_seconds": 8, "camera": {"name": "path"}},
        ]
        _draw_demo_timeline(image, turns, 4)
        self.assertFalse(image[:650].any())
        self.assertTrue(image[650:].any())

    def test_missing_key_events_do_not_invent_movement(self):
        self.assertFalse(any(_actions_at_time([], 1000).values()))


if __name__ == "__main__":
    unittest.main()
