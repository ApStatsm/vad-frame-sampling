"""User-run ver4 geometry and counterfactual tests; no checkpoints/GT required."""

import unittest

import numpy as np

from vadfs.context_ver4 import replace_neighbors, restore_full_layout, target_positions


class ContextVer4Test(unittest.TestCase):
    def test_target_selection_has_no_boundary_neighbors(self):
        for n in (1, 8, 9, 16, 17, 64, 1024, 2048, 2065, 8193):
            rows = list(range(0, n, 8))
            targets = target_positions(n)
            self.assertEqual(targets, sorted(set(targets)))
            self.assertLessEqual(len(targets), 2)
            for slot in targets:
                self.assertTrue(0 < slot < len(rows) - 1)
                self.assertEqual((slot - 1) // 256, slot // 256)
                self.assertEqual((slot + 1) // 256, slot // 256)
                self.assertEqual((slot - 1) // 64, slot // 64)
                self.assertEqual((slot + 1) // 64, slot // 64)
        self.assertEqual(target_positions(16), [])
        self.assertEqual(target_positions(17), [1])

    def test_only_neighbors_change_and_feature_target_is_preserved(self):
        for n in (17, 64, 1024, 4097):
            features = np.arange(n * 2).reshape(n, 2)
            for slot in target_positions(n):
                base, changed = replace_neighbors(n, slot)
                self.assertEqual(len(base), len(changed))
                self.assertEqual(changed, sorted(set(changed)))
                self.assertEqual([i for i, (a, b) in enumerate(zip(base, changed)) if a != b],
                                 [slot - 1, slot + 1])
                self.assertEqual(changed[slot - 1:slot + 2], [base[slot] - 1, base[slot], base[slot] + 1])
                np.testing.assert_array_equal(features[base[slot]], features[changed[slot]])

    def test_reject_cross_window_interventions(self):
        for slot in (0, 63, 64, 127, 128, 255, 256, 511, 512):
            with self.assertRaises(ValueError):
                replace_neighbors(8192, slot)

    def test_counterfactual_earlier_tie_and_endpoint(self):
        full = np.arange(8, dtype=np.float32) / 8
        grid = np.arange(8, dtype=np.int64) * 16
        actual = restore_full_layout(full, np.asarray([0, 64]), grid)
        expected = np.repeat(np.asarray([0, 0, 0, .5, .5, .5, .5, .5], dtype=np.float32), 16)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(restore_full_layout(full, grid, grid), np.repeat(full, 16))

    def test_counterfactual_rejects_nonmember_coordinates(self):
        with self.assertRaises(ValueError):
            restore_full_layout(np.zeros(3, dtype=np.float32), np.asarray([1]), np.asarray([0, 16, 32]))

    def test_target_summary_distinguishes_video_and_frame_weighting(self):
        from scripts.probe_context_ver4 import summarize_targets
        rows = [{'video': name, 'branch': 1, 'gt': 0, 'near_minus_baseline': delta,
                 'repeat_minus_baseline': 0., 'abs_error_change_vs_full': delta}
                for name, delta in (('a', -1.), ('a', -1.), ('b', 1.))]
        result = summarize_targets(rows)
        normal = result[0]
        self.assertEqual(normal['n_targets'], 3)
        self.assertEqual(normal['n_videos'], 2)
        self.assertAlmostEqual(normal['mean_delta'], -1 / 3)
        self.assertAlmostEqual(normal['video_mean_delta'], 0.)
        self.assertAlmostEqual(normal['closer_to_full_fraction'], 2 / 3)
        self.assertEqual(result[1]['n_targets'], 0)
        self.assertIsNone(result[1]['mean_delta'])


if __name__ == '__main__':
    unittest.main()
