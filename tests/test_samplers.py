"""User-run boundary tests; not executed during authoring."""

import math
import unittest

from vadfs.samplers import budget_count, choose, video_seed


class SamplerContractTests(unittest.TestCase):
    def test_rounding_uniqueness_order_and_candidate_membership(self):
        for count in (1, 2, 3, 7, 16, 31, 257):
            frames = [index * 16 for index in range(count)]
            for ratio in (1, .5, .25, .125):
                for method in ('uniform', 'random'):
                    with self.subTest(count=count, ratio=ratio, method=method):
                        selected = choose(frames, method, ratio, 7)
                        self.assertEqual(len(selected), min(count, max(1, math.ceil(count * ratio))))
                        self.assertEqual(selected, sorted(set(selected)))
                        self.assertTrue(set(selected) <= set(frames))

    def test_all_full_budget_selections_are_identical(self):
        frames = [0, 16, 32, 48, 64]
        for method in ('reference', 'uniform', 'random'):
            self.assertEqual(choose(frames, method, 1, 42), frames)

    def test_midpoint_and_local_random_reproducibility(self):
        self.assertEqual(choose([0, 16, 32], 'uniform', .125, 0), [16])
        seed = video_seed(3, 'ucf', 'example.mp4')
        self.assertEqual(seed, video_seed(3, 'ucf', 'example.mp4'))
        self.assertNotEqual(seed, video_seed(3, 'xd', 'example.mp4'))
        self.assertEqual(choose(list(range(100)), 'random', .25, seed),
                         choose(list(range(100)), 'random', .25, seed))

    def test_invalid_budgets_and_coordinates_are_rejected(self):
        for ratio in (0, -1, 1.1, float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                budget_count(10, ratio)
        for frames in ([], [0, 0], [16, 0], [-1, 0]):
            with self.assertRaises(ValueError):
                choose(frames, 'uniform', .25, 0)
        with self.assertRaises(ValueError):
            choose([0, 16], 'reference', .5, 0)


if __name__ == '__main__':
    unittest.main()
