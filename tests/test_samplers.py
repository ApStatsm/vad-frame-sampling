"""User-run boundary tests; not executed during authoring."""

import math
import unittest

from vadfs.samplers import budget_count, choose, video_seed
from vadfs.samplers.random_window import select_frames


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

    def test_uniform_starts_at_zero_and_keeps_stride_on_incomplete_tails(self):
        frames = [index * 16 for index in range(17)]
        for ratio, step in ((.5, 2), (.25, 4), (.125, 8)):
            with self.subTest(ratio=ratio):
                self.assertEqual(choose(frames, 'uniform', ratio, 0), frames[::step])
        self.assertEqual(choose([0, 16, 32], 'uniform', .125, 0), [0])

    def test_local_random_reproducibility(self):
        seed = video_seed(3, 'ucf', 'example.mp4')
        self.assertEqual(seed, video_seed(3, 'ucf', 'example.mp4'))
        self.assertNotEqual(seed, video_seed(3, 'xd', 'example.mp4'))
        self.assertEqual(choose(list(range(100)), 'random', .25, seed),
                         choose(list(range(100)), 'random', .25, seed))

    def test_random_windows_preserve_budget_and_clip_the_last_window(self):
        grid = [0, 16, 32]
        for ratio, width in ((1, 16), (.5, 32), (.25, 64), (.125, 128)):
            anchors = choose(grid, 'random_window', ratio, 3)
            selected = select_frames(anchors, width, 33, len(grid), 3)
            self.assertEqual(len(selected), budget_count(len(grid), ratio))
            self.assertEqual(selected, select_frames(anchors, width, 33, len(grid), 3))
            self.assertTrue(all(start <= frame < min(start + width, 33)
                                for start, frame in zip(anchors, selected)))
        self.assertEqual(select_frames([0, 32], 32, 33, 3, 3)[-1], 32)
        self.assertTrue(all(frame < 32 for frame in select_frames([0, 16], 16, 33, 2, 3)))

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
