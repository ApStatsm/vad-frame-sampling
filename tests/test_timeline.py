"""User-run alignment checks: sparse observations, endpoints, ties and full budget."""

import unittest

import numpy as np

from vadfs.timeline import candidates, restore_tokens, to_frames


class TimelineTests(unittest.TestCase):
    def test_full_preserves_original_repeat_for_every_method(self):
        grid = candidates(5)
        scores = np.asarray([.1, .8, .3, .4, .9], dtype=np.float32)
        for method in ('nearest', 'linear', 'previous'):
            restored = restore_tokens(scores, grid, grid, method)
            np.testing.assert_array_equal(to_frames(restored), np.repeat(scores, 16))
            self.assertEqual(restored.dtype, scores.dtype)

    def test_single_observation_covers_fixed_full_timeline(self):
        for method in ('nearest', 'linear', 'previous'):
            restored = restore_tokens(np.asarray([.7], dtype=np.float32), [64], candidates(9), method)
            self.assertEqual(to_frames(restored).shape, (144,))
            np.testing.assert_array_equal(restored, np.full(9, .7, dtype=np.float32))

    def test_ties_and_endpoint_extension(self):
        scores = np.asarray([0., 1.], dtype=np.float32)
        grid = candidates(5)
        np.testing.assert_array_equal(restore_tokens(scores, [16, 48], grid, 'nearest'), [0, 0, 0, 1, 1])
        np.testing.assert_array_equal(restore_tokens(scores, [16, 48], grid, 'previous'), [0, 0, 0, 1, 1])
        np.testing.assert_array_equal(restore_tokens(scores, [16, 48], grid, 'linear'), [0, 0, .5, 1, 1])

    def test_class_normalization_and_tail_block_length(self):
        scores = np.asarray([[.2, .8], [.8, .2]], dtype=np.float32)
        restored = restore_tokens(scores, [0, 32], candidates(3), 'linear')
        np.testing.assert_allclose(restored.sum(axis=1), 1, atol=1e-6)
        self.assertEqual(to_frames(restored).shape, (48, 2))

    def test_invalid_coordinates_nonfinite_and_shape(self):
        for selected in ([1], [0, 0], [32, 16], [80]):
            with self.assertRaises(ValueError):
                restore_tokens(np.ones(len(selected), dtype=np.float32), selected, candidates(4))
        with self.assertRaises(ValueError):
            restore_tokens(np.asarray([float('nan')]), [0], candidates(4))
        with self.assertRaises(ValueError):
            restore_tokens(np.ones(2, dtype=np.float32), [0], candidates(4))


if __name__ == '__main__':
    unittest.main()
