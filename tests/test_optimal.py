"""User-run objective checks; these tests were not executed during authoring."""

import unittest
import json
from pathlib import Path
import tempfile

import numpy as np

from vadfs.samplers.optimal import gains, objectives, select
from vadfs.sampling_features import FEATURE_SPEC, feature_row, fit_transform, load_artifact, transform
from vadfs.paths import package_hashes


class OptimalTests(unittest.TestCase):
    def test_marginal_gains_equal_direct_objective_differences(self):
        rows = np.array([[1., 2.], [-2., 1.], [0., 0.]])
        matrix = np.array([[3., .2], [.2, 2.]])
        old = objectives(matrix)
        for method, objective, sign in [('d_optimal', 'logdet', 1), ('a_optimal', 'inverse_trace', -1)]:
            actual = gains(matrix, rows, method)
            expected = [sign * (objectives(matrix + np.outer(row, row))[objective] - old[objective]) for row in rows]
            np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_greedy_choices_equal_small_brute_force(self):
        rows = np.random.default_rng(17).normal(size=(12, 8))
        for method, objective, sign in [('d_optimal', 'logdet', 1), ('a_optimal', 'inverse_trace', -1)]:
            selected, detail = select(rows, 8, method, .625)
            chosen = []
            for _ in range(8):
                best = max((i for i in range(12) if i not in chosen), key=lambda i: sign * objectives(
                    np.eye(8) + rows[chosen + [i]].T @ rows[chosen + [i]])[objective])
                chosen.append(best)
            self.assertEqual(selected, sorted(chosen))
            self.assertIsNone(detail['fallback'])

    def test_zero_rank_and_full_budget(self):
        for method in ('d_optimal', 'a_optimal', 'e_optimal'):
            selected, detail = select(np.zeros((5, 8)), 3, method, .5)
            self.assertEqual(selected, [0, 2, 4])
            self.assertEqual(detail['fallback'], 'zero_rank_uniform')
            selected, detail = select(np.zeros((5, 8)), 5, method, 1.)
            self.assertEqual(selected, list(range(5)))
            self.assertTrue(detail['full_budget_bypass'])

    def test_e_budget_bounds_and_improvement(self):
        rows = np.random.default_rng(3).normal(size=(15, 8))
        chosen, detail = select(rows, 8, 'e_optimal', .5)
        self.assertEqual(len(set(chosen)), 8)
        self.assertEqual(chosen, sorted(chosen))
        self.assertLessEqual(detail['evaluated_pairs'], 40960)
        self.assertGreaterEqual(detail['final_objectives']['minimum_eigenvalue'],
                                detail['initial_objectives']['minimum_eigenvalue'] - 1e-8)

    def test_rank_one_reduction_and_ties(self):
        rows = np.ones((9, 8))
        for method in ('d_optimal', 'a_optimal', 'e_optimal'):
            chosen, detail = select(rows, 1, method, .1)
            self.assertEqual(chosen, [0])
            self.assertEqual(detail['dimension'], 1)

    def test_invalid_rows_rejected(self):
        with self.assertRaises(ValueError):
            select(np.full((4, 8), np.nan), 2, 'd_optimal', .5)

    def test_e_one_round_matches_best_exhaustive_exchange(self):
        rows = np.random.default_rng(51).normal(size=(12, 8))
        chosen, _ = select(rows, 8, 'e_optimal', .625, {'e_max_rounds': 1})
        initial = [0, 1, 3, 4, 6, 8, 9, 11]
        # This is the same fixed first-candidate Uniform rule at b=.625.
        candidates = [initial]
        for remove in initial:
            for insert in sorted(set(range(12)) - set(initial)):
                candidates.append(sorted((set(initial) - {remove}) | {insert}))
        def minimum(indices):
            return np.linalg.eigvalsh(np.eye(8) + rows[indices].T @ rows[indices])[0]
        best = max(minimum(indices) for indices in candidates)
        self.assertAlmostEqual(minimum(chosen), best, places=9)

    def test_feature_layout_and_first_motion_zero(self):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[:8, :8] = 255
        row, gray = feature_row(image)
        self.assertAlmostEqual(row[0], 1.)
        np.testing.assert_allclose(row[1:], 0., atol=1e-15)
        second, _ = feature_row(np.zeros_like(image), gray)
        self.assertAlmostEqual(second[64], .25)
        np.testing.assert_allclose(second[65:], 0.)

    def test_fit_excludes_constants_and_does_not_whiten(self):
        rows = np.random.default_rng(4).normal(size=(100, 80))
        rows[:, 0] = 3.
        artifact = fit_transform(rows)
        self.assertNotIn(0, artifact['active'])
        z = transform(rows, artifact)
        self.assertEqual(z.shape, (100, 8))
        np.testing.assert_allclose(z.mean(axis=0), 0., atol=1e-12)
        self.assertFalse(np.allclose(z.std(axis=0), 1.))

    def test_artifact_rejects_test_name_hash_or_recipe_overlap(self):
        arrays = fit_transform(np.random.default_rng(4).normal(size=(100, 80)))
        hashes = package_hashes()
        recipe = {'color_conversion': 'opencv_bgr'}
        metadata = {'schema_version': 1, 'dataset_key': 'xd', 'split': 'development',
                    'feature_spec': FEATURE_SPEC, 'extraction_config': recipe,
                    'feature_source_sha256': hashes['vadfs/sampling_features.py'],
                    'color_source_sha256': hashes['vadfs/color.py'],
                    'training_videos': [{'filename': 'development.mp4', 'sha256': 'devhash'}]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'artifact.npz'
            np.savez(path, **arrays, metadata=np.asarray(json.dumps(metadata)))
            load_artifact(path, 'xd', recipe, [{'filename': 'test.mp4', 'sha256': 'testhash'}])
            for test in ({'filename': 'development.mp4', 'sha256': 'testhash'},
                         {'filename': 'renamed.mp4', 'sha256': 'devhash'}):
                with self.assertRaisesRegex(ValueError, 'overlap'):
                    load_artifact(path, 'xd', recipe, [test])
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                load_artifact(path, 'ucf', recipe, [])


if __name__ == '__main__':
    unittest.main()
