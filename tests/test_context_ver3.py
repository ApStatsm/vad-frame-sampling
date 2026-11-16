"""User-run protocol invariants; no model/data dependencies."""

import unittest
from vadfs.samplers.context_ver3 import METHODS_VER3, select_ver3


class ContextVer3Test(unittest.TestCase):
    def test_budget_endpoints_and_tail(self):
        for n in (1, 7, 8, 9, 49, 56, 57, 64, 65, 127, 128, 129, 1025):
            base = list(range(0, n, 8))
            for method in METHODS_VER3:
                rows, detail = select_ver3(n, method, 234)
                self.assertEqual(len(rows), (n + 7) // 8)
                self.assertEqual(rows, sorted(set(rows)))
                self.assertLess(rows[-1], n)
                self.assertEqual((rows[0], rows[-1]), (base[0], base[-1]))
                self.assertEqual(len(set(rows) - set(base)), detail['changed_slots'])
                tail = len(base) // 8 * 8
                self.assertEqual(rows[tail:], base[tail:])

    def test_pair_and_relocation_support(self):
        paired, _ = select_ver3(64, 'paired_ver3', 0)
        self.assertEqual(paired, [0, 8, 16, 27, 28, 40, 48, 56])
        for seed in range(5):
            rows, _ = select_ver3(64, 'relocated_ver3', seed)
            self.assertTrue(24 < rows[3] < rows[4] < 32)
            self.assertGreaterEqual(rows[4] - rows[3], 2)
            self.assertEqual(rows, select_ver3(64, 'relocated_ver3', seed)[0])

    def test_invalid_input(self):
        for n, method, seed in ((0, 'paired_ver3', 0), (8, 'bad', 0), (8, 'paired_ver3', -1)):
            with self.assertRaises(ValueError):
                select_ver3(n, method, seed)

    def test_cached_rows_are_original(self):
        import numpy as np
        from vadfs.clip_selection import select_cached_rows
        features = np.arange(64 * 512, dtype=np.float32).reshape(64, 512)
        grid = list(range(0, 64 * 16, 16))
        for method in METHODS_VER3:
            chosen, positions, _ = select_cached_rows(features, grid, method, 0.125, 0)
            np.testing.assert_array_equal(chosen, features[positions])
            self.assertEqual(chosen.dtype, features.dtype)
        with self.assertRaises(ValueError):
            select_cached_rows(features, grid, 'paired_ver3', 0.25, 0)

    def test_raw_ver3_rejected(self):
        from types import SimpleNamespace
        from vadfs.pipeline import validate_experiment
        experiment = {'input_type': 'raw', 'feature_source': 'raw_reference_cache',
                      'model_seed': 234, 'sampling': {'method': 'paired_ver3', 'budget': 0.125, 'seed': 0},
                      'protocol': {'restoration': 'nearest', 'status': 'draft'}}
        with self.assertRaises(ValueError):
            validate_experiment(experiment, SimpleNamespace(key='xd'), 'full', cached_features=False)
        validate_experiment(experiment, SimpleNamespace(key='xd'), 'full', cached_features=True)


if __name__ == '__main__':
    unittest.main()
