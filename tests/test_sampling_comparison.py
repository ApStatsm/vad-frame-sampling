"""User-run comparison provenance/seed/budget checks; no model execution."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.analyze.compare_sampling import main


def report(method, seed=0):
    return {'schema_version': 2, 'success': True, 'mode': 'full', 'input_type': 'cached_raw_features',
            'dataset': 'UCF-Crime', 'dataset_key': 'ucf', 'dataset_identity_sha256': 'data',
            'input_sha256': {}, 'package_sha256': {}, 'vendored_source_sha256': {},
            'extraction_config': {}, 'prompts': ['normal'],
            'experiment': {'model_seed': 234, 'protocol': {'restoration': 'nearest', 'status': 'draft'}},
            'sampling': {'method': method, 'seed': seed, 'budget': .25,
                         'representation': {'kind': 'clip_fixed_projection'}},
            'metrics': {'primary': {'branch': 'branch1', 'metric': 'roc_auc', 'value': .8},
                        'branch1': {'roc_auc': .8, 'ap': .4, 'ano_auc': .7}},
            'videos': [{'video': 'a.mp4', 'reference_tokens': 4, 'selected_tokens': 1, 'evaluation_frames': 64}],
            'comparison_signature': 'comparison', 'sampler_signature': method,
            'projection': {'q_sha256': 'same-q'}, 'cache_identity_sha256': 'same-cache'}


class ComparisonTests(unittest.TestCase):
    def invoke(self, reports, root):
        inputs = []
        for index, item in enumerate(reports):
            path = root / f'input_{index}.json'
            path.write_text(json.dumps(item))
            inputs.append(str(path))
        with patch('sys.argv', ['compare_sampling.py', *inputs, '--output-root', str(root / 'output')]):
            main()

    def test_different_q_cache_representation_or_actual_k_are_rejected(self):
        for change in ('q', 'cache', 'representation', 'count'):
            with tempfile.TemporaryDirectory() as folder:
                first, second = report('d_optimal'), report('a_optimal')
                if change == 'q':
                    second['projection']['q_sha256'] = 'different'
                elif change == 'cache':
                    second['cache_identity_sha256'] = 'different'
                elif change == 'representation':
                    second['sampling']['representation']['kind'] = 'cheap_feature_pca'
                else:
                    second['videos'][0]['selected_tokens'] = 2
                with self.assertRaises(ValueError):
                    self.invoke([first, second], Path(folder))

    def test_random_feature_seed_statistics_and_uniform_delta(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            uniform = report('uniform')
            uniform.pop('projection')
            uniform['sampling'].pop('representation')
            items = [uniform]
            for seed in range(5):
                item = report('random', seed)
                item['sampling'].pop('representation')
                item.pop('projection')
                item['metrics']['primary']['value'] = .8 + .001 * seed
                items.append(item)
            self.invoke(items, root)
            summary = json.loads(next((root / 'output').glob('*/summary.json')).read_text())
            random_row = next(row for row in summary['rows'] if row['method'] == 'random_feature')
            self.assertEqual(random_row['n_seeds'], 5)
            self.assertAlmostEqual(random_row['primary_mean_percent'], 80.2)
            self.assertAlmostEqual(random_row['uniform_delta_pp'], .2)
            self.assertIsNotNone(random_row['primary_std_pp'])
            self.assertEqual(summary['scope'], 'performance_only; no speedup comparison')


if __name__ == '__main__':
    unittest.main()
