"""User-run guards against mixing incomplete/different-environment baselines."""

import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from copy import deepcopy

import numpy as np

from vadfs.pipeline import check_reference, check_performance_reference, load_cached_selection


class ReferenceGuardTests(unittest.TestCase):
    def test_old_evaluator_is_cache_provenance_only_and_model_differences_still_fail(self):
        spec = SimpleNamespace(key='ucf', prompts=('normal',), primary_branch='branch1', primary_metric='roc_auc')
        inputs = {name: name for name in ('clip_checkpoint', 'vadclip_checkpoint', 'gt', 'segment_gt',
                  'class_gt', 'fine_grained_source', 'official_list', 'tokenizer_vocabulary')}
        modules = ('vadfs/vadclip.py', 'vadfs/evaluation.py', 'vadfs/metrics.py', 'vadfs/timeline.py', 'vadfs/color.py')
        report = {'input_sha256': inputs, 'package_sha256': dict.fromkeys(modules, 'current'),
                  'vendored_source_sha256': {'model': 'same'},
                  'experiment': {'model_seed': 234, 'protocol': {'restoration': 'nearest'}}}
        baseline = {'input_sha256': dict(inputs), 'dataset_key': 'ucf',
                    'dataset_identity_sha256': 'data', 'extraction_config': {}, 'prompts': ['normal'],
                    'vendored_source_sha256': {'model': 'same'}, 'experiment': report['experiment'],
                    'metrics': {'primary': {'branch': 'branch1', 'metric': 'roc_auc'}},
                    'package_sha256': dict.fromkeys(modules, 'current')}
        baseline['package_sha256']['vadfs/evaluation.py'] = 'historical'
        manifest = {'dataset_identity_sha256': 'data'}
        with patch('vadfs.pipeline.load_full_raw_reference', side_effect=lambda *args: deepcopy(baseline)):
            with self.assertRaisesRegex(ValueError, 'implementation differs'):
                check_performance_reference(Path('.'), manifest, report, spec, {})
            accepted = check_performance_reference(Path('.'), manifest, report, spec, {}, cache_only=True)
            self.assertIn('vadfs/evaluation.py', accepted['cache_source_evaluation_differences'])
            baseline['package_sha256']['vadfs/vadclip.py'] = 'different-model'
            with self.assertRaisesRegex(ValueError, 'implementation differs'):
                check_performance_reference(Path('.'), manifest, report, spec, {}, cache_only=True)

    def test_cached_rows_preserve_selected_indices_and_timestamps(self):
        features = np.arange(3 * 512, dtype=np.float32).reshape(3, 512)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'features.npz'
            np.savez(path, features=features, selected_frame_indices=[0, 16, 32],
                     selected_timestamps_seconds=[0.0, 0.5, 1.0])
            selected, arrays = load_cached_selection(path, [0, 16, 32], [0, 32], np.dtype('float32'))
            np.testing.assert_array_equal(selected, features[[0, 2]])
            np.testing.assert_array_equal(arrays['selected_frame_indices'], [0, 32])
            np.testing.assert_array_equal(arrays['selected_timestamps_seconds'], [0.0, 1.0])
            with self.assertRaisesRegex(ValueError, 'full reference grid'):
                load_cached_selection(path, [1, 17, 33], [1, 33], np.dtype('float32'))

    def test_only_full_matching_raw_reference_is_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            summary = {'schema_version': 2, 'success': True, 'mode': 'full', 'input_type': 'raw',
                       'sampling': {'budget': 1}, 'videos': [{}, {}], 'metrics': {'primary': {}},
                       'comparison_signature': 'same'}
            path = Path(folder) / 'summary.json'
            path.write_text(json.dumps(summary))
            experiment = {'reference_run': folder}
            manifest = {'videos': [{}, {}]}
            self.assertIsNotNone(check_reference(experiment, manifest, 'same'))
            with self.assertRaises(ValueError):
                check_reference(experiment, manifest, 'different')
            summary['mode'] = 'smoke'
            path.write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                check_reference(experiment, manifest, 'same')


if __name__ == '__main__':
    unittest.main()
