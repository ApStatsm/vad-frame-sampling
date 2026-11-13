"""User-run checks for fit-free selection and cache/model-input boundaries."""

from copy import deepcopy
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from vadfs.clip_selection import (PROJECTION_SPEC, fixed_projection, project_rows,
                                  select_cached_rows, validate_representation)
from vadfs.datasets import get_dataset
from vadfs.pipeline import load_cached_reference, validate_experiment
from scripts.run_optimal import experiment_cases


def clip_experiment():
    return {'input_type': 'raw', 'feature_source': 'raw_reference_cache',
            'model_seed': 234, 'reference_run': None,
            'sampling': {'method': 'd_optimal', 'budget': .25, 'seed': 0,
                         'representation': deepcopy(PROJECTION_SPEC)},
            'protocol': {'status': 'draft', 'restoration': 'nearest'}}


class ClipSelectionTests(unittest.TestCase):
    def test_fixed_projection_and_positive_qr_sign(self):
        q, record = fixed_projection()
        q2, record2 = fixed_projection()
        np.testing.assert_array_equal(q, q2)
        self.assertEqual(record['q_sha256'], record2['q_sha256'])
        self.assertFalse(record['data_fitted'])
        np.testing.assert_allclose(q.T @ q, np.eye(8), atol=1e-12)
        generator = np.random.Generator(np.random.PCG64(234))
        g = generator.standard_normal((512, 8))
        self.assertTrue(np.all(np.diag(q.T @ g) > 0))

    def test_projection_uses_fixed_columns_and_never_mutates_model_input(self):
        x = np.random.default_rng(9).normal(size=(9, 512)).astype(np.float16)
        saved = x.copy()
        q, _ = fixed_projection()
        z = project_rows(x, 3, q)
        normal = x.astype(np.float64)
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        np.testing.assert_allclose(z, normal @ (q[:, :3] * np.sqrt(512 / 3)), atol=1e-12)
        np.testing.assert_array_equal(x, saved)
        self.assertEqual(z.shape, (9, 3))
        for method in ('d_optimal', 'a_optimal', 'e_optimal', 'uniform', 'random'):
            with patch('numpy.linalg.svd', side_effect=AssertionError('SVD/fit forbidden')):
                selected, positions, detail = select_cached_rows(x, list(range(0, 144, 16)), method, .25, 4, q)
            self.assertEqual(selected.dtype, x.dtype)
            np.testing.assert_array_equal(selected, saved[positions])
            self.assertEqual(len(positions), 3)
            self.assertEqual(positions, sorted(set(positions)))
            if method.endswith('_optimal'):
                self.assertFalse(detail['rank_adaptation'])
                self.assertEqual(detail['dimension'], 3)

    def test_actual_full_budget_bypasses_projection_even_at_fractional_budget(self):
        x = np.ones((1, 512), dtype=np.float32)
        for method in ('d_optimal', 'a_optimal', 'e_optimal'):
            with patch('vadfs.clip_selection.project_rows', side_effect=AssertionError('Bypass must not project')):
                result, positions, detail = select_cached_rows(x, [0], method, .125, 0)
            np.testing.assert_array_equal(result, x)
            self.assertEqual(positions, [0])
            self.assertTrue(detail['full_budget_bypass'])

    def test_zero_norm_and_nonfinite_selection_inputs_fail(self):
        q, _ = fixed_projection()
        for x in (np.zeros((3, 512)), np.full((3, 512), np.nan)):
            with self.assertRaises(ValueError):
                select_cached_rows(x, [0, 16, 32], 'd_optimal', .5, 0, q)

    def test_rank_deficiency_does_not_change_projection_dimension(self):
        from vadfs.samplers.optimal import select
        rows = np.ones((12, 8))
        with patch('numpy.linalg.svd', side_effect=AssertionError('No SVD')):
            _, detail = select(rows, 8, 'e_optimal', .625, rank_adapt=False)
        self.assertEqual(detail['dimension'], 8)
        self.assertEqual(detail['rank'], 1)
        self.assertTrue(detail['minimum_at_ridge'])

    def test_cached_optimal_is_explicit_and_cannot_fall_back_to_raw(self):
        spec = get_dataset('ucf').SPEC
        experiment = clip_experiment()
        validate_experiment(experiment, spec, 'full', cached_features=True)
        with self.assertRaises(ValueError):
            validate_experiment(experiment, spec, 'full')
        experiment['sampling'].pop('representation')
        with self.assertRaises(ValueError):
            validate_experiment(experiment, spec, 'full', cached_features=True)
        bad = deepcopy(PROJECTION_SPEC)
        bad['seed'] = 999
        with self.assertRaises(ValueError):
            validate_representation(bad)

    def test_full_cache_fps_dtype_and_coordinates_are_checked(self):
        x = np.ones((3, 512), dtype=np.float16)
        grid = [0, 16, 32]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'cache.npz'
            np.savez(path, features=x, selected_frame_indices=grid,
                     selected_timestamps_seconds=np.asarray(grid) / 25.)
            result, _ = load_cached_reference(path, grid, x.dtype, expected_fps=25.)
            np.testing.assert_array_equal(result, x)
            with self.assertRaisesRegex(ValueError, 'timestamps'):
                load_cached_reference(path, grid, x.dtype, expected_fps=30.)
            with self.assertRaises(ValueError):
                load_cached_reference(path, grid, np.dtype('float32'))

    def test_default_sweep_has_37_runs_and_five_random_seeds_per_budget(self):
        for dataset in ('ucf', 'xd'):
            cases = experiment_cases(dataset, ['reference', 'uniform', 'random', 'd', 'a', 'e'],
                                     ['100', '50', '25', '12p5'])
            self.assertEqual(len(cases), 37)
            self.assertTrue(all(case['config'].is_file() for case in cases))
            for budget in ('100', '50', '25', '12p5'):
                self.assertEqual({case['seed'] for case in cases if case['method'] == 'random'
                                  and case['budget'] == budget}, set(range(5)))

    def test_pipeline_cache_path_sends_original_rows_without_decoder_or_pca(self):
        from vadfs.pipeline import run_experiment
        from vadfs import vadclip
        import vadfs.extract

        x = np.random.default_rng(23).normal(size=(12, 512)).astype(np.float16)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache = root / 'reference'
            (cache / 'features').mkdir(parents=True)
            metadata = root / 'metadata.json'
            metadata.write_text('{}')
            grid = np.arange(12) * 16
            np.savez(cache / 'features/000_test.npz', features=x, selected_frame_indices=grid,
                     selected_timestamps_seconds=grid / 25.)
            video = {'filename': 'test.mp4', 'official_csv_index': 0, 'reference_tokens': 12,
                     'feature_dtype': 'float16', 'evaluation_frames': 192, 'gt_offset': 0,
                     'sha256': 'rawhash', 'intervals_raw': [], 'is_anomaly': False}
            manifest = {'videos': [video], 'setup_id': 'synthetic',
                        'dataset_identity_sha256': 'data', 'frame_gt_sha256': 'gt'}
            source = {'run_id': 'raw-reference', 'loaded_summary_sha256': 'refhash',
                      'videos': [{'video': 'test.mp4', 'official_csv_index': 0, 'reference_tokens': 12,
                                  'selected_tokens': 12, 'evaluation_frames': 192, 'fps': 25.}]}
            recipe = {'temporal_mode': 'single_frame', 'frame_offset': 0, 'feature_storage_dtype': 'official'}
            dataset = {'key': 'ucf', 'extraction': recipe}
            resources = {key: str(metadata) for key in ('clip_checkpoint', 'vadclip_checkpoint')}
            resources['output_root'] = str(root / 'outputs')
            for budget in (1., .25):
                experiment = clip_experiment()
                experiment['sampling']['budget'] = budget
                calls = []
                def inference(model, feature, *args):
                    calls.append(feature.copy())
                    k = len(feature)
                    return np.zeros(k), np.zeros(k), np.zeros((k, 14)), {}
                output = root / ('run_' + str(budget))
                for subdir in ('scores', 'features'):
                    (output / subdir).mkdir(parents=True)
                with ExitStack() as stack:
                    replacements = {
                        'vadfs.pipeline.load_experiment': (experiment, dataset, resources, metadata, metadata),
                        'vadfs.pipeline.create_run_dir': ('test-run', output),
                        'vadfs.pipeline.load_prepared': (metadata, manifest),
                        'vadfs.pipeline.environment': {'ffmpeg': None},
                        'vadfs.pipeline.check_performance_reference': deepcopy(source),
                        'vadfs.vadclip.default_options': SimpleNamespace(visual_length=256),
                        'vadfs.vadclip.load_model': object(),
                    }
                    for target, value in replacements.items():
                        stack.enter_context(patch(target, return_value=value))
                    for target in ('vadfs.pipeline.record_provenance', 'vadfs.vadclip.check_device',
                                   'vadfs.vadclip.synchronize', 'vadfs.extract.validate_extraction'):
                        stack.enter_context(patch(target))
                    stack.enter_context(patch.object(vadclip, 'MODEL_OPTION_KEYS', ('visual_length',)))
                    stack.enter_context(patch('vadfs.vadclip.infer_feature', side_effect=inference))
                    for target in ('vadfs.extract.extract_reference', 'vadfs.sampling_features.collect',
                                   'vadfs.sampling_features.load_artifact', 'numpy.linalg.svd'):
                        stack.enter_context(patch(target, side_effect=AssertionError('Forbidden cached-path operation')))
                    code = run_experiment(metadata, mode='smoke', requested=['test.mp4'],
                                          cached_reference_override=cache)
                self.assertEqual(code, 0)
                report = json.loads((output / 'summary.json').read_text())
                self.assertFalse(report['speedup_eligible'])
                self.assertEqual(report['encoder_input_images'], 0)
                with np.load(output / 'scores/000_test.npz') as scores:
                    positions = scores['selected_row_positions']
                    np.testing.assert_array_equal(calls[0], x[positions])
                    self.assertEqual(calls[0].dtype, x.dtype)
                    self.assertEqual(len(scores['reference_branch1_scores']), 12)
                self.assertEqual(report['videos'][0]['evaluation_frames'], 192)


if __name__ == '__main__':
    unittest.main()
