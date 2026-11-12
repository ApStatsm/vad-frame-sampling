"""Synthetic XD Ano-AUC scope and ordering checks."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from vadfs.datasets.xd import SPEC
from vadfs.metrics import compute_xd_ano_auc


class XDAnoAUCTests(unittest.TestCase):
    def test_ano_auc_excludes_only_normal_only_videos(self):
        declared = [
            {'filename': name, 'official_csv_index': index, 'is_anomaly': label != 'A', 'category': label,
             'reference_tokens': 1, 'evaluation_frames': 16, 'gt_offset': 16 * index}
            for index, (name, label) in enumerate((('normal.mp4', 'A'), ('anomaly.mp4', 'B1-0-0')))
        ]
        records = [{**video, 'video': video['filename'], 'case': 'full_video', 'prefix_only': False}
                   for video in declared]
        manifest = {'dataset': 'xd', 'videos': declared}
        gt = np.repeat([0, 0, 1], [16, 8, 8])
        branch1 = np.repeat([0.9, 0.2, 0.7], [16, 8, 8])
        branch2 = np.repeat([0.1, 0.8, 0.3], [16, 8, 8])
        with patch('vadfs.metrics.xd.SPEC', replace(SPEC, expected_videos=2, expected_anomaly_videos=1)):
            scope, aucs = compute_xd_ano_auc(gt, branch1, branch2, records, manifest)
            self.assertEqual(scope['anomaly_evaluation_frames'], 16)
            self.assertEqual(aucs, {'branch1': 1.0, 'branch2': 0.0})
            with self.assertRaisesRegex(ValueError, 'order, scope or timeline'):
                compute_xd_ano_auc(gt, branch1, branch2, records[::-1], manifest)
