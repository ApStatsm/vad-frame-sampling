"""User-run manifest tests; synthetic files only, no video decode or model."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from vadfs.datasets.ucf import SPEC
from vadfs.preparation import identity_hash, index_files, unique_path, validate_gt


class PreparationTests(unittest.TestCase):
    def test_identity_is_portable_and_detects_content_changes(self):
        first = [{'filename': 'a.mp4', 'raw_path': '/one/a.mp4', 'sha256': 'content',
                  'reference_tokens': 2, 'evaluation_frames': 32, 'intervals_raw': []}]
        moved = deepcopy(first)
        moved[0]['raw_path'] = '/two/a.mp4'
        self.assertEqual(identity_hash(SPEC, first, 'gt'), identity_hash(SPEC, moved, 'gt'))
        moved[0]['sha256'] = 'other'
        self.assertNotEqual(identity_hash(SPEC, first, 'gt'), identity_hash(SPEC, moved, 'gt'))

    def test_gt_order_length_and_offsets(self):
        videos = [{'filename': 'anomaly.mp4', 'evaluation_frames': 16,
                   'is_anomaly': True, 'intervals_raw': [[2, 4]]},
                  {'filename': 'normal.mp4', 'evaluation_frames': 16,
                   'is_anomaly': False, 'intervals_raw': []}]
        gt = np.zeros(32, dtype=np.uint8)
        gt[2:4] = 1
        validate_gt(gt, videos)
        self.assertEqual(videos[1]['gt_offset'], 16)
        with self.assertRaises(ValueError):
            validate_gt(gt[:-1], videos)
        with self.assertRaises(ValueError):
            validate_gt(np.roll(gt, 16), videos)

    def test_duplicate_inputs_are_not_silently_chosen(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory in ('a', 'b'):
                (root / directory).mkdir()
                (root / directory / 'same.mp4').write_bytes(b'fixture')
            index = index_files(root, '.mp4', lambda path: path.name)
            with self.assertRaises(ValueError):
                unique_path(index, 'same.mp4')


if __name__ == '__main__':
    unittest.main()
