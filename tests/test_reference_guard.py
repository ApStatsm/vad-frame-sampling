"""User-run guards against mixing incomplete/different-environment baselines."""

import json
from pathlib import Path
import tempfile
import unittest

from vadfs.pipeline import check_reference


class ReferenceGuardTests(unittest.TestCase):
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
