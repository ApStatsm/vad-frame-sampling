"""UCF labels, audited reference lengths and the original evaluation protocol."""

import csv
from pathlib import Path

from .base import DatasetSpec
from ..config import read_json
from ..paths import project_path


LABELS = ('Normal', 'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary',
          'Explosion', 'Fighting', 'RoadAccidents', 'Robbery', 'Shooting',
          'Shoplifting', 'Stealing', 'Vandalism')
SPEC = DatasetSpec('ucf', 'UCF-Crime', LABELS, 290, 140, 'ucf_option', 'branch1', 'roc_auc')


def template(config):
    manifest = read_json(project_path(config['reference_manifest']))
    with project_path(config['official_list']).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 290 or len(manifest['videos']) != 290:
        raise ValueError('Expected 290 official UCF test entries')
    for index, (video, row) in enumerate(zip(manifest['videos'], rows)):
        feature = Path(row['path'])
        if (video['official_csv_index'] != index or video['category'] != row['label']
                or video['official_feature_filename'] != feature.name):
            raise ValueError('Portable UCF metadata and upstream list disagree')
        video['feature_parent'] = feature.parent.name
    return manifest
