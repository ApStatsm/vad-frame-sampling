"""XD official-feature adapter. Raw extraction recipe still requires validation."""

import csv
from pathlib import Path

from .base import DatasetSpec
from ..paths import project_path


PROMPTS = ('normal', 'fighting', 'shooting', 'riot', 'abuse', 'car accident', 'explosion')
SPEC = DatasetSpec('xd', 'XD-Violence', PROMPTS, 800, 500, 'xd_option', 'branch2', 'ap')


def template(config):
    intervals = {}
    for line in project_path(config['annotation_path']).read_text().splitlines():
        if not line.strip():
            continue
        name, *numbers = line.split()
        if len(numbers) % 2 or name in intervals:
            raise ValueError('Malformed or duplicate XD annotation')
        pairs = [[int(numbers[i]), int(numbers[i + 1])] for i in range(0, len(numbers), 2)]
        if any(start < 0 or end <= start for start, end in pairs):
            raise ValueError('Invalid XD annotation interval')
        intervals[name.removesuffix('.mp4')] = pairs
    with project_path(config['official_list']).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    videos = []
    for index, row in enumerate(rows):
        feature = Path(row['path'])
        if not feature.name.endswith('__0.npy'):
            raise ValueError('XD adapter expects the upstream __0 feature list')
        stem = feature.name.removesuffix('__0.npy')
        anomaly = row['label'] != 'A'
        if anomaly and stem not in intervals:
            raise ValueError(f'XD anomaly annotation missing: {stem}')
        videos.append({'official_csv_index': index, 'filename': stem + '.mp4',
                       'category': row['label'], 'is_anomaly': anomaly,
                       'intervals_raw': intervals.get(stem, []),
                       'official_feature_filename': feature.name,
                       'feature_parent': feature.parent.name})
    return {'schema_version': 1, 'dataset': 'xd', 'videos': videos,
            'annotation_convention': 'Upstream published intervals; validate against the bundled frame GT during setup.',
            'raw_recipe_status': 'unverified'}
