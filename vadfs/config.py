"""Portable defaults and strictly scoped, ignored machine-local overrides."""

from copy import deepcopy
import json
from pathlib import Path

from .paths import project_path


LOCAL_KEYS = {'videos_root', 'features_root', 'clip_checkpoint', 'vadclip_checkpoint',
              'output_root', 'setup_manifest', 'device'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_experiment(path, local_path=None):
    experiment_path = project_path(path)
    experiment = read_json(experiment_path)
    dataset_path = project_path(experiment['dataset_config'])
    dataset = read_json(dataset_path)
    resources = deepcopy(dataset['resources'])
    if local_path:
        local = read_json(project_path(local_path))
        unknown = set(local) - LOCAL_KEYS
        if unknown:
            raise ValueError(f'Local settings cannot change experiment rules: {sorted(unknown)}')
        resources.update(local)
    return experiment, dataset, resources, experiment_path, dataset_path


def resource_path(resources, name):
    value = resources.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f'Configure {name} in the dataset or ignored local JSON')
    return project_path(value)
