"""User-run, immutable input manifests; no model execution or video decoding.

UCF may be prepared from exactly the 290 test MP4s plus the checked-in portable
reference metadata. Original ZIPs are not required on this path. The legacy
ZIP/CRC preparation script is retained for the first archive verification.
"""

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .config import resource_path
from .datasets import get_dataset
from .datasets.base import validate_manifest
from .paths import ROOT, project_path, sha256
from .runs import create_run_dir, now, write_json


def index_files(root, suffix, key):
    found = defaultdict(set)
    if not root.is_dir():
        raise FileNotFoundError(f'Input directory does not exist: {root}')
    for path in root.rglob(f'*{suffix}'):
        if path.is_file():
            found[key(path)].add(path.resolve())
    return found


def unique_path(index, key):
    paths = index.get(key, set())
    if len(paths) != 1:
        raise ValueError(f'Expected one input for {key}, found {len(paths)}')
    return next(iter(paths))


def validate_gt(gt, videos):
    import numpy as np

    expected_length = sum(video['evaluation_frames'] for video in videos)
    if gt.ndim != 1 or len(gt) != expected_length or gt.dtype.kind not in 'biuf':
        raise ValueError('Bundled frame GT and the ordered reference lengths do not align')
    if any(not np.isin(gt[start:start + 65536], [0, 1]).all()
           for start in range(0, len(gt), 65536)):
        raise ValueError('Frame GT must contain only binary values')
    offset = 0
    for video in videos:
        length = video['evaluation_frames']
        expected = np.zeros(length, dtype=np.uint8)
        for start, end in video['intervals_raw']:
            if start < 0 or end <= start:
                raise ValueError('Invalid published interval')
            expected[start:end] = 1  # upstream slicing, including its tail clipping
        if not np.array_equal(expected, gt[offset:offset + length]):
            raise ValueError(f'Annotation/order mismatch against bundled GT: {video["filename"]}')
        if not video['is_anomaly'] and np.any(expected):
            raise ValueError('Normal video contains positive annotation')
        video['gt_offset'] = offset
        video['annotation_beyond_evaluation_timeline'] = [
            pair for pair in video['intervals_raw'] if pair[1] > length]
        offset += length


def identity_hash(spec, videos, gt_hash):
    """Content identity excludes machine paths, timestamps and setup locations."""
    content = {'dataset': spec.key, 'gt_sha256': gt_hash,
               'videos': [{key: video.get(key) for key in
                           ('filename', 'category', 'is_anomaly', 'reference_tokens',
                            'feature_dtype', 'evaluation_frames', 'gt_offset', 'sha256',
                            'official_feature_sha256', 'intervals_raw')}
                          for video in videos]}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def prepare(dataset_config, resources, input_type='both'):
    import numpy as np

    if input_type not in ('raw', 'official', 'both'):
        raise ValueError('Input type must be raw, official or both')
    adapter = get_dataset(dataset_config['key'])
    spec = adapter.SPEC
    manifest = deepcopy(adapter.template(dataset_config))
    videos = manifest['videos']
    want_features = input_type in ('official', 'both')
    want_videos = input_type in ('raw', 'both')
    if spec.key == 'xd' and not want_features:
        raise ValueError('XD reference lengths must first be established from official test features')
    if want_features:
        root = resource_path(resources, 'features_root')
        preserve_parent = dataset_config['feature_identity'] == 'parent_and_name'
        feature_key = (lambda path: (path.parent.name, path.name)) if preserve_parent else (lambda path: path.name)
        features = index_files(root, '.npy', feature_key)
    if want_videos:
        raw = index_files(resource_path(resources, 'videos_root'), '.mp4', lambda path: path.name)
    for video in videos:
        if want_features:
            key = ((video['feature_parent'], video['official_feature_filename']) if preserve_parent
                   else video['official_feature_filename'])
            path = unique_path(features, key)
            feature = np.load(path, mmap_mode='r', allow_pickle=False)
            if (feature.ndim != 2 or feature.shape[0] < 1 or feature.shape[1] != 512
                    or feature.dtype.kind != 'f'
                    or any(not np.isfinite(feature[start:start + 4096]).all()
                           for start in range(0, len(feature), 4096))):
                raise ValueError(f'Expected finite [T,512] feature: {path.name}')
            if video.get('reference_tokens', len(feature)) != len(feature):
                raise ValueError('Official feature length differs from portable UCF reference')
            if video.get('feature_dtype', str(feature.dtype)) != str(feature.dtype):
                raise ValueError('Official feature dtype differs from portable UCF reference')
            video.update(reference_tokens=int(len(feature)), evaluation_frames=int(len(feature)) * 16,
                         feature_dtype=str(feature.dtype), feature_path=str(path),
                         official_feature_sha256=sha256(path))
        if want_videos:
            path = unique_path(raw, video['filename'])
            size = path.stat().st_size
            digest = sha256(path)
            if ('bytes' in video and size != video['bytes']) or ('sha256' in video and digest != video['sha256']):
                raise ValueError(f'Raw video differs from the preserved reference: {video["filename"]}')
            video.update(raw_path=str(path), bytes=size, sha256=digest)
        if (video['official_csv_index'] + 1) % 25 == 0:
            print(f'Inputs checked: {video["official_csv_index"] + 1}/{len(videos)}', flush=True)
    validate_manifest(manifest, spec)
    gt_path = project_path(dataset_config['frame_gt'])
    validate_gt(np.load(gt_path, allow_pickle=False), videos)
    gt_hash = sha256(gt_path)
    manifest.update(prepared_at=now().isoformat(), input_type=input_type,
                    frame_gt_path=str(gt_path), frame_gt_sha256=gt_hash,
                    dataset_identity_sha256=identity_hash(spec, videos, gt_hash),
                    preparation_scope='Name/order/shape/finiteness/SHA/GT alignment; no decode or inference')
    setup_root = ROOT / '.local/setup' / spec.key
    setup_id, folder = create_run_dir(spec.key, 'setup', output_root=setup_root)
    manifest['setup_id'] = setup_id
    write_json(folder / 'manifest.json', manifest)
    write_json(setup_root / 'latest.json', {'manifest': str(folder / 'manifest.json'),
                                           'setup_id': setup_id})
    return folder / 'manifest.json'


def load_prepared(key, resources, dataset_config):
    from .config import read_json

    path = (resource_path(resources, 'setup_manifest') if resources.get('setup_manifest')
            else Path(read_json(ROOT / '.local/setup' / key / 'latest.json')['manifest']))
    manifest = read_json(path)
    spec = get_dataset(key).SPEC
    validate_manifest(manifest, spec)
    gt_path = project_path(dataset_config['frame_gt'])
    if sha256(gt_path) != manifest['frame_gt_sha256']:
        raise ValueError('Bundled frame GT changed since preparation')
    if identity_hash(spec, manifest['videos'], manifest['frame_gt_sha256']) != manifest['dataset_identity_sha256']:
        raise ValueError('Prepared input identity changed; prepare inputs again')
    # Bind preparation to current authoritative order/labels and UCF reference lengths.
    template = get_dataset(key).template(dataset_config)
    if len(template['videos']) != len(manifest['videos']):
        raise ValueError('Prepared and current dataset lists differ')
    for current, saved in zip(template['videos'], manifest['videos']):
        for field in ('filename', 'official_csv_index', 'category', 'is_anomaly', 'intervals_raw',
                      'official_feature_filename', 'reference_tokens', 'feature_dtype', 'sha256'):
            if field in current and current[field] != saved.get(field):
                raise ValueError(f'Prepared metadata differs from current {field}')
    return path, manifest
