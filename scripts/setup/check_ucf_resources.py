"""Inventory UCF resources without importing models, decoding videos or loading arrays.

Run from any working directory. Relative config values resolve against the project
root. Only derived reports/local CSV under outputs/setup are written. A successful
inventory does not establish feature/GT alignment or checkpoint loadability.
"""

import argparse
from collections import defaultdict
import csv
from datetime import datetime
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
from zoneinfo import ZoneInfo


# Expected ViT-B/16 archive hash from bundled VadCLIP/src/clip/clip.py.
CLIP_SHA256 = '5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_resource(value, root):
    if value is None or value == '':
        return None
    if not isinstance(value, str):
        raise ValueError('Resource paths must be strings or null')
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def feature_key(path):
    """Preserve the official parent folder, including the normal test split."""
    return path.parent.name, path.name


def checkpoint_record(path, expected_sha256=None):
    record = {'path': str(path) if path else None,
              'model_loading_performed': False, 'inventory_ok': False}
    if path is None:
        record['issue'] = 'path_not_configured'
    elif not path.is_file():
        record['issue'] = 'file_missing'
    elif path.suffix.lower() not in {'.pt', '.pth'}:
        record['issue'] = 'expected_complete_pt_or_pth_file_not_partial_or_archive'
    elif path.stat().st_size == 0:
        record['issue'] = 'empty_file'
    else:
        record.update(bytes=path.stat().st_size, sha256=sha256(path))
        if expected_sha256:
            record['expected_sha256'] = expected_sha256
            record['inventory_ok'] = record['sha256'] == expected_sha256
            if not record['inventory_ok']:
                record['issue'] = 'sha256_mismatch'
        else:
            record['inventory_ok'] = True
            record['identity_note'] = 'Dataset/architecture compatibility not yet verified'
    return record


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=root / 'configs/ucf_resources.json')
    parser.add_argument('--write-local-list', action='store_true',
                        help='Write a separate local CSV only if all 290 mappings are valid')
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text())
    features_root = resolve_resource(config.get('features_root'), root)
    original_csv = root / 'VadCLIP/list/ucf_CLIP_rgbtest.csv'
    manifest_path = root / 'UCF-Crime/test/manifest.json'
    with original_csv.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    manifest = json.loads(manifest_path.read_text())['videos']

    issues = []
    expected_names = [Path(row['path']).name for row in rows]
    expected_keys = [feature_key(Path(row['path'])) for row in rows]
    expected_key_set = set(expected_keys)
    expected_name_set = set(expected_names)
    if len(rows) != 290 or len(set(expected_names)) != 290:
        issues.append('Official CSV must contain 290 unique feature filenames')
    correspondence_ok = len(rows) == len(manifest) and all(
        Path(row['path']).name == video['official_feature_filename']
        and row['label'] == video['category']
        and video['official_csv_index'] == index
        for index, (row, video) in enumerate(zip(rows, manifest)))
    if not correspondence_ok:
        issues.append('Official CSV order/labels do not match the test manifest')

    missing_videos = [video['video_path'] for video in manifest
                      if not (root / 'UCF-Crime/test' / video['video_path']).is_file()]
    if missing_videos:
        issues.append('Some test video links are missing or broken')

    found = defaultdict(set)
    ignored_same_name_features = []
    if features_root is None or not features_root.is_dir():
        issues.append('features_root is not configured or does not exist')
    else:
        for path in features_root.rglob('*.npy'):
            if not path.is_file():
                continue
            key = feature_key(path)
            if key in expected_key_set:
                found[key].add(path.resolve())
            elif path.name in expected_name_set:
                ignored_same_name_features.append(str(path.resolve()))
    missing_features = ['/'.join(key) for key in expected_keys if not found[key]]
    duplicate_features = {'/'.join(key): sorted(map(str, paths)) for key, paths in found.items()
                          if len(paths) > 1}
    empty_features = [str(next(iter(found[key]))) for key in expected_keys
                      if len(found[key]) == 1 and next(iter(found[key])).stat().st_size == 0]
    mapping_ok = (len(rows) == 290 and len(set(expected_names)) == 290
                  and correspondence_ok and not missing_features
                  and not duplicate_features and not empty_features)
    if not mapping_ok:
        issues.append('Feature mapping is incomplete, ambiguous or contains empty files')

    checkpoints = {
        'ucf': checkpoint_record(resolve_resource(config.get('ucf_checkpoint'), root)),
        'clip': checkpoint_record(resolve_resource(config.get('clip_checkpoint'), root), CLIP_SHA256),
    }
    for name, record in checkpoints.items():
        if not record['inventory_ok']:
            issues.append(f'{name} checkpoint: {record["issue"]}')
    gt_path = root / 'VadCLIP/list/gt_ucf.npy'
    if not gt_path.is_file():
        issues.append('gt_ucf.npy is missing')

    setup = root / 'outputs/setup'
    setup.mkdir(parents=True, exist_ok=True)
    local_csv = setup / 'ucf_CLIP_rgbtest.local.csv'
    local_written = False
    if args.write_local_list and mapping_ok:
        with local_csv.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['path', 'label'])
            writer.writeheader()
            writer.writerows({'path': str(next(iter(found[feature_key(Path(row['path']))]))),
                              'label': row['label']} for row in rows)
        local_written = True

    packages = {}
    for name in ['numpy', 'torch', 'torchvision', 'pandas', 'scikit-learn']:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    report = {
        'generated_at': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
        'scope': 'Path/name/order inventory and checkpoint hashes only; no array/model loading',
        'project_root': str(root), 'config_path': str(config_path),
        'input_sha256': {'config': sha256(config_path), 'official_csv': sha256(original_csv),
                         'manifest': sha256(manifest_path), 'script': sha256(Path(__file__))},
        'environment': {'python': sys.version, 'executable': sys.executable,
                        'platform': platform.platform(), 'package_metadata': packages,
                        'imports_and_device_availability_verified': False},
        'official_csv_count': len(rows), 'manifest_order_and_labels_match': correspondence_ok,
        'missing_test_videos': missing_videos,
        'features_root': str(features_root) if features_root else None,
        'feature_identity': 'Official CSV parent directory + filename',
        'matched_feature_count': sum(len(found[key]) == 1 for key in expected_keys),
        'ignored_same_name_features': sorted(ignored_same_name_features),
        'missing_features': missing_features, 'duplicate_features': duplicate_features,
        'empty_features': empty_features, 'feature_mapping_ok': mapping_ok,
        'feature_arrays_loaded': False,
        'checkpoints': checkpoints,
        'gt': {'path': str(gt_path), 'exists': gt_path.is_file(),
               'array_loaded': False, 'alignment_verified': False},
        'local_csv_written_this_run': local_written,
        'local_csv': str(local_csv) if local_written else None,
        'path_inventory_complete': not issues,
        'step_1_complete': False,
        'pending': ['Feature shape/dtype/finiteness', 'Feature-to-GT timeline alignment',
                    'Checkpoint loading and dataset compatibility', 'Imports/device verification',
                    'Experiment protocol decisions'],
        'issues': issues,
    }
    report_path = setup / 'ucf_resource_check.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(f'Feature files matched: {report["matched_feature_count"]}/{len(rows)}')
    print(f'Path inventory complete: {report["path_inventory_complete"]}')
    for issue in issues:
        print(f'- {issue}')
    print(f'Report: {report_path}')
    if local_written:
        print(f'Local feature CSV: {local_csv}')
    print('Array alignment, model loading and device checks remain pending.')
    return 0 if not issues else 1


if __name__ == '__main__':
    raise SystemExit(main())
