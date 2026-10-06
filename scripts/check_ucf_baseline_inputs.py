"""User-run UCF feature/GT audit and optional CPU model-loading check.

Does not generate or overwrite GT, decode videos, run inference or calculate AUC.
The GT comparison mirrors the bundled official generator's [start:end] slicing
on a feature-length * 16 timeline; it does not settle raw-video coordinates.
"""

import argparse
from collections.abc import Mapping
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def project_path(value):
    if not isinstance(value, str) or not value:
        raise ValueError('A configured resource path is required')
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def all_finite(np, array):
    return all(bool(np.isfinite(array[start:start + 4096]).all())
               for start in range(0, len(array), 4096))


def check_arrays(report, local_csv, config):
    import numpy as np

    report['numpy_version'] = np.__version__
    report['input_sha256']['local_csv'] = sha256(local_csv)
    manifest_path = ROOT / 'UCF-Crime/test/manifest.json'
    report['input_sha256']['manifest'] = sha256(manifest_path)
    videos = json.loads(manifest_path.read_text())['videos']
    with local_csv.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    with (ROOT / 'VadCLIP/list/ucf_CLIP_rgbtest.csv').open(newline='') as handle:
        official_rows = list(csv.DictReader(handle))
    if len(rows) != 290 or len(videos) != 290 or len(official_rows) != 290:
        raise ValueError('Expected 290 rows in local CSV, official CSV and manifest')
    feature_root = project_path(config['features_root'])
    seen_paths = set()
    records = []
    total_timeline_length = 0
    arrays_ok = True
    for index, (row, original, video) in enumerate(zip(rows, official_rows, videos)):
        feature_path = project_path(row['path'])
        record = {'index': index, 'video': video['filename'], 'path': str(feature_path)}
        records.append(record)
        try:
            feature_path.relative_to(feature_root)
            if feature_path in seen_paths:
                raise ValueError('Repeated feature path')
            seen_paths.add(feature_path)
            original_path = Path(original['path'])
            if (feature_path.name != original_path.name
                    or feature_path.parent.name != original_path.parent.name
                    or feature_path.name != video['official_feature_filename']
                    or row['label'] != original['label']
                    or row['label'] != video['category']
                    or video['official_csv_index'] != index):
                raise ValueError('Feature identity, label or evaluation order mismatch')
            feature = np.load(feature_path, mmap_mode='r', allow_pickle=False)
            record.update(shape=list(feature.shape), dtype=str(feature.dtype),
                          bytes=feature_path.stat().st_size)
            if feature.ndim != 2 or feature.shape[0] < 1 or feature.shape[1] != 512:
                raise ValueError('Expected a nonempty feature array of shape [T, 512]')
            if feature.dtype.kind != 'f' or not all_finite(np, feature):
                raise ValueError('Features must contain finite floating-point values')
            length = int(feature.shape[0]) * 16
            record.update(feature_length=int(feature.shape[0]), finite=True,
                          gt_offset=total_timeline_length, official_timeline_length=length,
                          raw_header_frames=video.get('frame_count_header'))
            total_timeline_length += length
            # __0 / __5 need matching temporal lengths; feature values may differ.
            if not feature_path.name.endswith('__5.npy'):
                raise ValueError('Current official test list must use __5.npy')
            zero_path = feature_path.with_name(feature_path.name[:-7] + '__0.npy')
            zero = np.load(zero_path, mmap_mode='r', allow_pickle=False)
            record.update(zero_variant_path=str(zero_path), zero_variant_shape=list(zero.shape))
            if zero.shape != feature.shape:
                raise ValueError('__0 and __5 feature shapes differ')
            record['ok'] = True
        except Exception as error:
            arrays_ok = False
            record.update(ok=False, error=f'{type(error).__name__}: {error}')
            report['issues'].append(f'{video["filename"]}: {record["error"]}')
        if (index + 1) % 25 == 0 or index == len(rows) - 1:
            print(f'Features checked: {index + 1}/{len(rows)}', flush=True)
    report.update(feature_count=len(records), feature_arrays_ok=arrays_ok,
                  expected_gt_length=total_timeline_length, videos=records)
    if not arrays_ok:
        report['gt_check'] = {'performed': False, 'reason': 'Feature checks failed'}
        return False

    gt_path = ROOT / 'VadCLIP/list/gt_ucf.npy'
    report['input_sha256']['gt'] = sha256(gt_path)
    gt = np.load(gt_path, mmap_mode='r', allow_pickle=False)
    gt_report = {'performed': True, 'path': str(gt_path), 'shape': list(gt.shape),
                 'dtype': str(gt.dtype), 'length_match': gt.ndim == 1 and len(gt) == total_timeline_length,
                 'rule': 'Official feature T * 16 timeline; published intervals sliced [start:end]',
                 'raw_video_coordinate_convention_verified': False}
    report['gt_check'] = gt_report
    if not gt_report['length_match']:
        report['issues'].append('GT total length differs from the official feature score timeline')
        return False
    binary_ok = gt.dtype.kind in 'biuf' and all(
        bool(((gt[start:start + 65536] == 0) | (gt[start:start + 65536] == 1)).all())
        for start in range(0, len(gt), 65536))
    gt_report['binary_values_ok'] = binary_ok
    if not binary_ok:
        report['issues'].append('GT must contain only 0/1 values')
        return False

    mismatches = 0
    for video, record in zip(videos, records):
        offset, length = record['gt_offset'], record['official_timeline_length']
        expected = np.zeros(length, dtype=np.uint8)
        clipped_intervals = []
        for start, end in video['intervals_raw']:
            if start < 0 or end <= start:
                raise ValueError(f'Invalid published annotation: {video["filename"]}')
            expected[start:end] = 1
            if end > length:
                clipped_intervals.append([start, end])
        actual = gt[offset:offset + length]
        mismatch_positions = np.flatnonzero(actual != expected)
        count = int(len(mismatch_positions))
        record.update(gt_mismatch_count=count,
                      first_gt_mismatch_indices=mismatch_positions[:8].tolist(),
                      annotation_intervals_beyond_feature_timeline=clipped_intervals,
                      gt_positive_frames=int(np.count_nonzero(actual)))
        mismatches += count
        if count:
            report['issues'].append(f'{video["filename"]}: {count} GT positions differ')
    gt_report.update(mismatch_frames=mismatches, annotation_alignment_under_official_rule=mismatches == 0)
    return mismatches == 0


def check_model(report, config):
    import torch

    model_report = {'performed': True, 'torch_version': torch.__version__,
                    'load_device': 'cpu', 'inference_performed': False,
                    'cuda_available': torch.cuda.is_available(),
                    'mps_available': bool(hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()),
                    'ok': False}
    report['model_check'] = model_report
    sys.path.insert(0, str(ROOT / 'VadCLIP/src'))
    from model import CLIPVAD
    import ucf_option

    defaults = ucf_option.parser.parse_args([])
    clip_path = project_path(config['clip_checkpoint'])
    checkpoint_path = project_path(config['ucf_checkpoint'])
    if not clip_path.is_file() or not checkpoint_path.is_file():
        raise ValueError('Configured checkpoint file is missing')
    report['input_sha256']['model_source'] = sha256(ROOT / 'VadCLIP/src/model.py')
    model_report.update(clip_checkpoint=str(clip_path), ucf_checkpoint=str(checkpoint_path),
                        classes=defaults.classes_num, visual_length=defaults.visual_length,
                        visual_width=defaults.visual_width)
    # CPU and weights_only prevent implicit GPU placement or arbitrary pickle fallback.
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if not isinstance(state, Mapping) or not state or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()):
        raise ValueError('Expected the official checkpoint to be a direct tensor state_dict')
    model = CLIPVAD(defaults.classes_num, defaults.embed_dim, defaults.visual_length,
                    defaults.visual_width, defaults.visual_head, defaults.visual_layers,
                    defaults.attn_window, defaults.prompt_prefix, defaults.prompt_postfix,
                    'cpu', clip_checkpoint=str(clip_path))
    model.load_state_dict(state, strict=True)
    model.eval()
    model_report.update(ok=True, strict_state_dict_loading=True, parameter_entries=len(state))
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/ucf_resources.json')
    parser.add_argument('--local-list', type=Path, default=ROOT / 'outputs/setup/ucf_CLIP_rgbtest.local.csv')
    parser.add_argument('--check-model', action='store_true', help='Also import and load the model on CPU; no forward pass')
    args = parser.parse_args()
    report = {'generated_at': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
              'scope': 'Feature arrays/GT alignment and optional CPU model loading; no inference',
              'input_sha256': {'script': sha256(Path(__file__))}, 'issues': [],
              'data_checks_passed': False, 'model_check': {'performed': False},
              'ready_for_small_inference': False, 'step_1_complete': False}
    try:
        config_path = args.config.expanduser().resolve()
        report['input_sha256']['config'] = sha256(config_path)
        config = json.loads(config_path.read_text())
        report['data_checks_passed'] = check_arrays(report, args.local_list.expanduser().resolve(), config)
        if args.check_model and report['data_checks_passed']:
            report['ready_for_small_inference'] = check_model(report, config)
        elif args.check_model:
            report['model_check']['skip_reason'] = 'Resolve data issues before model loading'
    except Exception as error:
        report['issues'].append(f'{type(error).__name__}: {error}')
    report['pending'] = ['Small inference/window/padding checks', 'Actual evaluation device',
                         'Score branch/reporting protocol', 'Raw-video coordinate convention']
    report_path = ROOT / 'outputs/setup/ucf_baseline_input_check.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(f'Data checks passed: {report["data_checks_passed"]}')
    print(f'Model loading passed: {report["model_check"].get("ok", False)}')
    for issue in report['issues']:
        print(f'- {issue}')
    print(f'Report: {report_path}')
    success = report['data_checks_passed'] and not report['issues']
    if args.check_model:
        success = success and report['ready_for_small_inference']
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
