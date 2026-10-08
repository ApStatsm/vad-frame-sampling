"""Smoke/full VadCLIP evaluation on official UCF test features.

Smoke is optional; full evaluation does not require a prior smoke report.
Inference follows vadfs.vadclip (upstream window/mask/branches/16-fold restore).
Smoke prefixes are execution checks, not benchmark results. Cached-feature
timings are not raw-video efficiency. Full evaluates temporal mAP through the
unmodified upstream UCF evaluator, using its bundled segment/class GT and
excludeNormal=False (290 videos/14 classes).
"""

import argparse
import csv
import json
from pathlib import Path
import platform
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root: vadfs
from vadfs import ucf  # noqa: E402
from vadfs.paths import ROOT, audited_path, package_hashes, project_path, sha256  # noqa: E402
from vadfs.runs import create_run_dir, record_provenance, write_json  # noqa: E402


def prepare_cases(mode, audited_videos, window):
    if mode == 'full':
        return [(record, None, 'full_video') for record in audited_videos]
    anomaly = next(record for record in audited_videos if not record['video'].startswith('Normal_'))
    normals = [record for record in audited_videos if record['video'].startswith('Normal_')]
    normal = min(normals, key=lambda record: record['feature_length'])
    long_video = next(record for record in audited_videos if record['feature_length'] >= 2 * window)
    cases = [(anomaly, None, 'anomaly_video'), (normal, None, 'short_normal_video'),
             (long_video, None, 'long_video')]
    cases.extend((long_video, count, f'boundary_prefix_{count}')
                 for count in (window - 1, window, window + 1, 2 * window))
    return cases


def check_inputs(config_path, report):
    """Audited config/list/manifest/GT and checkpoints must be unchanged."""
    audit = json.loads(ucf.INPUT_AUDIT.read_text())
    if not audit.get('ready_for_small_inference') or not audit.get('data_checks_passed') or audit.get('issues'):
        raise ValueError('Resolve baseline input audit issues first')
    hashes = {name: sha256(path) for name, path in
              [('config', config_path), ('local_csv', ucf.LOCAL_FEATURE_LIST),
               ('manifest', ucf.MANIFEST), ('gt', ucf.FRAME_GT)]}
    if any(value != audit['input_sha256'].get(name) for name, value in hashes.items()):
        raise ValueError('Config, list, manifest or GT changed since the input audit')
    report['input_sha256'].update(hashes)
    with ucf.LOCAL_FEATURE_LIST.open(newline='') as handle:
        local_rows = list(csv.DictReader(handle))
    if len(local_rows) != len(audit['videos']) or len(local_rows) != 290:
        raise ValueError('Expected 290 audited test videos')
    # Order check on the recorded strings; both may predate the project folder rename.
    if any(project_path(row['path']) != project_path(record['path'])
           for row, record in zip(local_rows, audit['videos'])):
        raise ValueError('Local list order differs from audited feature order')
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['smoke', 'full'], default='smoke')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/ucf_resources.json')
    args = parser.parse_args()
    run_id, output = create_run_dir('ucf', args.mode, ('scores',))
    report = {'run_id': run_id, 'mode': args.mode, 'dataset': 'UCF-Crime',
              'input_type': 'official_features', 'device': args.device,
              'sampling': 'not_applied', 'budget': None,
              'success': False, 'videos': [], 'metrics': None,
              'input_sha256': {}, 'output_dir': str(output),
              'timing_scope': 'Cached-feature execution only; not raw-video efficiency',
              'smoke_outputs_are_benchmark_results': False}
    try:
        record_provenance(report)
        import numpy as np
        import torch
        from vadfs import vadclip
        from vadfs.metrics import compute_official_fine_grained, compute_ucf_metrics

        vadclip.check_device(args.device)
        config_path = args.config.expanduser().resolve()
        config = json.loads(config_path.read_text())
        audit = check_inputs(config_path, report)
        report['input_sha256'].update({
            'script': sha256(Path(__file__)), 'input_audit': sha256(ucf.INPUT_AUDIT),
            **vadclip.source_hashes(),
            'official_test_source': sha256(ucf.OFFICIAL_TEST_SOURCE),
            'metrics_source': sha256(ROOT / 'vadfs/metrics.py'),
            'fine_grained_source': sha256(ROOT / 'vadfs/metrics.py'),
            'official_fine_grained_source': sha256(ucf.DETECTION_MAP_SOURCE)})
        report['package_sha256'] = package_hashes()
        manifest = json.loads(ucf.MANIFEST.read_text())
        if args.mode == 'full':
            report['input_sha256'].update({
                'official_gt_segments': sha256(ucf.SEGMENT_GT),
                'official_gt_labels': sha256(ucf.CLASS_GT)})
            # Upstream ucf_test.py loads these bundled ragged object arrays.
            gt_segments = np.load(ucf.SEGMENT_GT, allow_pickle=True)
            gt_labels = np.load(ucf.CLASS_GT, allow_pickle=True)
            if len(gt_segments) != 290 or len(gt_labels) != 290:
                raise ValueError('Expected segment/class GT for 290 test videos')
        options = vadclip.default_options()
        clip_path, ucf_path = project_path(config['clip_checkpoint']), project_path(config['ucf_checkpoint'])
        resource_audit = json.loads(ucf.RESOURCE_AUDIT.read_text())
        for name, path in [('clip', clip_path), ('ucf', ucf_path)]:
            value = sha256(path)
            report['input_sha256'][f'{name}_checkpoint'] = value
            if value != resource_audit['checkpoints'][name]['sha256']:
                raise ValueError(f'{name} checkpoint changed since the resource audit')
        torch.manual_seed(options.seed)
        model = vadclip.load_model(options, args.device, clip_path, ucf_path)
        prompts = vadclip.prompt_text(ucf.LABELS)
        report.update(environment={'python': sys.version, 'executable': sys.executable,
                                   'torch': torch.__version__, 'numpy': np.__version__,
                                   'platform': platform.platform(), 'machine': platform.machine(),
                                   'torch_cpu_threads': torch.get_num_threads(),
                                   'cuda_device_name': torch.cuda.get_device_name() if args.device == 'cuda' else None},
                      precision={'visual_tokens': 'model casts to float32',
                                 'vad_parameters': str(model.classifier.weight.dtype),
                                 'clip_parameters': str(model.clipmodel.token_embedding.weight.dtype)},
                      model_options={key: getattr(options, key) for key in vadclip.MODEL_OPTION_KEYS},
                      prompt_text=prompts,
                      rules={'crop_variant': '__5.npy', 'window_policy': 'upstream process_split and declared lengths',
                             'padding_mask': 'upstream encode_video passes None to temporal',
                             'score_branch1': 'sigmoid(logits1)', 'score_branch2': '1 - softmax(logits2)[:, 0]',
                             'score_restore': 'repeat each official feature score 16 times',
                             'branch_combination': None, 'smoothing': None})
        gt = np.load(ucf.FRAME_GT, mmap_mode='r', allow_pickle=False)
        branch1_frames, branch2_frames, video_class_scores = [], [], []
        cases = prepare_cases(args.mode, audit['videos'], options.visual_length)
        run_started = time.perf_counter()
        for case_index, (audited, prefix, case_name) in enumerate(cases):
            path = audited_path(audited['path'])
            feature = np.load(path, allow_pickle=False)
            if list(feature.shape) != audited['shape'] or str(feature.dtype) != audited['dtype']:
                raise ValueError(f'Feature shape/dtype changed: {path.name}')
            feature_sha = sha256(path)
            if prefix is not None:
                feature = feature[:prefix]
            score1, score2, class_scores, details = vadclip.infer_feature(
                model, feature, options.visual_length, prompts, args.device)
            frame1, frame2 = vadclip.restore_frames(score1), vadclip.restore_frames(score2)
            offset = audited['gt_offset']
            gt_segment = np.asarray(gt[offset:offset + len(frame1)])
            if len(gt_segment) != len(frame1):
                raise ValueError('Restored scores and GT segment lengths differ')
            score_file = output / 'scores' / f'{case_index:03d}_{Path(audited["video"]).stem}_{case_name}.npz'
            np.savez_compressed(score_file, branch1_feature_scores=score1,
                                branch2_feature_scores=score2, gt=gt_segment,
                                branch2_class_feature_scores=class_scores,
                                class_labels=np.asarray(ucf.LABELS))
            report['videos'].append({'official_csv_index': audited['index'], 'video': audited['video'],
                                     'case': case_name, 'prefix_only': prefix is not None,
                                     'feature_sha256': feature_sha, 'gt_offset': offset,
                                     'score_file': str(score_file), **details})
            if args.mode == 'full':
                branch1_frames.append(frame1)
                branch2_frames.append(frame2)
                video_class_scores.append(class_scores)
            print(f'{case_index + 1}/{len(cases)}: {audited["video"]} [{case_name}] OK', flush=True)
        report['cached_feature_loop_seconds'] = time.perf_counter() - run_started
        if args.mode == 'full':
            frame1, frame2 = np.concatenate(branch1_frames), np.concatenate(branch2_frames)
            if len(frame1) != len(gt) or len(frame2) != len(gt):
                raise ValueError('Full restored score length does not equal the full GT length')
            report['metrics'] = compute_ucf_metrics(gt, frame1, frame2, report['videos'], manifest)
            np.savez_compressed(output / 'frame_scores.npz', branch1=frame1, branch2=frame2)
            print('Calculating original UCF fine-grained mAP (290 videos, 14 classes)...', flush=True)
            report['metrics']['fine_grained'] = compute_official_fine_grained(
                video_class_scores, report['videos'], manifest, ucf.LABELS,
                gt_segments, gt_labels, ucf.DETECTION_MAP_SOURCE)
            print(json.dumps(report['metrics'], indent=2))
        report['success'] = True
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        traceback.print_exc(file=sys.stderr)
    finally:
        summary_path = output / 'summary.json'
        write_json(summary_path, report)
        if args.mode == 'smoke':
            write_json(ucf.SMOKE_REPORT, report)
        print(f'Report: {summary_path}', flush=True)
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
