"""User-run UCF raw-video 100% reference, CLIP timing and official comparison.

The configured recipe and its verification status come from the raw config
(extraction_status). Smoke is optional; full runs all 290 videos directly.
Preserve all original inputs and prior runs.
"""

import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

from evaluate_ucf_official import ROOT, LABELS, infer_feature, project_path, sha256, synchronize


def create_output(mode):
    stamp = datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S')
    base = f'ucf_raw_{mode}_{stamp}'
    number = 1
    while True:
        run_id = base if number == 1 else f'{base}_{number}'
        output = ROOT / 'outputs/baseline' / run_id
        try:
            output.mkdir(parents=True, exist_ok=False)
            (output / 'features').mkdir()
            (output / 'scores').mkdir()
            return run_id, output
        except FileExistsError:
            number += 1


def audited_path(value):
    """Rebase an audited absolute path whose project folder was later renamed.

    The input audit stores absolute paths (e.g. the former '03_인공지능 프로젝트'
    folder). Keep the audit file unchanged; map the part after the project
    directory name onto the current ROOT. Callers still verify SHA-256.
    """
    path = project_path(value)
    if path.exists():
        return path
    parts = Path(value).parts
    if ROOT.name in parts:
        marker = len(parts) - 1 - parts[::-1].index(ROOT.name)
        rebased = ROOT.joinpath(*parts[marker + 1:])
        if rebased.exists():
            return rebased
    raise FileNotFoundError(f'Audited path not found, also after project-root rebasing: {value}')


def choose_videos(mode, requested, manifest, audit):
    declared = manifest['videos']
    if mode == 'full':
        if requested:
            raise ValueError('--video is only allowed in smoke; full must cover all 290 videos')
        return declared
    if requested:
        names = {video['filename'] for video in declared}
        if len(set(requested)) != len(requested) or set(requested) - names:
            raise ValueError('--video must name distinct videos in the official test manifest')
        return [video for video in declared if video['filename'] in requested]
    lengths = {record['video']: record['feature_length'] for record in audit['videos']}
    anomaly = min((video for video in declared if video['is_anomaly']), key=lambda v: lengths[v['filename']])
    normal = min((video for video in declared if not video['is_anomaly']), key=lambda v: lengths[v['filename']])
    long_video = min((video for video in declared if lengths[video['filename']] >= 512),
                     key=lambda v: lengths[v['filename']])
    selected = {video['filename'] for video in (anomaly, normal, long_video)}
    return [video for video in declared if video['filename'] in selected]


def read_inputs(config, official_run, report):
    resources_path = project_path(config['resources_config'])
    resources = json.loads(resources_path.read_text())
    audit_path = ROOT / 'outputs/setup/ucf_baseline_input_check.json'
    audit = json.loads(audit_path.read_text())
    resource_audit = json.loads((ROOT / 'outputs/setup/ucf_resource_check.json').read_text())
    manifest_path = ROOT / 'UCF-Crime/test/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    summary_path = official_run / 'summary.json'
    baseline = json.loads(summary_path.read_text())
    if (not audit.get('data_checks_passed') or audit.get('issues')
            or baseline.get('success') is not True or baseline.get('mode') != 'full'
            or baseline.get('input_type') != 'official_features'
            or baseline.get('dataset') != 'UCF-Crime'):
        raise ValueError('Use a passed input audit and a successful full official-feature baseline')
    if baseline['metrics'].get('fine_grained', {}).get('protocol') != 'original_ucf_test_excludeNormal_false':
        raise ValueError('Use an official run with the original 290-video/14-class fine-grained protocol')
    paths = {'config': resources_path, 'manifest': manifest_path,
             'local_csv': ROOT / 'outputs/setup/ucf_CLIP_rgbtest.local.csv',
             'gt': ROOT / 'VadCLIP/list/gt_ucf.npy'}
    for key, path in paths.items():
        value = sha256(path)
        if value != audit['input_sha256'].get(key) or value != baseline['input_sha256'].get(key):
            raise ValueError(f'{key} changed since the audited official baseline')
        report['input_sha256'][f'official_{key}'] = value
    report['input_sha256'].update(input_audit=sha256(audit_path), official_summary=sha256(summary_path))
    videos, audited, prior = manifest['videos'], audit['videos'], baseline['videos']
    if any(len(records) != 290 for records in (videos, audited, prior)):
        raise ValueError('Expected 290 manifest/audit/official-run records')
    if len({video['filename'] for video in videos}) != 290:
        raise ValueError('Duplicate manifest video names')
    offset = 0
    for index, (video, record, previous) in enumerate(zip(videos, audited, prior)):
        if (video['official_csv_index'] != index or record['index'] != index
                or previous['official_csv_index'] != index
                or video['filename'] != record['video'] or record['video'] != previous['video']
                or previous.get('case') != 'full_video' or previous.get('prefix_only') is not False
                or previous['token_count'] != record['feature_length']
                or record['gt_offset'] != offset or previous['gt_offset'] != offset):
            raise ValueError('Manifest/audit/official baseline order, length or scope mismatch')
        offset += record['feature_length'] * 16
    for name, key in (('clip', 'clip_checkpoint'), ('ucf', 'ucf_checkpoint')):
        path = project_path(resources[key])
        value = sha256(path)
        if value != resource_audit['checkpoints'][name]['sha256'] or value != baseline['input_sha256'][key]:
            raise ValueError(f'{name} checkpoint changed since the official baseline')
        report['input_sha256'][key] = value
    for key, relative in (
            ('model_source', 'VadCLIP/src/model.py'), ('layers_source', 'VadCLIP/src/utils/layers.py'),
            ('tools_source', 'VadCLIP/src/utils/tools.py'), ('metrics_source', 'scripts/ucf_metrics.py'),
            ('official_fine_grained_source', 'VadCLIP/src/utils/ucf_detectionMAP.py')):
        value = sha256(ROOT / relative)
        if value != baseline['input_sha256'].get(key):
            raise ValueError(f'{key} changed since the official baseline; reproduce that baseline first')
        report['input_sha256'][key] = value
    return resources, manifest, audit, baseline, offset


def metric_comparison(raw, official):
    differences = {}
    for branch in ('branch1', 'branch2'):
        differences[branch] = {key: 100 * (raw[branch][key] - official[branch][key])
                               for key in ('roc_auc', 'ap', 'ano_auc')}
    differences['fine_grained'] = {
        'map_by_iou': {key: 100 * (value - official['fine_grained']['map_by_iou'][key])
                       for key, value in raw['fine_grained']['map_by_iou'].items()},
        'mean_map': 100 * (raw['fine_grained']['mean_map'] - official['fine_grained']['mean_map'])}
    return {'unit': 'percentage_points', 'raw_minus_official': differences,
            'reproduction_status': 'Not judged automatically: compare against the tolerance fixed before the run'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('smoke', 'full'), default='smoke')
    parser.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cpu')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/ucf_raw_reference.json')
    parser.add_argument('--official-run', type=Path, help='Override the configured official reference run')
    parser.add_argument('--video', action='append', help='Exact manifest filename; repeat for optional smoke selection')
    parser.add_argument('--temporal-mode', choices=('single_frame', 'mean_block'))
    parser.add_argument('--frame-offset', type=int, help='0..15 within each block in single_frame mode')
    args = parser.parse_args()
    run_id, output = create_output(args.mode)
    report = {'run_id': run_id, 'mode': args.mode, 'dataset': 'UCF-Crime',
              'input_type': 'raw_video', 'device': args.device, 'sampling': 'Full / Reference',
              'budget': 1.0, 'success': False, 'videos': [], 'metrics': None,
              'input_sha256': {}, 'output_dir': str(output),
              'smoke_outputs_are_benchmark_results': False,
              'official_extraction_reproduced': False,
              'started_at': datetime.now(ZoneInfo('Asia/Seoul')).isoformat()}
    try:
        import numpy as np
        import torch
        import cv2
        import PIL
        import torchvision
        from ucf_raw_features import (ReferenceImageEncoder, validate_extraction, extract_reference,
                                      compare_features, event_coverage)
        from ucf_metrics import compute_ucf_metrics
        from ucf_finegrained import compute_official_fine_grained

        config_path = project_path(args.config)
        config = json.loads(config_path.read_text())
        if args.temporal_mode is not None:
            config['temporal_mode'] = args.temporal_mode
        if args.frame_offset is not None:
            config['frame_offset'] = args.frame_offset
        validate_extraction(config)
        report.update(extraction_config=config, command=sys.argv,
                      extraction_status=config['extraction_status'])
        report['input_sha256'].update(raw_config=sha256(config_path), raw_script=sha256(Path(__file__)),
                                      raw_helper=sha256(ROOT / 'scripts/ucf_raw_features.py'),
                                      color_helper=sha256(ROOT / 'scripts/ucf_color.py'),
                                      reused_evaluator=sha256(ROOT / 'scripts/evaluate_ucf_official.py'),
                                      crop_source=sha256(ROOT / 'VadCLIP/src/crop.py'),
                                      clip_preprocess_source=sha256(ROOT / 'VadCLIP/src/clip/clip.py'),
                                      clip_model_source=sha256(ROOT / 'VadCLIP/src/clip/model.py'),
                                      ucf_options_source=sha256(ROOT / 'VadCLIP/src/ucf_option.py'),
                                      fine_grained_source=sha256(ROOT / 'scripts/ucf_finegrained.py'))
        provenance_path = ROOT / 'VadCLIP/UPSTREAM.json'
        provenance = json.loads(provenance_path.read_text())
        report['input_sha256']['upstream_provenance'] = sha256(provenance_path)
        report.update(upstream_repository=provenance['repository'], upstream_commit=provenance['commit'])
        for key, command in (('research_commit', ['rev-parse', 'HEAD']),
                             ('research_dirty', ['status', '--porcelain', '--untracked-files=normal'])):
            try:
                result = subprocess.run(['git', '-C', str(ROOT), *command], capture_output=True, text=True, check=False)
                report[key] = (bool(result.stdout.strip()) if key == 'research_dirty' else result.stdout.strip()) if result.returncode == 0 else None
            except FileNotFoundError:
                report[key] = None
        official_run = project_path(args.official_run or config['official_reference_run'])
        report['official_reference_run'] = str(official_run)
        setup_started = time.perf_counter()
        resources, manifest, audit, baseline, frame_count = read_inputs(config, official_run, report)
        selected = choose_videos(args.mode, args.video, manifest, audit)
        report['selected_videos'] = [video['filename'] for video in selected]
        if args.device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable')
        if args.device == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            raise ValueError('MPS is unavailable')
        sys.path.insert(0, str(ROOT / 'VadCLIP/src'))
        from model import CLIPVAD
        from clip import clip
        from crop import image_crop
        from PIL import Image
        import ucf_option
        from utils.tools import get_prompt_text

        defaults = ucf_option.parser.parse_args([])
        defaults.seed = config['seed']
        options = {key: getattr(defaults, key) for key in baseline['model_options']}
        if options != baseline['model_options'] or tuple(baseline['prompt_text']) != LABELS:
            raise ValueError('Model options/seed/prompts differ from the official reference run')
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        model = CLIPVAD(defaults.classes_num, defaults.embed_dim, defaults.visual_length,
                        defaults.visual_width, defaults.visual_head, defaults.visual_layers,
                        defaults.attn_window, defaults.prompt_prefix, defaults.prompt_postfix,
                        args.device, clip_checkpoint=str(project_path(resources['clip_checkpoint'])))
        model.load_state_dict(torch.load(project_path(resources['ucf_checkpoint']), map_location='cpu', weights_only=True), strict=True)
        model.to(args.device).eval().requires_grad_(False)
        # CLIP dtype derives from visual.conv1. Changing only the original visual
        # module would also change text inference dtype. Keep the model untouched.
        visual = deepcopy(model.clipmodel.visual).eval().requires_grad_(False)
        if config['encoder_precision'] == 'float32':
            visual.float()
        encoder = ReferenceImageEncoder(visual)
        preprocess = clip._transform(visual.input_resolution)
        prompts = get_prompt_text(dict(zip(LABELS, LABELS)))
        gt = np.load(ROOT / 'VadCLIP/list/gt_ucf.npy', allow_pickle=False)
        if gt.ndim != 1 or len(gt) != frame_count:
            raise ValueError('Frame GT timeline differs from the audited feature list')
        if args.mode == 'full':
            segments_path = ROOT / 'VadCLIP/list/gt_segment_ucf.npy'
            labels_path = ROOT / 'VadCLIP/list/gt_label_ucf.npy'
            report['input_sha256'].update(official_gt_segments=sha256(segments_path),
                                          official_gt_labels=sha256(labels_path))
            for key in ('official_gt_segments', 'official_gt_labels'):
                if report['input_sha256'][key] != baseline['input_sha256'].get(key):
                    raise ValueError('Segment/class GT changed since the official baseline')
            gt_segments = np.load(segments_path, allow_pickle=True)
            gt_labels = np.load(labels_path, allow_pickle=True)
        report.update(model_options=options, prompt_text=prompts,
                      random_seeds={'python': config['seed'], 'numpy': config['seed'], 'torch': config['seed']},
                      encoder_weights_source='Independent frozen copy of the visual module from the strict-loaded UCF checkpoint',
                      environment={'python': sys.version, 'executable': sys.executable,
                                   'torch': torch.__version__, 'numpy': np.__version__, 'opencv': cv2.__version__,
                                   'pillow': PIL.__version__, 'torchvision': torchvision.__version__,
                                   'platform': platform.platform(), 'machine': platform.machine(),
                                   'torch_cpu_threads': torch.get_num_threads(),
                                   'opencv_threads': cv2.getNumThreads(),
                                   'cuda_device_name': torch.cuda.get_device_name() if args.device == 'cuda' else None},
                      precision={'clip_visual': str(visual.conv1.weight.dtype),
                                 'vad_parameters': str(model.classifier.weight.dtype),
                                 'vad_clip_visual': str(model.clipmodel.visual.conv1.weight.dtype),
                                 'vad_clip_text_embedding': str(model.clipmodel.token_embedding.weight.dtype)},
                      rules={'candidate_definition': 'One configured frame per complete 16-frame block' if config['temporal_mode'] == 'single_frame' else 'All frames in complete 16-frame blocks; 16 encodings averaged per token',
                             'frame_index_base': 0, 'annotation_endpoints': 'Preserve published [start,end) values; raw coordinate convention unverified',
                             'tail': ('Encode the tail block frame only where the official timeline has ceil(N/16) rows; never pad raw frames'
                                      if config.get('tail_policy') == 'match_official_rows' else 'Drop final incomplete block; never pad raw frames'),
                             'color_conversion': config.get('color_conversion', 'opencv_bgr'),
                             'image_geometry': config.get('image_geometry', 'published_340x256'),
                             'score_restore': 'Repeat token score over its original 16-frame block on the official evaluation timeline',
                             'window_and_mask': 'Unchanged infer_feature from official evaluator',
                             'primary_branch': 'branch1', 'fine_grained_branch': 'branch2',
                             'feature_normalization': 'L2 after block averaging' if config['feature_l2_normalize'] else 'No feature L2 normalization',
                             'retraining': False, 'sampler_uses_gt': False})
        report['setup_seconds'] = time.perf_counter() - setup_started
        warm_started = time.perf_counter()
        black = image_crop(np.zeros((256, 340, 3), dtype=np.uint8), config['crop_type'])
        warm_images = preprocess(Image.fromarray(black)).unsqueeze(0).repeat(config['encoder_batch_size'], 1, 1, 1)
        warm_images = warm_images.to(device=args.device, dtype=visual.conv1.weight.dtype)
        for _ in range(config['warmup_batches']):
            with torch.no_grad():
                encoder.encode_image(warm_images)
            synchronize(torch, args.device)
            infer_feature(np, torch, model, np.zeros((defaults.visual_length, 512), dtype=np.float32),
                          defaults.visual_length, prompts, args.device)
        del warm_images
        report['warmup'] = {'batches': config['warmup_batches'], 'seconds': time.perf_counter() - warm_started,
                            'input': 'Preprocessed synthetic black image batch and 256-token zero feature sequence; excluded from timings'}
        frames1, frames2, class_scores = [], [], []
        total_covered = total_events = 0
        loop_started = time.perf_counter()
        print(f'Raw {args.mode}: {len(selected)} videos, {args.device}, '
              f'{config["temporal_mode"]} offset {config["frame_offset"]}, crop {config["crop_type"]}, '
              f'{config.get("color_conversion", "opencv_bgr")}/{config.get("image_geometry", "published_340x256")}; '
              f'{config["extraction_status"]}', flush=True)
        for video in selected:
            report['current_video'] = video['filename']
            index = video['official_csv_index']
            audited, previous = audit['videos'][index], baseline['videos'][index]
            path = (ROOT / 'UCF-Crime/test' / video['video_path']).resolve()
            if not path.is_file() or path.stat().st_size != video['bytes']:
                raise ValueError(f'Raw video is missing or size changed: {video["filename"]}')
            source_hash = sha256(path)
            if source_hash != video['sha256']:
                raise ValueError(f'Raw video changed since the manifest: {video["filename"]}')
            dtype = np.dtype(audited['dtype'] if config['feature_storage_dtype'] == 'official' else 'float32')
            synchronize(torch, args.device)
            started = time.perf_counter()
            features, selection, metadata = extract_reference(path, encoder, preprocess, torch, cv2,
                                                               args.device, config, dtype,
                                                               expected_tokens=audited['feature_length'])
            metadata['manifest_fps'] = video['fps']
            metadata['manifest_header_frames'] = video['frame_count_header']
            metadata['fps_matches_manifest'] = abs(metadata['fps'] - video['fps']) < 1e-3
            if len(features) != audited['feature_length']:
                # Keep useful extraction diagnostics; do not force-align lengths.
                diagnostic = output / 'features' / f'{index:03d}_{Path(video["filename"]).stem}.npz'
                np.savez_compressed(diagnostic, features=features, **selection)
                report['failed_video'] = {'video': video['filename'], **metadata,
                                          'expected_tokens': audited['feature_length'], 'feature_file': str(diagnostic)}
                raise ValueError(f'Raw/official token count mismatch for {video["filename"]}: '
                                 f'{len(features)} != {audited["feature_length"]}; no clipping/resampling applied')
            vad_started = time.perf_counter()
            score1, score2, probabilities, details = infer_feature(np, torch, model, features,
                                                                  defaults.visual_length, prompts, args.device)
            metadata['timing_seconds']['vad_total'] = time.perf_counter() - vad_started
            restore_started = time.perf_counter()
            frame1, frame2 = np.repeat(score1, 16), np.repeat(score2, 16)
            metadata['timing_seconds']['score_restore'] = time.perf_counter() - restore_started
            synchronize(torch, args.device)
            elapsed = time.perf_counter() - started
            metadata['timing_seconds']['end_to_end'] = elapsed
            # Validation, GT analysis and output writes begin after E2E timing.
            validation_started = time.perf_counter()
            official_path = audited_path(audited['path'])
            if sha256(official_path) != previous['feature_sha256']:
                raise ValueError('Official feature changed since the reference run')
            official_features = np.load(official_path, allow_pickle=False)
            if list(official_features.shape) != audited['shape'] or str(official_features.dtype) != audited['dtype']:
                raise ValueError('Official feature shape/dtype changed since the input audit')
            feature_comparison = compare_features(features, official_features)
            score_path = official_run / 'scores' / Path(previous['score_file']).name
            with np.load(score_path, allow_pickle=False) as saved:
                gt_slice = gt[audited['gt_offset']:audited['gt_offset'] + len(frame1)]
                if not np.array_equal(saved['gt'], gt_slice):
                    raise ValueError('Saved official score GT differs from the audited timeline')
                score_comparison = {}
                for name, score in (('branch1', score1), ('branch2', score2)):
                    official_score = saved[f'{name}_feature_scores']
                    if score.shape != official_score.shape or not np.isfinite(official_score).all():
                        raise ValueError('Saved official score shape/values are invalid')
                    difference = np.abs(score.astype(np.float64) - official_score)
                    score_comparison[name] = {'mean_absolute_error': float(difference.mean()),
                                               'max_absolute_error': float(difference.max())}
            coverage = event_coverage(selection['selected_frame_indices'], video['intervals_raw'])
            total_covered += coverage['covered_events']
            total_events += coverage['total_events']
            stem = f'{index:03d}_{Path(video["filename"]).stem}'
            feature_file, score_file = output / 'features' / f'{stem}.npz', output / 'scores' / f'{stem}.npz'
            np.savez_compressed(feature_file, features=features, **selection)
            np.savez_compressed(score_file, branch1_feature_scores=score1, branch2_feature_scores=score2,
                                branch2_class_feature_scores=probabilities, class_labels=np.asarray(LABELS), gt=gt_slice)
            report['videos'].append({'official_csv_index': index, 'video': video['filename'],
                                     'case': 'full_video', 'prefix_only': False, 'gt_offset': audited['gt_offset'],
                                     'raw_video_sha256': source_hash, 'feature_file': str(feature_file),
                                     'score_file': str(score_file), 'official_score_sha256': sha256(score_path),
                                     'official_feature_sha256': previous['feature_sha256'],
                                     'feature_comparison': feature_comparison, 'score_comparison': score_comparison,
                                     'event_coverage': coverage, 'validation_and_save_seconds': time.perf_counter() - validation_started,
                                     **metadata, **details})
            report.pop('current_video', None)
            if args.mode == 'full':
                frames1.append(frame1)
                frames2.append(frame2)
                class_scores.append(probabilities)
            print(f'{len(report["videos"])}/{len(selected)}: {video["filename"]} '
                  f'CLIP {metadata["timing_seconds"]["clip_forward"]:.3f}s, E2E {elapsed:.3f}s, '
                  f'feature cosine {feature_comparison["cosine_mean"]:.6f}, '
                  f'relL2 {feature_comparison["relative_l2_error"]:.5f}', flush=True)
        report['processing_loop_wall_seconds'] = time.perf_counter() - loop_started
        stage_keys = report['videos'][0]['timing_seconds']
        report['timing_seconds'] = {key: sum(video['timing_seconds'][key] for video in report['videos']) for key in stage_keys}
        report['end_to_end_seconds'] = report['timing_seconds']['end_to_end']
        report['encoder_input_images'] = sum(video['encoder_input_images'] for video in report['videos'])
        report['timing_scope'] = {
            'end_to_end': 'Sum of sequential video calls: open/decode/select/preprocess/transfers/CLIP/feature aggregation/VadCLIP/score restoration',
            'excluded': 'Imports/input hashing/model loading/synthetic warm-up/official comparisons/GT analysis/metric computation/result serialization',
            'clip_forward': 'Synchronized image encoder forward only; input/output transfers and preprocessing measured separately',
            'io_condition': 'Raw source hash read immediately before timed decode; OS file cache uncontrolled and likely warm',
            'repeats': 1, 'speedup_reported': False, 'device_synchronization': args.device in ('cuda', 'mps')}
        cohorts = {}
        for record in report['videos']:
            dtype_name = audit['videos'][record['official_csv_index']]['dtype']
            cohorts.setdefault(dtype_name, []).append(record['feature_comparison']['relative_l2_error'])
        report['feature_agreement_by_official_dtype'] = {
            name: {'videos': len(values), 'relative_l2_median': float(np.median(values)),
                   'relative_l2_max': float(np.max(values)),
                   'videos_relative_l2_le_0_005': int(np.sum(np.asarray(values) <= 0.005))}
            for name, values in cohorts.items()}
        report['event_coverage'] = {'covered_events': total_covered, 'total_events': total_events,
                                    'coverage': total_covered / total_events if total_events else None,
                                    'unit': 'fraction', 'scope': 'Unclipped published intervals in selected complete videos; raw coordinate convention unverified'}
        if args.mode == 'full':
            metric_started = time.perf_counter()
            frame1, frame2 = np.concatenate(frames1), np.concatenate(frames2)
            report['metrics'] = compute_ucf_metrics(gt, frame1, frame2, report['videos'], manifest)
            report['metrics']['fine_grained'] = compute_official_fine_grained(class_scores, report['videos'],
                manifest, LABELS, gt_segments, gt_labels, ROOT / 'VadCLIP/src/utils/ucf_detectionMAP.py')
            report['metric_seconds'] = time.perf_counter() - metric_started
            report['official_comparison'] = metric_comparison(report['metrics'], baseline['metrics'])
            np.savez_compressed(output / 'frame_scores.npz', branch1=frame1, branch2=frame2)
            print(json.dumps(report['metrics'], indent=2), flush=True)
        report['success'] = True
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        traceback.print_exc(file=sys.stderr)
    finally:
        report['finished_at'] = datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
        summary_path = output / 'summary.json'
        summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(f'Report: {summary_path}', flush=True)
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
