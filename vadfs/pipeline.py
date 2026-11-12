"""Common user-run experiment pipeline; no experiments run during authoring.

Raw: verify inputs -> choose original indices -> encode only selected images ->
frozen VadCLIP -> reconstruct reference tokens -> fixed full-timeline evaluation.
Official: baseline reproduction with all precomputed test features, never speedup.
Cached raw: sample saved reference rows and rerun VadCLIP for performance only.
"""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback

from .config import load_experiment, read_json, resource_path
from .datasets import get_dataset
from .paths import ROOT, package_hashes, project_path, sha256
from .preparation import load_prepared
from .runs import create_run_dir, now, record_provenance, write_json
from .samplers import METHODS, budget_count, choose, video_seed
from .samplers.random_window import select_frames as select_window_frames, window_stride


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def select_videos(videos, mode, requested):
    if mode == 'full':
        if requested:
            raise ValueError('--video is only allowed in smoke mode')
        return videos
    if requested:
        if len(set(requested)) != len(requested):
            raise ValueError('Duplicate --video argument')
        selected = [video for video in videos if video['filename'] in requested]
        if len(selected) != len(requested):
            raise ValueError('--video must be an exact test-manifest filename')
        return selected
    # This only selects execution-check cases. Labels never reach the sampler.
    cases = [min((video for video in videos if video['is_anomaly']), key=lambda v: v['reference_tokens']),
             min((video for video in videos if not video['is_anomaly']), key=lambda v: v['reference_tokens'])]
    long = [video for video in videos if video['reference_tokens'] >= 512]
    if long:
        cases.append(min(long, key=lambda v: v['reference_tokens']))
    names = {video['filename'] for video in cases}
    return [video for video in videos if video['filename'] in names]


def validate_experiment(experiment, spec, mode, cached_features=False, performance_reference=False):
    if experiment['input_type'] not in ('raw', 'official') or mode not in ('smoke', 'full'):
        raise ValueError('Invalid input type or mode')
    sampling = experiment['sampling']
    if sampling['method'] not in METHODS:
        raise ValueError(f'Sampling method must be one of {tuple(METHODS)}')
    budget_count(1, sampling['budget'])
    if sampling['method'] == 'reference' and sampling['budget'] != 1:
        raise ValueError('Reference must use a 100% budget')
    video_seed(sampling['seed'], spec.key, 'validation')
    model_seed = experiment['model_seed']
    if type(model_seed) is not int or not 0 <= model_seed < 2**32:
        raise ValueError('Invalid fixed model seed')
    if experiment['input_type'] == 'official' and (sampling['method'] != 'reference' or sampling['budget'] != 1):
        raise ValueError('Official-feature mode is baseline reproduction only')
    if experiment['input_type'] == 'official' and experiment.get('reference_run'):
        raise ValueError('Raw reference comparisons are only defined for raw-video experiments')
    if (cached_features or performance_reference) and (experiment['input_type'] != 'raw' or experiment.get('reference_run')):
        raise ValueError('Cached or performance-only references require a raw experiment without --reference-run')
    if sampling['method'] == 'random_window':
        if cached_features or experiment['input_type'] != 'raw':
            raise ValueError('Random windows require the original raw video frames')
        window_stride(sampling['budget'])
    if experiment['protocol']['restoration'] not in ('nearest', 'linear', 'previous'):
        raise ValueError('Unknown restoration policy')
    if experiment['protocol']['status'] not in ('draft', 'fixed'):
        raise ValueError('Protocol status must be draft or fixed')
    if (mode == 'full' and experiment['input_type'] == 'raw' and sampling['budget'] < 1
            and not (cached_features or performance_reference)
            and not experiment.get('reference_run')):
        raise ValueError('Full sampling requires a 100% reference via --reference-run or --performance-reference-run')


def environment(device, torch):
    import cv2
    import numpy as np
    import PIL
    import torchvision

    info = {'python': platform.python_version(), 'platform': platform.platform(),
            'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
            'torchvision': torchvision.__version__, 'numpy': np.__version__,
            'opencv': cv2.__version__, 'pillow': PIL.__version__, 'device': device,
            'torch_threads': torch.get_num_threads(), 'opencv_threads': cv2.getNumThreads(),
            'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'cuda_allow_tf32': bool(torch.backends.cuda.matmul.allow_tf32),
            'cudnn_allow_tf32': bool(torch.backends.cudnn.allow_tf32),
            'cudnn_benchmark': bool(torch.backends.cudnn.benchmark)}
    if device == 'cuda':
        prop = torch.cuda.get_device_properties(torch.cuda.current_device())
        info.update(gpu_name=prop.name, gpu_uuid=str(getattr(prop, 'uuid', 'unavailable')),
                    gpu_capability=list(torch.cuda.get_device_capability()),
                    gpu_total_memory_bytes=prop.total_memory)
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=True)
        info['ffmpeg'] = result.stdout.splitlines()[0]
    except (OSError, subprocess.CalledProcessError):
        info['ffmpeg'] = None
    return info


def load_full_raw_reference(folder, manifest):
    summary_path = folder / 'summary.json'
    baseline = read_json(summary_path)
    if (baseline.get('schema_version') != 2 or not baseline.get('success') or baseline.get('mode') != 'full'
            or baseline.get('input_type') != 'raw' or baseline.get('sampling', {}).get('budget') != 1
            or baseline.get('sampling', {}).get('method') == 'random_window'
            or len(baseline.get('videos', [])) != len(manifest['videos'])
            or not baseline.get('metrics')):
        raise ValueError('Reference must be a successful complete 100% raw run from this pipeline')
    baseline['loaded_summary_sha256'] = sha256(summary_path)
    return baseline


def check_reference(experiment, manifest, signature):
    path = experiment.get('reference_run')
    if not path:
        return None
    baseline = load_full_raw_reference(project_path(path), manifest)
    if baseline.get('comparison_signature') != signature:
        raise ValueError('Reference differs in data/model/protocol/software/hardware settings; reproduce it with the same settings')
    return baseline


def check_performance_reference(folder, manifest, report, spec, recipe):
    baseline = load_full_raw_reference(folder, manifest)
    saved = baseline.get('input_sha256', {})
    for key in ('clip_checkpoint', 'vadclip_checkpoint', 'gt', 'official_list', 'tokenizer_vocabulary'):
        if saved.get(key) != report['input_sha256'][key]:
            raise ValueError(f'Performance reference uses a different {key}')
    primary = baseline['metrics']['primary']
    if (baseline.get('dataset_key') != spec.key
            or baseline.get('dataset_identity_sha256') != manifest['dataset_identity_sha256']
            or baseline.get('extraction_config') != recipe
            or baseline.get('prompts') != list(spec.prompts)
            or baseline.get('vendored_source_sha256') != report['vendored_source_sha256']
            or baseline['experiment']['model_seed'] != report['experiment']['model_seed']
            or (primary.get('branch'), primary.get('metric')) != (spec.primary_branch, spec.primary_metric)):
        raise ValueError('Performance reference differs in dataset, extraction recipe, or model settings')
    return baseline


def load_cached_selection(path, grid, selected_frames, dtype):
    import numpy as np

    with np.load(path, allow_pickle=False) as cache:
        features = cache['features']
        indices = cache['selected_frame_indices']
        timestamps = cache['selected_timestamps_seconds']
    if (features.shape != (len(grid), 512) or features.dtype != dtype or not np.isfinite(features).all()
            or indices.dtype.kind not in 'iu' or not np.array_equal(indices, grid)
            or timestamps.shape != indices.shape or not np.isfinite(timestamps).all()):
        raise ValueError(f'Cached features do not match the full reference grid and dtype: {path}')
    rows = np.searchsorted(indices, selected_frames)
    if np.any(rows >= len(indices)) or not np.array_equal(indices[rows], selected_frames):
        raise ValueError('Selected frames must belong to the cached reference grid')
    return features[rows], {'selected_frame_indices': indices[rows],
                            'selected_timestamps_seconds': timestamps[rows]}


def run_experiment(config_path, local_path=None, mode='smoke', requested=None,
                   device_override=None, seed_override=None, reference_override=None,
                   setup_override=None, save_features=False, *, cached_reference_override=None,
                   performance_reference_override=None):
    experiment, dataset_config, resources, exp_path, data_path = load_experiment(config_path, local_path)
    if seed_override is not None:
        experiment['sampling']['seed'] = seed_override
    if reference_override:
        experiment['reference_run'] = str(reference_override)
    if setup_override:
        resources['setup_manifest'] = str(setup_override)
    cache_folder = project_path(cached_reference_override) if cached_reference_override else None
    performance_folder = project_path(performance_reference_override) if performance_reference_override else None
    if cache_folder and performance_folder:
        raise ValueError('Choose one performance reference option')
    score_reference_folder = cache_folder or performance_folder
    if score_reference_folder and not resources.get('setup_manifest') and (score_reference_folder / 'manifest.json').is_file():
        resources['setup_manifest'] = str(score_reference_folder / 'manifest.json')
    adapter = get_dataset(dataset_config['key'])
    spec = adapter.SPEC
    validate_experiment(experiment, spec, mode, cached_features=cache_folder is not None,
                        performance_reference=performance_folder is not None)
    input_type = 'cached_raw_features' if cache_folder else experiment['input_type']
    raw_mode = input_type == 'raw'
    device = device_override or resources.get('device', 'cpu')
    if device not in ('cpu', 'cuda', 'mps'):
        raise ValueError('Device must be cpu, cuda or mps')
    sampling = experiment['sampling']
    random_windows = sampling['method'] == 'random_window'
    run_id, output = create_run_dir(
        f'{spec.key}_{input_type}_{sampling["method"]}_{sampling["budget"]:g}_s{sampling["seed"]}',
        mode, ('scores', 'features'), resource_path(resources, 'output_root'))
    report = {'schema_version': 2, 'run_id': run_id, 'started_at': now().isoformat(),
              'success': False, 'mode': mode, 'dataset': spec.name, 'dataset_key': spec.key,
              'input_type': input_type, 'sampling': deepcopy(sampling),
              'experiment': experiment, 'resources': resources, 'command': sys.argv,
              'output_dir': str(output), 'input_sha256': {}, 'videos': [], 'metrics': None,
              'eligible_for_final_benchmark': not cache_folder and mode == 'full' and experiment['protocol']['status'] == 'fixed',
              'validation_status': 'execution results only; external reproduction acceptance not automatic'}
    try:
        import numpy as np
        import torch
        from . import vadclip
        from .evaluation import evaluate
        from .extract import ReferenceImageEncoder, event_coverage, extract_reference, image_crop, validate_extraction
        from .timeline import candidates, restore_tokens, to_frames

        prepared_path, manifest = load_prepared(spec.key, resources, dataset_config)
        videos = select_videos(manifest['videos'], mode, requested)
        required_input = 'raw_path' if raw_mode else 'feature_path'
        if not cache_folder and any(not video.get(required_input) for video in videos):
            prepare_input = ('both' if spec.key == 'xd' else 'raw') if required_input == 'raw_path' else 'official'
            raise ValueError(f'Setup lacks {required_input}; run prepare_dataset.py with --input {prepare_input}')
        report.update(setup_manifest=str(prepared_path), setup_id=manifest['setup_id'],
                      dataset_identity_sha256=manifest['dataset_identity_sha256'])
        report['input_sha256'].update(experiment_config=sha256(exp_path), dataset_config=sha256(data_path),
                                      setup_manifest=sha256(prepared_path), gt=manifest['frame_gt_sha256'])
        if local_path:
            report['input_sha256']['local_config'] = sha256(project_path(local_path))
        for key in ('clip_checkpoint', 'vadclip_checkpoint'):
            report['input_sha256'][key] = sha256(resource_path(resources, key))
        for field in ('segment_gt', 'class_gt', 'fine_grained_source', 'official_list'):
            if dataset_config.get(field):
                report['input_sha256'][field] = sha256(project_path(dataset_config[field]))
        report['package_sha256'] = package_hashes()
        report['vendored_source_sha256'] = {
            path.relative_to(ROOT).as_posix(): sha256(path)
            for path in sorted((ROOT / 'VadCLIP/src').rglob('*.py'))}
        report['input_sha256']['tokenizer_vocabulary'] = sha256(ROOT / 'VadCLIP/src/clip/bpe_simple_vocab_16e6.txt.gz')
        record_provenance(report)
        vadclip.check_device(device)
        env = environment(device, torch)
        report['environment'] = env
        recipe = dataset_config.get('extraction') if experiment['input_type'] == 'raw' else None
        if experiment['input_type'] == 'raw' and not recipe:
            raise ValueError('Raw-video evaluation requires a dataset extraction recipe')
        if recipe:
            validate_extraction(recipe)
            report['extraction_config'] = recipe
            if recipe['temporal_mode'] != 'single_frame':
                raise ValueError('Common sampler pipeline requires one reference candidate per block')
            if random_windows and recipe['frame_offset'] != 0:
                raise ValueError('Random windows require a zero-offset reference grid')
            if raw_mode and not env['ffmpeg'] and recipe['color_conversion'] == 'x86_swscale_emulation':
                raise ValueError('Install ffmpeg before raw extraction')
        signature_payload = {'dataset': spec.key, 'data': manifest['dataset_identity_sha256'],
                             'dataset_config_sha256': sha256(data_path), 'recipe': recipe,
                             'model_seed': experiment['model_seed'], 'restoration': experiment['protocol']['restoration'],
                             'model_sources': report['vendored_source_sha256'], 'code': report['package_sha256'],
                             'checkpoints': {key: report['input_sha256'][key]
                                             for key in ('clip_checkpoint', 'vadclip_checkpoint')},
                             'evaluation_inputs': {key: report['input_sha256'].get(key) for key in
                                                   ('gt', 'segment_gt', 'class_gt', 'fine_grained_source', 'tokenizer_vocabulary')},
                             'environment': env}
        if score_reference_folder:
            baseline = check_performance_reference(score_reference_folder, manifest, report, spec, recipe)
            key = 'cached_reference_run' if cache_folder else 'performance_reference_run'
            report[key] = str(score_reference_folder)
            signature_payload[key + '_sha256'] = baseline['loaded_summary_sha256']
            if cache_folder and not (cache_folder / 'features').is_dir():
                raise ValueError('Cached reference must include features/ from --save-features')
        report['comparison_signature'] = fingerprint(signature_payload)
        if not score_reference_folder:
            baseline = check_reference(experiment, manifest, report['comparison_signature'])
        if baseline:
            report['reference_summary_sha256'] = baseline['loaded_summary_sha256']
        fixed_seed = experiment['model_seed']
        random.seed(fixed_seed)
        np.random.seed(fixed_seed)
        torch.manual_seed(fixed_seed)
        options = vadclip.default_options(seed=fixed_seed, dataset=spec.key)
        model = vadclip.load_model(options, device, resource_path(resources, 'clip_checkpoint'),
                                   resource_path(resources, 'vadclip_checkpoint'))
        report['model_options'] = {key: getattr(options, key) for key in vadclip.MODEL_OPTION_KEYS}
        report['prompts'] = list(spec.prompts)
        encoder = preprocess = None
        warm_started = time.perf_counter()
        if raw_mode:
            from clip import clip
            from PIL import Image
            visual = deepcopy(model.clipmodel.visual).eval().requires_grad_(False)
            if recipe['encoder_precision'] == 'float32':
                visual.float()
            encoder = ReferenceImageEncoder(visual)
            preprocess = clip._transform(visual.input_resolution)
            black = image_crop(np.zeros((256, 340, 3), dtype=np.uint8), recipe['crop_type'])
            images = preprocess(Image.fromarray(black)).unsqueeze(0).repeat(recipe['encoder_batch_size'], 1, 1, 1)
            images = images.to(device=device, dtype=visual.conv1.weight.dtype)
            for _ in range(recipe['warmup_batches']):
                with torch.no_grad():
                    encoder.encode_image(images)
                vadclip.synchronize(device)
                vadclip.infer_feature(model, np.zeros((options.visual_length, 512), dtype=np.float32),
                                     options.visual_length, spec.prompts, device)
            del images
        report['warmup_seconds'] = time.perf_counter() - warm_started
        frames1, frames2, dense_classes = [], [], []
        covered = events = 0
        window_frames = 16 * window_stride(sampling['budget']) if random_windows else None
        if random_windows:
            report['sampling'].update(window_frames=window_frames, score_anchor='window_start')
        loop_started = time.perf_counter()
        for video in videos:
            report['current_video'] = video['filename']
            stem = f'{video["official_csv_index"]:03d}_{Path(video["filename"]).stem}'
            if cache_folder:
                path = cache_folder / 'features' / f'{stem}.npz'
                input_sha = sha256(path)
            else:
                path = Path(video['raw_path'] if raw_mode else video['feature_path'])
                input_sha = video['sha256'] if raw_mode else video['official_feature_sha256']
                if sha256(path) != input_sha:
                    raise ValueError(f'Input content changed since preparation: {video["filename"]}')
            vadclip.synchronize(device)
            started = time.perf_counter()
            grid = candidates(video['reference_tokens'], frame_offset=recipe['frame_offset'] if recipe else 0)
            derived_seed = video_seed(sampling['seed'], spec.key, video['filename'])
            selected_frames = choose(grid, sampling['method'], sampling['budget'], derived_seed)
            sampling_seconds = time.perf_counter() - started
            if recipe:
                dtype = np.dtype(video['feature_dtype'] if recipe['feature_storage_dtype'] == 'official' else 'float32')
            if raw_mode:
                # None retains the legacy Full path; uniform/random at 100% still
                # exercise the explicit selection path for equivalence checks.
                selection_arg = None if sampling['method'] == 'reference' or random_windows else selected_frames
                frame_selector = None
                if random_windows:
                    def frame_selector(frame_count):
                        return select_window_frames(selected_frames, window_frames, frame_count,
                                                    video['reference_tokens'], derived_seed)
                feature, selection, metadata = extract_reference(
                    path, encoder, preprocess, device, recipe, dtype,
                    expected_tokens=video['reference_tokens'], selected_frame_indices=selection_arg,
                    frame_selector=frame_selector)
                if random_windows:
                    selection['selected_score_anchor_indices'] = np.asarray(selected_frames, dtype=np.int64)
                elif not np.array_equal(selection['selected_frame_indices'], selected_frames):
                    raise ValueError('Decoder did not encode precisely the sampler selection')
            elif cache_folder:
                feature, selection = load_cached_selection(path, grid, selected_frames, dtype)
            else:
                feature = np.load(path, allow_pickle=False)
                if feature.shape != (video['reference_tokens'], 512) or str(feature.dtype) != video['feature_dtype']:
                    raise ValueError('Official feature shape/dtype changed')
                selection = {'selected_frame_indices': np.asarray(selected_frames, dtype=np.int64)}
            if not raw_mode:
                metadata = {'timing_seconds': {}, 'encoder_input_images': 0,
                            'candidate_frames': len(grid), 'selected_frames': len(selected_frames),
                            'selected_fraction_of_candidates': len(selected_frames) / len(grid)}
            if feature.shape != (len(selected_frames), 512):
                raise ValueError('Feature count differs from the actual selection K')
            vad_started = time.perf_counter()
            score1, score2, probabilities, detail = vadclip.infer_feature(
                model, feature, options.visual_length, spec.prompts, device)
            vad_seconds = time.perf_counter() - vad_started
            restore_started = time.perf_counter()
            restoration = experiment['protocol']['restoration']
            dense1 = restore_tokens(score1, selected_frames, grid, restoration)
            dense2 = restore_tokens(score2, selected_frames, grid, restoration)
            class_grid = restore_tokens(probabilities, selected_frames, grid, restoration)
            frame1, frame2 = to_frames(dense1), to_frames(dense2)
            restore_seconds = time.perf_counter() - restore_started
            vadclip.synchronize(device)
            elapsed = time.perf_counter() - started
            if len(frame1) != video['evaluation_frames']:
                raise ValueError('Restoration failed to retain the original full evaluation length')
            coverage = event_coverage(selection['selected_frame_indices'], video['intervals_raw'])
            covered += coverage['covered_events']
            events += coverage['total_events']
            score_path = output / 'scores' / f'{stem}.npz'
            np.savez_compressed(score_path, branch1_selected_scores=score1, branch2_selected_scores=score2,
                                class_selected_scores=probabilities, reference_branch1_scores=dense1,
                                reference_branch2_scores=dense2, reference_class_scores=class_grid,
                                reference_frame_indices=np.asarray(grid), **selection)
            if save_features:
                np.savez_compressed(output / 'features' / f'{stem}.npz', features=feature, **selection)
            timing = {**metadata['timing_seconds'], 'sampler': sampling_seconds,
                      'vad_total': vad_seconds, 'score_restore': restore_seconds,
                      'end_to_end' if raw_mode else 'cached_feature_call': elapsed}
            record = {**metadata, **detail, 'official_csv_index': video['official_csv_index'],
                      'video': video['filename'], 'case': 'full_video', 'prefix_only': False,
                      'gt_offset': video['gt_offset'], 'reference_tokens': video['reference_tokens'],
                      'selected_tokens': len(selected_frames), 'evaluation_frames': len(frame1),
                      'input_sha256': input_sha, 'sampler_seed': derived_seed,
                      'score_file': str(score_path.relative_to(output)), 'event_coverage': coverage,
                      'timing_seconds': timing}
            report['videos'].append(record)
            report.pop('current_video', None)
            frames1.append(frame1)
            frames2.append(frame2)
            dense_classes.append(class_grid)
            write_json(output / 'summary.json', report)
            print(f'{len(report["videos"])}/{len(videos)} {video["filename"]}: '
                  f'{len(selected_frames)}/{len(grid)} candidates; call {elapsed:.3f}s', flush=True)
        report['processing_loop_wall_seconds'] = time.perf_counter() - loop_started
        report['encoder_input_images'] = sum(video['encoder_input_images'] for video in report['videos'])
        report['event_coverage'] = {'covered_events': covered, 'total_events': events,
                                    'fraction': covered / events if events else None}
        report['timing_scope'] = {
            'included': 'Candidate construction and selection; raw decoding/preprocess/transfers/CLIP if raw; VadCLIP and timeline restoration',
            'excluded': 'Setup/imports/input hashing/model loading/warm-up/GT analysis/metrics/output saves',
            'io_condition': 'Source hash read just before timed processing; file cache uncontrolled and likely warm',
            'concurrency': 'Not enforced; final timings require an otherwise idle shared machine',
            'repeats_in_this_run': 1, 'synchronization': device in ('cuda', 'mps'),
            'full_frame_image_dump': False, 'speedup_eligible': raw_mode and not performance_folder}
        time_key = 'end_to_end' if raw_mode else 'cached_feature_call'
        total_time = sum(video['timing_seconds'][time_key] for video in report['videos'])
        report['end_to_end_seconds' if raw_mode else 'cached_feature_seconds'] = total_time
        if mode == 'full':
            metric_started = time.perf_counter()
            report['metrics'], all1, all2 = evaluate(dataset_config, manifest, report['videos'], frames1, frames2, dense_classes)
            report['metric_seconds'] = time.perf_counter() - metric_started
            np.savez_compressed(output / 'frame_scores.npz', branch1=all1, branch2=all2)
            if baseline:
                current = report['metrics']['primary']['value']
                previous = baseline['metrics']['primary']['value']
                report['reference_comparison'] = {
                    'reference_run': baseline['run_id'], 'primary_delta_pp': 100 * (current - previous),
                }
                if score_reference_folder:
                    report['reference_comparison']['scope'] = 'performance_only'
                if raw_mode and not performance_folder:
                    report['reference_comparison'].update(speedup=baseline['end_to_end_seconds'] / total_time,
                                                          timing_repeated=False,
                                                          final_claim_requires_repeated_measurements=True)
        else:
            report['eligible_for_final_benchmark'] = False
        report['success'] = True
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        traceback.print_exc(file=sys.stderr)
    finally:
        report['finished_at'] = now().isoformat()
        write_json(output / 'summary.json', report)
        print(f'Report: {output / "summary.json"}', flush=True)
    return 0 if report['success'] else 1
