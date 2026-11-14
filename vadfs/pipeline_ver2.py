"""Separate raw-window ver2 pipeline; frozen CLIP/VadCLIP evaluation reused."""

from copy import deepcopy
from pathlib import Path
import random
import sys
import time
import traceback

from .config import load_experiment, resource_path
from .datasets import get_dataset
from .paths import ROOT, package_hashes, project_path, sha256
from .pipeline import environment, fingerprint, select_videos
from .preparation import load_prepared
from .runs import create_run_dir, now, record_provenance, write_json
from .samplers import video_seed
from .window_sampling_ver2 import (METHODS_VER2, SPEC_VER2, collect_ver2, projection_hash_ver2,
                                   projection_ver2, select_ver2, windows_ver2)


def validate_ver2(experiment, recipe, mode):
    sampling = experiment['sampling']
    if (experiment.get('protocol_version') != 2 or experiment.get('input_type') != 'raw'
            or experiment.get('reference_run') or experiment.get('feature_source')
            or mode not in ('smoke', 'full') or sampling['method'] not in METHODS_VER2
            or sampling['budget'] not in (1., .5, .25, .125)
            or experiment.get('selection_spec') != SPEC_VER2
            or experiment['protocol'].get('restoration') != 'nearest'
            or experiment['protocol'].get('score_anchor') != 'window_start'
            or experiment['protocol'].get('status') != 'draft'
            or recipe.get('frame_offset') != 0 or recipe.get('temporal_mode') != 'single_frame'
            or recipe.get('tail_policy') != 'match_official_rows'):
        raise ValueError('Experiment does not match frozen ver2 draft protocol')
    for value in (experiment['model_seed'], sampling['seed']):
        if type(value) is not int or not 0 <= value < 2**32:
            raise ValueError('Invalid seed')
    if sampling['method'] != 'random_window' and sampling['seed'] != 0:
        raise ValueError('Deterministic methods use sampler seed 0')


def run_experiment_ver2(config_path, local_path=None, mode='smoke', requested=None,
                         device_override=None, seed_override=None, setup_override=None, save_features=False):
    experiment, dataset, resources, exp_path, data_path = load_experiment(config_path, local_path)
    if seed_override is not None:
        experiment['sampling']['seed'] = seed_override
    if setup_override:
        resources['setup_manifest'] = str(setup_override)
    recipe = dataset.get('extraction', {})
    validate_ver2(experiment, recipe, mode)
    sampling = experiment['sampling']
    spec = get_dataset(dataset['key']).SPEC
    device = device_override or resources.get('device', 'cpu')
    if device not in ('cpu', 'cuda', 'mps'):
        raise ValueError('Unknown device')
    run_id, output = create_run_dir(
        f'{spec.key}_{sampling["method"]}_{sampling["budget"]:g}_s{sampling["seed"]}_ver2',
        mode, ('scores', 'features'), resource_path(resources, 'output_root'))
    report = {'schema_version': 2, 'protocol_version': 2, 'run_id': run_id, 'started_at': now().isoformat(),
              'success': False, 'mode': mode, 'dataset': spec.name, 'dataset_key': spec.key,
              'input_type': 'raw', 'sampling': deepcopy(sampling), 'experiment': experiment,
              'resources': resources, 'command': sys.argv, 'output_dir': str(output),
              'input_sha256': {}, 'videos': [], 'metrics': None,
              'experiment_scope': 'raw_window_exploration_ver2', 'candidate_space': 'raw_window',
              'score_anchor': 'window_start', 'eligible_for_final_benchmark': False,
              'speedup_eligible': True, 'validation_status': 'draft; user-run checks required'}
    try:
        import numpy as np
        import torch
        from . import vadclip
        from .evaluation import evaluate
        from .extract import ReferenceImageEncoder, extract_reference, image_crop, validate_extraction
        from .timeline import candidates, restore_tokens, to_frames

        prepared, manifest = load_prepared(spec.key, resources, dataset)
        videos = select_videos(manifest['videos'], mode, requested)
        if any(not v.get('raw_path') for v in videos):
            raise ValueError('Prepared raw manifest required; use prepare_dataset.py first')
        write_json(output / 'manifest.json', manifest)
        validate_extraction(recipe)
        report.update(setup_manifest=str(prepared), setup_id=manifest['setup_id'],
                      dataset_identity_sha256=manifest['dataset_identity_sha256'], extraction_config=recipe,
                      prompts=list(spec.prompts), package_sha256=package_hashes())
        report['input_sha256'].update(experiment_config=sha256(exp_path), dataset_config=sha256(data_path),
                                      setup_manifest=sha256(prepared), gt=manifest['frame_gt_sha256'])
        if local_path:
            report['input_sha256']['local_config'] = sha256(project_path(local_path))
        for key in ('clip_checkpoint', 'vadclip_checkpoint'):
            report['input_sha256'][key] = sha256(resource_path(resources, key))
        for key in ('segment_gt', 'class_gt', 'fine_grained_source', 'official_list'):
            report['input_sha256'][key] = sha256(project_path(dataset[key]))
        report['input_sha256']['tokenizer_vocabulary'] = sha256(ROOT / 'VadCLIP/src/clip/bpe_simple_vocab_16e6.txt.gz')
        report['entrypoint_sha256'] = sha256(ROOT / 'scripts/run_experiment_ver2.py')
        report['vendored_source_sha256'] = {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted((ROOT/'VadCLIP/src').rglob('*.py'))}
        record_provenance(report)
        vadclip.check_device(device)
        report['environment'] = environment(device, torch)
        if recipe['color_conversion'] == 'x86_swscale_emulation' and not report['environment']['ffmpeg']:
            raise ValueError('ffmpeg required for UCF raw extraction')
        q = projection_ver2()
        np.savez_compressed(output / 'projection_ver2.npz', q=q)
        report['projection'] = {'q_sha256': projection_hash_ver2(q), 'file': 'projection_ver2.npz',
                                'file_sha256': sha256(output / 'projection_ver2.npz'), 'specification': SPEC_VER2}
        # Deliberately independent of method/budget/seed; sampler protocol is separate.
        report['comparison_signature'] = fingerprint({
            'protocol_version': 2, 'selection_spec': SPEC_VER2, 'projection': projection_hash_ver2(q),
            'dataset': spec.key, 'identity': manifest['dataset_identity_sha256'], 'recipe': recipe,
            'model_seed': experiment['model_seed'], 'prompts': list(spec.prompts),
            'package': report['package_sha256'], 'vendored': report['vendored_source_sha256'],
            'inputs': {k:v for k,v in report['input_sha256'].items() if k not in ('experiment_config','local_config')},
            'environment': report['environment']})
        fixed = experiment['model_seed']
        random.seed(fixed)
        np.random.seed(fixed)
        torch.manual_seed(fixed)
        options = vadclip.default_options(seed=fixed, dataset=spec.key)
        model = vadclip.load_model(options, device, resource_path(resources, 'clip_checkpoint'), resource_path(resources, 'vadclip_checkpoint'))
        report['model_options'] = {k: getattr(options,k) for k in vadclip.MODEL_OPTION_KEYS}
        from clip import clip
        from PIL import Image
        visual = deepcopy(model.clipmodel.visual).eval().requires_grad_(False)
        if recipe['encoder_precision'] == 'float32':
            visual.float()
        encoder = ReferenceImageEncoder(visual)
        preprocess = clip._transform(visual.input_resolution)
        tick = time.perf_counter()
        black = image_crop(np.zeros((256,340,3), dtype=np.uint8), recipe['crop_type'])
        images = preprocess(Image.fromarray(black)).unsqueeze(0).repeat(recipe['encoder_batch_size'],1,1,1)
        images = images.to(device=device, dtype=visual.conv1.weight.dtype)
        for _ in range(recipe['warmup_batches']):
            with torch.no_grad():
                encoder.encode_image(images)
            vadclip.synchronize(device)
            vadclip.infer_feature(model, np.zeros((options.visual_length,512), dtype=np.float32), options.visual_length, spec.prompts, device)
        del images
        report['warmup_seconds'] = time.perf_counter()-tick
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        frame1_all, frame2_all, class_all = [], [], []
        for video in videos:
            report['current_video'] = video['filename']
            path = Path(video['raw_path'])
            input_hash = sha256(path)
            if input_hash != video['sha256']:
                raise ValueError('Raw input changed since preparation')
            n = video['reference_tokens']
            grid = candidates(n)
            seed = video_seed(sampling['seed'], spec.key, video['filename'])
            dtype = np.dtype(video['feature_dtype'] if recipe['feature_storage_dtype']=='official' else 'float32')
            vadclip.synchronize(device)
            start = time.perf_counter()
            first, selected, detail = None, None, None
            if sampling['method'] in ('d_optimal','a_optimal','e_optimal'):
                rows, first = collect_ver2(path, n)
                bounds = windows_ver2(n, first['decoded_frames'], sampling['budget'])
                tick = time.perf_counter()
                selected, detail = select_ver2(rows, bounds, sampling['method'], seed, q)
                selection_seconds = time.perf_counter()-tick
                del rows
            else:
                selection_seconds = 0.
            def selector(frame_count):
                nonlocal selected, detail, selection_seconds
                bounds = windows_ver2(n, frame_count, sampling['budget'])
                if first:
                    if frame_count != first['decoded_frames']:
                        raise ValueError('Two decode passes disagree on frame count')
                else:
                    tick = time.perf_counter()
                    selected, detail = select_ver2(None, bounds, sampling['method'], seed)
                    selection_seconds = time.perf_counter()-tick
                return selected
            feature, arrays, metadata = extract_reference(path, encoder, preprocess, device, recipe, dtype,
                                                          expected_tokens=n, frame_selector=selector)
            if first and (metadata['decoded_frames'] != first['decoded_frames'] or abs(metadata['fps']-first['fps']) > 1e-8):
                raise ValueError('Two passes disagree on decoded metadata')
            bounds = windows_ver2(n, metadata['decoded_frames'], sampling['budget'])
            anchors = [a for a,_ in bounds]
            if (not np.array_equal(arrays['selected_frame_indices'],selected) or feature.shape != (len(bounds),512)
                    or metadata['encoder_input_images'] != len(bounds)):
                raise ValueError('Encoder input differs from one-per-window selection')
            arrays['selected_score_anchor_indices'] = np.asarray(anchors, dtype=np.int64)
            tick = time.perf_counter()
            score1, score2, probabilities, infer_detail = vadclip.infer_feature(model, feature, options.visual_length, spec.prompts, device)
            vadclip.synchronize(device)
            vad_seconds = time.perf_counter()-tick
            tick = time.perf_counter()
            dense1 = restore_tokens(score1, anchors, grid, 'nearest')
            dense2 = restore_tokens(score2, anchors, grid, 'nearest')
            dense_class = restore_tokens(probabilities, anchors, grid, 'nearest')
            frame1, frame2 = to_frames(dense1), to_frames(dense2)
            if len(frame1) != video['evaluation_frames']:
                raise ValueError('Full evaluation timeline length changed')
            vadclip.synchronize(device)
            restore_seconds = time.perf_counter()-tick
            elapsed = time.perf_counter()-start
            # All GT analysis and file writes occur after timed processing.
            intervals = [(max(0,a),min(len(frame1),b)) for a,b in video['intervals_raw'] if min(len(frame1),b)>max(0,a)]
            covered = sum(any(a <= i < b for i in selected) for a,b in intervals)
            stem = f'{video["official_csv_index"]:03d}_{Path(video["filename"]).stem}'
            np.savez_compressed(output/'scores'/f'{stem}.npz', branch1_selected_scores=score1, branch2_selected_scores=score2,
                                class_selected_scores=probabilities, reference_branch1_scores=dense1,
                                reference_branch2_scores=dense2, reference_class_scores=dense_class,
                                reference_frame_indices=np.asarray(grid), **arrays)
            if save_features:
                np.savez_compressed(output/'features'/f'{stem}.npz', features=feature, **arrays)
            timing = {**metadata['timing_seconds'], **(first['timing_seconds'] if first else {}),
                      'optimal_selection': selection_seconds, 'vad_total':vad_seconds,
                      'score_restore':restore_seconds, 'end_to_end':elapsed}
            record = {**metadata, **infer_detail, 'official_csv_index':video['official_csv_index'],
                      'video':video['filename'], 'case':'full_video', 'prefix_only':False,
                      'gt_offset':video['gt_offset'], 'reference_tokens':n, 'selected_tokens':len(selected),
                      'evaluation_frames':len(frame1), 'input_sha256':input_hash, 'sampler_seed':seed,
                      'score_file':f'scores/{stem}.npz', 'sampler_diagnostics':detail,
                      'selection_coverage_stop':bounds[-1][1], 'decode_passes':2 if first else 1,
                      'event_coverage':{'valid_covered_events':covered,'valid_total_events':len(intervals)},
                      'timing_seconds':timing}
            report['videos'].append(record)
            report.pop('current_video',None)
            frame1_all.append(frame1); frame2_all.append(frame2); class_all.append(dense_class)
            write_json(output/'summary.json',report)
            print(f'{len(report["videos"])}/{len(videos)} {video["filename"]}: {len(selected)} frames; E2E {elapsed:.3f}s',flush=True)
        report['encoder_input_images'] = sum(v['encoder_input_images'] for v in report['videos'])
        covered = sum(v['event_coverage']['valid_covered_events'] for v in report['videos'])
        total_events = sum(v['event_coverage']['valid_total_events'] for v in report['videos'])
        report['event_coverage'] = {'valid_covered_events': covered, 'valid_total_events': total_events,
                                    'valid_fraction': covered / total_events if total_events else None}
        report['end_to_end_seconds'] = sum(v['timing_seconds']['end_to_end'] for v in report['videos'])
        report['memory'] = {'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None,
                            'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved() if device=='cuda' else None,
                            'process_peak_rss_bytes':None, 'scope':'whole run; CPU/FFmpeg child RSS unmeasured'}
        report['timing_scope'] = {'included':'wall time: all decode passes, thumbnail, projection/selection, CLIP, VAD, restore',
                                  'excluded':'hashing, model setup, warmup, GT analysis, metrics, output saves',
                                  'repeats_in_this_run':1, 'synchronization':device in ('cuda','mps'),
                                  'io_condition':'raw hash before processing; cache uncontrolled',
                                  'concurrency':'not enforced; user must ensure idle machine'}
        if mode=='full':
            tick = time.perf_counter()
            report['metrics'], all1, all2 = evaluate(dataset,manifest,report['videos'],frame1_all,frame2_all,class_all)
            report['metric_seconds'] = time.perf_counter()-tick
            np.savez_compressed(output/'frame_scores.npz',branch1=all1,branch2=all2)
        report['success'] = True
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        traceback.print_exc(file=sys.stderr)
    finally:
        report['finished_at'] = now().isoformat()
        write_json(output/'summary.json',report)
        print(f'Report: {output / "summary.json"}',flush=True)
    return 0 if report['success'] else 1
