"""Stage B: fixed-target neighbor replacement. Does not change model computation."""

import argparse
import csv
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vadfs.config import load_experiment, resource_path
from vadfs.context_ver4 import SPEC_VER4, load_ver3, replace_neighbors, score_arrays, target_positions
from vadfs.datasets import get_dataset
from vadfs.paths import package_hashes, project_path, sha256
from vadfs.preparation import load_prepared
from vadfs.runs import create_run_dir, now, record_provenance, write_json


def check_current(report, experiment, dataset, resources):
    """Allow new diagnostic files but keep the archived model/evaluation fixed."""
    if (experiment['context_ver4'] != SPEC_VER4 or experiment['model_seed'] != 234
            or experiment['model_seed'] != report['experiment']['model_seed']
            or experiment['sampling'] != {'method': 'uniform_ver3', 'budget': 0.125, 'seed': 0}
            or experiment['protocol']['restoration'] != 'nearest'
            or experiment['protocol']['status'] != 'draft' or dataset['key'] != 'xd'
            or dataset['extraction'] != report['extraction_config']):
        raise ValueError('Configuration differs from the fixed ver4 diagnostic protocol')
    paths = {key: resource_path(resources, key) for key in ('clip_checkpoint', 'vadclip_checkpoint')}
    paths.update({key: project_path(dataset[field]) for key, field in (
        ('gt', 'frame_gt'), ('segment_gt', 'segment_gt'), ('class_gt', 'class_gt'),
        ('official_list', 'official_list'), ('fine_grained_source', 'fine_grained_source'))})
    paths['tokenizer_vocabulary'] = ROOT / 'VadCLIP/src/clip/bpe_simple_vocab_16e6.txt.gz'
    inputs = {key: sha256(path) for key, path in paths.items()}
    if any(inputs[key] != report['input_sha256'].get(key) for key in inputs):
        raise ValueError('Checkpoint/GT/split/tokenizer content differs from ver3')
    current = package_hashes()
    for name in ('vadfs/vadclip.py', 'vadfs/evaluation.py', 'vadfs/metrics.py',
                 'vadfs/timeline.py', 'vadfs/color.py'):
        if not current.get(name) or current[name] != report['package_sha256'].get(name):
            raise ValueError(f'Frozen source changed: {name}')
    vendored = {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted((ROOT / 'VadCLIP/src').rglob('*.py'))}
    if vendored != report['vendored_source_sha256']:
        raise ValueError('Vendored model implementation changed')
    return inputs, current, vendored


def write_csv(path, rows, fields):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_targets(rows):
    """Descriptive paired effects; no independence or significance assumptions."""
    output = []
    for branch in (1, 2):
        for label in (0, 1):
            subset = [r for r in rows if r['branch'] == branch and r['gt'] == label]
            differences = np.asarray([r['near_minus_baseline'] for r in subset])
            videos = sorted({r['video'] for r in subset})
            video_means = [np.mean([r['near_minus_baseline'] for r in subset if r['video'] == name])
                           for name in videos]
            output.append({
                'branch': branch, 'gt': label, 'n_targets': len(subset), 'n_videos': len(videos),
                'mean_delta': float(differences.mean()) if len(subset) else None,
                'median_delta': float(np.median(differences)) if len(subset) else None,
                'increased_fraction': float(np.mean(differences > 0)) if len(subset) else None,
                'video_mean_delta': float(np.mean(video_means)) if videos else None,
                'video_mean_sample_std': float(np.std(video_means, ddof=1)) if len(videos) > 1 else None,
                'mean_abs_error_change_vs_full': float(np.mean([r['abs_error_change_vs_full'] for r in subset])) if subset else None,
                'closer_to_full_fraction': float(np.mean([r['abs_error_change_vs_full'] < 0 for r in subset])) if subset else None,
                'max_repeat_abs_delta': max((abs(r['repeat_minus_baseline']) for r in subset), default=None)})
    return output


def run(sweep, config, local, mode, device):
    experiment, dataset, resources, config_path, dataset_path = load_experiment(config, local)
    groups, manifest, gt, provenance = load_ver3(sweep)
    source_path, archived = groups[('uniform_ver3', 0)]
    full_path, full_report = groups[('reference', 0)]
    inputs, sources, vendored = check_current(archived, experiment, dataset, resources)
    resources['setup_manifest'] = archived['setup_manifest']
    prepared_path, prepared = load_prepared('xd', resources, dataset)
    if prepared != manifest:
        raise ValueError('Prepared manifest differs from source sweep')
    cache_folder = project_path(archived['cached_reference_run'])
    if sha256(cache_folder / 'summary.json') != archived['reference_summary_sha256']:
        raise ValueError('Raw cache source summary changed')
    # Heavy imports occur only when this user-run stage is actually invoked.
    import torch
    from vadfs import vadclip
    from vadfs.pipeline import environment, load_cached_reference, select_videos

    videos = select_videos(manifest['videos'], mode, None)
    _, output = create_run_dir('xd_context_probe_ver4', mode, ('scores',), resource_path(resources, 'output_root'))
    status = {'success': False, 'error': None, 'mode': mode, 'started_at': now().isoformat(),
              'scope': 'targeted_diagnostic_no_whole_timeline_AP_no_speedup',
              'protocol': SPEC_VER4, 'experiment': experiment, 'resources': resources,
              'inputs': provenance, 'input_sha256': inputs, 'package_sha256': sources,
              'vendored_source_sha256': vendored, 'videos': [], 'targets': [],
              'dataset_identity_sha256': archived['dataset_identity_sha256'],
              'command': sys.argv, 'model_seed': 234, 'expected_videos': len(videos)}
    status['input_sha256'].update(config=sha256(config_path), dataset_config=sha256(dataset_path),
                                  setup_manifest=sha256(prepared_path), entrypoint=sha256(Path(__file__)))
    if local:
        status['input_sha256']['local_config'] = sha256(project_path(local))
    try:
        record_provenance(status)
        vadclip.check_device(device)
        status['environment'] = environment(device, torch)
        random.seed(234)
        np.random.seed(234)
        torch.manual_seed(234)
        options = vadclip.default_options(seed=234, dataset='xd')
        status['model_options'] = {key: getattr(options, key) for key in vadclip.MODEL_OPTION_KEYS}
        if (status['model_options'] != archived['model_options'] or options.visual_length != 256
                or options.attn_window != 64):
            raise ValueError('Model options/window differ from archived Uniform')
        spec = get_dataset('xd').SPEC
        if list(spec.prompts) != archived['prompts']:
            raise ValueError('Prompts differ')
        model = vadclip.load_model(options, device, resource_path(resources, 'clip_checkpoint'),
                                  resource_path(resources, 'vadclip_checkpoint'))
        for video in videos:
            name = video['filename']
            n = video['reference_tokens']
            targets = target_positions(n)
            record = {'video': name, 'reference_tokens': n, 'selected_tokens': (n + 7) // 8,
                      'target_slots': targets, 'skip_reason': None if targets else 'no_eligible_interior_slot'}
            if targets:
                stem = f'{video["official_csv_index"]:03d}_{Path(name).stem}'
                cache_path = cache_folder / 'features' / f'{stem}.npz'
                cache_hash = sha256(cache_path)
                if cache_hash != archived['cache_input_sha256'][name]:
                    raise ValueError(f'Cached feature bytes differ: {name}')
                grid = np.arange(n, dtype=np.int64) * 16
                features, _ = load_cached_reference(cache_path, grid, np.dtype(video['feature_dtype']))
                base_rows = list(range(0, n, 8))
                baseline1, baseline2, _, baseline_detail = vadclip.infer_feature(
                    model, features[base_rows], 256, spec.prompts, device)
                repeat1, repeat2, _, _ = vadclip.infer_feature(
                    model, features[base_rows], 256, spec.prompts, device)
                _, _, saved_scores, score_source = score_arrays(source_path, archived, video)
                _, _, full_scores, full_source = score_arrays(full_path, full_report, video)
                baseline = {1: baseline1, 2: baseline2}
                repeat = {1: repeat1, 2: repeat2}
                for b in (1, 2):
                    archived_error = float(np.max(np.abs(baseline[b] - saved_scores[b])))
                    replay_error = float(np.max(np.abs(baseline[b] - repeat[b])))
                    if archived_error > SPEC_VER4['stored_score_atol']:
                        raise ValueError(f'Uniform reproduction failed: {name} branch{b} maxabs={archived_error}')
                    if replay_error > SPEC_VER4['repeat_score_atol']:
                        raise ValueError(f'Unchanged replay is unstable: {name} branch{b} maxabs={replay_error}')
                    record[f'branch{b}_archived_maxabs'] = archived_error
                    record[f'branch{b}_repeat_maxabs'] = replay_error
                variant_rows, variant1, variant2 = [], [], []
                for slot in targets:
                    original, changed = replace_neighbors(n, slot)
                    if original != base_rows or not np.array_equal(features[original[slot]], features[changed[slot]]):
                        raise ValueError('Target feature changed')
                    near1, near2, _, near_detail = vadclip.infer_feature(
                        model, features[changed], 256, spec.prompts, device)
                    for field in ('token_count', 'window_count', 'declared_lengths', 'physical_valid_lengths'):
                        if near_detail[field] != baseline_detail[field]:
                            raise ValueError('Target intervention changed window/padding layout')
                    variant_rows.append(changed)
                    variant1.append(near1)
                    variant2.append(near2)
                    frame = original[slot] * 16
                    label = int(gt[video['gt_offset'] + frame])  # GT enters only after target selection/inference.
                    for branch, near in ((1, near1), (2, near2)):
                        full_score = float(full_scores[branch][original[slot]])
                        baseline_error = abs(float(baseline[branch][slot]) - full_score)
                        near_error = abs(float(near[slot]) - full_score)
                        status['targets'].append({
                            'video': name, 'target_slot': slot, 'target_frame': frame,
                            'window_index': slot // 256, 'position_in_window': slot % 256,
                            'attention_block_index': slot // 64, 'position_in_attention_block': slot % 64,
                            'reference_tokens': n, 'selected_tokens': len(base_rows), 'gt': label,
                            'branch': branch, 'baseline_score': float(baseline[branch][slot]),
                            'full_score': full_score, 'baseline_abs_error_vs_full': baseline_error,
                            'near_abs_error_vs_full': near_error, 'abs_error_change_vs_full': near_error - baseline_error,
                            'repeat_score': float(repeat[branch][slot]), 'near_score': float(near[slot]),
                            'near_minus_baseline': float(near[slot] - baseline[branch][slot]),
                            'repeat_minus_baseline': float(repeat[branch][slot] - baseline[branch][slot])})
                score_path = output / 'scores' / f'{stem}_ver4.npz'
                np.savez_compressed(score_path, uniform_rows=np.asarray(base_rows),
                                    target_slots=np.asarray(targets), variant_rows=np.asarray(variant_rows),
                                    baseline_branch1=baseline1, baseline_branch2=baseline2,
                                    repeat_branch1=repeat1, repeat_branch2=repeat2,
                                    near_branch1=np.asarray(variant1), near_branch2=np.asarray(variant2))
                record.update(cache_sha256=cache_hash, archived_score=score_source, full_score_source=full_source,
                              score_file=str(score_path.relative_to(output)), score_sha256=sha256(score_path),
                              baseline_inference=baseline_detail)
            status['videos'].append(record)
            write_json(output / 'summary_ver4.json', status)
            print(f'{len(status["videos"])}/{len(videos)} {name}: {len(targets)} fixed targets', flush=True)
        aggregate = summarize_targets(status['targets'])
        status['aggregates'] = aggregate
        target_fields = ['video', 'target_slot', 'target_frame', 'window_index', 'position_in_window',
                         'attention_block_index', 'position_in_attention_block',
                         'reference_tokens', 'selected_tokens', 'gt', 'branch', 'baseline_score',
                         'full_score', 'baseline_abs_error_vs_full', 'near_abs_error_vs_full', 'abs_error_change_vs_full',
                         'repeat_score', 'near_score', 'near_minus_baseline', 'repeat_minus_baseline']
        write_csv(output / 'target_scores_ver4.csv', status['targets'], target_fields)
        write_csv(output / 'target_comparison_ver4.csv', aggregate, list(aggregate[0]))
        status['success'] = True
    except BaseException as error:
        status['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        status['finished_at'] = now().isoformat()
        write_json(output / 'summary_ver4.json', status)
        print(f'Report: {output / "summary_ver4.json"}', flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep-ver3', required=True, type=Path)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/experiments/xd_context_ver4.json')
    parser.add_argument('--local', type=Path)
    parser.add_argument('--mode', choices=('smoke', 'full'), default='smoke')
    parser.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cuda')
    args = parser.parse_args()
    run(args.sweep_ver3, args.config, args.local, args.mode, args.device)


if __name__ == '__main__':
    main()
