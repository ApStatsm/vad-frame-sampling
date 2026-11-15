"""Compare successful full performance runs at equal per-video N/K.

Accept run folders, sweep.json or seed aggregate folders. Aggregate run IDs are
resolved under --random-runs-root. Historical source changes are reported and
require explicit acknowledgement; this report never infers speedup from them.
"""

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vadfs.paths import sha256
from vadfs.runs import create_run_dir, write_json
from vadfs.samplers import OPTIMAL_METHODS, budget_count


def expand(path, random_root):
    path = path.expanduser().resolve()
    if path.is_dir():
        path /= 'summary.json'
    report = json.loads(path.read_text())
    if 'complete' in report and 'runs' in report:
        if not report['complete']:
            raise ValueError(f'Incomplete sweep: {path}')
        result = []
        for item in report['runs']:
            source = Path(item['summary'])
            if sha256(source) != item['sha256']:
                raise ValueError('Sweep input changed')
            result.extend(expand(source, random_root))
        return result
    if 'primary_distribution' in report:
        if len(report['runs']) != len(set(report['runs'])):
            raise ValueError('Duplicate aggregate run IDs')
        result = []
        for run_id in report['runs']:
            result.extend(expand(random_root / run_id, random_root))
        if {r['sampling']['seed'] for _, r in result} != set(report['seeds']):
            raise ValueError('Aggregate seed provenance mismatch')
        return result
    if (report.get('schema_version') != 2 or not report.get('success')
            or report.get('mode') != 'full' or not report.get('metrics')):
        raise ValueError(f'Expected a successful full run: {path}')
    return [(path, report)]


def frozen_identity(report):
    fields = ('clip_checkpoint', 'vadclip_checkpoint', 'gt', 'segment_gt', 'class_gt',
              'official_list', 'tokenizer_vocabulary', 'fine_grained_source')
    return {'dataset': report['dataset_key'], 'data': report['dataset_identity_sha256'],
            'inputs': {key: report['input_sha256'].get(key) for key in fields},
            'recipe': report.get('extraction_config'), 'prompts': report['prompts'],
            'model_seed': report['experiment']['model_seed'],
            'restoration': report['experiment']['protocol']['restoration'],
            'model_sources': report['vendored_source_sha256'],
            'frozen_common_sources': {key: report['package_sha256'].get(key) for key in
                                     ('vadfs/vadclip.py', 'vadfs/evaluation.py', 'vadfs/metrics.py',
                                      'vadfs/timeline.py', 'vadfs/color.py')},
            'primary': {key: report['metrics']['primary'][key] for key in ('branch', 'metric')}}


def describe(values):
    return {'n': len(values), 'mean': statistics.mean(values),
            'sample_std': statistics.stdev(values) if len(values) > 1 else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--random-runs-root', type=Path, default=Path('/workspace/vad/results/random/runs'))
    parser.add_argument('--allow-source-differences', action='store_true',
                        help='Acknowledge exploratory performance comparison across recorded source versions')
    parser.add_argument('--output-root', type=Path, default=Path('outputs/analyses'))
    args = parser.parse_args()
    expanded = [item for path in args.inputs for item in expand(path, args.random_runs_root)]
    if len({str(path) for path, _ in expanded}) != len(expanded):
        raise ValueError('Duplicate input runs')
    first = expanded[0][1]
    identity = frozen_identity(first)
    source_differences = any(r['package_sha256'] != first['package_sha256'] for _, r in expanded)
    if source_differences and not args.allow_source_differences:
        raise ValueError('Source versions differ; review before using --allow-source-differences for performance only')
    groups, counts = {}, {}
    q_hashes = {r['projection']['q_sha256'] for _, r in expanded if r.get('projection')}
    cache_ids = {r['cache_identity_sha256'] for _, r in expanded if r.get('cache_identity_sha256')}
    representations = {r['sampling'].get('representation', {}).get('kind', 'cheap_feature_pca')
                       for _, r in expanded if r['sampling']['method'] in OPTIMAL_METHODS}
    if len(q_hashes) > 1 or len(cache_ids) > 1 or len(representations) > 1:
        raise ValueError('Cannot mix different projection matrices, feature caches, or optimal representations')
    expected_videos = {v['video']: (v['reference_tokens'], v['evaluation_frames']) for v in first['videos']}
    for path, report in expanded:
        if frozen_identity(report) != identity:
            raise ValueError(f'Dataset/model/evaluation/timeline differs: {path}')
        if report['input_type'] not in ('raw', 'cached_raw_features'):
            raise ValueError('Use raw Reference and raw/cached sampling results; official features are a separate baseline')
        sampling = report['sampling']
        budget, method = sampling['budget'], sampling['method']
        if {v['video']: (v['reference_tokens'], v['evaluation_frames']) for v in report['videos']} != expected_videos:
            raise ValueError('Different video set/reference timeline across budgets')
        if any(v['selected_tokens'] != budget_count(v['reference_tokens'], budget) for v in report['videos']):
            raise ValueError('Reported per-video selection count violates the declared budget')
        counts_for_run = {v['video']: (v['reference_tokens'], v['selected_tokens'], v['evaluation_frames'])
                          for v in report['videos']}
        if len(counts_for_run) != len(report['videos']):
            raise ValueError('Duplicate videos in a run')
        if budget in counts and counts[budget] != counts_for_run:
            raise ValueError(f'Per-video N/K/evaluation lengths differ at budget {budget}: {path}')
        counts[budget] = counts_for_run
        group = groups.setdefault((budget, method), [])
        if any(r['sampling']['seed'] == sampling['seed'] for _, r in group):
            raise ValueError('Duplicate seeds; timing repeats must be compared separately')
        if group and (report.get('sampler_signature') != group[0][1].get('sampler_signature')
                      or report['comparison_signature'] != group[0][1]['comparison_signature']):
            raise ValueError('Same-method repeats differ in sampler configuration, artifact or environment')
        group.append((path, report))
    rows = []
    for (budget, method), items in sorted(groups.items(), reverse=True):
        reports = [report for _, report in items]
        branch = first['metrics']['primary']['branch']
        distribution = describe([100 * r['metrics']['primary']['value'] for r in reports])
        uniform = groups.get((budget, 'uniform'))
        reference = groups.get((1, 'reference'))
        row = {'dataset': first['dataset_key'],
               'method': 'random_feature' if method == 'random' and reports[0]['input_type'] == 'cached_raw_features' else method,
               'budget': budget,
               'primary_metric': first['metrics']['primary']['metric'],
               'primary_mean_percent': distribution['mean'], 'primary_std_pp': distribution['sample_std'],
               'n_seeds': distribution['n'], 'seeds': [r['sampling']['seed'] for r in reports],
               'selected_frames': sum(v['selected_tokens'] for v in reports[0]['videos']),
               'reference_candidates': sum(v['reference_tokens'] for v in reports[0]['videos']),
               'uniform_delta_pp': (distribution['mean'] - statistics.mean(
                   100 * r['metrics']['primary']['value'] for _, r in uniform)) if uniform else None,
               'reference_drop_pp': (statistics.mean(100 * r['metrics']['primary']['value'] for _, r in reference)
                                     - distribution['mean']) if reference else None,
               'candidate_space': 'original frames within windows; scores anchored to window starts' if method == 'random_window'
                                  else 'fixed reference grid; scores anchored to selected candidates',
               'selection_representation': reports[0]['sampling'].get('representation', {}).get('kind'),
               'projection_q_sha256': reports[0].get('projection', {}).get('q_sha256'),
               'ap_mean_percent': statistics.mean(100 * r['metrics'][branch]['ap'] for r in reports),
               'ano_auc_mean_percent': statistics.mean(100 * r['metrics'][branch]['ano_auc'] for r in reports)}
        for field in ('roc_auc', 'ap', 'ano_auc'):
            values = describe([100 * r['metrics'][branch][field] for r in reports])
            row[field + '_mean_percent'] = values['mean']
            row[field + '_std_pp'] = values['sample_std']
        fine = [r['metrics'].get('fine_grained', {}) for r in reports]
        for iou in ('0.1', '0.2', '0.3', '0.4', '0.5'):
            values = [100 * f['map_by_iou'][iou] for f in fine if iou in f.get('map_by_iou', {})]
            stats = describe(values) if len(values) == len(fine) else {'mean': None, 'sample_std': None}
            row['map_iou_' + iou + '_mean_percent'] = stats['mean']
            row['map_iou_' + iou + '_std_pp'] = stats['sample_std']
        maps = [100 * f['mean_map'] for f in fine if 'mean_map' in f]
        map_stats = describe(maps) if len(maps) == len(fine) else {'mean': None, 'sample_std': None}
        row['mean_map_percent'] = map_stats['mean']
        row['mean_map_std_pp'] = map_stats['sample_std']
        for field in ('valid_fraction', 'reference_observable_fraction'):
            values = [r.get('event_coverage', {}).get(field) for r in reports]
            row[field + '_mean'] = statistics.mean(values) if all(v is not None for v in values) else None
        rows.append(row)
    _, output = create_run_dir(first['dataset_key'] + '_sampling', 'comparison', output_root=args.output_root)
    write_json(output / 'summary.json', {
        'schema_version': 1, 'scope': 'performance_only; no speedup comparison',
        'historical_source_differences': source_differences,
        'exploratory': source_differences or any(r['experiment']['protocol']['status'] != 'fixed' for _, r in expanded),
        'frozen_identity': identity, 'rows': rows,
        'inputs': [{'summary': str(path), 'sha256': sha256(path), 'source_sha256': r['package_sha256']}
                   for path, r in expanded]})
    with (output / 'comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Comparison: {output / "comparison.csv"}')


if __name__ == '__main__':
    main()
