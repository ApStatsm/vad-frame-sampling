"""Compare the saved 12.5% UCF/XD sampling runs with the same contingency rules."""

import argparse
from collections import defaultdict
import csv
from pathlib import Path
import statistics

from contingency_tables import collect_tables, write_table
from vadfs.config import read_json
from vadfs.datasets import ucf, xd
from vadfs.paths import ROOT
from vadfs.runs import create_run_dir, write_json


BUDGET = 0.125
SHARED = Path('/workspace/vad/results')
RESULTS = ROOT / 'docs/results'
ADAPTIVE = {
    'ucf': 'ucf_raw_adaptive_window_0.125_s0_full_20261113_132917',
    'xd': 'xd_raw_adaptive_window_0.125_s0_full_20261115_165418',
}
OPTIMAL = {'a_optimal', 'd_optimal', 'e_optimal'}


def input_runs(key):
    yield 'window', SHARED / f'uniform/{key}_raw_uniform_0.125', None
    for run in sorted((SHARED / 'random/runs').glob(f'{key}_raw_random_window_0.125_s*_full_*')):
        yield 'window', run, None
    yield 'window', ROOT / 'outputs/runs' / ADAPTIVE[key], None
    sources = {
        'window': (RESULTS / f'{key}_window_optimal_ver2_20261114/runs', f'{key}_?_optimal_0.125_*.json'),
        'cached': (RESULTS / 'clip_optimal_20261113/runs', f'{key}_cached_raw_features_*_0.125_*.json'),
    }
    for family, (folder, pattern) in sources.items():
        for archive in sorted(folder.glob(pattern)):
            yield family, Path(read_json(archive)['output_dir']), archive


def collect_dataset(spec, threshold):
    expected = {
        'window': {'uniform': {0}, 'random_window': set(range(5)), 'adaptive_window': {0}},
        'cached': {'uniform': {0}, 'random': set(range(5))},
    }
    for methods in expected.values():
        methods.update({name: {0} for name in OPTIMAL})
    seen = defaultdict(set)
    tables = []
    for family, run, archive in input_runs(spec.key):
        report = read_json(archive or run / 'summary.json')
        method = report['sampling']['method']
        seed = report['sampling']['seed']
        group = (family, method)
        if report['sampling']['budget'] != BUDGET or seed in seen[group]:
            raise ValueError(f'Unexpected budget or duplicate seed: {run}')
        seen[group].add(seed)
        rows, multiclass, binary, metadata = collect_tables(run.resolve(), spec, threshold, archive)
        metadata.update(family=family, sampling_method=method, seed=seed, budget=BUDGET)
        tables.append((rows, multiclass, binary, metadata))
    if seen != {(family, method): seeds for family, methods in expected.items() for method, seeds in methods.items()}:
        raise ValueError(f'Incomplete 12.5% methods or seeds: {spec.key}')
    return tables


def rates(spec, rows, multiclass, binary):
    tn, fp, fn, tp = map(int, binary.ravel())
    anomaly_frames = int(multiclass[1:].sum())
    as_normal = int(multiclass[1:, 0].sum())
    correct = 0
    columns = {name: index for index, name in enumerate(spec.prompts)}
    for label, row in zip(rows[1:], multiclass[1:]):
        correct += sum(int(row[columns[name]]) for name in label.split(' + '))
    return {
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'precision_pct': 100 * tp / (tp + fp) if tp + fp else 0.0,
        'recall_pct': 100 * tp / (tp + fn),
        'fpr_pct': 100 * fp / (tn + fp),
        'type_correct_pct': 100 * correct / anomaly_frames,
        'type_as_normal_pct': 100 * as_normal / anomaly_frames,
        'type_wrong_anomaly_pct': 100 * (anomaly_frames - correct - as_normal) / anomaly_frames,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs/analyses')
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error('--threshold must be in [0, 1]')

    specs = (ucf.SPEC, xd.SPEC)
    tables = {spec.key: collect_dataset(spec, args.threshold) for spec in specs}
    _, output = create_run_dir('contingency_tables', '12p5', output_root=args.output_root)
    groups = defaultdict(list)
    summaries = []
    for spec in specs:
        for rows, multiclass, binary, metadata in tables[spec.key]:
            family, method, seed = (metadata[key] for key in ('family', 'sampling_method', 'seed'))
            prefix = f'{spec.key}_{family}_{method}_s{seed}'
            labels = ('Normal', 'Anomaly')
            write_table(output / f'{prefix}_binary.csv', labels, labels, binary)
            write_table(output / f'{prefix}_multiclass.csv', rows, spec.prompts, multiclass)
            groups[spec.key, family, method].append(rates(spec, rows, multiclass, binary))
            summaries.append(metadata)
    comparison = []
    for (dataset, family, method), runs in groups.items():
        row = dict(dataset=dataset, family=family, method=method, n_runs=len(runs))
        for metric in runs[0]:
            values = [run[metric] for run in runs]
            row[f'{metric}_mean'] = statistics.mean(values)
            row[f'{metric}_std'] = statistics.stdev(values) if len(values) > 1 else None
        comparison.append(row)
    with (output / 'comparison.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    write_json(output / 'summary.json', {'threshold': args.threshold, 'runs': summaries})
    print(output.resolve())


if __name__ == '__main__':
    main()
