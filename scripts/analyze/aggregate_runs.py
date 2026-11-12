"""Aggregate user-run repeated seeds; reject different protocols or missing seeds."""

import argparse
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vadfs.config import read_json
from vadfs.runs import create_run_dir, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path, help='Run directories or summary JSON paths')
    parser.add_argument('--expected-seeds', nargs='+', type=int)
    parser.add_argument('--output-root', type=Path, default=Path('outputs/analyses'))
    args = parser.parse_args()
    reports = [read_json(path / 'summary.json' if path.is_dir() else path) for path in args.runs]
    if any(report.get('schema_version') != 2 or not report.get('success')
           or report.get('mode') != 'full' or not report.get('metrics') for report in reports):
        raise ValueError('Only successful full common-pipeline runs can be aggregated')
    def group(report):
        return (report['dataset_key'], report['input_type'], report['sampling']['method'],
                report['sampling']['budget'], report['comparison_signature'],
                report['metrics']['primary']['branch'], report['metrics']['primary']['metric'],
                report.get('reference_summary_sha256'))
    if any(group(report) != group(reports[0]) for report in reports):
        raise ValueError('Dataset, method, budget, environment, protocol or reference differs')
    seeds = [report['sampling']['seed'] for report in reports]
    if len(set(seeds)) != len(seeds):
        raise ValueError('Duplicate sampler seeds; timing repetitions need a separate analysis')
    if args.expected_seeds is not None:
        if len(set(args.expected_seeds)) != len(args.expected_seeds) or set(seeds) != set(args.expected_seeds):
            raise ValueError('Runs do not cover exactly the declared seed list')
    def describe(values):
        return {'n': len(values), 'mean': statistics.mean(values),
                'sample_std': statistics.stdev(values) if len(values) > 1 else None}
    summary = {'schema_version': 1, 'dataset': reports[0]['dataset'], 'input_type': reports[0]['input_type'],
               'sampling_method': reports[0]['sampling']['method'], 'budget': reports[0]['sampling']['budget'],
               'seeds': seeds, 'runs': [report['run_id'] for report in reports],
               'primary': reports[0]['metrics']['primary'].copy(),
               'metric_unit': 'fraction; multiply mean/std by 100 for percentage points',
               'primary_distribution': describe([report['metrics']['primary']['value'] for report in reports]),
               'comparison_signature': reports[0]['comparison_signature'],
               'protocol_statuses': [report['experiment']['protocol']['status'] for report in reports],
               'automatic_claim_of_final_validation': False}
    summary['primary'].pop('value')
    branch = summary['primary']['branch']
    summary['metric_distributions'] = {branch: {
        field: describe([report['metrics'][branch][field] for report in reports])
        for field in ('roc_auc', 'ap', 'ano_auc')
        if all(field in report['metrics'][branch] for report in reports)
    }}
    fine = [report['metrics'].get('fine_grained', {}) for report in reports]
    if all('map_by_iou' in metrics and 'mean_map' in metrics for metrics in fine):
        summary['metric_distributions']['fine_grained'] = {
            'map_by_iou': {iou: describe([metrics['map_by_iou'][iou] for metrics in fine])
                           for iou in fine[0]['map_by_iou']},
            'mean_map': describe([metrics['mean_map'] for metrics in fine]),
        }
    key = 'end_to_end_seconds' if reports[0]['input_type'] == 'raw' else 'cached_feature_seconds'
    summary[key] = describe([report[key] for report in reports])
    for field in ('primary_delta_pp', 'speedup'):
        if all(field in report.get('reference_comparison', {}) for report in reports):
            summary[field] = describe([report['reference_comparison'][field] for report in reports])
    _, output = create_run_dir('seeds', 'aggregate', output_root=args.output_root.expanduser().resolve())
    write_json(output / 'summary.json', summary)
    primary = summary['primary_distribution']
    if primary['sample_std'] is not None:
        print(f'{summary["dataset"]} {summary["budget"]:.1%} {summary["primary"]["metric"]}: '
              f'{primary["mean"]:.4%} ± {100 * primary["sample_std"]:.4f} pp (n={primary["n"]})')
    print(f'Aggregate: {output / "summary.json"}')


if __name__ == '__main__':
    main()
