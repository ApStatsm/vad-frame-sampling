"""Run all five seeds for each requested Random window budget and aggregate the scores."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vadfs.paths import package_hashes, sha256

BUDGETS = ('100', '50', '25', '12p5')
SEEDS = ('0', '1', '2', '3', '4')


def run_and_get_summary(command):
    report = None
    with subprocess.Popen(command, cwd=ROOT, text=True, stdout=subprocess.PIPE) as process:
        for line in process.stdout:
            print(line, end='', flush=True)
            if line.startswith('Report: '):
                report = Path(line.removeprefix('Report: ').strip())
        status = process.wait()
    if status:
        raise SystemExit(status)
    if report is None or not report.is_file():
        raise RuntimeError('Experiment did not return a summary file')
    return str(report)


def reusable_runs(args, configs, reference, local):
    if not args.reuse_run:
        return {}
    reference_hash = sha256(reference / 'summary.json')
    local_hash = sha256(local) if local else None
    source_hashes = package_hashes()
    config_hashes = {sha256(path): budget for budget, path in configs.items()}
    reused = {}
    for path in args.reuse_run:
        path = path.expanduser().resolve()
        if path.is_dir():
            path /= 'summary.json'
        report = json.loads(path.read_text(encoding='utf-8'))
        sampling = report.get('sampling', {})
        inputs = report.get('input_sha256', {})
        env = report.get('environment', {})
        budget = config_hashes.get(inputs.get('experiment_config'))
        seed = str(sampling.get('seed'))
        if (report.get('schema_version') != 2 or not report.get('success') or not report.get('metrics')
                or report.get('mode') != 'full' or report.get('input_type') != 'raw'
                or report.get('dataset_key') != args.dataset or sampling.get('method') != 'random_window'
                or budget is None or seed not in SEEDS
                or report.get('reference_summary_sha256') != reference_hash
                or report.get('reference_comparison', {}).get('scope') != 'performance_only'
                or report.get('package_sha256') != source_hashes or inputs.get('local_config') != local_hash
                or env.get('device') != args.device
                or (args.device == 'cuda' and env.get('CUDA_VISIBLE_DEVICES') != os.environ.get('CUDA_VISIBLE_DEVICES'))):
            raise ValueError(f'Reuse requires a completed seed from the same experiment settings: {path}')
        key = budget, seed
        if key in reused:
            raise ValueError(f'Duplicate completed run for budget {budget}, seed {seed}')
        reused[key] = str(path)
    return reused


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('ucf', 'xd'), required=True)
    parser.add_argument('--reference-run', type=Path, required=True,
                        help='Existing raw 100%% Reference folder for performance comparison')
    parser.add_argument('--budgets', nargs='+', choices=BUDGETS, default=BUDGETS)
    parser.add_argument('--reuse-run', action='append', type=Path, default=[],
                        help='Reuse one completed seed from a run directory or summary JSON; repeatable')
    parser.add_argument('--local', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cuda')
    args = parser.parse_args()
    if len(set(args.budgets)) != len(args.budgets):
        parser.error('Budgets must be unique')
    reference = args.reference_run.expanduser().resolve()
    local = args.local.expanduser().resolve() if args.local else None
    configs = {budget: ROOT / 'configs' / 'experiments' / f'{args.dataset}_random_{budget}.json'
               for budget in args.budgets}
    reused = reusable_runs(args, configs, reference, local)
    for budget, config in configs.items():
        base_command = [sys.executable, '-u', 'scripts/run_experiment.py',
                        '--config', str(config),
                        '--device', args.device, '--mode', 'full',
                        '--performance-reference-run', str(reference)]
        if local:
            base_command += ['--local', str(local)]
        reports = []
        for seed in SEEDS:
            report = reused.get((budget, seed))
            if report:
                print(f'Reusing budget {budget}, seed {seed}: {report}', flush=True)
            else:
                report = run_and_get_summary([*base_command, '--seed', seed])
            reports.append(report)
        subprocess.run([sys.executable, 'scripts/analyze/aggregate_runs.py', *reports,
                        '--expected-seeds', *SEEDS], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
