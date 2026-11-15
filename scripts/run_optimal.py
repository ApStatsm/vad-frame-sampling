"""Fit-free CLIP-cache D/A/E and matched baselines, sequential on one device."""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_random_windows import run_and_get_summary
from vadfs.paths import project_path, sha256
from vadfs.runs import create_run_dir, write_json

BUDGETS = ('100', '50', '25', '12p5')
METHODS = ('reference', 'uniform', 'random', 'd', 'a', 'e')


def experiment_cases(dataset, methods, budgets):
    """Reference is always 100%; Random-feature uses the declared five seeds."""
    cases = []
    for method in methods:
        for budget in (('100',) if method == 'reference' else budgets):
            name = method + '_optimal' if method in ('d', 'a', 'e') else method
            for seed in (range(5) if method == 'random' else (0,)):
                config = ROOT / f'configs/experiments/{dataset}_clip_{name}_{budget}.json'
                cases.append({'method': name, 'budget': budget, 'seed': seed, 'config': config})
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('ucf', 'xd'), required=True)
    parser.add_argument('--local', required=True, type=Path)
    parser.add_argument('--reference-run', required=True, type=Path,
                        help='Full raw Reference with manifest.json and saved features/*.npz')
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS),
                        help='Default: cached Reference, Uniform, five Random-feature seeds, D/A/E')
    parser.add_argument('--budgets', nargs='+', choices=BUDGETS, default=list(BUDGETS))
    parser.add_argument('--mode', choices=('smoke', 'full'), default='smoke')
    parser.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cuda')
    args = parser.parse_args()
    if len(set(args.methods)) != len(args.methods) or len(set(args.budgets)) != len(args.budgets):
        parser.error('Methods and budgets must be unique')
    local = project_path(args.local)
    reference = project_path(args.reference_run)
    if not local.is_file() or not (reference / 'summary.json').is_file() or not (reference / 'features').is_dir():
        parser.error('Local JSON and a full raw Reference summary/features directory are required')
    cases = experiment_cases(args.dataset, args.methods, args.budgets)
    _, folder = create_run_dir(f'{args.dataset}_clip_optimal', args.mode, output_root=ROOT / 'outputs/analyses')
    sweep = {'schema_version': 2, 'dataset_key': args.dataset, 'mode': args.mode,
             'experiment_scope': 'cached_feature_exploration', 'speedup_eligible': False,
             'reference_summary_sha256': sha256(reference / 'summary.json'),
             'local_config_sha256': sha256(local), 'complete': False, 'runs': []}
    try:
        for case in cases:
            command = [sys.executable, '-u', str(ROOT / 'scripts/run_experiment.py'),
                       '--config', str(case['config']), '--local', str(local),
                       '--device', args.device, '--mode', args.mode, '--seed', str(case['seed']),
                       '--cached-reference-run', str(reference)]
            summary = run_and_get_summary(command)
            sweep['runs'].append({key: value for key, value in case.items() if key != 'config'}
                                 | {'summary': summary, 'sha256': sha256(Path(summary))})
            write_json(folder / 'sweep.json', sweep)
        if args.mode == 'full':
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/analyze/compare_sampling.py'),
                                     *[item['summary'] for item in sweep['runs']],
                                     '--output-root', str(ROOT / 'outputs/analyses')],
                                    cwd=ROOT, check=True, text=True, capture_output=True)
            print(result.stdout, end='', flush=True)
            if result.stderr:
                print(result.stderr, file=sys.stderr, end='', flush=True)
            for line in result.stdout.splitlines():
                if line.startswith('Comparison: '):
                    sweep['comparison_csv'] = line.removeprefix('Comparison: ').strip()
        sweep['complete'] = True
    finally:
        write_json(folder / 'sweep.json', sweep)
        print(f'Sweep: {folder / "sweep.json"}', flush=True)


if __name__ == '__main__':
    main()
