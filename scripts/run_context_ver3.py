"""Run the fixed XD cached-context ver3 sweep; user execution only."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_random_windows import run_and_get_summary
from vadfs.paths import sha256
from vadfs.runs import create_run_dir, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run', type=Path, required=True)
    parser.add_argument('--local', type=Path)
    parser.add_argument('--mode', choices=('smoke', 'full'), default='smoke')
    parser.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cuda')
    args = parser.parse_args()
    reference = args.reference_run.expanduser().resolve()
    if not all((reference / p).exists() for p in ('summary.json', 'manifest.json', 'features')):
        parser.error('A full raw Reference summary, manifest and feature cache are required')
    _, output = create_run_dir('xd_context_ver3', args.mode, output_root=ROOT / 'outputs/analyses')
    sweep = {'complete': False, 'mode': args.mode, 'runs': [], 'error': None,
             'reference_summary_sha256': sha256(reference / 'summary.json'),
             'scheduler_sha256': sha256(Path(__file__)), 'scope': 'exploratory_cached_features_no_speedup'}
    try:
        for method in ('reference', 'uniform_ver3', 'paired_ver3', 'relocated_ver3'):
            config = ROOT / f'configs/experiments/xd_{method.removesuffix("_ver3")}_ver3.json'
            for seed in (range(5) if method == 'relocated_ver3' else (0,)):
                command = [sys.executable, '-u', str(ROOT / 'scripts/run_experiment.py'),
                           '--config', str(config), '--cached-reference-run', str(reference),
                           '--setup-manifest', str(reference / 'manifest.json'),
                           '--mode', args.mode, '--device', args.device, '--seed', str(seed)]
                if args.local:
                    command += ['--local', str(args.local.expanduser().resolve())]
                path = Path(run_and_get_summary(command))
                report = json.loads(path.read_text())
                if report.get('success') is not True:
                    raise ValueError(f'Failed result: {path}')
                sweep['runs'].append({'method': method, 'seed': seed, 'summary': str(path),
                                      'sha256': sha256(path)})
                write_json(output / 'sweep_ver3.json', sweep)
        if args.mode == 'full':
            subprocess.run([sys.executable, str(ROOT / 'scripts/analyze/compare_context_ver3.py'),
                            *[r['summary'] for r in sweep['runs']], '--output', str(output)],
                           cwd=ROOT, check=True)
        sweep['complete'] = True
    except BaseException as error:
        sweep['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        write_json(output / 'sweep_ver3.json', sweep)
        print(f'Sweep: {output / "sweep_ver3.json"}', flush=True)


if __name__ == '__main__':
    main()
