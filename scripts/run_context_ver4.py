"""Entry point for ver4 diagnostics. Authoring never runs this script."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vadfs.paths import sha256
from vadfs.runs import create_run_dir, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep-ver3', type=Path, required=True)
    parser.add_argument('--stage', choices=('decomposition', 'context', 'all'), default='decomposition')
    parser.add_argument('--mode', choices=('smoke', 'full'), default='smoke', help='Context stage only')
    parser.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cuda')
    parser.add_argument('--local', type=Path)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/experiments/xd_context_ver4.json')
    args = parser.parse_args()
    sweep = args.sweep_ver3.expanduser().resolve()
    if not sweep.is_file():
        parser.error('--sweep-ver3 must be an existing completed ver3 sweep JSON')
    _, output = create_run_dir('xd_context_ver4', args.stage, output_root=ROOT / 'outputs/analyses')
    status = {'success': False, 'error': None, 'stage': args.stage, 'context_mode': args.mode,
              'source_sweep': str(sweep), 'source_sweep_sha256': sha256(sweep),
              'entrypoint_sha256': sha256(Path(__file__)), 'outputs': {}}
    try:
        if args.stage in ('decomposition', 'all'):
            from scripts.analyze.decompose_context_ver4 import run
            status['outputs']['decomposition'] = str(run(sweep, ROOT / 'outputs/analyses'))
            write_json(output / 'sweep_ver4.json', status)
        if args.stage in ('context', 'all'):
            from scripts.probe_context_ver4 import run
            status['outputs']['context'] = str(run(sweep, args.config, args.local, args.mode, args.device))
        status['success'] = True
    except BaseException as error:
        status['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        write_json(output / 'sweep_ver4.json', status)
        print(f'Sweep: {output / "sweep_ver4.json"}', flush=True)


if __name__ == '__main__':
    main()
