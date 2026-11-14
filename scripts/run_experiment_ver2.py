"""Run a separate raw-window ver2 experiment; default smoke, never auto-launch."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vadfs.pipeline_ver2 import run_experiment_ver2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--local', required=True, type=Path)
    parser.add_argument('--mode', choices=('smoke','full'), default='smoke')
    parser.add_argument('--device', choices=('cpu','cuda','mps'))
    parser.add_argument('--seed', type=int)
    parser.add_argument('--setup-manifest', type=Path)
    parser.add_argument('--video', action='append', help='Exact manifest filename; smoke only')
    parser.add_argument('--save-features', action='store_true')
    args = parser.parse_args()
    return run_experiment_ver2(args.config,args.local,args.mode,args.video,args.device,args.seed,args.setup_manifest,args.save_features)


if __name__=='__main__':
    raise SystemExit(main())
