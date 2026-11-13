"""Run UCF/XD raw, official, or cached raw-feature experiments."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vadfs.pipeline import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--local', type=Path)
    parser.add_argument('--mode', choices=('smoke', 'full'), default='smoke')
    parser.add_argument('--device', choices=('cpu', 'cuda', 'mps'))
    parser.add_argument('--seed', type=int, help='Sampler seed only; model seed remains fixed')
    references = parser.add_mutually_exclusive_group()
    references.add_argument('--reference-run', type=Path)
    references.add_argument('--cached-reference-run', type=Path,
                            help='Reuse saved features from a full raw 100%% run for performance evaluation')
    references.add_argument('--performance-reference-run', type=Path,
                            help='Compare scores to an existing raw 100%% run without computing speedup')
    parser.add_argument('--setup-manifest', type=Path)
    parser.add_argument('--video', action='append', help='Exact filename; smoke only')
    parser.add_argument('--save-features', action='store_true', help='Optional selected feature cache; excluded from E2E time')
    args = parser.parse_args()
    return run_experiment(args.config, args.local, args.mode, args.video, args.device,
                          args.seed, args.reference_run, args.setup_manifest, args.save_features,
                          cached_reference_override=args.cached_reference_run,
                          performance_reference_override=args.performance_reference_run)


if __name__ == '__main__':
    raise SystemExit(main())
