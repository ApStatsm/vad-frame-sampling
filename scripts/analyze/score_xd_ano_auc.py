"""Add XD Ano-AUC to a copy of a completed run's summary using saved frame scores."""

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vadfs.config import read_json
from vadfs.metrics import compute_xd_ano_auc
from vadfs.paths import ROOT, audited_path, project_path, sha256
from vadfs.runs import now, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path, help='Completed XD official or raw run directory')
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    summary_path = run_dir / 'summary.json'
    report = read_json(summary_path)
    if (report.get('schema_version') != 2 or report.get('success') is not True
            or report.get('dataset_key') != 'xd' or report.get('mode') != 'full'
            or report.get('input_type') not in ('official', 'raw') or not report.get('metrics')):
        parser.error('Expected a successful full XD official or raw run')

    config = read_json(ROOT / 'configs/datasets/xd.json')
    inputs = {'setup_manifest': audited_path(report['setup_manifest']),
              'gt': project_path(config['frame_gt']), 'official_list': project_path(config['official_list'])}
    hashes = {key: sha256(path) for key, path in inputs.items()}
    if any(digest != report['input_sha256'].get(key) for key, digest in hashes.items()):
        raise ValueError('Prepared manifest, frame GT or official list differs from the original run')

    manifest = read_json(inputs['setup_manifest'])
    gt = np.load(inputs['gt'], allow_pickle=False)
    scores_path = run_dir / 'frame_scores.npz'
    with np.load(scores_path, allow_pickle=False) as scores:
        scope, ano_auc = compute_xd_ano_auc(gt, scores['branch1'], scores['branch2'], report['videos'], manifest)

    report['metrics'].update(scope)
    for branch, value in ano_auc.items():
        report['metrics'][branch]['ano_auc'] = value
        print(f'{branch} Ano-AUC: {value:.8f} ({value:.4%})')
    report['ano_auc_postprocessing'] = {
        'source_summary_sha256': sha256(summary_path), 'frame_scores_sha256': sha256(scores_path),
        'input_sha256': hashes, 'metric_source_sha256': sha256(ROOT / 'vadfs/metrics.py'),
        'evaluated_at': now().isoformat(),
    }
    output = run_dir / 'summary_with_ano_auc.json'
    write_json(output, report)
    print(f'Report: {output}')


if __name__ == '__main__':
    main()
