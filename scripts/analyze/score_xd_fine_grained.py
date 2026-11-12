"""Add official XD temporal mAP to a copy of a completed run's summary."""

import argparse
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vadfs.config import read_json
from vadfs.datasets.xd import PROMPTS
from vadfs.evaluation import evaluate_xd_fine_grained
from vadfs.paths import ROOT, audited_path, project_path, sha256
from vadfs.runs import now, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path, help='Completed XD official run directory')
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    summary_path = run_dir / 'summary.json'
    report = read_json(summary_path)
    if (report.get('schema_version') != 2 or report.get('success') is not True
            or report.get('dataset_key') != 'xd' or report.get('input_type') != 'official'
            or report.get('mode') != 'full' or not report.get('metrics')
            or report.get('prompts') != list(PROMPTS)):
        parser.error('Expected a successful full XD official run with the original prompt order')

    config_path = ROOT / 'configs/datasets/xd.json'
    dataset_config = read_json(config_path)
    manifest_path = audited_path(report['setup_manifest'])
    manifest_hash = sha256(manifest_path)
    official_list_hash = sha256(project_path(dataset_config['official_list']))
    if (manifest_hash != report['input_sha256']['setup_manifest']
            or official_list_hash != report['input_sha256']['official_list']):
        raise ValueError('Prepared manifest or official list differs from the original run')
    manifest = read_json(manifest_path)
    records = report['videos']
    class_tokens = []
    for record in records:
        score_path = (run_dir / record['score_file']).resolve()
        if not score_path.is_relative_to(run_dir / 'scores'):
            raise ValueError(f'Score file outside the run score directory: {score_path}')
        with np.load(score_path, allow_pickle=False) as scores:
            class_tokens.append(scores['reference_class_scores'])

    started = time.perf_counter()
    report['metrics']['fine_grained'] = evaluate_xd_fine_grained(dataset_config, manifest, records, class_tokens)
    report['fine_grained_postprocessing'] = {
        'source_summary': str(summary_path),
        'source_summary_sha256': sha256(summary_path),
        'dataset_config_sha256': sha256(config_path),
        'input_sha256': {field: sha256(project_path(dataset_config[field])) for field in
                         ('segment_gt', 'class_gt', 'fine_grained_source')},
        'source_sha256': {path: sha256(ROOT / path) for path in
                          ('scripts/analyze/score_xd_fine_grained.py', 'vadfs/evaluation.py', 'vadfs/metrics.py')},
        'metric_seconds': time.perf_counter() - started,
        'evaluated_at': now().isoformat(),
    }
    output = run_dir / 'summary_with_fine_grained.json'
    write_json(output, report)
    print(f'Report: {output}')


if __name__ == '__main__':
    main()
