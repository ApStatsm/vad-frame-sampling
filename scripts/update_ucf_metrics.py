"""Add Ano-AUC from saved scores; never load a model."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback
from zoneinfo import ZoneInfo

import numpy as np
from ucf_metrics import compute_ucf_metrics


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path, help='Full run folder, relative to project root or absolute')
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser()
    run_dir = (run_dir if run_dir.is_absolute() else ROOT / run_dir).resolve()
    summary_path = run_dir / 'summary.json'
    original = summary_path.read_text()
    report = json.loads(original)
    if (report.get('success') is not True or report.get('mode') != 'full'
            or report.get('dataset') != 'UCF-Crime'
            or report.get('input_type') != 'official_features'):
        raise ValueError('Select a successful full UCF official-feature run')
    manifest_path = ROOT / 'UCF-Crime/test/manifest.json'
    gt_path = ROOT / 'VadCLIP/list/gt_ucf.npy'
    for key, path in (('manifest', manifest_path), ('gt', gt_path)):
        if sha256(path) != report['input_sha256'][key]:
            raise ValueError(f'{key} differs from the original run')
    scores_path = run_dir / 'frame_scores.npz'
    scores_hash = sha256(scores_path)
    gt = np.load(gt_path, allow_pickle=False)
    manifest = json.loads(manifest_path.read_text())
    with np.load(scores_path, allow_pickle=False) as scores:
        frame1, frame2 = scores['branch1'], scores['branch2']
        computed = compute_ucf_metrics(
            gt, frame1, frame2, report['videos'], manifest)
    # Verify saved-score alignment before adding anything to the original report.
    metrics = report['metrics']
    for key in ('evaluation_frames', 'test_videos', 'includes_normal'):
        if metrics[key] != computed[key]:
            raise ValueError(f'Original metric scope differs: {key}')
    for branch in ('branch1', 'branch2'):
        for key in ('roc_auc', 'ap'):
            if not np.isclose(metrics[branch][key], computed[branch][key], rtol=0, atol=1e-12):
                raise ValueError(f'Saved scores do not reproduce {branch}.{key}')
    for branch in ('branch1', 'branch2'):
        metrics[branch]['ano_auc'] = computed[branch]['ano_auc']
    for key in ('anomaly_videos', 'anomaly_evaluation_frames', 'ano_auc_scope'):
        metrics[key] = computed[key]
    report['metrics_update'] = {
        'updated_at': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
        'operation': 'Add Ano-AUC from saved scores; original inference provenance retained',
        'source_summary_sha256': hashlib.sha256(original.encode('utf-8')).hexdigest(),
        'frame_scores_sha256': scores_hash,
        'script_sha256': sha256(Path(__file__)),
        'metrics_source_sha256': sha256(ROOT / 'scripts/ucf_metrics.py'),
    }
    if summary_path.read_text() != original or sha256(scores_path) != scores_hash:
        raise ValueError('Run files changed during metric update; retry with unchanged files')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=run_dir,
                                         prefix='.summary-', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        os.replace(temporary, summary_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(json.dumps(metrics, indent=2))
    print(f'Updated: {summary_path}')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
