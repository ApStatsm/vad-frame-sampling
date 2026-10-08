"""Attribute raw-vs-official UCF metric differences to official-feature dtype cohorts.

Uses only saved frame scores (no model inference). Swaps the frame-score
segments of the selected cohort between a raw run and the official-feature run,
and also evaluates the timeline with that cohort excluded. Output:
<raw_run>/cohort_sensitivity.json.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root: vadfs
from vadfs import ucf  # noqa: E402
from vadfs.metrics import compute_ucf_metrics  # noqa: E402
from vadfs.paths import ROOT, project_path  # noqa: E402


def frame_metrics(gt, score, anomaly_mask, keep):
    gt, score, anomaly_mask = gt[keep], score[keep], anomaly_mask[keep]
    return {'roc_auc': float(roc_auc_score(gt, score)), 'ap': float(average_precision_score(gt, score)),
            'ano_auc': float(roc_auc_score(gt[anomaly_mask], score[anomaly_mask]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-run', type=Path, required=True)
    parser.add_argument('--official-run', type=Path, default=ROOT / 'outputs/baseline/ucf_full_20261007_010204')
    parser.add_argument('--cohort-dtype', default='float32')
    args = parser.parse_args()
    raw_dir, official_dir = project_path(args.raw_run), project_path(args.official_run)
    raw = json.loads((raw_dir / 'summary.json').read_text())
    official = json.loads((official_dir / 'summary.json').read_text())
    audit = json.loads(ucf.INPUT_AUDIT.read_text())
    manifest = json.loads(ucf.MANIFEST.read_text())
    gt = np.load(ucf.FRAME_GT, allow_pickle=False)
    if not raw.get('success') or raw.get('mode') != 'full' or len(raw['videos']) != 290:
        raise ValueError('Use a successful full raw run')
    with np.load(raw_dir / 'frame_scores.npz') as f:
        raw_scores = {b: f[b].astype(np.float64) for b in ('branch1', 'branch2')}
    with np.load(official_dir / 'frame_scores.npz') as f:
        official_scores = {b: f[b].astype(np.float64) for b in ('branch1', 'branch2')}
    for scores in (raw_scores, official_scores):
        if any(s.shape != gt.shape for s in scores.values()):
            raise ValueError('Frame score length differs from GT')

    cohort = np.zeros(len(gt), dtype=bool)
    anomaly = np.zeros(len(gt), dtype=bool)
    by_name = {v['filename']: v for v in manifest['videos']}
    members = []
    for record, video in zip(audit['videos'], raw['videos']):
        if record['video'] != video['video'] or record['gt_offset'] != video['gt_offset']:
            raise ValueError('Audit/raw run order mismatch')
        segment = slice(record['gt_offset'], record['gt_offset'] + record['official_timeline_length'])
        anomaly[segment] = by_name[record['video']]['is_anomaly']
        if record['dtype'] == args.cohort_dtype:
            cohort[segment] = True
            members.append(record['video'])

    def full_metrics(scores):
        m = compute_ucf_metrics(gt, scores['branch1'], scores['branch2'], raw['videos'], manifest)
        return {b: m[b] for b in ('branch1', 'branch2')}

    raw_with_official_cohort = {b: np.where(cohort, official_scores[b], raw_scores[b]) for b in raw_scores}
    official_with_raw_cohort = {b: np.where(cohort, raw_scores[b], official_scores[b]) for b in raw_scores}
    results = {
        'raw': full_metrics(raw_scores),
        'official': full_metrics(official_scores),
        'raw_with_official_cohort_scores': full_metrics(raw_with_official_cohort),
        'official_with_raw_cohort_scores': full_metrics(official_with_raw_cohort),
        'raw_excluding_cohort': {b: frame_metrics(gt, raw_scores[b], anomaly, ~cohort) for b in raw_scores},
        'official_excluding_cohort': {b: frame_metrics(gt, official_scores[b], anomaly, ~cohort) for b in raw_scores},
    }
    base = results['official']
    differences = {name: {b: {k: 100 * (v[b][k] - base[b][k]) for k in v[b]} for b in v}
                   for name, v in results.items() if name not in ('official', 'official_excluding_cohort')}
    differences['raw_excluding_cohort_vs_official_excluding_cohort'] = {
        b: {k: 100 * (results['raw_excluding_cohort'][b][k] - results['official_excluding_cohort'][b][k])
            for k in results['raw_excluding_cohort'][b]} for b in raw_scores}
    report = {'raw_run': str(raw_dir), 'official_run': str(official_dir), 'cohort_dtype': args.cohort_dtype,
              'cohort_videos': members, 'cohort_frames': int(cohort.sum()), 'total_frames': len(gt),
              'metrics': results, 'differences_pp_vs_official': differences,
              'note': 'Saved scores only; excluding-cohort metrics are diagnostic, not the benchmark protocol'}
    (raw_dir / 'cohort_sensitivity.json').write_text(json.dumps(report, indent=2) + '\n')
    for name, diff in differences.items():
        print(f'{name:55s} ' + ' | '.join(f"{b} AUC {d['roc_auc']:+.3f} AP {d['ap']:+.3f} Ano {d['ano_auc']:+.3f}"
                                          for b, d in diff.items()))
    print('Saved:', raw_dir / 'cohort_sensitivity.json')


if __name__ == '__main__':
    main()
