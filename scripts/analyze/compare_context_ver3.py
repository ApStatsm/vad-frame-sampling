"""Validate a complete ver3 sweep and analyze saved scores; user-run only."""

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_curve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analyze.compare_sampling import frozen_identity
from vadfs.paths import project_path, sha256
from vadfs.runs import write_json
from vadfs.samplers import video_seed
from vadfs.samplers.context_ver3 import select_ver3
from vadfs.timeline import restore_tokens, to_frames


def save_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fpr_at_recall(gt, scores, target):
    fpr, tpr, _ = roc_curve(gt, scores, drop_intermediate=False)
    # First ROC vertex reaching target; report actual achieved recall as well.
    i = int(np.searchsorted(tpr, target, side='left'))
    if i == 0 or tpr[i] == target:
        interpolated = fpr[i]
    else:
        j = i - 1
        interpolated = fpr[j] + (fpr[i] - fpr[j]) * (target - tpr[j]) / (tpr[i] - tpr[j])
    return float(interpolated), float(fpr[i]), float(tpr[i])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summaries', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    items = [(p.resolve(), json.loads(p.read_text())) for p in args.summaries]
    groups = {}
    first = items[0][1]
    for path, report in items:
        if (not report.get('success') or report.get('mode') != 'full'
                or report.get('dataset_key') != 'xd' or report.get('input_type') != 'cached_raw_features'
                or report.get('comparison_signature') != first.get('comparison_signature')
                or frozen_identity(report) != frozen_identity(first)
                or not report.get('comparison_signature') or not report.get('cache_identity_sha256')
                or report.get('cache_identity_sha256') != first.get('cache_identity_sha256')
                or report['input_sha256']['setup_manifest'] != first['input_sha256']['setup_manifest']):
            raise ValueError(f'Incomplete/incompatible result: {path}')
        key = report['sampling']['method'], report['sampling']['seed']
        if key in groups:
            raise ValueError('Duplicate method/seed')
        groups[key] = (path, report)
    expected = {('reference', 0), ('uniform_ver3', 0), ('paired_ver3', 0)} | {
        ('relocated_ver3', seed) for seed in range(5)}
    if set(groups) != expected:
        raise ValueError('Need Full + Uniform + paired + relocated seeds0-4 (8 runs)')
    manifest_path = project_path(first['setup_manifest'])
    if sha256(manifest_path) != first['input_sha256']['setup_manifest']:
        raise ValueError('Manifest changed')
    manifest = json.loads(manifest_path.read_text())
    config = json.loads((ROOT / 'configs/datasets/xd.json').read_text())
    gt_path = project_path(config['frame_gt'])
    if sha256(gt_path) != first['input_sha256']['gt']:
        raise ValueError('GT changed')
    gt = np.load(gt_path, allow_pickle=False)
    if len(manifest['videos']) != 800:
        raise ValueError('Require complete XD test split')
    ref_path, ref = groups[('reference', 0)]
    uniform_path, uniform = groups[('uniform_ver3', 0)]
    with np.load(ref_path.parent / 'frame_scores.npz', allow_pickle=False) as z:
        full2 = z['branch2'].copy()
    if full2.shape != gt.shape:
        raise ValueError('Full/GT timeline mismatch')
    target = float(np.mean(full2[gt == 1] >= 0.5))
    source_summary = Path(first['cached_reference_run']) / 'summary.json'
    if sha256(source_summary) != first['reference_summary_sha256']:
        raise ValueError('Cache source summary changed')
    source = json.loads(source_summary.read_text())
    fps = {v['video']: v['fps'] for v in source['videos']}
    rows, deltas = [], []
    for (method, seed), (path, report) in groups.items():
        if len(report['videos']) != 800:
            raise ValueError('Missing videos')
        totals = np.zeros(4, dtype=np.int64)  # all events, covered, short events, short covered
        changes = {branch: {label: [] for label in (0, 1)} for branch in (1, 2)}
        restored = []
        for v, record, urecord in zip(manifest['videos'], report['videos'], uniform['videos']):
            if (record['video'] != v['filename'] or urecord['video'] != v['filename']
                    or record['official_csv_index'] != v['official_csv_index']
                    or record['reference_tokens'] != v['reference_tokens']
                    or record['gt_offset'] != v['gt_offset']
                    or record['evaluation_frames'] != v['evaluation_frames']):
                raise ValueError('Video order/timeline mismatch')
            with np.load(path.parent / record['score_file'], allow_pickle=False) as z:
                frames = z['selected_frame_indices'].copy()
                scores = {b: z[f'branch{b}_selected_scores'].copy() for b in (1, 2)}
                grid = z['reference_frame_indices'].copy()
            n = v['reference_tokens']
            expected_rows = (list(range(n)) if method == 'reference' else select_ver3(
                n, method, video_seed(seed, 'xd', v['filename']))[0])
            if (len(grid) != n or not np.array_equal(grid, np.arange(n) * 16)
                    or not np.array_equal(frames, grid[expected_rows])
                    or record['selected_tokens'] != len(expected_rows)):
                raise ValueError('Selection does not follow frozen ver3 protocol')
            dense = to_frames(restore_tokens(scores[2], frames, grid))
            restored.append(dense)
            fps_value = fps[v['filename']]
            if not np.isfinite(fps_value) or fps_value <= 0:
                raise ValueError('Invalid cache source FPS')
            for a, b in v['intervals_raw']:
                a, b = max(0, a), min(v['evaluation_frames'], b)
                if b <= a:
                    continue
                covered = bool(np.any((frames >= a) & (frames < b)))
                short = (b - a) / fps_value <= 1.0
                totals += [1, int(covered), int(short), int(short and covered)]
            if method == 'reference':
                continue
            with np.load(uniform_path.parent / urecord['score_file'], allow_pickle=False) as z:
                uframes = z['selected_frame_indices']
                common, ri, ui = np.intersect1d(frames, uframes, return_indices=True)
                labels = gt[v['gt_offset'] + common]
                for branch in (1, 2):
                    difference = scores[branch][ri] - z[f'branch{branch}_selected_scores'][ui]
                    for label in (0, 1):
                        changes[branch][label].extend(difference[labels == label].tolist())
        full_scores = np.concatenate(restored)
        with np.load(path.parent / 'frame_scores.npz', allow_pickle=False) as z:
            if not np.array_equal(full_scores, z['branch2']):
                raise ValueError('Saved scores disagree with re-restored timeline')
        ap = float(average_precision_score(gt, full_scores))
        if abs(ap - report['metrics']['primary']['value']) > 1e-10:
            raise ValueError('AP does not reproduce summary')
        interp, actual_fpr, actual_recall = fpr_at_recall(gt, full_scores, target)
        rows.append({'method': method, 'seed': seed, 'ap_percent': 100 * ap,
                     'fpr_interpolated_percent': 100 * interp,
                     'fpr_attainable_percent': 100 * actual_fpr,
                     'attainable_recall_percent': 100 * actual_recall,
                     'target_recall_percent': 100 * target,
                     'events': int(totals[0]), 'covered': int(totals[1]),
                     'events_le_1s': int(totals[2]), 'covered_le_1s': int(totals[3]),
                     'selected_frames': sum(v['selected_tokens'] for v in report['videos'])})
        for branch in (1, 2):
            for label in (0, 1):
                values = np.asarray(changes[branch][label])
                deltas.append({'method': method, 'seed': seed, 'branch': branch,
                               'gt': label, 'n_common_frames': len(values),
                               'mean_delta_vs_uniform': float(values.mean()) if len(values) else None,
                               'median_delta_vs_uniform': float(np.median(values)) if len(values) else None,
                               'increased_fraction': float(np.mean(values > 0)) if len(values) else None})
    aggregate = []
    uniform_row = next(row for row in rows if row['method'] == 'uniform_ver3')
    for row in rows:
        row['ap_delta_vs_uniform_pp'] = row['ap_percent'] - uniform_row['ap_percent']
        row['fpr_delta_vs_uniform_pp'] = row['fpr_interpolated_percent'] - uniform_row['fpr_interpolated_percent']
    for method in ('reference', 'uniform_ver3', 'paired_ver3', 'relocated_ver3'):
        subset = [r for r in rows if r['method'] == method]
        row = {'method': method, 'n_runs': len(subset)}
        for field in rows[0]:
            if field in ('method', 'seed'):
                continue
            values = [r[field] for r in subset]
            row[field + '_mean'] = statistics.mean(values)
            row[field + '_sample_std'] = statistics.stdev(values) if len(values) > 1 else None
        aggregate.append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    save_csv(args.output / 'runs_ver3.csv', rows)
    save_csv(args.output / 'comparison_ver3.csv', aggregate)
    save_csv(args.output / 'common_frame_changes_ver3.csv', deltas)
    write_json(args.output / 'analysis_ver3.json', {
        'scope': 'exploratory_cached_features_no_speedup_no_causal_attribution',
        'rows': rows, 'aggregates': aggregate,
        'inputs': [{'summary': str(p), 'sha256': sha256(p)} for p, _ in items],
        'analyzer_sha256': sha256(Path(__file__)),
        'identity_helper_sha256': sha256(ROOT / 'scripts/analyze/compare_sampling.py'),
        'common_frames': 'intersection with Uniform; method-specific population; pooled frame statistics'})
    print(f'Comparison: {args.output / "comparison_ver3.csv"}')


if __name__ == '__main__':
    main()
