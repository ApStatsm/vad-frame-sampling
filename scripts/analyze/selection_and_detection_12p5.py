"""Compare anomaly-frame selection with detection on previously saved scores."""

import argparse
from collections import defaultdict
import csv
from pathlib import Path
import statistics

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from contingency_tables import REFERENCE_ROOT, checked_input
from vadfs.config import read_json
from vadfs.datasets import ucf, xd
from vadfs.datasets.base import validate_manifest
from vadfs.paths import ROOT, project_path, sha256
from vadfs.runs import create_run_dir, write_json
from vadfs.samplers import budget_count
from vadfs.timeline import to_frames


ANALYSES = ROOT / 'outputs/analyses'
BATCH_SUMMARY = ANALYSES / 'contingency_tables_12p5_20261115_183125/summary.json'
THRESHOLD = 0.5
BUDGET = 0.125
DURATION_LIMITS = (1, 2, 4)
DURATION_BINS = ('le_1s', '1_to_2s', '2_to_4s', 'gt_4s')


def primary_scores(run_dir, spec, gt, *, branch=None):
    with np.load(run_dir / 'frame_scores.npz', allow_pickle=False) as scores:
        primary = scores[branch or spec.primary_branch]
    if (primary.shape != gt.shape or not np.isfinite(primary).all()
            or np.any(primary < 0) or np.any(primary > 1)):
        raise ValueError(f'Invalid primary scores: {run_dir}')
    return primary


def detection_metrics(gt, scores, reference_tp):
    positives = np.count_nonzero(gt)
    recall_target = reference_tp / positives
    fpr, recall, _ = roc_curve(gt, scores, drop_intermediate=False)
    tp = np.rint(recall * positives).astype(np.int64)
    index = int(np.searchsorted(tp, reference_tp, side='left'))
    # If tied scores skip the target, also report the first attainable point above it.
    exact_fpr = fpr[index]
    if tp[index] != reference_tp:
        fraction = (reference_tp - tp[index - 1]) / (tp[index] - tp[index - 1])
        exact_fpr = fpr[index - 1] + fraction * (fpr[index] - fpr[index - 1])
    return {
        'ap_pct': 100 * float(average_precision_score(gt, scores)),
        'roc_auc_pct': 100 * float(roc_auc_score(gt, scores)),
        'recall_target_pct': 100 * recall_target,
        'fpr_at_exact_recall_pct': 100 * float(exact_fpr),
        'fpr_at_attainable_recall_pct': 100 * float(fpr[index]),
        'attainable_recall_pct': 100 * float(recall[index]),
        'fpr_at_0_5_pct': 100 * float(np.mean(scores[gt == 0] >= THRESHOLD)),
        'recall_at_0_5_pct': 100 * float(np.mean(scores[gt == 1] >= THRESHOLD)),
    }


def selection_metrics(run_dir, videos, fps, gt, primary, branch, budget):
    selected, selected_in_anomaly_videos, selected_anomaly = 0, 0, 0
    events = dict.fromkeys(DURATION_BINS, 0)
    covered = dict.fromkeys(DURATION_BINS, 0)
    for video, video_fps in zip(videos, fps):
        index = video['official_csv_index']
        stem = Path(video['filename']).stem
        path = (run_dir / 'scores' / f'{index:03d}_{stem}.npz').resolve()
        if not path.is_relative_to(run_dir / 'scores'):
            raise ValueError(f'Score path outside run directory: {path}')
        tokens = video['reference_tokens']
        length = video['evaluation_frames']
        offset = video['gt_offset']
        with np.load(path, allow_pickle=False) as archive:
            frames = archive['selected_frame_indices']
            scores = archive[f'reference_{branch}_scores']
        if (frames.shape != (budget_count(tokens, budget),) or frames.dtype.kind not in 'iu'
                or frames[0] < 0 or frames[-1] >= length or np.any(frames[1:] <= frames[:-1])):
            raise ValueError(f'Invalid selected frame coordinates or budget: {path}')
        if scores.shape != (tokens,) or not np.array_equal(to_frames(scores), primary[offset:offset + length]):
            raise ValueError(f'Per-video scores do not match the aggregate timeline: {path}')
        selected += len(frames)
        if video['is_anomaly']:
            selected_in_anomaly_videos += len(frames)
        selected_anomaly += int(np.count_nonzero(gt[offset + frames]))
        for start, stop in video['intervals_raw']:
            start, stop = max(0, start), min(length, stop)
            if stop <= start:
                continue
            seconds = (stop - start) / video_fps
            bucket = DURATION_BINS[np.searchsorted(DURATION_LIMITS, seconds, side='left')]
            events[bucket] += 1
            first = int(np.searchsorted(frames, start, side='left'))
            covered[bucket] += int(first < len(frames) and frames[first] < stop)

    result = {
        'selected_frames': selected,
        'selected_frames_in_anomaly_videos': selected_in_anomaly_videos,
        'selected_anomaly_frames': selected_anomaly,
        'selection_precision_all_pct': 100 * selected_anomaly / selected,
        'selection_precision_anomaly_videos_pct': 100 * selected_anomaly / selected_in_anomaly_videos,
        'anomaly_frame_recall_pct': 100 * selected_anomaly / np.count_nonzero(gt),
        'events_total': sum(events.values()),
        'events_covered': sum(covered.values()),
    }
    for bucket in DURATION_BINS:
        result[f'events_{bucket}'] = events[bucket]
        result[f'covered_{bucket}'] = covered[bucket]
        result[f'coverage_{bucket}_pct'] = 100 * covered[bucket] / events[bucket] if events[bucket] else None
    return result


def analyze_dataset(spec, sample_records):
    reference = REFERENCE_ROOT / f'{spec.key}_raw_reference'
    reference_summary = reference / 'summary.json'
    full = read_json(reference_summary)
    manifest_hash = full['input_sha256']['setup_manifest']
    manifest = read_json(checked_input(reference / 'manifest.json', manifest_hash))
    videos = validate_manifest(manifest, spec)
    config = read_json(ROOT / 'configs/datasets' / f'{spec.key}.json')
    gt_hash = full['input_sha256']['gt']
    gt = np.load(checked_input(project_path(config['frame_gt']), gt_hash), allow_pickle=False)
    fps = []
    offset = 0
    if len(full['videos']) != len(videos):
        raise ValueError(f'Incomplete reference run: {reference}')
    for video, record in zip(videos, full['videos']):
        if video['gt_offset'] != offset or record['video'] != video['filename'] or record['fps'] <= 0:
            raise ValueError(f'Reference manifest and FPS records differ: {video["filename"]}')
        offset += video['evaluation_frames']
        fps.append(record['fps'])
    if gt.shape != (offset,) or not np.isin(gt, [0, 1]).all() or np.unique(gt).size != 2:
        raise ValueError(f'Invalid binary GT timeline: {reference}')
    full_scores = primary_scores(reference, spec, gt)
    reference_tp = int(np.count_nonzero(full_scores[gt == 1] >= THRESHOLD))
    run_inputs = [('reference', reference, reference_summary, full, full_scores)]
    for record in sample_records:
        source = checked_input(Path(record['source_summary']), record['source_summary_sha256'])
        summary = read_json(source)
        run_dir = source.parent if record['source_summary_kind'] == 'original' else Path(summary['output_dir'])
        run_inputs.append((record['family'], run_dir.resolve(), source, summary, None))

    rows = []
    for family, run_dir, source, report, cached_scores in run_inputs:
        sampling = report['sampling']
        if (not report['success'] or report['mode'] != 'full' or report['dataset_key'] != spec.key
                or report['dataset_identity_sha256'] != manifest['dataset_identity_sha256']
                or report['input_sha256']['setup_manifest'] != manifest_hash
                or report['input_sha256']['gt'] != gt_hash
                or sampling['budget'] != (1 if family == 'reference' else BUDGET)):
            raise ValueError(f'Incompatible run summary: {source}')
        scores = cached_scores if cached_scores is not None else primary_scores(run_dir, spec, gt)
        row = {'dataset': spec.key, 'family': family, 'method': sampling['method'], 'seed': sampling['seed'],
               'budget': sampling['budget'], 'run_id': report['run_id'], 'summary_sha256': sha256(source)}
        row.update(selection_metrics(run_dir, videos, fps, gt, scores, spec.primary_branch, sampling['budget']))
        row.update(detection_metrics(gt, scores, reference_tp))
        rows.append(row)
        print(f'{spec.key} {family} {sampling["method"]} seed {sampling["seed"]}', flush=True)
    return rows


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    groups = defaultdict(list)
    baseline = {row['dataset']: row for row in rows if row['family'] == 'reference'}
    for row in rows:
        ref = baseline[row['dataset']]
        for metric in ('ap_pct', 'fpr_at_exact_recall_pct'):
            row[f'{metric[:-4]}_delta_vs_full_pp'] = row[metric] - ref[metric]
        groups[row['dataset'], row['family'], row['method']].append(row)
    comparison = []
    for (dataset, family, method), runs in groups.items():
        aggregate = dict(dataset=dataset, family=family, method=method, n_runs=len(runs))
        for key, value in runs[0].items():
            if key in ('seed', 'budget') or isinstance(value, str):
                continue
            values = [row[key] for row in runs if row[key] is not None]
            aggregate[f'{key}_mean'] = statistics.mean(values) if values else None
            aggregate[f'{key}_std'] = statistics.stdev(values) if len(values) > 1 else None
        comparison.append(aggregate)
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-summary', type=Path, default=BATCH_SUMMARY)
    parser.add_argument('--output-root', type=Path, default=ANALYSES)
    args = parser.parse_args()
    batch = read_json(args.batch_summary)
    if (len(batch['runs']) != 38 or len({r['source_run_id'] for r in batch['runs']}) != 38
            or batch['threshold'] != THRESHOLD):
        raise ValueError('Expected the completed 38-run 12.5% contingency batch at threshold 0.5')
    rows = []
    for spec in (ucf.SPEC, xd.SPEC):
        records = [record for record in batch['runs'] if record['dataset'] == spec.name]
        if len(records) != 19:
            raise ValueError(f'Expected 19 sample runs for {spec.name}')
        rows.extend(analyze_dataset(spec, records))
    comparison = summarize(rows)
    _, output = create_run_dir('selection_detection', '12p5', output_root=args.output_root)
    write_csv(output / 'runs.csv', rows)
    write_csv(output / 'comparison.csv', comparison)
    write_json(output / 'summary.json', {
        'batch_summary': str(args.batch_summary.resolve()),
        'batch_summary_sha256': sha256(args.batch_summary),
        'recall_target': 'Per-dataset Full reference recall at 0.5, analysis only',
        'tied_scores': 'Linear ROC interpolation for exact recall; first achievable point also recorded',
        'event_duration_bins_seconds': ['(0,1]', '(1,2]', '(2,4]', '(4,∞)'],
        'event_scope': 'Binary GT annotation intervals clipped to the official evaluation timeline',
    })
    print(output.resolve())


if __name__ == '__main__':
    main()
