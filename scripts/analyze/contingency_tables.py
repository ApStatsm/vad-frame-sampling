"""Build frame-level contingency tables from saved UCF and XD test predictions."""

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vadfs.config import read_json
from vadfs.datasets import ucf, xd
from vadfs.datasets.base import validate_manifest
from vadfs.paths import ROOT, audited_path, project_path, sha256
from vadfs.runs import create_run_dir, write_json
from vadfs.timeline import to_frames


REFERENCE_ROOT = Path('/workspace/vad/results/baselines')
XD_CODES = ('A', 'B1', 'B2', 'B4', 'B5', 'B6', 'G')


def checked_input(path, expected_hash):
    if sha256(path) != expected_hash:
        raise ValueError(f'Input hash differs from the source run: {path}')
    return path


def load_ground_truth(spec, hashes):
    config = read_json(ROOT / 'configs/datasets' / f'{spec.key}.json')
    frame_path = checked_input(project_path(config['frame_gt']), hashes['gt'])
    segments_path = checked_input(project_path(config['segment_gt']), hashes['segment_gt'])
    classes_path = checked_input(project_path(config['class_gt']), hashes['class_gt'])
    frame_gt = np.load(frame_path, allow_pickle=False)
    # The bundled, hash-checked segment/class annotations are ragged object arrays.
    segments = np.load(segments_path, allow_pickle=True)
    classes = np.load(classes_path, allow_pickle=True)
    if frame_gt.ndim != 1 or not np.isin(frame_gt, [0, 1]).all():
        raise ValueError(f'Invalid binary frame GT: {frame_path}')
    if len(segments) != spec.expected_videos or len(classes) != spec.expected_videos:
        raise ValueError(f'Segment/class GT must cover the complete {spec.name} test split')
    return frame_gt, segments, classes


def class_mask(segments, labels, label_indices, video):
    """Zero represents Normal and other masks represent sets of anomaly labels."""
    if len(segments) != len(labels):
        raise ValueError(f'Segment and label counts differ: {video["filename"]}')
    masks = np.zeros(video['evaluation_frames'], dtype=np.int64)
    for bounds, label in zip(segments, labels):
        start, stop = map(int, bounds)
        if not 0 <= start < stop or str(label) not in label_indices:
            raise ValueError(f'Invalid segment or class {label!r}: {video["filename"]}')
        class_index = label_indices[str(label)]
        if class_index == 0:
            if video['is_anomaly']:
                raise ValueError(f'Unexpected normal segment label: {video["filename"]}')
        else:
            if not video['is_anomaly']:
                raise ValueError(f'Anomaly class in a normal-only video: {video["filename"]}')
            masks[start:stop] |= 1 << (class_index - 1)
    return masks


def class_predictions(run_dir, score_file, tokens, class_count, primary_check=None):
    path = (run_dir / score_file).resolve()
    if not path.is_relative_to(run_dir / 'scores'):
        raise ValueError(f'Score file outside the run score directory: {path}')
    with np.load(path, allow_pickle=False) as scores:
        probabilities = scores['reference_class_scores']
        if primary_check is not None:
            branch, expected = primary_check
            branch_tokens = scores[f'reference_{branch}_scores']
            if branch_tokens.shape != (tokens,) or not np.array_equal(to_frames(branch_tokens), expected):
                raise ValueError(f'Per-video scores do not match the aggregate run scores: {path}')
    if (probabilities.shape != (tokens, class_count)
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0) or np.any(probabilities > 1)
            or not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-5)):
        raise ValueError(f'Invalid reference class probabilities: {path}')
    return to_frames(probabilities.argmax(axis=1))


def collect_tables(run_dir, spec, threshold, archived_summary=None):
    summary_path = archived_summary or run_dir / 'summary.json'
    report = read_json(summary_path)
    if (report.get('success') is not True or report.get('mode') != 'full'
            or report.get('dataset_key') != spec.key or report.get('prompts') != list(spec.prompts)
            or report.get('metrics', {}).get('test_videos') != spec.expected_videos):
        raise ValueError(f'Expected a complete, successful {spec.name} run: {summary_path}')
    records = report.get('videos')
    if archived_summary is not None:
        if (records is not None or report['run_id'] != run_dir.name
                or Path(report['output_dir']).resolve() != run_dir):
            raise ValueError(f'Archived summary is not for the specified score directory: {summary_path}')
    elif len(records or ()) != spec.expected_videos:
        raise ValueError(f'Expected all per-video records: {summary_path}')
    hashes = report['input_sha256']
    manifest_path = run_dir / 'manifest.json'
    if archived_summary is not None or not manifest_path.is_file():
        manifest_path = audited_path(report['setup_manifest'])
    manifest = read_json(checked_input(manifest_path, hashes['setup_manifest']))
    if manifest['dataset_identity_sha256'] != report['dataset_identity_sha256']:
        raise ValueError(f'Manifest identity differs from the source run: {manifest_path}')
    videos = validate_manifest(manifest, spec)
    frame_gt, segments, classes = load_ground_truth(spec, hashes)
    with np.load(run_dir / 'frame_scores.npz', allow_pickle=False) as scores:
        primary = scores[spec.primary_branch]
    if (primary.shape != frame_gt.shape or not np.isfinite(primary).all()
            or np.any(primary < 0) or np.any(primary > 1)):
        raise ValueError(f'Invalid primary branch scores: {run_dir}')

    codes = XD_CODES if spec.key == 'xd' else spec.prompts
    label_indices = {code: index for index, code in enumerate(codes)}
    label_indices['A'] = 0  # Bundled UCF normal annotations also use A.
    class_count = len(spec.prompts)
    counts = np.zeros((1 << (class_count - 1), class_count), dtype=np.int64)
    offset = 0
    disagreeing_gt_frames = 0
    for index, video in enumerate(videos):
        frames = video['evaluation_frames']
        tokens = video['reference_tokens']
        if video['gt_offset'] != offset:
            raise ValueError(f'Annotation timeline mismatch: {video["filename"]}')
        primary_check = None
        if archived_summary is not None:
            stem = Path(video['filename']).stem
            score_file = f'scores/{index:03d}_{stem}.npz'
            primary_check = (spec.primary_branch, primary[offset:offset + frames])
        else:
            record = records[index]
            if (record['video'] != video['filename'] or record['official_csv_index'] != index
                    or record.get('case') != 'full_video' or record.get('prefix_only') is not False
                    or record['reference_tokens'] != tokens or record['evaluation_frames'] != frames
                    or record['gt_offset'] != offset):
                raise ValueError(f'Score and annotation timeline mismatch: {video["filename"]}')
            score_file = record['score_file']
        if offset + frames > len(frame_gt):
            raise ValueError(f'Score and annotation timeline mismatch: {video["filename"]}')
        actual = class_mask(segments[index], classes[index], label_indices, video)
        predicted = class_predictions(run_dir, score_file, tokens, class_count, primary_check)
        cells, tallies = np.unique(actual * class_count + predicted, return_counts=True)
        counts.flat[cells] += tallies
        disagreeing_gt_frames += int(np.count_nonzero(
            (actual > 0) != (frame_gt[offset:offset + frames] == 1)))
        offset += frames
    if offset != len(frame_gt) or counts.sum() != offset:
        raise ValueError(f'Expected one prediction per evaluation frame: {run_dir}')

    binary = np.bincount(
        2 * frame_gt.astype(np.intp) + (primary >= threshold), minlength=4).reshape(2, 2)
    metadata = {
        'dataset': spec.name,
        'source_summary': str(summary_path),
        'source_summary_kind': 'archived_without_video_records' if archived_summary else 'original',
        'source_run_id': report['run_id'],
        'source_summary_sha256': sha256(summary_path),
        'units': 'evaluation frames',
        'total_frames': offset,
        'binary_branch': spec.primary_branch,
        'binary_prediction': f'score >= {threshold}',
        'multiclass_prediction': 'argmax of restored Branch 2 class probabilities',
        'multiclass_actual': 'set of active segment GT labels; normal outside anomaly segments',
        'binary_vs_multiclass_gt_disagreement_frames': disagreeing_gt_frames,
    }
    rows = [0, *(1 << index for index in range(class_count - 1))]
    rows.extend(mask for mask in np.flatnonzero(counts.sum(axis=1)).tolist() if mask not in rows)
    row_labels = [
        spec.prompts[0] if mask == 0 else ' + '.join(
            name for index, name in enumerate(spec.prompts[1:]) if mask & (1 << index))
        for mask in rows
    ]
    return row_labels, counts[rows], binary, metadata


def write_table(path, row_labels, column_labels, counts):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['actual / predicted', *column_labels, 'Total'])
        for label, row in zip(row_labels, counts):
            writer.writerow([label, *row.tolist(), int(row.sum())])
        writer.writerow(['Total', *counts.sum(axis=0).tolist(), int(counts.sum())])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ucf-run', type=Path, default=REFERENCE_ROOT / 'ucf_raw_reference')
    parser.add_argument('--xd-run', type=Path, default=REFERENCE_ROOT / 'xd_raw_reference')
    parser.add_argument('--threshold', type=float, default=0.5, help='Binary threshold (default: 0.5)')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs/analyses')
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error('--threshold must be in [0, 1]')

    runs = [(ucf.SPEC, args.ucf_run), (xd.SPEC, args.xd_run)]
    tables = [collect_tables(run.expanduser().resolve(), spec, args.threshold) for spec, run in runs]
    _, output = create_run_dir('contingency_tables', 'full', output_root=args.output_root)
    binary_labels = ('Normal', 'Anomaly')
    for (spec, _), (rows, multiclass, binary, _) in zip(runs, tables):
        write_table(output / f'{spec.key}_multiclass.csv', rows, spec.prompts, multiclass)
        write_table(output / f'{spec.key}_binary.csv', binary_labels, binary_labels, binary)
    write_json(output / 'summary.json', {spec.key: table[3] for (spec, _), table in zip(runs, tables)})
    print(output.resolve())


if __name__ == '__main__':
    main()
