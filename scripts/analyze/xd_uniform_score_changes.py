"""Compare pre-restoration XD scores at identical Full and Uniform frame coordinates."""

import argparse
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from sklearn.metrics import average_precision_score

from contingency_tables import REFERENCE_ROOT, checked_input
from selection_and_detection_12p5 import ANALYSES, BUDGET, primary_scores, write_csv
from vadfs.config import read_json
from vadfs.datasets import xd
from vadfs.datasets.base import validate_manifest
from vadfs.paths import ROOT, project_path, sha256
from vadfs.runs import create_run_dir, write_json
from vadfs.samplers import budget_count
from vadfs.timeline import restore_tokens, to_frames


BRANCH = xd.SPEC.primary_branch
BRANCHES = ('branch1', 'branch2')
CONDITIONS = ('full_reference', 'full_scores_restored_at_uniform_positions', 'uniform_12p5')
DEFAULT_FULL = REFERENCE_ROOT / 'xd_raw_reference'
DEFAULT_UNIFORM = REFERENCE_ROOT.parent / 'uniform/xd_raw_uniform_0.125'


def load_reports(full_dir, uniform_dir):
    full = read_json(full_dir / 'summary.json')
    uniform = read_json(uniform_dir / 'summary.json')
    for report, method, budget in ((full, 'reference', 1.0), (uniform, 'uniform', BUDGET)):
        if (not report['success'] or report['mode'] != 'full' or report['dataset_key'] != 'xd'
                or report['sampling']['method'] != method or report['sampling']['budget'] != budget
                or len(report['videos']) != xd.SPEC.expected_videos
                or report['metrics']['primary']['branch'] != BRANCH):
            raise ValueError(f'Expected a complete XD {method} run at budget {budget}')
    for field in ('dataset_identity_sha256', 'prompts', 'model_options', 'vendored_source_sha256'):
        if full[field] != uniform[field]:
            raise ValueError(f'Full and Uniform have different {field}')
    for field in ('setup_manifest', 'gt', 'clip_checkpoint', 'vadclip_checkpoint'):
        if full['input_sha256'][field] != uniform['input_sha256'][field]:
            raise ValueError(f'Full and Uniform have different {field} hashes')
    if (full['input_type'] != 'raw' or uniform['input_type'] != 'cached_raw_features'
            or sha256(full_dir / 'summary.json') != uniform['reference_summary_sha256']):
        raise ValueError('Uniform must use the supplied Full run as its cached feature source')
    return full, uniform


def load_video_scores(run_dir, record, video, expected_count, branch):
    for report_key, manifest_key in (('video', 'filename'), ('official_csv_index', 'official_csv_index'),
                                     ('gt_offset', 'gt_offset'), ('evaluation_frames', 'evaluation_frames'),
                                     ('reference_tokens', 'reference_tokens')):
        if record[report_key] != video[manifest_key]:
            raise ValueError(f'Run and manifest disagree on {report_key}: {record["video"]}')
    path = (run_dir / record['score_file']).resolve()
    if not path.is_relative_to(run_dir / 'scores'):
        raise ValueError(f'Invalid score file path: {path}')
    with np.load(path, allow_pickle=False) as archive:
        grid = archive['reference_frame_indices']
        indices = archive['selected_frame_indices']
        scores = archive[f'{branch}_selected_scores']
    if (grid.shape != (video['reference_tokens'],) or indices.shape != (expected_count,)
            or scores.shape != indices.shape or record['selected_tokens'] != expected_count
            or grid.dtype.kind not in 'iu' or indices.dtype.kind not in 'iu'
            or grid[0] < 0 or grid[-1] >= video['evaluation_frames']
            or np.any(grid[1:] <= grid[:-1]) or np.any(indices[1:] <= indices[:-1])
            or scores.dtype.kind != 'f' or not np.isfinite(scores).all()
            or np.any(scores < 0) or np.any(scores > 1)):
        raise ValueError(f'Invalid coordinates, scores or frame count: {path}')
    return grid, indices, scores


def collect_pairs(full_dir, uniform_dir, full, uniform, videos, gt, branch):
    full_timeline = primary_scores(full_dir, xd.SPEC, gt, branch=branch)
    uniform_timeline = primary_scores(uniform_dir, xd.SPEC, gt, branch=branch)
    restored_timeline = np.empty_like(full_timeline)
    method = uniform['experiment']['protocol']['restoration']
    pairs = []
    offset = 0
    for video, full_record, uniform_record in zip(videos, full['videos'], uniform['videos']):
        count = video['reference_tokens']
        end = offset + video['evaluation_frames']
        grid, full_indices, full_scores = load_video_scores(full_dir, full_record, video, count, branch)
        uniform_grid, selected, uniform_scores = load_video_scores(
            uniform_dir, uniform_record, video, budget_count(count, BUDGET), branch)
        if video['gt_offset'] != offset or not np.array_equal(grid, full_indices) or not np.array_equal(grid, uniform_grid):
            raise ValueError(f'Expected identical reference grids and ordered GT: {video["filename"]}')
        positions = np.searchsorted(grid, selected)
        if np.any(positions >= count) or not np.array_equal(grid[positions], selected):
            raise ValueError(f'Uniform selection is outside the Full grid: {video["filename"]}')
        matched_full = full_scores[positions]
        if (not np.array_equal(to_frames(full_scores), full_timeline[offset:end])
                or not np.array_equal(to_frames(restore_tokens(uniform_scores, selected, grid, method)),
                                      uniform_timeline[offset:end])):
            raise ValueError(f'Could not reproduce the saved {branch} timeline: {video["filename"]}')
        restored_timeline[offset:end] = to_frames(restore_tokens(matched_full, selected, grid, method))

        fps = float(full_record['fps'])
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f'Invalid Full FPS: {video["filename"]}')
        local_gt = gt[offset:end]
        short_event = np.zeros(len(local_gt), dtype=bool)
        for start, stop in video['intervals_raw']:
            start, stop = max(0, start), min(len(local_gt), stop)
            if 0 < stop - start <= fps:
                short_event[start:stop] = True
        for frame, full_score, uniform_score, position in zip(selected, matched_full, uniform_scores, positions):
            frame = int(frame)
            pairs.append({
                'video_index': video['official_csv_index'], 'video': video['filename'],
                'original_frame': frame, 'full_token': int(position),
                'gt_is_anomaly': int(local_gt[frame]), 'in_event_le_1s': bool(short_event[frame]),
                'full_score': float(full_score), 'uniform_score': float(uniform_score),
                'uniform_minus_full': float(uniform_score) - float(full_score),
            })
        offset = end
    if offset != gt.size:
        raise ValueError('Manifest and binary GT have different evaluation lengths')
    return pairs, (full_timeline, restored_timeline, uniform_timeline)


def summarize(group, label):
    changes = np.asarray([row['uniform_minus_full'] for row in group])
    if not len(changes):
        raise ValueError(f'No matching frames for {label}')
    q1, median, q3 = np.quantile(changes, [0.25, 0.5, 0.75])
    return {
        'frame_group': label, 'n_frames': len(changes),
        'n_videos': len({row['video_index'] for row in group}),
        'mean_delta': float(changes.mean()), 'median_delta': float(median),
        'q25_delta': float(q1), 'q75_delta': float(q3),
        'fraction_decreased': float(np.mean(changes < 0)),
        'fraction_unchanged': float(np.mean(changes == 0)),
        'fraction_increased': float(np.mean(changes > 0)),
    }


def write_case_figure(path, case, video, fps, pairs, gt, timelines, branch):
    offset = video['gt_offset']
    focus = case['original_frame']
    radius = round(6 * fps)
    start = max(0, focus - radius)
    stop = min(video['evaluation_frames'], focus + radius + 1)
    left, right, top, bottom = 74, 866, 91, 352

    def x(frame):
        return left + (frame - start) * (right - left) / (stop - start)

    def y(score):
        return bottom - score * (bottom - top)

    elements = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="430" viewBox="0 0 900 430">',
                '<rect width="900" height="430" fill="white"/>',
                '<g font-family="Arial, sans-serif" fill="#183244">']

    def label(px, py, text, size=14, anchor='start'):
        elements.append(f'<text x="{px}" y="{py}" font-size="{size}" text-anchor="{anchor}">{escape(text)}</text>')

    label(left, 29, f'XD-Violence · Uniform 12.5% · target frame {focus}', 22)
    label(left, 54, f'Full score {case["full_score"]:.3f} · Uniform score {case["uniform_score"]:.3f}')
    local_gt = gt[offset + start:offset + stop].astype(np.int8)
    edges = np.flatnonzero(np.diff(local_gt, prepend=0, append=0))
    for a, b in zip(edges[::2], edges[1::2]):
        elements.append(f'<rect x="{x(start+a):.2f}" y="{top}" width="{x(start+b)-x(start+a):.2f}" '
                        f'height="{bottom-top}" fill="#FDE8BD"/>')
    for score in (0, 0.25, 0.5, 0.75, 1):
        elements.append(f'<path d="M {left} {y(score)} H {right}" stroke="#DCE4EB"/>')
        label(left - 10, y(score) + 5, f'{score:g}', anchor='end')
    label(left, top - 10, f'Branch {branch[-1]} anomaly score')
    colors = ('#185D97', '#596D7F', '#008172')
    names = ('Full', 'Full scores at Uniform positions, restored', 'Uniform 12.5%')
    for scores, color, name, legend_x in zip(timelines, colors, names, (74, 176, 526)):
        local = scores[offset + start:offset + stop]
        changes = np.flatnonzero(local[1:] != local[:-1]) + 1
        commands = [f'M {x(start):.2f} {y(local[0]):.2f}']
        commands.extend(f'H {x(start+i):.2f} V {y(local[i]):.2f}' for i in changes)
        commands.append(f'H {x(stop):.2f}')
        dash = ' stroke-dasharray="6 4"' if name == names[1] else ''
        elements.append(f'<path d="{" ".join(commands)}" fill="none" stroke="{color}" stroke-width="2.4"{dash}/>')
        elements.append(f'<path d="M {legend_x} 411 h 18" stroke="{color}" stroke-width="2.4"{dash}/>')
        label(legend_x + 24, 416, name)
    for row in pairs:
        frame = row['original_frame']
        if row['video_index'] == case['video_index'] and start <= frame < stop:
            for key, color in (('full_score', colors[0]), ('uniform_score', colors[2])):
                radius = 5 if frame == focus else 3
                elements.append(f'<circle cx="{x(frame):.2f}" cy="{y(row[key]):.2f}" r="{radius}" fill="{color}"/>')
    for frame in np.linspace(start, stop - 1, num=5, dtype=int):
        label(x(frame), 375, f'{frame / fps:.1f}s', anchor='middle')
    label((left + right) / 2, 394, 'Video time', anchor='middle')
    elements.append('<rect x="681" y="403" width="12" height="12" fill="#FDE8BD"/>')
    label(699, 416, 'Annotated anomaly')
    elements.append('</g></svg>')
    path.write_text('\n'.join(elements) + '\n', encoding='utf-8')


def analyze_branch(full_dir, uniform_dir, full, uniform, videos, gt, branch):
    pairs, timelines = collect_pairs(full_dir, uniform_dir, full, uniform, videos, gt, branch)
    groups = [
        summarize([row for row in pairs if row['gt_is_anomaly'] == target], label)
        for target, label in ((0, 'normal'), (1, 'anomaly'))
    ]
    full_ap, restored_ap, uniform_ap = (float(average_precision_score(gt, scores)) for scores in timelines)
    for value, report in ((full_ap, full), (uniform_ap, uniform)):
        if not np.isclose(value, report['metrics'][branch]['ap'], rtol=0, atol=1e-8):
            raise ValueError(f'Recomputed frame AP differs from the saved {branch} AP: {report["run_id"]}')
    summary = {
        'frame_ap_pct': dict(zip(CONDITIONS, (100 * full_ap, 100 * restored_ap, 100 * uniform_ap))),
        'frame_ap_changes_pp': {
            'full_to_restored': 100 * (restored_ap - full_ap),
            'restored_to_uniform': 100 * (uniform_ap - restored_ap),
        },
        'frame_groups': groups,
    }
    return pairs, timelines, summary


def compare_pair_directions(branch1_pairs, branch2_pairs):
    """Count all nine direction combinations on the same GT-labeled frames."""
    identity_fields = ('video_index', 'video', 'original_frame', 'full_token', 'gt_is_anomaly', 'in_event_le_1s')
    score_fields = ('full_score', 'uniform_score', 'uniform_minus_full')
    paired = []
    counts = np.zeros((2, 3, 3), dtype=int)
    for branch1, branch2 in zip(branch1_pairs, branch2_pairs, strict=True):
        identity = {key: branch1[key] for key in identity_fields}
        if any(branch2[key] != value for key, value in identity.items()):
            raise ValueError('The branches do not cover the same frames in the same order')
        paired.append({
            **identity,
            **{f'branch1_{key}': branch1[key] for key in score_fields},
            **{f'branch2_{key}': branch2[key] for key in score_fields},
        })
        direction1 = int(np.sign(branch1['uniform_minus_full'])) + 1
        direction2 = int(np.sign(branch2['uniform_minus_full'])) + 1
        counts[identity['gt_is_anomaly'], direction1, direction2] += 1

    directions = ('decreased', 'unchanged', 'increased')
    rows = []
    for label, table in zip(('normal', 'anomaly'), counts):
        total = int(table.sum())
        for i, direction1 in enumerate(directions):
            for j, direction2 in enumerate(directions):
                count = int(table[i, j])
                rows.append({
                    'frame_group': label, 'branch1_direction': direction1,
                    'branch2_direction': direction2, 'n_frames': count,
                    'fraction_of_group': count / total,
                })
    return paired, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-run', type=Path, default=DEFAULT_FULL)
    parser.add_argument('--uniform-run', type=Path, default=DEFAULT_UNIFORM)
    parser.add_argument('--output-root', type=Path, default=ANALYSES)
    parser.add_argument('--compare-branches', action='store_true',
                        help='Compare both saved branches and count their joint score changes.')
    args = parser.parse_args()
    full_dir, uniform_dir = args.full_run.resolve(), args.uniform_run.resolve()
    full, uniform = load_reports(full_dir, uniform_dir)
    hashes = full['input_sha256']
    manifest = read_json(checked_input(full_dir / 'manifest.json', hashes['setup_manifest']))
    videos = validate_manifest(manifest, xd.SPEC)
    if manifest['dataset_identity_sha256'] != full['dataset_identity_sha256']:
        raise ValueError('Manifest and Full run identify different datasets')
    config = read_json(ROOT / 'configs/datasets/xd.json')
    gt = np.load(checked_input(project_path(config['frame_gt']), hashes['gt']), allow_pickle=False)
    if gt.ndim != 1 or not np.array_equal(np.unique(gt), [0, 1]):
        raise ValueError('Expected a binary frame timeline containing both classes')

    branches = BRANCHES if args.compare_branches else (BRANCH,)
    analyses = {branch: analyze_branch(full_dir, uniform_dir, full, uniform, videos, gt, branch)
                for branch in branches}
    pairs, timelines, primary_summary = analyses[BRANCH]
    anomaly = [row for row in pairs if row['gt_is_anomaly'] == 1]
    short = [row for row in anomaly if row['in_event_le_1s']]
    examples = short or anomaly
    median = np.median([row['uniform_minus_full'] for row in examples])
    case = min(examples, key=lambda row: abs(row['uniform_minus_full'] - median))
    video_position = case['video_index']
    video = videos[video_position]
    fps = full['videos'][video_position]['fps']
    report = {
        'dataset': xd.SPEC.name, 'branch': BRANCH, 'uniform_budget': BUDGET,
        'ap_scale': 'percent', **primary_summary,
        'delta_score_definition': 'Uniform direct score minus Full direct score at the same frame',
        'example': case,
        'example_rule': ('Nearest to median primary-branch delta among selected GT-positive frames in events '
                         '<=1 second; fallback to all selected GT-positive frames'),
        'restoration': uniform['experiment']['protocol']['restoration'],
        'full_run': str(full_dir), 'uniform_run': str(uniform_dir),
        'full_summary_sha256': sha256(full_dir / 'summary.json'),
        'uniform_summary_sha256': sha256(uniform_dir / 'summary.json'),
        'interpretation': 'The sparse input changes context, token positions and window composition together.',
    }
    if args.compare_branches:
        paired, joint = compare_pair_directions(analyses['branch1'][0], analyses['branch2'][0])
        report['branches'] = {branch: result[2] for branch, result in analyses.items()}
        report['joint_score_directions'] = joint
        report['direction_rule'] = 'Exact sign of each score difference, with zero counted as unchanged'
        _, output = create_run_dir('xd_uniform_branch_comparison', '12p5', output_root=args.output_root)
        write_csv(output / 'paired_frames.csv', paired)
        write_csv(output / 'joint_score_directions.csv', joint)
        write_csv(output / 'branch_ap.csv', [
            {'branch': branch, **summary['frame_ap_pct'],
             **{f'{key}_pp': value for key, value in summary['frame_ap_changes_pp'].items()}}
            for branch, summary in report['branches'].items()
        ])
        write_csv(output / 'score_changes_by_gt.csv', [
            {'branch': branch, **group}
            for branch, summary in report['branches'].items() for group in summary['frame_groups']
        ])
        for branch, (branch_pairs, branch_timelines, _) in analyses.items():
            branch_case = next(row for row in branch_pairs if row['video_index'] == video_position
                               and row['original_frame'] == case['original_frame'])
            write_case_figure(output / f'example_{branch}.svg', branch_case, video, fps,
                              branch_pairs, gt, branch_timelines, branch)
    else:
        _, output = create_run_dir('xd_uniform_score_changes', '12p5', output_root=args.output_root)
        write_csv(output / 'paired_frames.csv', pairs)
        write_csv(output / 'score_changes_by_gt.csv', primary_summary['frame_groups'])
        write_case_figure(output / 'example.svg', case, video, fps, pairs, gt, timelines, BRANCH)
    write_json(output / 'summary.json', report)
    for branch, (_, _, summary) in analyses.items():
        for name, value in summary['frame_ap_pct'].items():
            prefix = f'{branch} ' if args.compare_branches else ''
            print(f'{prefix}{name}: AP {value:.2f}%')
    print(output.resolve())


if __name__ == '__main__':
    main()
