"""Shared ver4 input validation and geometry. No model import or execution here."""

import json
from pathlib import Path

import numpy as np

from .paths import ROOT, project_path, sha256
from .samplers import video_seed
from .samplers.context_ver3 import select_ver3

EXPECTED = {('reference', 0), ('uniform_ver3', 0), ('paired_ver3', 0)} | {
    ('relocated_ver3', seed) for seed in range(5)}
SPEC_VER4 = {'version': 4, 'dataset': 'xd', 'budget': 0.125,
             'target_quantiles': [1 / 3, 2 / 3], 'neighbor_radius_slots': 1,
             'near_gap_reference_rows': 1, 'window': 256, 'attention_block': 64,
             'stored_score_atol': 1e-5, 'repeat_score_atol': 1e-6,
             'target_selection': 'geometry_only; no_GT_or_score_input',
             'scope': 'exploratory_cached_diagnostics; no_speedup'}


def read(path):
    return json.loads(Path(path).read_text())


def load_ver3(sweep_path):
    """Bind eight completed source runs; do not require all current sources equal."""
    sweep_path = Path(sweep_path).expanduser().resolve()
    sweep = read(sweep_path)
    if sweep.get('complete') is not True or sweep.get('mode') != 'full' or sweep.get('error'):
        raise ValueError('Require a completed full ver3 sweep')
    groups = {}
    first = None
    for item in sweep['runs']:
        path = Path(item['summary'])
        if sha256(path) != item['sha256']:
            raise ValueError(f'Archived summary changed: {path}')
        report = read(path)
        key = report['sampling']['method'], report['sampling']['seed']
        if key != (item['method'], item['seed']) or key in groups:
            raise ValueError('Duplicate or inconsistent source method/seed')
        if (report.get('success') is not True or report.get('mode') != 'full'
                or report.get('dataset_key') != 'xd' or len(report.get('videos', [])) != 800
                or report.get('input_type') != 'cached_raw_features' or not report.get('metrics')
                or not report.get('comparison_signature') or not report.get('cache_identity_sha256')
                or report['experiment']['protocol']['restoration'] != 'nearest'
                or report['metrics']['primary']['branch'] != 'branch2'
                or report['metrics']['primary']['metric'] != 'ap'):
            raise ValueError(f'Invalid source run: {path}')
        if report['sampling']['budget'] != (1 if key[0] == 'reference' else 0.125):
            raise ValueError('Source budget differs from ver3 protocol')
        if first is None:
            first = report
        for field in ('comparison_signature', 'cache_identity_sha256', 'cache_input_sha256',
                      'reference_summary_sha256', 'model_options', 'prompts', 'package_sha256',
                      'vendored_source_sha256', 'dataset_identity_sha256', 'environment'):
            if report.get(field) != first.get(field):
                raise ValueError(f'Source runs differ in {field}')
        if report['input_sha256']['setup_manifest'] != first['input_sha256']['setup_manifest']:
            raise ValueError('Source manifests differ')
        groups[key] = (path, report)
    if set(groups) != EXPECTED:
        raise ValueError('Need exactly the eight prescribed ver3 runs')
    manifest_path = project_path(first['setup_manifest'])
    if sha256(manifest_path) != first['input_sha256']['setup_manifest']:
        raise ValueError('Source manifest changed')
    manifest = read(manifest_path)
    if len(manifest['videos']) != 800:
        raise ValueError('Incomplete test manifest')
    gt_path = project_path(read(ROOT / 'configs/datasets/xd.json')['frame_gt'])
    if sha256(gt_path) != first['input_sha256']['gt']:
        raise ValueError('GT content changed')
    gt = np.load(gt_path, allow_pickle=False)
    if gt.ndim != 1 or not np.isin(gt, [0, 1]).all() or np.unique(gt).size != 2:
        raise ValueError('Require binary frame GT with both labels')
    offset = 0
    for i, video in enumerate(manifest['videos']):
        if (video['official_csv_index'] != i or video['gt_offset'] != offset
                or video['evaluation_frames'] != 16 * video['reference_tokens']):
            raise ValueError('Invalid source timeline/order')
        for key, (_, report) in groups.items():
            record = report['videos'][i]
            expected_k = video['reference_tokens'] if key[0] == 'reference' else (video['reference_tokens'] + 7) // 8
            if any(record.get(k) != v for k, v in {
                    'video': video['filename'], 'official_csv_index': i, 'gt_offset': offset,
                    'reference_tokens': video['reference_tokens'], 'evaluation_frames': video['evaluation_frames'],
                    'selected_tokens': expected_k}.items()):
                raise ValueError('Archived video identity/timeline/K mismatch')
        offset += video['evaluation_frames']
    if offset != len(gt):
        raise ValueError('GT length and manifest disagree')
    return groups, manifest, gt, {'sweep': str(sweep_path), 'sha256': sha256(sweep_path),
                                 'summaries': sweep['runs']}


def score_arrays(path, report, video):
    """Read only direct valid-token scores; verify geometry for this source run."""
    record = report['videos'][video['official_csv_index']]
    score_path = path.parent / record['score_file']
    digest = sha256(score_path)
    with np.load(score_path, allow_pickle=False) as z:
        frames = z['selected_frame_indices'].copy()
        grid = z['reference_frame_indices'].copy()
        scores = {b: z[f'branch{b}_selected_scores'].copy() for b in (1, 2)}
    n = video['reference_tokens']
    method, seed = report['sampling']['method'], report['sampling']['seed']
    rows = (list(range(n)) if method == 'reference' else select_ver3(
        n, method, video_seed(seed, 'xd', video['filename']))[0])
    if (frames.ndim != 1 or grid.ndim != 1 or frames.dtype.kind not in 'iu'
            or grid.dtype.kind not in 'iu' or not np.array_equal(grid, np.arange(n) * 16)
            or not np.array_equal(frames, grid[rows])):
        raise ValueError('Source selection/grid does not follow ver3')
    for values in scores.values():
        if (values.shape != frames.shape or values.dtype.kind != 'f' or not np.isfinite(values).all()
                or np.any(values < 0) or np.any(values > 1)):
            raise ValueError('Invalid source token probabilities')
    return frames, grid, scores, {'file': str(score_path), 'sha256': digest}


def restore_full_layout(full_scores, selected_frames, reference_frames):
    """Counterfactual only: Full scores at observed coordinates, then fixed restoration."""
    from .timeline import restore_tokens, to_frames
    full_scores = np.asarray(full_scores)
    selected_frames, reference_frames = np.asarray(selected_frames), np.asarray(reference_frames)
    if full_scores.shape != reference_frames.shape:
        raise ValueError('Full scores and reference grid must have one row per candidate')
    rows = np.searchsorted(reference_frames, selected_frames)
    if np.any(rows >= len(reference_frames)) or not np.array_equal(reference_frames[rows], selected_frames):
        raise ValueError('Layout coordinates must belong to Full reference grid')
    return to_frames(restore_tokens(full_scores[rows], selected_frames, reference_frames, 'nearest'))


def target_positions(n):
    """At most two predeclared quantiles of eligible compressed input slots."""
    if type(n) is not int or n < 1:
        raise ValueError('Invalid candidate count')
    rows = list(range(0, n, 8))
    eligible = [p for p in range(1, len(rows) - 1) if 1 <= p % 64 <= 62]
    if not eligible:
        return []
    return sorted({eligible[(len(eligible) - 1) * q // 3] for q in (1, 2)})


def replace_neighbors(n, target_slot):
    """Keep target row/slot/K/window; replace only immediate neighboring rows."""
    base = list(range(0, n, 8))
    if target_slot not in range(1, len(base) - 1) or not 1 <= target_slot % 64 <= 62:
        raise ValueError('Target requires both neighbors within the same 64-slot attention block')
    changed = base.copy()
    target = base[target_slot]
    changed[target_slot - 1], changed[target_slot + 1] = target - 1, target + 1
    if (changed != sorted(set(changed)) or len(changed) != len(base)
            or changed[target_slot] != target or changed[-1] >= n or changed[0] < 0
            or [i for i, (a, b) in enumerate(zip(base, changed)) if a != b]
            != [target_slot - 1, target_slot + 1]):
        raise ValueError('Neighbor replacement violated geometry invariants')
    return base, changed
