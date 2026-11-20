"""Compare distance graphs with fixed Uniform inputs and unchanged VadCLIP weights."""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vadfs import vadclip
from vadfs.config import read_json
from vadfs.datasets.base import validate_manifest
from vadfs.datasets import ucf, xd
from vadfs.paths import ROOT, sha256
from vadfs.pipeline import load_cached_reference
from vadfs.runs import create_run_dir, now, record_provenance, write_json
from vadfs.timeline import restore_tokens, to_frames


SHARED = Path('/workspace/vad')
CONDITIONS = ('token_index', 'original_distance', 'self_only')
MASS_CONTROLS = ('original_distance_matched_mass', 'token_index_reduced_mass')
BRANCHES = ('branch1', 'branch2')
CONTRASTS = ((1, 0), (2, 0), (1, 2))
MASS_CONTRASTS = ((3, 0), (4, 0), (3, 1), (4, 1))
BLOCK = vadclip.FRAMES_PER_TOKEN
ARCHIVED_SCORE_ATOL = 5e-4
ARCHIVED_AP_ATOL_PP = 0.05
BOOTSTRAP_SEED = 234


def verified_path(path, digest):
    if sha256(path) != digest:
        raise ValueError(f'Input hash differs from recorded run: {path}')
    return path


def load_inputs(args, spec, stride):
    full = read_json(args.full_run / 'summary.json')
    sparse = read_json(args.uniform_run / 'summary.json')
    for report, input_type, method, budget in (
            (full, 'raw', 'reference', 1), (sparse, 'cached_raw_features', 'uniform', 1 / stride)):
        if (not report['success'] or report['mode'] != 'full' or report['dataset_key'] != spec.key
                or report['input_type'] != input_type or report['sampling']['method'] != method
                or report['sampling']['budget'] != budget or len(report['videos']) != spec.expected_videos
                or report['experiment']['protocol']['restoration'] != 'nearest'
                or report['prompts'] != list(spec.prompts)):
            raise ValueError(f'Unexpected archived input: {report["run_id"]}')
    verified_path(args.full_run / 'summary.json', sparse['reference_summary_sha256'])
    for name in ('dataset_identity_sha256', 'model_options', 'vendored_source_sha256', 'extraction_config'):
        if full[name] != sparse[name]:
            raise ValueError(f'Full and Uniform differ in {name}')
    hashes = full['input_sha256']
    for key in ('setup_manifest', 'gt', 'clip_checkpoint', 'vadclip_checkpoint', 'tokenizer_vocabulary'):
        if hashes[key] != sparse['input_sha256'][key]:
            raise ValueError(f'Full and Uniform differ in {key}')
    for path, digest in full['vendored_source_sha256'].items():
        verified_path(ROOT / path, digest)
    verified_path(ROOT / 'VadCLIP/src/clip/bpe_simple_vocab_16e6.txt.gz', hashes['tokenizer_vocabulary'])
    manifest = read_json(verified_path(args.full_run / 'manifest.json', hashes['setup_manifest']))
    if manifest['dataset_identity_sha256'] != full['dataset_identity_sha256']:
        raise ValueError('Manifest belongs to a different dataset')
    videos = validate_manifest(manifest, spec)
    dataset = read_json(ROOT / f'configs/datasets/{spec.key}.json')
    gt_path = verified_path(ROOT / dataset['frame_gt'], hashes['gt'])
    gt = np.load(gt_path, allow_pickle=False)
    if gt.ndim != 1 or not np.array_equal(np.unique(gt), [0, 1]):
        raise ValueError('Expected the complete binary ground-truth timeline')
    return full, sparse, videos, gt


def saved_scores(folder, record, video, grid, expected_indices):
    for key, manifest_key in (('video', 'filename'), ('gt_offset', 'gt_offset'),
                              ('official_csv_index', 'official_csv_index'),
                              ('reference_tokens', 'reference_tokens'), ('evaluation_frames', 'evaluation_frames')):
        if record[key] != video[manifest_key]:
            raise ValueError(f'{key} does not match the manifest: {video["filename"]}')
    path = (folder / record['score_file']).resolve()
    if not path.is_relative_to(folder / 'scores'):
        raise ValueError(f'Invalid score path: {path}')
    with np.load(path, allow_pickle=False) as scores:
        if (not np.array_equal(scores['reference_frame_indices'], grid)
                or not np.array_equal(scores['selected_frame_indices'], expected_indices)):
            raise ValueError(f'Unexpected frame selection: {path}')
        values = [scores[f'{branch}_selected_scores'] for branch in BRANCHES]
    if any(v.shape != expected_indices.shape or not np.isfinite(v).all() for v in values):
        raise ValueError(f'Invalid direct scores: {path}')
    return np.stack(values), sha256(path)


def distance_block(stock, positions, scale, condition):
    if condition == 'self_only':
        return torch.eye(len(positions), dtype=stock.dtype, device=stock.device)
    distances = (positions[:, None] - positions[None, :]).abs().to(stock.dtype) / BLOCK
    physical = torch.exp(-distances / scale)
    if condition == 'original_distance':
        return physical
    stock_mass = stock.sum(dim=-1, keepdim=True)
    physical_mass = physical.sum(dim=-1, keepdim=True)
    if condition == 'original_distance_matched_mass':
        return physical * (stock_mass / physical_mass)
    if condition == 'token_index_reduced_mass':
        return stock * (physical_mass / stock_mass)
    raise ValueError(f'Unknown distance graph: {condition}')


def infer_with_graph(model, features, indices, window, prompts, device, condition, check_full=False):
    if condition == 'token_index':
        return np.stack(vadclip.infer_feature(model, features, window, prompts, device)[:2])

    def substitute_valid_edges(module, args, original):
        if original.shape != (len(indices) // window + 1, window, window):
            raise ValueError('Unexpected upstream distance graph shape')
        adjusted = original.clone()
        scale = torch.exp(torch.tensor(1.)).to(original.device)
        for batch, start in enumerate(range(0, len(indices), window)):
            positions = torch.as_tensor(indices[start:start + window], device=original.device)
            n = len(positions)
            block = distance_block(original[batch, :n, :n], positions, scale, condition)
            adjusted[batch, :n, :n] = block
        # Edges involving padding, including the upstream extra empty window, stay identical.
        if check_full and not torch.equal(original, adjusted):
            raise ValueError('Original-frame graph differs from the stock graph at 100%')
        return adjusted

    handle = model.disAdj.register_forward_hook(substitute_valid_edges)
    try:
        return np.stack(vadclip.infer_feature(model, features, window, prompts, device)[:2])
    finally:
        handle.remove()


def grouped_metrics(score_groups, group_count, positive, negative, recall_target):
    """AP, interpolated FPR, and AUC from tied score blocks and frame counts."""
    tp = np.bincount(score_groups, weights=positive, minlength=group_count).cumsum()
    fp = np.bincount(score_groups, weights=negative, minlength=group_count).cumsum()
    if tp[-1] == 0 or fp[-1] == 0:
        raise ValueError('Both labels must be present in every evaluation sample')
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=tp + fp != 0)
    ap = np.dot(np.diff(tp, prepend=0), precision) / tp[-1]
    auc = np.dot(np.diff(fp, prepend=0), tp + np.concatenate(([0.], tp[:-1]))) / (2 * tp[-1] * fp[-1])
    target = recall_target * tp[-1]
    if abs(target - round(target)) < 1e-8:
        target = round(target)
    index = int(np.searchsorted(tp, target, side='left'))
    previous_tp = tp[index - 1] if index else 0
    previous_fp = fp[index - 1] if index else 0
    exact_fp = fp[index]
    if tp[index] != target:
        exact_fp = previous_fp + (exact_fp - previous_fp) * (target - previous_tp) / (tp[index] - previous_tp)
    return np.array([100 * ap, 100 * exact_fp / fp[-1], 100 * auc])


def ranks(scores):
    unique, inverse = np.unique(-scores, return_inverse=True)
    return inverse, len(unique)


def bootstrap(primary_scores, positive, video_index, strata, recall_target, reps, conditions, contrasts):
    groups = [ranks(scores) for scores in primary_scores]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas = np.empty((reps, len(contrasts), 3))
    for rep in range(reps):
        # Draw the same videos for every condition and preserve the two video stratum sizes.
        sampled = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in strata])
        weights = np.bincount(sampled, minlength=sum(map(len, strata)))[video_index]
        tp, fp = positive * weights, (BLOCK - positive) * weights
        values = np.array([grouped_metrics(*group, tp, fp, recall_target) for group in groups])
        deltas[rep] = [values[left] - values[right] for left, right in contrasts]
    intervals = np.percentile(deltas, (2.5, 97.5), axis=0)
    return {f'{conditions[left]}_vs_{conditions[right]}':
            dict(zip(('ap_delta_pp', 'fpr_delta_pp', 'roc_auc_delta_pp'), intervals[:, i].T.tolist(), strict=True))
            for i, (left, right) in enumerate(contrasts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('xd', 'ucf'), default='xd')
    parser.add_argument('--budget', choices=(.5, .25, .125), type=float, default=.125)
    parser.add_argument('--row-mass-controls', action='store_true')
    parser.add_argument('--full-run', type=Path)
    parser.add_argument('--uniform-run', type=Path)
    parser.add_argument('--clip-checkpoint', type=Path, default=SHARED / 'checkpoints/clip/ViT-B-16.pt')
    parser.add_argument('--vadclip-checkpoint', type=Path)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs/analyses')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--bootstrap-reps', type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap_reps < 2:
        parser.error('--bootstrap-reps must be at least 2')
    spec = {'xd': xd.SPEC, 'ucf': ucf.SPEC}[args.dataset]
    stride = int(1 / args.budget)
    conditions = (*CONDITIONS, *MASS_CONTROLS) if args.row_mass_controls else CONDITIONS
    contrasts = (*CONTRASTS, *MASS_CONTRASTS) if args.row_mass_controls else CONTRASTS
    args.full_run = (args.full_run or SHARED / f'results/baselines/{spec.key}_raw_reference').resolve()
    args.uniform_run = (args.uniform_run or SHARED / f'results/uniform/{spec.key}_raw_uniform_{args.budget:g}').resolve()
    full, sparse, videos, gt = load_inputs(args, spec, stride)
    hashes = full['input_sha256']
    clip = verified_path(args.clip_checkpoint, hashes['clip_checkpoint'])
    checkpoint = verified_path(args.vadclip_checkpoint or SHARED / f'checkpoints/vadclip/model_{spec.key}.pth',
                               hashes['vadclip_checkpoint'])
    seed = full['model_options']['seed']
    options = vadclip.default_options(seed, dataset=spec.key)
    if {key: getattr(options, key) for key in vadclip.MODEL_OPTION_KEYS} != full['model_options']:
        raise ValueError('Current VadCLIP options differ from the archived runs')
    vadclip.check_device(args.device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = vadclip.load_model(options, args.device, clip, checkpoint)
    window, prompts = options.visual_length, full['prompts']
    budget_label = f'{100 * args.budget:g}'.replace('.', 'p')
    prefix = f'{spec.key}_distance_graph_{budget_label}'
    if args.row_mass_controls:
        prefix += '_mass'
    run_id, output = create_run_dir(prefix, 'full', output_root=args.output_root.resolve())
    print(f'Output: {output}', flush=True)

    n = sum(video['reference_tokens'] for video in videos)
    if n * BLOCK != len(gt):
        raise ValueError('Manifest and GT timelines differ')
    positive = gt.reshape(n, BLOCK).sum(axis=1)
    video_index = np.repeat(np.arange(len(videos)), [v['reference_tokens'] for v in videos])
    restored = np.empty((len(conditions), len(BRANCHES), n), dtype=np.float32)
    replayed = np.empty((len(BRANCHES), n), dtype=np.float32)
    selected_points, sources = [], []
    archived_max_error = dict(full=[0., 0.], uniform=[0., 0.])
    full_identity_error = 0.
    offset = 0
    with (np.load(args.full_run / 'frame_scores.npz', allow_pickle=False) as full_npz,
          np.load(args.uniform_run / 'frame_scores.npz', allow_pickle=False) as sparse_npz):
        full_saved = [full_npz[k] for k in BRANCHES]
        sparse_saved = [sparse_npz[k] for k in BRANCHES]
    if any(scores.shape != gt.shape for scores in [*full_saved, *sparse_saved]):
        raise ValueError('Archived frame scores do not cover the official test timeline')
    for index, (video, full_row, sparse_row) in enumerate(zip(videos, full['videos'], sparse['videos'], strict=True)):
        count = video['reference_tokens']
        stop = offset + count
        frame_slice = slice(offset * BLOCK, stop * BLOCK)
        if frame_slice.start != video['gt_offset']:
            raise ValueError(f'Unexpected GT offset: {video["filename"]}')
        grid = np.arange(count, dtype=np.int64) * BLOCK
        indices = grid[::stride]
        full_direct, full_hash = saved_scores(args.full_run, full_row, video, grid, grid)
        sparse_direct, sparse_hash = saved_scores(args.uniform_run, sparse_row, video, grid, indices)
        feature_path = args.full_run / 'features' / Path(full_row['score_file']).name
        feature_hash = sparse_row['input_sha256']
        features, _ = load_cached_reference(verified_path(feature_path, feature_hash), grid,
                                            np.dtype(video['feature_dtype']), expected_fps=full_row['fps'])
        full_before = infer_with_graph(model, features, grid, window, prompts, args.device, CONDITIONS[0])
        full_after = infer_with_graph(model, features, grid, window, prompts, args.device, CONDITIONS[1], check_full=True)
        full_identity_error = max(full_identity_error, float(np.max(np.abs(full_after - full_before))))
        if full_identity_error > 1e-6:
            raise ValueError(f'Full score identity failed: {video["filename"]}')
        sparse_features = features[::stride]
        direct = np.stack([infer_with_graph(model, sparse_features, indices, window, prompts, args.device, condition)
                           for condition in conditions])
        for j, branch in enumerate(BRANCHES):
            archived_uniform = restore_tokens(sparse_direct[j], indices, grid)
            if (not np.array_equal(to_frames(full_direct[j]), full_saved[j][frame_slice])
                    or not np.array_equal(to_frames(archived_uniform), sparse_saved[j][frame_slice])):
                raise ValueError(f'Saved direct scores disagree with the archived timeline: {video["filename"]}')
            for key, observed, saved in (('full', full_before[j], full_direct[j]),
                                         ('uniform', direct[0, j], sparse_direct[j])):
                error = float(np.max(np.abs(observed - saved)))
                archived_max_error[key][j] = max(error, archived_max_error[key][j])
                if error > ARCHIVED_SCORE_ATOL:
                    raise ValueError(f'{key} archived score replay fails for {branch}: {video["filename"]} ({error:g})')
            for k in range(len(conditions)):
                restored[k, j, offset:stop] = restore_tokens(direct[k, j], indices, grid)
            replayed[j, offset:stop] = restore_tokens(full_direct[j, ::stride], indices, grid)
        selected_points.append((np.full(len(indices), index), indices, gt[frame_slice][indices], direct))
        sources.append({'video': video['filename'], 'feature_sha256': feature_hash,
                        'full_score_sha256': full_hash, 'uniform_score_sha256': sparse_hash})
        offset = stop
        if (index + 1) % 50 == 0:
            print(f'{index + 1}/{len(videos)} videos checked', flush=True)

    selected_gt = np.concatenate([p[2] for p in selected_points])
    selected = np.concatenate([p[3] for p in selected_points], axis=-1)
    np.savez_compressed(output / 'selected_scores.npz', conditions=conditions, branches=BRANCHES,
                        video_index=np.concatenate([p[0] for p in selected_points]),
                        frame_index=np.concatenate([p[1] for p in selected_points]),
                        gt=selected_gt, scores=selected)
    np.savez_compressed(output / 'restored_scores.npz', conditions=conditions, branches=BRANCHES,
                        tokens=restored, video_index=video_index, positive_frames_per_token=positive)

    metrics, score_replay = {}, {}
    for j, branch in enumerate(BRANCHES):
        recall_target = float(np.mean(full_saved[j][gt == 1] >= 0.5))
        rows = {}
        for k, condition in enumerate(conditions):
            ap, fpr, auc = grouped_metrics(*ranks(restored[k, j]), positive, BLOCK - positive, recall_target)
            dense = to_frames(restored[k, j])
            exact_ap = float(average_precision_score(gt, dense)) * 100
            dense_fpr, dense_recall, _ = roc_curve(gt, dense, drop_intermediate=False)
            i = int(np.searchsorted(dense_recall, recall_target, side='left'))
            expected_fpr = 100 * (dense_fpr[i] if dense_recall[i] == recall_target else np.interp(
                recall_target, dense_recall[i - 1:i + 1], dense_fpr[i - 1:i + 1]))
            if not np.allclose([ap, fpr, auc], [exact_ap, expected_fpr, 100 * roc_auc_score(gt, dense)],
                               rtol=0, atol=1e-8):
                raise ValueError('Grouped token metrics disagree with expanded frame metrics')
            shifts = selected[k, j] - selected[0, j]
            groups = {name: {'n': int(len(delta)), 'mean': float(np.mean(delta)),
                             'median': float(np.median(delta)), 'increased_fraction': float(np.mean(delta > 0))}
                      for name, label in (('normal', 0), ('anomaly', 1))
                      if len(delta := shifts[selected_gt == label])}
            rows[condition] = {'ap_pct': float(ap), 'roc_auc_pct': float(auc),
                               'fpr_at_matched_recall_pct': float(fpr),
                               'selected_score_shift': groups}
        if abs(rows[CONDITIONS[0]]['ap_pct'] - 100 * sparse['metrics'][branch]['ap']) > ARCHIVED_AP_ATOL_PP:
            raise ValueError(f'{branch} replay AP differs from archived Uniform AP')
        for row in rows.values():
            row['ap_delta_pp'] = row['ap_pct'] - rows[CONDITIONS[0]]['ap_pct']
            row['roc_auc_delta_pp'] = row['roc_auc_pct'] - rows[CONDITIONS[0]]['roc_auc_pct']
            row['fpr_delta_pp'] = row['fpr_at_matched_recall_pct'] - rows[CONDITIONS[0]]['fpr_at_matched_recall_pct']
        metrics[branch] = {'recall_target_pct': recall_target * 100, 'conditions': rows}
        score_replay[branch] = {}
        for name, scores in (('full', full_saved[j][::BLOCK]), ('full_scores_at_selected_frames', replayed[j]),
                             ('sparse_inference', restored[0, j])):
            values = grouped_metrics(*ranks(scores), positive, BLOCK - positive, recall_target)
            score_replay[branch][name] = dict(zip(
                ('ap_pct', 'fpr_at_matched_recall_pct', 'roc_auc_pct'), values.tolist(), strict=True))

    print(f'Computing {args.bootstrap_reps} paired video bootstrap replicates', flush=True)
    labels = np.array([v['is_anomaly'] for v in videos])
    primary = metrics[spec.primary_branch]
    ci = bootstrap(restored[:, BRANCHES.index(spec.primary_branch)], positive, video_index,
                   (np.flatnonzero(~labels), np.flatnonzero(labels)), primary['recall_target_pct'] / 100,
                   args.bootstrap_reps, conditions, contrasts)
    report = {
        'run_id': run_id, 'finished_at': now().isoformat(), 'success': True, 'dataset': spec.key, 'videos': len(videos),
        'scope': 'cached-feature model diagnostic', 'budget': args.budget, 'selected_frames': len(selected_gt),
        'primary': {'branch': spec.primary_branch, 'metric': spec.primary_metric},
        'graph_change': 'valid-token edges only; edges involving padding remain unchanged',
        'sources': {'full_run': str(args.full_run), 'uniform_run': str(args.uniform_run), 'videos': sources},
        'input_sha256': {'full_summary': sha256(args.full_run / 'summary.json'),
                         'uniform_summary': sha256(args.uniform_run / 'summary.json'),
                         'manifest': hashes['setup_manifest'], 'ground_truth': hashes['gt'],
                         'clip_checkpoint': hashes['clip_checkpoint'], 'vadclip_checkpoint': hashes['vadclip_checkpoint'],
                         'script': sha256(Path(__file__)), 'runner': sha256(ROOT / 'vadfs/vadclip.py')},
        'vendored_source_sha256': full['vendored_source_sha256'], 'model_options': full['model_options'],
        'device': args.device, 'torch': torch.__version__, 'command': sys.argv,
        'gpu': torch.cuda.get_device_name() if args.device == 'cuda' else None,
        'full_graph_exact_identity': True, 'full_score_identity_max_abs_error': full_identity_error,
        'archived_score_max_abs_error': archived_max_error, 'archived_score_atol': ARCHIVED_SCORE_ATOL,
        'archived_ap_atol_pp': ARCHIVED_AP_ATOL_PP, 'metrics': metrics, 'score_replay': score_replay,
        f'{spec.primary_branch}_bootstrap': {'replicates': args.bootstrap_reps, 'seed': BOOTSTRAP_SEED,
                                            'unit': 'paired videos stratified by manifest class',
                                            'fixed_recall_target': True, 'percentile_ci95_pp': ci},
    }
    record_provenance(report)
    write_json(output / 'summary.json', report)
    for condition, row in primary['conditions'].items():
        print(f'{condition}: AP {row["ap_pct"]:.4f}% ({row["ap_delta_pp"]:+.4f} pp), '
              f'AUC {row["roc_auc_pct"]:.4f}% ({row["roc_auc_delta_pp"]:+.4f} pp), '
              f'FPR {row["fpr_at_matched_recall_pct"]:.4f}% ({row["fpr_delta_pp"]:+.4f} pp)')
    print(f'Summary: {output / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
