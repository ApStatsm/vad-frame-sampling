"""Isolate the effect of padded tokens in the XD VadCLIP Branch 2 prompt pool."""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

from xd_distance_graph import (
    ARCHIVED_SCORE_ATOL, BLOCK, BRANCHES, SHARED, grouped_metrics, load_inputs, ranks,
    saved_scores, verified_path,
)
from vadfs import vadclip
from vadfs.datasets.xd import SPEC
from vadfs.paths import ROOT, sha256
from vadfs.pipeline import load_cached_reference
from vadfs.runs import create_run_dir, now, record_provenance, write_json
from vadfs.timeline import restore_tokens


STRIDES = (1, 2, 4, 8)
SEED = 234
REPLICATES = 1000


def infer_without_padded_pool(model, features, window, prompts, device):
    def zero_padded_logits(module, args, logits):
        starts = torch.arange(logits.shape[0], device=logits.device)[:, None] * window
        positions = starts + torch.arange(window, device=logits.device)
        partial_window_padding = (starts < len(features)) & (positions >= len(features))
        return logits.masked_fill(partial_window_padding.unsqueeze(-1), 0)

    # The classifier's raw logits weight the visual features pooled for the text prompt.
    # Upstream discards padded classifier scores after forward; valid logits stay identical.
    handle = model.classifier.register_forward_hook(zero_padded_logits)
    try:
        return vadclip.infer_feature(model, features, window, prompts, device)[:2]
    finally:
        handle.remove()


def bootstrap(ranked, positive, video_index, strata, recall_target):
    rng = np.random.default_rng(SEED)
    deltas = np.empty((REPLICATES, len(STRIDES), 2, 3))
    for rep in range(REPLICATES):
        sampled = np.concatenate([rng.choice(group, len(group), replace=True) for group in strata])
        weights = np.bincount(sampled, minlength=sum(map(len, strata)))[video_index]
        tp, fp = positive * weights, (BLOCK - positive) * weights
        values = np.array([[grouped_metrics(*group, tp, fp, recall_target) for group in pair]
                           for pair in ranked])
        deltas[rep, :, 0] = values[:, 1] - values[:, 0]
        deltas[rep, :, 1] = deltas[rep, :, 0] - deltas[rep, 0, 0]
    return np.percentile(deltas, (2.5, 97.5), axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs/analyses')
    args = parser.parse_args()
    args.full_run = (SHARED / 'results/baselines/xd_raw_reference').resolve()
    uniform_dirs = [(SHARED / f'results/uniform/xd_raw_uniform_{1 / stride:g}').resolve()
                    for stride in STRIDES[1:]]
    archives = []
    for stride, uniform_dir in zip(STRIDES[1:], uniform_dirs, strict=True):
        args.uniform_run = uniform_dir
        full, uniform, videos, gt = load_inputs(args, SPEC, stride)
        archives.append(uniform)
    paths = (args.full_run, *uniform_dirs)
    rows = (full['videos'], *(archive['videos'] for archive in archives))
    hashes = full['input_sha256']
    clip = verified_path(SHARED / 'checkpoints/clip/ViT-B-16.pt', hashes['clip_checkpoint'])
    checkpoint = verified_path(SHARED / 'checkpoints/vadclip/model_xd.pth', hashes['vadclip_checkpoint'])
    options = vadclip.default_options(seed=full['model_options']['seed'], dataset=SPEC.key)
    if {key: getattr(options, key) for key in vadclip.MODEL_OPTION_KEYS} != full['model_options']:
        raise ValueError('VadCLIP options differ from the archives')
    random.seed(options.seed)
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)
    vadclip.check_device(args.device)
    model = vadclip.load_model(options, args.device, clip, checkpoint)
    prompts, window = full['prompts'], options.visual_length
    run_id, output = create_run_dir('xd_valid_prompt_pool', 'full', output_root=args.output_root.resolve())
    print(f'Output: {output}', flush=True)

    n = sum(video['reference_tokens'] for video in videos)
    if n * BLOCK != len(gt):
        raise ValueError('Manifest and GT timelines differ')
    positive = gt.reshape(n, BLOCK).sum(axis=1)
    scores = np.empty((len(STRIDES), 2, n), dtype=np.float32)
    count_selected = np.zeros(len(STRIDES), dtype=np.int64)
    max_branch1_error = np.zeros(len(STRIDES))
    max_unpadded_branch2_error = np.zeros(len(STRIDES))
    offset = 0
    with np.load(args.full_run / 'frame_scores.npz', allow_pickle=False) as full_npz:
        full_scores = full_npz['branch2']
    recall_target = float(np.mean(full_scores[gt == 1] >= 0.5))
    for i, video in enumerate(videos):
        stop = offset + video['reference_tokens']
        if offset * BLOCK != video['gt_offset']:
            raise ValueError(f'Unexpected GT offset: {video["filename"]}')
        grid = np.arange(stop - offset, dtype=np.int64) * BLOCK
        feature_record = rows[1][i]
        feature_hash = feature_record['input_sha256']
        if any(budget_rows[i]['input_sha256'] != feature_hash for budget_rows in rows[2:]):
            raise ValueError(f'Uniform feature hashes differ: {video["filename"]}')
        feature_file = args.full_run / 'features' / Path(rows[0][i]['score_file']).name
        features, _ = load_cached_reference(verified_path(feature_file, feature_hash), grid,
                                            np.dtype(video['feature_dtype']), expected_fps=rows[0][i]['fps'])
        for b, (stride, path, budget_rows) in enumerate(zip(STRIDES, paths, rows, strict=True)):
            indices = grid[::stride]
            direct, _ = saved_scores(path, budget_rows[i], video, grid, indices)
            branch1, branch2 = infer_without_padded_pool(model, features[::stride], window, prompts, args.device)
            error = float(np.max(np.abs(branch1 - direct[0])))
            max_branch1_error[b] = max(error, max_branch1_error[b])
            if error > ARCHIVED_SCORE_ATOL:
                raise ValueError(f'Valid Branch 1 scores changed: {video["filename"]} ({error:g})')
            unpadded = len(indices) // window * window
            if unpadded:
                error = float(np.max(np.abs(branch2[:unpadded] - direct[1, :unpadded])))
                max_unpadded_branch2_error[b] = max(error, max_unpadded_branch2_error[b])
                if error > ARCHIVED_SCORE_ATOL:
                    raise ValueError(f'An unpadded window changed: {video["filename"]} ({error:g})')
            scores[b, 0, offset:stop] = restore_tokens(direct[1], indices, grid)
            scores[b, 1, offset:stop] = restore_tokens(branch2, indices, grid)
            count_selected[b] += len(indices)
        offset = stop
        if (i + 1) % 50 == 0:
            print(f'{i + 1}/{len(videos)} videos checked', flush=True)
    if not np.array_equal(scores[0, 0], full_scores[::BLOCK]):
        raise ValueError('Full archived direct and restored scores disagree')

    ranked = [[ranks(values) for values in pair] for pair in scores]
    metrics = np.array([[grouped_metrics(*group, positive, BLOCK - positive, recall_target) for group in pair]
                        for pair in ranked])
    for observed, archive in zip(metrics[:, 0, 0], (full, *archives), strict=True):
        if abs(observed - 100 * archive['metrics']['branch2']['ap']) > 1e-8:
            raise ValueError('Baseline ranking differs from the archive')
    video_index = np.repeat(np.arange(len(videos)), [video['reference_tokens'] for video in videos])
    classes = np.array([video['is_anomaly'] for video in videos])
    print(f'Computing {REPLICATES} paired video bootstrap replicates', flush=True)
    interval = bootstrap(ranked, positive, video_index,
                         (np.flatnonzero(~classes), np.flatnonzero(classes)), recall_target)
    names = ('ap_pct', 'fpr_at_matched_recall_pct', 'roc_auc_pct')
    rows = {}
    full_delta = metrics[0, 1] - metrics[0, 0]
    for b, stride in enumerate(STRIDES):
        delta = metrics[b, 1] - metrics[b, 0]
        rows[f'{1 / stride:g}'] = {
            'selected_frames': int(count_selected[b]),
            'stock': dict(zip(names, metrics[b, 0].tolist(), strict=True)),
            'valid_pool': dict(zip(names, metrics[b, 1].tolist(), strict=True)),
            'delta_pp': dict(zip(names, delta.tolist(), strict=True)),
            'delta_ci95_pp': dict(zip(names, interval[:, b, 0].T.tolist(), strict=True)),
            'gap_difference_vs_full_pp': dict(zip(names, (delta - full_delta).tolist(), strict=True)),
            'gap_difference_ci95_pp': dict(zip(names, interval[:, b, 1].T.tolist(), strict=True)),
            'max_archived_branch1_score_error': float(max_branch1_error[b]),
            'max_unpadded_branch2_score_error': float(max_unpadded_branch2_error[b]),
        }
        print(f'{100 / stride:g}% AP {metrics[b, 0, 0]:.4f}% -> {metrics[b, 1, 0]:.4f}%', flush=True)

    np.savez_compressed(output / 'restored_scores.npz', scores=scores, budgets=1 / np.asarray(STRIDES),
                        conditions=('stock', 'valid_pool'), video_index=video_index,
                        positive_frames_per_token=positive)
    report = {
        'run_id': run_id, 'success': True, 'finished_at': now().isoformat(), 'scope': 'cached-feature model diagnostic',
        'videos': len(videos), 'branch': BRANCHES[1], 'matched_recall_pct': 100 * recall_target, 'budgets': rows,
        'bootstrap': {'replicates': REPLICATES, 'seed': SEED, 'unit': 'paired videos stratified by manifest class'},
        'change': 'Only padded logits weighting the Branch 2 text prompt pool are zeroed',
        'sources': {str(path): sha256(path / 'summary.json') for path in paths},
        'input_sha256': {'script': sha256(Path(__file__)),
                         'archive_helpers': sha256(Path(__file__).with_name('xd_distance_graph.py')),
                         'manifest': hashes['setup_manifest'], 'ground_truth': hashes['gt'],
                         'clip_checkpoint': hashes['clip_checkpoint'], 'vadclip_checkpoint': hashes['vadclip_checkpoint']},
        'vendored_source_sha256': full['vendored_source_sha256'], 'model_options': full['model_options'],
        'device': args.device, 'torch': torch.__version__, 'command': sys.argv,
        'gpu': torch.cuda.get_device_name() if args.device == 'cuda' else None,
    }
    record_provenance(report)
    write_json(output / 'summary.json', report)
    print(f'Summary: {output / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
