"""Test distance graphs on the exact frames selected by an archived window sampler."""

import argparse
from copy import deepcopy
from pathlib import Path
import random
import sys

import numpy as np
import torch

from xd_distance_graph import (
    ARCHIVED_AP_ATOL_PP as METRIC_ATOL_PP, ARCHIVED_SCORE_ATOL, BLOCK, BRANCHES, SHARED,
    bootstrap, grouped_metrics, infer_with_graph, ranks, verified_path,
)
from vadfs import vadclip
from vadfs.config import read_json
from vadfs.datasets import ucf, xd
from vadfs.datasets.base import validate_manifest
from vadfs.extract import ReferenceImageEncoder, extract_reference, validate_extraction
from vadfs.paths import ROOT, sha256
from vadfs.runs import create_run_dir, now, record_provenance, write_json
from vadfs.timeline import restore_tokens
from clip import clip


CONDITIONS = ('token_index', 'original_distance', 'original_distance_matched_mass',
              'fixed_spacing_matched_mass')
CONTRASTS = ((1, 0), (2, 0), (2, 1), (3, 0), (2, 3))
WINDOW_METHODS = ('random_window', 'adaptive_window')
OPTIMAL_METHODS = ('d_optimal', 'a_optimal', 'e_optimal')


def archived_selection(folder, relative_score, grid, anchors, window_frames):
    score_file = (folder / relative_score).resolve()
    if not score_file.is_relative_to(folder / 'scores'):
        raise ValueError(f'Invalid score path: {score_file}')
    with np.load(score_file, allow_pickle=False) as data:
        positions = data['selected_frame_indices']
        saved_anchors = data['selected_score_anchor_indices']
        saved_grid = data['reference_frame_indices']
        direct = np.stack([data[f'{branch}_selected_scores'] for branch in BRANCHES])
    if (positions.dtype.kind not in 'iu' or positions.shape != anchors.shape
            or not np.array_equal(saved_grid, grid)
            or not np.array_equal(saved_anchors, anchors)
            or np.any(positions < anchors) or np.any(positions >= anchors + window_frames)
            or np.any(positions >= len(grid) * BLOCK)
            or direct.shape != (len(BRANCHES), len(positions)) or not np.isfinite(direct).all()):
        raise ValueError(f'Archived frame/anchor positions disagree with the window protocol: {score_file}')
    return positions, direct, sha256(score_file)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True, help='Completed raw window-sampler run')
    parser.add_argument('--summary', type=Path, help='Shared experiment summary for the same run')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs/analyses')
    parser.add_argument('--limit', type=int, help='Check the first N videos without reporting split-level metrics')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    archive = args.archive.resolve()
    archive_summary = args.summary.resolve() if args.summary else archive / 'summary.json'
    source = read_json(archive_summary)
    if args.summary and source['run_id'] != archive.name:
        raise ValueError('Shared summary and archive directory identify different runs')
    spec = {'ucf': ucf.SPEC, 'xd': xd.SPEC}[source['dataset_key']]
    sampling = source['sampling']
    budget = sampling['budget']
    is_ver2 = source.get('protocol_version') == 2
    allowed_methods = (*WINDOW_METHODS, *OPTIMAL_METHODS) if is_ver2 else WINDOW_METHODS
    stored_rows = source.get('videos')
    if (not source['success'] or source['mode'] != 'full' or source['input_type'] != 'raw'
            or sampling['method'] not in allowed_methods
            or sampling.get('score_anchor', source.get('score_anchor')) != 'window_start'
            or source['experiment']['protocol']['restoration'] != 'nearest'
            or source['prompts'] != list(spec.prompts) or source['metrics']['test_videos'] != spec.expected_videos
            or (stored_rows is None and not (args.summary and is_ver2))
            or (stored_rows is not None and len(stored_rows) != spec.expected_videos)):
        raise ValueError('Expected a complete raw window-sampler archive')
    if budget not in (1., .5, .25, .125):
        raise ValueError('Unexpected archived window size or budget')
    window_frames = int(BLOCK / budget)
    if sampling.get('window_frames', window_frames) != window_frames:
        raise ValueError('Archived window size differs from the reported budget')
    recipe = source['extraction_config']
    validate_extraction(recipe)
    if recipe['temporal_mode'] != 'single_frame' or recipe['frame_offset'] != 0 or recipe['block_frames'] != BLOCK:
        raise ValueError('Expected a zero-offset, single-frame extraction recipe')

    try:
        manifest_path = Path(source['setup_manifest'])
        if not manifest_path.is_file():
            manifest_path = SHARED / f'results/baselines/{spec.key}_raw_reference/manifest.json'
        hashes = source['input_sha256']
        manifest = read_json(verified_path(manifest_path, hashes['setup_manifest']))
        if manifest['dataset_identity_sha256'] != source['dataset_identity_sha256']:
            raise ValueError('Manifest has a different dataset identity')
        videos = validate_manifest(manifest, spec)
        for path, digest in source['vendored_source_sha256'].items():
            verified_path(ROOT / path, digest)
        verified_path(ROOT / 'VadCLIP/src/clip/bpe_simple_vocab_16e6.txt.gz', hashes['tokenizer_vocabulary'])
        dataset = read_json(ROOT / f'configs/datasets/{spec.key}.json')
        gt = np.load(verified_path(ROOT / dataset['frame_gt'], hashes['gt']), allow_pickle=False)
        clip_path = verified_path(SHARED / 'checkpoints/clip/ViT-B-16.pt', hashes['clip_checkpoint'])
        checkpoint = verified_path(SHARED / f'checkpoints/vadclip/model_{spec.key}.pth', hashes['vadclip_checkpoint'])
    except FileNotFoundError as error:
        raise SystemExit(f'Archived experiment input unavailable: {error.filename}') from error
    n = sum(v['reference_tokens'] for v in videos)
    if gt.shape != (n * BLOCK,) or not np.array_equal(np.unique(gt), [0, 1]):
        raise ValueError('Ground truth disagrees with the official timeline')
    with np.load(archive / 'frame_scores.npz', allow_pickle=False) as arrays:
        archived_frames = np.stack([arrays[b] for b in BRANCHES])
    if archived_frames.shape != (len(BRANCHES), len(gt)) or not np.isfinite(archived_frames).all():
        raise ValueError('Invalid archived frame scores')
    sparse_scores = archived_frames[:, ::BLOCK]
    if not np.array_equal(np.repeat(sparse_scores, BLOCK, axis=1), archived_frames):
        raise ValueError('Archived frame scores disagree with the reference block size')

    seed = source['model_options']['seed']
    options = vadclip.default_options(seed, dataset=spec.key)
    if {key: getattr(options, key) for key in vadclip.MODEL_OPTION_KEYS} != source['model_options']:
        raise ValueError('Model options differ from the archived experiment')
    vadclip.check_device(args.device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = vadclip.load_model(options, args.device, clip_path, checkpoint)
    visual = deepcopy(model.clipmodel.visual).eval().requires_grad_(False)
    if recipe['encoder_precision'] == 'float32':
        visual.float()
    encoder = ReferenceImageEncoder(visual)
    preprocess = clip._transform(visual.input_resolution)
    run_id, output = create_run_dir(f'{source["run_id"]}_distance_graph',
                                    'smoke' if args.limit is not None else 'full',
                                    output_root=args.output_root.resolve())
    print(f'Output: {output}', flush=True)

    restored = np.empty((len(CONDITIONS), len(BRANCHES), n), dtype=np.float32)
    errors = np.zeros(len(BRANCHES))
    selections = []
    offset = 0
    rows = list(zip(videos, stored_rows if stored_rows is not None else [None] * len(videos),
                    strict=True))[:args.limit]
    for i, (video, row) in enumerate(rows, start=1):
        count = video['reference_tokens']
        grid = np.arange(count, dtype=np.int64) * BLOCK
        anchors = grid[::window_frames // BLOCK]
        stem = f'{video["official_csv_index"]:03d}_{Path(video["filename"]).stem}'
        relative_score = f'scores/{stem}.npz'
        positions, archived, digest = archived_selection(archive, relative_score, grid, anchors, window_frames)
        stop = offset + count
        if offset * BLOCK != video['gt_offset'] or (row is not None and (
                row['video'] != video['filename'] or row['score_file'] != relative_score
                or len(positions) != row['selected_frames'] or video['sha256'] != row['input_sha256'])):
            raise ValueError(f'Archived video metadata mismatch: {video["filename"]}')
        raw = verified_path(Path(video['raw_path']), video['sha256'])
        dtype = np.dtype(video['feature_dtype'] if recipe['feature_storage_dtype'] == 'official' else 'float32')
        features, selection, metadata = extract_reference(
            raw, encoder, preprocess, args.device, recipe, dtype, expected_tokens=count,
            frame_selector=lambda frame_count: positions.tolist(),
        )
        if not np.array_equal(selection['selected_frame_indices'], positions) or (row is not None and any(
                metadata[k] != row[k] for k in ('decoded_frames', 'fps', 'encoder_batches', 'feature_dtype'))):
            raise ValueError(f'Cannot recreate archived feature input: {video["filename"]}')
        for k, condition in enumerate(CONDITIONS):
            # Regular anchor spacing provides the same narrower kernel, without
            # using the actual irregular gaps between frames in the input.
            actual = condition != 'fixed_spacing_matched_mass'
            scores = infer_with_graph(model, features, positions if actual else anchors,
                                      options.visual_length, source['prompts'], args.device,
                                      condition if actual else 'original_distance_matched_mass')
            if k == 0:
                error = np.max(np.abs(scores - archived), axis=1)
                errors = np.maximum(errors, error)
                if np.any(error > ARCHIVED_SCORE_ATOL):
                    raise ValueError(f'Archived stock scores differ for {video["filename"]}: {error.tolist()}')
            for j, values in enumerate(scores):
                restored[k, j, offset:stop] = restore_tokens(values, anchors, grid)
                if k == 0 and not np.array_equal(restore_tokens(archived[j], anchors, grid),
                                                 sparse_scores[j, offset:stop]):
                    raise ValueError(f'Archived score restoration mismatch: {video["filename"]}')
        selections.append({'video': video['filename'], 'raw_sha256': video['sha256'],
                           'archive_score_sha256': digest, 'selected_frames': len(positions)})
        offset = stop
        if i % 25 == 0 or i == len(rows):
            print(f'{i}/{len(rows)} videos checked', flush=True)

    positive = gt.reshape(n, BLOCK).sum(axis=1)
    video_index = np.repeat(np.arange(len(videos)), [v['reference_tokens'] for v in videos])
    report = {'run_id': run_id, 'finished_at': now().isoformat(), 'success': True,
              'complete_split': args.limit is None, 'dataset': spec.key, 'sampling': sampling,
              'archive': str(archive), 'archive_summary': str(archive_summary), 'extraction_config': recipe, 'device': args.device,
              'model_options': source['model_options'], 'gpu': torch.cuda.get_device_name() if args.device == 'cuda' else None,
              'command': sys.argv, 'scope': 'same original selected frames; original window-start restoration',
              'conditions': CONDITIONS, 'videos': selections, 'primary': {'branch': spec.primary_branch, 'metric': spec.primary_metric},
              'archived_stock_score_max_abs_error': errors.tolist(), 'archived_stock_score_atol': ARCHIVED_SCORE_ATOL,
              'input_sha256': {**hashes, 'archive_summary': sha256(archive_summary),
                               'archive_frame_scores': sha256(archive / 'frame_scores.npz'), 'script': sha256(Path(__file__)),
                               **{p: sha256(ROOT / p) for p in ('vadfs/extract.py', 'vadfs/vadclip.py',
                                                              'scripts/analyze/xd_distance_graph.py')}}}
    if args.limit is None:
        j = BRANCHES.index(spec.primary_branch)
        # Match the archived sampler's recall at its existing 0.5 threshold.
        recall = float(np.mean(archived_frames[j, gt == 1] >= .5))
        values = np.array([grouped_metrics(*ranks(scores), positive, BLOCK - positive, recall)
                           for scores in restored[:, j]])
        archived_metrics = source['metrics'][spec.primary_branch]
        for column, metric in ((0, 'ap'), (2, 'roc_auc')):
            if abs(values[0, column] - 100 * archived_metrics[metric]) > METRIC_ATOL_PP:
                raise ValueError(f'Stock replay {metric} differs from the archive')
        names = ('ap_pct', 'fpr_at_matched_recall_pct', 'roc_auc_pct')
        report['metrics'] = {name: dict(zip(names, row.tolist(), strict=True))
                             for name, row in zip(CONDITIONS, values, strict=True)}
        labels = np.array([v['is_anomaly'] for v in videos])
        print('Computing 1,000 stratified paired video bootstrap replicates', flush=True)
        report['paired_bootstrap'] = {'replicates': 1000, 'recall_target_pct': 100 * recall,
                                      'percentile_ci95_pp': bootstrap(
                                          restored[:, j], positive, video_index,
                                          (np.flatnonzero(~labels), np.flatnonzero(labels)),
                                          recall, 1000, CONDITIONS, CONTRASTS)}
    np.savez_compressed(output / 'restored_scores.npz', conditions=CONDITIONS, branches=BRANCHES,
                        tokens=restored[:, :, :offset], video_index=video_index[:offset],
                        positive_frames_per_token=positive[:offset])
    record_provenance(report)
    write_json(output / 'summary.json', report)
    if args.limit is None:
        key = spec.primary_metric + '_pct'
        for name in CONDITIONS:
            print(f'{name}: {report["metrics"][name][key]:.4f}%')
    print(f'Summary: {output / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
