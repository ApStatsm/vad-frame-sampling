"""Compare first-frame CLIP features from selected raw videos to official test rows."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from vadfs.config import read_json, resource_path
from vadfs.extract import ReferenceImageEncoder, compare_features, extract_reference, validate_extraction
from vadfs.paths import ROOT, sha256
from vadfs.preparation import index_files, load_prepared, unique_path
from vadfs.runs import create_run_dir, now, record_provenance, write_json
from vadfs.vadclip import check_device
from clip import clip


def compare_video(video, raw_index, encoder, preprocess, device, recipe):
    raw = unique_path(raw_index, video['filename'])
    official_path = Path(video['feature_path'])
    official_sha = sha256(official_path)
    if official_sha != video['official_feature_sha256']:
        raise ValueError('Official feature changed since input preparation')
    raw_sha = sha256(raw)
    if video.get('sha256') and raw_sha != video['sha256']:
        raise ValueError('Raw video SHA-256 differs from the preserved UCF reference')
    official = np.load(official_path, allow_pickle=False)
    features, selection, metadata = extract_reference(
        raw, encoder, preprocess, device, recipe, official.dtype, expected_tokens=len(official),
    )
    indices = selection['selected_frame_indices']
    if not np.array_equal(indices, np.arange(len(official)) * 16):
        raise ValueError('Selected frames differ from the expected 0, 16, 32, … sequence')
    return {
        'raw_video': str(raw), 'raw_video_sha256': raw_sha,
        'official_feature': str(official_path), 'official_feature_sha256': official_sha,
        'official_shape': list(official.shape), 'extraction': metadata,
        'selected_frame_indices': indices.tolist(),
        'selected_timestamps_seconds': selection['selected_timestamps_seconds'].tolist(),
        'comparison': compare_features(features, official),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('ucf', 'xd'), required=True)
    parser.add_argument('--videos-root', type=Path, required=True)
    parser.add_argument('--video', action='append', required=True, help='Exact test video filename; repeatable')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--xd-geometry', choices=('shortside_fxfy', 'published_340x256'),
                        help='XD resize geometry; default is shortside_fxfy')
    args = parser.parse_args()
    if args.xd_geometry and args.dataset != 'xd':
        parser.error('--xd-geometry applies only to XD')

    config_path = ROOT / 'configs/datasets' / f'{args.dataset}.json'
    config = read_json(config_path)
    resources = config['resources']
    manifest_path, manifest = load_prepared(args.dataset, resources, config)
    videos = {video['filename']: video for video in manifest['videos']}
    if len(set(args.video)) != len(args.video) or set(args.video) - videos.keys():
        parser.error('--video must name distinct videos in the official test list')
    raw_index = index_files(args.videos_root, '.mp4', lambda path: path.name)
    recipe = dict(read_json(ROOT / 'configs/datasets/ucf.json')['extraction'])
    if args.dataset == 'xd':
        recipe.update(crop_type=0, color_conversion='opencv_bgr',
                      image_geometry=args.xd_geometry or 'shortside_fxfy',
                      extraction_status='provisional_not_verified_against_official_recipe')
    validate_extraction(recipe)
    check_device(args.device)
    checkpoint = resource_path(resources, 'clip_checkpoint')
    run_id, output = create_run_dir(f'{args.dataset}_raw_feature_comparison', 'sample',
                                    output_root=resource_path(resources, 'output_root'))
    summary_path = output / 'summary.json'
    report = {
        'run_id': run_id, 'started_at': now().isoformat(), 'dataset': args.dataset,
        'command': sys.argv, 'recipe': recipe, 'scope': 'selected test videos; feature alignment only',
        'device': args.device, 'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
        'input_sha256': {'script': sha256(Path(__file__)), 'dataset_config': sha256(config_path),
                         'manifest': sha256(manifest_path), 'clip_checkpoint': sha256(checkpoint),
                         'extract': sha256(ROOT / 'vadfs/extract.py'), 'color': sha256(ROOT / 'vadfs/color.py')},
        'videos': [], 'success': False,
    }
    if args.device == 'cuda':
        gpu = torch.cuda.get_device_properties(torch.cuda.current_device())
        report['gpu'] = {'name': gpu.name, 'uuid': str(gpu.uuid)}
    record_provenance(report)
    write_json(summary_path, report)
    print(f'Report: {summary_path}', flush=True)

    model, preprocess = clip.load(str(checkpoint), device='cpu', jit=False)
    visual = model.visual.to(device=args.device, dtype=torch.float32).eval().requires_grad_(False)
    del model
    encoder = ReferenceImageEncoder(visual)
    for name in args.video:
        row = {'filename': name, 'official_tokens': videos[name]['reference_tokens']}
        try:
            row.update(compare_video(videos[name], raw_index, encoder, preprocess, args.device, recipe))
        except (ValueError, OSError, RuntimeError) as error:
            row['error'] = str(error)
            print(f'{name}: {error}', flush=True)
        else:
            comparison = row['comparison']
            print(f'{name}: {row["official_tokens"]} rows, '
                  f'relative L2 {comparison["relative_l2_error"]:.6f}, '
                  f'mean cosine {comparison["cosine_mean"]:.6f}', flush=True)
        report['videos'].append(row)
        write_json(summary_path, report)
    report.update(finished_at=now().isoformat(), success=all('error' not in row for row in report['videos']))
    write_json(summary_path, report)
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
