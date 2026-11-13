"""User-run development-only cheap-feature PCA fit; never fit on test videos."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('ucf', 'xd'), required=True)
    parser.add_argument('--development-list', required=True, type=Path,
                        help='JSON {dataset_key, split, videos: [{raw_path}]}')
    parser.add_argument('--test-manifest', required=True, type=Path,
                        help='Prepared full test manifest, used only for overlap rejection')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--rows-per-video', type=int, default=1024)
    parser.add_argument('--max-fit-rows', type=int, default=100000)
    args = parser.parse_args()
    import cv2
    import numpy as np
    from vadfs.config import read_json
    from vadfs.datasets import get_dataset
    from vadfs.paths import ROOT, package_hashes, sha256
    from vadfs.preparation import validate_manifest
    from vadfs.sampling_features import FEATURE_SPEC, collect, fit_transform
    from vadfs.timeline import candidates

    if args.rows_per_video < 9 or args.max_fit_rows < 9:
        parser.error('Row limits must be at least 9')
    if args.output.exists():
        raise FileExistsError('Choose a new artifact path; artifacts are immutable')
    listing = read_json(args.development_list)
    if listing.get('dataset_key') != args.dataset or listing.get('split') not in ('train', 'development', 'validation'):
        raise ValueError('Development list needs matching dataset_key and a non-test split')
    test = read_json(args.test_manifest)
    validate_manifest(test, get_dataset(args.dataset).SPEC)
    recipe = read_json(ROOT / 'configs/datasets' / f'{args.dataset}.json')['extraction']
    test_names = {v['filename'] for v in test['videos']}
    test_hashes = {v['sha256'] for v in test['videos']}
    records, seen_names, seen_hashes = [], set(), set()
    # Validate ALL identities before decoding ANY development video.
    for entry in listing['videos']:
        path = Path(entry['raw_path']).expanduser()
        if not path.is_absolute():
            path = args.development_list.resolve().parent / path
        path = path.resolve()
        digest = sha256(path)
        if path.name in test_names or digest in test_hashes or path.name in seen_names or digest in seen_hashes:
            raise ValueError(f'Duplicate or test-overlapping development input: {path}')
        seen_names.add(path.name)
        seen_hashes.add(digest)
        records.append({'raw_path': str(path), 'filename': path.name, 'sha256': digest})
    rows, total = [], 0
    for record in records:
        path = Path(record['raw_path'])
        cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
        try:
            if not cap.isOpened():
                raise ValueError(f'Cannot open {path}')
            header = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not np.isfinite(header) or header < 16 or abs(header - round(header)) > 1e-3:
                raise ValueError('Development video needs at least one full block and integral frame count')
            tokens = int(round(header)) // 16
        finally:
            cap.release()
        # Development has no official rows; use complete reference blocks only.
        cheap, detail = collect(path, recipe, candidates(tokens, frame_offset=recipe['frame_offset']))
        keep = min(len(cheap), args.rows_per_video)
        indices = np.arange(keep, dtype=np.int64) * len(cheap) // keep
        rows.append(cheap[indices])
        total += keep
        if total > args.max_fit_rows:
            raise ValueError('Fit row cap exceeded; declare a smaller development list or larger memory cap')
        record.update(reference_tokens=tokens, fit_row_positions=indices.tolist(),
                      decoded_frames=detail['decoded_frames'])
        print(f'Development {path.name}: {keep}/{tokens} rows', flush=True)
    arrays = fit_transform(np.concatenate(rows) if rows else np.empty((0, 80)))
    metadata = {'schema_version': 1, 'dataset_key': args.dataset, 'split': listing['split'],
                'feature_spec': FEATURE_SPEC, 'extraction_config': recipe, 'training_videos': records,
                'development_list_sha256': sha256(args.development_list),
                'test_manifest_sha256': sha256(args.test_manifest),
                'feature_source_sha256': package_hashes()['vadfs/sampling_features.py'],
                'color_source_sha256': package_hashes()['vadfs/color.py'],
                'source_sha256': package_hashes(), 'dtype': 'float64', 'constant_scale_tolerance': 1e-12,
                'fit_rows': total, 'rows_per_video': args.rows_per_video,
                'numpy': np.__version__, 'opencv': cv2.__version__,
                'note': 'Training IDs/hashes verified disjoint from the supplied full test manifest'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as handle:
        np.savez_compressed(handle, **arrays, metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
    print(f'Artifact: {args.output.resolve()} SHA256 {sha256(args.output)}')


if __name__ == '__main__':
    main()
