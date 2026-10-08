"""Search raw-video CLIP features for the nearest official feature rows.

A nearest neighbour is not a historical recipe confirmation. This diagnostic
tests for large time offsets/order differences using a fixed raw frame stride.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
import traceback

from verify_ucf_extraction import (ROOT, diagnostic_image, file_hash, index_features,
                                   project_path, save_csv, save_json, validate_device)


def nearest_metrics(target, prediction, np):
    """Use direct differences for distances and avoid platform BLAS warnings."""
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if (target.ndim != 2 or prediction.ndim != 2 or target.shape[1] != prediction.shape[1]
            or not np.isfinite(target).all() or not np.isfinite(prediction).all()):
        raise ValueError('Query and candidate vectors must be finite compatible matrices')
    tn = np.sqrt(np.einsum('qd,qd->q', target, target, optimize=False))
    pn = np.sqrt(np.einsum('nd,nd->n', prediction, prediction, optimize=False))
    if np.any(tn <= 0) or np.any(pn <= 0):
        raise ValueError('Zero query or candidate vectors')
    dot = np.einsum('qd,nd->qn', target, prediction, optimize=False)
    cosine = np.clip(dot / (tn[:, None] * pn[None, :]), -1, 1)
    distance = np.empty_like(dot)
    for q, query in enumerate(target):
        difference = prediction - query
        distance[q] = np.sqrt(np.einsum('nd,nd->n', difference, difference,
                                       optimize=False)) / tn[q]
    if not np.isfinite(cosine).all() or not np.isfinite(distance).all():
        raise ValueError('Non-finite nearest-neighbour metrics')
    return distance, cosine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True)
    parser.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='mps')
    parser.add_argument('--precision', choices=('float32', 'float16'), default='float16')
    parser.add_argument('--crop-type', type=int, choices=(0, 5), default=5)
    parser.add_argument('--preprocess-variant', choices=('published', 'shortside_opencv_linear',
                                                       'shortside_opencv_scale', 'clip_direct'),
                        default='shortside_opencv_scale')
    parser.add_argument('--stride', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--query-rows', type=int, nargs='+', default=[0, 1, 2, 3, 8, 16, 32, 64, 128])
    args = parser.parse_args()
    if args.stride < 1 or args.batch_size < 1 or min(args.query_rows) < 0:
        parser.error('Positive stride/batch and nonnegative rows required')
    output = ROOT / 'outputs/setup' / datetime.now().strftime('ucf_timeline_search_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    report = {'success': False, 'historical_recipe_confirmed': False, 'settings': vars(args),
              'output_dir': str(output), 'scope': 'Raw-video nearest-neighbour diagnostic, not VAD performance'}
    started = time.perf_counter()
    save_json(output / 'summary.json', report)
    print('Output:', output, flush=True)
    try:
        import numpy as np
        import torch
        import cv2
        from PIL import Image
        validate_device(torch, args.device, args.precision)
        manifest = json.loads((ROOT / 'UCF-Crime/test/manifest.json').read_text())
        video = next(v for v in manifest['videos'] if v['filename'] == args.video)
        config = json.loads((ROOT / 'configs/ucf_resources.json').read_text())
        video_path = (ROOT / 'UCF-Crime/test' / video['video_path']).resolve()
        checksum = file_hash(video_path)
        if checksum != video['sha256']:
            raise ValueError('Video differs from manifest')
        official_path = index_features(project_path(config['features_root']), [video], args.crop_type)[args.video]
        official = np.load(official_path, mmap_mode='r', allow_pickle=False)
        if official.ndim != 2 or official.shape[1] != 512 or official.dtype.kind != 'f':
            raise ValueError('Official features must be floating [T,512]')
        queries = sorted({r for r in args.query_rows if r < len(official)})
        if not queries:
            raise ValueError('No valid query rows')
        checkpoint = project_path(config['clip_checkpoint'])
        sys.path.insert(0, str(ROOT / 'VadCLIP/src'))
        import clip
        from clip.model import convert_weights
        from crop import image_crop
        model, preprocess = clip.load(str(checkpoint), device='cpu', jit=False)
        visual = model.visual.float().eval().requires_grad_(False)
        if args.precision == 'float16':
            convert_weights(visual)
        visual = visual.to(args.device)
        dtype = visual.conv1.weight.dtype
        chunks, batch, indices = [], [], []
        def flush():
            if not batch:
                return
            images = torch.stack(batch).to(device=args.device, dtype=dtype)
            batch.clear()
            with torch.no_grad():
                features = visual(images).float().cpu().numpy()
            if features.shape != (len(images), 512) or not np.isfinite(features).all():
                raise ValueError('Invalid CLIP features')
            chunks.append(features)
            if sum(len(c) for c in chunks) % 256 == 0:
                print('Encoded', sum(len(c) for c in chunks), 'images', flush=True)
        cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
        try:
            if not cap.isOpened():
                raise ValueError('Cannot open video')
            header_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            count = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                index = count
                count += 1
                if index % args.stride == 0:
                    indices.append(index)
                    image = diagnostic_image(frame, args.preprocess_variant, args.crop_type,
                                             image_crop, np, cv2, Image)
                    batch.append(preprocess(image))
                    if len(batch) >= args.batch_size:
                        flush()
            if count != header_count:
                raise ValueError(f'Decoded/header mismatch: {count}/{header_count}')
            flush()
        finally:
            cap.release()
        raw = np.concatenate(chunks)
        np.savez_compressed(output / 'raw_features.npz', features=raw,
                            frame_indices=np.asarray(indices), fps=np.asarray(fps))
        target = np.asarray(official[queries], dtype=np.float64)
        prediction = raw.astype(np.float64)
        distance, cosine = nearest_metrics(target, prediction, np)
        rows = []
        for q, official_row in enumerate(queries):
            best = np.argsort(distance[q])[:5]
            for rank, candidate in enumerate(best, 1):
                rows.append({'official_row': official_row, 'rank': rank,
                             'raw_frame': int(indices[candidate]), 'raw_time_seconds': indices[candidate] / fps,
                             'expected_frame_under_stride16': official_row * 16,
                             'frame_offset_from_expected': int(indices[candidate] - official_row * 16),
                             'relative_l2_error': float(distance[q, candidate]),
                             'cosine': float(cosine[q, candidate])})
            print('Official row', official_row, 'nearest:', rows[-5], flush=True)
        save_csv(output / 'nearest_rows.csv', rows)
        report.update(success=True, decoded_frames=count, header_frames=header_count, fps=fps,
                      encoder_images=len(raw), raw_feature_shape=list(raw.shape),
                      official_shape=list(official.shape), official_dtype=str(official.dtype),
                      queries=queries, nearest=rows, input_sha256={
                          'video': checksum, 'official': file_hash(official_path),
                          'clip_checkpoint': file_hash(checkpoint), 'script': file_hash(Path(__file__)),
                          'preprocess_helper': file_hash(ROOT / 'scripts/verify_ucf_extraction.py'),
                          'official_test_list': file_hash(ROOT / 'VadCLIP/list/ucf_CLIP_rgbtest.csv')},
                      environment={'torch': torch.__version__, 'opencv': cv2.__version__,
                                   'numpy': np.__version__, 'encoder_dtype': str(dtype),
                                   'preprocess': repr(preprocess)})
    except Exception as error:
        report['error'] = {'message': str(error), 'traceback': traceback.format_exc()}
        print(report['error']['traceback'], file=sys.stderr, flush=True)
    finally:
        report['elapsed_seconds'] = time.perf_counter() - started
        save_json(output / 'summary.json', report)
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
