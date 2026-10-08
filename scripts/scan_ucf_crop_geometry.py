"""Narrow geometry diagnostic at a fixed frame offset, without VAD inference.

Use this to identify input geometry, then verify that condition with all 16 frame
positions, independent videos and complete video/tail handling. Rank is evidence,
not an automatic historical recipe confirmation.
"""

import argparse
from datetime import datetime
import itertools
import json
from pathlib import Path
import sys
import time
import traceback

from verify_ucf_extraction import (ROOT, choose_videos, file_hash, index_features,
                                   project_path, rank_candidates, save_csv, save_json,
                                   validate_device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='mps')
    parser.add_argument('--precision', choices=('float32', 'float16'), default='float16')
    parser.add_argument('--resize-backend', choices=('opencv', 'av_rgb_opencv', 'av_bilinear', 'av_bicubic', 'av_fast_bilinear'),
                        default='opencv', help='Native YUV->RGB resize vs resize of decoded RGB/BGR pixels')
    parser.add_argument('--resize-scale', action='store_true', help='Use OpenCV fx/fy instead of explicit dsize')
    parser.add_argument('--source-colorspace', choices=('bt601', 'bt709', 'fcc'),
                        help='Explicit YUV source matrix for PyAV conversion; default: stream/decoder default')
    parser.add_argument('--no-flip', action='store_true', help='Diagnostic only: keep __5 counterpart but omit flip')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/ucf_resources.json')
    parser.add_argument('--video', action='append')
    parser.add_argument('--blocks', type=int, nargs='+', default=[1, 8, 32])
    parser.add_argument('--frame-offset', type=int, choices=range(16), default=0)
    parser.add_argument('--widths', type=int, nargs='+', default=[340, 341, 342, 343])
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--crop-xs', type=int, nargs='+', default=[57, 58, 59, 60])
    parser.add_argument('--crop-ys', type=int, nargs='+', default=[15, 16, 17])
    parser.add_argument('--crop-type', type=int, choices=(0, 5), default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--atol', type=float, default=1e-4)
    parser.add_argument('--rtol', type=float, default=1e-3)
    args = parser.parse_args()
    if args.source_colorspace and args.resize_backend == 'opencv':
        parser.error('Source colorspace is available for PyAV decoding only')
    if args.resize_scale and args.resize_backend not in ('opencv', 'av_rgb_opencv'):
        parser.error('resize-scale applies to decoded-pixel OpenCV resize only')
    if args.batch_size < 1 or args.height < 224 or min(args.blocks) < 0:
        parser.error('Invalid batch size, height or blocks')
    if args.atol < 0 or args.rtol < 0:
        parser.error('Tolerances must be non-negative')
    geometry = sorted(set(itertools.product(args.widths, args.crop_xs, args.crop_ys)))
    if any(w < 224 or x < 0 or x + 224 > w or y < 0 or y + 224 > args.height for w, x, y in geometry):
        parser.error('Every 224x224 crop must fit within every resize geometry')
    args.blocks = sorted(set(args.blocks))
    stem = datetime.now().astimezone().strftime('ucf_geometry_scan_%Y%m%d_%H%M%S')
    output = ROOT / 'outputs/setup' / stem
    attempt = 0
    while True:
        try:
            output.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            attempt += 1
            output = ROOT / 'outputs/setup' / f'{stem}_{attempt}'
    (output / 'arrays').mkdir()
    report = {'success': False, 'historical_recipe_confirmed': False,
              'purpose': 'Fixed frame-offset geometry diagnosis; mean pooling/tail not checked',
              'settings': {**vars(args), 'config': str(project_path(args.config))},
              'geometry': [{'width': w, 'height': args.height, 'crop_x': x, 'crop_y': y,
                            'interpolation': args.resize_backend, 'crop_size': 224,
                            'resize_by_scale': args.resize_scale, 'source_colorspace': args.source_colorspace,
                            'horizontal_flip': args.crop_type == 5 and not args.no_flip} for w, x, y in geometry],
              'videos': [], 'output_dir': str(output)}
    save_json(output / 'summary.json', report)
    print(f'Output: {output}', flush=True)
    started = time.perf_counter()
    try:
        import numpy as np
        import torch
        import cv2
        from PIL import Image
        if args.resize_backend != 'opencv':
            import av
        validate_device(torch, args.device, args.precision)
        config = json.loads(project_path(args.config).read_text())
        manifest_path = ROOT / 'UCF-Crime/test/manifest.json'
        selected = choose_videos(json.loads(manifest_path.read_text()), args.video)
        paths = index_features(project_path(config['features_root']), selected, args.crop_type)
        checkpoint = project_path(config['clip_checkpoint'])
        sys.path.insert(0, str(ROOT / 'VadCLIP/src'))
        import clip
        from clip.model import convert_weights
        model, preprocess = clip.load(str(checkpoint), device='cpu', jit=False)
        visual = model.visual.float().eval().requires_grad_(False)
        if args.precision == 'float16':
            convert_weights(visual)
        visual = visual.to(args.device)
        dtype = visual.conv1.weight.dtype
        report['input_sha256'] = {'script': file_hash(Path(__file__)),
                                 'verifier_source': file_hash(ROOT / 'scripts/verify_ucf_extraction.py'),
                                 'clip_checkpoint': file_hash(checkpoint),
                                 'manifest': file_hash(manifest_path),
                                 'official_test_list': file_hash(ROOT / 'VadCLIP/list/ucf_CLIP_rgbtest.csv')}
        report['environment'] = {'torch': torch.__version__, 'opencv': cv2.__version__,
                                 'numpy': np.__version__, 'dtype': str(dtype),
                                 'preprocess': repr(preprocess)}
        all_rows = []
        for video in selected:
            filename = video['filename']
            path = (ROOT / 'UCF-Crime/test' / video['video_path']).resolve()
            checksum = file_hash(path)
            if checksum != video['sha256']:
                raise ValueError(f'{filename}: video differs from manifest')
            official = np.load(paths[filename], mmap_mode='r', allow_pickle=False)
            if official.ndim != 2 or official.shape[1] != 512 or official.dtype.kind != 'f':
                raise ValueError('Official features must be floating [T,512]')
            blocks = [b for b in args.blocks if b < len(official)]
            if not blocks:
                raise ValueError(f'{filename}: no usable blocks')
            target = np.asarray(official[blocks], dtype=np.float64)
            target_norm = np.linalg.norm(target, axis=1)
            if not np.isfinite(target).all() or np.any(target_norm <= 0):
                raise ValueError('Invalid official vectors')
            wanted = {16 * b + args.frame_offset for b in blocks}
            cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG) if args.resize_backend == 'opencv' else None
            container = av.open(str(path)) if args.resize_backend != 'opencv' else None
            iterator = container.decode(video=0) if container is not None else None
            tensors, chunks, indices = [], [], []
            def flush():
                if not tensors:
                    return
                images = torch.stack(tensors).to(device=args.device, dtype=dtype)
                tensors.clear()
                with torch.no_grad():
                    encoded = visual(images).float().cpu().numpy()
                if encoded.shape != (len(images), 512) or not np.isfinite(encoded).all():
                    raise ValueError('Invalid encoded features')
                chunks.append(encoded)
            try:
                if cap is not None and not cap.isOpened():
                    raise ValueError(f'Cannot decode {path}')
                decoded = 0
                while decoded <= max(wanted):
                    if cap is not None:
                        ok, frame = cap.read()
                        if not ok:
                            raise ValueError(f'Video ended before requested frame {max(wanted)}')
                    else:
                        try:
                            frame = next(iterator)
                        except StopIteration as error:
                            raise ValueError(f'Video ended before requested frame {max(wanted)}') from error
                    index = decoded
                    decoded += 1
                    if index not in wanted:
                        continue
                    indices.append(index)
                    for width, x, y in geometry:
                        if args.resize_backend in ('opencv', 'av_rgb_opencv'):
                            source = frame
                            if args.resize_backend == 'av_rgb_opencv':
                                options = {}
                                if args.source_colorspace:
                                    matrix = {'bt601': 'ITU601', 'bt709': 'ITU709', 'fcc': 'FCC'}[args.source_colorspace]
                                    options['src_colorspace'] = getattr(av.video.reformatter.Colorspace, matrix)
                                source = frame.reformat(format='rgb24', **options).to_ndarray()
                            if args.resize_scale:
                                scale = args.height / source.shape[0]
                                resized = cv2.resize(source, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                                if resized.shape[:2] != (args.height, width):
                                    raise ValueError('Requested width does not match scale-derived output geometry')
                            else:
                                resized = cv2.resize(source, (width, args.height), interpolation=cv2.INTER_LINEAR)
                            cropped = resized[y:y+224, x:x+224]
                            if args.resize_backend == 'opencv':
                                cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                        else:
                            method = {'av_bilinear': 'BILINEAR', 'av_bicubic': 'BICUBIC',
                                      'av_fast_bilinear': 'FAST_BILINEAR'}[args.resize_backend]
                            interpolation = getattr(av.video.reformatter.Interpolation, method)
                            options = {}
                            if args.source_colorspace:
                                matrix = {'bt601': 'ITU601', 'bt709': 'ITU709', 'fcc': 'FCC'}[args.source_colorspace]
                                options['src_colorspace'] = getattr(av.video.reformatter.Colorspace, matrix)
                            resized = frame.reformat(width=width, height=args.height,
                                                     format='rgb24', interpolation=interpolation, **options).to_ndarray()
                            cropped = resized[y:y+224, x:x+224]
                        if args.crop_type == 5 and not args.no_flip:
                            cropped = cv2.flip(cropped, 1)
                        tensors.append(preprocess(Image.fromarray(cropped)))
                        if len(tensors) >= args.batch_size:
                            flush()
                flush()
            finally:
                if cap is not None:
                    cap.release()
                if container is not None:
                    container.close()
            features = np.concatenate(chunks).reshape(len(blocks), len(geometry), 512)
            rows = []
            for g, (width, x, y) in enumerate(geometry):
                name = f'{args.resize_backend}_w{width}_h{args.height}_x{x}_y{y}'
                if args.no_flip:
                    name += '_no_flip'
                prediction = features[:, g, :].astype(np.float64)
                norm = np.linalg.norm(prediction, axis=1)
                delta = prediction - target
                absolute = np.abs(delta)
                close = absolute <= args.atol + args.rtol * np.abs(target)
                cosine = np.clip(np.sum(prediction * target, axis=1) / (norm * target_norm), -1, 1)
                for i, block in enumerate(blocks):
                    rows.append({'video': filename, 'block_index': block, 'frame_index': indices[i],
                                 'candidate': f'single_offset_{args.frame_offset:02d}', 'feature_form': 'raw',
                                 'row_shift': 0, 'preprocess_variant': name,
                                 'cosine': float(cosine[i]), 'mean_absolute_error': float(absolute[i].mean()),
                                 'rmse': float(np.sqrt(np.mean(delta[i] ** 2))),
                                 'max_absolute_error': float(absolute[i].max()),
                                 'relative_l2_error': float(np.linalg.norm(delta[i]) / target_norm[i]),
                                 'candidate_norm': float(norm[i]), 'official_norm': float(target_norm[i]),
                                 'exact_element_fraction': float(np.mean(prediction[i] == target[i])),
                                 'close_element_fraction': float(close[i].mean()),
                                 'tolerance_pass': bool(close[i].all())})
            ranking = rank_candidates(rows)
            all_rows.extend(rows)
            np.savez_compressed(output / 'arrays' / f'{Path(filename).stem}.npz',
                                features=features, official_features=official[blocks],
                                block_indices=np.asarray(blocks), frame_indices=np.asarray(indices),
                                geometry=np.asarray(geometry))
            report['videos'].append({'video': filename, 'video_sha256': checksum,
                                     'official_path': str(paths[filename]),
                                     'official_sha256': file_hash(paths[filename]), 'blocks': blocks,
                                     'decoder': args.resize_backend, 'decoded_prefix_frames': decoded,
                                     'encoder_images': len(blocks) * len(geometry), 'ranking': ranking})
            print(filename, 'best:', ranking[0], flush=True)
            save_json(output / 'summary.json', report)
        report['ranking'] = rank_candidates(all_rows)
        save_csv(output / 'ranking.csv', report['ranking'])
        save_csv(output / 'per_frame.csv', all_rows)
        report['success'] = True
        print('AGGREGATE TOP 10', flush=True)
        for row in report['ranking'][:10]:
            print(row, flush=True)
    except Exception as error:
        report['error'] = {'message': str(error), 'traceback': traceback.format_exc()}
        print(report['error']['traceback'], file=sys.stderr, flush=True)
    finally:
        report['elapsed_seconds'] = time.perf_counter() - started
        save_json(output / 'summary.json', report)
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
