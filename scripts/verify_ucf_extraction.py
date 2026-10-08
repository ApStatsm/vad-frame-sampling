"""Compare official UCF features with all 16 frame positions and mean pooling.

This is an extraction diagnostic, not a sampler benchmark or VAD evaluation.
Every candidate uses identical blocks, decoder and CLIP weights; an optional
preprocessing diagnostic changes explicitly named image transforms. A winning
rank is not proof of the historical extraction recipe; inspect numeric agreement.
"""

import argparse
from collections import defaultdict
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import platform
import sys
import time
import traceback
from types import MethodType


ROOT = Path(__file__).resolve().parents[1]
BLOCK_FRAMES = 16
DEFAULT_BLOCKS = (0, 1, 2, 3, 8, 16, 32)
PREPROCESS_RECIPES = {
    'published': 'Upstream image_crop: OpenCV linear resize 340x256, crop, BGR->RGB, flip',
    'bgr_as_rgb': 'Published crop, then swap R/B: encode BGR channel order as RGB',
    'opencv_cubic': 'Published geometry, OpenCV INTER_CUBIC instead of INTER_LINEAR',
    'opencv_area': 'Published geometry, OpenCV INTER_AREA instead of INTER_LINEAR',
    'pil_bicubic': 'Published geometry, RGB Pillow bicubic resize instead of OpenCV resize',
    'pil_bilinear': 'Published geometry, RGB Pillow bilinear resize instead of OpenCV resize',
    'shortside_opencv_linear': 'OpenCV linear resize short side to 256 preserving aspect, center crop',
    'shortside_opencv_scale': 'OpenCV linear resize with fx=fy=256/min(height,width), then center crop',
    'scale_jpeg_raw95': 'JPEG quality95 roundtrip of decoded frame before fx/fy linear resize and crop',
    'scale_jpeg_raw100': 'JPEG quality100 roundtrip of decoded frame before fx/fy linear resize and crop',
    'scale_jpeg_resized95': 'JPEG quality95 roundtrip after fx/fy linear resize, before crop and flip',
    'scale_jpeg_resized100': 'JPEG quality100 roundtrip after fx/fy linear resize, before crop and flip',
    'shortside_opencv_scale_preflip': 'Horizontal flip of raw frame before fx/fy aspect resize and center crop',
    'shortside_opencv_scale_floor': 'Float32 RGB fx/fy bilinear resize, floor to uint8, center crop and flip',
    'shortside_opencv_scale_round': 'Float32 RGB fx/fy bilinear resize, nearest rounding to uint8',
    'shortside_opencv_scale_ceil': 'Float32 RGB fx/fy bilinear resize, ceiling to uint8',
    'shortside_opencv_scale_unit_floor': 'Float32 RGB /255 fx/fy resize, multiply255 then truncate to uint8',
    'shortside_opencv_scale_area': 'OpenCV uint8 INTER_AREA fx/fy resize, center crop and flip',
    'shortside_opencv_scale_cubic': 'OpenCV uint8 INTER_CUBIC fx/fy resize, center crop and flip',
    'shortside_opencv_scale_exact': 'OpenCV uint8 INTER_LINEAR_EXACT fx/fy resize, center crop and flip',
    'shortside_opencv_scale_lanczos': 'OpenCV uint8 INTER_LANCZOS4 fx/fy resize, center crop and flip',
    'shortside_bicubic': 'RGB Pillow bicubic resize short side to 256 preserving aspect, center crop',
    'shortside_bilinear': 'RGB Pillow bilinear resize short side to 256 preserving aspect, center crop',
    'shortside_opencv_linear_exact': 'OpenCV INTER_LINEAR_EXACT aspect-preserving short-side 256 resize',
    'shortside_tensor_bilinear': 'Float RGB tensor torchvision bilinear short-side 256 resize, antialias=True',
    'shortside_tensor_bicubic': 'Float RGB tensor torchvision bicubic short-side 256 resize, antialias=True',
    'shortside_cv2_float': 'Float RGB OpenCV linear short-side 256 resize without uint8 rounding',
    'clip_direct': 'RGB original frame, optional horizontal flip, then CLIP preprocess only',
    'jpeg_q95': 'OpenCV JPEG quality 95 roundtrip of decoded BGR frame before published crop',
    'crop_x_minus1': 'Published resize with center crop shifted one pixel left before optional flip',
    'crop_x_plus1': 'Published resize with center crop shifted one pixel right before optional flip',
    'crop_y_minus1': 'Published resize with center crop shifted one pixel up before optional flip',
    'crop_y_plus1': 'Published resize with center crop shifted one pixel down before optional flip',
}


def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def save_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_output():
    stamp = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
    for attempt in range(10000):
        suffix = '' if attempt == 0 else f'_{attempt}'
        output = ROOT / 'outputs/setup' / f'ucf_extraction_verify_{stamp}{suffix}'
        try:
            output.mkdir(parents=True, exist_ok=False)
            (output / 'arrays').mkdir()
            return output
        except FileExistsError:
            continue
    raise RuntimeError('Could not allocate a unique output directory')


def choose_videos(manifest, requested):
    videos = manifest['videos']
    names = [video['filename'] for video in videos]
    if len(names) != len(set(names)):
        raise ValueError('Manifest contains duplicate video filenames')
    if requested:
        if len(requested) != len(set(requested)) or set(requested) - set(names):
            raise ValueError('--video must name distinct exact manifest filenames')
        by_name = {video['filename']: video for video in videos}
        return [by_name[name] for name in requested]
    # Use manifest order and video labels, never anomaly intervals or VAD scores.
    anomaly = next(video for video in videos if video['is_anomaly'])
    normal = next(video for video in videos if not video['is_anomaly'])
    other = next(video for video in videos
                 if video['is_anomaly'] and video['category'] != anomaly['category'])
    return [anomaly, normal, other]


def index_features(root, videos, crop_type):
    """Retain upstream test folders while rebasing its obsolete absolute root.

    Some normal basenames also exist in a training/event folder. Basename-only
    lookup is ambiguous; the official test CSV specifies the correct counterpart.
    """
    root = root.resolve()
    test_list = ROOT / 'VadCLIP/list/ucf_CLIP_rgbtest.csv'
    listed = {}
    with test_list.open(newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ['path', 'label']:
            raise ValueError(f'Unexpected test CSV header in {test_list}')
        for row in reader:
            original = PurePosixPath(row['path'].strip())
            if 'UCFClipFeatures' not in original.parts:
                raise ValueError(f'Missing feature-root marker in {test_list}')
            marker = original.parts.index('UCFClipFeatures')
            relative = Path(*original.parts[marker + 1:])
            if not relative.parts or '..' in relative.parts or not relative.name.endswith('__5.npy'):
                raise ValueError(f'Invalid test counterpart: {original}')
            video_stem = relative.name.removesuffix('__5.npy')
            filename = video_stem + '.mp4'
            if filename in listed:
                raise ValueError(f'Duplicate official test video: {filename}')
            candidate = (root / relative.with_name(f'{video_stem}__{crop_type}.npy')).resolve()
            if not candidate.is_relative_to(root):
                raise ValueError('Feature counterpart resolves outside configured root')
            listed[filename] = candidate
    paths = {}
    for video in videos:
        filename = video['filename']
        if filename not in listed or not listed[filename].is_file():
            raise ValueError(f'Official test counterpart missing: {filename}, crop {crop_type}')
        paths[filename] = listed[filename]
    return paths


def validate_device(torch, device, precision):
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is not available; use --device cpu or mps')
    if device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS is not available; use --device cpu or cuda')
    if device == 'cpu' and precision == 'float16':
        raise ValueError('Use float32 on CPU; float16 is supported here on cuda/mps only')


def legacy_attention(block, values):
    """Explicit attention weights force the unfused bmm/softmax path in torch 2.x.

    This tests computation order, not new learned weights. It is not a guarantee
    of bitwise agreement with an unavailable historical CUDA/PyTorch build.
    """
    mask = block.attn_mask
    if mask is not None:
        mask = mask.to(dtype=values.dtype, device=values.device)
    return block.attn(values, values, values, need_weights=True,
                      average_attn_weights=False, attn_mask=mask)[0]


def jpeg_roundtrip(frame, quality, cv2):
    ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError('JPEG encoding failed')
    restored = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if restored is None or restored.shape != frame.shape:
        raise ValueError('JPEG decoding failed')
    return restored


def diagnostic_image(frame, variant, crop_type, image_crop, np, cv2, Image):
    """Named hypotheses; baseline delegates to the upstream crop without changes."""
    if variant not in PREPROCESS_RECIPES:
        raise ValueError(f'Unknown preprocessing variant: {variant}')
    if variant.startswith('scale_jpeg_raw'):
        quality = int(variant.removeprefix('scale_jpeg_raw'))
        frame = jpeg_roundtrip(frame, quality, cv2)
        variant = 'shortside_opencv_scale'
    jpeg_after_resize = None
    if variant.startswith('scale_jpeg_resized'):
        jpeg_after_resize = int(variant.removeprefix('scale_jpeg_resized'))
        variant = 'shortside_opencv_scale'
    if variant == 'published':
        return Image.fromarray(image_crop(frame, crop_type))
    if variant == 'bgr_as_rgb':
        cropped = image_crop(frame, crop_type)
        return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
    if variant.startswith('jpeg_q'):
        quality = int(variant.removeprefix('jpeg_q'))
        ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise ValueError(f'JPEG encoding failed: {variant}')
        restored = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if restored is None or restored.shape != frame.shape:
            raise ValueError(f'JPEG decoding failed: {variant}')
        return Image.fromarray(image_crop(restored, crop_type))
    # Non-baseline geometry diagnostics are restricted to center crop 0/5.
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    preflip = variant == 'shortside_opencv_scale_preflip'
    if preflip and crop_type == 5:
        rgb = cv2.flip(rgb, 1)
    if variant == 'clip_direct':
        return Image.fromarray(cv2.flip(rgb, 1) if crop_type == 5 else rgb)
    if variant.startswith('shortside_opencv_scale'):
        scale = 256 / min(rgb.shape[:2])
        kernels = {'shortside_opencv_scale_area': cv2.INTER_AREA,
                   'shortside_opencv_scale_cubic': cv2.INTER_CUBIC,
                   'shortside_opencv_scale_exact': cv2.INTER_LINEAR_EXACT,
                   'shortside_opencv_scale_lanczos': cv2.INTER_LANCZOS4}
        if variant in ('shortside_opencv_scale', 'shortside_opencv_scale_preflip') or variant in kernels:
            resized = cv2.resize(rgb, None, fx=scale, fy=scale,
                                 interpolation=kernels.get(variant, cv2.INTER_LINEAR))
        else:
            source = rgb.astype(np.float32)
            if variant.endswith('unit_floor'):
                source /= 255
            resized = cv2.resize(source, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            if variant.endswith('unit_floor'):
                resized *= 255
            if variant.endswith('_round'):
                resized = np.rint(resized)
            elif variant.endswith('_ceil'):
                resized = np.ceil(resized)
            resized = np.clip(resized, 0, 255).astype(np.uint8)
    elif variant in ('pil_bicubic', 'pil_bilinear', 'shortside_bicubic', 'shortside_bilinear',
                   'shortside_opencv_linear', 'shortside_opencv_linear_exact'):
        image = Image.fromarray(rgb)
        size = (340, 256)
        if variant.startswith('shortside_'):
            width, height = image.size
            size = ((int(256 * width / height), 256) if width >= height else
                    (256, int(256 * height / width)))
        if variant in ('shortside_opencv_linear', 'shortside_opencv_linear_exact'):
            interpolation = (cv2.INTER_LINEAR_EXACT if variant.endswith('_exact') else cv2.INTER_LINEAR)
            resized = cv2.resize(rgb, size, interpolation=interpolation)
        else:
            interpolation = Image.Resampling.BILINEAR if variant.endswith('bilinear') else Image.Resampling.BICUBIC
            resized = np.asarray(image.resize(size, interpolation))
    else:
        interpolation = {'opencv_cubic': cv2.INTER_CUBIC, 'opencv_area': cv2.INTER_AREA}
        resized = cv2.resize(rgb, (340, 256),
                             interpolation=interpolation.get(variant, cv2.INTER_LINEAR))
    if jpeg_after_resize is not None:
        resized = cv2.cvtColor(jpeg_roundtrip(cv2.cvtColor(resized, cv2.COLOR_RGB2BGR),
                                            jpeg_after_resize, cv2), cv2.COLOR_BGR2RGB)
    y = int(round((resized.shape[0] - 224) / 2))
    x = int(round((resized.shape[1] - 224) / 2))
    shifts = {'crop_x_minus1': (-1, 0), 'crop_x_plus1': (1, 0),
              'crop_y_minus1': (0, -1), 'crop_y_plus1': (0, 1)}
    dx, dy = shifts.get(variant, (0, 0))
    cropped = resized[y + dy:y + dy + 224, x + dx:x + dx + 224]
    if cropped.shape != (224, 224, 3):
        raise ValueError(f'Unexpected crop shape for {variant}: {cropped.shape}')
    return Image.fromarray(cv2.flip(cropped, 1) if crop_type == 5 and not preflip else cropped)


def diagnostic_tensor(frame, variant, crop_type, preprocess, np, torch, cv2):
    from torchvision.transforms import functional as functional
    from torchvision.transforms import InterpolationMode
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if variant == 'shortside_cv2_float':
        height, width = rgb.shape[:2]
        size = ((int(256 * width / height), 256) if width >= height else
                (256, int(256 * height / width)))
        resized = cv2.resize(rgb.astype(np.float32) / 255, size, interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized).permute(2, 0, 1)
    else:
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255
        interpolation = (InterpolationMode.BICUBIC if variant.endswith('bicubic') else
                         InterpolationMode.BILINEAR)
        tensor = functional.resize(tensor, [256], interpolation=interpolation, antialias=True)
    tensor = functional.center_crop(tensor, [224, 224])
    if crop_type == 5:
        tensor = functional.hflip(tensor)
    normalizer = preprocess.transforms[-1]
    if hasattr(normalizer, 'mean') and hasattr(normalizer, 'std'):
        tensor = normalizer(tensor)
    return tensor


def encode_blocks(path, blocks, visual, preprocess, image_crop, args, np, torch, cv2, Image):
    """Decode sequentially from frame zero; encode only requested complete blocks.

    Prefix diagnostics deliberately stop after the last requested block. They do
    not validate video EOF, the full decode count, or the incomplete final block.
    """
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    encoded_chunks = {variant: [] for variant in args.preprocess_variants}
    batches = {variant: [] for variant in args.preprocess_variants}
    frame_indices, decoder_pts = [], []
    parameter_dtype = visual.conv1.weight.dtype
    wanted = set(blocks)
    last_frame_exclusive = (max(blocks) + 1) * BLOCK_FRAMES
    decoded = 0

    def flush(variant):
        batch = batches[variant]
        if not batch:
            return
        images = torch.stack(batch).to(device=args.device, dtype=parameter_dtype)
        batch.clear()
        with torch.no_grad():
            features = visual(images)
        array = features.detach().float().cpu().numpy()
        if array.shape != (len(images), 512) or not np.isfinite(array).all():
            raise ValueError('CLIP visual encoder must produce finite [image_count,512] features')
        encoded_chunks[variant].append(array)

    try:
        if not cap.isOpened():
            raise ValueError(f'Cannot open video with OpenCV FFmpeg: {path}')
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        header_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f'Invalid decoder FPS: {fps}')
        backend = cap.getBackendName()
        while decoded < last_frame_exclusive:
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f'Decode ended at frame {decoded}; required {last_frame_exclusive}. '
                                 'Select earlier --blocks or investigate video/feature alignment.')
            index = decoded
            decoded += 1
            if index // BLOCK_FRAMES in wanted:
                frame_indices.append(index)
                pts = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000
                decoder_pts.append(pts if math.isfinite(pts) else float('nan'))
                if args.image_channel_order == 'bgr':
                    # All image helpers assume BGR decode and convert to RGB.
                    # Swap beforehand to deliberately feed BGR-valued pixels to
                    # the RGB encoder, including before pixel normalization.
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                for variant in args.preprocess_variants:
                    if variant in ('shortside_tensor_bilinear', 'shortside_tensor_bicubic', 'shortside_cv2_float'):
                        tensor = diagnostic_tensor(frame, variant, args.crop_type, preprocess, np, torch, cv2)
                    else:
                        image = diagnostic_image(frame, variant, args.crop_type, image_crop, np, cv2, Image)
                        tensor = preprocess(image)
                    batches[variant].append(tensor)
                    if len(batches[variant]) >= args.batch_size:
                        flush(variant)
        for variant in args.preprocess_variants:
            flush(variant)
    finally:
        cap.release()
    frame_features = {variant: np.concatenate(chunks).reshape(len(blocks), BLOCK_FRAMES, 512)
                      for variant, chunks in encoded_chunks.items()}
    indices = np.asarray(frame_indices, dtype=np.int64).reshape(len(blocks), BLOCK_FRAMES)
    expected = np.asarray(blocks, dtype=np.int64)[:, None] * BLOCK_FRAMES + np.arange(BLOCK_FRAMES)
    if not np.array_equal(indices, expected):
        raise ValueError('Decoded frame order does not match requested block indices')
    info = {
        'decoded_prefix_frames': decoded,
        'encoder_input_images': len(frame_indices) * len(args.preprocess_variants),
        'encoder_input_images_per_variant': len(frame_indices),
        'encoder_batches': sum(len(chunks) for chunks in encoded_chunks.values()),
        'encoder_batches_per_variant': {name: len(chunks) for name, chunks in encoded_chunks.items()},
        'decoder_backend': backend, 'fps': fps,
        'header_frames': header_count if math.isfinite(header_count) else None,
        'decode_scope': 'prefix through last requested complete block; EOF and tail not checked',
        'timestamp_source': 'frame index / FPS is nominal CFR time; decoder PTS also retained',
        'decoder_pts_all_finite': bool(np.isfinite(decoder_pts).all()),
    }
    return frame_features, indices, np.asarray(decoder_pts).reshape(indices.shape), info


def unit_vectors(values, np):
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError('Cannot normalize non-finite or zero features')
    return values / norms


def candidates(frame_features, np):
    values = {f'single_offset_{offset:02d}': frame_features[:, offset, :]
              for offset in range(BLOCK_FRAMES)}
    values['mean16'] = frame_features.mean(axis=1, dtype=np.float32)
    values['mean16_of_unit_frames'] = unit_vectors(frame_features, np).mean(axis=1, dtype=np.float32)
    return values


def comparisons(video, blocks, frame_features, official, shifts, args, np, variant):
    rows = []
    for name, candidate in candidates(frame_features, np).items():
        # Cast-to-official variants test storage quantization, not new frame recipes.
        normalized = unit_vectors(candidate, np)
        forms = {'raw': candidate, 'raw_cast_official': candidate.astype(official.dtype),
                 'l2': normalized, 'l2_cast_official': normalized.astype(official.dtype)}
        for form, prediction in forms.items():
            prediction = prediction.astype(np.float64)
            if not np.isfinite(prediction).all():
                raise ValueError(f'{name}/{form}: feature conversion produced non-finite values')
            for shift in shifts:
                indices = np.asarray(blocks, dtype=np.int64) + shift
                target = np.asarray(official[indices], dtype=np.float64)
                target_norm = np.linalg.norm(target, axis=1)
                prediction_norm = np.linalg.norm(prediction, axis=1)
                if np.any(target_norm <= 0) or np.any(prediction_norm <= 0):
                    raise ValueError('Zero official/candidate vectors prevent cosine comparison')
                difference = prediction - target
                absolute = np.abs(difference)
                cosine = np.clip(np.sum(prediction * target, axis=1) /
                                 (prediction_norm * target_norm), -1, 1)
                close = absolute <= args.atol + args.rtol * np.abs(target)
                for position, block in enumerate(blocks):
                    rows.append({
                        'video': video, 'block_index': block, 'start_frame': block * BLOCK_FRAMES,
                        'end_frame_exclusive': (block + 1) * BLOCK_FRAMES,
                        'official_row': int(indices[position]), 'row_shift': shift,
                        'candidate': name, 'feature_form': form, 'preprocess_variant': variant,
                        'cosine': float(cosine[position]),
                        'mean_absolute_error': float(absolute[position].mean()),
                        'rmse': float(np.sqrt(np.mean(difference[position] ** 2))),
                        'max_absolute_error': float(absolute[position].max()),
                        'relative_l2_error': float(np.linalg.norm(difference[position]) / target_norm[position]),
                        'candidate_norm': float(prediction_norm[position]),
                        'official_norm': float(target_norm[position]),
                        'exact_element_fraction': float(np.mean(prediction[position] == target[position])),
                        'close_element_fraction': float(close[position].mean()),
                        'tolerance_pass': bool(close[position].all()),
                    })
    return rows


def rank_candidates(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['preprocess_variant'], row['candidate'], row['feature_form'], row['row_shift'])].append(row)
    ranked = []
    for (variant, name, form, shift), records in groups.items():
        n = len(records)
        # Each row has 512 dimensions; pooled L2 weights official vector energy.
        difference_sq = sum(record['rmse'] ** 2 * 512 for record in records)
        target_sq = sum(record['official_norm'] ** 2 for record in records)
        ranked.append({
            'candidate': name, 'feature_form': form, 'row_shift': shift, 'preprocess_variant': variant,
            'compared_blocks': n, 'compared_videos': len({r['video'] for r in records}),
            'cosine_mean': sum(r['cosine'] for r in records) / n,
            'cosine_min': min(r['cosine'] for r in records),
            'mean_absolute_error': sum(r['mean_absolute_error'] for r in records) / n,
            'rmse': math.sqrt(difference_sq / (n * 512)),
            'max_absolute_error': max(r['max_absolute_error'] for r in records),
            'relative_l2_error': math.sqrt(difference_sq / target_sq),
            'candidate_norm_mean': sum(r['candidate_norm'] for r in records) / n,
            'official_norm_mean': sum(r['official_norm'] for r in records) / n,
            'exact_element_fraction': sum(r['exact_element_fraction'] for r in records) / n,
            'close_element_fraction': sum(r['close_element_fraction'] for r in records) / n,
            'tolerance_pass_fraction': sum(r['tolerance_pass'] for r in records) / n,
            'all_blocks_within_tolerance': all(r['tolerance_pass'] for r in records),
        })
    ranked.sort(key=lambda r: (r['relative_l2_error'], -r['cosine_mean'],
                               r['preprocess_variant'], r['candidate'], r['feature_form'], r['row_shift']))
    return [{'rank': index, **row} for index, row in enumerate(ranked, 1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu')
    parser.add_argument('--precision', choices=('float32', 'float16'), default='float32')
    parser.add_argument('--activation', choices=('quick_gelu', 'gelu', 'gelu_tanh'), default='quick_gelu',
                        help='Diagnostic visual activation; weights remain fixed, upstream default QuickGELU')
    parser.add_argument('--attention-backend', choices=('default', 'legacy'), default='default',
                        help='Default need_weights=False vs unfused need_weights=True diagnostic')
    parser.add_argument('--image-normalization', choices=('clip', 'imagenet', 'symmetric', 'none'),
                        default='clip', help='Pixel normalization before CLIP; distinct from feature L2')
    parser.add_argument('--image-channel-order', choices=('rgb', 'bgr'), default='rgb',
                        help='Pixel channel order supplied to the RGB encoder; BGR is a diagnostic hypothesis')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/ucf_resources.json')
    parser.add_argument('--video', action='append', help='Exact test filename; repeat to select multiple videos')
    parser.add_argument('--blocks', type=int, nargs='+', default=list(DEFAULT_BLOCKS),
                        help='Zero-based 16-frame block indices; default: 0 1 2 3 8 16 32')
    parser.add_argument('--row-shifts', type=int, nargs='+', default=[0],
                        help='Compare decoded block b against official row b+shift; default: 0')
    parser.add_argument('--crop-type', type=int, choices=range(10), default=5,
                        help='Compare against matching __N.npy crop variant; official UCF test default: 5')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--preprocess-variants', nargs='+', default=['published'],
                        choices=['all', *PREPROCESS_RECIPES],
                        help='Explicit image-preprocessing hypotheses, or all; default: published only')
    parser.add_argument('--atol', type=float, default=1e-4)
    parser.add_argument('--rtol', type=float, default=1e-3)
    args = parser.parse_args()
    if (args.batch_size < 1 or not all(math.isfinite(x) and x >= 0 for x in (args.atol, args.rtol))
            or any(block < 0 for block in args.blocks)):
        parser.error('Use a positive batch size and finite non-negative blocks/tolerances')
    args.blocks = sorted(set(args.blocks))
    args.row_shifts = sorted(set(args.row_shifts))
    if 'all' in args.preprocess_variants:
        if args.preprocess_variants != ['all']:
            parser.error('Use --preprocess-variants all by itself')
        args.preprocess_variants = list(PREPROCESS_RECIPES)
    else:
        args.preprocess_variants = list(dict.fromkeys(args.preprocess_variants))
    if args.crop_type not in (0, 5) and args.preprocess_variants != ['published']:
        parser.error('Preprocessing diagnostics support center crop 0/5 only; published supports 0..9')
    output = make_output()
    started = time.perf_counter()
    report = {
        'success': False, 'output_dir': str(output), 'dataset': 'UCF-Crime',
        'purpose': 'Official-feature extraction diagnostic; no VAD inference or sampling benchmark',
        'historical_recipe_confirmed': False, 'settings': vars(args).copy(),
        'tolerance_rule': 'abs(candidate - official) <= atol + rtol * abs(official), all 512 elements',
        'ranking_rule': 'pooled relative L2 error ascending; cosine and agreement reported separately',
        'preprocess_recipes': {name: PREPROCESS_RECIPES[name] for name in args.preprocess_variants},
        'alignment_assumption': 'decoded block b spans frames [16*b,16*b+16); official row is b+shift',
        'warnings': [
            'The lowest error candidate is not automatically a verified original recipe.',
            'High cosine alone is insufficient, especially for adjacent/static frames.',
            'Cast variants or normalization forms can be numerically identical and tie.',
            'If no candidate matches, check video versions, decoding, preprocessing and precision.',
            'Only complete sampled blocks are compared; EOF/tail behavior remains unverified.',
        ],
        'videos': [], 'input_sha256': {},
    }
    report['settings']['config'] = str(project_path(args.config))
    save_json(output / 'summary.json', report)
    print(f'Output: {output}', flush=True)
    try:
        import numpy as np
        import torch
        import cv2
        from PIL import Image

        validate_device(torch, args.device, args.precision)
        config_path = project_path(args.config)
        resources = json.loads(config_path.read_text())
        manifest_path = ROOT / 'UCF-Crime/test/manifest.json'
        manifest = json.loads(manifest_path.read_text())
        selected = choose_videos(manifest, args.video)
        feature_root = project_path(resources['features_root'])
        checkpoint = project_path(resources['clip_checkpoint'])
        if not feature_root.is_dir() or not checkpoint.is_file():
            raise FileNotFoundError('Configured feature directory or local CLIP checkpoint does not exist')
        feature_paths = index_features(feature_root, selected, args.crop_type)
        report['input_sha256'].update(config=file_hash(config_path), manifest=file_hash(manifest_path),
                                      clip_checkpoint=file_hash(checkpoint), script=file_hash(Path(__file__)))
        sys.path.insert(0, str(ROOT / 'VadCLIP/src'))
        import clip
        from clip.model import convert_weights
        from crop import image_crop

        for name, relative in (('official_test_list', 'VadCLIP/list/ucf_CLIP_rgbtest.csv'),
                               ('crop_source', 'VadCLIP/src/crop.py'),
                               ('clip_source', 'VadCLIP/src/clip/clip.py'),
                               ('clip_model_source', 'VadCLIP/src/clip/model.py')):
            report['input_sha256'][name] = file_hash(ROOT / relative)
        # Local checkpoint path prevents implicit downloading. Use the upstream
        # preprocessing and image_crop, then explicitly set visual precision.
        model, preprocess = clip.load(str(checkpoint), device='cpu', jit=False)
        if args.image_normalization != 'clip':
            from torchvision.transforms import Compose, Normalize
            transforms = list(preprocess.transforms[:-1])
            if args.image_normalization == 'imagenet':
                transforms.append(Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
            elif args.image_normalization == 'symmetric':
                transforms.append(Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
            preprocess = Compose(transforms)
        visual = model.visual.float().eval().requires_grad_(False)
        if args.attention_backend == 'legacy':
            for block in visual.transformer.resblocks:
                block.attention = MethodType(legacy_attention, block)
        activation_replacements = 0
        if args.activation != 'quick_gelu':
            from clip.model import QuickGELU
            names = [name for name, module in visual.named_modules() if isinstance(module, QuickGELU)]
            if not names:
                raise ValueError('Expected upstream visual QuickGELU modules')
            for name in names:
                parent, child = name.rsplit('.', 1)
                visual.get_submodule(parent)._modules[child] = torch.nn.GELU(
                    approximate='tanh' if args.activation == 'gelu_tanh' else 'none')
            activation_replacements = len(names)
        if args.precision == 'float16':
            # CLIP LayerNorm runs in fp32: preserve its parameters and use the
            # upstream selective fp16 conversion rather than visual.half().
            convert_weights(visual)
        visual = visual.to(device=args.device)
        dtype = visual.conv1.weight.dtype
        report['environment'] = {'python': platform.python_version(), 'platform': platform.platform(),
                                 'torch': torch.__version__, 'numpy': np.__version__,
                                 'opencv': cv2.__version__, 'encoder_dtype': str(dtype),
                                 'visual_activation': args.activation,
                                 'attention_backend': args.attention_backend,
                                 'activation_replacements': activation_replacements,
                                 'precision_policy': 'fp32 or upstream selective fp16; LayerNorm stays fp32',
                                 'preprocess': repr(preprocess)}
        detail_rows = []
        for video in selected:
            filename = video['filename']
            video_path = (ROOT / 'UCF-Crime/test' / video['video_path']).resolve()
            official_path = feature_paths[filename]
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
            video_sha256 = file_hash(video_path)
            if video_sha256 != video['sha256']:
                raise ValueError(f'{filename}: video SHA-256 differs from prepared manifest')
            official = np.load(official_path, mmap_mode='r', allow_pickle=False)
            if (official.ndim != 2 or official.shape[1] != 512 or official.shape[0] < 1
                    or official.dtype.kind != 'f'):
                raise ValueError(f'Expected floating [tokens,512] official feature: {official_path}')
            # Intersection guarantees equal block counts for all row shifts.
            blocks = [b for b in args.blocks if b < len(official)
                      and all(0 <= b + shift < len(official) for shift in args.row_shifts)]
            if not blocks:
                raise ValueError(f'{filename}: no common valid --blocks for the requested row shifts')
            official_indices = np.asarray(blocks)[:, None] + np.asarray(args.row_shifts)[None, :]
            if not np.isfinite(official[official_indices.ravel()]).all():
                raise ValueError(f'{filename}: selected official rows contain non-finite values')
            print(f'{filename}: blocks {blocks}; encode {len(blocks) * BLOCK_FRAMES} images '
                  f'x {len(args.preprocess_variants)} preprocessing variants', flush=True)
            feature_sets, indices, pts, info = encode_blocks(
                video_path, blocks, visual, preprocess, image_crop, args, np, torch, cv2, Image)
            rows = []
            for variant, frame_features in feature_sets.items():
                rows.extend(comparisons(filename, blocks, frame_features, official,
                                        args.row_shifts, args, np, variant))
            detail_rows.extend(rows)
            ranking = rank_candidates(rows)
            np.savez_compressed(
                output / 'arrays' / f'{Path(filename).stem}.npz',
                block_indices=np.asarray(blocks), frame_indices=indices,
                nominal_timestamps_seconds=indices / info['fps'], decoder_timestamps_seconds=pts,
                frame_features=feature_sets[args.preprocess_variants[0]],
                frame_features_variant=np.asarray(args.preprocess_variants[0]),
                frame_features_by_preprocess=np.stack(list(feature_sets.values())),
                preprocess_variants=np.asarray(args.preprocess_variants), official_rows=official_indices,
                official_features=np.asarray(official[official_indices]),
                row_shifts=np.asarray(args.row_shifts),
            )
            video_record = {
                'video': filename, 'video_path': str(video_path), 'official_path': str(official_path),
                'video_sha256': video_sha256, 'official_sha256': file_hash(official_path),
                'video_matches_manifest_sha256': True,
                'official_shape': list(official.shape), 'official_dtype': str(official.dtype),
                'blocks': blocks, 'skipped_requested_blocks': sorted(set(args.blocks) - set(blocks)),
                **info, 'ranking': ranking,
            }
            report['videos'].append(video_record)
            save_csv(output / 'per_block.csv', detail_rows)
            save_json(output / 'summary.json', report)
            best = ranking[0]
            print(f"  Best: {best['preprocess_variant']} / {best['candidate']} / "
                  f"{best['feature_form']} / shift {best['row_shift']}; "
                  f"cos={best['cosine_mean']:.8f}, relL2={best['relative_l2_error']:.6g}, "
                  f"tolerance pass={best['tolerance_pass_fraction']:.1%}", flush=True)
        report['ranking'] = rank_candidates(detail_rows)
        best = report['ranking'][0]
        report['numeric_assessment'] = (
            'Best candidate is within the declared tolerance on all compared blocks; '
            'check ties, cross-video consistency, and larger independent blocks before adopting it.'
            if best['all_blocks_within_tolerance'] else
            'No tested candidate is within the declared tolerance on all compared blocks; '
            'the original extraction recipe is unresolved. Do not adopt rank 1 solely for winning.'
        )
        save_csv(output / 'ranking.csv', report['ranking'])
        report['preprocess_ranking'] = [next(row for row in report['ranking']
                                            if row['preprocess_variant'] == variant)
                                        for variant in args.preprocess_variants]
        report['preprocess_ranking'].sort(key=lambda row: row['rank'])
        save_csv(output / 'preprocess_ranking.csv', report['preprocess_ranking'])
        report['success'] = True
        if len(args.preprocess_variants) > 1:
            print('\nBest candidate per preprocessing variant:', flush=True)
            for row in report['preprocess_ranking']:
                print(f"  {row['preprocess_variant']}: {row['candidate']} / {row['feature_form']} / "
                      f"shift {row['row_shift']}; relL2={row['relative_l2_error']:.6g}, "
                      f"pass={row['tolerance_pass_fraction']:.1%}", flush=True)
        print('\nAggregate top 5 (ranked by numeric error, not detection performance):', flush=True)
        for row in report['ranking'][:5]:
            print(f"  {row['rank']}. {row['preprocess_variant']} / {row['candidate']} / "
                  f"{row['feature_form']} / shift {row['row_shift']}: "
                  f"cos={row['cosine_mean']:.8f}, relL2={row['relative_l2_error']:.6g}, "
                  f"maxErr={row['max_absolute_error']:.6g}, "
                  f"pass={row['tolerance_pass_fraction']:.1%}", flush=True)
        print(report['numeric_assessment'], flush=True)
    except Exception as error:
        report['error'] = {'type': type(error).__name__, 'message': str(error),
                           'traceback': traceback.format_exc()}
        print(report['error']['traceback'], file=sys.stderr, flush=True)
    finally:
        report['elapsed_seconds'] = time.perf_counter() - started
        save_json(output / 'summary.json', report)
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
