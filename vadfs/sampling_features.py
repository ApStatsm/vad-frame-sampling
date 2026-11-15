"""80D cheap features and fixed development-only standardization/PCA artifacts.

No CLIP, GT or labels are used by collection/selection. Training metadata is
checked against test identities before any transform may be used for inference.
"""

import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from .paths import package_hashes, sha256


FEATURE_SPEC = {'version': 1, 'gray': '0.299R+0.587G+0.114B; RGB uint8 / 255',
                'resize': 'OpenCV INTER_AREA 64x64 after grayscale; float64',
                'appearance': '8x8 cells, each 8x8 pixels; row-major means',
                'motion': '4x4 cells, each 16x16 pixels; abs current-previous candidate; first zero',
                'dimensions': 80, 'pca_dimensions': 8, 'whitening': False}


def feature_row(rgb, previous=None):
    image = rgb.astype(np.float64) / 255.0
    gray = cv2.resize(image @ np.array([0.299, 0.587, 0.114]), (64, 64), interpolation=cv2.INTER_AREA)
    appearance = gray.reshape(8, 8, 8, 8).mean(axis=(1, 3)).ravel()
    difference = np.zeros_like(gray) if previous is None else np.abs(gray - previous)
    motion = difference.reshape(4, 16, 4, 16).mean(axis=(1, 3)).ravel()
    return np.concatenate((appearance, motion)), gray


def collect(path, config, grid):
    """First decode pass; retain only 80 floats per reference candidate."""
    from .color import yuv420p_to_rgb_x86
    from .extract import FfmpegYuvReader

    started = time.perf_counter()
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    reader = None
    timings = {'first_pass_decode': 0.0, 'cheap_features': 0.0}
    try:
        if not cap.isOpened():
            raise ValueError(f'Cannot open {path}')
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        header = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(header) or header < 1 or abs(header - round(header)) > 1e-3 or not np.isfinite(fps) or fps <= 0:
            raise ValueError('Invalid video metadata for first pass')
        if config['color_conversion'] == 'x86_swscale_emulation':
            cap.release()
            reader = FfmpegYuvReader(path, width, height)
        timings['first_pass_decode'] += time.perf_counter() - started
        source = reader or cap
        count, cursor, previous, rows = 0, 0, None, []
        while True:
            tick = time.perf_counter()
            ok, frame = source.read()
            timings['first_pass_decode'] += time.perf_counter() - tick
            if not ok:
                break
            if cursor < len(grid) and count == grid[cursor]:
                tick = time.perf_counter()
                rgb = (yuv420p_to_rgb_x86(frame, width, height) if reader
                       else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                row, previous = feature_row(rgb, previous)
                rows.append(row)
                cursor += 1
                timings['cheap_features'] += time.perf_counter() - tick
            count += 1
        offset = config['frame_offset']
        tail = (config.get('tail_policy') == 'match_official_rows' and count % 16
                and len(grid) == count // 16 + 1 and offset < count % 16)
        expected = count // 16 + int(bool(tail))
        if count != round(header) or cursor != len(grid) or expected != len(grid):
            raise ValueError('First-pass decode/header/candidate/tail mismatch')
    finally:
        tick = time.perf_counter()
        if reader:
            reader.release(check=sys.exc_info()[0] is None)
        else:
            cap.release()
        timings['first_pass_decode'] += time.perf_counter() - tick
    result = np.asarray(rows, dtype=np.float64)
    if result.shape != (len(grid), 80) or not np.isfinite(result).all():
        raise ValueError('Invalid cheap feature matrix')
    return result, {'decoded_frames': count, 'fps': fps, 'timing_seconds': timings}


def fit_transform(rows):
    """Fit once on development rows, with constant columns excluded."""
    x = np.asarray(rows, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 80 or len(x) < 9 or not np.isfinite(x).all():
        raise ValueError('PCA fitting requires at least 9 finite 80D development rows')
    mean, scale = x.mean(axis=0), x.std(axis=0)
    active = np.flatnonzero(scale > 1e-12)
    if len(active) < 8:
        raise ValueError('Fewer than eight nonconstant development dimensions')
    standardized = (x[:, active] - mean[active]) / scale[active]
    pca_mean = standardized.mean(axis=0)
    _, singular, vt = np.linalg.svd(standardized - pca_mean, full_matrices=False)
    if singular[7] <= 1e-8 * singular[0]:
        raise ValueError('Development PCA has fewer than eight effective dimensions')
    return {'mean': mean, 'scale': scale, 'active': active,
            'pca_mean': pca_mean, 'components': vt[:8]}


def load_artifact(path, dataset, recipe, test_videos):
    path = Path(path)
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved['metadata'].item()))
        arrays = {key: saved[key].copy() for key in ('mean', 'scale', 'active', 'pca_mean', 'components')}
    if (metadata.get('schema_version') != 1 or metadata.get('dataset_key') != dataset
            or metadata.get('split') not in ('train', 'development', 'validation')
            or metadata.get('feature_spec') != FEATURE_SPEC or metadata.get('extraction_config') != recipe):
        raise ValueError('Artifact dataset/split/feature definition/recipe mismatch')
    if (metadata.get('feature_source_sha256') != package_hashes()['vadfs/sampling_features.py']
            or metadata.get('color_source_sha256') != package_hashes()['vadfs/color.py']):
        raise ValueError('Cheap feature implementation changed since artifact fitting')
    training = metadata.get('training_videos', [])
    test_names = {video['filename'] for video in test_videos}
    test_hashes = {video['sha256'] for video in test_videos}
    if not training or any(video['filename'] in test_names or video['sha256'] in test_hashes for video in training):
        raise ValueError('Artifact training identities overlap test or provenance is empty')
    active = arrays['active']
    if (active.ndim != 1 or active.dtype.kind not in 'iu' or len(active) < 8
            or np.any(active < 0) or np.any(active >= 80) or np.any(np.diff(active) <= 0)
            or arrays['mean'].shape != (80,) or arrays['scale'].shape != (80,)
            or arrays['pca_mean'].shape != (len(active),) or arrays['components'].shape != (8, len(active))
            or any(not np.isfinite(value).all() for value in arrays.values())
            or np.any(arrays['scale'][active] <= 0)):
        raise ValueError('Malformed PCA artifact arrays')
    return arrays, {'path': str(path), 'sha256': sha256(path), 'metadata': metadata}


def transform(rows, artifact):
    active = artifact['active']
    z = (((rows[:, active] - artifact['mean'][active]) / artifact['scale'][active]
          - artifact['pca_mean']) @ artifact['components'].T)
    if not np.isfinite(z).all():
        raise ValueError('Nonfinite transformed features')
    return np.asarray(z, dtype=np.float64)
