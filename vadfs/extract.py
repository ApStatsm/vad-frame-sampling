"""Streaming raw UCF reference extraction.

Complete consecutive 16-frame blocks define tokens. single_frame encodes one
configured position per block; mean_block encodes all 16 and averages features.
No annotation or official feature values enter frame selection or encoding.

Reconstructed recipe (2026-10-08 diagnostics, docs/UCF_RAW_REFERENCE.md): frame
16k (offset 0), x86 libswscale YUV->RGB emulation, OpenCV linear resize with
fx=fy=256/min(H,W), crop [16:240, 58:282] (+flip for crop 5). It matched the 286
float16 official files on sampled blocks at relative L2 ~0.0015; the 4 longest
float32 videos remain unmatched. Full 290-video agreement is checked per run.
"""

import math
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch

from .color import yuv420p_to_rgb_x86
from .paths import add_vadclip_src
from .vadclip import synchronize

add_vadclip_src()
from crop import image_crop  # noqa: E402


class ReferenceImageEncoder:
    """Independent frozen visual-module copy; leave VadCLIP text precision intact."""

    def __init__(self, visual):
        self.visual = visual

    def encode_image(self, images):
        return self.visual(images)


EXTRACTION_STATUSES = ('provisional_not_verified_against_official_recipe',
                       'reconstructed_sample_verified_float16_cohort')
COLOR_CONVERSIONS = ('opencv_bgr', 'x86_swscale_emulation')
IMAGE_GEOMETRIES = ('published_340x256', 'shortside_fxfy')
TAIL_POLICIES = ('drop_incomplete', 'match_official_rows')


def config_value(config, key, default):
    """Older configs predate these keys; their defaults keep the earlier behaviour."""
    return config.get(key, default)


def validate_extraction(config):
    if type(config['block_frames']) is not int or config['block_frames'] != 16:
        raise ValueError('This reference uses 16-frame blocks to retain the official evaluation timeline')
    if config['temporal_mode'] not in ('single_frame', 'mean_block'):
        raise ValueError('temporal_mode must be single_frame or mean_block')
    for key in ('frame_offset', 'crop_type', 'encoder_batch_size', 'warmup_batches', 'seed'):
        if type(config[key]) is not int:
            raise ValueError(f'{key} must be an integer')
    if not 0 <= config['frame_offset'] < 16 or not 0 <= config['crop_type'] <= 9:
        raise ValueError('frame_offset must be 0..15; crop_type must be 0..9')
    batch = config['encoder_batch_size']
    if batch < 1 or (config['temporal_mode'] == 'mean_block' and batch % 16):
        raise ValueError('Batch size must be positive and divisible by 16 for mean_block')
    if config['encoder_precision'] not in ('float32', 'native'):
        raise ValueError('encoder_precision must be float32 or native')
    if config['feature_storage_dtype'] not in ('official', 'float32'):
        raise ValueError('feature_storage_dtype must be official or float32')
    if type(config['feature_l2_normalize']) is not bool or config['warmup_batches'] < 0:
        raise ValueError('Invalid feature normalization or warm-up setting')
    if not 0 <= config['seed'] < 2**32:
        raise ValueError('seed must be in 0..2**32-1')
    if config['extraction_status'] not in EXTRACTION_STATUSES:
        raise ValueError(f'extraction_status must be one of {EXTRACTION_STATUSES}')
    if config_value(config, 'color_conversion', 'opencv_bgr') not in COLOR_CONVERSIONS:
        raise ValueError(f'color_conversion must be one of {COLOR_CONVERSIONS}')
    if config_value(config, 'image_geometry', 'published_340x256') not in IMAGE_GEOMETRIES:
        raise ValueError(f'image_geometry must be one of {IMAGE_GEOMETRIES}')
    tail = config_value(config, 'tail_policy', 'drop_incomplete')
    if tail not in TAIL_POLICIES:
        raise ValueError(f'tail_policy must be one of {TAIL_POLICIES}')
    if tail == 'match_official_rows' and config['temporal_mode'] != 'single_frame':
        raise ValueError('match_official_rows only adds a single tail frame; use single_frame')


def crop_image(rgb, crop_type, geometry):
    """RGB uint8 frame -> 224x224 RGB crop in the upstream 10-crop index layout."""
    if geometry == 'published_340x256':
        return image_crop(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), crop_type)
    scale = 256 / min(rgb.shape[:2])
    img = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    height, width = img.shape[:2]
    y, x = (height - 224) // 2, (width - 224) // 2  # 341x256 -> [16:240, 58:282]
    base = crop_type % 5
    if base == 0:
        img = img[y:y + 224, x:x + 224]
    elif base == 1:
        img = img[:224, :224]
    elif base == 2:
        img = img[:224, -224:]
    elif base == 3:
        img = img[-224:, :224]
    else:
        img = img[-224:, -224:]
    return cv2.flip(img, 1) if crop_type >= 5 else img


class FfmpegYuvReader:
    """Sequential yuv420p frames from the ffmpeg CLI; no seeking, every decoded frame."""

    def __init__(self, path, width, height):
        self.frame_bytes = width * height * 3 // 2
        self.process = subprocess.Popen(
            ['ffmpeg', '-v', 'error', '-nostdin', '-i', str(path), '-map', '0:v:0', '-fps_mode', 'passthrough',
             '-f', 'rawvideo', '-pix_fmt', 'yuv420p', '-'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=self.frame_bytes * 4)

    def read(self):
        data = self.process.stdout.read(self.frame_bytes)
        if not data:
            return False, None
        if len(data) != self.frame_bytes:
            raise ValueError('Truncated yuv420p frame from ffmpeg')
        return True, data

    def release(self, check=True):
        self.process.stdout.close()
        if not check:  # an earlier error is propagating; do not mask it
            self.process.kill()
            self.process.wait()
            return
        stderr = self.process.stderr.read().decode(errors='replace').strip()
        code = self.process.wait()
        if code != 0 or stderr:
            raise ValueError(f'ffmpeg decode failed (exit {code}): {stderr[:500]}')


def extract_reference(path, encoder, preprocess, device, config, output_dtype, expected_tokens=None):
    """Decode to EOF, encode complete blocks, and keep bounded image-batch memory.

    Frame positions use a zero-based decode counter. Nominal timestamps are
    index/fps (explicit CFR assumption); OpenCV backend PTS are saved separately
    (NaN for the ffmpeg yuv420p reader). Decode/header disagreements fail rather
    than treating an early read failure as a complete video. No seeking, feature
    cache or padding. The incomplete tail is dropped unless tail_policy is
    match_official_rows and the official timeline has exactly one extra row
    (ceil(N/16) cohort); then the tail block's configured frame is encoded.
    expected_tokens is the audited official row count, never annotation labels.
    """
    color = config_value(config, 'color_conversion', 'opencv_bgr')
    geometry = config_value(config, 'image_geometry', 'published_340x256')
    tail_policy = config_value(config, 'tail_policy', 'drop_incomplete')
    stage = {name: 0.0 for name in ('decode', 'selection', 'color_conversion', 'preprocess', 'clip_input_transfer',
                                  'clip_forward', 'clip_output_transfer', 'feature_aggregation')}
    chunks, batch, chosen_indices, chosen_pts = [], [], [], []
    block_images, block_indices, block_pts = [], [], []
    group = 1 if config['temporal_mode'] == 'single_frame' else 16
    encoded = 0
    batches = 0
    next_progress = 1024
    clip_dtype = encoder.visual.conv1.weight.dtype

    def to_rgb(frame):
        started = time.perf_counter()
        rgb = (yuv420p_to_rgb_x86(frame, width, height) if color == 'x86_swscale_emulation'
               else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        stage['color_conversion'] += time.perf_counter() - started
        return rgb

    def encode_images(images):
        for image in images:
            rgb = to_rgb(image)
            started = time.perf_counter()
            cropped = crop_image(rgb, config['crop_type'], geometry)
            batch.append(preprocess(Image.fromarray(cropped)))
            stage['preprocess'] += time.perf_counter() - started
            if len(batch) == config['encoder_batch_size']:
                flush()

    def flush():
        nonlocal encoded, batches, next_progress
        if not batch:
            return
        started = time.perf_counter()
        images = torch.stack(batch)
        batch.clear()
        stage['preprocess'] += time.perf_counter() - started
        synchronize(device)
        started = time.perf_counter()
        images = images.to(device=device, dtype=clip_dtype)
        synchronize(device)
        stage['clip_input_transfer'] += time.perf_counter() - started
        started = time.perf_counter()
        with torch.no_grad():
            features = encoder.encode_image(images)
        synchronize(device)
        stage['clip_forward'] += time.perf_counter() - started
        started = time.perf_counter()
        features = features.detach().float().cpu().numpy()
        stage['clip_output_transfer'] += time.perf_counter() - started
        started = time.perf_counter()
        if features.shape != (len(images), 512) or not np.isfinite(features).all():
            raise ValueError('CLIP must produce finite [image_count,512] features')
        # Batch/group divisibility preserves complete block membership.
        features = features.reshape(-1, group, 512).mean(axis=1)
        if config['feature_l2_normalize']:
            norm = np.linalg.norm(features, axis=1, keepdims=True)
            if np.any(norm == 0):
                raise ValueError('Cannot normalize zero CLIP features')
            features = features / norm
        chunks.append(features.astype(output_dtype))
        stage['feature_aggregation'] += time.perf_counter() - started
        encoded += len(images)
        batches += 1
        if encoded >= next_progress:
            print(f'  {path.name}: CLIP encoded {encoded} images', flush=True)
            next_progress = encoded + 1024

    started = time.perf_counter()
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    stage['decode'] += time.perf_counter() - started
    count = 0
    tail_frames_encoded = 0
    reader = None
    try:
        if not cap.isOpened():
            raise ValueError(f'Could not open video with the FFmpeg backend: {path}')
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        header_frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or not math.isfinite(header_frames) or header_frames < 1:
            raise ValueError('Video requires positive FPS and a usable header frame count')
        if abs(header_frames - round(header_frames)) > 1e-3:
            raise ValueError('Non-integer header frame count')
        backend = cap.getBackendName()
        if color == 'x86_swscale_emulation':
            # OpenCV supplies metadata only; frames come as bit-exact yuv420p planes.
            started = time.perf_counter()
            cap.release()
            reader = FfmpegYuvReader(path, width, height)
            backend = 'ffmpeg_cli_yuv420p'
            stage['decode'] += time.perf_counter() - started
        source = reader or cap
        while True:
            started = time.perf_counter()
            ok, frame = source.read()
            stage['decode'] += time.perf_counter() - started
            if not ok:
                break
            started = time.perf_counter()
            if config['temporal_mode'] == 'mean_block' or count % 16 == config['frame_offset']:
                block_images.append(frame)
                block_indices.append(count)
                block_pts.append(float('nan') if reader else float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000)
            count += 1
            complete = count % 16 == 0
            stage['selection'] += time.perf_counter() - started
            if complete:
                started = time.perf_counter()
                chosen_indices.extend(block_indices)
                chosen_pts.extend(block_pts)
                stage['selection'] += time.perf_counter() - started
                encode_images(block_images)
                block_images.clear()
                block_indices.clear()
                block_pts.clear()
        if count != round(header_frames):
            raise ValueError(f'Decode/header count mismatch ({count} != {round(header_frames)}); '
                             'check truncation or decode errors before evaluating')
        if (tail_policy == 'match_official_rows' and expected_tokens == count // 16 + 1
                and count % 16 and block_images):
            # Official ceil(N/16) cohort: one row exists for the incomplete tail block.
            chosen_indices.extend(block_indices)
            chosen_pts.extend(block_pts)
            tail_frames_encoded = len(block_images)
            encode_images(block_images)
        flush()
    finally:
        started = time.perf_counter()
        if reader is not None:
            reader.release(check=sys.exc_info()[0] is None)
        else:
            cap.release()
        stage['decode'] += time.perf_counter() - started
    started = time.perf_counter()
    if not chunks:
        raise ValueError('Video contains no complete 16-frame block')
    features = np.concatenate(chunks)
    if not np.isfinite(features).all():
        raise ValueError('Stored feature precision caused non-finite values')
    indices = np.asarray(chosen_indices, dtype=np.int64)
    pts = np.asarray(chosen_pts, dtype=np.float64)
    nominal = indices / fps
    starts = np.arange(len(features), dtype=np.int64) * 16
    members = indices.reshape(-1, group)
    if (len(features) != count // 16 + (1 if tail_frames_encoded else 0) or encoded != len(indices)
            or np.any(np.diff(indices) <= 0) or members.shape[0] != len(features)):
        raise ValueError('Candidate/encoding/token order or count mismatch')
    pts_valid = bool(reader is None and np.isfinite(pts).all() and (len(pts) == 1 or np.all(np.diff(pts) > 0)))
    stage['feature_aggregation'] += time.perf_counter() - started
    metadata = {
        'decoded_frames': count, 'header_frames': int(round(header_frames)),
        'fps': fps, 'decoder_backend': backend,
        'discarded_tail_frames': 0 if tail_frames_encoded else count % 16,
        'tail_frames_encoded': tail_frames_encoded, 'tail_policy': tail_policy,
        'color_conversion': color, 'image_geometry': geometry,
        'candidate_frames': len(indices), 'selected_frames': len(indices),
        'selected_fraction_of_candidates': 1.0,
        'encoder_input_fraction_of_decoded_frames': encoded / count,
        'encoder_input_images': encoded, 'encoder_batches': batches,
        'encoder_parameter_dtype': str(clip_dtype), 'feature_dtype': str(features.dtype),
        'feature_shape': list(features.shape),
        'timestamp_source': 'zero-based decoded frame index / OpenCV FPS; CFR assumption',
        'decoder_pts_source': 'unavailable (ffmpeg yuv420p pipe)' if reader else 'OpenCV CAP_PROP_POS_MSEC',
        'cfr_and_annotation_coordinates_verified': False,
        'decoder_pts_strictly_increasing': pts_valid,
        'decoder_pts_max_nominal_error_seconds': float(np.max(np.abs(pts - nominal))) if pts_valid else None,
        'timing_seconds': stage,
    }
    arrays = {'selected_frame_indices': indices, 'selected_timestamps_seconds': nominal,
              'decoder_timestamps_seconds': pts, 'token_member_frame_indices': members,
              'token_start_frames': starts, 'token_end_frames': starts + 16}
    return features, arrays, metadata


def compare_features(raw, official):
    """Aligned numeric comparison; never choose an extraction recipe by metrics."""
    if raw.shape != official.shape:
        return {'shape_match': False, 'raw_shape': list(raw.shape), 'official_shape': list(official.shape)}
    raw, official = raw.astype(np.float64), official.astype(np.float64)
    if not np.isfinite(official).all():
        raise ValueError('Non-finite official features')
    raw_norm = np.linalg.norm(raw, axis=1)
    official_norm = np.linalg.norm(official, axis=1)
    if np.any(raw_norm == 0) or np.any(official_norm == 0):
        raise ValueError('Zero feature vector prevents cosine comparison')
    cosine = np.sum(raw * official, axis=1) / (raw_norm * official_norm)
    difference = raw - official
    return {'shape_match': True, 'cosine_mean': float(cosine.mean()),
            'cosine_min': float(cosine.min()),
            'mean_absolute_error': float(np.abs(difference).mean()),
            'max_absolute_error': float(np.abs(difference).max()),
            'relative_l2_error': float(np.linalg.norm(difference) / np.linalg.norm(official)),
            'raw_norm_mean': float(raw_norm.mean()), 'official_norm_mean': float(official_norm.mean())}


def event_coverage(indices, intervals):
    """Post-inference analysis only; intervals retain official half-open endpoints."""
    covered = 0
    for start, end in intervals:
        if start < 0 or end <= start:
            raise ValueError('Invalid annotation interval for coverage analysis')
        position = np.searchsorted(indices, start)
        covered += int(position < len(indices) and indices[position] < end)
    return {'covered_events': covered, 'total_events': len(intervals)}
