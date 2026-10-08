"""Continuous crop-geometry search against official UCF CLIP features.

Extraction diagnostic only (no VAD scores, GT or sampler). Frame 16*b of each
requested block is rendered with a separable affine sampling map

    x_src = ax * u + bx,  y_src = ay * v + by   (u, v = 0..223 crop pixels)

using float bilinear interpolation, rounded to uint8, then optionally flipped
(crop type 5) and encoded by the local CLIP ViT-B/16 visual tower. Parameters
are fitted on --fit-blocks and reported on --holdout-blocks. A lower error is a
geometric diagnostic, not proof of the historical recipe.

Reference parameter sets:
  published 340x256 resize + crop(16,58): ax=320/340, bx=ax*58.5-0.5, ay=0.9375, by=14.96875
  fx=fy=256/240 resize + crop(16,58):      ax=ay=0.9375, bx=54.34375, by=14.96875
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from verify_ucf_extraction import index_features, file_hash  # noqa: E402

DEFAULT_VIDEOS = ['Abuse028_x264.mp4', 'Normal_Videos_003_x264.mp4', 'Arrest001_x264.mp4',
                  'Arson007_x264.mp4', 'Burglary005_x264.mp4', 'RoadAccidents001_x264.mp4',
                  'Normal_Videos_006_x264.mp4']
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
FX = 256 / 240
REFERENCES = {
    'published_340x256': (320 / 340, 320 / 340 * 58.5 - 0.5, 0.9375, 0.9375 * 16.5 - 0.5),
    'fxfy_256_over_240': (0.9375, 0.9375 * 58.5 - 0.5, 0.9375, 0.9375 * 16.5 - 0.5),
}


def decode_frames(path, frame_indices):
    wanted = set(frame_indices)
    frames = {}
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    index = 0
    try:
        while len(frames) < len(wanted):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f'{path.name}: decode ended at {index}')
            if index in wanted:
                frames[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            index += 1
    finally:
        cap.release()
    return [frames[i] for i in frame_indices]


def decode_frames_x86(path, frame_indices, width, height):
    """Bit-exact yuv420p from ffmpeg, converted with the x86 swscale SIMD emulation."""
    from test_ucf_yuv_conversion import ffmpeg_raw, numpy_x86_simd
    bufs, _ = ffmpeg_raw(path, frame_indices, width, height, 'yuv420p')
    return [numpy_x86_simd(b, width, height) for b in bufs]


def render(rgb, params, flip):
    ax, bx, ay, by = params
    u = np.arange(224, dtype=np.float64)
    xs = np.clip(ax * u + bx, 0, rgb.shape[1] - 1)
    ys = np.clip(ay * u + by, 0, rgb.shape[0] - 1)
    x0 = np.minimum(np.floor(xs).astype(int), rgb.shape[1] - 2)
    y0 = np.minimum(np.floor(ys).astype(int), rgb.shape[0] - 2)
    wx = (xs - x0)[None, :, None]
    wy = (ys - y0)[:, None, None]
    src = rgb.astype(np.float64)
    top = src[y0][:, x0] * (1 - wx) + src[y0][:, x0 + 1] * wx
    bottom = src[y0 + 1][:, x0] * (1 - wx) + src[y0 + 1][:, x0 + 1] * wx
    out = np.clip(np.rint(top * (1 - wy) + bottom * wy), 0, 255).astype(np.uint8)
    return out[:, ::-1] if flip else out


def opencv_fxfy(rgb, flip):
    resized = cv2.resize(rgb, None, fx=FX, fy=FX, interpolation=cv2.INTER_LINEAR)
    out = resized[16:240, 58:282]
    return out[:, ::-1] if flip else out


class Encoder:
    def __init__(self, checkpoint, device, precision='float32'):
        sys.path.insert(0, str(ROOT / 'VadCLIP/src'))
        import clip
        from clip.model import convert_weights
        model, _ = clip.load(str(checkpoint), device='cpu', jit=False)
        visual = model.visual.float().eval().requires_grad_(False)
        if precision == 'float16':
            convert_weights(visual)  # upstream selective fp16 (as clip.load on CUDA)
        self.visual = visual.to(device)
        self.dtype = self.visual.conv1.weight.dtype
        self.device = device
        self.mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)
        self.images = 0

    def __call__(self, crops, batch=64):
        out = []
        for start in range(0, len(crops), batch):
            array = np.stack(crops[start:start + batch]).astype(np.float32) / 255
            tensor = torch.from_numpy(array).permute(0, 3, 1, 2).to(self.device)
            with torch.no_grad():
                out.append(self.visual(((tensor - self.mean) / self.std).to(self.dtype)).float().cpu().numpy())
            self.images += len(array)
        return np.concatenate(out)


def rel_l2(candidate, official):
    return float(np.sqrt(((candidate - official) ** 2).sum() / (official ** 2).sum()))


def cosine_mean(candidate, official):
    num = (candidate * official).sum(1)
    den = np.linalg.norm(candidate, axis=1) * np.linalg.norm(official, axis=1)
    return float((num / den).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--video', action='append')
    parser.add_argument('--fit-blocks', type=int, nargs='+', default=[1, 8, 32])
    parser.add_argument('--holdout-blocks', type=int, nargs='+', default=[4, 16, 24])
    parser.add_argument('--device', default='mps')
    parser.add_argument('--crop-type', type=int, default=5, choices=[0, 5])
    parser.add_argument('--pixel-source', default='x86', choices=['opencv', 'x86'])
    parser.add_argument('--precision', default='float32', choices=['float32', 'float16'])
    parser.add_argument('--grid-ax', type=float, nargs=3, default=[0.930, 0.945, 0.001])
    parser.add_argument('--grid-bx', type=float, nargs=3, default=[53.6, 55.2, 0.2])
    parser.add_argument('--grid-ay', type=float, nargs=3, default=[0.930, 0.945, 0.001])
    parser.add_argument('--grid-by', type=float, nargs=3, default=[14.2, 15.8, 0.2])
    parser.add_argument('--nelder-mead-iters', type=int, default=150)
    args = parser.parse_args()

    stamp = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
    output = ROOT / 'outputs/setup' / f'ucf_affine_search_{stamp}_{args.pixel_source}_{args.precision}'
    output.mkdir(parents=True, exist_ok=False)
    resources = json.loads((ROOT / 'configs/ucf_resources.json').read_text())
    manifest = json.loads((ROOT / 'UCF-Crime/test/manifest.json').read_text())
    by_name = {v['filename']: v for v in manifest['videos']}
    names = args.video or DEFAULT_VIDEOS
    videos = [by_name[n] for n in names]
    feature_paths = index_features(ROOT / resources['features_root'], videos, args.crop_type)
    flip = args.crop_type == 5
    t0 = time.time()

    sets = {'fit': [], 'holdout': []}
    for video in videos:
        official = np.load(feature_paths[video['filename']], allow_pickle=False)
        path = ROOT / 'UCF-Crime/test' / video['video_path']
        if file_hash(path) != video['sha256']:
            raise ValueError(f"{video['filename']}: SHA-256 mismatch")
        for split, blocks in (('fit', args.fit_blocks), ('holdout', args.holdout_blocks)):
            usable = [b for b in blocks if b < official.shape[0]]
            idx = [16 * b for b in usable]
            frames = (decode_frames_x86(path, idx, video['width'], video['height'])
                      if args.pixel_source == 'x86' else decode_frames(path, idx))
            for b, frame in zip(usable, frames):
                sets[split].append({'video': video['filename'], 'block': b, 'frame': frame,
                                    'official': official[b].astype(np.float32),
                                    'official_dtype': str(official.dtype)})
    encoder = Encoder(ROOT / resources['clip_checkpoint'], args.device, args.precision)

    def evaluate(params, split='fit'):
        items = sets[split]
        crops = [render(it['frame'], params, flip) for it in items]
        feats = encoder(crops)
        off = np.stack([it['official'] for it in items])
        return rel_l2(feats, off), cosine_mean(feats, off), feats

    report = {'scope': 'geometry diagnostic only; no VAD/GT/sampler', 'historical_recipe_confirmed': False,
              'settings': vars(args), 'videos': names,
              'samples': {k: [(it['video'], it['block']) for it in v] for k, v in sets.items()},
              'environment': {'torch': torch.__version__, 'opencv': cv2.__version__, 'device': args.device,
                              'precision': args.precision, 'pixel_source': args.pixel_source}}

    # Sanity: float renderer vs OpenCV fx/fy path at the reference parameters.
    ref = REFERENCES['fxfy_256_over_240']
    diffs = [np.abs(render(it['frame'], ref, flip).astype(int) - opencv_fxfy(it['frame'], flip).astype(int)).max()
             for it in sets['fit']]
    off_fit = np.stack([it['official'] for it in sets['fit']])
    cv_feats = encoder([opencv_fxfy(it['frame'], flip) for it in sets['fit']])
    report['sanity'] = {'max_pixel_diff_float_vs_opencv_resize_fxfy': int(max(diffs)),
                        'opencv_resize_fxfy_fit_rel_l2': rel_l2(cv_feats, off_fit)}
    report['references'] = {}
    for name, params in REFERENCES.items():
        fit = evaluate(params)
        hold = evaluate(params, 'holdout')
        report['references'][name] = {'params': params, 'fit_rel_l2': fit[0], 'fit_cos': fit[1],
                                      'holdout_rel_l2': hold[0], 'holdout_cos': hold[1]}
    print(json.dumps({'sanity': report['sanity'], 'references': report['references']}, indent=1), flush=True)

    def grid(lo_hi_step):
        lo, hi, step = lo_hi_step
        return np.round(np.arange(lo, hi + step / 2, step), 6)

    # Stage 1: horizontal grid with reference vertical mapping.
    best = list(ref)
    rows = []
    for ax in grid(args.grid_ax):
        for bx in grid(args.grid_bx):
            p = (ax, bx, best[2], best[3])
            rows.append((rel_l2_val := evaluate(p)[0], *p))
    rows.sort()
    report['grid_horizontal_top10'] = rows[:10]
    best = list(rows[0][1:])
    print('horizontal best', rows[0], flush=True)
    # Stage 2: vertical grid with best horizontal mapping.
    rows = []
    for ay in grid(args.grid_ay):
        for by in grid(args.grid_by):
            p = (best[0], best[1], ay, by)
            rows.append((evaluate(p)[0], *p))
    rows.sort()
    report['grid_vertical_top10'] = rows[:10]
    best = list(rows[0][1:])
    print('vertical best', rows[0], flush=True)
    # Stage 3: joint local refinement.
    from scipy.optimize import minimize
    result = minimize(lambda p: evaluate(tuple(p))[0], np.array(best), method='Nelder-Mead',
                      options={'maxiter': args.nelder_mead_iters, 'xatol': 1e-4, 'fatol': 1e-6,
                               'initial_simplex': np.array(best) + np.vstack([np.zeros(4), np.diag([0.001, 0.2, 0.001, 0.2])])})
    best = tuple(float(v) for v in result.x)
    fit = evaluate(best)
    hold = evaluate(best, 'holdout')
    per_sample = []
    for split in ('fit', 'holdout'):
        feats = evaluate(best, split)[2]
        for it, f in zip(sets[split], feats):
            per_sample.append({'split': split, 'video': it['video'], 'block': it['block'],
                               'rel_l2': rel_l2(f[None], it['official'][None]),
                               'cos': cosine_mean(f[None], it['official'][None])})
    implied = {'scale_x_src_per_out': best[0], 'scale_y_src_per_out': best[2],
               'implied_resized_width_for_320': 320 / best[0], 'implied_resized_height_for_240': 240 / best[2],
               'implied_crop_x_in_resized': (best[1] + 0.5) / best[0] - 0.5,
               'implied_crop_y_in_resized': (best[3] + 0.5) / best[2] - 0.5}
    report['best'] = {'params': best, 'fit_rel_l2': fit[0], 'fit_cos': fit[1], 'holdout_rel_l2': hold[0],
                      'holdout_cos': hold[1], 'nelder_mead_success': bool(result.success),
                      'implied_geometry': implied, 'per_sample': per_sample}
    report['encoder_images'] = encoder.images
    report['elapsed_seconds'] = time.time() - t0
    (output / 'summary.json').write_text(json.dumps(report, indent=2, default=float) + '\n')
    print(json.dumps({k: report['best'][k] for k in ('params', 'fit_rel_l2', 'holdout_rel_l2', 'implied_geometry')},
                     indent=1))
    print('output', output)


if __name__ == '__main__':
    main()
