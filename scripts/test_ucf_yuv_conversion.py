"""Test YUV420p -> RGB conversion hypotheses for official UCF CLIP features.

Extraction diagnostic only (no VAD scores, GT or sampler). H.264 decoding is
bit-exact, but the YUV->RGB conversion (libswscale SIMD path, rounding flags,
chroma upsampling) differs across CPU architectures and builds. This script
obtains frame 16*b in several ways and encodes each with the same fx/fy crop
geometry:

  opencv_direct          OpenCV CAP_FFMPEG BGR decode (current pipeline)
  ffmpeg_rgb_<flags>     ffmpeg CLI rgb24 rawvideo with named swscale options
  numpy_<model>          exact-float conversion of the bit-exact yuv420p planes
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from verify_ucf_extraction import index_features, file_hash  # noqa: E402
from ucf_color import yuv420p_to_rgb_x86  # noqa: E402
from search_ucf_affine_geometry import (DEFAULT_VIDEOS, REFERENCES, Encoder, decode_frames,  # noqa: E402
                                        render, rel_l2, cosine_mean)

FFMPEG_VARIANTS = {
    'ffmpeg_rgb_default': ([], []),
    'ffmpeg_rgb_cpuflags0': (['-cpuflags', '0'], []),
    'ffmpeg_rgb_accurate_rnd': ([], ['-sws_flags', 'bicubic+accurate_rnd']),
    'ffmpeg_rgb_full_chroma': ([], ['-sws_flags', 'bicubic+accurate_rnd+full_chroma_int']),
    'ffmpeg_rgb_bilinear_full_chroma': ([], ['-sws_flags', 'bilinear+accurate_rnd+full_chroma_int']),
    'ffmpeg_rgb_cpuflags0_bitexact': (['-cpuflags', '0'], ['-sws_flags', 'bicubic+bitexact']),
}


def ffmpeg_raw(video_path, frame_indices, width, height, pix_fmt, pre=(), post=()):
    expr = '+'.join(f'eq(n\\,{i})' for i in frame_indices)
    cmd = ['ffmpeg', '-v', 'error', '-nostdin', *pre, '-i', str(video_path), '-vf', f"select='{expr}'",
           '-fps_mode', 'passthrough', *post, '-f', 'rawvideo', '-pix_fmt', pix_fmt, '-']
    data = subprocess.run(cmd, check=True, capture_output=True).stdout
    per = width * height * 3 if pix_fmt == 'rgb24' else width * height * 3 // 2
    if len(data) != per * len(frame_indices):
        raise ValueError(f'{video_path.name}: unexpected {pix_fmt} byte count {len(data)}')
    return [data[i * per:(i + 1) * per] for i in range(len(frame_indices))], cmd


def yuv_planes(buf, width, height):
    a = np.frombuffer(buf, dtype=np.uint8)
    y = a[:width * height].reshape(height, width).astype(np.float64)
    u = a[width * height:width * height * 5 // 4].reshape(height // 2, width // 2).astype(np.float64)
    v = a[width * height * 5 // 4:].reshape(height // 2, width // 2).astype(np.float64)
    return y, u, v


def upsample(c, mode):
    if mode == 'nearest':
        return c.repeat(2, 0).repeat(2, 1)
    # Bilinear, MPEG-2/H.264 default chroma siting: left (co-sited horizontally), centre vertically.
    h, w = c.shape
    out_h = np.empty((h * 2, w))
    pad = np.vstack([c[:1], c, c[-1:]])
    out_h[0::2] = 0.25 * pad[0:-2] + 0.75 * pad[1:-1]
    out_h[1::2] = 0.75 * pad[1:-1] + 0.25 * pad[2:]
    out = np.empty((h * 2, w * 2))
    padw = np.hstack([out_h, out_h[:, -1:]])
    out[:, 0::2] = out_h
    out[:, 1::2] = 0.5 * padw[:, :-1] + 0.5 * padw[:, 1:]
    return out


def numpy_rgb(buf, width, height, chroma, matrix, rounding):
    y, u, v = yuv_planes(buf, width, height)
    u, v = upsample(u, chroma) - 128, upsample(v, chroma) - 128
    kr, kb = {'bt601': (0.299, 0.114), 'bt709': (0.2126, 0.0722)}[matrix]
    kg = 1 - kr - kb
    yy = (y - 16) * 255 / 219
    cs = 255 / 224
    r = yy + cs * 2 * (1 - kr) * v
    b = yy + cs * 2 * (1 - kb) * u
    g = yy - cs * 2 * (1 - kb) * kb / kg * u - cs * 2 * (1 - kr) * kr / kg * v
    rgb = np.stack([r, g, b], -1)
    rgb = np.floor(rgb) if rounding == 'floor' else np.rint(rgb)
    return np.clip(rgb, 0, 255).astype(np.uint8)



def numpy_x86_simd(buf, width, height):
    """Emulate libswscale x86 MMX/SSSE3 yuv420p->rgb; shared implementation in ucf_color."""
    return yuv420p_to_rgb_x86(buf, width, height)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--video', action='append')
    parser.add_argument('--blocks', type=int, nargs='+', default=[1, 4, 8, 16, 24, 32])
    parser.add_argument('--device', default='mps')
    parser.add_argument('--params', type=float, nargs=4, default=list(REFERENCES['fxfy_256_over_240']))
    args = parser.parse_args()

    stamp = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
    output = ROOT / 'outputs/setup' / f'ucf_yuv_conversion_{stamp}'
    output.mkdir(parents=True, exist_ok=False)
    resources = json.loads((ROOT / 'configs/ucf_resources.json').read_text())
    manifest = json.loads((ROOT / 'UCF-Crime/test/manifest.json').read_text())
    by_name = {v['filename']: v for v in manifest['videos']}
    names = args.video or DEFAULT_VIDEOS
    videos = [by_name[n] for n in names]
    feature_paths = index_features(ROOT / resources['features_root'], videos, 5)
    t0 = time.time()
    sources, officials, samples, commands = {'opencv_direct': []}, [], [], {}
    for video in videos:
        path = ROOT / 'UCF-Crime/test' / video['video_path']
        if file_hash(path) != video['sha256']:
            raise ValueError(f"{video['filename']}: SHA-256 mismatch")
        w, h = video['width'], video['height']
        official = np.load(feature_paths[video['filename']], allow_pickle=False)
        blocks = [b for b in args.blocks if b < official.shape[0]]
        idx = [16 * b for b in blocks]
        sources['opencv_direct'] += decode_frames(path, idx)
        officials += [official[b].astype(np.float32) for b in blocks]
        samples += [(video['filename'], b) for b in blocks]
        for name, (pre, post) in FFMPEG_VARIANTS.items():
            bufs, cmd = ffmpeg_raw(path, idx, w, h, 'rgb24', pre, post)
            commands[name] = cmd
            sources.setdefault(name, []).extend(np.frombuffer(b, np.uint8).reshape(h, w, 3) for b in bufs)
        yuv, cmd = ffmpeg_raw(path, idx, w, h, 'yuv420p')
        commands['yuv420p'] = cmd
        for chroma in ('nearest', 'bilinear'):
            for matrix in ('bt601', 'bt709'):
                for rounding in ('round', 'floor'):
                    key = f'numpy_{chroma}_{matrix}_{rounding}'
                    sources.setdefault(key, []).extend(numpy_rgb(b, w, h, chroma, matrix, rounding) for b in yuv)
        sources.setdefault('numpy_x86_simd_emulation', []).extend(numpy_x86_simd(b, w, h) for b in yuv)
    officials = np.stack(officials)
    encoder = Encoder(ROOT / resources['clip_checkpoint'], args.device)
    base = sources['opencv_direct']
    results = []
    for source, frames in sources.items():
        d = [a.astype(int) - b.astype(int) for a, b in zip(frames, base)]
        feats = encoder([render(f, tuple(args.params), True) for f in frames])
        results.append({'source': source, 'rel_l2': rel_l2(feats, officials), 'cos': cosine_mean(feats, officials),
                        'mean_abs_pixel_diff_vs_opencv': float(np.mean([np.abs(x).mean() for x in d])),
                        'max_abs_pixel_diff_vs_opencv': int(max(np.abs(x).max() for x in d)),
                        'mean_signed_diff_rgb_vs_opencv': [float(np.mean([x[..., c].mean() for x in d])) for c in range(3)]})
        r = results[-1]
        print(f"{source:34s} relL2={r['rel_l2']:.5f} cos={r['cos']:.6f} pix={r['mean_abs_pixel_diff_vs_opencv']:.3f} "
              f"max={r['max_abs_pixel_diff_vs_opencv']} signed={np.round(r['mean_signed_diff_rgb_vs_opencv'], 3)}", flush=True)
    results.sort(key=lambda r: r['rel_l2'])
    report = {'scope': 'colour-conversion diagnostic only; no VAD/GT/sampler', 'historical_recipe_confirmed': False,
              'settings': vars(args), 'samples': samples, 'commands': commands,
              'ffmpeg_version': subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True).stdout.split('\n')[0],
              'results': results, 'encoder_images': encoder.images, 'elapsed_seconds': time.time() - t0}
    (output / 'summary.json').write_text(json.dumps(report, indent=2, default=float) + '\n')
    print('output', output)


if __name__ == '__main__':
    main()
