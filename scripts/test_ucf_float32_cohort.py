"""Frame-source hypotheses for the float32 (longest-video, ceil(N/16)) UCF cohort.

Extraction diagnostic only (no VAD scores, GT or sampler). For official row r the
candidates are frames 16r+d (d in --offsets). Pixel sources: bit-exact yuv420p
with the x86 swscale emulation (float16-cohort recipe), followed optionally by a
lossy image round trip (OpenCV/Pillow JPEG at several qualities, ffmpeg JPEG),
then a named crop geometry and CLIP ViT-B/16.
"""

import argparse
from datetime import datetime
import io
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'VadCLIP/src')]
from verify_ucf_extraction import index_features, file_hash  # noqa: E402
from search_ucf_affine_geometry import Encoder, opencv_fxfy, rel_l2, cosine_mean  # noqa: E402
from ucf_color import yuv420p_to_rgb_x86  # noqa: E402
from crop import image_crop  # noqa: E402

DEFAULT_VIDEOS = ['Shooting032_x264.mp4', 'Normal_Videos_940_x264.mp4', 'Normal_Videos_924_x264.mp4',
                  'Normal_Videos_935_x264.mp4']


def ffmpeg_select(path, indices, pix_fmt=None, image=None, extra=()):
    expr = '+'.join(f'eq(n\\,{i})' for i in indices)
    cmd = ['ffmpeg', '-v', 'error', '-nostdin', '-i', str(path), '-vf', f"select='{expr}'",
           '-fps_mode', 'passthrough', '-frames:v', str(len(indices)), *extra]
    if image:
        cmd += ['-f', 'image2pipe', '-c:v', image, '-']
    else:
        cmd += ['-f', 'rawvideo', '-pix_fmt', pix_fmt, '-']
    return subprocess.run(cmd, check=True, capture_output=True).stdout


def split_jpegs(data):
    frames, start = [], 0
    while True:
        begin = data.find(b'\xff\xd8', start)
        if begin < 0:
            return frames
        end = data.find(b'\xff\xd9', begin) + 2
        frames.append(cv2.cvtColor(cv2.imdecode(np.frombuffer(data[begin:end], np.uint8), cv2.IMREAD_COLOR),
                                   cv2.COLOR_BGR2RGB))
        start = end


def cv2_jpeg(rgb, quality):
    ok, enc = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def pil_jpeg(rgb, quality):
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format='JPEG', quality=quality)
    return np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert('RGB'))


GEOMETRIES = {
    'fxfy': lambda rgb: opencv_fxfy(rgb, True),
    'published_340x256': lambda rgb: image_crop(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 5),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', action='append')
    parser.add_argument('--rows', type=int, nargs='+', default=[1, 8, 32, 64, 128])
    parser.add_argument('--offsets', type=int, nargs='+', default=[-1, 0, 1])
    parser.add_argument('--device', default='mps')
    parser.add_argument('--precision', default='float32', choices=['float32', 'float16'])
    args = parser.parse_args()
    stamp = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
    output = ROOT / 'outputs/setup' / f'ucf_float32_cohort_{stamp}'
    output.mkdir(parents=True, exist_ok=False)
    resources = json.loads((ROOT / 'configs/ucf_resources.json').read_text())
    manifest = {v['filename']: v for v in json.loads((ROOT / 'UCF-Crime/test/manifest.json').read_text())['videos']}
    names = args.video or DEFAULT_VIDEOS
    paths = index_features(ROOT / resources['features_root'], [manifest[n] for n in names], 5)
    encoder = Encoder(ROOT / resources['clip_checkpoint'], args.device, args.precision)
    t0 = time.time()
    sources = {}
    official, keys = [], []
    for name in names:
        video = manifest[name]
        path = ROOT / 'UCF-Crime/test' / video['video_path']
        if file_hash(path) != video['sha256']:
            raise ValueError(f'{name}: SHA-256 mismatch')
        off = np.load(paths[name], allow_pickle=False).astype(np.float32)
        w, h = video['width'], video['height']
        rows = [r for r in args.rows if r < len(off)]
        idx = sorted({16 * r + d for r in rows for d in args.offsets if 0 <= 16 * r + d < video['frame_count_header']})
        raw = ffmpeg_select(path, idx, 'yuv420p')
        per = w * h * 3 // 2
        x86 = [yuv420p_to_rgb_x86(raw[i * per:(i + 1) * per], w, h) for i in range(len(idx))]
        frames = {'x86': x86}
        for q in (95, 90, 75, 50):
            frames[f'x86_cv2jpeg{q}'] = [cv2_jpeg(f, q) for f in x86]
        for q in (75, 95):
            frames[f'x86_piljpeg{q}'] = [pil_jpeg(f, q) for f in x86]
        for label, extra in (('ffmpegjpeg_default', ()), ('ffmpegjpeg_q2', ('-q:v', '2')), ('ffmpegjpeg_q5', ('-q:v', '5'))):
            frames[label] = split_jpegs(ffmpeg_select(path, idx, image='mjpeg', extra=extra))
            if len(frames[label]) != len(idx):
                raise ValueError(f'{name}: {label} produced {len(frames[label])} frames')
        position = {i: k for k, i in enumerate(idx)}
        for r in rows:
            for d in args.offsets:
                i = 16 * r + d
                if i not in position:
                    continue
                keys.append((name, r, d))
                official.append(off[r])
                for label, fs in frames.items():
                    sources.setdefault(label, []).append(fs[position[i]])
    official = np.stack(official)
    results = []
    for label, fs in sources.items():
        for gname, geometry in GEOMETRIES.items():
            feats = encoder([geometry(f) for f in fs])
            by_d = {}
            for d in args.offsets:
                mask = np.array([k[2] == d for k in keys])
                by_d[d] = rel_l2(feats[mask], official[mask])
            per_sample = [rel_l2(f[None], o[None]) for f, o in zip(feats, official)]
            results.append({'source': label, 'geometry': gname, 'rel_l2_by_offset': by_d,
                            'cos_d0': cosine_mean(feats[[k[2] == 0 for k in keys]], official[[k[2] == 0 for k in keys]]),
                            'per_sample': [dict(video=k[0], row=k[1], d=k[2], rel_l2=v) for k, v in zip(keys, per_sample)]})
            print(f"{label:22s} {gname:18s} " + ' '.join(f'd={d:+d}:{v:.4f}' for d, v in by_d.items()), flush=True)
    results.sort(key=lambda r: min(r['rel_l2_by_offset'].values()))
    report = {'scope': 'float32 cohort frame-source diagnostic; no VAD/GT/sampler', 'historical_recipe_confirmed': False,
              'settings': vars(args), 'results': results, 'encoder_images': encoder.images,
              'opencv': cv2.__version__, 'elapsed_seconds': time.time() - t0}
    (output / 'summary.json').write_text(json.dumps(report, indent=2, default=float) + '\n')
    print('output', output)


if __name__ == '__main__':
    main()
