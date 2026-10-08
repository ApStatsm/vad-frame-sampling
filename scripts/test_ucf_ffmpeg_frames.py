"""Test frame-dump hypotheses: ffmpeg image files (JPEG/PNG) instead of direct decode.

Extraction diagnostic only (no VAD scores, GT or sampler). For frame 16*b of each
requested block, the pixels come from (a) OpenCV direct decode, or (b) ffmpeg CLI
writing the frame to an image file which is read back with OpenCV or Pillow. Each
pixel source is then rendered with the same named crop geometries and encoded by
the local CLIP ViT-B/16 visual tower. Temporary images are written only under
the run's output directory.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from verify_ucf_extraction import index_features, file_hash  # noqa: E402
from search_ucf_affine_geometry import (DEFAULT_VIDEOS, REFERENCES, Encoder, decode_frames,  # noqa: E402
                                        render, rel_l2, cosine_mean)

# name -> (extension, extra ffmpeg output args)
DUMPS = {
    'ffmpeg_png': ('png', []),
    'ffmpeg_jpg_default': ('jpg', []),
    'ffmpeg_jpg_q1': ('jpg', ['-q:v', '1']),
    'ffmpeg_jpg_q2': ('jpg', ['-q:v', '2']),
    'ffmpeg_jpg_q5': ('jpg', ['-q:v', '5']),
    'ffmpeg_jpg_q10': ('jpg', ['-q:v', '10']),
}


def ffmpeg_dump(video_path, frame_indices, out_dir, ext, extra):
    out_dir.mkdir(parents=True, exist_ok=True)
    expr = '+'.join(f'eq(n\\,{i})' for i in frame_indices)
    cmd = ['ffmpeg', '-v', 'error', '-nostdin', '-i', str(video_path), '-vf', f"select='{expr}'",
           '-fps_mode', 'passthrough', *extra, str(out_dir / f'%04d.{ext}')]
    subprocess.run(cmd, check=True)
    files = sorted(out_dir.glob(f'*.{ext}'))
    if len(files) != len(frame_indices):
        raise ValueError(f'{video_path.name}: ffmpeg wrote {len(files)} files for {len(frame_indices)} frames')
    return files, cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', action='append')
    parser.add_argument('--blocks', type=int, nargs='+', default=[1, 4, 8, 16, 24, 32])
    parser.add_argument('--device', default='mps')
    parser.add_argument('--dumps', nargs='+', default=list(DUMPS), choices=list(DUMPS))
    parser.add_argument('--extra-params', type=float, nargs=4, action='append', default=[],
                        metavar=('AX', 'BX', 'AY', 'BY'), help='additional affine geometry to test')
    parser.add_argument('--keep-images', action='store_true')
    args = parser.parse_args()

    stamp = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
    output = ROOT / 'outputs/setup' / f'ucf_ffmpeg_frames_{stamp}'
    output.mkdir(parents=True, exist_ok=False)
    resources = json.loads((ROOT / 'configs/ucf_resources.json').read_text())
    manifest = json.loads((ROOT / 'UCF-Crime/test/manifest.json').read_text())
    by_name = {v['filename']: v for v in manifest['videos']}
    names = args.video or DEFAULT_VIDEOS
    videos = [by_name[n] for n in names]
    feature_paths = index_features(ROOT / resources['features_root'], videos, 5)
    t0 = time.time()

    geometries = dict(REFERENCES)
    for i, p in enumerate(args.extra_params):
        geometries[f'extra_{i}'] = tuple(p)
    sources = {'opencv_direct': []}
    officials, samples, commands = [], [], {}
    for video in videos:
        path = ROOT / 'UCF-Crime/test' / video['video_path']
        if file_hash(path) != video['sha256']:
            raise ValueError(f"{video['filename']}: SHA-256 mismatch")
        official = np.load(feature_paths[video['filename']], allow_pickle=False)
        blocks = [b for b in args.blocks if b < official.shape[0]]
        indices = [16 * b for b in blocks]
        sources['opencv_direct'] += decode_frames(path, indices)
        officials += [official[b].astype(np.float32) for b in blocks]
        samples += [(video['filename'], b) for b in blocks]
        for name in args.dumps:
            ext, extra = DUMPS[name]
            files, cmd = ffmpeg_dump(path, indices, output / 'images' / name / path.stem, ext, extra)
            commands[name] = cmd[:4] + ['<video>'] + cmd[5:-1] + ['<out>']
            sources.setdefault(f'{name}_cv2', []).extend(
                cv2.cvtColor(cv2.imread(str(f), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for f in files)
            if ext == 'jpg':
                sources.setdefault(f'{name}_pil', []).extend(
                    np.asarray(Image.open(f).convert('RGB')) for f in files)
    officials = np.stack(officials)
    encoder = Encoder(ROOT / resources['clip_checkpoint'], args.device)

    results = []
    for source, frames in sources.items():
        pixel_diff = float(np.mean([np.abs(a.astype(int) - b.astype(int)).mean()
                                    for a, b in zip(frames, sources['opencv_direct'])]))
        for gname, params in geometries.items():
            feats = encoder([render(f, params, True) for f in frames])
            per = [rel_l2(f[None], o[None]) for f, o in zip(feats, officials)]
            results.append({'source': source, 'geometry': gname, 'params': params,
                            'rel_l2': rel_l2(feats, officials), 'cos': cosine_mean(feats, officials),
                            'median_sample_rel_l2': float(np.median(per)),
                            'mean_abs_pixel_diff_vs_opencv': pixel_diff})
            print(f"{source:28s} {gname:20s} relL2={results[-1]['rel_l2']:.5f} "
                  f"cos={results[-1]['cos']:.6f} pix={pixel_diff:.3f}", flush=True)
    results.sort(key=lambda r: r['rel_l2'])
    report = {'scope': 'frame-source diagnostic only; no VAD/GT/sampler', 'historical_recipe_confirmed': False,
              'settings': vars(args), 'videos': names, 'samples': samples, 'ffmpeg_commands': commands,
              'ffmpeg_version': subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True).stdout.split('\n')[0],
              'opencv': cv2.__version__, 'results': results, 'encoder_images': encoder.images,
              'elapsed_seconds': time.time() - t0}
    (output / 'summary.json').write_text(json.dumps(report, indent=2, default=float) + '\n')
    if not args.keep_images:
        shutil.rmtree(output / 'images')
    print('output', output)


if __name__ == '__main__':
    main()
