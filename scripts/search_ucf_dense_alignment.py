"""Dense local temporal alignment between official rows and every nearby raw frame.

Extraction diagnostic only (no VAD scores, GT or sampler). For each requested
official row r, every decoded frame in [16r - window, 16r + window] is encoded
with a fixed crop geometry and compared with official row r. The best-matching
frame offset d = frame - 16r is reported per row, revealing whether the row->frame
mapping is 16r everywhere or drifts (e.g. different decode count or seeking).
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from verify_ucf_extraction import index_features, file_hash  # noqa: E402
from search_ucf_affine_geometry import REFERENCES, Encoder, render, rel_l2  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', action='append', required=True)
    parser.add_argument('--rows', type=int, nargs='+', required=True)
    parser.add_argument('--window', type=int, default=48)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--params', type=float, nargs=4, default=list(REFERENCES['fxfy_256_over_240']),
                        metavar=('AX', 'BX', 'AY', 'BY'))
    args = parser.parse_args()

    stamp = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
    output = ROOT / 'outputs/setup' / f'ucf_dense_alignment_{stamp}'
    output.mkdir(parents=True, exist_ok=False)
    resources = json.loads((ROOT / 'configs/ucf_resources.json').read_text())
    manifest = json.loads((ROOT / 'UCF-Crime/test/manifest.json').read_text())
    by_name = {v['filename']: v for v in manifest['videos']}
    videos = [by_name[n] for n in args.video]
    feature_paths = index_features(ROOT / resources['features_root'], videos, 5)
    encoder = Encoder(ROOT / resources['clip_checkpoint'], args.device)
    t0 = time.time()
    report = {'scope': 'temporal alignment diagnostic only; no VAD/GT/sampler',
              'historical_recipe_confirmed': False, 'settings': vars(args), 'videos': {}}
    for video in videos:
        path = ROOT / 'UCF-Crime/test' / video['video_path']
        if file_hash(path) != video['sha256']:
            raise ValueError(f"{video['filename']}: SHA-256 mismatch")
        official = np.load(feature_paths[video['filename']], allow_pickle=False).astype(np.float32)
        rows = [r for r in args.rows if r < official.shape[0]]
        windows = {r: range(max(0, 16 * r - args.window), 16 * r + args.window + 1) for r in rows}
        wanted = sorted({i for w in windows.values() for i in w})
        last = wanted[-1]
        frames = {}
        cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
        index = 0
        wanted_set = set(wanted)
        while index <= last:
            ok, frame = cap.read()
            if not ok:
                break
            if index in wanted_set:
                frames[index] = render(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), tuple(args.params), True)
            index += 1
        cap.release()
        decoded = sorted(frames)
        feats = dict(zip(decoded, encoder([frames[i] for i in decoded])))
        per_row = []
        for r in rows:
            errors = {i - 16 * r: rel_l2(feats[i][None], official[r][None]) for i in windows[r] if i in feats}
            best = min(errors, key=errors.get)
            per_row.append({'row': r, 'best_offset': best, 'best_rel_l2': errors[best],
                            'rel_l2_at_0': errors.get(0), 'errors': errors})
            print(f"{video['filename']} row {r:5d}: best d={best:+d} relL2={errors[best]:.4f} "
                  f"(d=0: {errors.get(0, float('nan')):.4f})", flush=True)
        report['videos'][video['filename']] = {'official_rows': int(official.shape[0]),
                                               'decoded_prefix': index, 'rows': per_row}
    report['encoder_images'] = encoder.images
    report['elapsed_seconds'] = time.time() - t0
    (output / 'summary.json').write_text(json.dumps(report, indent=2, default=float) + '\n')
    print('output', output)


if __name__ == '__main__':
    main()
