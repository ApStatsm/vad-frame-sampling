"""Pixel comparison of OpenCV FFmpeg and PyAV sequential video decoding."""

from datetime import datetime
import json
from pathlib import Path

from verify_ucf_extraction import ROOT, choose_videos, file_hash, save_json


def main():
    import av
    import cv2
    import numpy as np

    output = ROOT / 'outputs/setup' / datetime.now().strftime('ucf_decoder_compare_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    report = {'success': False, 'frames': [16, 128, 512], 'videos': [],
              'av_version': av.__version__, 'opencv_version': cv2.__version__,
              'script_sha256': file_hash(Path(__file__))}
    save_json(output / 'summary.json', report)
    for video in choose_videos(json.loads((ROOT / 'UCF-Crime/test/manifest.json').read_text()), None):
        path = (ROOT / 'UCF-Crime/test' / video['video_path']).resolve()
        if file_hash(path) != video['sha256']:
            raise ValueError('Video differs from manifest')
        wanted = set(report['frames'])
        opencv, pyav = {}, {}
        cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
        try:
            if not cap.isOpened():
                raise ValueError('Cannot open OpenCV decoder')
            for index in range(max(wanted) + 1):
                ok, frame = cap.read()
                if not ok:
                    raise ValueError('OpenCV decode ended before requested frame')
                if index in wanted:
                    opencv[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            for index, frame in enumerate(container.decode(stream)):
                if index in wanted:
                    pyav[index] = frame.to_ndarray(format='rgb24')
                if index == max(wanted):
                    break
        if set(pyav) != wanted:
            raise ValueError('PyAV decode ended before requested frame')
        entries = []
        for index in sorted(wanted):
            a, b = opencv[index], pyav[index]
            if a.shape != b.shape:
                raise ValueError('Decoder shape disagreement')
            difference = np.abs(a.astype(np.int16) - b.astype(np.int16))
            entries.append({'index': index, 'shape': list(a.shape),
                            'pixel_equal_fraction': float(np.mean(a == b)),
                            'mean_abs_pixel_difference': float(difference.mean()),
                            'max_abs_pixel_difference': int(difference.max())})
        np.savez_compressed(output / f"{Path(video['filename']).stem}.npz",
                            indices=np.asarray(sorted(wanted)),
                            opencv_rgb=np.stack([opencv[i] for i in sorted(wanted)]),
                            pyav_rgb=np.stack([pyav[i] for i in sorted(wanted)]))
        report['videos'].append({'video': video['filename'], 'rows': entries})
        print(video['filename'], entries, flush=True)
        save_json(output / 'summary.json', report)
    report['success'] = True
    save_json(output / 'summary.json', report)
    print('Output:', output, flush=True)


if __name__ == '__main__':
    main()
