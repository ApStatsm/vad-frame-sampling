"""Create a test-only view of existing UCF videos; preserve all source inputs."""

import argparse
from collections import Counter
import csv
from datetime import datetime
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zlib
from zipfile import ZipFile
from zoneinfo import ZoneInfo


def read_inputs(root):
    annotation = root / 'UCF-Crime/Temporal_Anomaly_Annotation_for_Testing_Videos.txt'
    annotations = {}
    for line in annotation.read_text().splitlines():
        if not line.strip():
            continue
        name, category, *values = line.split()
        assert name not in annotations, f'Duplicate annotation: {name}'
        assert Path(name).name == name, f'Unsafe filename: {name}'
        assert category.isalnum(), f'Unexpected category: {category}'
        raw = list(map(int, values))
        assert len(raw) % 2 == 0, f'Unpaired interval: {name}'
        intervals = []
        for start, end in zip(raw[::2], raw[1::2]):
            if (start, end) == (-1, -1):
                continue
            assert 0 <= start < end, f'Invalid interval: {name}: {start}, {end}'
            intervals.append([start, end])
        assert bool(intervals) == (category != 'Normal'), name
        annotations[name] = {'category': category, 'intervals_raw': intervals,
                             'annotation_values_raw': raw}
    split = root / 'VadCLIP/list/Anomaly_Test.txt'
    split_names = [Path(s.strip()).name for s in split.read_text().splitlines() if s.strip()]
    csv_path = root / 'VadCLIP/list/ucf_CLIP_rgbtest.csv'
    with csv_path.open(newline='') as handle:
        feature_rows = list(csv.DictReader(handle))
    names = [Path(r['path']).name.removesuffix('__5.npy') + '.mp4' for r in feature_rows]
    assert len(set(names)) == len(names) == 290
    assert len(set(split_names)) == len(split_names) == 290
    assert set(names) == set(split_names) == set(annotations)
    for name, row in zip(names, feature_rows):
        assert annotations[name]['category'] == row['label'], name
    sources = {}
    for source in (root / 'UCF-Crime/extracted').rglob('*.mp4'):
        if source.name in annotations:
            assert source.name not in sources, f'Duplicate extracted source: {source.name}'
            sources[source.name] = source
    assert set(sources) == set(names), f'Missing sources: {set(names) - set(sources)}'
    archive_entries = {}
    archives = [root / f'UCF-Crime/Anomaly-Videos-Part-{i}.zip' for i in range(1, 5)]
    archives.append(root / 'UCF-Crime/Testing_Normal_Videos.zip')
    for archive in archives:
        with ZipFile(archive) as handle:
            for entry in handle.infolist():
                name = Path(entry.filename).name
                if name in annotations:
                    assert name not in archive_entries, f'Duplicate archive entry: {name}'
                    archive_entries[name] = (archive, entry)
    assert set(archive_entries) == set(names)
    return annotation, split, csv_path, annotations, feature_rows, names, sources, archive_entries


def inspect_video(source, entry):
    assert source.stat().st_size == entry.file_size, f'ZIP size mismatch: {source}'
    checksum = 0
    sha = hashlib.sha256()
    with source.open('rb') as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            checksum = zlib.crc32(chunk, checksum)
            sha.update(chunk)
    assert checksum == entry.CRC, f'ZIP CRC mismatch: {source}'
    result = subprocess.run([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration:format=duration',
        '-of', 'json', str(source)
    ], capture_output=True, text=True, check=True, timeout=60)
    assert not result.stderr.strip(), f'ffprobe error: {source}: {result.stderr}'
    probe = json.loads(result.stdout)
    assert len(probe['streams']) == 1, f'Missing video stream: {source}'
    stream = probe['streams'][0]
    fps = float(Fraction(stream['avg_frame_rate']))
    duration = float(stream.get('duration', probe['format']['duration']))
    frame_count = int(stream['nb_frames']) if stream.get('nb_frames', 'N/A') != 'N/A' else None
    assert fps > 0 and duration > 0 and stream['width'] > 0 and stream['height'] > 0
    return {'bytes': entry.file_size, 'crc32': f'{checksum:08x}', 'sha256': sha.hexdigest(),
            'width': stream['width'], 'height': stream['height'], 'fps': fps,
            'fps_rational': stream['avg_frame_rate'], 'duration_seconds': duration,
            'frame_count_header': frame_count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    annotation, split, feature_csv, annotations, feature_rows, names, sources, entries = read_inputs(root)
    destination = root / 'UCF-Crime/test'
    planned = []
    for name in names:
        category = annotations[name]['category']
        rel = Path('Normal') / name if category == 'Normal' else Path('Anomaly') / category / name
        target = destination / rel
        if target.exists() or target.is_symlink():
            assert target.is_symlink() and target.resolve() == sources[name].resolve(), f'Conflicting target: {target}'
        planned.append(rel)
    existing = {p.relative_to(destination) for p in destination.rglob('*.mp4')}
    assert not existing - set(planned), f'Unexpected test videos: {existing - set(planned)}'
    summary = {'videos': len(names), 'normal': sum(a['category'] == 'Normal' for a in annotations.values()),
               'anomaly': sum(a['category'] != 'Normal' for a in annotations.values()),
               'selected_bytes': sum(entries[n][1].file_size for n in names),
               'destination': str(destination), 'storage': 'relative symlinks to existing extracted videos'}
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if args.dry_run:
        return
    assert shutil.which('ffprobe'), 'ffprobe is required'
    records = []
    warnings = []
    for index, (name, rel, feature_row) in enumerate(zip(names, planned, feature_rows)):
        source = sources[name]
        archive, entry = entries[name]
        metadata = inspect_video(source, entry)
        ann = annotations[name]
        frame_count = metadata['frame_count_header']
        if frame_count is not None and any(end > frame_count for _, end in ann['intervals_raw']):
            warnings.append({'filename': name, 'issue': 'annotation endpoint exceeds header frame count',
                             'frame_count_header': frame_count, 'intervals_raw': ann['intervals_raw']})
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_symlink():
            target.symlink_to(os.path.relpath(source, target.parent))
        records.append({'official_csv_index': index, 'filename': name,
                        'category': ann['category'], 'is_anomaly': ann['category'] != 'Normal',
                        'video_path': str(rel), 'source_path': str(source.relative_to(root)),
                        'source_archive': str(archive.relative_to(root)), 'zip_entry': entry.filename,
                        'official_feature_filename': Path(feature_row['path']).name,
                        **ann, **metadata})
        if (index + 1) % 10 == 0 or index + 1 == len(names):
            print(f'Verified {index + 1}/{len(names)}: {name}', flush=True)
    annotation_dir = destination / 'annotations'
    annotation_dir.mkdir(exist_ok=True)
    shutil.copyfile(annotation, annotation_dir / annotation.name)
    split_dir = destination / 'splits'
    split_dir.mkdir(exist_ok=True)
    for group in ['test', 'anomaly', 'normal']:
        selected = [r for r in records if group == 'test' or r['is_anomaly'] == (group == 'anomaly')]
        (split_dir / f'{group}.txt').write_text(''.join(r['video_path'] + '\n' for r in selected))
    doc = {'generated_at': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
           'order': 'VadCLIP/list/ucf_CLIP_rgbtest.csv',
           'paths': 'video_path relative to UCF-Crime/test; source_path and source_archive relative to project root',
           'annotation_convention': 'Raw published frame indices preserved; base and inclusive/exclusive convention not established here.',
           'summary': summary, 'videos': records}
    (destination / 'manifest.json').write_text(json.dumps(doc, ensure_ascii=False, indent=2) + '\n')
    columns = ['official_csv_index', 'filename', 'category', 'is_anomaly', 'video_path',
               'source_path', 'source_archive', 'zip_entry', 'official_feature_filename',
               'bytes', 'crc32', 'sha256', 'width', 'height', 'fps', 'fps_rational',
               'duration_seconds', 'frame_count_header', 'intervals_raw']
    with (destination / 'manifest.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for record in records:
            writer.writerow({**record, 'intervals_raw': json.dumps(record['intervals_raw'])})
    report = {'generated_at': doc['generated_at'], 'summary': summary,
              'verified_zip_crc_and_size': len(records), 'metadata_probe_passed': len(records),
              'frame_count_source': 'ffprobe nb_frames header; not a full decode count',
              'full_video_decode_performed': False, 'annotation_warnings': warnings,
              'input_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [annotation, split, feature_csv]},
              'preparation_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    report_path = root / 'outputs/setup/ucf_test_preparation_report.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (destination / 'README.md').write_text('''# UCF-Crime test set

Official test split only: 140 anomaly videos and 150 normal videos.

- `Anomaly/<category>/*.mp4`, `Normal/*.mp4`: relative symlinks to existing
  `../extracted/` source videos. Keep that source folder with this test folder.
- `manifest.json`, `manifest.csv`: one row per video, in the official VadCLIP
  test-feature CSV order; source archive, SHA-256, ZIP CRC, video metadata and
  unmodified annotation intervals are recorded.
- `splits/test.txt`, `splits/anomaly.txt`, `splits/normal.txt`: video paths
  relative to this directory. Subsets preserve the official CSV order.
- `annotations/`: byte-identical copy of the downloaded temporal annotation.
  Normal videos have no valid anomaly intervals (`-1 -1` sentinel pairs).

Frame interval indices retain the published values. Their base and endpoint
convention must be established when implementing timeline evaluation; no GT
arrays or sampling candidates are generated here. GT is for evaluation and
analysis only, not sampler input. Header frame counts are not full decode counts.

Rebuild/revalidate from the project root:

```sh
.venv/bin/python scripts/setup/prepare_ucf_test.py --dry-run
.venv/bin/python scripts/setup/prepare_ucf_test.py
```

Verification report: `../../outputs/setup/ucf_test_preparation_report.json`.
Original archives, extracted videos and upstream VadCLIP lists are preserved.
''')
    print(f'Completed: {len(records)} videos; {len(warnings)} annotation warnings; {report_path}', flush=True)


if __name__ == '__main__':
    main()
