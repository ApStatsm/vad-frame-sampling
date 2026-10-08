# UCF-Crime test set

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
