# UCF-Crime local data

Git includes the published test annotation text and small test lists, but no videos, archives, generated manifest or video symlinks.

To recreate the current data view, place the four `Anomaly-Videos-Part-{1,2,3,4}.zip` archives and `Testing_Normal_Videos.zip` here. Extract their videos under `extracted/`, preserving category folders and unique test video filenames. The published annotation must be named `Temporal_Anomaly_Annotation_for_Testing_Videos.txt` in this directory.

The user then runs from the project root:

```sh
.venv/bin/python scripts/prepare_ucf_test.py
```

This requires `ffprobe`. It checks the test split, CRC/size and metadata, creates relative symlinks under `test/`, and writes `test/manifest.{json,csv}` plus an `outputs/setup/` report. It reads every selected video for CRC/hash verification, so this is a real data-preparation step. Codex has not rerun it during GitHub preparation.

Keep `extracted/` with the test symlinks. Expected test count: 290 (140 anomaly, 150 normal), ordered by `VadCLIP/list/ucf_CLIP_rgbtest.csv`. Existing users retain their prepared data and need not recreate it solely for GitHub integration.

See [TEAM_SETUP.md](../docs/TEAM_SETUP.md) for feature/checkpoint preparation and subsequent checks.
