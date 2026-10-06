# Official features — local input

Official UCF-Crime CLIP features are not included in Git. Download them using the links and instructions in [UCF_RESOURCE_SETUP.md](../docs/UCF_RESOURCE_SETUP.md), preserving the archive category/split layout and filenames.

Default extracted location: `official/ucf/UCFClipFeatures/`. The shared config resolves this from the project root. The test selector uses the official immediate parent directory plus basename; normal test files use `Testing_Normal_Videos_Anomaly/`. Do not merge similarly named normal directories or rename `__0` to `__5`.

Archives and arrays stay local. Feature-based metrics and timings must be distinguished from raw-video end-to-end measurements.
