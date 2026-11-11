
from pathlib import Path
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, snapshot_download

REPO = "mteb/ucf-crime"
ROOT = Path("/workspace/vad/datasets/UCF-Crime")
ROOT.mkdir(parents=True, exist_ok=True)

# Test Split 메타데이터 다운로드
metadata = hf_hub_download(
    repo_id=REPO,
    filename="data/test-00000-of-00001.parquet",
    repo_type="dataset",
    local_dir=str(ROOT)
)

# Test 영상 290개 경로 추출
table = pq.read_table(metadata)
paths = table.column("video_path").to_pylist()
paths = list(dict.fromkeys(paths))

print(f"Test videos: {len(paths)}")
assert len(paths) == 290, "Test Split을 확인하세요."

# Test 영상만 선택적으로 다운로드
snapshot_download(
    repo_id=REPO,
    repo_type="dataset",
    allow_patterns=paths,
    local_dir=str(ROOT),
    max_workers=4
)

# 다운로드 결과 검증
missing = [p for p in paths if not (ROOT / p).is_file()]

print(f"Downloaded: {len(paths) - len(missing)}/290")
print(f"Missing: {len(missing)}")

if missing:
    print("Missing files:", missing[:10])
    raise SystemExit(1)

print("UCF-Crime Test download completed.")
