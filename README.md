# VAD Frame Sampling

Pretrained VadCLIP을 고정하고 frame sampling 방법에 따른 이상 탐지 성능과 계산 비용을 비교하는 프로젝트입니다. UCF-Crime을 먼저 검증한 뒤 XD-Violence로 확장합니다.

현재 UCF test 290개(anomaly 140, normal 150)의 자원·feature/GT 정렬·checkpoint loading 점검과 이전 버전 MPS smoke 7사례가 통과했습니다. 전체 AUC/AP 평가와 raw-video sampling 비교는 아직 진행하지 않았습니다. 자세한 상태는 [PROGRESS.md](PROGRESS.md), 연구 설계는 [연구 계획](VAD_Frame_Sampling_Research_Plan.md)을 참고하세요.

## 저장소 파일 구조

```text
├── VadCLIP/                            # 모델 소스, 공식 목록·GT, 원본 LICENSE
├── scripts/                            # 데이터 준비·입력 점검·평가 코드
├── configs/
│   └── ucf_resources.json              # feature·checkpoint 경로 설정
├── docs/                               # 자원 준비·baseline 실행 안내
├── UCF-Crime/
│   ├── Temporal_Anomaly_Annotation_for_Testing_Videos.txt
│   └── test/
│       ├── manifest.json               # 영상별 경로·metadata·annotation
│       ├── manifest.csv                # 동일 정보를 표 형식으로 저장
│       ├── splits/
│       │   ├── test.txt                # 전체 test 목록
│       │   ├── anomaly.txt             # anomaly 영상 목록
│       │   └── normal.txt              # normal 영상 목록
│       └── annotations/                # 원본 temporal annotation 복사본
├── outputs/
│   ├── setup/                          # 점검 JSON, local CSV, 로그
│   └── baseline/
│       └── <run_id>/
│           └── summary.json            # 실행별 보고서
├── .gitignore                          # 업로드 제외 규칙
├── README.md                           # 프로젝트·데이터 구조 안내
├── AGENTS.md                           # 프로젝트 작업 지침
├── PROGRESS.md                         # 진행 기록
├── VAD_Frame_Sampling_Research_Plan.md  # 연구 계획
└── requirements.txt                    # 실행 환경의 의존성 버전
```

작은 manifest·목록·점검 보고서는 현재 파일을 그대로 공유합니다. 기존 CSV와 보고서에 들어 있는 절대 경로는 실행 당시 컴퓨터의 경로입니다. 다른 컴퓨터에서는 자신의 경로로 설정하고 점검 파일을 다시 생성해야 합니다. 원본 공식 feature CSV의 배포자 경로도 변경하지 않고 보존합니다.

## GitHub에서 제외한 대용량 파일

아래 자원은 로컬 파일을 보존하면서 `.gitignore`로 업로드에서 제외합니다. 새 clone에는 들어 있지 않으므로 직접 확보해야 합니다.

```text
├── UCF-Crime/
│   ├── Anomaly-Videos-Part-{1,2,3,4}.zip    # 원본 anomaly 영상 ZIP
│   ├── Testing_Normal_Videos.zip           # normal test 영상 ZIP
│   ├── extracted/                          # 압축 해제한 원본 영상
│   │   ├── Anomaly-Videos-Part-1/
│   │   │   └── Abuse/
│   │   │       └── *.mp4
│   │   └── ...
│   └── test/
│       ├── Anomaly/
│       │   └── <category>/
│       │       └── *.mp4                   # extracted 원본으로 연결되는 symlink
│       └── Normal/
│           └── *.mp4                       # extracted 원본으로 연결되는 symlink
├── features/
│   └── official/
│       └── ucf/
│           ├── UCFClipFeatures.shared.zip  # 다운로드한 feature ZIP
│           └── UCFClipFeatures/
│               ├── Abuse/
│               │   └── Abuse028_x264__5.npy
│               ├── Testing_Normal_Videos_Anomaly/
│               │   └── *.npy               # 공식 test normal feature
│               └── <배포 폴더>/
│                   └── <video_stem>__<crop>.npy
├── checkpoints/
│   ├── clip/
│   │   └── ViT-B-16.pt                      # CLIP encoder 가중치
│   └── vadclip/
│       └── model_ucf.pth                    # UCF용 VadCLIP 가중치
├── outputs/
│   └── baseline/
│       └── <run_id>/
│           ├── scores/
│           │   └── *.npz                   # 영상·사례별 score 배열
│           └── frame_scores.npz            # full 평가의 전체 timeline score
├── .venv/                                  # 로컬 Python 가상환경
└── .local-backups/                          # 이전 Git 이력·정리 파일 백업
```

| 제외 자원 | 파일 형식과 대응 규칙 |
|---|---|
| Raw 영상 | `.mp4`. 공식 test 목록의 영상명과 일치해야 합니다. `test/`의 영상은 실제 복사본 대신 상대 symlink이므로 `extracted/`도 함께 보존합니다. |
| Official CLIP features | NumPy `.npy`. 검증한 test 290개는 `[T, 512]`, floating point이며 286개 float16/4개 float32입니다. 파일명은 `<video_stem>__5.npy`, 선택 기준은 공식 CSV의 바로 위 폴더명+파일명입니다. `__0`와 `__5`를 임의로 바꾸지 않습니다. |
| CLIP checkpoint | 완성된 PyTorch 모델 아카이브 `ViT-B-16.pt`. `.part`는 미완성 파일입니다. |
| VadCLIP checkpoint | UCF용 PyTorch checkpoint `model_ucf.pth`. 현재 모델의 state dict를 strict loading합니다. XD 평가에는 XD용 가중치를 별도로 사용합니다. |
| 영상별 score | 압축 NumPy `.npz`. 현재 evaluator가 `branch1_feature_scores`, `branch2_feature_scores`(각 `[T]`)와 `gt`(대응하는 `16*T` 길이)를 저장합니다. |
| 전체 score | Full 성공 시 `frame_scores.npz`에 `branch1`, `branch2`를 저장합니다. 각각 전체 official-feature timeline 길이입니다. 현재 full 결과는 아직 없습니다. |
| 가상환경·캐시 | 설치 환경과 생성 파일입니다. 코드는 `requirements.txt`로 환경을 준비하며 캐시는 공유하지 않습니다. |

작고 필요한 `VadCLIP/list/gt*.npy`와 CLIP tokenizer의 `.txt.gz`는 저장소에 포함합니다. Feature 폴더 전체와 영상·weights·score 배열은 제외하지만 모든 `.npy`/`.gz`를 일괄 제외하지는 않습니다.

### 함께 공유하는 metadata 형식

- Annotation `.txt`: `영상명 category start1 end1 start2 end2`. 예: `Abuse028_x264.mp4 Abuse 165 240 -1 -1`. `-1 -1`은 유효 구간 없음이며 원본 frame index 값을 보존합니다.
- `manifest.json`: 최상위 `generated_at`, `order`, `paths`, `annotation_convention`, `summary`, `videos`. `videos`에는 official CSV index, 영상명/category, video/source/archive 경로, feature 파일명, size/CRC/hash, width/height/FPS/duration/frame count, 원본 annotation 구간을 기록합니다.
- `manifest.csv`: 영상별 한 행. 주요 열은 `official_csv_index, filename, category, is_anomaly, video_path, source_path, source_archive, zip_entry, official_feature_filename, bytes, crc32, sha256, width, height, fps, fps_rational, duration_seconds, frame_count_header, intervals_raw`입니다.
- `splits/*.txt`: 한 줄에 `test/` 기준 상대 영상 경로 하나. 공식 feature CSV 순서를 유지합니다.
- Feature CSV: `path,label` 두 열. `outputs/setup/ucf_CLIP_rgbtest.local.csv`는 해당 컴퓨터에서 생성한 경로를 사용합니다.
- 점검/실행 JSON: 성공 여부, issues, 입력 hash, 장치와 실행 설정, 영상별 검사·metric 등을 기록합니다. Smoke는 metric을 계산하지 않으며 이전 smoke를 최신 코드의 검증으로 해석하지 않습니다.

### 자원 배치와 재생성

공식 feature/model 링크와 다운로드 안내는 [UCF_RESOURCE_SETUP.md](docs/UCF_RESOURCE_SETUP.md)에 있습니다. CLIP/UCF checkpoint의 기존 확인 SHA-256은 [PROGRESS.md](PROGRESS.md)에 기록했습니다.

[configs/ucf_resources.json](configs/ucf_resources.json)의 기본 경로:

```json
{
  "features_root": "features/official/ucf/UCFClipFeatures",
  "ucf_checkpoint": "checkpoints/vadclip/model_ucf.pth",
  "clip_checkpoint": "checkpoints/clip/ViT-B-16.pt"
}
```

세 경로는 프로젝트 루트 기준입니다. 실제 보관 위치가 다르면 설정을 맞춥니다. Raw test 구조를 새로 만들 때는 위 영상 ZIP 5개와 `extracted/` 영상, 원본 annotation을 준비한 뒤 사용자가 실행합니다. 영상 metadata 확인에는 `ffprobe`가 필요합니다.

```sh
.venv/bin/python scripts/prepare_ucf_test.py
.venv/bin/python scripts/check_ucf_resources.py --write-local-list
.venv/bin/python scripts/check_ucf_baseline_inputs.py --check-model
```

이미 같은 입력에서 준비·점검을 통과한 기존 로컬 작업자는 GitHub 공유만을 이유로 이 단계를 반복할 필요가 없습니다. 다른 컴퓨터에서는 manifest·symlink·local CSV·audit를 자신의 입력 기준으로 생성합니다.

## 실행 환경과 다음 평가

기존 환경은 Python 3.10.0, PyTorch 2.8.0, macOS arm64/MPS입니다. 새 환경 설치는 아직 검증하지 않았습니다.

```sh
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

현재 evaluator와 모델 파일은 이전 smoke 이후 버전 기록/변경 표시가 보완되어 hash가 다릅니다. 기존 입력은 보존하고 현재 코드의 smoke를 먼저 실행한 뒤 full 평가를 진행합니다.

```sh
.venv/bin/python scripts/evaluate_ucf_official.py --mode smoke --device mps
.venv/bin/python scripts/evaluate_ucf_official.py --mode full --device mps
```

다른 장치를 쓰면 smoke/full 모두 같은 `--device cpu` 또는 `--device cuda`를 지정합니다. Smoke의 success=true/7사례 통과, full의 영상 290개·평가 frame 1,109,888·두 branch AUC/AP를 확인합니다. 출력 위치와 평가 규칙은 [UCF_OFFICIAL_BASELINE.md](docs/UCF_OFFICIAL_BASELINE.md)를 참고하세요. Cached-feature 시간은 raw-video 효율 측정이 아닙니다. 이번 공유 정리에서는 프로젝트 코드를 실행하지 않았습니다.

## 원본 코드

[nwpu-zxr/VadCLIP](https://github.com/nwpu-zxr/VadCLIP)의 기준 commit은 `c41067f07d252efcda18008bea367886070c33b0`입니다. 원본 [LICENSE](VadCLIP/LICENSE)·README와 출처를 보존하며, local 변경은 [LOCAL_CHANGES.md](VadCLIP/LOCAL_CHANGES.md)에 기록했습니다.
