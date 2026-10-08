# VAD Frame Sampling

Pretrained VadCLIP을 고정하고 frame sampling 방법에 따른 이상 탐지 성능과 계산 비용을 비교하는 프로젝트입니다. UCF-Crime을 먼저 검증한 뒤 XD-Violence로 확장합니다.

현재 UCF test 290개(anomaly 140, normal 150)의 official-feature full 평가가 사용자 MPS 실행에서 성공했습니다. Branch1 AUC 88.02%/Ano-AUC 70.23%, branch2 AUC 85.69%, 원본 fine-grained AVG 6.68%입니다. Raw-video reference 코드를 준비했으며 raw 성능·시간과 sampling 비교는 아직 미측정입니다. [실험 결과표](docs/EXPERIMENT_RESULTS.md), 자세한 상태는 [PROGRESS.md](PROGRESS.md), 연구 설계는 [연구 계획](VAD_Frame_Sampling_Research_Plan.md)을 참고하세요.

## 저장소 파일 구조

```text
├── VadCLIP/                            # 모델 소스, 공식 목록·GT, 원본 LICENSE
├── scripts/                            # 데이터 준비·입력 점검·평가 코드
├── configs/
│   ├── ucf_resources.json              # feature·checkpoint 경로 설정
│   └── ucf_raw_reference.json          # provisional raw 추출·batch·precision 설정
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
│           ├── features/                # raw run의 feature·선택 index/timestamp NPZ
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
| 영상별 score | 압축 NumPy `.npz`. `branch1_feature_scores`, `branch2_feature_scores`(각 `[T]`), `gt`(`[16*T]`)와 새 `branch2_class_feature_scores`(`[T,14]`), `class_labels`(문자열 14개)를 저장합니다. 기존 scalar-only run에는 새 key가 없습니다. |
| 전체 score | Full 성공 시 `frame_scores.npz`에 `branch1`, `branch2`를 저장합니다. 각각 전체 official-feature timeline 길이입니다. 사용자 최신 full run `ucf_full_20261007_010204`에 저장됩니다. |
| 가상환경·캐시 | 설치 환경과 생성 파일입니다. 코드는 `requirements.txt`로 환경을 준비하며 캐시는 공유하지 않습니다. |

작고 필요한 `VadCLIP/list/gt*.npy`와 CLIP tokenizer의 `.txt.gz`는 저장소에 포함합니다. Feature 폴더 전체와 영상·weights·score 배열은 제외하지만 모든 `.npy`/`.gz`를 일괄 제외하지는 않습니다.

### 함께 공유하는 metadata 형식

- Annotation `.txt`: `영상명 category start1 end1 start2 end2`. 예: `Abuse028_x264.mp4 Abuse 165 240 -1 -1`. `-1 -1`은 유효 구간 없음이며 원본 frame index 값을 보존합니다.
- `manifest.json`: 최상위 `generated_at`, `order`, `paths`, `annotation_convention`, `summary`, `videos`. `videos`에는 official CSV index, 영상명/category, video/source/archive 경로, feature 파일명, size/CRC/hash, width/height/FPS/duration/frame count, 원본 annotation 구간을 기록합니다.
- `manifest.csv`: 영상별 한 행. 주요 열은 `official_csv_index, filename, category, is_anomaly, video_path, source_path, source_archive, zip_entry, official_feature_filename, bytes, crc32, sha256, width, height, fps, fps_rational, duration_seconds, frame_count_header, intervals_raw`입니다.
- `splits/*.txt`: 한 줄에 `test/` 기준 상대 영상 경로 하나. 공식 feature CSV 순서를 유지합니다.
- Feature CSV: `path,label` 두 열. `outputs/setup/ucf_CLIP_rgbtest.local.csv`는 해당 컴퓨터에서 생성한 경로를 사용합니다.
- 점검/실행 JSON: 성공 여부, issues, 입력 hash, 장치와 실행 설정, 영상별 검사·metric 등을 기록합니다. Smoke는 metric을 계산하지 않으며 이전 smoke를 최신 코드의 검증으로 해석하지 않습니다.
- 실행 폴더명: `ucf_<mode>_<YYYYMMDD_HHMMSS>`입니다. 예: `ucf_full_20261006_234239`. 한국 시간 기준이며 이름이 겹치면 `_2`, `_3` 등을 붙입니다. 장치와 입력 종류는 `summary.json`에서 확인합니다.

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

전체 평가는 바로 실행할 수 있습니다. Smoke는 소규모 추론을 별도로 확인하고 싶을 때 선택합니다. Full은 기존 smoke 보고서의 존재·통과 여부·hash와 무관하게 실행하며 입력 audit와 checkpoint 점검은 유지합니다.

```sh
.venv/bin/python scripts/evaluate_ucf_official.py --mode full --device mps
```

선택적 소규모 점검:

```sh
.venv/bin/python scripts/evaluate_ucf_official.py --mode smoke --device mps
```

다른 장치를 쓰면 `--device cpu` 또는 `--device cuda`를 지정합니다. Smoke를 실행한 경우 success=true/7사례 통과, full에서는 영상 290개·평가 frame 1,109,888·두 branch AUC/AP를 확인합니다. 출력 위치와 평가 규칙은 [UCF_OFFICIAL_BASELINE.md](docs/UCF_OFFICIAL_BASELINE.md)를 참고하세요. Cached-feature 시간은 raw-video 효율 측정이 아닙니다. 이번 변경에서는 프로젝트 코드를 실행하지 않았습니다.

Full 평가의 `metrics.branch1`/`branch2`에는 `roc_auc`, `ap`, `ano_auc`(0~1)를 저장합니다. Ano-AUC는 manifest의 이상 영상 140개 전체 프레임을 대상으로 합니다. 이미 성공한 run에는 모델을 다시 돌리지 않고 저장 score로 Ano-AUC를 추가할 수 있습니다.

```sh
.venv/bin/python scripts/update_ucf_metrics.py outputs/baseline/ucf_full_20261007_010204
```

원래 run의 `summary.json`에 Ano-AUC와 계산 출처를 추가하고 기존 AUC/AP·추론 기록은 유지합니다. 기존 사용자 추가 계산 결과는 branch1 Ano-AUC 70.23%, branch2 65.52%로 확인했으며 Codex는 재계산하지 않았습니다.

갱신된 full evaluator는 모델을 다시 평가하고 `metrics.fine_grained.map_by_iou`(IoU 0.1~0.5의 mAP)와 `mean_map`(AVG)를 함께 계산합니다. 원본 `ucf_test.py`처럼 정상 영상을 포함한 290개/14개 클래스의 A-branch 점수를 변경하지 않은 공개 evaluator에 `excludeNormal=False`로 전달합니다. 원본 segment/class GT를 사용하며 값은 0~1입니다. 별도 metric 갱신 명령 없이 위 full 명령을 실행하면 됩니다. [평가 범위](docs/UCF_OFFICIAL_BASELINE.md#fine-grained-map의-평가-범위)를 참고하세요.

사용자 `ucf_full_20261007_010204`에서 원본 방식 mAP 11.72/7.83/6.40/4.53/2.93%, AVG 6.68%를 확인했습니다. 이전 140개/13개 클래스 AVG 7.46%는 다른 protocol의 과거 기록입니다. Codex는 저장 JSON만 읽었고 모델/metric 계산을 실행하지 않았습니다. `update_ucf_metrics.py`는 기존 Ano-AUC 추가 기능만 제공합니다.

## Raw-video reference와 CLIP 시간 측정

[raw evaluator](scripts/evaluate_ucf_raw.py)와 [설정](configs/ucf_raw_reference.json)을 추가했습니다. 원본 영상에서 feature를 직접 만들고 기존 고정 VadCLIP/metric으로 평가하며, CLIP forward와 raw end-to-end 시간을 따로 저장합니다. 비교 입력은 현재 성공 official run의 summary/score NPZ입니다. [실행·출력·검증 안내](docs/UCF_RAW_REFERENCE.md)를 참고하세요.

```sh
.venv/bin/python scripts/evaluate_ucf_raw.py --mode full --device mps
```

검증은 선택 사항입니다. `--mode smoke`는 짧은 anomaly·짧은 normal·긴 영상 사례를 확인하고, `--video Abuse028_x264.mp4`로 특정 영상만 지정할 수 있습니다. Full은 smoke 없이 직접 실행 가능합니다.

기본 추출은 완전한 16-frame 구간에서 offset 8 한 장, crop 5, CLIP float32, batch 32입니다. 이는 **공식 추출 recipe로 확인되지 않은 provisional 조건**이며 공개 feature 유사도·기존 score 차이·full metric 차이를 함께 기록합니다. 100%는 이 후보를 모두 사용한다는 뜻입니다. `--temporal-mode mean_block`은 구간의 16장 전체를 encode/평균하는 별도 가설이며 효율 조건을 섞어 비교하지 않습니다.

출력은 `outputs/baseline/ucf_raw_<mode>_<timestamp>/summary.json`, `features/*.npz`, `scores/*.npz` 및 full `frame_scores.npz`입니다. 시간·선택 frame index/timestamp·feature/score 대조와 Event coverage를 기록합니다. 전체 290개 full에서 두 branch AUC/AP/Ano-AUC와 원본 fine-grained mAP/AVG를 계산합니다. 프로젝트 코드·테스트·decoding·encoder/inference는 아직 실행하지 않았고, raw 재현/시간 값은 사용자 실행 후 확인해야 합니다. Uniform/Random은 reference 검증 후 추가합니다.

## 원본 코드

[nwpu-zxr/VadCLIP](https://github.com/nwpu-zxr/VadCLIP)의 기준 commit은 `c41067f07d252efcda18008bea367886070c33b0`입니다. 원본 [LICENSE](VadCLIP/LICENSE)·README와 출처를 보존하며, local 변경은 [LOCAL_CHANGES.md](VadCLIP/LOCAL_CHANGES.md)에 기록했습니다.
