# VAD Frame Sampling

Pretrained VadCLIP을 고정하고, frame sampling 방법만 바꿨을 때 이상 탐지 성능과 계산 비용이 어떻게 달라지는지 비교하는 프로젝트입니다. UCF-Crime에서 먼저 검증한 뒤 XD-Violence로 확장합니다. 연구 설계는 [연구 계획](VAD_Frame_Sampling_Research_Plan.md), 결과는 [실험 결과표](docs/EXPERIMENT_RESULTS.md)를 참고하세요.

## 현재 상태

UCF-Crime test 290개(anomaly 140, normal 150), 평가 frame 1,109,888개 기준입니다.

| 입력 | Run | Branch1 AUC | AP | Ano-AUC | Fine-grained AVG | 상태 |
|---|---|---:|---:|---:|---:|---|
| Official CLIP features | `ucf_full_20261007_010204` | 88.02% | 33.56% | 70.23% | 6.68% | 공식 재현 (논문 값과 일치) |
| Raw video, 직접 추출 | `ucf_raw_full_20261008_150142` | 87.19% | 31.81% | 69.14% | 6.83% | **Sampling 비교의 기준(reference)** |
| Uniform / Random / Adaptive / Ours | — | — | — | — | — | 미측정 |

- Raw reference는 공식 feature 대비 AUC −0.83pp입니다. 저장 점수로 민감도 분석을 한 결과, 이 차이는 전부 공식 feature를 재현하지 못한 가장 긴 float32 영상 4개 때문입니다(4개만 공식 점수로 바꾸면 +0.002pp). 나머지 float16 286개는 상대 L2 ≤ 0.005로 재현됩니다.
- 공용 코드는 `vadfs/` 패키지로 정리했습니다. 리팩토링한 코드로 다시 실행한 `ucf_full_20261008_163913`과 `ucf_raw_full_20261008_164022`는 위 두 run과 290개 전부 feature·score·metric이 bit 단위로 같습니다.
- 다음 작업: raw reference 위에 Uniform / Random sampler 구현(budget 100/50/25/12.5%).

## 실행

프로젝트 루트에서 실행합니다. 어느 위치에서 실행해도 프로젝트 루트는 스크립트 위치로 결정됩니다. MPS 기준 소요 시간은 official full 약 30초, raw full 약 25분입니다.

```sh
# Official feature 평가 (smoke: 7개 사례 실행 점검, full: 290개 + metric)
.venv/bin/python scripts/evaluate_ucf_official.py --mode full --device mps

# Raw-video reference: 원본 영상 → CLIP feature 추출 → VadCLIP → metric, 단계별 시간 측정
.venv/bin/python scripts/evaluate_ucf_raw.py --mode full --device mps

# Raw vs official 차이를 feature dtype cohort별로 분석 (저장 점수만 사용, 모델 실행 없음)
.venv/bin/python scripts/analyze_ucf_raw_cohorts.py --raw-run outputs/baseline/ucf_raw_full_20261008_150142
```

- `--device`는 `cpu` / `cuda` / `mps` 중 하나입니다. 기본값은 `cpu`입니다.
- Smoke는 선택 사항입니다. Full은 smoke 결과 없이 바로 실행할 수 있습니다.
- Raw smoke는 짧은 anomaly·짧은 normal·긴 영상 3개를 처리합니다. `--video Abuse028_x264.mp4`로 특정 영상을 지정할 수 있고 여러 번 쓸 수 있습니다.
- 추출 설정은 [configs/ucf_raw_reference.json](configs/ucf_raw_reference.json)에서 읽습니다. 다른 설정은 `--config`, 다른 official 비교 run은 `--official-run`으로 지정합니다.
- 실행 규칙과 출력의 상세 설명: [official 평가](docs/UCF_OFFICIAL_BASELINE.md), [raw reference](docs/UCF_RAW_REFERENCE.md).

## 저장소 구조

```text
├── vadfs/                              # 공용 연구 코드 (scripts에서 import)
│   ├── paths.py                        # 프로젝트 루트·경로 해석·SHA-256
│   ├── runs.py                         # run 폴더 생성·provenance·summary 저장
│   ├── ucf.py                          # UCF label·고정 입력 경로
│   ├── vadclip.py                      # VadCLIP 로딩·upstream 동일 inference·score 복원
│   ├── color.py                        # x86 libswscale YUV→RGB 재현
│   ├── extract.py                      # raw 영상 decode·후보 선택·crop·CLIP encode
│   └── metrics.py                      # AUC/AP/Ano-AUC·원본 fine-grained mAP·run 비교
├── scripts/
│   ├── evaluate_ucf_official.py        # official feature 평가
│   ├── evaluate_ucf_raw.py             # raw-video reference 평가·시간 측정
│   ├── analyze_ucf_raw_cohorts.py      # 저장 점수 기반 cohort 민감도 분석
│   └── setup/                          # 데이터 준비·자원/입력 점검 (완료된 일회성 단계)
├── configs/
│   ├── ucf_resources.json              # feature·checkpoint 경로
│   └── ucf_raw_reference.json          # 확정된 raw 추출 설정과 official 비교 run
├── docs/
│   ├── EXPERIMENT_RESULTS.md           # 결과표
│   ├── UCF_OFFICIAL_BASELINE.md        # official 평가 규칙·출력
│   ├── UCF_RAW_REFERENCE.md            # raw 추출 recipe·근거·시간 측정·출력
│   ├── UCF_RESOURCE_SETUP.md           # 자원 다운로드·배치
│   └── archive/                        # 추출 방식 탐색 기록 등 지난 문서
├── VadCLIP/                            # upstream 모델 소스, 공식 목록·GT, LICENSE
├── data/                               # 데이터·feature·checkpoint (저장소에는 작은 metadata만)
│   └── UCF-Crime/
│       ├── Temporal_Anomaly_Annotation_for_Testing_Videos.txt   # 원본 annotation
│       └── test/
│           ├── manifest.json, manifest.csv # 영상별 경로·metadata·annotation
│           ├── splits/{test,anomaly,normal}.txt
│           └── annotations/            # 원본 annotation 복사본
├── outputs/
│   ├── setup/                          # 입력 점검 보고서, local feature CSV, 추출 재구성 근거
│   └── baseline/<run_id>/summary.json  # 실행별 보고서
├── AGENTS.md                           # 작업 지침
├── VAD_Frame_Sampling_Research_Plan.md # 연구 계획
└── requirements.txt
```

`PROGRESS.md`(진행 기록)는 로컬 전용 파일이라 저장소에 포함하지 않습니다.

### 코드 흐름

```text
evaluate_ucf_raw.py
  └─ extract.extract_reference   영상 전체 decode → 16-frame 구간마다 후보 선택 → color/crop → CLIP encode
       └─ vadclip.infer_feature  [T,512] feature → 256-token window → branch1/branch2 token score
            └─ vadclip.restore_frames → metrics.compute_ucf_metrics / compute_official_fine_grained
```

Sampler는 `extract.extract_reference`의 후보 선택 단계(현재 `count % 16 == frame_offset`)에 들어갑니다. 따라서 CLIP encoding 이전에 프레임을 고릅니다. Upstream의 window·mask 동작은 알려진 이상 동작까지 그대로 유지합니다(`vadfs/vadclip.py` docstring 참고).

### 기준 run과 재현성

- 각 run의 `summary.json`에는 입력·checkpoint·소스 SHA-256(`input_sha256`), `vadfs` 전체 파일 hash(`package_sha256`), research commit, upstream VadCLIP commit이 기록됩니다.
- Raw 평가기는 모델·metric 소스 hash가 현재 코드와 같은 official run만 비교 대상으로 받습니다. 그래서 현재 설정의 `official_reference_run`은 리팩토링 코드로 다시 실행한 `ucf_full_20261008_163913`입니다(`010204`와 bit 동일).
- Raw reference를 확정할 때의 코드는 git tag `ucf-raw-reference-v1`에 보존했습니다. 정리하면서 제거한 추출 진단 스크립트도 `git show ucf-raw-reference-v1:scripts/<file>`로 볼 수 있습니다.

## 출력 형식

Run 폴더는 `outputs/baseline/` 아래에 생성됩니다. 이름은 official은 `ucf_<mode>_<YYYYMMDD_HHMMSS>`, raw는 `ucf_raw_<mode>_<YYYYMMDD_HHMMSS>`이며 한국 시간 기준입니다. 이름이 겹치면 `_2`, `_3`을 붙이고, 기존 run은 덮어쓰지 않습니다.

| 파일 | 내용 |
|---|---|
| `summary.json` | 설정, 입력/소스 hash, 장비·precision, 영상별 기록, 시간, `metrics`(branch별 `roc_auc`·`ap`·`ano_auc`, `fine_grained`; 값은 0~1). Raw는 공식 feature·점수 대조와 `official_comparison`(pp)도 기록합니다 |
| `scores/*.npz` | 영상별 `branch1_feature_scores`·`branch2_feature_scores` `[T]`, `branch2_class_feature_scores` `[T,14]`, `class_labels`, `gt` `[16T]` |
| `features/*.npz` (raw) | `features` `[T,512]`, `selected_frame_indices`, `selected_timestamps_seconds`, `decoder_timestamps_seconds`, `token_member_frame_indices`, `token_start_frames`, `token_end_frames` |
| `frame_scores.npz` (full) | 전체 평가 시간축의 `branch1`, `branch2` frame score |
| `cohort_sensitivity.json` | `analyze_ucf_raw_cohorts.py` 결과 (raw run 폴더에 저장) |

Git에는 `summary.json`과 `cohort_sensitivity.json`만 올라가고 NPZ 배열은 제외됩니다.

- Branch1은 `sigmoid(logits1)`, branch2는 `1 − softmax(logits2)[Normal]`입니다. UCF 주 지표는 branch1 frame-level ROC-AUC입니다.
- Ano-AUC는 이상 영상 140개의 전체 프레임(정상 구간 포함)을 모아 계산합니다.
- Fine-grained mAP는 원본 `ucf_test.py`처럼 정상 영상까지 포함한 290개·14개 클래스로 계산합니다. 원본 evaluator(`excludeNormal=False`)와 segment/class GT를 수정 없이 사용하고, branch2 점수로 계산합니다.
- Official feature 평가 시간(`cached_feature_loop_seconds`)은 decoding·CLIP encoding을 포함하지 않으므로 raw 효율이나 speedup 근거로 쓰지 않습니다.

## 데이터와 자원

### GitHub에서 제외한 대용량 파일

새 clone에는 들어 있지 않으므로 직접 확보해야 합니다. 다운로드 링크와 절차는 [UCF_RESOURCE_SETUP.md](docs/UCF_RESOURCE_SETUP.md)에 있습니다.

```text
├── data/
│   ├── UCF-Crime/
│   │   ├── Anomaly-Videos-Part-{1,2,3,4}.zip, Testing_Normal_Videos.zip   # 원본 영상 ZIP
│   │   ├── extracted/                      # 압축 해제한 원본 영상 1,100개
│   │   └── test/{Anomaly/<category>,Normal}/*.mp4   # extracted로 연결되는 상대 symlink 290개
│   ├── features/official/ucf/
│   │   ├── UCFClipFeatures.shared.zip      # 다운로드한 feature ZIP
│   │   └── UCFClipFeatures/<배포 폴더>/<video_stem>__<crop>.npy
│   └── checkpoints/
│       ├── clip/ViT-B-16.pt                # CLIP encoder
│       └── vadclip/model_ucf.pth           # UCF용 VadCLIP
├── outputs/baseline/<run_id>/{features,scores,frame_scores.npz}
├── .venv/
└── .local-backups/                         # 이전 Git 이력·정리 전 진단 출력 백업
```

| 자원 | 규칙 |
|---|---|
| Raw 영상 | `.mp4`. 공식 test 목록의 영상명과 일치해야 합니다. `test/`는 symlink이므로 `extracted/`도 함께 보존합니다. |
| Official CLIP features | `.npy` `[T,512]`. Test 290개 중 286개는 float16, 4개는 float32입니다. UCF test는 `__5`(중앙 crop 좌우 반전)를 사용하며, 공식 CSV의 바로 위 폴더명+파일명으로 선택합니다. |
| Checkpoints | `ViT-B-16.pt`(완성본, `.part` 아님), `model_ucf.pth`(strict loading). XD는 XD용 가중치를 따로 씁니다. 확인한 SHA-256은 `outputs/setup/ucf_resource_check.json`에 있습니다. |

`VadCLIP/list/gt*.npy`와 CLIP tokenizer `.txt.gz`는 작아서 저장소에 포함합니다.

### 자원 경로와 준비 단계

[configs/ucf_resources.json](configs/ucf_resources.json)의 경로는 프로젝트 루트 기준입니다.

```json
{
  "features_root": "data/features/official/ucf/UCFClipFeatures",
  "ucf_checkpoint": "data/checkpoints/vadclip/model_ucf.pth",
  "clip_checkpoint": "data/checkpoints/clip/ViT-B-16.pt"
}
```

준비·점검은 이 컴퓨터에서 이미 통과했습니다. 2026-11-05에 대용량 입력을 `data/` 아래로 옮겼으므로, 이 컴퓨터에서도 아래 두 점검과 official full을 한 번 다시 실행해야 평가기의 hash 대조를 통과합니다. 다른 컴퓨터에서는 자신의 입력 기준으로 manifest·symlink·local CSV·audit를 다시 만듭니다. 이 단계에는 영상 ZIP 5개, `extracted/` 영상, 원본 annotation, `ffprobe`가 필요합니다.

```sh
.venv/bin/python scripts/setup/prepare_ucf_test.py
.venv/bin/python scripts/setup/check_ucf_resources.py --write-local-list
.venv/bin/python scripts/setup/check_ucf_baseline_inputs.py --check-model
```

기존 CSV와 점검 보고서의 절대 경로는 실행 당시 컴퓨터의 경로입니다. 평가기는 프로젝트 폴더 이름이 바뀐 경로도 현재 루트로 다시 연결한 뒤 SHA-256으로 같은 파일인지 확인합니다.

### 공유 metadata 형식

- Annotation `.txt`: `영상명 category start1 end1 start2 end2` 형식입니다. 예: `Abuse028_x264.mp4 Abuse 165 240 -1 -1`. `-1 -1`은 구간 없음을 뜻하며, 원본 frame index를 그대로 둡니다.
- `manifest.json` / `manifest.csv`: 영상별 official CSV index, category, `is_anomaly`, 영상·원본·ZIP 경로, feature 파일명, 크기·CRC32·SHA-256, 해상도·FPS·길이·header frame 수, 원본 annotation 구간.
- `splits/*.txt`: 한 줄에 `test/` 기준 상대 영상 경로 하나이며, 공식 feature CSV 순서를 따릅니다.
- `outputs/setup/ucf_CLIP_rgbtest.local.csv`: `path,label` 형식의 이 컴퓨터용 feature 목록입니다.

## 실행 환경

기존 환경은 Python 3.10.0, PyTorch 2.8.0, macOS arm64(MPS)입니다. Raw 추출에는 `ffmpeg` CLI가 필요합니다. 새 환경 설치는 검증하지 않았습니다.

```sh
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## 원본 코드

[nwpu-zxr/VadCLIP](https://github.com/nwpu-zxr/VadCLIP)의 기준 commit은 `c41067f07d252efcda18008bea367886070c33b0`입니다. 원본 [LICENSE](VadCLIP/LICENSE)·README와 출처를 보존하며, local 변경은 [LOCAL_CHANGES.md](VadCLIP/LOCAL_CHANGES.md)에 기록했습니다.
