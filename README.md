# VAD Frame Sampling — 별도 개선 버전

고정된 CLIP/VadCLIP에서 프레임 선택 방법만 바꾸는 추론 연구 코드입니다.
원본 저장소 commit `632dd952eaa917de2b7d89778527d05bb75e920b`를 별도 폴더에 복사해 정리했습니다.
원본 저장소, 데이터와 기존 결과는 수정하지 않았습니다.

**상태: 코드 작성·정적 검토 완료 / import·설치·테스트·추론·수치 재현은 미실행.**
저장소 AGENTS 규칙에 따라 실제 실행은 사용자가 수행합니다. 아래 명령도 작성 과정에서 실행하지 않았습니다.

## 구현 범위

| 기능 | 상태 |
|---|---|
| UCF official-feature 평가 | 공통 pipeline 연결, 실행 검증 필요 |
| UCF raw Reference 100% | 기존 계산 규칙을 유지하도록 연결, 새 환경 재현 필요 |
| Uniform / Random | CLIP 이전 선택 구현, 100/50/25/12.5% 설정 제공, 실행 검증 필요 |
| 점수 복원 | 원본 candidate 위치에 nearest/linear/previous 복원 후 고정 16-frame 평가 |
| 테스트 영상만으로 준비 | 원본 ZIP 없이 파일 SHA·이름·순서·GT 정렬 점검 경로 구현 |
| XD official-feature 평가 | dataset별 옵션·prompt·AP 평가 연결, 실행 검증 필요 |
| XD raw 추출 | **미지원**: 공식 추출 recipe와 tail/crop 조건 검증 후 추가 |
| Adaptive / Ours | 미구현 |

기존 보고 성능(UCF official AUC 88.02%, raw AUC 87.19%)은 과거 Mac 실행 결과입니다.
이번 개선 버전이나 CUDA 서버의 성능으로 간주하면 안 됩니다.

## 구조

```text
vadfs/
  datasets/{ucf,xd,base}.py      # 라벨, 목록, 데이터셋별 조건
  samplers/{reference,uniform,random}.py
  preparation.py               # 테스트 입력 검증, 로컬 immutable manifest
  extract.py                   # 전체 decode, 선택한 프레임만 CLIP encode
  vadclip.py                   # 고정 모델, 기존 window/mask 동작
  timeline.py                  # 선택 시점 -> 전체 reference grid -> 평가 frame
  evaluation.py                # 데이터셋 adapter와 기존 metric 연결
  metrics.py                   # 보존한 UCF 평가 구현
  pipeline.py                  # 공통 실행 순서
  runs.py, paths.py, color.py
scripts/
  run_experiment.py            # 새 실험의 공통 진입점
  setup/prepare_dataset.py     # ZIP 없는 테스트 입력 준비
  setup/                      # 기존 최초 ZIP/CRC 검증 경로도 보존
  analyze/aggregate_runs.py    # seed별 결과 집계
configs/
  datasets/                   # 공용 데이터·checkpoint·추출 조건
  experiments/                # 방법·budget·seed·protocol 초안
  local/                      # 컴퓨터별 경로, 예제 외 Git 제외
.local/setup/                 # 준비 실행별 결과, Git 제외
outputs/runs/                 # 새 실험 결과, Git 제외
VadCLIP/                      # upstream snapshot·LICENSE·변경 기록
requirements/                 # 공통 패키지와 장치별 PyTorch 설치 분리
sources/                      # 존재하면 읽기 전용 참조 자료
```

과거 setup 보고서와 run 요약은 `docs/reference_snapshot/outputs/`에 보존했습니다.
이 사본에는 원본 `.git` 이력, 영상, feature, checkpoint와 과거 점수 NPZ가 없습니다.
`SOURCE_SNAPSHOT.json`이 출처 commit을 기록합니다. Git 저장소를 새로 만들기 전에는 실행 보고서의
research commit이 null이고, source snapshot과 전체 코드 hash로 식별합니다.

## 빠른 시작 — Linux / Vast.ai

작업 폴더를 연 뒤 서버 터미널에서 수행합니다. 정상 동작하는 Vast PyTorch 템플릿은
그 torch/torchvision 조합을 유지하세요. 별도 clean 환경 설치는 [SETUP](docs/SETUP.md)에 있습니다.

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
cp configs/local/example.json configs/local/me.json
```

`configs/local/me.json`의 경로를 실제 공용 데이터 위치로 바꿉니다.
예제의 `/workspace/shared-data/...`는 설명용 경로입니다. Linux에는 `ffmpeg`가 필요합니다.
지원 방식은 **이 checkout에서 직접 실행 또는 editable 설치**이며 일반 wheel 단독 배포는 지원하지 않습니다.

### 1. 입력 준비

공식 test 순서는 저장소의 목록으로 고정합니다. UCF는 영상 290개를 한 폴더 또는 하위 폴더들에
보관해도 되며, 중복 이름은 조용히 선택하지 않고 오류로 보고합니다.

```bash
python scripts/setup/prepare_dataset.py --dataset configs/datasets/ucf.json --local configs/local/me.json --input both
```

이 명령은 영상·feature를 읽어 파일 식별값, 배열 형식과 GT 정렬을 확인합니다.
영상 decoding과 모델 추론은 하지 않습니다. 원본 ZIP은 필요하지 않습니다.
결과는 `.local/setup/ucf/<setup_id>/manifest.json`에 생성됩니다.
기본 실행기는 마지막 준비 결과를 사용하고, 명시적으로 고정하려면 `--setup-manifest`를 사용합니다.

공개 feature 없이 raw 영상만 준비할 때는 `--input raw`를 사용합니다.
UCF의 reference token 수와 dtype은 기존 audit에서 보존한 portable metadata를 사용합니다.
Official 평가에는 `--input official` 또는 `both`로 준비한 feature가 필요합니다.

### 2. 작은 검증과 official 재현

```bash
python -m unittest discover -s tests -v
python scripts/run_experiment.py --config configs/experiments/ucf_official.json --local configs/local/me.json --device cuda --mode smoke
python scripts/run_experiment.py --config configs/experiments/ucf_official.json --local configs/local/me.json --device cuda --mode full
```

첫 명령은 인위적인 입력으로 budget·좌표·tail·복원을 점검하는 테스트입니다.
실제 모델 계산 재현을 대신하지 않습니다. Full official의 primary AUC를 기존 0.8802와 비교하세요.

### 3. Raw 기준 실험

```bash
python scripts/run_experiment.py --config configs/experiments/ucf_reference_100.json --local configs/local/me.json --device cuda --mode smoke --save-features
python scripts/run_experiment.py --config configs/experiments/ucf_reference_100.json --local configs/local/me.json --device cuda --mode full
```

전체 raw run에서 출력되는 run 폴더를 이후 `--reference-run`으로 지정합니다.
기존 0.8719와 비교하되 장비 차이·허용 오차를 사전에 정하고 차이를 설명하세요.
새 구현의 수치 동일성은 아직 확인하지 않았습니다.

### 4. Sampling

먼저 `ucf_uniform_100.json`, `ucf_random_100.json`으로 100% 선택의 Reference 동등성을 확인합니다.
이후 아래 명령의 `실제_REFERENCE_RUN_폴더`를 앞 단계의 전체 raw run 경로로 교체합니다.

```bash
python scripts/run_experiment.py --config configs/experiments/ucf_uniform_25.json --local configs/local/me.json --device cuda --mode smoke
python scripts/run_experiment.py --config configs/experiments/ucf_uniform_25.json --local configs/local/me.json --device cuda --mode full --reference-run 실제_REFERENCE_RUN_폴더
python scripts/run_experiment.py --config configs/experiments/ucf_random_25.json --local configs/local/me.json --device cuda --mode full --seed 0 --reference-run 실제_REFERENCE_RUN_폴더
```

50%·12.5% 설정도 제공합니다. 12.5% 파일 이름은 `ucf_uniform_12p5.json` 형식입니다.
Random seed 0~4는 제안이며, 본 실험 전에 protocol과 함께 확정하세요.
기본 설정은 `protocol.status: draft`라 결과가 최종 benchmark 승인 상태로 표시되지 않습니다.
[PROTOCOL](docs/PROTOCOL.md)을 읽고 test 결과를 보기 전에 설정을 고정하세요.

### 5. XD official 확장

```bash
cp configs/local/xd.example.json configs/local/xd.me.json
python scripts/setup/prepare_dataset.py --dataset configs/datasets/xd.json --local configs/local/xd.me.json --input official
python scripts/run_experiment.py --config configs/experiments/xd_official.json --local configs/local/xd.me.json --device cuda --mode smoke
```

XD용 checkpoint와 test feature를 별도로 확보해야 합니다. XD는 모델 layer/window/prompt가 UCF와 다릅니다.
GT 정렬과 소규모 실행을 확인한 뒤 `--mode full`로 AP를 재현합니다.
XD raw를 실행하면 미지원 안내로 중단합니다. 추출 방식이 검증되었다고 가정하지 않습니다.

## 결과와 협업

- run 폴더마다 `summary.json`, `scores/*.npz`, full일 때 `frame_scores.npz`를 생성합니다.
- `--save-features`를 지정한 경우에만 selected feature를 추가 저장합니다. 프레임 이미지는 저장하지 않습니다.
- 선택 index·timestamp, 실제 K/N, 모델·sampler seed, 입력/소스 hash, 장비·버전을 기록합니다.
- 실패·중단 시 마지막 완료 영상까지의 요약을 남기며 기존 run은 덮어쓰지 않습니다. 자동 resume은 아직 없습니다.
- metric은 전체 **공식 평가 시간축**에서 계산합니다. 원본 decoded frame 수와 GT 시간축 차이는 기존 제한점으로 보존합니다.
- decoding·선택·CLIP·VAD·복원 비용을 측정합니다. 공식 feature 평가는 speedup의 근거가 아닙니다.
- 같은 비교 signature의 full raw 100% 결과만 speedup 기준으로 받습니다.
- 두 사람은 코드 checkout·local JSON·출력 폴더를 분리하고 영상·checkpoint만 공용으로 두세요.
- 처음 검증한 공용 Python 환경의 패키지 변경은 한 명이 관리하세요. 환경을 분리한다면 같은 설치 조건과 version 기록을 유지합니다.
- 최종 시간 측정 중에는 다른 GPU 작업·다운로드·압축 해제를 멈춥니다.

[설치와 준비](docs/SETUP.md) · [실험 규칙 초안](docs/PROTOCOL.md) · [변경 내역·검증](docs/REFACTOR.md)

## 기존 실행 경로

`scripts/evaluate_ucf_official.py`, `evaluate_ucf_raw.py`, `analyze_ucf_raw_cohorts.py`는 재현 진단용으로 남겨뒀습니다.
이 경로는 `.local/setup/ucf/`의 기존 형식 audit를 별도로 사용하며 새 `prepare_dataset.py` 산출물과 혼용하지 않습니다.
과거 raw 진단은 새 local official run을 `--official-run`으로 명시해야 합니다.
이전 사용법은 `docs/archive/SETUP_before_refactor.md`에 보존했고,
경로 차이는 [REFACTOR](docs/REFACTOR.md)에 적었습니다.
