> 이 문서는 원본 저장소의 과거 실험 기록입니다. 새 개선 버전의 실행 결과가 아닙니다.
> 과거 보고서 사본은 docs/reference_snapshot/outputs/에 있으며 새 run은 outputs/runs/에 생성됩니다.

# 실험 진행과 결과

최종 갱신: 2026-11-05. 실제로 실행한 실험만 기록한다. 논문 보고값이나 계획을 측정값처럼 적지 않는다. Raw 추출 방식의 근거는 [UCF_RAW_REFERENCE.md](UCF_RAW_REFERENCE.md)에 있다.

## 한눈에 보기

**UCF-Crime에서 baseline 재현과 sampling 비교의 기준(raw-video reference) 확정까지 끝났다. Sampling 실험(Uniform / Random / Adaptive / Ours)과 XD-Violence는 아직 시작하지 않았다.**

| 단계 (연구 계획 순서) | 상태 |
|---|---|
| 1. UCF-Crime test split·annotation·feature·checkpoint 확인 | 완료 |
| 2. Official feature + pretrained checkpoint로 baseline 재현 | 완료 — AUC 88.02%, 논문 값과 일치 |
| 3. Raw video에서 CLIP feature를 직접 뽑는 reference pipeline | 완료 — AUC 87.19%, sampling 비교의 기준으로 확정 |
| 4. Uniform / Random, budget별 성능·시간 | **다음 작업** |
| 5. XD-Violence baseline | 미착수 |
| 6. Adaptive baseline, failure analysis | 미착수 |
| 7. Ours 설계·최종 평가 | 미착수 |

| 입력 | Run | Branch1 AUC | AP | Ano-AUC | Fine-grained AVG |
|---|---|---:|---:|---:|---:|
| Official CLIP features | `ucf_full_20261007_010204` | 88.02% | 33.56% | 70.23% | 6.68% |
| Raw video, 직접 추출 | `ucf_raw_full_20261008_150142` | 87.19% | 31.80% | 69.14% | 6.83% |
| 차이 (raw − official) | | −0.83pp | −1.75pp | −1.09pp | +0.15pp |

## 고정한 실험 조건

연구 질문은 "pretrained VadCLIP을 고정했을 때, frame sampling만 바꾸면 성능과 계산 비용이 어떻게 달라지는가"다. 아래는 모든 run에서 같다.

| 항목 | 값 |
|---|---|
| 모델 | VadCLIP (upstream commit `c41067f`), 재학습 없음 |
| Checkpoint | `model_ucf.pth` (SHA-256 `d064aba0…`), CLIP ViT-B/16 (`5806e77c…`) |
| 데이터 | UCF-Crime 공식 test 290개 (anomaly 140, normal 150) |
| 평가 시간축 | 공식 GT 전체 1,109,888 frame. Sampling 후에도 전체 시간축에서 평가 |
| 주 지표 | Branch1(`sigmoid(logits1)`) frame-level ROC-AUC |
| 보조 지표 | AP, Ano-AUC(이상 영상 140개 전체 frame), fine-grained mAP(branch2, 290개·14개 클래스, 원본 evaluator) |
| Score 복원 | Token 점수를 16 frame씩 반복 (upstream 방식) |
| Seed / 장비 | 234 / macOS arm64, MPS |

## 실험 1. Official feature 재현

공개된 CLIP feature를 그대로 넣어 upstream VadCLIP 평가를 재현했다. Run `ucf_full_20261007_010204` ([summary.json](reference_snapshot/outputs/baseline/ucf_full_20261007_010204/summary.json)).

| Branch | AUC | AP | Ano-AUC |
|---|---:|---:|---:|
| Branch1 (C-branch) | 88.02% | 33.56% | 70.23% |
| Branch2 (A-branch) | 85.69% | 26.50% | 65.52% |

| Fine-grained (branch2) | mAP@0.1 | @0.2 | @0.3 | @0.4 | @0.5 | AVG |
|---|---:|---:|---:|---:|---:|---:|
| 원본 방식 (290개·14개 클래스) | 11.72 | 7.83 | 6.40 | 4.53 | 2.93 | 6.68 |
| 이전 anomaly-only 13개 클래스 (폐기) | 13.09 | 8.71 | 7.15 | 5.05 | 3.28 | 7.46 |

- 원본 방식은 논문 [Table 4](https://ojs.aaai.org/index.php/AAAI/article/download/28423/28826#page=6)와 소수점 둘째 자리까지 일치한다.
- 처음에는 normal을 뺀 13개 클래스로 계산해 7.46%가 나왔다. 원본 코드(`excludeNormal=False`)와 같은 방식으로 바로잡았으며, 7.46%는 다른 protocol의 값이라 성능 개선으로 해석하지 않는다.
- Upstream의 window 처리 이상 동작은 그대로 유지했다. 길이가 256의 배수여도 padded window를 하나 더 만들고, padding mask를 전달하지 않는다. 바꾸면 재현이 깨지기 때문이다.
- 실행 시간(`cached_feature_loop_seconds`) 27.9초는 decode·CLIP encoding을 포함하지 않으므로 효율 비교에 쓰지 않는다.

## 실험 2. Raw-video reference 구축

Sampling은 CLIP encoder에 넣기 **전에** 프레임을 골라야 효율 이득이 생긴다. 그래서 원본 영상에서 직접 feature를 뽑는 pipeline이 필요했다. 공개 코드에는 feature 추출 코드가 없어서 공식 feature와 수치를 맞춰 가며 추출 방식을 재구성했다.

### 확정한 추출 방식

| 항목 | 값 | 근거 |
|---|---|---|
| 시간 단위 | 16 frame마다 token 1개 | 저자 issue 답변 |
| 구간 내 선택 | 첫 프레임 1장 (offset 0) | 16장 평균은 시간축 오차 패턴이 맞지 않아 기각 |
| 색 변환 | x86 libswscale YUV→RGB를 numpy로 재현 | Mac OpenCV는 약 1 grey level 어두워 상대 L2 0.042 |
| Resize / crop | 짧은 변 256 비율 resize 후 중앙 crop을 좌우 반전 (crop 5) | 공개 crop.py의 340×256 고정 resize는 상대 L2 0.12 |
| 꼬리 처리 | 공식 feature 길이에 맞춤 (float16은 floor, float32는 ceil) | 공식 배열 길이만 사용, GT는 쓰지 않음 |
| Precision | CLIP float32, 저장은 공식 feature dtype | |

기각한 다른 가설(JPEG 저장, chroma 보간 방식, `floor((N−1)/16)` 꼬리 규칙 등)은 [UCF_RAW_REFERENCE.md](UCF_RAW_REFERENCE.md#재구성-근거와-기각한-가설-요약)에 정리했다.

### 결과

Run `ucf_raw_full_20261008_150142` (MPS, float32, seed 234, 290개 성공, [summary.json](reference_snapshot/outputs/baseline/ucf_raw_full_20261008_150142/summary.json)).

| Branch | AUC | AP | Ano-AUC | 공식 대비 AUC |
|---|---:|---:|---:|---:|
| Branch1 | 87.19% | 31.80% | 69.14% | −0.83pp |
| Branch2 | 84.90% | 25.22% | 64.48% | −0.80pp |

- Fine-grained(branch2): mAP@0.1~0.5 = 11.79 / 7.97 / 6.57 / 4.71 / 3.12, AVG 6.83%(공식 대비 +0.15pp).
- Feature 일치도: 공식 feature가 float16인 286개는 모두 상대 L2 ≤ 0.005다(중앙값 0.0019, 최대 0.0047). CPU와 MPS 사이의 차이와 같은 수준이고, token 점수 차이는 최대 0.008이다.
- 공식 feature가 float32인 가장 긴 4개 영상(Shooting032, Normal924/935/940)은 상대 L2 0.29~0.35로 재현하지 못했다. 이 4개가 평가 시간축의 24.7%를 차지한다.

### 차이의 원인 분석

저장된 점수만으로 민감도 분석을 했다(`scripts/analyze_ucf_raw_cohorts.py`, 모델은 실행하지 않음, 결과 `outputs/baseline/ucf_raw_full_20261008_150142/cohort_sensitivity.json`).

| 비교 | Branch1 AUC 차이 |
|---|---:|
| Raw 점수에서 float32 4개 구간만 공식 점수로 교체 | +0.002pp (공식과 사실상 같음) |
| 공식 점수에 raw 4개 구간을 넣음 | −0.836pp |
| 4개를 뺀 시간축에서 raw vs 공식 | +0.002pp |

→ −0.83pp는 **전부 float32 4개 영상 때문**이다. 같은 교체에서 공식 대비 AP는 +0.013pp, Ano-AUC는 +0.004pp다. 사전에 정한 허용 오차(|ΔAUC| ≤ 0.5pp)는 넘었지만, 이 4개의 미재현을 제한점으로 기록하고 raw reference를 sampling 비교의 기준으로 확정했다(2026-10-08). 앞으로의 sampling Δ는 87.19%를 기준으로 계산하고, 4개 영상도 빼지 않고 290개 전체로 평가한다.

## 실험 3. 재현성 검증

코드나 경로를 바꿀 때마다 다시 실행해 결과가 bit 단위로 같은지 확인했다.

| 날짜 | 변경 | 재실행 run | 결과 |
|---|---|---|---|
| 2026-10-08 | 코드를 `vadfs` 패키지로 리팩토링 | `ucf_full_20261008_163913`, `ucf_raw_full_20261008_164022` | 290개 전부 feature·선택 index·score·metric bit 동일 |
| 2026-11-05 | 대용량 입력을 `data/`로 이동 | `ucf_full_20261105_222638` | Metric 동일, 점수 NPZ 291개 byte 동일 |

각 run의 `summary.json`에는 입력·checkpoint·소스 파일 SHA-256과 commit이 기록된다. Raw 평가기는 모델·metric 소스 hash가 같은 official run만 비교 대상으로 받는다. 현재 비교 기준 official run은 `ucf_full_20261105_222638`이다.

## 시간 측정 (raw reference, 1회)

| 단계 | 시간 (초) |
|---|---:|
| End-to-end (290개 전체) | 1465.8 |
| CLIP forward | 1221.1 |
| x86 색 변환 재현 (numpy) | 95.7 |
| Decode | 69.9 |
| 전처리 | 43.5 |
| VAD | 24.7 |

- CLIP encoder 입력 69,368장이다. CLIP forward가 전체의 약 83%를 차지하므로 프레임 수를 줄이면 효과가 클 것으로 보이지만, 아직 측정하지 않았다.
- 1회 측정이고, warm-up 2회는 제외했으며 OS cache는 통제하지 않았다. 색 변환 재현은 실제 x86 비용이 아니므로 효율 비교에서 따로 보고한다.
- 영상은 항상 전부 decode하므로, 프레임을 덜 뽑아도 decode 시간은 줄지 않는다고 본다.

## Raw-video sampling 비교표

`미측정`은 실제 측정값이 아직 없다는 뜻이다. Reference의 Δ=0과 Speedup=1.00×는 기준 정의이며 실측 성과가 아니다. Random은 최종 확정한 seed 반복의 평균±표준편차로 기록한다.

| Dataset | Sampling | Budget | AUC 또는 AP (%) | Δ (pp) | End-to-end (s) | Speedup | Event coverage (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| UCF-Crime | Full / Reference | 100% | 87.19 | 0 | 1465.8 (색 변환 재현 95.7 포함) | 1.00× | 100.0 (156/156) |
| UCF-Crime | Uniform | 25% | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 |
| UCF-Crime | Random | 25% | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 |
| UCF-Crime | Adaptive | 25% | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 |
| UCF-Crime | Ours | 25% | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 |
| XD-Violence | Full / Reference | 100% | 미측정 | 0 | 미측정 | 1.00× | 미측정 |
| XD-Violence | Uniform | 25% | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 |
| XD-Violence | Random | 25% | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 |
| XD-Violence | Adaptive | 25% | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 |
| XD-Violence | Ours | 25% | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 |

50%·12.5% budget도 같은 형식으로 행을 추가한다.

기입 규칙:

- AUC 또는 AP: UCF는 frame-level ROC-AUC, XD는 frame-level AP를 %로 기록한다. 같은 데이터셋의 모든 sampler에서 score branch와 후처리를 고정한다.
- Δ: `해당 sampling 성능(%) − raw reference 성능(%)`. 단위는 percentage point다.
- End-to-end: 같은 test split 전체를 처리한 총 시간(초). Decoding, sampler 점수·선택, 전처리/CLIP encoding, VAD, 원본 시간축 score 복원을 포함한다. 장비·precision·batch/I/O 조건과 warm-up·동기화·반복 방식을 먼저 고정한다.
- Speedup: `raw reference End-to-end / 해당 sampling End-to-end`. 25% budget을 4배 speedup으로 가정하지 않는다.
- Event coverage: `선택한 원본 frame이 하나 이상 포함된 anomaly interval 수 / 전체 anomaly interval 수 × 100`. GT는 분석에만 사용한다. 100%도 정해진 reference 후보를 모두 쓴다는 뜻일 뿐이므로 coverage=100%로 가정하지 않는다.

## 아직 하지 않은 것

**Sampling 구현 전에 정해야 할 것**

- 주 score branch를 branch1로 고정할지, 두 branch를 결합할지
- 일부 프레임만 골랐을 때 window·padding·위치 정보를 어떻게 처리할지
- 비균일 선택의 점수를 원래 시간축으로 복원하는 방법 (nearest, linear 등)
- Random의 반복 횟수와 seed 목록 (5개로 제안만 한 상태)
- 시간 측정 반복 횟수와 조건

**측정하지 않은 실험**

- UCF-Crime의 Uniform / Random / Adaptive / Ours, budget 100 / 50 / 25 / 12.5%
- XD-Violence 전체 (데이터·feature·checkpoint 확인 전)
- Ours 설계 (test와 분리된 개발 데이터 확보 방법도 미정)

## Run 목록

모든 run은 `outputs/baseline/<run ID>/`에 있다. Git에는 `summary.json`과 `cohort_sensitivity.json`만 올라가고, 점수 NPZ는 로컬에만 있다.

| Run ID | 입력 | 목적 | 결과 |
|---|---|---|---|
| `ucf_full_20261007_010204` | Official features | Baseline 재현 | AUC 88.02% |
| `ucf_raw_full_20261008_150142` | Raw video | Reference 확정 (+ `cohort_sensitivity.json`) | AUC 87.19% |
| `ucf_full_20261008_163913` | Official features | 리팩토링 검증 | 010204와 bit 동일 |
| `ucf_raw_full_20261008_164022` | Raw video | 리팩토링 검증 | 150142와 bit 동일 |
| `ucf_full_20261105_222638` | Official features | `data/` 이동 검증, **현재 비교 기준** | 163913과 bit 동일 |
