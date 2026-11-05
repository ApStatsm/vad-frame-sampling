# VAD frame sampling 실험 결과

확인일: 2026-10-07, raw reference 갱신 2026-10-08. 사용자 실행 보고서를 읽어 기록했다.

## Official-feature baseline 재현

이 결과는 미리 추출된 공식 CLIP features를 입력한 평가다. Raw-video reference 및 아래 sampling 비교표와 구분한다. Frame budget은 적용하지 않았다.

| Dataset | 입력 | Score branch | AUC (%) | AP (%) | Ano-AUC (%) | 평가 영상 수 | 평가 frame 수 |
|---|---|---|---:|---:|---:|---:|---:|
| UCF-Crime | Official CLIP features | Branch 1 / C-branch | 88.02 | 33.56 | 70.23 | 290 | 1,109,888 |
| UCF-Crime | Official CLIP features | Branch 2 / A-branch | 85.69 | 26.50 | 65.52 | 290 | 1,109,888 |

Ano-AUC는 이상 영상 140개의 전체 프레임(정상 구간 포함)을 모아 계산한다. 사용자가 기존 저장 score로 추가한 summary의 branch1=0.7023225741073721, branch2=0.6552197493487624를 읽어 %로 기록했다. 계산 경위는 [기존 결과에 Ano-AUC 추가](UCF_OFFICIAL_BASELINE.md#기존-결과에-ano-auc-추가-기록)를 참고한다.

사용자 새 full run `ucf_full_20261007_010204`에서 원본 290개/14개 클래스 fine-grained 재현을 확인했다. 기존 두 branch AUC/AP/Ano-AUC도 같은 값이다.

| Dataset / 입력 | Score branch | mAP@0.1 (%) | mAP@0.2 (%) | mAP@0.3 (%) | mAP@0.4 (%) | mAP@0.5 (%) | AVG (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| UCF-Crime / official features / original-code reproduction | Branch 2 / A-branch | 11.72 | 7.83 | 6.40 | 4.53 | 2.93 | 6.68 |
| UCF-Crime / earlier anomaly-only 13-class protocol | Branch 2 / A-branch | 13.09 | 8.71 | 7.15 | 5.05 | 3.28 | 7.46 |

원본 방식 결과는 논문 [Table 4](https://ojs.aaai.org/index.php/AAAI/article/download/28423/28826#page=6)의 보고값과 소수점 둘째 자리까지 일치한다. 이전 anomaly-only 값은 다른 protocol의 과거 기록이며 모델 성능 개선으로 해석하지 않는다. 이전 004524 폴더는 사용자 삭제 상태로 복구하지 않았다.

- 최신 Run: `ucf_full_20261007_010204`, MPS, seed 234, success=true, 정상 포함 290개/1,109,888 frame. 근거: [summary.json](../outputs/baseline/ucf_full_20261007_010204/summary.json).
- 원본 `excludeNormal=False` evaluator와 bundled segment/class GT를 그대로 사용했다. Normal의 label 'A'도 보존되어 결과에 기록된다.
- UCF 주 지표는 branch1 AUC 88.02%다. Official-feature 성능 재현이며 raw-video 재현 완료를 뜻하지 않는다.
- `cached_feature_loop_seconds`=27.88649941701442초. 모델 초기 로딩, 최종 metric 계산·전체 score 저장, decoding·CLIP visual encoding을 제외한다. Raw E2E/Speedup에 사용하지 않는다.
- Codex는 기존 JSON만 읽었고 score NPZ/GT 배열을 로딩하거나 metric을 다시 계산하지 않았다.

## Raw-video reference (sampling 비교의 기준)

2026-10-08 사용자 결정: UCF sampling 비교의 기준은 직접 추출한 raw-video reference다. 추출 방식과 근거는 [UCF_RAW_REFERENCE.md](UCF_RAW_REFERENCE.md)와 [EXTRACTION_DIAGNOSTIC_RUNS.md](archive/EXTRACTION_DIAGNOSTIC_RUNS.md)에 있다. 사용자 실행 run은 `ucf_raw_full_20261008_150142`(MPS, float32, seed 234, 290개 성공)이다. 근거: [summary.json](../outputs/baseline/ucf_raw_full_20261008_150142/summary.json).

| Dataset | 입력 | Score branch | AUC (%) | AP (%) | Ano-AUC (%) | 공식 feature 대비 AUC (pp) |
|---|---|---|---:|---:|---:|---:|
| UCF-Crime | Raw video, 재구성 추출 | Branch 1 / C-branch | 87.19 | 31.81 | 69.14 | −0.83 |
| UCF-Crime | Raw video, 재구성 추출 | Branch 2 / A-branch | 84.90 | 25.22 | 64.48 | −0.80 |

Fine-grained(branch 2): mAP@0.1~0.5 = 11.79 / 7.97 / 6.57 / 4.71 / 3.12, AVG 6.83%(공식 대비 +0.15pp).

- Feature 일치: 공식 float16 286개 모두 상대 L2 ≤ 0.005(중앙값 0.0019, 최대 0.0047), token 점수 차이 최대 0.008. 가장 긴 4개(공식 float32: Shooting032, Normal924/935/940)는 0.29~0.35로 재현하지 못했다. 이 4개는 frame-level 시간축의 24.7%다.
- AUC 차이의 원인은 4개 영상으로 **확인했다**(2026-10-08, `scripts/analyze_ucf_raw_cohorts.py`, 저장 점수만 사용, 결과 `outputs/baseline/ucf_raw_full_20261008_150142/cohort_sensitivity.json`). Raw 점수에서 float32 4개 구간만 공식 점수로 바꾸면 공식 대비 branch1 AUC +0.002pp, AP +0.013pp, Ano-AUC +0.004pp(branch2 +0.003/+0.008/+0.008)로 사실상 같다. 반대로 공식 점수에 raw 4개 구간을 넣으면 −0.836pp가 된다. 4개를 뺀 시간축에서도 raw와 공식의 차이는 +0.002pp다.
- 리팩토링 검증(2026-10-08): `vadfs` 패키지 코드로 재실행한 `ucf_raw_full_20261008_164022`(raw)와 `ucf_full_20261008_163913`(official)이 각각 150142, 010204와 290개 전부 feature·score·metric bit 동일. 수치 기준은 바뀌지 않는다.
- 입력 경로 이동 검증(2026-11-05): 대용량 입력을 `data/`로 옮기고 입력 점검을 다시 만든 뒤 실행한 `ucf_full_20261105_222638`(official, MPS)이 `163913`과 metric 동일, 점수 NPZ 291개 byte 동일. Raw 비교 기준 run을 이 run으로 바꿨다.
- 시간(1회): E2E 1465.8초, CLIP forward 1221.1초, x86 색 변환 재현 95.7초(numpy 에뮬레이션, 실제 x86 비용 아님), decode 69.9초, 전처리 43.5초, VAD 24.7초. Encoder 입력 69,368장. Warm-up 2회 제외, OS cache 미통제.

## Raw-video sampling 비교표

`미측정`은 실제 측정값이 아직 확인되지 않았다는 뜻이다. Reference의 Δ=0 및 Speedup=1.00×는 기준 정의이며 실측 성과가 아니다. Random은 최종 확정한 seed 반복의 평균±표준편차로 기록한다.

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

기입 규칙:

- AUC 또는 AP: UCF는 frame-level ROC-AUC, XD는 frame-level AP를 %로 기록한다. 같은 데이터셋의 모든 sampler에서 score branch와 후처리를 고정한다.
- Δ: `해당 sampling 성능(%) − raw reference 성능(%)`. 차이의 단위는 percentage point다.
- End-to-end: 같은 test split 전체를 처리한 총 시간을 초로 기록한다. Decoding, sampler 점수·선택, 전처리/CLIP encoding, VAD 및 원본 시간축 score 복원을 포함한다. 동일 장비·precision·batch/I/O 조건과 warm-up·동기화·반복 방식을 먼저 고정한다.
- Speedup: `raw reference End-to-end / 해당 sampling End-to-end`. 25% budget을 4배 speedup으로 가정하지 않는다.
- Event coverage: `선택한 원본 frame이 하나 이상 포함된 anomaly interval 수 / 전체 anomaly interval 수 × 100`. GT는 분석에만 사용한다. 100%도 정해진 reference 후보를 모두 사용한다는 뜻이므로 coverage=100%로 가정하지 않는다.

## 바로 다음 작업

1. (완료) UCF raw reference full 실행과 기준 고정(`ucf_raw_full_20261008_150142`). 민감도 분석으로 AUC 차이가 float32 4개 영상 때문임을 확인했다.
2. Annotation 좌표 규칙은 공식 GT 시간축을 그대로 사용하므로 별도 변경하지 않는다.
3. 공통 규칙과 주 score branch(branch1)를 고정한 뒤 Uniform/Random 25%를 먼저 비교한다. 연구 계획의 50%/12.5% budget도 이후 같은 형식으로 기록한다.
4. XD-Violence 자원과 데이터셋별 checkpoint·AP 평가를 검증한 뒤 확장한다. Adaptive/Ours는 계획 순서에 따라 개발하며 Ours는 별도 개발 데이터에서 설계·튜닝한다.
