# Frame Sampling 성능 평가

2026-11-13 완료된 CLIP cached-feature 실험 결과다. 저장된 raw Reference CLIP feature에서 동일한 수의 후보를 선택하고, 고정된 VadCLIP으로 원본 annotation 시간축 전체를 평가했다.

수치는 **%**이며, Random-feature는 **seed 0–4의 평균 ± 표본 표준편차**다. Budget은 Reference 후보 수 대비 선택 비율이다.

## UCF-Crime — Frame-level ROC-AUC (%)

Test 영상 290개. Reference 100%: **87.19%**.

| 방법 | 100% | 50% | 25% | 12.5% |
|---|---:|---:|---:|---:|
| Uniform | 87.19 | 87.01 | 86.51 | 85.81 |
| Random-feature | 87.19 ± 0.00 | 87.06 ± 0.20 | 86.54 ± 0.26 | 85.50 ± 0.58 |
| D-optimal | 87.19 | 86.97 | 86.10 | 85.07 |
| A-optimal | 87.19 | 87.15 | 86.31 | 85.45 |
| E-optimal | 87.19 | **87.21** | **86.75** | **86.02** |

## XD-Violence — Frame-level AP (%)

Test 영상 800개. Reference 100%: **84.50%**.

| 방법 | 100% | 50% | 25% | 12.5% |
|---|---:|---:|---:|---:|
| Uniform | 84.50 | **83.34** | **81.67** | **78.83** |
| Random-feature | 84.50 ± 0.00 | 83.12 ± 0.08 | 81.16 ± 0.11 | 77.90 ± 0.47 |
| D-optimal | 84.50 | 83.11 | 80.93 | 77.68 |
| A-optimal | 84.50 | 82.94 | 80.98 | 77.76 |
| E-optimal | 84.50 | 82.97 | 81.18 | 78.67 |

## Optimal 방법의 Uniform 대비 성능 차이 (pp)

양수는 Uniform 대비 향상, 음수는 하락이다. 차이는 반올림 전 원본 수치로 계산했다.

| 데이터셋 | 방법 | 50% | 25% | 12.5% |
|---|---|---:|---:|---:|
| UCF-Crime | D-optimal | -0.04 | -0.41 | -0.74 |
| UCF-Crime | A-optimal | +0.14 | -0.20 | -0.35 |
| UCF-Crime | E-optimal | **+0.20** | **+0.24** | **+0.21** |
| XD-Violence | D-optimal | -0.23 | -0.75 | -1.14 |
| XD-Violence | A-optimal | -0.40 | -0.70 | -1.07 |
| XD-Violence | E-optimal | -0.38 | -0.49 | -0.16 |

## 평가 조건 및 해석

- 굵은 성능 수치는 각 budget에서 가장 높은 주 지표다. 통계적 유의성을 의미하지 않는다.
- Random-feature는 동일한 Reference 후보에서 선택한다. 기존 원본 영상의 구간 내 Random-window 실험과는 다르다.
- D/A/E-optimal은 선택에만 고정 무작위 투영을 사용하고, 선택된 원본 512D CLIP feature를 VadCLIP에 전달한다.
- 100%에서는 모든 방법이 Reference와 동일한 성능을 보였다.
- UCF-Crime에서는 E-optimal이 모든 축소 budget에서 Uniform보다 높은 ROC-AUC를 보였다.
- XD-Violence에서는 Uniform이 모든 축소 budget에서 가장 높은 AP를 보였다.
- 전체 CLIP feature를 미리 계산한 cache 기반 탐색 결과이며, raw-video 처리 속도 향상을 입증하는 실험은 아니다.
- 데이터셋별 37회, 총 74회 full 실행이 성공했다. 이 문서 작성 중 테스트·추론·재집계는 실행하지 않았다.

## 원본 결과 및 재현 안내

- [UCF 비교 CSV](docs/results/clip_optimal_20261113/ucf/comparison.csv)
- [XD 비교 CSV](docs/results/clip_optimal_20261113/xd/comparison.csv)
- [집계·실행 요약 및 provenance](docs/results/clip_optimal_20261113/README.md)
- [실행 명령과 필요한 입력·출력 경로](docs/OPTIMAL_RUNS.md)
