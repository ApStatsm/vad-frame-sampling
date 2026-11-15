# Frame Sampling 성능 평가 — ver2

2026-11-14까지 완료된 raw-video 구간 선택 실험 결과다. 원본 영상에서 프레임을 선택한 뒤 CLIP feature를 새로 추출하고, 고정된 VadCLIP으로 원본 annotation 시간축 전체를 평가했다.

수치는 **%**이며, Random은 **구간 내 Random-window, seed 0–4의 평균 ± 표본 표준편차**다. Budget은 Reference token 수 대비 선택 입력 수 비율이다.

## UCF-Crime — Frame-level ROC-AUC (%)

Test 영상 290개. Uniform 100% (raw Reference와 같은 선택): **87.19%**.

| 방법 | 100% | 50% | 25% | 12.5% |
|---|---:|---:|---:|---:|
| Uniform | **87.19** | 87.01 | 86.51 | 85.81 |
| Random | 87.18 ± 0.04 | **87.11 ± 0.13** | **86.62 ± 0.16** | **85.89 ± 0.06** |
| D-optimal | 87.14 | 87.03 | 86.56 | 85.61 |
| A-optimal | 87.11 | 87.01 | 86.50 | 85.50 |
| E-optimal | 87.16 | 87.03 | 86.59 | 85.63 |

## XD-Violence — Frame-level AP (%)

Test 영상 800개. Uniform 100% (raw Reference와 같은 선택): **84.50%**.

| 방법 | 100% | 50% | 25% | 12.5% |
|---|---:|---:|---:|---:|
| Uniform | **84.50** | 83.34 | 81.67 | **78.83** |
| Random | 84.44 ± 0.14 | 83.41 ± 0.12 | **81.79 ± 0.20** | 78.37 ± 0.15 |
| D-optimal | 84.43 | 83.10 | 80.84 | 77.80 |
| A-optimal | 84.36 | 83.03 | 81.13 | 77.63 |
| E-optimal | 84.44 | **83.45** | 81.61 | 78.39 |

## Optimal 방법의 Uniform 대비 성능 차이 (pp)

양수는 Uniform 대비 향상, 음수는 하락이다. 차이는 반올림 전 원본 수치로 계산했다.

| 데이터셋 | 방법 | 50% | 25% | 12.5% |
|---|---|---:|---:|---:|
| UCF-Crime | D-optimal | +0.02 | +0.05 | -0.20 |
| UCF-Crime | A-optimal | +0.00 | -0.01 | -0.30 |
| UCF-Crime | E-optimal | +0.02 | +0.09 | -0.18 |
| XD-Violence | D-optimal | -0.24 | -0.84 | -1.03 |
| XD-Violence | A-optimal | -0.31 | -0.54 | -1.20 |
| XD-Violence | E-optimal | +0.11 | -0.06 | -0.44 |

## 평가 조건 및 해석

- 굵은 성능 수치는 각 budget에서 반올림 전 주 지표가 가장 높은 방법이다. 통계적 유의성을 의미하지 않는다.
- 100/50/25/12.5%에서 원본 16/32/64/128프레임 구간마다 한 장을 선택한다. Uniform은 구간 시작점, Random은 구간 내 균등 무작위 위치를 사용한다.
- D/A/E는 저해상도 색상·윤곽·움직임 특징과 고정 투영으로 영상 전체의 구간 제약 목적함수를 개선한다. 선택된 원본 프레임만 기존 CLIP으로 512D feature를 추출한다.
- CLIP/VadCLIP 가중치·prompt·데이터셋별 전처리·branch를 고정한다. 모든 방법은 구간 시작점을 score anchor로 사용하고 nearest 복원 후 전체 시간축을 평가한다.
- 100%는 원본 프레임 전체가 아니라 16프레임당 한 장이다. 구간 내 선택 위치가 달라지므로 방법별 성능이 같을 필요가 없다.
- UCF는 이번 ver2의 36회 완료 결과를 사용했다. Random은 모든 budget에서 5개 seed가 완료됐다.
- XD Uniform은 이번 ver2의 4회 완료 결과, D/A/E는 GPU 0·1 분할 실행의 12회 완료 결과다. XD Random은 기존에 완료된 20회(4 budgets × 5 seeds) 집계를 재사용했다. 중단된 ver2 Random의 일부 seed만 평균내거나 미완료 결과를 채우지 않았다.
- 기존 XD Random 20개 원본 JSON에서 데이터 identity, checkpoint/GT, prompt, 추출 recipe, 모델 옵션 및 모델·평가·복원·추출·색 변환·Random-window 소스 hash가 현재 ver2와 일치함을 확인했다. 실행 시점·GPU·일부 제어 코드는 서로 다르며, 과거 결과 재사용을 표시한 성능 비교다.
- UCF에서는 축소 budget 모두 Random 평균이 가장 높다. D/E는 50%·25%에서 Uniform보다 소폭 높지만, 모든 optimal이 Random 평균보다 낮다.
- XD에서는 50%에서 E-optimal이 가장 높고, 25%는 Random 평균, 100%·12.5%는 Uniform이 가장 높다. E의 50% Random 대비 차이는 약 +0.04pp로 작으며 유의한 개선이라고 단정하지 않는다.
- 기존 test 결과를 본 뒤 변경한 후속 탐색 실험이다. 최종 일반화 성능이나 raw-video speedup을 주장하지 않는다. 병렬·단일 실행 시간과 seed 반복을 최종 효율 측정으로 혼동하지 않는다.
- 문서 작성 중 프로젝트 코드·테스트·추론·metric 재평가는 실행하지 않았다. 저장된 CSV/JSON의 수치를 읽어 반올림·차이 계산 및 문서 편집만 수행했다.

## 원본 결과 및 재현 안내

- [UCF ver2 성능표 및 실행 요약](docs/results/ucf_window_optimal_ver2_20261114/README.md)
- [UCF 비교 CSV](docs/results/ucf_window_optimal_ver2_20261114/comparison_ver2.csv)
- [XD D/A/E 성능표 및 실행 요약](docs/results/xd_window_optimal_ver2_20261114/README.md)
- [XD optimal 비교 CSV](docs/results/xd_window_optimal_ver2_20261114/comparison_optimal_ver2.csv)
- [XD Uniform·기존 Random의 표 작성 근거](docs/results/evaluation_baselines_ver2_20261114/sources_ver2.json)
- [실험 방법·입력·실행 명령·출력 경로](method_ver2.md)
- [이전 cached CLIP 실험 결과](evaluation.md) — 이번 ver2와 별도 실험이다.
