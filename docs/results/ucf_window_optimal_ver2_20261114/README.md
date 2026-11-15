# UCF-Crime Window Optimal ver2 — 2026-11-14

원본 영상의 구간당 한 프레임을 선택하는 ver2 실험 결과다. 고정된 CLIP/VadCLIP을 사용했으며, 36회 실행 모두 290개 test 영상 전체 평가를 완료했다.

## Frame-level ROC-AUC (%)

Random-window는 seed 0–4의 평균 ± 표본 표준편차다. 나머지 방법은 조건당 한 번 실행했다.

| 방법 | 100% | 50% | 25% | 12.5% |
|---|---:|---:|---:|---:|
| Uniform | 87.19 | 87.01 | 86.51 | 85.81 |
| Random-window | 87.18 ± 0.04 | 87.11 ± 0.13 | 86.62 ± 0.16 | 85.89 ± 0.06 |
| D-optimal | 87.14 | 87.03 | 86.56 | 85.61 |
| A-optimal | 87.11 | 87.01 | 86.50 | 85.50 |
| E-optimal | 87.16 | 87.03 | 86.59 | 85.63 |

## 평가 조건과 기록

- Budget별 원본 16/32/64/128프레임 구간에서 각각 한 장을 선택한다.
- D/A/E는 저비용 thumbnail 특징으로 구간 제약 선택 후 선택된 프레임만 CLIP으로 추출한다.
- 실제 선택 위치는 달라도 score anchor는 구간 시작점이며, 기존 nearest 복원과 전체 annotation 시간축 평가를 유지한다.
- 100%에서도 구간 내부 선택이 달라질 수 있으므로 방법별 성능이 같아야 하는 실험이 아니다.
- 이번 표는 기존 cached CLIP optimal 결과와 다른 실험이다.
- CSV의 시간은 각 seed별 단일 실행의 처리 시간이다. 3회 시간 반복 결과나 확정된 speedup은 아니다.
- 기존 test 결과를 본 뒤 설계를 변경한 후속 탐색적 평가이며 통계적 유의성·최종 일반화 성능을 주장하지 않는다.

## 보존 파일

- [비교 CSV](comparison_ver2.csv): 원본 자동 집계 표.
- [집계 JSON](summary_ver2.json): 비교 signature와 원본 입력 summary 해시.
- [Sweep 완료 기록](sweep_ver2.json): complete=true, 완료된 36개 실행과 원본 해시.
- [실행 요약](runs/): 원본 summary에서 top-level videos 배열만 제거한 36개 JSON.

실행 요약 사본은 videos 배열 제거로 원본과 바이트가 다르다. Sweep/집계에 기록된 summary 해시는 서버의 원본 파일에 해당하며, 보존 전 36개 모두 일치함을 확인했다.

원본 상세 summary·선택 좌표·score NPZ는 `/workspace/hyeongjun/vad/outputs/runs/ucf_*_ver2_full_*/`에 남아 있다. 이 폴더의 보존본에는 영상별 상세 배열과 NPZ를 포함하지 않는다.

원본 비교 결과: `outputs/analyses/ucf_comparison_ver2_full_20261114_030924/`. 원본 sweep: `outputs/analyses/ucf_sweep_ver2_full_20261113_154059/sweep_ver2.json`.

이 커밋은 UCF 결과만 보존한다. ver2 구현·방법론은 아직 로컬 미커밋 상태이며 실행 당시 research_commit/dirty 및 package hash를 원문대로 유지했다. XD는 진행 중이므로 포함하지 않는다. 이번 보존 작업에서는 테스트·추론·재집계를 실행하지 않았다.

후속 코드 보존: UCF 실행에 필요한 공통 `_ver2` 코드·테스트, UCF 설정 20개와
[method_ver2.md](../../../method_ver2.md)를 결과 커밋 다음에 보존했다.
보존 전 UCF 36개 실행에 기록된 모든 package source hash와 현재 파일을 대조해 일치를 확인했다.
위의 미커밋 설명은 결과 전용 커밋 당시 상태이며, 과거 실행의 research_commit/dirty 값은 변경하지 않았다.
XD 전용 설정은 이번 코드 커밋에서 제외했다.
