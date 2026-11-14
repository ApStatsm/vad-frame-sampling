# XD-Violence Window Optimal ver2 — 2026-11-14

고정 CLIP/VadCLIP에서 원본 구간당 한 프레임을 선택한 D/A/E 성능표다.
세 방법 × 네 budget 총 12회 모두 성공했고, 각 실행이 800개 test 영상 전체를 평가했다.

## Frame-level AP (%)

| 방법 | 100% | 50% | 25% | 12.5% |
|---|---:|---:|---:|---:|
| D-optimal | 84.43 | 83.10 | 80.84 | 77.80 |
| A-optimal | 84.36 | 83.03 | 81.13 | 77.63 |
| E-optimal | **84.44** | **83.45** | **81.61** | **78.39** |

- 각 방법은 결정적 seed0으로 조건당 한 번 실행했다. 굵은 값은 해당 budget의 최고 AP이며 통계적 유의성을 뜻하지 않는다.
- Budget별 원본 16/32/64/128프레임 구간에서 한 장씩 선택한다. 저비용 thumbnail 표현으로 선택하고, 선택된 원본 프레임만 CLIP으로 추출한다.
- Score anchor는 구간 시작점이며 기존 nearest 복원과 전체 annotation 시간축 평가를 유지한다.
- GPU 0·1에 분할 실행했다. GPU별 환경은 집계 JSON에 보존했고, 병렬 실행 시간은 최종 speedup 근거로 사용하지 않는다.
- 기존 cached CLIP 선택과 다른 후속 탐색 실험이다. 이번 표에는 Uniform/Random을 포함하지 않는다.

## 보존 파일

- [자동 생성 Markdown 표](evaluation_optimal_ver2.md)
- [원본 비교 CSV](comparison_optimal_ver2.csv)
- [집계 JSON 및 환경·입력 해시](summary_optimal_ver2.json)
- [12회 완료 기록](sweep_optimal_ver2.json)
- [개별 실행 요약](runs/)

실행 요약은 서버 원본 summary의 top-level videos 배열만 제외했다. 바이트가 달라지므로
sweep/집계의 summary 해시는 서버의 원본 파일에 해당한다. 보존 전 원본 12개 해시 일치를 확인했다.
영상별 상세 기록·좌표·score NPZ는 `/workspace/hyeongjun/vad/outputs/runs/xd_{d,a,e}_optimal_*_ver2_full_*/`에 남는다.
전체 원본 NPZ·로그를 Git에 올린 백업은 아니다.

원본 비교 폴더는 `outputs/analyses/xd_optimal_only_ver2_full_20261114_215901/`,
원본 sweep 폴더는 `outputs/analyses/xd_optimal_split_ver2_full_20261114_105943/`다.

이번 커밋은 XD 결과만 보존한다. 공통 ver2 모델 경로와 UCF 코드는 앞서 보존했지만,
GPU 분할 실행기·optimal 전용 비교 도구·XD 전용 설정·방법론 추가분은 아직 로컬 미커밋 상태다.
실행 당시 source hash와 research_commit/dirty 기록은 원본 그대로 유지했다.
이번 보존 작업에서는 테스트·추론·재집계를 실행하지 않았다.
