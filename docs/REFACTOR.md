# 개선 내역과 검증 상태

작성: 2026-11-08 / 원본 commit: 632dd952eaa917de2b7d89778527d05bb75e920b.
별도 소스 폴더를 생성했으며 원본 Git 저장소·로컬 review 사본·대용량 입력은 변경하지 않았습니다.
구현은 정적으로 검토했고 프로젝트 코드를 import하거나 실행하지 않았습니다.

## 변경 대응표

| 기존 상태 | 개선 버전 |
|---|---|
| Mac에서 관측한 requirements 하나 | 공통 패키지, CUDA/CPU/Mac torch 설치, 과거 환경 기록 분리 |
| CLIP이 pkg_resources import | public packaging.version 직접 사용; 계산 변경 없음 |
| UCF 옵션을 공용 loader에서 고정 | dataset별 ucf_option/xd_option 선택 |
| Full 전용 선택을 encoder loop에 내장 | samplers/가 K와 원본 frame index를 결정하고 CLIP 전에 적용 |
| token score를 무조건 16회 반복 | 누락 reference token 복원 후 기존 16회 평가; Full 입력은 그대로 |
| raw runner에 모든 처리·진단 혼합 | pipeline/데이터 준비/모델/선택/복원/평가를 공통 진입점으로 연결 |
| 개인 절대 경로와 audit를 Git 추적 | local JSON, immutable .local setup, run별 outputs를 Git에서 제외 |
| 원본 ZIP이 없으면 준비 불가 | 기존 reference SHA/길이/순서로 test-only 영상 폴더 검증 |
| 새 clone 기본값이 과거 점수 run 참조 | 과거 run 기본 참조 제거, 현재 full raw Reference를 명시 |
| 과거 setup/run 파일과 live 출력 혼합 | 과거 사본은 docs/reference_snapshot, 새 실행은 outputs/runs |

## 유지한 계산과 제한점

원본 model.py, utils/tools.py, utils/layers.py, vadfs/metrics.py, vadfs/color.py와 주요 GT 파일은
원본 commit 사본과 바이트 비교했습니다. ReferenceImageEncoder, resize/crop·색 변환 계산,
모델 가중치 로딩·추론 및 upstream window/mask 규칙은 유지하도록 연결했습니다.
CLIP의 import 호환성과 모델 옵션 선택, 선택 index를 받는 extraction 분기 등은 수정되어
이전 run의 source hash와 다릅니다. **바이트 보존은 전체 pipeline의 수치 동등성 증명이 아닙니다.**

원본 286개 float16 feature의 일치와 긴 float32 4개 미재현, 공식 feature 시간축과 raw frame 좌표의
제한점, 순차 전체 decoding, 입력 hash 확인에 따른 warm cache 조건을 보존합니다.
새 CUDA/Python 환경에서 기존 AUC가 재현되는지는 아직 확인하지 않았습니다.

## 소스 구조

- datasets: 라벨, 공식 test 순서, checkpoint 옵션, 평가 길이.
- samplers: candidate 좌표·budget·seed만 입력. GT·모델 호출 없음.
- preparation: 파일 식별/feature/GT 정렬. decoder/모델을 실행하지 않는 사용자 준비 단계.
- extract: 기존 Full math + 선택된 프레임만 encoder에 전달하는 경로.
- timeline: reference candidate 좌표에서 누락 score 복원 후 고정 평가 block 확장.
- evaluation: UCF는 기존 dense evaluator, XD official은 branch별 ROC-AUC/AP와 원본 temporal mAP.
- pipeline: 공통 처리와 측정, checkpoint/source/data/environment 기록, raw Reference 비교.
- runs: 충돌 방지 run 폴더, atomic JSON 저장, 출처 기록.

기존 raw 진단은 official feature와 점수를 대조하는 별도 보존 경로입니다.
새 공통 raw 경로는 공개 feature·과거 score NPZ 없이도 보존한 UCF reference metadata로 실행 가능합니다.

## 새로 작성한 테스트 — 미실행

- test_samplers.py: 짧은 영상의 ceil budget, 순서·중복, Full 선택, 독립 seed.
- test_timeline.py: Full repeat 동등성, 한 관측의 전체 길이, nearest tie와 끝 구간, class 합.
- test_preparation.py: 경로와 독립적인 데이터 identity, GT 길이/순서와 중복 파일 거부.
- test_extraction_selection.py: 가짜 capture/encoder로 selected-only encoding, Full 분기 동등성, floor/ceil tail.
- test_reference_guard.py: smoke·다른 signature를 full raw 기준으로 사용하지 못하게 검사.

사용자 명령: `python -m unittest discover -s tests -v`.
테스트는 작성됐지만 실행하지 않았습니다. 실제 ffmpeg/OpenCV·CUDA·checkpoint 재현을 별도로 확인해야 합니다.

## 사용자가 확인할 검증 순서

1. 설치 환경의 import와 실제 GPU 연산. 기존에 성공한 torch 조합을 유지.
2. 위 테스트와 prepare_dataset의 입력 identity·GT 정렬 확인.
3. UCF official full의 AUC 0.8802 근처 재현과 raw full의 0.8719 근처 재현을 확인.
   허용 오차는 sampling을 보기 전에 정하고, 차이를 설명할 때까지 성능 개선으로 해석하지 않음.
4. 같은 짧은 영상에서 legacy raw와 새 Reference raw의 선택·feature·score를 대조.
   legacy raw에는 current local official run을 --official-run으로 넘겨야 함.
5. 새 Reference / Uniform 100 / Random 100의 선택과 feature/score/metric 동등성 확인.
6. 25%에서도 전체 평가 길이를 유지하는지 확인하고 두 seed 이상의 smoke로 기록 분리 점검.
7. protocol을 고정한 뒤 전체 budget과 선언한 Random seed 목록 실행.
8. 같은 raw 기준 run, 같은 comparison signature만 집계. 시간 반복과 seed 반복은 구분.
9. XD official의 test 800개·7개 prompt·checkpoint·GT 정렬과 AP를 확인했다. 새 temporal mAP 경로는 실행 검증 대기 중이다.
10. XD raw Reference는 공식 feature 대조로 재구성한 전처리와 tail을 사용한다. 새 공통 경로의 전체 평가와 공식 AP 대비 차이는 실행 검증 대기 중이다.

## 아직 구현하지 않은 범위

- Adaptive, Ours, sampler 학습/튜닝, train split, 온라인·실시간 주장.
- 자동 resume, 자동 대량 sweep, 실행 큐·GPU 예약, 실험 추적 SaaS 연동.
- 일반 wheel만으로 동작하는 독립 배포. 현재는 source checkout/editable workflow.

## 기존 진입점 이관

scripts/evaluate_ucf_official.py와 evaluate_ucf_raw.py는 기존 UCF 진단을 위해 유지했습니다.
기존 준비 스크립트의 고정 audit 출력은 outputs/setup에서 .local/setup/ucf로 이동했습니다.
새 immutable manifest와 legacy audit 형식은 다르므로 두 준비 절차를 혼용하지 않습니다.
기존 cohort 분석 스크립트에는 실제 보존한 legacy raw/official run의 경로를 모두 명시하세요.
과거 안내는 archive/README_before_refactor.md와 archive/SETUP_before_refactor.md에 있습니다.

## 두 사람 운영

각자 source checkout과 configs/local/<이름>.json, output_root를 사용합니다. 대용량 영상·checkpoint는
공용 디렉터리를 읽기 전용 입력으로 공유합니다. 처음에는 한 사람이 의존성 변경을 관리합니다.
GPU 두 장은 개발 시 독립 작업에 쓰되 최종 시간 측정은 다른 GPU/CPU/I/O 작업이 없는 상태에서 수행합니다.

Random 집계 예시(실제 run 경로로 교체):

```bash
python scripts/analyze/aggregate_runs.py RUN_SEED0 RUN_SEED1 RUN_SEED2 RUN_SEED3 RUN_SEED4 --expected-seeds 0 1 2 3 4
```

결과는 outputs/analyses의 고유 폴더에 평균·표본 표준편차로 저장됩니다. 한 run의 표준편차는 null입니다.
seed 중복이나 서로 다른 환경·방법·budget·기준 run은 거부합니다.
