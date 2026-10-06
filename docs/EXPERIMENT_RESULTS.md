# VAD frame sampling 실험 결과

확인일: 2026-10-07. 기존 사용자 실행 보고서를 읽어 기록했다. Codex는 평가·데이터 처리를 실행하지 않았다.

## Official-feature baseline 재현

이 결과는 미리 추출된 공식 CLIP features를 입력한 평가다. Raw-video reference 및 아래 sampling 비교표와 구분한다. Frame budget은 적용하지 않았다.

| Dataset | 입력 | Score branch | AUC (%) | AP (%) | Ano-AUC (%) | 평가 영상 수 | 평가 frame 수 |
|---|---|---|---:|---:|---:|---:|---:|
| UCF-Crime | Official CLIP features | Branch 1 / C-branch | 88.02 | 33.56 | 미계산 | 290 | 1,109,888 |
| UCF-Crime | Official CLIP features | Branch 2 / A-branch | 85.69 | 26.50 | 미계산 | 290 | 1,109,888 |

Ano-AUC는 이상 영상 140개의 전체 프레임(정상 구간 포함)을 모아 계산한다. 기존 저장 score로 추가하는 코드를 준비했으며 아직 실행하지 않아 수치는 기입하지 않았다. 실행 명령과 출력은 [기존 결과에 Ano-AUC 추가](UCF_OFFICIAL_BASELINE.md#기존-결과에-ano-auc-추가--모델-재실행-없음)를 참고한다.

- Run: `ucf_full_20261006_235442`, MPS, float32 model parameters, seed 234, success=true, normal 영상 포함.
- 근거: [summary.json](../outputs/baseline/ucf_full_20261006_235442/summary.json). 실행 source hash, checkpoint hash, 연구 commit 및 dirty 상태는 해당 보고서에 기록되어 있다.
- UCF 주 지표는 AUC다. Branch 1은 논문 보고값 88.02%와 소수점 둘째 자리에서 일치한다. 이는 현재 official-feature run의 관측이며 raw-video 재현 성공을 뜻하지 않는다.
- 보고서의 `cached_feature_loop_seconds`는 24.088초다. 모델 초기 로딩, 최종 AUC/AP 계산·전체 score 저장을 포함하지 않는 feature 처리 loop 시간이며 영상 decoding·CLIP visual encoding도 제외한다. 아래 End-to-end 또는 Speedup 열에 넣지 않는다.
- Score NPZ 파일은 경로·존재만 확인했으며 내용을 다시 로딩하거나 metric을 재계산하지 않았다.

## Raw-video sampling 비교표

`미측정`은 실제 측정값이 아직 확인되지 않았다는 뜻이다. Reference의 Δ=0 및 Speedup=1.00×는 기준 정의이며 실측 성과가 아니다. Random은 최종 확정한 seed 반복의 평균±표준편차로 기록한다.

| Dataset | Sampling | Budget | AUC 또는 AP (%) | Δ (pp) | End-to-end (s) | Speedup | Event coverage (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| UCF-Crime | Full / Reference | 100% | 미측정 | 0 | 미측정 | 1.00× | 미측정 |
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

1. UCF raw-video reference의 후보 FPS/stride·전처리·feature aggregation과 annotation 시간축/경계·score 복원 규칙, 재현 허용 오차를 확정하고 코드를 준비한다.
2. 사용자가 raw reference를 실행해 AUC와 전체 시간, 선택 index/timestamp 및 event coverage를 확인한다. Official-feature 결과와의 차이를 설명한다.
3. 공통 규칙과 주 score branch를 고정하고 Uniform/Random 25%를 먼저 비교한다. 연구 계획의 50%/12.5% budget도 이후 동일 형식으로 기록한다.
4. XD-Violence 자원과 데이터셋별 checkpoint·AP 평가를 검증한 뒤 확장한다. Adaptive/Ours는 계획 순서에 따라 개발하며 Ours는 별도 개발 데이터에서 설계·튜닝한다.
