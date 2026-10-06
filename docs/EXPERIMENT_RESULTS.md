# VAD frame sampling 실험 결과

확인일: 2026-10-07. 기존 사용자 실행 보고서를 읽어 기록했다. Codex는 평가·데이터 처리를 실행하지 않았다.

## Official-feature baseline 재현

이 결과는 미리 추출된 공식 CLIP features를 입력한 평가다. Raw-video reference 및 아래 sampling 비교표와 구분한다. Frame budget은 적용하지 않았다.

| Dataset | 입력 | Score branch | AUC (%) | AP (%) | Ano-AUC (%) | 평가 영상 수 | 평가 frame 수 |
|---|---|---|---:|---:|---:|---:|---:|
| UCF-Crime | Official CLIP features | Branch 1 / C-branch | 88.02 | 33.56 | 70.23 | 290 | 1,109,888 |
| UCF-Crime | Official CLIP features | Branch 2 / A-branch | 85.69 | 26.50 | 65.52 | 290 | 1,109,888 |

Ano-AUC는 이상 영상 140개의 전체 프레임(정상 구간 포함)을 모아 계산한다. 사용자가 기존 저장 score로 추가한 summary의 branch1=0.7023225741073721, branch2=0.6552197493487624를 읽어 %로 기록했다. Codex는 계산을 실행하지 않았다. 실행 명령과 출력은 [기존 결과에 Ano-AUC 추가](UCF_OFFICIAL_BASELINE.md#기존-결과에-ano-auc-추가--모델-재실행-없음)를 참고한다.

새 사용자 full run `ucf_full_20261007_004524`의 fine-grained 결과를 확인했다. 평가 대상은 이상 영상 140개/13개 사건 클래스이며 공개 UCF test의 290개/14개 클래스 호출과 구분한다. 논문 표 대비 수치가 높지만 동일 모델의 성능 개선으로 해석하지 않는다. 원본 공개 evaluator·GT를 사용하는 추가 비교는 아직 미실행이다.

| Dataset / 입력 | Score branch | mAP@0.1 (%) | mAP@0.2 (%) | mAP@0.3 (%) | mAP@0.4 (%) | mAP@0.5 (%) | AVG (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| UCF-Crime / official features / anomaly-only 13 classes | Branch 2 / A-branch | 13.09 | 8.71 | 7.15 | 5.05 | 3.28 | 7.46 |
| UCF-Crime / original-code reproduction | Branch 2 / A-branch | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 |

논문 [Table 4](https://ojs.aaai.org/index.php/AAAI/article/download/28423/28826#page=6)의 UCF 보고값은 11.72/7.83/6.40/4.53/2.93%, AVG 6.68%다. 현재 anomaly-only 결과와 직접 비교 가능한 조건인지 확인 중이다. 새 run의 AUC/AP/Ano-AUC는 이전 run과 같은 값이다. [최신 결과 JSON](../outputs/baseline/ucf_full_20261007_004524/summary.json)에 class score 경로·scope·source hash가 저장되어 있다.

사용자 요청으로 full 평가의 기본 mAP를 원본 공개 코드 방식으로 수정했다. 다음 명령으로 모델부터 다시 평가하여 새 run을 만든다.

```sh
.venv/bin/python scripts/evaluate_ucf_official.py --mode full --device mps
```

새 summary의 `metrics.fine_grained`에 원본 `excludeNormal=False` 조건(290개/14개 클래스)의 결과를 저장한다. 원본 segment/label GT를 순서·구간·category와 대조한 뒤 그대로 사용하며, 'A' 등의 label도 remap하지 않는다. 기존 run의 anomaly-only 결과는 보존한다. 원본 방식은 사용자 실행 전까지 미측정이며 별도 저장-score mAP 갱신은 사용하지 않는다.

- 최신 Run: `ucf_full_20261007_004524`, MPS, float32 model parameters, seed 234, success=true, normal 영상 포함. 이전 run과 AUC/AP/Ano-AUC는 같다.
- 근거: [summary.json](../outputs/baseline/ucf_full_20261007_004524/summary.json). 실행 source hash, checkpoint hash, 연구 commit 및 dirty 상태는 해당 보고서에 기록되어 있다.
- UCF 주 지표는 AUC다. Branch 1은 논문 보고값 88.02%와 소수점 둘째 자리에서 일치한다. 이는 현재 official-feature run의 관측이며 raw-video 재현 성공을 뜻하지 않는다.
- 최신 보고서의 `cached_feature_loop_seconds`는 24.368초다. 모델 초기 로딩, 최종 metric 계산·전체 score 저장을 포함하지 않는 feature 처리 loop 시간이며 영상 decoding·CLIP visual encoding도 제외한다. 아래 End-to-end 또는 Speedup 열에 넣지 않는다.
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

1. 수정된 full evaluator를 실행해 원본 fine-grained mAP/AVG를 확인하고 논문과의 차이를 점검한다.
2. UCF raw-video reference의 후보 FPS/stride·전처리·feature aggregation과 annotation 시간축/경계·score 복원 규칙, 재현 허용 오차를 확정하고 코드를 준비한다.
3. 사용자가 raw reference를 실행해 AUC와 전체 시간, 선택 index/timestamp 및 event coverage를 확인한다. Official-feature 결과와의 차이를 설명한다.
4. 공통 규칙과 주 score branch를 고정하고 Uniform/Random 25%를 먼저 비교한다. 연구 계획의 50%/12.5% budget도 이후 동일 형식으로 기록한다.
5. XD-Violence 자원과 데이터셋별 checkpoint·AP 평가를 검증한 뒤 확장한다. Adaptive/Ours는 계획 순서에 따라 개발하며 Ours는 별도 개발 데이터에서 설계·튜닝한다.
