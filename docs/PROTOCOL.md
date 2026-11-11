# Sampling 실험 protocol — 초안

## 상태와 순서

설정 파일의 `protocol.status`는 모두 draft입니다. 기본 보간·seed 제안은 새 실험 결과로 검증된 선택이 아닙니다.
본 실험 전에 설정을 확정한 사본을 만들고 status를 fixed로 변경합니다. test 결과를 보고 보간·후처리를
sampler별로 조정하지 않습니다. fixed라는 설정만으로 재현 검증·통계 검증이 완료되지는 않습니다.

진행 순서: 입력 준비 → 단위 검증 → official 재현 → raw Reference 재현 → 100% 선택 동일성 → Uniform/Random 25%.

## 고정 규칙과 제안 규칙

| 항목 | 구현/현재 제안 |
|---|---|
| UCF 후보 | 보존한 영상별 reference_tokens개의 frame 16k, offset 0 |
| UCF tail | 기존 official row 수에 따라 floor/ceil; 긴 float32 4개 제한점 유지 |
| Budget | K=min(N,max(1,ceil(r*N))), 100/50/25/12.5% |
| Uniform | 후보 순서를 K개 동일 너비 bin으로 나누고 각 bin의 중앙 위치를 정수 연산으로 선택 |
| Random | 중복 없는 K개 선택 후 시간순 정렬, 영상별 독립 RNG |
| Sampler seed | base seed와 dataset+filename의 SHA-256으로 영상별 seed 도출 |
| Model seed | 모든 sampler·seed 반복에서 234로 고정 |
| Seed 반복 제안 | 0,1,2,3,4; 최종 확정 전 |
| 점수 복원 제안 | 선택 frame 좌표에서 reference candidate grid로 nearest; 동점은 이전 관측 |
| 다른 복원 선택지 | linear, previous; 동일 비교군에서는 하나로 고정 |
| 최종 frame 점수 | 복원된 reference token마다 16번 반복, 기존 공식 평가 길이 유지 |
| Positional encoding | 선택 token을 시간순 압축해서 기존 model에 입력; timestamp를 새 embedding으로 넣지 않음 |
| Window/mask | 원본 process_split의 256 배수 추가 window와 temporal mask 무시 동작 보존 |
| 100% | 어떤 sampler·보간도 같은 후보와 원래 token 점수를 유지하도록 구현, 실제 수치 검증 필요 |
| GT | 평가·coverage 분석용. sampler에 전달하지 않음 |

비균일 입력의 실제 시간 간격은 모델의 압축 token 위치에 직접 반영되지 않습니다. 이 시간 정보 손실은
방법의 제한점이며 보고서에 기록합니다. 기본 protocol은 offline이며 온라인·실시간 성능을 주장하지 않습니다.

UCF 주 지표는 branch1 ROC-AUC, 보조 AP/Ano-AUC와 원본 290영상·14클래스 fine-grained 평가입니다.
XD official 주 지표는 원본 test가 반환하는 branch2 AP입니다. 보조 지표는 원본 XD evaluator의
800영상·7클래스 temporal mAP@IoU 0.1~0.5입니다. Temporal mAP의 실행 검증과 XD raw recipe 검증은 대기 중입니다.

## 시간 측정

각 영상의 원본 입력 hash 확인 후, candidate 구성·sampling부터 decoding·전처리·전송·CLIP·VAD·복원까지 측정합니다.
원본 hash 확인 직후라 OS 파일 cache가 따뜻할 수 있습니다. 모든 방법에 같은 조건을 적용하고 한계를 기록합니다.
모델 로딩·synthetic warm-up·GT 분석·metric 계산·결과 저장은 측정에서 제외합니다.
run의 end_to_end_seconds는 영상별 호출 시간 합입니다. processing_loop_wall_seconds는 저장 등도 포함합니다.

단일 run은 반복 측정값이 아닙니다. 다른 작업이 없는 동일 장비·같은 batch/precision에서 반복 측정하세요.
샘플링 결과의 seed 변동과 동일 설정의 시간 반복은 별도로 관리합니다.
기준 run은 동일 입력 identity·코드·checkpoint·restoration·runtime·GPU signature의 full raw 100%여야 합니다.

## 사용자 검증 항목

- tests의 budget, tie, tail, full-selection, GT 정렬 검증 통과.
- official 전체 AUC 및 raw 전체 AUC가 기존 결과와 어떤 차이가 있는지 기록.
- Reference 100 / Uniform 100 / Random 100의 selected index, feature, score, metric 동등성 확인.
- 25%에서도 평가 frame 수가 100%와 같고 실제 선택 K가 공식 budget 정의와 같은지 확인.
- 실패 run이나 smoke를 최종 결과에 포함하지 않음.
- Random 반복은 동일 signature·동일 baseline·서로 다른 선언 seed 목록만 집계.
- 허용할 장비별 재현 오차와 timing 반복 횟수는 sampling 성능을 보기 전에 확정.
