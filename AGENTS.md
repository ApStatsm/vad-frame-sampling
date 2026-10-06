# VAD Frame Sampling — Codex 작업 지침

## 코드 작성·검증·실행 역할 (2026-10-06 사용자 지시)

- Codex는 프로젝트 코드를 작성·수정하고 정적으로 검토한다. 사용자가 직접 코드를 실행하고 결과를 확인한다.
- 사용자가 별도로 실행을 명시적으로 요청하지 않으면 프로젝트 코드, 테스트, dry-run, import를 통한 실행, inference, 데이터 처리·디코딩·벤치마크를 실행하지 않는다. 이전 작업의 실행 허용을 이후 작업에 포괄적으로 적용하지 않는다.
- 실행 없이 코드·설정·기존 파일/로그를 읽고 논리, 경로와 입출력 규칙을 검토한다. 코드·문서 편집과 파일 읽기는 가능하다.
- 코드 전달 시 목적, 실행 명령, 필요한 입력, 예상 출력 위치와 사용자가 확인할 항목을 안내한다. 정적 검토와 실제 실행 검증을 구분하고 미실행 상태를 명시한다.
- 사용자가 제공한 결과·로그를 바탕으로 오류를 분석하고 수정한다. 이 역할 규칙은 아래 검증·실험 수행 지침보다 우선한다.

## 작업 시작과 인계

- 작업 시작 전에 같은 프로젝트 루트의 `VAD_Frame_Sampling_Research_Plan.md`와 `PROGRESS.md`를 읽는다.
- 연구 계획서에서 목표·실험 원칙을 확인하고, 진행 기록에서 실제 완료 상태·미해결 문제·다음 작업을 확인한다.
- 사용자의 최신 지시를 우선한다. 지시가 연구 계획을 변경하면 변경 내용과 이유를 기록한다.
- 기존 코드, 설정, 실행 로그를 확인하여 이미 완료된 작업을 반복하지 않는다.
- 진행 기록과 실제 파일·로그가 다르면 증거를 확인하고 기록을 수정한다. 확인되지 않은 상태는 추정하지 않는다.
- 작업 종료 시 `PROGRESS.md`에 수행한 작업, 검증 결과, 결과 파일 위치, 미해결 문제와 바로 실행할 다음 작업을 기록한다.

## 연구 목표와 범위

- 고정된 pretrained VadCLIP의 inference에서 frame sampling strategy만 변경하여 탐지 성능과 계산 효율을 비교한다.
- Main benchmarks는 UCF-Crime과 XD-Violence다. UCF-Crime에서 재현과 raw-video pipeline을 먼저 완성한 뒤 XD-Violence로 확장한다.
- 데이터셋별로 해당 pretrained checkpoint를 사용한다. 같은 데이터셋의 sampler 비교에서 CLIP image/text encoder, VadCLIP 가중치, prompt, score branch와 후처리를 고정한다.
- 기본 비교 방법은 Full / Reference, Uniform, Random, Simple Adaptive, Ours다.
- 기본 frame budget은 100%, 50%, 25%, 12.5%다.
- Ours는 아직 설계되지 않았다. 재현과 기본 sampling 비교를 검증한 뒤 별도 개발 데이터의 분석을 바탕으로 구체화한다.
- VadCLIP 재학습과 다른 VAD 모델로의 확장은 현재 필수 범위가 아니다.

## 구현 순서

1. UCF-Crime test split, annotation, official test features와 checkpoint의 위치 및 대응 관계를 확인한다.
2. Official features와 pretrained checkpoint로 baseline 평가를 재현한다.
3. Raw video에서 동일 조건의 CLIP features를 추출하는 reference pipeline을 검증한다.
4. Uniform / Random을 적용하여 budget별 탐지 성능과 실제 시간을 측정한다.
5. XD-Violence에서도 데이터셋별 baseline과 평가를 검증한다.
6. 두 benchmark의 Adaptive baseline과 failure analysis를 진행한다.
7. 개발 데이터에서 Ours를 설계·튜닝한 뒤 고정된 방법으로 최종 평가한다.

## Sampling과 시간축 규칙

- 최종 효율 실험에서는 CLIP visual encoder 이전에 프레임을 선택한다.
- 전체 CLIP features를 계산한 뒤 일부를 선택하는 실험은 탐색용으로 구분하고, visual encoding 비용 절감의 증거로 사용하지 않는다.
- 영상별 reference 후보 수 `N_v`와 budget `r`에 대해 선택 수는 `K_v = min(N_v, max(1, ceil(r * N_v)))`로 맞춘다. 동일 영상·budget에서 모든 sampler의 K를 동일하게 유지한다.
- 후보 생성 FPS/stride와 budget의 분모를 명시한다. Reference 100%를 모든 원본 프레임과 같은 것으로 가정하지 않는다.
- 선택 index와 timestamp를 저장하고, 중복 없이 시간순으로 모델에 입력한다.
- 전처리, window 구성, positional encoding 처리, padding/mask와 원본 시간축으로의 score 복원 규칙을 사전에 정하고 sampler 간 동일하게 적용한다.
- 100% 입력에서는 reference와 같은 탐지 결과가 나오는지 확인한다.

## 평가와 검증

- UCF-Crime은 frame-level ROC-AUC, XD-Violence는 frame-level AP를 주요 지표로 사용한다.
- 공식 평가의 test 목록, normal 영상 포함 여부, GT 정렬과 집계 규칙을 확인한다.
- Sampling 후에도 원본 annotation timeline 전체에 대해 평가한다. 선택된 프레임만 평가하지 않는다.
- GT annotation은 평가와 분석에만 사용한다. Sampler 입력이나 test set 기반 hyperparameter 선택에 사용하지 않는다.
- Sampler 학습·튜닝이 필요하면 test set과 분리된 개발/validation 데이터를 사용한다. 설계에 활용한 test 결과는 탐색적 분석으로 표시한다.
- Random 등의 확률적 방법은 고정된 seed 목록으로 반복하고 평균·표준편차를 기록한다. 반복 횟수는 최종 평가 전에 확정한다.
- Official-feature 재현과 raw-video reference 결과를 구분한다. 설명되지 않는 재현 차이를 해결한 뒤 sampling 효과를 해석한다.
- 변경 범위에 필요한 검증을 수행하고, 실제로 실행한 검증과 실행하지 못한 검증을 구분하여 보고한다.

## 효율과 실험 기록

- 같은 frame budget을 같은 총 계산 비용으로 표현하지 않는다.
- Decoding, sampler의 점수 계산·추가 모델, 전처리/CLIP encoding, VAD와 score 복원 비용을 측정에 포함한다.
- Feature cache를 이용한 성능 실험과 raw-video end-to-end 시간 실험을 구분한다.
- 같은 장비, precision, batch 정책과 I/O 조건에서 측정하고 warm-up·GPU 동기화 조건을 기록한다.
- Encoder 입력 수 감소만으로 speedup을 추정하지 않는다. Speedup은 실제 end-to-end 시간으로 계산한다.
- 각 실험의 코드 버전, checkpoint 식별값, split, 설정, seed, 입력 수, 장비, 실행 방법, metric, 시간과 출력 경로를 저장한다.
- 논문 보고값, 계획, 실제 측정값을 명확히 구분한다. 실행하지 않은 실험의 결과를 채우지 않는다.

## 데이터와 문서 관리

- 원본 영상, official features, checkpoints와 annotations는 실험 입력으로 보존한다. 파생 결과는 별도 출력 위치에 저장한다.
- ChatGPT 프로젝트의 `sources/` 등 동기화된 참조 파일이 있으면 읽기 전용으로 취급한다.
- 데이터·checkpoint·로그 경로는 실제 확인한 위치만 `PROGRESS.md`에 기록한다. 대용량 자원을 이 안내 파일에 포함하지 않는다.
- 연구 계획서는 목표와 설계를, 진행 기록은 현재 상태와 실험 증거를 담도록 유지한다.
