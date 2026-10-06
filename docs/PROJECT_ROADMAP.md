# VAD Frame Sampling — 실행 로드맵

작성일: 2026-10-06 (Asia/Seoul)

최신 구조 확인: UCF 공식 feature ZIP·압축 해제 폴더와 UCF/CLIP checkpoint 파일이 확보된 위치에 존재한다. `configs/ucf_resources.json`을 실제 경로로 연결했다. 파일 내용·hash·모델 로딩과 평가 검증은 미실행이며, 아래 최초 점검의 미확인 상태보다 최신 `PROGRESS.md`와 `UCF_RESOURCE_SETUP.md`를 우선한다.

이 문서는 연구 계획을 현재 파일·코드·기존 로그와 대조하여 정리한 작업 순서다. 연구 목표는 `VAD_Frame_Sampling_Research_Plan.md`, 실제 수행 상태는 `PROGRESS.md`가 기준이다. 이번 검토에서는 코드 실행, 배열 로딩, 영상 디코딩, 다운로드, 모델 평가를 수행하지 않았다.

## 1. 연구의 중심

데이터셋별 pretrained VadCLIP과 CLIP encoder를 고정하고, visual encoder에 들어가기 전에 선택하는 프레임만 바꾼다. 같은 영상·같은 frame budget에서 탐지 성능과 실제 처리 시간을 비교한다.

- 데이터셋: UCF-Crime부터 완성한 뒤 XD-Violence로 확장.
- 방법: Reference, Uniform, Random, Simple Adaptive, Ours.
- Budget: reference 후보 대비 100%, 50%, 25%, 12.5%.
- 주 지표: UCF-Crime frame-level ROC-AUC, XD-Violence frame-level AP.
- 모델 재학습과 다른 VAD 모델 확장은 현재 필수 범위가 아니다.
- Ours는 기본 비교와 별도 개발 데이터 분석 이후 설계한다.

## 2. 현재 파일 구조와 준비 상태

```text
03_Project/
├── AGENTS.md                         작업 역할·실험 원칙
├── VAD_Frame_Sampling_Research_Plan.md 연구 설계
├── PROGRESS.md                       실제 진행 상태와 증거
├── VadCLIP/                          공식 모델·평가 코드
│   ├── src/                          모델, 데이터 로더, 평가, crop 예제
│   └── list/                         feature CSV, split, annotation, GT 배열
├── UCF-Crime/
│   ├── *.zip                         원본 배포 압축 파일
│   ├── extracted/                    기존 추출 영상
│   └── test/
│       ├── Anomaly/<category>/       test anomaly 영상 링크
│       ├── Normal/                   test normal 영상 링크
│       ├── annotations/              원본 annotation 복사본
│       ├── splits/                   test/anomaly/normal 목록
│       └── manifest.{json,csv}       영상별 대응·metadata·식별값
├── checkpoints/clip/                 ViT-B-16.pt.part 존재
├── scripts/prepare_ucf_test.py        데이터 정리 코드
├── configs/                          현재 비어 있음
├── docs/                             이 로드맵
├── outputs/setup/                    기존 자원 점검·데이터 정리 보고서
└── .venv/                            기존 Python 환경
```

| 구분 | 확인 상태 | 의미 |
|---|---|---|
| UCF raw test 정리 | 기존 보고서에 290개: anomaly 140 + normal 150 | 재정리부터 시작할 필요 없음 |
| 무결성·metadata | 기존 보고서에 290개 ZIP CRC/크기·metadata 통과 | 이번 검토에서 검증을 재실행한 것은 아님 |
| 전체 디코딩 | 기존 보고서에 경계 경고 영상 5개만 수행 | 나머지 285개는 전체 디코딩 미검증 |
| Test manifest·split | 파일 존재, 공식 feature CSV 순서 기록 | 이후 영상·feature·GT 연결의 기준 |
| VadCLIP 코드·GT 배열 | 파일 존재 | 모델 평가 성공이나 GT 정렬 검증을 의미하지 않음 |
| Official test feature 배열 | 프로젝트 내 미확인 | CSV는 배열 자체가 아니며 기존 외부 경로를 가리킴 |
| UCF VadCLIP checkpoint | 프로젝트 내 미확인 | 실제 자원 위치·로딩 확인 필요 |
| CLIP 가중치 | `.part` 파일 존재 | 다운로드 완료·유효성 미검증 |
| Raw 추출·sampling 실험 코드 | 완성된 실험 진입점 미확인 | `crop.py`는 crop 함수와 예제이며 전체 pipeline이 아님 |
| 성능·시간 실험 | 결과 미기록 | 아직 baseline 실행 전 |
| XD-Violence·개발 데이터 | 확보 상태 미확인 | 이후 단계에서 준비 |

Test 영상은 상대 symlink이므로 `UCF-Crime/extracted/`도 보존해야 한다. 데이터 정리 스크립트를 불필요하게 재실행하지 않는다.

## 3. 단계별 진행 순서

### Step 1 — 자원 연결과 실험 규칙 확정

Codex는 자원 점검 코드·로컬 경로 설정·실행 안내를 준비하고, 사용자가 실행하여 결과를 확인한다.

1. Official UCF test features, UCF checkpoint, 완성된 CLIP 가중치의 실제 위치를 확인한다. 외부에 확보된 자원이 있으면 재사용한다.
2. 기존 공식 CSV를 보존하고 같은 순서·label의 로컬 feature CSV를 별도 생성한다.
3. 영상 290개와 feature 파일을 일대일 연결하고 shape, 길이, dtype, 파일 식별값을 기록한다.
4. 실제 장비·패키지 import·checkpoint 로딩을 확인한다. 현재 평가 코드는 CUDA가 없으면 CPU를 선택하므로 macOS 가속 사용 여부는 별도 결정한다.
5. 두 score branch를 초기 재현에서 모두 기록하고, sampling 비교의 주 branch를 결과에 따라 sampler별로 바꾸지 않도록 사전에 정한다.
6. 설정·로그·결과의 저장 형식과 run ID 규칙을 정한다.

완료 기준: 입력 경로가 실제로 유효하고, 290개 영상·feature·GT의 대응 및 모델 로딩을 사용자 실행 결과로 확인한다. 설정 파일과 자원 점검 보고서를 남긴다.

### Step 2 — Official-feature baseline 재현

1. 작은 입력으로 feature 길이, window 분할, padding, score 길이와 해당 GT 구간의 대응을 확인한다.
2. 정렬을 검증한 전체 290개 feature로 평가한다. Normal 150개를 포함한다.
3. `sigmoid(logits1)`와 `1 - P(Normal)` 두 branch의 AUC를 각각 저장한다.
4. 논문 보고값과 차이가 있으면 checkpoint, 입력 feature 버전, branch, GT와 후처리 차이를 설명한다.

완료 기준: 재현 가능한 설정·영상별 score·전체 AUC·실행 로그가 있고, 차이의 원인 또는 확인 한계를 기록한다. 설명되지 않은 차이를 남긴 채 sampling 효과를 해석하지 않는다.

### Step 3 — Raw-video 100% reference 구축

```text
영상 → 후보 frame index/timestamp → 전처리 → CLIP features
     → 고정 VadCLIP → 원본 평가 시간축의 score → AUC
```

1. 공식 추출의 CLIP 종류, FPS/stride, crop, normalization, 시간·crop aggregation과 tail 처리를 확인한다.
2. Reference 후보 수 `N_v`를 정의한다. 100%가 원본의 모든 프레임이라는 가정은 하지 않는다.
3. 짧은/긴/normal/anomaly 영상에서 후보 시점, feature shape와 score 길이를 검증한 뒤 전체 test로 확장한다.
4. 원본 annotation 좌표, 공식 feature/GT 시간축과 raw-video 시간축의 관계를 문서화한다.
5. Sampling 결과를 보기 전에 재현 허용 오차를 정하고 official-feature 결과와 비교한다.

완료 기준: raw reference의 전체 AUC와 영상별 시점·score가 저장되고, official-feature 재현과의 차이가 설명된다. 공식 추출 조건을 완전히 복원하지 못하면 제한을 기록하고 검증된 raw reference를 sampling 비교의 기준으로 사용한다.

### Step 4 — Uniform / Random으로 공통 sampling pipeline 검증

1. Sampler 입력은 영상에서 만든 후보·시각 정보이며 GT를 전달하지 않는다.
2. 선택 수는 `K_v = min(N_v, max(1, ceil(r × N_v)))`로 통일한다.
3. Uniform은 균등하게 K개, Random은 중복 없이 K개를 선택하고 시간순으로 정렬한다.
4. CLIP 전 단계에서 선택하고 index/timestamp와 실제 encoder 입력 수를 저장한다.
5. Window·padding·positional embedding 처리와 score 복원 규칙을 고정한다. 원본 전체 평가 시간축으로 복원한다.
6. 100%에서 sampler 선택과 score가 reference와 일치하는지 먼저 확인한다.
7. 네 budget에서 AUC를 측정한다. Random seed 목록·반복 횟수를 실행 전에 확정하고 평균·표준편차를 기록한다. 5개 seed는 기존 계획의 제안이며 아직 확정값이 아니다.
8. Decoding, sampling, 전처리/encoding, VAD, score 복원과 전체 경과 시간을 함께 측정한다. 장비·precision·batch·I/O·warm-up·동기화 조건을 동일하게 유지한다.

완료 기준: UCF의 budget별 성능·실제 시간·선택 정보가 있고 100% 일치 검증을 통과한다. Feature cache 평가와 raw end-to-end 시간 측정을 구분한다.

### Step 5 — XD-Violence로 동일 절차 확장

1. XD의 test split·raw video·annotation·official features·XD checkpoint를 확보한다.
2. Official-feature AP를 먼저 재현하고 raw 100% reference를 검증한다.
3. UCF와 같은 sampler 정의·budget 원칙으로 Uniform/Random을 비교한다.
4. XD의 label/prompt·GT 정렬·집계는 데이터셋별 공식 규칙을 확인한다.

완료 기준: XD baseline과 raw reference, budget별 AP·시간 결과를 확보한다. UCF checkpoint를 XD에 그대로 사용하지 않는다.

### Step 6 — Simple Adaptive와 failure analysis

1. Test와 분리된 development/validation 데이터 및 사용 목적을 확정한다.
2. Frame difference 또는 motion 기반의 단순 adaptive를 구현한다. 해상도·비교 간격·동점 처리·coverage 규칙과 모든 추가 비용을 기록한다.
3. 고정된 adaptive를 두 benchmark에서 비교한다.
4. 개발 데이터에서 짧은 사건 누락, 큰 정상 motion 편향, 정적 anomaly 누락, 문맥 손실, 특정 구간의 선택 집중을 분석한다.
5. GT 구간·선택 시점·reference/sample score를 한 시간축에 표시하고 event coverage 및 사건 길이별 분석을 남긴다.

완료 기준: 기본 방법의 성능·비용과 실패 원인에 대한 증거가 있고, Ours가 해결할 문제를 구체적으로 정의한다. Test 결과를 보고 설계를 수정한 경우 탐색적 분석으로 표시하며 최종 평가 데이터 분리 문제를 해결한다.

### Step 7 — Ours 설계·튜닝·고정

1. Step 6의 실패 원인에서 설계 가설을 세운다. 현재 특정 알고리즘을 확정하지 않는다.
2. 별도 개발 데이터에서만 구조·hyperparameter를 비교한다.
3. 같은 K, GT 미사용, encoder 전 선택 조건을 유지한다.
4. 추가 모델·점수 계산 비용까지 포함하여 유용성을 확인한다.
5. 최종 평가 전에 코드·설정·seed·hyperparameter를 고정한다.

완료 기준: 고정된 Ours 구현과 개발 근거, 구성 요소별 ablation 계획을 확보한다.

### Step 8 — 최종 비교와 보고

고정된 방법을 두 benchmark와 네 budget에서 평가한다. Reference 대비 metric 변화(pp), 실제 end-to-end speedup, Random 변동, 단계별 비용, failure case와 ablation을 보고한다. 효율은 encoder 입력 수가 아니라 실제 전체 시간으로 판단한다.

완료 기준: 모든 결과를 코드·checkpoint·split·설정·seed·장비·로그로 추적할 수 있고, 논문 보고값·계획·실제 측정값을 구분한 최종 표와 그래프를 확보한다.

## 4. 현재 코드에서 먼저 확인할 사항

아래는 정적 검토 결과이며 실행 오류가 관측되었다는 뜻은 아니다.

| 항목 | 관찰 | 필요한 조치 |
|---|---|---|
| GT 생성 스크립트와 CSV | `make_gt_ucf.py`는 `__0.npy`만 처리하지만 현재 test CSV는 `__5.npy` 파일을 참조 | 기존 `gt_ucf.npy`의 생성 입력·순서·길이·crop 대응을 검증. 스크립트를 그대로 실행하여 GT를 덮어쓰지 않음 |
| 16배 score 반복 | `ucf_test.py`는 score를 16회 반복 | 공식 feature 시간축의 복원 규칙을 확인. Raw 후보 stride가 16이라고 단정하지 않음 |
| Annotation 경계 | 기존 보고서에 5개 영상 끝값이 실제 frame 수보다 1~2 큼 | 좌표 base·끝점 규칙과 공식 GT를 확인하고 명시적인 경계 정책을 적용. 원본 annotation 보존 |
| Window 분할 | 길이 256의 정수배에서도 `process_split`은 추가 padded window 생성 | 짧은 길이·256·512 등 경계 입력의 shape/유효 길이를 사용자가 검증. 재현에 미치는 영향 확인 |
| Padding mask | `encode_video`는 전달받은 mask 대신 temporal에 `None` 전달 | 기존 동작을 baseline에서 확인하고 raw/sampled 공통 정책 결정. 재현 전 임의 수정하지 않음 |
| 원본 시점과 모델 위치 | 위치 embedding은 window 내 순번 기준 | 비균일 sampling의 시간 간격 변화를 기록하고 sampler 간 동일 정책 적용 |
| CLIP 경로 | 모델은 이름으로 CLIP을 로딩하며 기본 cache는 `~/.cache/clip` | 프로젝트 `checkpoints/clip`과 실제 로딩 경로 연결 필요. `.part` 존재만으로 완료 처리하지 않음 |
| 실행 위치·장치 | 기본 list 경로는 작업 디렉터리에 의존; CUDA 없으면 CPU | 프로젝트 wrapper/설정에서 경로와 장치를 명시하고 사용자 실행으로 검증 |

## 5. 파일 관리 방식

현재 구조를 유지하고 필요 단계에서 파일을 추가한다. 원본 `VadCLIP` 코드는 재현 기준으로 보존하고, 연구용 실행 진입점·모듈과 설정은 별도 관리하는 것을 권장한다.

- `configs/`: 확보된 실제 경로와 공통 실험 설정. Baseline, raw reference, sampling 설정을 단계별로 추가.
- `scripts/`: 자원 점검·평가·추출·sampling 실행 진입점. 공통 구현이 커지면 연구용 모듈로 분리.
- `docs/`: 시간축·추출 조건·실험 규칙과 분석 문서.
- `outputs/setup/`: 기존 준비 보고서 유지.
- `outputs/<run_id>/`: 실행별 설정, 자원 식별값, metrics, 단계별 시간, 영상별 score, 선택 index/timestamp, 로그.
- 향후 feature/cache 저장 위치는 원본 입력과 파생 cache를 구분하여 실제 자원 확보 시 지정.

위 신규 저장 경로·파일은 제안이며 이번에 실험 산출물을 생성한 것은 아니다.

## 6. 바로 다음 작업과 역할

바로 다음 작업은 **UCF official test features·UCF checkpoint·완성된 CLIP 가중치의 위치를 확인하고, local feature CSV와 GT 정렬 점검을 준비하는 것**이다. 그 결과가 확인되면 소규모 official-feature 평가부터 진행한다.

Codex는 코드·설정·실행 명령·필요 입력·예상 출력·확인 항목을 준비하고 정적 검토한다. 사용자는 코드를 실행하고 결과·로그를 확인한다. 이번 작업은 문서 검토와 정리만 수행했으며 새 실행 명령이나 실험 코드를 작성하지 않았다.
