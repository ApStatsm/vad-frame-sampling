# VAD Frame Sampling — 진행 기록

최종 업데이트: 2026-10-06 (Asia/Seoul)  
현재 단계: 사용자 자원·baseline 입력 점검 및 이전 버전 MPS 소규모 추론 7사례 통과 / GitHub 공유 준비·현재 버전 smoke 재점검 필요  
다음 목표: 사용자가 현재 코드의 `evaluate_ucf_official.py --mode smoke --device mps`를 통과한 뒤 `--mode full --device mps`로 전체 test 290개의 AUC를 평가한다.

이 파일은 새 Codex 작업에서 이어서 진행하기 위한 기록이다. 아래 상태는 이 대화에서 확인한 내용 기준이며, 다른 컴퓨터나 서버에서 수행한 작업은 확인 후 반영한다.

## 1. 읽을 문서

- `AGENTS.md`: Codex 작업 지침과 실험 원칙
- `VAD_Frame_Sampling_Research_Plan.md`: 상세 연구 계획, 지표 정의와 단계별 완료 기준
- 이 파일: 실제 진행 상태, 실행 환경, 결과와 다음 작업
- `docs/PROJECT_ROADMAP.md`: 현재 파일·코드와 대조한 단계별 작업, 완료 기준과 사전 점검 항목
- `docs/UCF_RESOURCE_SETUP.md`: Step 1 자원 확보·설정·사용자 실행 안내
- `docs/UCF_OFFICIAL_BASELINE.md`: 입력 점검 통과 이후 smoke·전체 official-feature 평가 실행 안내

세 파일을 실제 연구 프로젝트 루트에 함께 둔다. 기존 `AGENTS.md`가 있는 프로젝트에서는 기존 지침을 보존하며 연구 지침을 합친다.

## 2. 현재 상태

| 항목 | 상태 | 확인 근거 / 남은 작업 |
|---|---|---|
| 연구 주제와 실험 범위 | 정리 완료 | VadCLIP 고정 inference에서 sampling만 변경 |
| Main benchmarks | 결정 완료 | UCF-Crime, XD-Violence |
| 상세 연구 계획서 | 작성 완료 | `VAD_Frame_Sampling_Research_Plan.md` |
| Codex 지침과 진행 기록 | 작성 완료 | `AGENTS.md`, `PROGRESS.md` |
| 공식 참고 자료 | 문서 안내 확인, 확보 자원 검증 진행 | 공개 자원 파일·CLIP hash·UCF CPU checkpoint loading 확인. Inference/논문 metric 재현은 아직 미수행 |
| 실제 연구 코드 폴더 / 서버 | 로컬 폴더 확인 | 프로젝트 루트와 VadCLIP checkout 확인. 서버는 미확인 |
| VadCLIP 코드 및 실행 환경 | 사용자 CPU loading 및 MPS smoke 성공 | torch 2.8.0, strict loading 349 entries. MPS 7사례 finite score·길이 검사 통과. CPU forward/전체 평가 미실행 |
| UCF-Crime 데이터·feature·checkpoint | Raw 정리·자원·배열/GT 점검 통과 | Feature 290개 finite `[T,512]`. __0/__5 shape 동일. GT 1,109,888 및 공식 feature timeline/annotation mismatch 0. CPU loading 성공 |
| XD-Violence 데이터·feature·checkpoint | 확인 필요 | UCF pipeline 검증 후 확장 준비 |
| Official-feature baseline 평가 | 이 작업에서 미수행 | UCF-Crime부터 실행 |
| Raw-video reference 검증 | 이 작업에서 미수행 | 공식 추출 조건 확인 후 구축 |
| Sampling 비교 / failure analysis | 이 작업에서 미수행 | Reference 검증 후 진행 |
| Ours 설계 / 최종 평가 | 미착수 | 별도 개발 데이터와 baseline 분석 필요 |

## 3. 유지할 결정

- 2026-10-06부터 코드 실행·결과 확인은 사용자가 담당한다. Codex는 코드 작성·수정과 정적 검토만 수행하며, 사용자의 별도 명시적 실행 요청이 없는 한 테스트·dry-run·프로젝트 import·데이터 처리·inference도 실행하지 않는다. 실행 명령과 확인 항목을 전달하고, 사용자 결과로 후속 분석한다. 이전 실행 기록은 이 지시 이전에 수행한 작업이다.
- Baseline은 VadCLIP이며, 데이터셋별 pretrained checkpoint로 inference한다.
- CLIP encoder와 VadCLIP 가중치를 고정하고 sampling strategy만 비교한다.
- UCF-Crime의 주요 지표는 frame-level ROC-AUC, XD-Violence는 frame-level AP다.
- Full / Reference, Uniform, Random, Simple Adaptive, Ours를 비교한다.
- Frame budget은 100%, 50%, 25%, 12.5%다.
- 최종 효율 실험에서는 visual encoder 이전에 sampling한다.
- UCF-Crime official features 재현 → raw-video reference → sampling 비교 순서로 진행한다.
- 현재 inference 계획에는 VadCLIP 재학습용 train data가 필수는 아니다. Sampler 개발·튜닝에 필요한 데이터는 test와 분리한다.

## 4. 환경과 자원 위치

2026-10-06 로컬 파일과 패키지 metadata를 확인했다. 파일 존재와 실제 로딩·실행 성공은 구분한다. 아래 상대 경로는 확인된 프로젝트 루트 기준이다.

| 항목 | 확인된 값 / 위치 |
|---|---|
| 연구 프로젝트 루트 | `/Users/apstat/Desktop/01_4학년2학기/03_인공지능 프로젝트/03_Project` |
| 실행 위치: 로컬 / 서버 | 로컬 macOS (`Darwin arm64`); 별도 서버 미확인 |
| OS, Python, PyTorch, CUDA | 사용자 보고서: Python 3.10.0, torch 2.8.0, numpy 2.2.6. CPU 모델 로딩 성공. CUDA unavailable, MPS smoke 7사례 통과 |
| GPU / 메모리 | 미확인 |
| VadCLIP 코드 경로와 commit | `VadCLIP/`의 일반 소스 폴더; 원본 base `c41067f07d252efcda18008bea367886070c33b0`는 `UPSTREAM.json`에 고정. Local 변경: 선택적 CLIP path 생성자 인수, DistanceAdj 장치 연결과 변경 표시. 연구 commit/dirty 상태와 source hash를 구분하여 기록. 이전 내부 Git 이력은 `.local-backups/vadclip-before-integration/git-history/`에 보존 |
| CLIP 종류와 가중치 | `checkpoints/clip/ViT-B-16.pt`(350,837,078 bytes), SHA-256 `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f` 기대값 일치. 사용자 CPU 모델 생성/로딩 성공 |
| UCF-Crime raw test videos | `UCF-Crime/test/Anomaly/<category>/`, `UCF-Crime/test/Normal/`: 290개 상대 symlink. 원본은 `UCF-Crime/extracted/`의 1,100개 영상(30,448,047,108 bytes). ZIP 원본 보존. Test 290개 CRC/크기 및 metadata 검증 |
| UCF-Crime test 목록 / annotation | `UCF-Crime/test/manifest.json`, `manifest.csv`, `splits/{test,anomaly,normal}.txt`, `annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt`; official feature CSV 순서 유지, annotation byte 일치 |
| UCF-Crime official test features | `features/official/ucf/UCFClipFeatures/`: 19,500개 .npy 경로 확인. Test 290개 mapping/배열 검증 통과. 286 float16 + 4 float32, finite `[T,512]`, __0/__5 shape 일치. local CSV 생성 |
| UCF-Crime checkpoint / hash | `checkpoints/vadclip/model_ucf.pth`(649,186,055 bytes), SHA-256 `d064aba05190ce5f6652151e6259b24ca6dda83a1e6289bf343afb6f905a2c79`. CPU strict loading 성공 (14 classes, 349 entries), MPS smoke 통과. 전체 평가 미실행 |
| XD-Violence raw test videos | 미확인 |
| XD-Violence test 목록 / annotation | 미확인 |
| XD-Violence official test features | 미확인 |
| XD-Violence checkpoint / hash | 미확인 |
| Development / validation data | 미확인 |
| 실험 설정 / 로그 / 결과 저장 위치 | Setup 자원·raw 정리 보고서 보존. 사용자 자원 점검: `ucf_resource_check.json`, `ucf_CLIP_rgbtest.local.csv`. 배열/GT/모델 점검: `ucf_baseline_input_check.json`. Inference metric 없음 |

## 5. 다음 작업

Raw test 데이터 정리는 완료했다. 데이터 목록은 `UCF-Crime/test/manifest.json`을 사용하고, 현재 상대 symlink 구조에서는 원본 `UCF-Crime/extracted/`도 함께 보존한다.

아래 실행 검증·평가는 사용자가 수행한다. Codex는 필요한 코드·설정·명령과 확인 항목을 준비하고 정적으로 검토한다. 사용자에게 명시적인 실행 요청을 받은 경우에만 해당 범위에서 직접 실행한다.

1. 자원 경로 점검은 사용자 실행 보고서에서 통과했고 local CSV도 생성됐다. 반복 실행·재다운로드 없이 다음 점검으로 진행한다.
2. Feature/GT/CPU model loading도 사용자 보고서에서 통과했다. 반복 점검 없이 소규모 inference로 진행한다.
3. 사용자 MPS smoke 보고서에서 7개 실제/경계 사례의 유효 score와 복원 길이 검증 통과를 확인했다. CPU forward 및 장치 간 수치 일치는 미검증이다.
4. GitHub 공유 준비에서 변경 표시와 evaluator의 연구/upstream 버전 기록을 보완하여 source hash가 바뀌었다. 사용자가 `.venv/bin/python scripts/evaluate_ucf_official.py --mode smoke --device mps`로 현재 버전 7사례를 다시 통과한 뒤 `--mode full --device mps`로 전체 official-feature baseline AUC를 두 score branch별로 기록한다. 자원/config/GT는 이번 정리에서 변경하지 않아 기존 입력 점검을 반복할 필요는 없다.
5. 재현 차이를 설명한 뒤 raw-video reference 추출 조건과 시간축을 검증한다.
6. Reference 검증 후 Uniform / Random 및 budget 비교로 넘어간다.

Raw test 입력은 official test 목록의 290개를 기준으로 선택한다. Annotation의 category를 함께 확인하여 anomaly 140개와 normal 150개를 구분한다. 현재 fixed-checkpoint inference에는 training normal 데이터가 필수가 아니며, sampler 튜닝 데이터는 test와 분리한다.

## 6. 구현 전에 확정할 사항

| 항목 | 현재 상태 | 확정 시점 |
|---|---|---|
| Reference FPS / stride, crop, normalization | 공식 추출 조건 확인 필요 | Raw-video reference 구현 전 |
| 모델의 score branch / 결합 방식 | 코드상 sigmoid와 `1 - P(Normal)` 두 branch 확인; 비교의 주 branch/결합 정책 미확정 | Baseline 평가 전 |
| Raw-video 재현 허용 오차 | 미정 | Sampling 결과를 보기 전 |
| Window, padding/mask, 시간 정보 처리 | 미정 | Sampling 구현 전 |
| 비균일 선택의 score 복원 | Nearest 또는 linear 등 검토 | Sampling 평가 전 |
| Random 반복 횟수와 seed 목록 | 5개 seed가 초기 제안, 미확정 | 본 실험 전 |
| Adaptive score와 선택 규칙 | Frame difference 또는 motion 검토 | Adaptive 실험 전 |
| 개발 / 최종 평가 분리 | 데이터 확보 방식 미정 | Ours 개발 전 |
| Ours 구조와 hyperparameters | 미정 | Baseline 검증과 개발 데이터 분석 후 |
| 시간 측정 조건과 반복 횟수 | 미정 | 효율 실험 전 |

## 7. 실험 결과

**현재 기록된 실제 측정 결과는 없다.** 논문의 수치를 우리 결과로 기입하지 않는다.

| Run ID | Dataset / 입력 | Sampling | Budget | AUC 또는 AP (%) | End-to-end (s) | 상태 | 설정 / 로그 / 결과 경로 |
|---|---|---|---|---|---|---|---|
| — | UCF-Crime / official features | Reference | 해당 없음 | — | — | 미실행 | — |
| — | UCF-Crime / raw video | Full / Reference | 100% | — | — | 미실행 | — |
| — | XD-Violence / official features | Reference | 해당 없음 | — | — | 미실행 | — |

실험을 수행하면 행을 추가하고 아래 정보를 함께 남긴다.

- 날짜, run ID, 실행 방법과 코드 commit
- Dataset split과 입력 유형: official features / raw video / cached features
- Checkpoint 식별값, encoder, 전처리, 후보 수와 실제 선택 수
- Sampling 방식, budget, seed와 score 복원 설정
- 장비, precision, batch, warm-up과 동기화 조건
- 주요 metric, reference 대비 차이, 단계별 시간과 총 시간
- 로그, 설정, 선택 index/timestamp, score 및 결과 파일 위치
- 검증 범위, 실패 원인 또는 해석상의 제한

## 8. 미해결 문제와 인계 메모

- 사용자 자원 경로·CLIP hash·feature 배열·official-feature GT 정렬·CPU checkpoint 로딩과 MPS 소규모 forward/window 검증은 통과했다. 전체 test 평가와 실제 metric 재현, 장치 간 수치 비교는 남아 있다.
- Feature 배포의 `Normal_Videos_event/`와 `Testing_Normal_Videos_Anomaly/`에 공식 test normal 50개의 같은 파일명이 존재한다. 배열 동일성은 미확인이다. 공식 CSV가 지정한 parent 폴더와 파일명을 함께 기준으로 선택하며, 원본 패키지의 다른 폴더를 삭제하거나 임의 병합하지 않는다.
- Annotation 파일에는 normal 150개도 포함된다. Annotation 전체를 anomaly로 분류하지 않는다.
- Feature CSV와 annotation은 파일명 집합은 같지만 행 순서는 다르다. GT 배열과의 정렬은 공식 CSV/GT 생성 규칙 기준으로 별도 검증해야 한다.
- `make_gt_ucf.py`의 __0 필터와 test CSV의 __5 차이에 대해 사용자 점검으로 290개 variant shape 일치, GT 전체 길이와 영상별 annotation mismatch 0을 확인했다. 기존 GT의 정확한 생성 이력과 crop 인덱스의 원본 추출 조건은 별도 확인 대상이다. GT를 덮어쓰지 않는다.
- `process_split`은 길이가 256의 정수배인 경우에도 추가 padded window를 만들며, `encode_video`는 전달된 padding mask 대신 temporal에 `None`을 전달한다. 실행 오류·성능 영향은 미검증이다. Baseline에서 기존 동작을 확인하고 raw/sampled 공통 window·padding 정책을 정한다.
- Raw test 290개는 추출 영상의 전체 bytes를 읽어 ZIP 크기·CRC32와 대조하고 SHA-256을 저장했으며 ffprobe metadata도 통과했다. 전체 프레임 디코딩은 annotation 경계 경고가 있는 5개에서만 수행했다. 나머지 285개는 metadata 검증 범위다.
- 5개 영상의 annotation 끝값이 디코딩 프레임 수보다 1~2 크다: Arson011 (1267 vs 1266), Arson016 (1796 vs 1795), Explosion033 (3156 vs 3154), Fighting003 (3103 vs 3102), Shooting015 (1715 vs 1713). Header와 실제 디코딩 수는 일치했고 decoder 오류는 없었다. 원본 annotation은 보존했다. Raw-video 평가 구현 시 좌표 규칙과 경계 처리, 공식 GT의 feature timeline을 확인해야 한다.
- 처음 이어받는 작업은 baseline 재현 준비다. Ours 구현부터 시작하지 않는다.
- 데이터 다운로드와 checkpoint 로딩 여부를 먼저 확인하여 이미 확보한 자원을 중복 다운로드하지 않는다.
- Test 결과를 보고 sampler를 수정했는지 기록하고, 설계에 사용한 데이터와 최종 평가 데이터를 구분한다.

## 9. 작업 기록

### 2026-10-05 — 연구 문서 준비

- 수행: 기존 대화를 바탕으로 상세 연구 계획서 작성. Codex용 연구 지침과 초기 진행 기록 작성.
- 확인: 계획서의 필수 항목과 Markdown 코드 블록 구성을 점검. 모델 평가나 데이터 처리는 수행하지 않음.
- 준비 파일: `AGENTS.md`, `PROGRESS.md`, `VAD_Frame_Sampling_Research_Plan.md`.
- 다음: 실제 연구 프로젝트의 환경과 UCF-Crime 자원 위치 확인 후 official-feature baseline 평가.

후속 작업 기록은 날짜별로 추가한다. 각 기록에는 수행 내용, 실제 검증 결과, 출력 위치, 미해결 문제와 다음 작업을 포함하고, 위의 현재 상태와 자원 위치 표도 함께 갱신한다.

### 2026-10-06 — 프로젝트 맥락 인계 및 로컬 자원 점검

- 수행: 연구 계획·지침·진행 기록을 읽고 사용자의 최신 맥락과 로컬 코드/ZIP/환경을 대조했다. 연구 목표 변경 없음. 모델 코드나 sampler는 수정하지 않았다.
- 검증: annotation 290개(140 anomaly + 150 normal), `Anomaly_Test.txt`의 290개, feature CSV의 290개 파일명 집합 일치. ZIP별 test 포함 수 19/39/51/31 + normal 150; 누락 없음. Annotation과 feature CSV 행 순서는 다름.
- 코드 확인: `ucf_test.py`는 feature 입력을 받아 sigmoid branch와 `1 - P(Normal)` branch를 평가하고 score를 16회 반복한다. 이 반복만으로 raw-frame 후보 생성 stride/aggregation이 확정되는 것은 아니다. `model.py`의 encoder는 `ViT-B/16`이다.
- 환경: `.venv` 존재, Python 3.10.0. Metadata로 torch 2.8.0, torchvision 0.23.0, numpy 2.2.6, scikit-learn 1.7.2, pandas 2.2.3, opencv-python-headless 5.0.0.93, ftfy 6.3.1, regex 2025.11.3 확인. `.venv`는 system-site-packages를 포함한다. 실제 모델/장치 실행은 검증하지 않았다.
- 출력: `outputs/setup/resource_audit_2026-10-06.json`; 이 `PROGRESS.md`의 현재 상태·경로·다음 작업 갱신.
- 한계: ZIP central directory만 검사했다. 전체 CRC, 영상 디코딩, feature/GT 배열 로딩, checkpoint 로딩, inference와 성능·시간 측정은 수행하지 않았다. 기존 부분 다운로드 파일은 보존했다.
- 미해결/바로 다음: official test features와 UCF checkpoint 및 완성된 CLIP 가중치를 확인한 뒤 소규모 official-feature baseline 검증. 추가 다운로드·압축 해제는 이번 인계 점검에서 수행하지 않았다.

### 2026-10-06 — 사용자 우선순위 반영: raw test 데이터 정리

- 변경 이유: 사용자가 baseline 준비 전에 데이터셋 정리를 요청했다. 연구 목표는 유지하고 연구 계획서에 데이터 정리 순서·산출물 구조를 명시했다.
- 기존 상태 수정: 앞선 점검에서 `UCF-Crime/extracted/`를 놓쳤다. 이번에 기존 추출 영상 1,100개와 test 290개 존재를 확인했다. 재압축 해제 없이 이 파일을 재사용했다.
- 수행: `scripts/prepare_ucf_test.py` 작성·실행. 공식 test 목록과 annotation, official feature CSV의 파일명/label을 대조한 뒤 test-only 상대 symlink 구조 생성. 140 anomaly는 13개 category 폴더로, 150 normal은 Normal 폴더로 정리했다.
- 출력: `UCF-Crime/test/README.md`, `manifest.json`, `manifest.csv`, `splits/test.txt`, `splits/anomaly.txt`, `splits/normal.txt`, `annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt`; `outputs/setup/ucf_test_preparation_report.json`, `ucf_test_preparation.log`.
- 실제 실행: `.venv/bin/python scripts/prepare_ucf_test.py --dry-run`, `.venv/bin/python -u scripts/prepare_ucf_test.py > outputs/setup/ucf_test_preparation.log 2>&1`; 완료 후 dry-run과 layout 검증 재실행.
- 검증: test 290개 누락·중복·추가 영상 없음. 모든 상대 link 유효. Source bytes/ZIP entry 크기·CRC32 290개 일치, SHA-256 저장. ffprobe video metadata 290개 통과. Manifest와 split 목록은 official feature CSV 순서 유지. Annotation 복사본은 원본과 byte 일치. 선택 원본 용량 7,988,909,023 bytes이나 symlink로 영상 복사 비용을 줄였다.
- 추가 검증: annotation 경계 경고 5개를 ffprobe `-count_frames`로 전체 디코딩하여 header frame count와 일치 및 decoder stderr 없음 확인. Endpoint의 1~2 frame 차이는 남아 있고 annotation을 임의 수정하지 않았다. 나머지 285개 전체 디코딩은 미실행.
- 보존: ZIP, 기존 extracted 영상, upstream test/feature 목록과 GT 배열 보존. 학습 영상 810개는 test 폴더에 포함하지 않았다. GT 좌표 변환·GT 배열 생성·sampling·baseline inference는 미실행.
- 미해결/다음: official test features와 UCF checkpoint/CLIP 가중치를 확인한 뒤 소규모 official-feature baseline 검증. Raw-video 평가 전에 위 5개 annotation 경계와 GT 시간축 규칙을 확인한다.

### 2026-10-06 — 코드 실행 역할 변경

- 사용자 지시: 앞으로 작성된 코드는 사용자가 직접 실행하고 결과를 확인한다. 별도 실행 요청이 없으면 Codex는 정적 검토만 수행한다.
- 수행: `AGENTS.md`에 실행 역할과 허용 범위를 명시하고 이 파일의 유지할 결정·다음 작업에 반영했다. 연구 목표와 기존 결과는 유지했다.
- 확인: 기존 연구 문서와 지침을 읽고 새 규칙이 기존 실행 검증 지침보다 우선하도록 문구를 검토했다. 프로젝트 코드·테스트·dry-run은 실행하지 않았다.
- 출력: `AGENTS.md`, `PROGRESS.md`.
- 다음: baseline 준비에 필요한 코드·설정·실행 방법을 정적으로 검토하여 전달하고, 사용자가 제공한 실행 결과를 분석한다. 현재 추가 미해결 문제 없음.

### 2026-10-06 — 파일 구조·프로젝트 설명 검토와 실행 로드맵 정리

- 수행: 연구 계획·진행 기록·작업 지침과 실제 폴더, upstream README, 모델·평가·데이터 로더·GT 생성·crop·데이터 준비 코드를 읽었다. 기존 setup 보고서·로그·manifest를 대조했다. 연구 목표 변경 없음.
- 확인: Raw test 정리 완료 기록과 산출물 존재를 확인했다. `configs/`는 비어 있고 연구용 실행 코드는 데이터 정리 스크립트만 확인된다. Official feature 배열·완성된 UCF checkpoint는 프로젝트 내 미확인이며 CLIP `.part`만 존재한다. 실제 metric·시간 결과는 없다.
- 정적 발견: GT 생성의 `__0.npy` 필터와 CSV의 `__5.npy` 대응 확인 필요. 16배 score 반복은 raw 후보 stride의 확정 근거가 아니다. 정수배 길이의 추가 padded window, temporal mask 미전달, CLIP 기본 cache와 프로젝트 checkpoint 경로, CUDA/CPU 선택도 재현 준비 점검 항목으로 기록했다. 영향은 실행 미검증이다.
- 출력: `docs/PROJECT_ROADMAP.md`. 자원/규칙 확정 → official-feature 재현 → raw reference → Uniform/Random → XD 확장 → Adaptive·failure analysis → 개발 데이터에서 Ours 고정 → 최종 평가의 완료 기준과 파일 관리 제안을 정리했다.
- 검증 범위: 파일 목록·텍스트·기존 로그의 읽기 및 정적 검토만 수행했다. 프로젝트 코드, 테스트, dry-run, import, 배열 로딩, 영상 처리, 다운로드, inference는 실행하지 않았다. 기존 검증 결과를 이번에 재실행한 것으로 취급하지 않는다.
- 미해결/바로 다음: UCF official test features·UCF checkpoint·완성된 CLIP 가중치의 실제 위치를 확인하고 local feature CSV와 GT 정렬 점검 코드·설정·실행 안내를 준비한다. 실행과 결과 확인은 사용자가 수행한다.

### 2026-10-06 — Step 1 자원 경로 점검 준비

- 수행: 연구 계획·진행 기록을 읽고 현재 자원 목록을 재확인했다. `configs/ucf_resources.json`, `scripts/check_ucf_resources.py`, `docs/UCF_RESOURCE_SETUP.md`를 작성했다. 자원 경로는 확인되지 않아 설정의 세 필드를 `null`로 유지했다.
- 확인: 프로젝트 안에 official feature 배열·UCF checkpoint 미확인, CLIP `.part` 존재. 기본 cache 위치 `/Users/apstat/.cache/clip`은 존재하지 않는다. 공식 GitHub README의 UCF feature/model 링크 등록을 웹으로 확인했다. 배포 파일 다운로드·압축 내용은 미검증이다.
- 코드 목적: 설정한 경로에서 공식 CSV/manifest 순서·label을 대조하고 정확한 feature 파일명 290개의 누락·중복·빈 파일, 영상 링크와 checkpoint hash를 점검한다. 유효한 feature 매핑에 한하여 별도 local CSV를 출력한다. 배열·모델 로딩과 GT 정렬·실제 장치 검증은 다음 점검 범위다.
- 정적 검토: 원본 입력을 쓰지 않고 `outputs/setup/`의 파생 보고서·local CSV만 생성하도록 검토했다. 부분 다운로드를 checkpoint로 받지 않고 ViT-B/16 완성본의 기대 SHA-256을 비교한다. GT `__0`/feature `__5` 불일치는 자동 수정하지 않는다.
- 실제 실행: 프로젝트 코드·테스트·dry-run·import·다운로드·모델 로딩은 수행하지 않았다. 작성 코드의 실제 동작은 미검증이다.
- 준비 파일: 위 코드·설정·안내. 사용자 실행 후 예상 산출물은 `outputs/setup/ucf_resource_check.json`과 조건부 `ucf_CLIP_rgbtest.local.csv`이며 아직 생성하지 않았다.
- 바로 다음: 이미 보유한 자원 경로를 확인하거나 공식 배포를 확보한 뒤 설정을 채운다. 사용자가 `.venv/bin/python scripts/check_ucf_resources.py --write-local-list`를 실행하고 결과를 전달하면 배열 형식·GT 정렬·모델 로딩 점검을 이어서 준비한다. Step 1은 아직 완료되지 않았다.

### 2026-10-06 — 사용자 자원 미확보 확인과 저장 폴더 준비

- 사용자 확인: UCF official features와 UCF pretrained checkpoint는 아직 없다.
- 수행: `features/official/ucf/`, `checkpoints/vadclip/` 빈 폴더를 생성하고 존재를 확인했다. 설정의 feature root를 준비 폴더로 지정했으며 실제 모델 파일 경로는 `null`을 유지했다. `docs/UCF_RESOURCE_SETUP.md`에 직접 다운로드 링크와 저장·설정 절차를 보완했다.
- 검증: OneDrive 직접 링크 두 개는 웹 도구에서 접근할 수 없어 다운로드 가능 여부 미확인이다. 공식 README의 링크 등록과 실제 다운로드 성공을 구분한다. 프로젝트 코드·다운로드·압축 해제·배열 로딩·모델 실행은 수행하지 않았다.
- 출력: 준비 폴더, `configs/ucf_resources.json`, `docs/UCF_RESOURCE_SETUP.md`, 이 진행 기록. 준비 폴더는 실제 feature/checkpoint 보유를 뜻하지 않는다.
- 바로 다음: 사용자가 공식 UCF feature/model과 CLIP 완성본을 확보하고 저장한다. 실제 checkpoint 경로를 설정한 후 준비된 자원 점검 코드를 사용자 실행한다. 링크 접근이 실패하면 사용자 오류 내용을 바탕으로 공식 대체 배포 경로를 확인한다.

### 2026-10-06 — UCF feature wget 다운로드 명령 준비

- 사용자 요청: OneDrive의 UCFClipFeatures.zip을 wget으로 다운로드하고 싶다고 요청했다. 프로젝트 실행 역할에 따라 직접 다운로드를 시작하지 않고 사용자 실행 명령을 준비했다.
- 수행: 연구 계획·진행 기록을 읽고, 제공 URL의 파일 경로를 Microsoft 안내의 download.aspx?SourceUrl 형식으로 변환했다. 이어받기·저장 경로·HTTP 응답 출력 옵션과 ZIP/HTML 식별 명령을 docs/UCF_RESOURCE_SETUP.md에 추가했다.
- 확인: wget, file, zipinfo 실행 파일이 로컬에 존재하고 features/official/ucf/는 비어 있다. 제공 OneDrive 화면은 웹 도구 접근 실패. 변환 다운로드 주소의 실제 접근·응답 내용·파일 크기는 미검증이다.
- 검증 범위: URL 구성·shell 인용·저장 위치를 정적으로 검토했다. wget 실행·다운로드·압축 해제·프로젝트 코드 실행은 하지 않았다.
- 출력: docs/UCF_RESOURCE_SETUP.md의 다운로드 절차. 예상 다운로드 파일은 features/official/ucf/UCFClipFeatures.zip이며 아직 존재하지 않는다.
- 바로 다음: 사용자가 준비 명령을 실행하고 ZIP 여부를 확인한다. HTML 저장이나 401/403이면 실행 결과로 접근 상태를 확인한 뒤 공유 링크/다운로드 요청에 맞춰 명령을 수정한다.

### 2026-10-06 — wget HTTP 403 후 공유 session을 사용하는 명령 준비

- 사용자 결과: 앞서 안내한 SourceUrl 단독 wget 요청에서 HTTP 403 발생. 다운로드 성공으로 기록하지 않는다.
- 확인: 원래 공식 공유 링크를 in-app browser로 열어 UCFClipFeatures.zip 페이지와 다운로드 버튼 표시를 확인했다. 버튼은 누르지 않았고 다운로드는 시작하지 않았다. 서버가 wget 요청을 거부하는 상세 원인은 미확정이다.
- 수행: scripts/download_ucf_features.sh 작성. 공식 공유 링크에서 받은 임시 session cookie를 동일 호스트의 다운로드 요청에 전달하도록 준비했다. 기존 실패 파일을 재개하지 않도록 별도 .shared.zip.part 경로를 쓰고 ZIP central directory 확인 후 완료 이름으로 이동한다. 임시 cookie는 종료 시 삭제한다.
- 정적 검토: 원본 데이터·GT·공식 목록을 수정하지 않으며 사용자 브라우저 cookie를 추출하지 않는다. 사용자에게 HTTP 오류를 전달받는 경우를 대비하여 cookie/응답 header를 출력하지 않는다. 전체 CRC·압축 해제·모델 평가 범위는 포함하지 않는다.
- 출력: scripts/download_ucf_features.sh, docs/UCF_RESOURCE_SETUP.md. 예상 다운로드 완료 파일은 features/official/ucf/UCFClipFeatures.shared.zip이며 아직 확보하지 않았다.
- 검증 범위: 파일 읽기·코드 편집·URL 및 shell 흐름의 정적 검토와 공유 페이지 UI 조회만 수행. 스크립트·wget·다운로드·테스트는 실행하지 않았다. Cookie 방식의 실제 성공은 미검증이며 별도 JavaScript/token이 필요하면 추가 대응이 필요하다.
- 바로 다음: 사용자가 bash scripts/download_ucf_features.sh 실행. 실패하면 HTTP 상태와 어느 단계에서 실패했는지 확인하여 실제 브라우저 다운로드 링크/요청을 사용하는 방법으로 이어간다.

### 2026-10-06 — 다운로드 이후 구조 검토와 실제 경로 연결

- 사용자 요청: 현재 구조가 괜찮은지 확인. 연구 계획·진행 기록과 현재 파일 목록·설정을 읽고 대조했다.
- 새 확인: `features/official/ucf/UCFClipFeatures.shared.zip` 12,996,298,147 bytes와 압축 해제된 UCFClipFeatures 폴더 존재. 16개 하위 폴더에 .npy 파일 경로 19,500개. `model_ucf.pth` 649,186,055 bytes와 `ViT-B-16.pt` 350,837,078 bytes 존재. CLIP `.part`는 현재 목록에 없다. 다운로드 방법·전체 압축 검증 성공은 파일 존재만으로 추정하지 않았다.
- 구조 판단: raw 입력, 공식 feature, checkpoint, upstream 코드, 연구 scripts/configs, outputs가 분리되어 현재 단계에 적절하다. 원본 feature 패키지의 category/train/test 폴더는 보존한다. ZIP과 압축 해제 폴더의 공존은 문제로 보지 않는다.
- 파일 목록 정적 대조: 공식 CSV의 test feature 290개 basename 누락 0. 이 중 normal 50개는 다른 두 폴더에서 같은 이름을 가짐. 공식 parent 폴더+파일명 기준으로 대조하면 누락 0이며 전체 feature 목록에서도 동일 parent+파일명 중복 0. Feature/GT 배열은 읽지 않았다.
- 수정: 기존 점검 코드가 basename만 비교하여 이 50개를 중복으로 처리하는 문제를 확인했다. `check_ucf_resources.py`를 공식 parent+파일명으로 매핑하도록 수정했고 비선택 폴더의 동명 파일은 별도 기록한다. `configs/ucf_resources.json`을 확인한 세 실제 자원 경로로 갱신했다.
- 출력: 코드·설정·자원 준비 안내·로드맵·이 진행 기록. 자원 점검 보고서와 local CSV는 아직 생성하지 않았다.
- 검증 범위: 디렉터리/파일 metadata·텍스트 읽기와 파일 목록 대조 및 코드 정적 검토만 수행. 프로젝트 코드·테스트·dry-run·배열 로딩·hash 계산·디코딩·모델 로딩·inference는 미실행이다.
- 바로 다음: 사용자가 자원 점검 명령 실행. 예상은 test feature 290개 매핑 성공, 공식 parent 기준 중복 없음, 비선택 동명 feature 50개 기록이다. CLIP hash와 실제 실행 결과는 사용자 로그로 확인한 뒤 GT/모델 로딩 점검을 이어간다.

### 2026-10-06 — 사용자 자원 점검 통과 확인 및 baseline 입력 점검 준비

- 사용자 실행: check_ucf_resources를 실행했다고 전달. Codex는 재실행 없이 생성된 보고서와 local CSV를 읽었다.
- 실제 보고서 결과: `ucf_resource_check.json`, 생성 시각 2026-10-06 19:59:32 KST. Feature 290개 match, 공식 CSV/manifest 순서·label 일치, missing/duplicate/empty feature 및 missing video 없음. 비선택 normal 동명 feature 50개가 별도 기록됐다. CLIP hash 기대값 일치, UCF checkpoint hash 저장, issues 없음, path_inventory_complete true. Step 1 전체 완료는 아직 false다.
- 산출물 확인: `outputs/setup/ucf_CLIP_rgbtest.local.csv`는 header 포함 291줄. 공식 CSV와 parent+filename/label 순서를 텍스트로 대조하여 차이 없음을 확인했다. 사용자 실행 결과와 Codex의 읽기·정적 검토를 구분한다.
- 다음 코드: `scripts/check_ucf_baseline_inputs.py` 작성. 290개 feature의 shape/dtype/유효값, `__0`/`__5` shape, 공식 feature T*16 길이와 GT, manifest annotation을 공식 slicing 규칙으로 구성한 영상별 GT를 비교한다. 통과 후 선택적 `--check-model`로 CPU 모델 strict loading과 장치 availability를 점검한다. Inference/AUC·raw decode·GT 수정은 수행하지 않는다.
- 모델 경로 연결: `VadCLIP/src/model.py` 생성자에 기본값 None의 `clip_checkpoint` 인수만 추가해 로컬 CLIP 파일을 명시적으로 선택할 수 있게 했다. 기존 호출 기본값·구조·forward·가중치 로딩 규칙은 유지했다. 새 source hash는 다음 사용자 보고서에 기록한다.
- 검증 범위: 사용자 보고서·CSV·코드를 읽고 새 코드 흐름을 정적으로 검토했다. Codex는 새 코드·import·테스트·hash 계산·모델 로딩·inference를 실행하지 않았다. 사용자 자원 점검을 모델 실행 성공으로 확장하여 기록하지 않는다.
- 출력: 새 입력 점검 코드, 모델의 선택적 경로 인수, `docs/UCF_RESOURCE_SETUP.md`와 진행 기록. 다음 사용자 실행의 예상 보고서는 `outputs/setup/ucf_baseline_input_check.json`이며 아직 미생성이다.
- 바로 다음: 사용자가 `.venv/bin/python scripts/check_ucf_baseline_inputs.py --check-model` 실행. 데이터/GT/모델 loading이 통과하면 작은 입력의 inference/window/padding 검증과 전체 official-feature AUC 평가 준비로 이어간다.

### 2026-10-06 — 사용자 baseline 입력 점검 통과와 소규모 inference 준비

- 사용자 실행 결과 확인: `outputs/setup/ucf_baseline_input_check.json`, 생성 시각 20:05:51 KST. Data checks true, issues 없음. 290개 feature finite `[T,512]`, 286 float16+4 float32, __0/__5 shape 일치. 
- GT 검증 결과: 1,109,888 float32 binary labels. Feature T*16 timeline 길이와 일치하고 manifest annotation을 공식 [start:end]로 비교한 mismatch 0. Raw-video 좌표 규칙까지 검증된 것으로 확장하지 않는다. 
- 모델/장치 결과: torch 2.8.0, CPU strict loading 성공 (classes14, visual_length256, width512, 349 entries). CUDA unavailable, MPS available. Forward/inference는 아직 미실행. 
- 정적 발견/수정: DistanceAdj.forward가 `.to('cuda')`를 고정 사용하여 CPU/MPS forward가 실패할 수 있음을 확인. 모델 parameter 장치를 사용하도록 수정했다. 원래 거리 행렬/exp 상수·식은 유지. 관측된 사용자 runtime 오류가 아니라 정적 발견이며 실제 수정 검증은 다음 smoke에서 수행한다. 
- 코드 준비: `scripts/evaluate_ucf_official.py`에 smoke/full 모드를 작성. 원본 split, 길이 선언, mask, prompt, branch와 16배 반복 규칙 유지. Smoke는 실제 anomaly/짧은normal/긴 영상 및255/256/257/512 token prefix 7사례이며 AUC 미계산. Full은 동일 source/config/device/checkpoint의 smoke 통과 후 test290 전체 frame AUC/AP를 두 branch별 저장. 
- 출력 설계: 실행별 `outputs/baseline/<run_id>/summary.json`와 scores, 실패 traceback. Smoke 최신 요약은 `outputs/setup/ucf_smoke_check.json`. 실행 source/input/checkpoint hash, base commit, device/precision, 옵션/prompt를 기록. Cached-feature 시간은 raw 효율 근거로 사용하지 않는다. 
- 출력 준비 파일: 새 평가 코드, DistanceAdj 장치 연결, `docs/UCF_OFFICIAL_BASELINE.md`, 진행 기록. 새 inference 산출물·metric은 아직 없다. 
- Codex 검증 범위: 사용자 보고서/기존 코드 읽기와 정적 검토·편집만 수행. 프로젝트 코드·테스트·import·forward·inference는 실행하지 않았다. 
- 바로 다음: 사용자가 CPU smoke 명령 실행. 통과하면 동일 device 전체 official-feature 평가로 이어간다. MPS 평가를 선택하면 먼저 동일 코드의 MPS smoke를 별도 실행하고 device/precision을 기록한다. 

### 2026-10-06 — CUDA 수정 내용 재검토

- 사용자 요청에 따라 upstream 대비 `src/utils/layers.py` 변경을 정적으로 확인했다. DistanceAdj의 CUDA 고정 전송 두 곳을 제거하고 `self.sigma.device`로 모델과 같은 장치를 선택한다. exp(1) 상수도 거리 행렬 장치로 이동한다.
- `sigma`의 값은 계산에 사용하지 않는다. 거리 계산, float32 변환, `exp(-distance / exp(1))`, batch 반복과 checkpoint parameter 구성은 유지된다. `model.py`의 별도 변경은 로컬 CLIP checkpoint 경로 선택 인수다.
- 검증 범위: 코드와 Git diff 읽기만 수행. 프로젝트 코드·테스트·forward는 실행하지 않았으며 CPU/CUDA/MPS 실제 inference 성공과 장치 간 수치 일치는 미검증이다. 새 inference 결과 파일 없음.
- 바로 다음: 사용자가 `.venv/bin/python scripts/evaluate_ucf_official.py --mode smoke --device cpu`를 실행하여 7사례 통과와 `outputs/setup/ucf_smoke_check.json`의 성공 상태를 확인한다.

### 2026-10-06 — Mac MPS 장치 선택 안내

- 평가 코드의 기본 장치는 CPU이며 자동 MPS 선택은 하지 않음을 정적으로 확인했다. `--device mps`를 지정하면 availability 검사 후 모델과 입력을 MPS로 이동하고 시간 측정에서 MPS를 동기화한다.
- 기존 사용자 보고서의 MPS available=true는 추론 통과를 의미하지 않는다. 코드 실행·테스트는 수행하지 않았고 새 결과 파일도 없다.
- MPS를 사용할 경우 바로 다음 사용자 실행은 `.venv/bin/python scripts/evaluate_ucf_official.py --mode smoke --device mps`다. `outputs/setup/ucf_smoke_check.json`의 device=mps, success=true와 7사례 통과를 확인한 뒤 동일 MPS 장치의 full 평가를 진행한다.

### 2026-10-06 — 사용자 MPS smoke 7사례 통과 확인

- 사용자 완료 안내 후 기존 `outputs/setup/ucf_smoke_check.json`을 읽었다. Run `ucf_official_smoke_mps_20261006_202116_841716`, device=mps, success=true, metrics=null. Codex는 재실행하지 않았다.
- 실제 anomaly(88 tokens), 짧은 normal(13), 긴 영상(540) 및 255/256/257/512 prefix의 7사례 모두 finite score, [0,1] 범위, score/GT 복원 길이 검사 통과. 256/512 정수배의 추가 padded window는 원본 정책대로 기록되어 있다. 장치 간 동등성이나 전체 AUC 통과로 확장하지 않는다.
- 출력 확인: 해당 run의 `summary.json` 및 최신 smoke JSON. 사례별 NPZ 경로가 보고서에 기록되어 있으며 NPZ 내용은 Codex가 로딩하지 않았다. VAD/CLIP parameter precision은 float32다. 보고서의 시간은 cached-feature 실행 범위다.
- 수행/검증: 사용자 결과 파일·문서 읽기와 진행 기록/실행 안내 갱신만 수행. 코드·테스트·import·inference는 실행하지 않았고 평가 코드는 변경하지 않았다.
- 미해결/바로 다음: 사용자가 `.venv/bin/python scripts/evaluate_ucf_official.py --mode full --device mps` 실행. 예상 출력 `outputs/baseline/ucf_official_full_mps_<timestamp>/summary.json`과 `frame_scores.npz`. 성공 여부, 영상 290개, 평가 frame 1,109,888 및 두 branch AUC/AP를 확인한다. 전체 metric은 아직 없다.

### 2026-10-06 — GitHub 팀 공유·지속 커밋 관리 방식 검토

- 사용자 요청: 현재 진행 내용을 GitHub에서 팀원과 공유하고 이후에도 지속 커밋하기 위한 관리 방식 상담. 이번 요청에서는 저장소 생성·커밋·push를 수행하지 않았다.
- 확인: 프로젝트 루트는 Git 저장소가 아니고 루트 `.gitignore`·`README.md`·의존성 명세가 없다. `VadCLIP/`만 공식 `https://github.com/nwpu-zxr/VadCLIP.git`에 연결된 별도 Git 저장소이며 base commit은 기존 기록과 같다. `src/model.py`, `src/utils/layers.py`에 미커밋 변경이 있다. 원본 LICENSE 존재를 확인했다.
- 제안(아직 적용·확정하지 않음): 별도 private 연구 저장소 하나에서 scripts/configs/docs/연구 문서와 수정한 VadCLIP 소스를 함께 관리한다. 기존 내부 Git 이력과 변경을 백업하고 원본 URL/commit/라이선스 및 변경 내역을 보존한 뒤 일반 하위 폴더로 통합하는 방식을 우선 제안한다. 원본 업데이트를 자주 병합해야 하면 별도 저장소+submodule/subtree를 재검토한다.
- 공유 범위 제안: raw videos, feature 배열/ZIP, checkpoints, 가상환경/cache, 영상 symlink와 원시 outputs는 Git에서 제외한다. 작은 실험 요약은 별도 results 폴더에서 코드 commit·설정·checkpoint hash·장치·실행 상태와 함께 공유한다. 데이터 확보/폴더 재구성 절차와 의존성 명세를 README로 안내하며 로컬 절대 경로가 담긴 결과를 그대로 팀의 실행 설정으로 사용하지 않는다.
- 정적 발견: 평가 코드는 현재 `git -C VadCLIP rev-parse HEAD`를 `upstream_commit`으로 저장한다. 일반 하위 폴더 통합 시 연구 저장소 HEAD를 원본 commit으로 오인할 수 있어 통합 작업에서 연구 commit과 고정 upstream commit 기록을 분리해야 한다. 이번에는 코드를 변경하지 않았다.
- 협업 제안: 작업별 짧은 branch에서 작은 단위로 commit/push하고 PR 검토 후 main에 병합한다. 미실행 코드와 사용자 검증 완료를 구분하고 주요 재현 단계에는 tag를 남긴다. 다음 할 일·담당자는 Issues, 실제 상태·실험 증거는 PROGRESS에 기록한다.
- 검증/출력: 문서·설정·기존 smoke JSON·파일 목록과 Git 상태를 읽고 GitHub/Git 공식 문서를 확인했다. 프로젝트 코드·테스트·import·inference는 실행하지 않았다. 수정 파일은 이 진행 기록뿐이며 원격 산출물은 없다.
- 바로 다음: 공유 준비 시 루트 README·제외 규칙·의존성 명세·팀 공유용 결과 요약과 원본 코드 출처 기록을 작성하고 중첩 Git을 백업·정리한 뒤 업로드 대상 목록을 검토한다. 연구 실행의 다음 작업은 기존 MPS 전체 official-feature 평가다.

### 2026-10-06 — GitHub 공유 파일 정리와 연구 Git 초기화

- 사용자 승인/대상: 프로젝트를 정리하여 GitHub에 올리라는 요청을 받았으며 `https://github.com/ApStatsm/vad-frame-sampling` 저장소 주소를 사용자가 제공했다.
- 수행: 루트 `.gitignore`, `README.md`, 설치된 직접 의존성 metadata 기준 `requirements.txt`, `CONTRIBUTING.md`, PR template, `docs/TEAM_SETUP.md`와 데이터/feature/checkpoint 안내를 작성했다. 기존 raw 영상·ZIP·feature·weights·가상환경·outputs는 보존하고 Git에서 제외했다. 작은 annotation·test 목록과 upstream GT/tokenizer는 공유 입력으로 포함한다.
- 원본 코드 통합: 기존 `VadCLIP/.git`을 `.local-backups/vadclip-before-integration/git-history/`로 이동하여 이력을 보존했다. 수정 patch와 두 수정 파일 사본도 같은 백업에 저장했다. 루트 `main` 연구 Git을 초기화하고 `VadCLIP/`을 일반 하위 소스 폴더로 관리한다. 원본 URL/commit은 `VadCLIP/UPSTREAM.json`, 변경 내역은 `LOCAL_CHANGES.md`에 남기고 LICENSE/README/원본 파일을 보존했다. 두 변경 파일에 변경 표시 주석을 추가했다.
- 실행 버전 기록: `evaluate_ucf_official.py`가 고정 upstream commit과 research commit/tracked dirty 상태를 별도로 기록하도록 수정했다. 원본 provenance 파일 hash도 smoke/full 비교에 포함했다. 모델 계산·가중치·prompt·window·score 복원은 변경하지 않았다.
- 기존 결과 공유: `results/setup_20261006/summary.json`과 `results/ucf_official_smoke_mps_20261006_202116_841716/summary.json`을 기존 사용자 JSON 보고서에서 추렸다. 원시 보고서를 덮어쓰지 않았고 공유 요약의 실행 경로는 프로젝트 상대 경로로 표시했다. 새 실행이나 benchmark metric은 없다. 기존 run은 연구 Git 초기화 이전이므로 research commit은 null이고 당시 source hash를 보존한다.
- 검증/실행 범위: 파일·설치 metadata·기존 JSON 읽기, Git 상태/제외 규칙 확인 및 코드 정적 검토만 진행했다. 프로젝트 코드·테스트·dry-run·import·배열 로딩·영상 처리·inference·환경 재설치는 실행하지 않았다. 변경 표시/버전 기록 때문에 현재 source hash는 과거 smoke와 달라져 현재 버전 smoke 재실행이 필요하다. 원본 입력/config/GT는 수정하지 않았다.
- GitHub 연결 상태: 저장된 HTTPS 인증 없음, 기존 SSH 키는 GitHub에서 Permission denied(publickey). GitHub 공식 서버 키를 대조했으며 시스템 SSH 파일은 변경하지 않았다. 공식 GitHub CLI 2.102.0을 Git 제외 로컬 도구 폴더에 준비하고 배포 SHA-256을 확인했다. 사용자의 device 로그인 완료를 기다리는 중이며 이 기록 작성 시점에는 remote push 미완료다.
- 업로드 사전 확인: 첫 commit 대상 75개, 총 21,583,771 bytes(진행 기록 보완 전 metadata 합계), 최대 파일은 원본 XD GT 9,325,312 bytes. 영상 symlink·대용량 입력·가상환경·outputs·이력 백업·로컬 인증 도구가 제외되었고 알려진 인증 token/private-key 표식 검사에서 후보가 없었다. 수정한 evaluator/model/layers는 AST 파싱으로 정적 문법만 확인했고 provenance/공유 요약 JSON 텍스트도 파싱했다. 프로젝트 코드의 실행 검증으로 취급하지 않는다.
- 바로 다음: 업로드 파일 목록·크기·정적 문법을 확인하여 첫 commit을 만들고 인증 후 지정 저장소로 push한다. 연구 실행은 사용자가 `.venv/bin/python scripts/evaluate_ucf_official.py --mode smoke --device mps`의 7사례 success를 확인한 뒤 동일 장치 full 평가를 실행한다. 새 팀원은 `docs/TEAM_SETUP.md`의 입력 준비부터 수행한다.

### 2026-10-06 — 첫 로컬 commit 완료 / 원격 인증 대기

- 완료: `main`의 첫 commit `6d43d48` (`chore: initialize VAD frame sampling research repository`), 75개 파일. `origin`은 사용자가 제공한 `https://github.com/ApStatsm/vad-frame-sampling.git`이다. 내부 소스를 Git link(160000)로 처리한 항목이 없으며 첫 commit 직후 tracked 작업 트리는 clean이다.
- 검증: AST/JSON 텍스트 파싱과 업로드 목록·제외 규칙·크기 확인 통과. 프로젝트 코드·테스트·inference·데이터 처리·환경 재설치는 미실행이다. 원격 커밋 및 저장소 visibility는 아직 인증 후 확인해야 한다.
- 남은 작업: GitHub CLI device 인증 요청을 시작하여 사용자 로그인 완료를 기다린다. 로컬 도구는 `.local-backups/tools/gh/gh_2.102.0_macOS_arm64/bin/gh`, 인증 설정 위치는 `.local-backups/github-cli-config/`로 Git에서 제외된다. 인증 정보는 채팅/커밋에 포함하지 않는다. 이 기록 시점에 push는 미완료다.
- 바로 다음: 로그인 후 GitHub API로 계정/대상 저장소·기존 branch를 확인하고 HTTPS 인증을 연결한 뒤 force 없이 push한다. 이후 remote main commit과 로컬 HEAD의 일치, 추적 상태·작업 트리를 확인하여 업로드 완료를 기록한다. 팀원 초대는 계정 식별 정보가 아직 없으므로 별도 사용자 작업이다.

### 2026-10-06 — 사용자 터미널에서 직접 업로드하도록 인계

- 최신 지시: GitHub 인증·업로드를 사용자가 자신의 터미널에서 직접 수행하도록 안내한다. Codex가 진행하던 device 인증 대기 세션은 종료했으며 이전 일회용 코드는 사용하지 않는다.
- 확인: 기존 로컬 commit `6d43d48`, `4936d4b`와 HTTPS origin 연결을 확인했다. 인계 기록 추가 전 작업 트리는 clean이다. Homebrew는 존재하고 전역 `gh`는 아직 없다. 다운로드한 로컬 GitHub CLI는 보존한다.
- 안내: 사용자가 Homebrew로 GitHub CLI를 설치하고 `gh auth login --hostname github.com --git-protocol https --web`, `gh auth setup-git --hostname github.com`으로 인증한다. 이번 PROGRESS 인계 기록을 commit한 뒤 `git push -u origin main`을 실행한다. 이미 첫 코드 commit과 origin을 준비했으므로 git init/remote add는 반복하지 않는다.
- 검증 범위/출력: 문서·Git 상태·로컬 도구 존재와 공식 CLI 안내만 읽었다. 이 기록만 수정했으며 Codex는 추가 commit·설치·인증·push 또는 프로젝트 코드 실행을 하지 않았다. 원격 업로드는 아직 미완료다.
- 바로 다음: 사용자가 자신의 터미널에서 안내 순서를 실행하고 업로드 결과를 확인한다. 인증 실패 또는 non-fast-forward 등 오류가 나오면 force push 없이 오류 메시지를 바탕으로 대응한다. 연구 실행의 다음 작업은 현재 버전 MPS smoke 후 full 평가다.

### 2026-10-06 — 기존 파일 중심으로 공유 범위 단순화

- 최신 지시: 기존 GitHub 저장소에 현재 로컬 파일을 올리되 대용량은 제외하고 제외 자원의 구조·양식을 README에 설명한다. 별도 협업 문서나 공유용 결과 사본을 늘리지 않는다. 인증/업로드는 앞선 지시에 따라 사용자 터미널에서 직접 진행하도록 안내한다.
- 정리: 앞서 추가한 CONTRIBUTING, PR template, TEAM_SETUP, 데이터 폴더별 중복 README와 results 요약 사본을 작업 트리에서 걷어내고 `.local-backups/share-cleanup-20261006/`에 보관했다. 기존 연구 문서·코드·원본 LICENSE/출처/변경 내역은 보존했다. 원본 영상·feature·checkpoint·원시 결과는 삭제하지 않았다.
- README: 현재 코드 구조, 제외 영상 ZIP/extracted/test symlink, feature 폴더·파일명·[T,512]/dtype, checkpoint, NPZ score key/shape, 가상환경과 작은 manifest/CSV/annotation/보고서의 형식을 루트에 모았다. 입력 배치·재생성 명령과 실행 조건을 기존 실제 파일/코드에서 정적으로 대조했다.
- Git 제외 규칙: 대용량 입력 폴더·영상·weights·ZIP·score NPZ·가상환경/cache/백업은 제외한다. 작은 기존 manifest·split·annotation 및 outputs의 JSON/CSV/TXT/LOG는 원래 위치에서 공유한다. 기존 보고서의 절대 경로와 hash는 해당 사용자 실행의 기록이며 다른 컴퓨터의 통과 audit를 대신하지 않는다고 설명했다.
- GitHub 연결 확인: origin은 이미 `https://github.com/ApStatsm/vad-frame-sampling.git`에 연결되어 있고 `gh` 설치는 현재 확인했다. `gh auth status`는 미로그인을 보고했다. 추가 인증 세션이나 원격 쓰기는 시작하지 않았다.
- 검증 범위/출력: 문서·Git 상태·기존 파일 목록/형식과 코드 읽기, README/제외 규칙 정리만 수행했다. 프로젝트 코드·테스트·import·inference는 미실행이며 이번에는 모델/평가 코드도 변경하지 않았다. 정리된 파일은 아직 미커밋이며 원격 업로드도 미완료다.
- 최종 공유 후보 정적 확인: 파일 76개, 진행 기록 마지막 보완 전 총 22,381,459 bytes. 최대 원본 GT 파일 9,325,312 bytes이며 100 MiB 초과 파일·영상 symlink·제외 입력/백업·알려진 인증 token/private-key 표식 후보 없음. README의 로컬 링크 모두 존재하고 작은 기존 outputs 보고서 8개와 manifest 2개가 포함된다. Git 제외 자원 5개 경로를 대표로 확인했다. 실제 push 검증은 아니다.
- 바로 다음: 최종 공유 후보 목록의 크기·제외 자원·README 연결을 정적으로 확인한 후 사용자가 프로젝트 루트에서 `gh auth login --hostname github.com --git-protocol https --web`, `gh auth setup-git --hostname github.com`, `git add -A`, `git commit`, `git push -u origin main`을 실행한다. 설치/저장소 생성/remote add는 다시 할 필요 없다.

### 2026-10-06 — 사용자 터미널의 인증 저장 권한·실행 위치 오류

- 사용자 화면: 브라우저 authentication complete 뒤 `/Users/apstat/.config/gh` 생성 permission denied, 이후 not logged in 및 Git 명령의 not a git repository가 보고됐다. 업로드 성공으로 기록하지 않는다.
- 읽기 확인: `/Users/apstat/.config`는 root 소유의 755 디렉터리이고 gh 하위 폴더는 없다. 프로젝트 루트에는 기존 Git/HTTPS origin과 미커밋 공유 정리가 정상 존재한다. Git 명령의 오류는 사용자 터미널의 프로젝트 밖 실행으로 해석한다.
- 다음 안내: 사용자가 먼저 실제 프로젝트 루트로 이동하고 `.config` 디렉터리 자체의 소유자만 자신의 계정으로 바꾼 뒤 로그인/인증 연결을 재실행한다. 성공 후 기존 정리를 commit/push한다. Codex는 권한 변경·로그인·commit/push나 프로젝트 코드 실행을 하지 않았으며 이 진행 기록만 보완했다.

### 2026-10-06 — GitHub 업로드 완료 확인

- 사용자 터미널에서 생성한 commit `545d8ef6c0e9de6e4aa8acc2366c73c10d768130` (`chore: simplify sharing and document excluded data`)을 확인했다. GitHub origin의 실제 main을 읽어 로컬 HEAD 및 origin/main과 같은 commit임을 확인했다. 확인 시작 시 작업 트리는 clean이었다.
- 최신 commit에는 정리된 기존 파일과 README가 있으며 제외 영상·feature·weights·score·가상환경/백업 및 걷어낸 중복 공유 파일은 추적되지 않는다. Codex는 Git/파일 읽기만 수행했으며 프로젝트 코드 실행이나 추가 commit/push는 하지 않았다. 이 업로드 확인 기록은 로컬에서 추가되었으므로 다음 일반 commit에 포함하면 된다.
- 다음 연구 작업: 사용자가 현재 버전 MPS smoke를 통과한 뒤 full official-feature 평가를 실행한다. 전체 metric은 여전히 미측정이다.
