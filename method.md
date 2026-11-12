# D·A·E-optimal 프레임 샘플링의 baseline 적용 사양

작성일: 2026-11-12  
대상 코드: HJ / `41fda1951f0026287f37b7c404db2a9fdd896eba`  
근거: `VAD_Optimal_Sampling_Handoff.md`, 현재 baseline 코드와 공용 서버의 기존 summary  
상태: 구현 예정 사양. 이 문서 작성으로 코드·설정·패키지를 변경하거나 실험을 실행하지 않았다.

## 1. 목표와 적용 범위

UCF-Crime raw 영상에서 CLIP 입력 프레임을 줄이면서 탐지 성능 하락을 최소화한다. D-, A-, E-optimal을 독립 실험군으로 구현하고 동일 budget의 Uniform·Random과 비교한다. 프레임 감소와 실제 처리 시간 감소를 구분하며, 저비용 특징 수집과 최적화 비용도 자원 평가에 포함한다.

CLIP·VadCLIP 가중치, prompt, 전처리, feature dtype, window·mask, score branch와 평가 GT는 고정한다. 이번 적용의 우선 범위는 UCF다. XD adapter는 보존하고, XD의 전처리를 UCF와 같게 바꾸지 않는다. XD 확장은 별도 검증 후 수행한다.

기존 GitHub 코드는 현재 상태로 보존한다. 아래 파일명과 설정 키 중 ‘신설’로 표시한 것은 향후 구현 제안이며 현재 사용 가능한 기능이나 CLI가 아니다.

## 2. 현재 baseline과 연결 위치

| 역할 | 현재 파일 | 적용 원칙 |
|---|---|---|
| 공통 진입점 | `scripts/run_experiment.py` | 새 방법도 이 진입점을 사용 |
| 준비·실행·측정·저장 | `vadfs/pipeline.py`, `vadfs/preparation.py` | 새 단계와 기록을 연결 |
| 후보·budget·seed·선택 검증 | `vadfs/samplers/__init__.py` | 정확한 K와 시간순·중복 없음 검사를 유지 |
| 기존 선택 규칙 | `vadfs/samplers/{reference,uniform,random}.py` | Uniform·Random 규칙을 변경하지 않음 |
| raw decoding·CLIP encoding | `vadfs/extract.py` | 기존 전처리와 tail 처리를 보존 |
| pretrained 추론 | `vadfs/vadclip.py` | 같은 모델과 window 동작 사용 |
| 전체 후보 grid 복원 | `vadfs/timeline.py` | 모든 방법에 nearest 적용 |
| 전체 평가 | `vadfs/evaluation.py`, `vadfs/metrics.py` | 기존 evaluator 재사용 |
| 반복 집계 | `scripts/analyze/aggregate_runs.py` | sampler 설정과 시간 반복을 구분하도록 확장 |

`scripts/evaluate_ucf_raw.py`는 legacy 진단 경로다. 신규 방법을 이 경로에만 구현하거나 legacy audit와 공통 pipeline의 준비 manifest를 혼용하지 않는다.

현재 `choose(candidates, method, ratio, seed)`는 후보 위치만 받고 decoding 전에 호출된다. 따라서 영상 특징이 필요한 D·A·E를 METHODS에 등록하는 것만으로는 적용할 수 없다. 특징 수집과 선택을 연결하는 별도 단계가 필요하다.

## 3. 입력과 사전 준비

### 서버에서 확인한 자원

| 자원 | 경로 |
|---|---|
| UCF 영상 | `/workspace/vad/datasets/UCF-Crime/videos` |
| UCF official feature | `/workspace/vad/features/official/ucf/UCFClipFeatures` |
| CLIP | `/workspace/vad/checkpoints/clip/ViT-B-16.pt` |
| UCF VadCLIP | `/workspace/vad/checkpoints/vadclip/model_ucf.pth` |
| 공식 평가 목록 | `VadCLIP/list/ucf_CLIP_rgbtest.csv` |
| frame GT | `VadCLIP/list/gt_ucf.npy` |
| segment/class GT | `VadCLIP/list/gt_segment_ucf.npy`, `VadCLIP/list/gt_label_ucf.npy` |
| 고정 reference metadata | `configs/datasets/ucf_reference.json` |

UCF는 정상 150개·이상 140개, 총 290개를 공식 순서로 평가한다. 공식 feature는 부모 폴더명과 `__5.npy` 이름으로 대응하며 `[T,512]` 부동소수점 배열이어야 한다. 원본 영상은 기존 준비 코드의 SHA·크기 검사를 통과해야 한다.

컴퓨터 경로는 `configs/local/`에 두고 새 준비 결과는 `.local/setup/ucf/`, 새 실험은 `outputs/runs/`에 둔다. 공용 데이터·checkpoint·기존 결과를 덮어쓰지 않는다. 다른 checkout의 절대 경로를 담은 manifest를 그대로 복사해 쓰지 않는다.

### 표준화·PCA artifact — 신설

테스트와 분리된 학습/개발 영상에서 한 번 적합하고, 테스트 실행에서는 변환만 허용한다. 현재 확인한 공용 UCF 영상은 test 290개이므로 **적합용 영상 위치와 분리 목록을 먼저 확보해야 한다.** test 영상으로 대신 적합하지 않는다.

artifact에는 학습 영상 ID·파일 hash·split, 특징 정의, 표준화 평균/scale, 제거한 상수 차원, PCA 평균/components, whitening 여부, dtype, 코드·설정 hash를 저장한다. 파일 자체의 SHA-256도 run에 기록한다. 적합용 파일과 test 파일의 중복을 ID와 가능한 content hash로 확인한다.

## 4. 후보와 budget 계약

영상별 후보 수 N은 manifest의 `reference_tokens`를 사용한다. 후보는 기존 UCF 설정과 동일한 frame `16k`, offset 0이며, tail은 `match_official_rows` 규칙을 유지한다. 원본 모든 프레임을 100%의 분모로 삼지 않는다.

`K = min(N, max(1, ceil(b × N)))`, budget b는 1, 0.5, 0.25, 0.125다. N=0은 오류다. 모든 sampler는 같은 영상·budget에서 정확히 K개를 중복 없이 선택하고 시간순으로 반환한다.

**실제 K=N이면 방법명과 관계없이 저비용 특징 계산·PCA·최적화를 우회하고 기존 Reference 추출 경로를 사용한다.** 짧은 영상에서 b<1이어도 이 조건이 성립할 수 있다. `full_budget_bypass`와 사유를 기록한다. 현재 baseline의 Uniform/Random 100% 명시 선택 경로와 비교하는 진단 검증도 유지한다.

Uniform은 현재 구현의 정수 연산 중앙 bin 선택을 그대로 사용한다. Random은 현재 `video_seed(base_seed, dataset, filename)`와 로컬 RNG의 비복원 선택을 유지한다. 선택 단계에 GT·category·anomaly 여부를 전달하지 않는다.

## 5. decoding과 선택의 구현 구조

첫 구현은 전체 후보 특징을 먼저 모으는 offline 방식이다. 메모리에 전체 해상도 프레임을 모두 저장하는 방식은 피하고 다음 두 pass를 명시적으로 구현한다.

1. 첫 pass: 기존 Reference와 같은 decoder·후보·tail 규칙으로 읽고 후보별 저비용 특징만 저장한다. 후보 프레임 ID와 명목 timestamp를 함께 보존한다.
2. 특징 변환 및 선택: 고정 artifact로 표준화·PCA를 적용하고 영상별 공통 차원 처리 후 D·A·E를 실행한다.
3. 두 번째 pass: 기존 `extract_reference`의 선택 인덱스 경로로 동일 영상을 읽어 선택한 프레임만 CLIP에 입력한다.
4. 선택 feature를 기존 VadCLIP과 전체 점수 복원·평가에 전달한다.

Reference·Uniform·Random은 불필요한 첫 pass를 추가하지 않는다. D·A·E의 두 pass 비용은 그대로 비교 비용에 포함한다. 별도 캐시를 사용하는 성능 진단은 raw end-to-end 효율 실험과 구분한다.

UCF의 현재 decoder 경로는 FFmpeg YUV420p와 기존 x86 색 변환 계산을 사용한다. 특징 수집에서도 후보 RGB 변환 기준을 명시적으로 재사용하고, CLIP 입력의 기존 resize/crop/flip 계산은 변경하지 않는다. 두 pass의 후보 목록·실제 decode 수·tail이 일치하지 않으면 실패 처리한다.

신설 모듈 제안은 `vadfs/sampling_features.py`와 `vadfs/samplers/{d_optimal,a_optimal,e_optimal}.py`다. 선택기는 라벨 없는 특징 행렬과 후보 좌표·K·수치 설정만 받으며, 선택 positions와 진단 정보를 반환한다. 반환값은 기존 budget·bounds·중복·정렬 검사와 동일한 검사를 통과해야 한다.

## 6. 공통 특징과 D·A·E 계산

각 후보에서 64×64 grayscale을 만들고, 8×8 격자 평균 64차원과 직전 후보 대비 절대 차이의 4×4 격자 평균 16차원을 연결한다. 첫 후보의 움직임은 0이다. grayscale 변환식, resize 보간, 입력 범위, 계산 dtype과 격자 순서를 구현 설정에 고정한다. 초기 영상 특징을 바꿔 CLIP 전처리까지 변경하지 않는다.

고정 표준화·PCA로 8차원 z를 얻으며 whitening은 끈다. 영상별 비중심 SVD로 유효 랭크 r을 구하고, 공통 차원을 `d=min(8,K,r)`로 둔다. 필요할 때 같은 후보 특징의 상위 방향으로 투영한다. 초기 상대 랭크 허용오차는 `1e-8 × 최대 특이값`이다. r=0이면 Uniform으로 대체하고 사유를 기록한다. 모든 선형대수는 float64, 초기 λ=1을 사용한다. 이 값은 초기 설정안이며 개발 데이터에서 확정한다.

`M(S)=λI+Σ z_i z_iᵀ`이며 이상 점수나 VadCLIP Fisher 행렬로 해석하지 않는다.

| 방법 | 선택 구현 |
|---|---|
| D | 빈 집합에서 시작. `Mu_i=z_i`를 풀고 `log1p(z_iᵀu_i)` 최대 후보를 추가 |
| A | 빈 집합에서 시작. `||u_i||²/(1+z_iᵀu_i)` 최대 후보를 추가 |
| E | 기존 Uniform K개에서 시작. 교환 후 실제 최소 고유값 개선이 최대인 쌍을 적용 |

D·A는 단계별 Cholesky와 배치 우변 풀이를 공유하고 후보마다 역행렬을 만들지 않는다. 유효 동률은 앞선 원본 후보로 결정한다. NaN·분해 실패를 무조건 정상 결과로 바꾸지 않고 오류 또는 명시적 fallback으로 기록한다.

E 초기 상한은 라운드당 4,096쌍·10라운드다. 전체 쌍이 상한 이하면 전수 평가하고, 초과하면 현재 최소 고유벡터 투영 제곱이 작은 선택 후보 64개와 큰 미선택 후보 64개를 조합한다. 후보 축소의 동률도 시간순으로 고정한다. 최소 고유값 개선 허용오차는 `1e-10 × max(1,||M||₂)`다. 최소 고유값 동률에서 log determinant가 엄격히 개선되는 교환만 보조적으로 허용하고 횟수를 기록한다. 교환 쌍 간 최종 동률 순서와 logdet 수치 허용오차도 설정에 명시한다. E는 교환 근사해이며 최적해 보장을 주장하지 않는다.

영상별로 rank·d·fallback, 초기/최종 목적값, 평가 쌍 수, 교환 수, 보조 교환 수와 종료 사유를 저장한다. 선택·교환 후 누적 행렬은 필요 시 선택 집합에서 재구성하여 수치 오차를 점검한다.

## 7. 추론과 평가 — 기존 계산 유지

선택 feature는 시간순으로 압축해 기존 256-token window에 넣는다. 모델 positional embedding에 원본 timestamp를 새로 주입하지 않는다. `process_split`의 256 배수 추가 padded window와 temporal module의 padding mask 무시 동작은 baseline 재현을 위해 보존한다. 이를 수정하는 별도 실험을 sampling 변경에 섞지 않는다.

복원은 `vadfs/timeline.py`의 nearest를 모든 방법에 동일하게 사용한다. 원본 frame index에서 전체 Reference grid로 Branch1·Branch2·클래스 점수를 복원하고, token마다 16번 반복한다. 동률은 앞선 관측, 양 끝은 최근접 관측을 사용한다. 현재 timestamp는 frame index/fps이며 CFR 가정 아래 frame 좌표 nearest와 같다. decoder PTS 기반 복원으로 표현하지 않는다.

평가는 전체 공식 GT 시간축에서 수행한다. UCF 주 지표는 Branch1 frame ROC-AUC, 보조는 AP·Ano-AUC다. Ano-AUC는 이상 영상 전체 프레임을 모으며 그 안의 정상 구간도 포함한다. 정상 영상 포함 290개와 기존 fine-grained 14-class evaluator·GT·`excludeNormal=False`도 유지한다. frame/segment GT를 sampling 결과에 맞춰 줄이거나 다시 생성하지 않는다.

raw decoded frame 수와 공식 feature 기반 GT 길이가 다를 수 있는 기존 제한점은 유지·기록한다. 긴 float32 feature 4개에서 알려진 official/raw 불일치를 sampling 구현 과정에서 임의 보정하거나 해당 영상을 제외하지 않는다.

성능 차이는 두 필드를 분리한다.

- `delta_pp = 100 × (sampled − raw_reference)`: 양수는 개선. 기존 `primary_delta_pp` 의미 유지.
- `drop_pp = 100 × (raw_reference − sampled)`: 양수는 하락. 최종 표의 하락량에 사용.

동일 budget Uniform 대비 차이도 별도로 계산한다. 분수 metric을 %로 표시할 때만 100을 곱하며 pp와 상대 변화율을 혼동하지 않는다.

## 8. Coverage 진단의 분모

기존 half-open interval `[start,end)` 경계 규칙을 유지한다. 평가 유효 구간은 annotation을 영상별 `[0,evaluation_frames)`에 교차시킨 비어 있지 않은 구간으로 정의한다. clipping과 제외된 빈 구간 수를 기록하고 기존 원본 interval coverage도 구분해 보존할 수 있다.

두 분모를 함께 보고한다.

1. 전체 유효 구간 coverage: 선택 frame이 하나 이상 포함된 유효 구간 수 / 전체 유효 구간 수.
2. Reference 관측 가능 구간 coverage: 선택 frame이 포함된 구간 수 / 전체 Reference 후보가 하나 이상 포함된 유효 구간 수.

분모 0은 null로 기록한다. 영상별 비율의 단순 평균 대신 전체 covered/total 합계로 집계한다. GT 시간축과 raw 좌표의 대응은 기존 한계가 있으므로 이 지표는 공식 interval 좌표 기준의 선택 진단으로 표시한다. GT는 선택 완료 후 분석에만 사용한다.

## 9. 시간·메모리 측정

영상별 측정 시작은 첫 decoding/특징 수집 이전, 종료는 전체 점수 복원 직후다. GPU 측정 경계에서 동기화한다. 두 pass, 저비용 색 변환·resize·특징 계산, 표준화/PCA·SVD, 최적화, CLIP 전처리·전송·추론, VAD·복원을 모두 포함한다. pass 간 준비와 버퍼 관리도 총 wall time에서 누락하지 않는다.

단계별 키는 첫 pass decode, 저비용 특징, 특징 변환, 선택 최적화, 두 번째 pass decode, CLIP 전처리/전송/추론, VAD, 복원을 구분한다. 전체 시간은 독립 wall clock으로 측정하며 단계 합으로 대신하지 않는다. 기존 `processing_loop_wall_seconds`는 hash·저장 등도 포함하는 별도 지표임을 유지한다.

모델 초기화·artifact 로딩·입력 hash·warm-up·GT 분석·metric·파일 저장은 추론 시간에서 제외하고 범위를 명시한다. 현재 hash 확인 직후의 warm cache 성향을 모든 방법에서 유지·기록한다. decoder·스레드·장비·batch·precision·TF32·캐시 조건이 다른 결과를 같은 시간 비교로 합치지 않는다.

메모리는 모델과 CLIP visual 복사본을 포함한 실행 상태에서 측정한다. GPU peak allocated/reserved는 warm-up 후 측정 구간 직전 peak 통계를 초기화하고 baseline 사용량과 함께 기록한다. 프로세스 RSS는 측정 구간에서 관측한 peak와 sampling 간격을 기록한다. FFmpeg 자식 프로세스 포함 여부도 표시한다. 측정하지 못한 값은 0이 아닌 null과 사유로 남긴다.

`입력 비율=ΣK/ΣN`, `시간 감소율=1−T_method/T_reference`, `speedup=T_reference/T_method`다. 감소된 CLIP 입력 수만으로 speedup을 추정하지 않는다. 시간은 영상별 측정 시간 합으로 비교하며 wall time과 구분한다.

## 10. Reference 보호·설정·결과 저장

공용 서버에 보존된 기존 결과는 다음과 같다. 이번 문서 작성에서 재실행한 값이 아니다.

| 경로: `/workspace/vad/results/baselines/` 아래 | 주요 지표 |
|---|---:|
| `ucf_official/summary.json` | AUC 88.0199% |
| `ucf_raw_reference/summary.json` | AUC 87.1853% |
| `xd_official/summary.json` | AP 84.5045% |
| `xd_raw_reference/summary.json` | AP 84.5041% |

이 run들은 실행 당시 dirty code 기록이 있으므로 현재 clean HEAD와 완전 동일한 소스라고 간주하지 않는다. sampler 구현 후에는 패키지 hash가 달라지므로 같은 완성 코드·환경으로 full raw Reference를 다시 생성한다. 비교 검사를 우회해 과거 run을 속도 분모로 강제 사용하지 않는다.

현재 comparison signature의 데이터·모델·평가·공통 코드·환경 일치 검사는 보존한다. 별도로 sampler 특징 설정·artifact hash·λ·차원/허용오차·E 탐색 상한 등을 포함한 `sampler_signature`를 신설해 같은 방법의 반복 집계 시 검사한다. 서로 다른 방법을 raw Reference와 비교할 때 sampler_signature가 같아야 한다고 요구하지는 않는다.

새 experiment 설정은 기존 `configs/experiments/` 구조에 맞춰 신설한다. sampler 규칙은 실험 설정에, 데이터 경로는 local 설정에 둔다. 새 artifact 경로 키를 추가할 때는 `vadfs/config.py`의 local 허용 키와 검증도 함께 설계한다. artifact 내용 hash와 실험 규칙은 별도로 고정한다.

run에는 기존 score/feature NPZ와 summary를 유지하고 N/K, 원본 선택 index·명목 timestamp, artifact 식별값, sampler 설정/진단, 모든 단계 시간·메모리를 추가한다. 새 출력 schema는 버전을 표시하고 구버전 summary와 집계의 호환 여부를 검사한다. 실패 시 현재 영상·완료 영상·실패 사유를 저장하며 기존 결과를 덮어쓰지 않는다.

## 11. 반복과 집계

Random seed는 기존 protocol 제안인 `[0,1,2,3,4]`를 초기 실행안으로 사용하고 최종 평가 전에 고정한다. model seed는 모든 방법에서 234로 유지한다. 같은 seed를 반복하는 시간 측정은 별도 `timing_repeat_id`로 관리한다. 반복 횟수는 성능·시간 비교 전에 확정한다.

현재 집계기는 같은 seed 중복을 거부하므로 timing 반복용 집계가 필요하다. 먼저 seed별 시간 반복을 요약하고, Random 성능은 서로 다른 seed의 평균·표본 표준편차로 보고한다. 시간 반복을 독립 sampler seed처럼 세지 않는다. AP·Ano-AUC·coverage·입력 수·메모리·drop_pp도 집계한다. 다른 sampler_signature·Reference·protocol을 묵시적으로 섞지 않는다.

최종 표는 방법 / budget / 실제 CLIP 입력 수 / ROC-AUC / raw 대비 하락(pp) / AP / Ano-AUC / 두 coverage / 시간 / speedup / 메모리로 구성한다. 시간 반복과 seed 반복의 수·범위도 표시한다. `protocol.status=fixed`만으로 검증 완료를 주장하지 않는다.

## 12. 사용자 실행 검증과 완료 기준

Codex는 코드 작성·정적 검토만 수행한다. 아래는 향후 사용자 실행 항목이며 아직 실행한 검증이 아니다.

1. 준비 검증: 공식 목록·이름·SHA·feature shape/dtype·GT 순서 확인. artifact 적합 데이터와 test 분리 확인.
2. 합성 입력: D/A 개선량을 직접 logdet/inverse trace 차이와 대조. E 교환 목적값·동률·상한·종료를 확인. rank 0, 동일 특징, K=1, K=N, tail, 분해 실패와 비균일 좌표를 점검.
3. 100%: Reference와 각 방법의 index·feature·선택 및 복원 score를 같은 영상에서 비교. 특징 수집·최적화가 우회되는지 확인. 256 경계 전후와 긴 영상도 포함.
4. 25% smoke: 정상·이상·긴 영상을 포함해 정확한 K, 선택된 프레임만 encoding, 전체 길이 복원, 두 pass 비용과 artifact 기록 확인. 현재 smoke는 최종 metric을 계산하지 않으므로 score 배열을 비교한다.
5. full: 새로운 raw Reference를 만든 뒤 전체 290개 sampling 평가. 평가 프레임 수는 manifest/GT와 일치해야 한다. 기대 UCF 공식 평가 길이는 1,109,888이며 실행 준비 시 실제 GT로 확인한다.
6. 반복: 고정 seed 목록과 시간 반복을 실행·별도 집계하고 최종 표 작성. 모델·복원·전처리 조건은 sampler별로 바꾸지 않는다.

현재 존재하는 검증/실행 명령의 형태는 다음과 같다. `configs/local/hj.ucf.json`은 실제 경로를 담아 별도로 생성해야 하며 현재 존재한다고 가정하지 않는다.

```bash
cd /workspace/hyeongjun/vad
export VAD_PY=/workspace/sangmin/.venv/bin/python

"$VAD_PY" -m unittest discover -s tests -v
"$VAD_PY" scripts/setup/prepare_dataset.py \
  --dataset configs/datasets/ucf.json --local configs/local/hj.ucf.json --input both

CUDA_VISIBLE_DEVICES=0 "$VAD_PY" scripts/run_experiment.py \
  --config configs/experiments/ucf_reference_100.json \
  --local configs/local/hj.ucf.json --device cuda --mode smoke --save-features

CUDA_VISIBLE_DEVICES=0 "$VAD_PY" scripts/run_experiment.py \
  --config configs/experiments/ucf_reference_100.json \
  --local configs/local/hj.ucf.json --device cuda --mode full
```

Python 경로는 기존 baseline 환경으로 확인한 위치이며 이 환경을 수정하지 않는다. 실행 직전 GPU 가용 상태를 확인한다. D·A·E CLI는 구현 전이므로 실행 가능한 명령으로 제시하지 않는다. 구현 후 실제 설정 파일과 새 Reference 출력 경로를 사용해 smoke/full 명령을 추가한다. full sampling에는 현재 실행기가 요구하는 `--reference-run`을 지정한다.

## 13. 바로 다음 작업과 미해결 사항

- 적합용 train/development 영상의 실제 위치와 고정 목록 확보.
- 기존 baseline을 보존한 작업 범위에서 특징 수집·두 pass·sampler 인터페이스 구현.
- 100% 우회, 비용·메모리, coverage, 설정 fingerprint와 반복 집계 보완.
- 새 설정·테스트 작성 후 사용자가 실행하고 결과를 전달. 수치 동등성·재현성·실제 speedup은 그 결과로 판단.

이 문서는 구현 사양이며 D·A·E의 성능·속도 결과는 아직 없다.
