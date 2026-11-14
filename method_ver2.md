# 구간당 한 프레임 D/A/E 실험 방법 ver2

> 2026-11-14 UCF 인계: UCF ver2 full 36회가 사용자 실행으로 완료됐고 결과는
> [UCF ver2 성능표](docs/results/ucf_window_optimal_ver2_20261114/README.md)에 보존했다.
> 이번 GitHub 보존 범위는 공통 ver2 코드·테스트와 UCF 설정 20개다.
> 아래 XD 명령과 나머지 설정 20개는 로컬 후속 실험 안내이며, XD 전용 설정은 이번 커밋에 포함하지 않는다.
> 테스트 수행 여부는 확인하지 않았고, 이번 커밋 작업에서 테스트·추론을 재실행하지 않았다.


상태: 2026-11-13 별도 `_ver2` 코드·설정 작성, 정적 검사 완료. 테스트·추론·실험 실행은 미검증이다.
기존 cached CLIP 실험과 구별하며 기존 결과·설정의 의미를 바꾸지 않는다.

## 1. 목적과 범위

고정된 pretrained CLIP ViT-B/16과 데이터셋별 VadCLIP을 사용한다.
원본 영상의 각 구간에서 한 장을 선택하는 규칙만 바꿔 성능과 실제 비용을 비교한다.
UCF 290개와 XD 800개 전체를 대상으로 한다. CLIP/VadCLIP 가중치·prompt·branch·전처리·평가를 고정한다.
후보는 기존 16프레임 간격 CLIP cache가 아니라 각 구간 내부의 실제 원본 프레임 전체다.
선택 전에 CLIP을 돌리지 않고 저비용 영상 특징을 사용한다. 기존 CLIP 표현에 optimal을 적용하는
실험에서 저비용 표현의 구간 제약 optimal로 연구 범위가 바뀐다는 점을 명시한다.
학습·PCA 적합·모델 예측·GT·영상 category는 선택에 사용하지 않는다.

## 2. 구간과 정확한 입력 수

N은 기존 manifest의 reference_tokens, F는 실제 decode한 총 프레임 수다.
프레임 index는 zero-based, 구간은 half-open이다.

| Budget r | q=1/r | 구간 길이 L=16q | 구간당 CLIP 입력 |
|---|---:|---:|---:|
| 100% | 1 | 16 | 1장 |
| 50% | 2 | 32 | 1장 |
| 25% | 4 | 64 | 1장 |
| 12.5% | 8 | 128 | 1장 |

K=ceil(N/q)=ceil(rN), T=min(F,16N), anchor a_j=16qj, j=0,...,K-1.
후보 집합 C_j={a_j,...,min(a_j+L,T)-1}에서 정확히 한 장 s_j를 선택한다.
N=0, 빈 C_j, 실제 decode와 manifest 불일치는 오류로 중단한다. 임의 중복·K 감소로 보정하지 않는다.
전체 decode 중 T 이후 프레임은 확인만 하고 선택·특징 집계에 포함하지 않는다.
공식 평가 길이 자체는 줄이지 않는다. T는 GT 값에서 계산하지 않고 기존 Random-window와 같은
reference_tokens와 실제 프레임 수로 정한다. annotations는 sampler API에 넘기지 않는다.
현재 random_window.select_frames의 stop=min(frame_count,reference_tokens*16)과 일치한다.

100%는 원본 전 프레임이 아니라 16프레임당 한 장이다. 모든 방법의 K는 같아도 s_j는 달라질 수 있다.
따라서 optimal 100%와 Reference의 성능 동등성을 요구하지 않는다.

## 3. 공통 비교군

| 표시 이름 | 규칙 |
|---|---|
| Window-start / Uniform | 각 C_j의 첫 프레임 a_j 선택 |
| Random-window | 각 C_j에서 균등 무작위 한 장, 기존 video_seed와 Python random 규칙 유지 |
| Window-D | 구간당 한 장 제약으로 전체 영상의 logdet 목적함수 개선 |
| Window-A | 같은 제약으로 전체 영상의 inverse trace 목적함수 개선 |
| Window-E | 같은 제약으로 전체 영상의 최소 고유값 목적함수 개선 |

Random-window seed는 0–4로 고정하고 평균·표본 표준편차를 보고한다.
D/A/E의 초기 선택은 모두 window-start이며 결정적으로 동작한다.
각 구간을 독립적으로 1개 최적화하지 않는다. 영상 전체 선택 집합을 공동 최적화한다.
기존 cached Random-feature와는 별도 방법·별도 표다.
Official-feature baseline 및 raw Reference100은 참고 행으로 보존한다.

## 4. 선택용 저비용 표현 — 제안 기본값

이하 표현·탐색 상한은 아직 실행하지 않은 v1 제안값이다. 기존 test 결과에서 최적값을 찾았다고
주장하지 않는다. 변경은 개발 데이터나 GT 없는 계산 가능성 점검에 근거해 버전으로 기록한다.

1. 원본 영상을 순차 decode한다. 선택용 영상은 OpenCV BGR→RGB 후 cv2.INTER_AREA로
   종횡비 보정 없이 32×32에 resize하고 float64 RGB [0,1]로 변환한다.
   이 thumbnail 변환은 선택 전용이며 실제 CLIP용 데이터셋별 색 변환·crop을 대체하지 않는다.
2. Y=0.299R+0.587G+0.114B. 프레임별 h_t는 아래 8차원으로 고정한다.
   [mean(R),mean(G),mean(B),2std(R),2std(G),2std(B),edge(Y),motion(Y)].
   std는 1024 pixel 모집단 표준편차(ddof=0)다.
   edge는 수평·수직 인접 pixel 절대 차이 평균의 산술평균이다.
   motion은 mean(abs(Y_t-Y_(t-1)))이며 첫 프레임은 0이다. 구간 경계에서도 이전 프레임을 사용한다.
3. h_t를 L2 정규화한다. h=0은 영벡터로 유지하고 개수를 기록한다.
4. 데이터와 무관한 PCG64(234) 표준정규 8×8 행렬의 reduced QR로 Q를 만든다.
   R 대각이 양수가 되도록 Q 열 부호를 고정하고 실제 Q와 hash를 저장한다.
   d=min(8,K), z_t=normalized(h_t) Q[:,:d] sqrt(8/d).
   모든 데이터셋·방법·영상은 같은 Q를 쓴다. 투영 후 재정규화하지 않는다.
5. PCA/SVD·영상별 표준화·rank에 따른 차원 적합은 하지 않는다. rank는 진단으로만 기록한다.

thumbnail 특징은 의미론적 CLIP 표현이 아니다. 저조도·카메라 움직임·작은 사건에 취약할 수 있다.
특징 다양성의 증가가 이상 탐지 성능 증가를 보장하지 않는다.
NaN/Inf는 오류이며 전체 z가 0이면 window-start fallback을 명시한다.

## 5. 공통 구간 제약 교환 알고리즘

M(S)=I_d+Σ_j z_(s_j)z_(s_j)^T, ridge=1, float64.
D: logdet(M) 최대화. A: tr(M^-1) 최소화. E: λ_min(M) 최대화.
모두 window-start 집합에서 시작하며 동일한 coordinate sweep을 적용한다.

- 1 sweep은 시간순으로 모든 K개 구간을 한 번 방문한다.
- 현재 구간 j에서 선택 s_j를 C_j의 모든 다른 프레임으로 바꾼 trial을 평가한다.
  M'=M-z_(s_j)z_(s_j)^T+z_tz_t^T. 다른 구간 선택은 그대로다.
- D/A는 해당 목적을 가장 개선하는 후보가 현재 값보다 허용오차 이상 좋을 때만 교체한다.
- E는 최소 고유값 개선을 우선한다. 현재 최소값과 허용오차 내 동률인 후보 중에서는
  logdet이 개선될 때 보조 교체를 허용한다. rank 부족으로 λ_min=1에 머무는 상태도 기록한다.
- 후보 간 허용오차 내 동률은 원본 frame index가 작은 후보로 결정한다.
  목적 개선 없는 단순 동률 이동은 하지 않는다.
- 교체가 즉시 다음 구간의 M에 반영된다. sweep 종료 시 선택 집합에서 M을 재구성한다.
- 최대 5 sweeps. 한 sweep에서 교체가 0이면 조기 종료한다.
  고정 10개 교환 상한을 쓰지 않으며 최대 5K번 교체할 수 있다.
- D/A objective tolerance=1e-12 max(1,abs(current objective)).
  E tolerance=1e-10 max(1,λ_max(M)), 보조 logdet tolerance는 D와 같다.
- 초기·최종 모든 목적값, selected rank, 교체·보조 교체 수, sweep별 변경 수와
  후보 평가 수, 종료 사유를 저장한다. 선택 결과를 목적함수로 재계산해 유한성·단조성을 검사한다.

이는 구간 제약하의 좌표 교환 근사다. 전역 최적해나 AUC/AP 최적화를 주장하지 않는다.
최악 후보 평가 수는 약 5Σ_j(|C_j|-1)이며 각 trial에 d≤8 선형대수가 필요하다.
실제 계산 가능성을 확인하기 전 실행 시간을 예측하거나 효율적이라고 단정하지 않는다.

## 6. 실제 feature 추출과 시간축

Pass 1: 전체 decode → thumbnail 특징/좌표 → 구간 제약 선택.
Pass 2: 원본 재decode → 선택한 K장만 고정 CLIP 전처리·encoding → 기존 VadCLIP.
두 pass의 프레임 수·nominal FPS·좌표를 검사한다. 프레임 단위 random seek는 기본으로 쓰지 않는다.
저비용 z는 모델에 전달하지 않는다. 원본 영상에서 새로 추출한 512D feature와 기존 저장 dtype을 사용한다.
UCF/XD 각각 configs/datasets의 색 변환·geometry·crop·encoder precision/batch를 고정한다.

선택 s_j와 timestamp를 저장하되 score anchor는 모든 비교군에서 a_j로 고정한다.
기존 positional encoding·compressed token 순서·window·padding/mask를 유지한다.
선택 점수를 a_j에 놓고 전체 reference grid(16k)로 nearest 복원한다.
동률은 앞선 anchor, 양 끝은 최근접 관측으로 처리한 뒤 기존 16-frame 확장으로 전체 GT 길이를 평가한다.
실제 선택 좌표로 복원하는 ablation은 주 실험과 섞지 않는다.

## 7. 평가와 비용

UCF: Branch1 frame ROC-AUC. XD: Branch2 frame AP.
기존 보조 branch·Ano-AUC·fine-grained mAP도 보존한다.
영상별 K·N·F·T와 실제 encoder 입력 수를 기록한다. anomaly coverage는 선택 후 GT로만 분석한다.
주 표: dataset/method/budget/주 지표/Random std/Uniform Δ/Random 평균 Δ/Reference drop.
별도 비용 표: decode pass1, thumbnail, selection, decode pass2, CLIP 전처리·encoding,
VadCLIP·복원, 총 E2E, GPU peak, CPU 메모리와 측정 범위.

성능용 feature cache 실행과 raw E2E를 분리한다. optimal E2E에는 두 pass와 모든 선택 비용을 포함한다.
Uniform/Random은 기존 한 pass 구현을 그대로 측정한다. 불필요한 pass를 강제로 추가하지 않는다.
따라서 비용 차이는 프레임 수와 별개다. CLIP 입력 수가 같으면 optimal은 추가 비용이 들 수 있다.
E2E 구간은 첫 decode 시작부터 최종 score 복원까지의 동기화된 wall time이며 내부 compute/I/O를
빠짐없이 포함한다. setup·모델 로딩·warmup·GT 분석·metric·결과 저장 시간은 별도로 기록한다.
같은 장비·precision·batch·I/O 조건에서 GPU별 동시 작업 없이 실행하고 캐시 상태를 기록한다.
효율 보고 전 모든 방법·budget을 3회 측정한다. 결정적 방법은 동일 선택을 재계산하며 Random은 seed0을
3회 측정한다. Random seed0–4 성능 반복과 시간 반복을 혼동하지 않는다.
같은 budget Uniform 대비 speedup 및 raw Reference100 대비 speedup은 실제 총 시간 비율로 계산한다.

## 8. 실험 규모와 검증 순서

데이터셋별 Uniform4 + Random-window20 + D/A/E12 = 36회, 양쪽 72회 성능 실행.
raw Reference100을 재검증할 경우 데이터셋별 1회 추가한다. 시간 반복은 별도 run으로 기록한다.

1. 구현 후 합성 입력으로 구간 제약·K·tail·동률·K=1·rank 부족·0 특징·비유한값을 검사한다.
2. 작은 입력에서 모든 구간별 후보 trial의 직접 목적값과 선택 결정을 대조한다.
   초기/최종 목적값 단조성과 각 구간 정확히 1장 선택을 검사한다.
3. window-start는 기존 Uniform과 index/feature/score가 동등해야 한다.
   Random-window는 동일 seed에서 기존 선택 index와 score anchor가 같아야 한다.
4. 짧은 영상으로 두 pass 일치, 실제 encoder K개, 원본 512D 입력·dtype 및 전체 평가 길이를 확인한다.
   tail floor/ceil, 마지막 짧은 구간과 공식 coverage 밖 프레임을 포함해 검증한다.
5. GT·category·예측 점수가 sampler에 전달되지 않는지 인터페이스를 검사한다.
6. GT 없는 계산 가능성 점검 후 사양을 고정한다. 필요한 튜닝은 별도 개발 데이터에서 한다.
7. 사용자 직접 full 실행 후 성능과 시간 결과를 분리해서 해석한다.

기존 test 결과를 본 뒤 설계를 변경했으므로 새 UCF/XD 결과도 후속 탐색적 평가임을 표시한다.
같은 test에서 여러 특징·차원·sweep 수를 비교해 최고 설정을 최종 성능으로 보고하지 않는다.
최종 일반화 주장을 위한 미사용 평가 조건 또는 별도 검증 절차는 이후 확보한다.

## 9. 구현 인계와 산출물

예정 method ID: window_d_optimal/window_a_optimal/window_e_optimal.
예정 구성: 별도 window selector와 thumbnail 표현 모듈, 두 pass raw pipeline 연결,
configs/experiments의 별도 window 설정, 선택·시간·signature 검증 및 사용자 테스트.
기존 clip_fixed_projection, cached optimal, raw PCA 설정을 새 의미로 덮어쓰지 않는다.
출력은 outputs/runs/ 고유 폴더에 선택 index·anchor·timestamp·원본 feature(선택 저장),
목적값/교환 진단·recipe/checkpoint/소스/hash·환경·전체 metric·시간을 기록한다.
집계는 outputs/analyses/에 candidate_space=raw_window, representation, anchor, protocol version이
같은 run만 허용한다. source cache 성능표와 자동 합치지 않는다.

별도 `_ver2` CLI·설정·검증 코드를 작성했다. 실제 입력과 명령·출력·미검증 항목은 다음 절을 따른다.

## 10. ver2 구현 파일과 실행 안내

2026-11-13 구현 작성. 정적 검토만 수행했으며 프로젝트 import·테스트·추론은 미실행이다.
새 파일명은 모두 `_ver2`로 구분한다. 기존 pipeline/selector/CLI는 수정하지 않았다.
공통 모델·추출·평가·manifest 로더는 기존 모듈을 그대로 import한다.

| 새 파일 | 역할 |
|---|---|
| `vadfs/window_sampling_ver2.py` | 원본 후보 thumbnail 특징·고정 투영·구간 제약 교환 |
| `vadfs/pipeline_ver2.py` | raw 두 pass 선택, 선택된 프레임 CLIP 추출, 기존 평가 |
| `scripts/run_experiment_ver2.py` | 단일 smoke/full 실행 |
| `scripts/run_optimal_ver2.py` | 데이터셋별 36회 sweep |
| `scripts/analyze/compare_sampling_ver2.py` | ver2 전용 성능 집계·CSV |
| `tests/test_window_sampling_ver2.py` | 구간·tail·목적함수·Random 호환·추출 오류 검증 |
| `tests/test_comparison_ver2.py` | 집계 signature·seed·K 혼합 거부 검증 |
| `configs/experiments/*_ver2.json` | 두 데이터셋 × 5방법 × 4budget, 40개 |

### 입력 준비

기존 `configs/local/hj.ucf.json`, `hj.xd.json`의 원본 영상·checkpoint 경로를 사용한다.
반드시 raw_path가 준비된 full manifest를 명시한다. 이 서버의 기존 manifest 후보는
`/workspace/vad/results/baselines/{ucf_raw_reference,xd_raw_reference}/manifest.json`이다.
새 pipeline은 그 manifest를 검증하지만 과거 결과의 metric이나 cache feature를 입력으로 쓰지 않는다.
manifest가 없거나 검증에 실패하면 기존 준비 절차로 `.local/setup/`에 준비하고 그 실제 경로를 넘긴다.
데이터셋별 모델과 CLIP checkpoint, 원본 test 영상, 기존 evaluator의 annotation 입력이 필요하다.
별도 개발 영상/PCA artifact는 이 고정 사양 실행에 필요하지 않는다.

### 사용자가 실행할 검증

```bash
/workspace/sangmin/.venv/bin/python -m unittest discover -s tests -p '*ver2.py' -v
```

먼저 Uniform/Random의 기존 경로 대비 선택·feature·score 일치와 100/25% D/A/E smoke를 확인한다.
기본 smoke는 기존 helper와 동일하게 짧은 normal/anomaly와 긴 영상 사례를 고르며,
이때 label은 실행 점검 사례 선택에만 쓰고 sampler에는 전달하지 않는다.
긴 영상 smoke도 계산량이 클 수 있으므로 `--video`로 정확한 manifest 파일명을 지정할 수 있다.

```bash
CUDA_VISIBLE_DEVICES=0 /workspace/sangmin/.venv/bin/python -u scripts/run_experiment_ver2.py \
  --config configs/experiments/ucf_d_optimal_25_ver2.json \
  --local configs/local/hj.ucf.json \
  --setup-manifest /workspace/vad/results/baselines/ucf_raw_reference/manifest.json \
  --mode smoke --device cuda --save-features
```

A/E는 config의 `d_optimal`을 `a_optimal`/`e_optimal`로 바꾼다. budget 100도 같은 방식으로 점검한다.
XD는 `ucf`를 `xd`로 바꾸고 XD manifest와 local 설정을 사용한다.
검증 후 full sweep 명령은 다음과 같다. 성능 탐색은 GPU0/1 분리가 가능하지만 최종 시간 측정은
다른 GPU·CPU·I/O 작업이 없는 상태에서 한 번에 하나씩 수행한다.

```bash
CUDA_VISIBLE_DEVICES=0 /workspace/sangmin/.venv/bin/python -u scripts/run_optimal_ver2.py \
  --dataset ucf --local configs/local/hj.ucf.json \
  --setup-manifest /workspace/vad/results/baselines/ucf_raw_reference/manifest.json \
  --mode full --device cuda

CUDA_VISIBLE_DEVICES=1 /workspace/sangmin/.venv/bin/python -u scripts/run_optimal_ver2.py \
  --dataset xd --local configs/local/hj.xd.json \
  --setup-manifest /workspace/vad/results/baselines/xd_raw_reference/manifest.json \
  --mode full --device cuda
```

`scripts/run_optimal_ver2.py`의 `--methods`와 `--budgets`로 범위를 제한할 수 있다.
Uniform 및 Random-window가 포함된 full sweep은 완료 후 자동 비교 CSV를 생성한다.
일부 방법만 실행하면 독립 실행 결과가 저장되며, 비교 시 같은 budget의 Uniform 및 Random 5 seed가 필요하다.
시간 3회 반복은 별도 사용자 실행으로 수행하며, 현재 도구는 시간 반복을 자동 집계하지 않는다.

### 출력과 사용자가 확인할 항목

- 각 실행: `outputs/runs/*_ver2_{smoke,full}_*/summary.json`, `scores/*.npz`.
- `--save-features` 지정 시 선택된 새 CLIP feature와 좌표가 `features/*.npz`에 저장된다.
- `projection_ver2.npz`, manifest 사본, 코드·입력 hash와 환경도 실행 폴더에 저장된다.
- Sweep: `outputs/analyses/*_sweep_ver2_*/sweep_ver2.json`.
- 비교: `outputs/analyses/*_comparison_ver2_*/comparison_ver2.csv`, `summary_ver2.json`.
- `success`, 모든 영상 수, 정확한 K/구간당 한 장, 실제 encoder 입력 K개, 두 pass FPS/프레임 수,
  score anchor와 전체 평가 길이, 목적값/교환 횟수/Uniform 중복률을 확인한다.
- 최종 시간 효율은 3회 별도 측정 전까지 미검증이다. 비교 CSV의 시간은 seed별 단일 실행 평균이며
  반복 측정 speedup으로 해석하지 않는다. CPU/FFmpeg 메모리는 현재 null이다.

실제 실행 검증 완료를 뜻하지 않는다. 최초 설계 문서 `docs/WINDOW_OPTIMAL_PROTOCOL.md`는 설계 당시 기록으로 보존하며 이 파일이 ver2 구현 인계의 최신 사양이다.
