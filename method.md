# 기존 CLIP feature에 적용하는 D·A·E-optimal sampling

수정일: 2026-11-12. 대상: UCF-Crime 및 XD-Violence.
상태: **적합 없는 feature 기반 실행 코드 작성·정적 검사 완료. 테스트·추론·실행 검증은 미실행이다.**
이전 저비용 특징·개발 데이터 PCA·두 pass 사양은 `docs/archive/METHOD_CHEAP_FEATURE_PCA_20261112.md`에 보존한다.

## 1. 목적과 변경 이유

기존 raw Reference에서 추출해 저장한 CLIP feature에 D-, A-, E-optimal 선택을 적용하고,
동일 budget의 Uniform·Random과 탐지 성능을 비교한다. CLIP과 VadCLIP의 가중치는 고정한다.
개발 영상 확보, 표준화 통계 적합, PCA 적합, sampler 학습은 이번 실험에 필요하지 않다.

현재 공유 영상은 UCF test 290개, XD test 800개다. 이전 사양은 별도 개발 데이터로 PCA를
적합해야 했으나 사용자 요청에 따라 먼저 기존 CLIP 표현의 선택 효과를 탐색한다.
이는 기존 raw-video 연구의 최종 효율 실험에 앞선 **탐색적 feature 기반 성능 실험**이다.
전체 CLIP feature를 사용해 선택하므로 CLIP 계산 절감이나 raw-video speedup의 근거로 쓰지 않는다.

흐름은 다음과 같다.

```text
고정된 raw Reference CLIP feature [N,512]
  → 선택 전용 정규화·고정 8D 투영
  → D / A / E로 정확히 K개 행 선택
  → 선택된 원본 512D feature를 시간순으로 기존 VadCLIP에 입력
  → 전체 Reference grid로 nearest 복원
  → 전체 공식 GT 시간축 평가
```

## 2. 입력과 baseline

기본 입력은 다음 raw Reference run의 영상별 `features/*.npz`다.

| Dataset | Reference 폴더 | 주 지표 |
|---|---|---|
| UCF-Crime | `/workspace/vad/results/baselines/ucf_raw_reference` | Branch1 frame ROC-AUC |
| XD-Violence | `/workspace/vad/results/baselines/xd_raw_reference` | Branch2 frame AP |

기존 README·실행 코드에 feature cache 사용 경로가 있다. 새 실행 전 해당 폴더의 실제
feature 파일 존재·읽기 권한·내용·출처를 확인한다. 이번 문서 수정에서 NPZ를 읽거나
feature 전체를 검증하지 않았으므로 준비 완료로 간주하지 않는다.

각 cache는 원본 feature `X ∈ R^(N×512)`, `selected_frame_indices`,
`selected_timestamps_seconds`를 포함해야 한다. Reference summary, manifest, feature 파일 hash로
출처를 고정한다. 영상별 행 수 N·dtype·유한값과 저장 좌표가 전체 Reference grid와 맞아야 한다.
누락·중복·다른 checkpoint·다른 전처리·sampled cache는 거부한다.

선택에는 feature와 후보 좌표, K 및 고정 수치 설정만 전달한다. GT·category·anomaly 여부·
VadCLIP 예측 점수는 선택 입력으로 쓰지 않는다. 공식 test 영상 순서와 평가 GT는 기존 규칙을 유지한다.

Raw Reference cache가 없으면 동일하게 고정한 추출 설정으로 전체 feature를 먼저 생성한다.
이 경우에도 전체 CLIP 계산을 수행한 것으로 기록한다. Official feature 실험은 입력 출처가 다른
별도 실험군이며 raw Reference cache를 조용히 대체하지 않는다.

## 3. 정확한 budget과 공통 비교군

영상별 Reference 후보 수 N에서 다음을 적용한다.

```text
b ∈ {1, 0.5, 0.25, 0.125}
K = min(N, max(1, ceil(b × N)))
```

N=0은 오류다. 모든 선택은 정확히 K개이며 중복 없이 시간순으로 정렬한다.
실제 K=N이면 정규화·투영·최적화를 모두 우회하고 원본 feature 전부를 기존 추론에 전달한다.
짧은 영상에서 b<1이어도 동일하게 우회한다. 100%에서는 feature·score·metric이 cached Reference와
일치해야 한다. CLIP 입력 자체가 100% 줄거나 재계산되는 것은 아니다.

새로운 공통 feature 비교군은 아래와 같다.

| 방법 | 선택 공간 및 규칙 |
|---|---|
| Reference | 원본 cache의 모든 행 |
| Uniform | 현재 baseline과 동일하게 후보 0부터 1/b 간격으로 K개 행 선택 |
| Random-feature | 동일한 N개 후보 중 K개를 비복원 추출; seed 0~4 |
| D / A / E | 같은 N개 후보 중 아래 공통 표현·목적함수로 K개 행 선택 |

Random-feature는 기존 `random` selector 및 영상별 `video_seed` 규칙을 사용한다.
기존에 완료한 **Random-window**는 16/b프레임 구간 안에서 원본 프레임을 고르므로 이 비교군과 다르다.
기존 Random-window 결과는 별도 행으로 추가할 수 있으나 후보 공간과 score anchor 차이를 표시한다.
새 공통 후보 Random 결과인 것처럼 이름을 바꾸거나 동일 후보 비교라고 표현하지 않는다.

기존 Uniform 결과도 같은 raw cache·행 선택·모델·nearest 복원·평가 조건을 확인한 뒤 재사용한다.

## 4. 적합 없는 공통 선택 표현

D/A/E는 아래와 같은 고정 표현을 사용한다. 세 방법에서 변환을 다르게 하지 않는다.

1. Cache의 각 행을 float64로 복사해 `u_i = x_i / ||x_i||₂`로 만든다. 원본 X는 변경하지 않는다.
2. `numpy.random.Generator(numpy.random.PCG64(234))`로 표준정규 행렬 G `[512,8]`을 생성한다.
3. `Q,R = numpy.linalg.qr(G, mode='reduced')`를 계산한다. R 대각 성분이 음수인 열은
   해당 Q 열의 부호를 반전해 QR 부호 규칙을 고정한다. 대각 0은 오류다.
4. 고정 투영 `P = sqrt(512/8) × Q`를 만들고 `z_i = u_i P`로 8차원 선택 표현을 얻는다.
5. K<8이면 P의 앞 `d=K`개 열만 쓰며 배율은 `sqrt(512/d)`로 다시 정의한다.
   따라서 일반적으로 `d=min(8,K)`, `P_d=sqrt(512/d)×Q[:,:d]`다.

Q는 UCF·XD, 모든 영상·방법·budget에서 같은 고정 행렬의 앞 열을 사용한다. 선택용 row normalization은
CLIP 표현의 크기보다 방향을 비교하도록 정한 규칙이며, 투영 후 추가 정규화는 하지 않는다.
서로 다른 budget의 d 변화는 기록하고, 동일 영상·budget의 모든 optimal 방법은 같은 z를 사용한다.

이 과정은 데이터 평균·분산·주성분을 추정하지 않는 **고정 무작위 직교 투영**이다.
PCA, whitening, 영상별 SVD, dataset 표준화 적합을 수행하지 않는다.
초기 seed 234·8차원·배율을 test metric을 보고 고르거나 바꾸지 않는다.
Q의 실제 값·shape·생성 규칙·NumPy 버전·float64 바이트의 SHA-256을 실행 결과에 보존한다.
동일 seed만으로 모든 플랫폼에서 bit 동일하다고 가정하지 않고 행렬 hash를 대조한다.
여기서 저장하는 행렬은 재현용 상수이며 학습된 PCA artifact가 아니다.

NaN/Inf 또는 원본 feature의 0 norm은 오류로 처리한다. 투영된 z 전체가 0인 경우만
Uniform fallback과 사유를 기록한다. 그 외 rank 부족은 자동 차원 적합으로 보정하지 않는다.
E의 퇴화와 보조 교환 규칙으로 드러내고 진단에 남긴다.

### 왜 고정 8D 투영을 쓰는가

512D에서 K<512이면 `λI + Σ z_i z_iᵀ`의 최소 고유값이 항상 λ에 머물러
일반적인 E 목적함수로 선택 집합을 구별할 수 없다. D/A의 반복 선형대수 비용도 커진다.
고정 8D 투영과 `d≤K` 규칙으로 이 문제를 줄인다. 이것이 모든 rank 부족을 해소한다는 뜻은 아니다.

이번 D/A/E는 **투영한 CLIP 표현에서의 optimal 기준을 근사적으로 최적화하는 방법**이다.
원본 512D에서의 최적해나 anomaly-relevant frame의 최적해라고 주장하지 않는다.
투영 없는 512D D/A와 성능이 같다고 가정하지도 않는다.

## 5. 공통 설계 행렬과 수치 규칙

```text
M(S) = λ I_d + Σ(i∈S) z_i z_iᵀ
λ = 1
```

이는 CLIP 기반 선택용 설계 행렬이며 VadCLIP의 Fisher 행렬이나 이상 점수가 아니다.
모든 행렬 연산은 float64다. 초기 설정은 실행 전 고정하고, GT·test metric 기반 튜닝은 하지 않는다.
최적화 실패·분해 실패·비유한 목적값은 오류로 저장하며 묵시적으로 성공 처리하지 않는다.

D/A gain 동률 허용오차는 `1e-12 × max(1, |최대 gain|)`이며 동률에서는 앞선 후보를 선택한다.
K를 채울 때까지 진행하며 선택된 원본 행 위치를 마지막에 시간순으로 정렬한다.

## 6. D-optimal: log determinant 최대화

빈 선택에서 `M=λI`로 시작한다. 미선택 후보를 배치로 풀어 `Mu_i=z_i`를 계산하고
다음 개선량이 가장 큰 후보를 추가한다.

```text
Δ_D(i) = log1p(z_iᵀ u_i)
```

후보를 추가한 뒤 M을 갱신하고 K개까지 반복한다. 단계마다 Cholesky 분해를 사용하며
후보별 역행렬은 만들지 않는다. 초기/최종 logdet과 선택 수를 저장한다.
이 greedy 선택은 전역 최적해 보장을 의미하지 않는다.

## 7. A-optimal: inverse trace 최소화

D와 같은 초기화와 선형계 풀이를 사용하고 다음 inverse trace 감소량을 최대화한다.

```text
Δ_A(i) = ||u_i||₂² / (1 + z_iᵀ u_i)
```

K개까지 추가한다. `tr(M⁻¹)`의 초기/최종 값과 선택 수를 저장한다.
D/A가 동률 등으로 같은 집합을 반환할 수 있으며 서로 다른 결과를 강제로 만들지 않는다.

## 8. E-optimal: 최소 고유값 최대화

동일 budget의 Uniform K개에서 시작한다. 선택된 j와 미선택 i를 교환해
`M' = M - z_j z_jᵀ + z_i z_iᵀ`를 만들고 실제 최소 고유값을 평가한다.

- 최대 10라운드, 라운드당 최대 4096쌍으로 고정한다.
- 전체 교환 쌍이 4096 이하면 전수 평가한다.
- 초과하면 현재 최소 고유벡터 v의 투영 제곱 `(vᵀz)²`이 작은 선택 후보 최대 64개,
  큰 미선택 후보 최대 64개의 조합을 평가한다. 후보 축소 동률은 시간순이다.
- 최소 고유값 개선 허용오차는 `τ_E=1e-10×max(1,||M||₂)`다.
- `λ_min(M')>λ_min(M)+τ_E`인 교환을 우선한다. 그 중 최소 고유값이 가장 큰 쌍을 선택한다.
- 1차 개선이 없을 때만, 최소 고유값이 현재 값과 τ_E 이내이고 logdet이
  `1e-12×max(1,|logdet(M)|)`보다 크게 개선되는 교환을 보조적으로 허용한다.
- 비교 중 최소 고유값이 허용오차 내 동률이면 logdet이 큰 쌍을 우선하고,
  최종 동률은 `(제거 후보 index, 추가 후보 index)` 사전순으로 정한다.
- 개선이 없거나 라운드 상한에 도달하면 종료한다. 교환 후 M을 선택 집합에서 재구성한다.

초기/최종 최소 고유값·logdet·선택 행렬 rank, 평가 쌍 수·교환 수·보조 교환 수·종료 사유를 저장한다.
rank 부족으로 최소 고유값이 λ에 머무르면 그 사실과 보조 교환 여부를 표시한다.
최소 고유값 대신 최소 양의 고유값을 쓰는 등 목적함수를 조용히 변경하지 않는다.
E는 후보 축소와 상한이 있는 교환 근사이며 전역 최적해 보장을 주장하지 않는다.

## 9. 기존 VadCLIP 추론과 전체 시간축 평가

선택용 정규화·투영 결과 z는 sampler에만 사용한다. **VadCLIP에는 cache에서 선택한 원본
512D feature 행을 기존 dtype 그대로 전달한다.** CLIP 재인코딩과 모델 재학습은 수행하지 않는다.

선택 feature는 시간순으로 압축해 기존 window·positional embedding·padding/mask 처리에 넣는다.
원본 timestamp를 새로 positional encoding에 주입하지 않는다.
Branch1/Branch2 및 클래스 점수 모두 선택된 원본 후보 위치에서 전체 Reference grid로 nearest 복원한다.
동률은 앞선 관측, 양 끝은 가장 가까운 관측을 사용하고 기존 16-frame 확장을 유지한다.

UCF는 normal 포함 290개 전체에서 Branch1 ROC-AUC를 주 지표로, XD는 800개 전체에서
Branch2 AP를 주 지표로 쓴다. 두 branch의 기존 보조 지표·Ano-AUC 및 fine-grained 평가도 유지한다.
선택 프레임만 평가하거나 영상/GT를 줄이지 않는다. anomaly 영상 내부 정상 구간을 포함하는
기존 Ano-AUC 정의와 공식 순서·집계를 그대로 사용한다.

## 10. 결과 비교와 기록

기본 비교표는 dataset / method / budget / 실제 K 합계 / ROC-AUC / AP / Ano-AUC /
fine-grained mAP / raw Reference 대비 drop / Uniform 대비 Δ로 구성한다.

```text
delta_pp = 100 × (sampled metric − raw Reference metric)
drop_pp  = 100 × (raw Reference metric − sampled metric)
uniform_delta_pp = 100 × (method metric − 동일 budget Uniform metric)
```

Random-feature는 seed `[0,1,2,3,4]`의 평균·표본 표준편차와 개별 값을 기록한다.
D/A/E와 고정 투영은 결정적이며 model seed는 234로 고정한다.
Random-window의 기존 seed 결과는 별도 후보 공간으로 표시한다.

GT coverage는 선택 완료 후에만 분석한다. half-open interval을 전체 평가 길이로 clip하고,
전체 유효 구간 및 Reference 관측 가능 구간 두 분모의 covered/total 합계를 기록한다.
분모 0은 null이며 영상별 비율을 단순 평균하지 않는다.

summary/선택 NPZ에는 다음을 저장한다.

- source cache run/summary/manifest/영상별 feature hash, 모델·checkpoint·전처리·평가 식별값.
- sampler 설정, projection 생성 규칙·실제 Q/hash, λ/d/허용오차/E 상한, 소스·환경 버전.
- 영상별 N/K, 선택 row·원본 frame index·timestamp, fallback·목적값·교환 진단.
- 원본 512D feature가 변환 없이 모델에 전달됐는지와 전체 복원/평가 길이.
- `experiment_scope=cached_feature_exploration`, `speedup_eligible=false` 및 Random 후보 공간.

Cache 로딩, 선택 표현 생성, 최적화, VAD, score 복원 시간은 각각 기록할 수 있다.
이는 cached-feature 시간이다. 모델 초기화·warmup·metric·저장은 별도 범위로 기록한다.
전체 feature 생성 비용을 생략한 시간을 raw end-to-end라고 부르거나 CLIP speedup으로 환산하지 않는다.
입력 token 감소는 `K/N`으로 표시하되 실제 수행한 CLIP 입력 수 감소와 구분한다.

## 11. 구현 연결과 실행 상태

`vadfs/clip_selection.py`에 고정 투영·원본 feature 행 선택을 추가하고 공통 cached pipeline에
연결했다. 새 설정은 `configs/experiments/{ucf,xd}_clip_*.json` 42개이며
feature_source=raw_reference_cache와 명시적 clip_fixed_projection 설정을 사용한다.
기존 raw PCA 설정을 같은 의미로 재해석하지 않고 별도 경로로 보존했다.
원본 인수인계 `VAD_Optimal_Sampling_Handoff.md`는 과거 근거로 남긴다.

구현된 공통 경로는 다음과 같다.

1. 영상별 전체 cache를 읽고 행/좌표·출처를 검증한 뒤 선택 전용 고정 투영을 생성한다.
2. `optimal.py`의 선택 연산에 공통 z를 전달하고 선택 positions로 원본 X를 slice한다.
3. cached optimal에 대한 현재 거부 조건을 새 mode에 맞춰 변경하고 sampler signature를 분리한다.
4. 별도 명시적인 `feature_source`/표현 설정을 추가해 이전 raw PCA 설정을 다른 의미로 재해석하지 않는다.
5. Random-feature를 같은 cache 경로에 연결하고 기존 Random-window는 별도 유지한다.
6. wrapper의 PCA artifact 필수 조건을 이 새 경로에서 제거하고 UCF/XD별 설정·비교 도구를 연결한다.
7. cache의 모델·GT·복원·recipe 검사를 유지하고 서로 다른 표현·투영 hash·N/K·평가 조건의 집계를 거부한다.
   과거 Reference의 평가 소스만 현재와 다르면 feature provenance로만 재사용한다.
   해당 과거 metric 대비 차이는 생성하지 않으며 현재 코드의 cached Reference 100%를 다시 평가해 비교한다.

`scripts/run_optimal.py`는 PCA artifact 없이 같은 cache의 Reference/Uniform/Random-feature/D/A/E를
기본 37회 실행하며 full 완료 후 비교 CSV를 생성한다. 실제 사용자 명령은 `docs/OPTIMAL_RUNS.md`에 있다.
입력 cache 290개/800개와 summary·manifest의 존재·읽기 권한은 확인했지만 NPZ 내용은 읽어 검증하지 않았다.
새 출력은 `outputs/runs/`와 `outputs/analyses/` 아래 고유 폴더에 저장한다.
기존 공용 원본·checkpoint·cache·과거 결과는 덮어쓰지 않는다.

## 12. 사용자 실행 검증과 다음 단계

코드 수정 후 사용자 실행으로 다음을 확인한다. 현재 이 검증은 미실행이다.

1. 고정 투영이 데이터와 무관하고 반복·방법·데이터셋 간 같은 Q/hash를 사용하는지 확인한다.
2. 작은 합성 입력에서 D/A gain을 직접 목적값 차이와, E 한 라운드를 전수교환 결과와 대조한다.
3. K=1, K=N, rank 부족, feature 0 norm/NaN, 동률, 정확한 K와 시간순 정렬을 확인한다.
4. 100%에서 변환 우회·원본 feature/score/전체 metric의 cached Reference 동등성을 확인한다.
5. UCF/XD 25% smoke에서 원본 512D 행 전달·전체 시간축 복원과 출처 기록을 확인한다.
6. 100/50/25/12.5% full과 Random-feature seed 0~4를 실행하고 같은 budget 성능표를 생성한다.

이 탐색 결과를 이용해 방법을 수정하면 test 기반 탐색 이력을 남긴다. 최종 일반화 성능을
주장할 때에는 설계에 사용하지 않은 평가 조건이나 별도 검증 데이터가 필요하다.
실제 CLIP 이전 선택과 raw end-to-end 효율 검증은 후속 연구 단계로 유지한다.
