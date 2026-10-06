# VAD Frame Sampling 연구 계획

작성일: 2026-10-05  
계획 수정: 2026-10-07 — 사용자 요청으로 UCF fine-grained 기본 평가를 원본 공개 코드의 290개/14개 클래스 방식으로 변경; Ano-AUC 유지
대화 출처: 「베이스라인 VAD 모델 추천」  
문서 상태: 현재까지 합의한 연구 방향과 이를 실행하기 위한 실험 계획. 실험 결과는 아직 포함하지 않는다.

## 1. 프로젝트 개요

**Pretrained VadCLIP을 고정하고, UCF-Crime과 XD-Violence의 raw video에서 frame sampling strategy만 변경하여 VAD 성능과 계산 효율을 비교한다.**

연구 주제는 Video Anomaly Detection(VAD)을 위한 효율적인 frame sampling이다. 제한된 frame budget에서 이상 탐지에 중요한 프레임을 선택하여 visual encoding 비용을 줄이고, 탐지 성능을 최대한 유지하는 것이 목표다.

핵심 연구 질문은 다음과 같다.

> 같은 수의 프레임을 처리할 때, 어떤 sampling strategy가 이상 탐지 성능을 가장 잘 보존하는가?

추가로 다음을 확인한다.

- Sampling 비율을 줄일 때 성능과 실제 처리 시간이 어떻게 변하는가?
- 짧은 이상 사건이나 움직임이 작은 이상 상황에서도 중요한 프레임을 관측할 수 있는가?
- 서로 다른 두 benchmark에서 제안 방법의 효과가 일관되는가?

## 2. 문제의식

Uniform Sampling은 영상 전체를 고르게 관측하지만, 짧은 anomaly가 sampling 간격 사이에서 발생하면 사건 자체를 놓칠 수 있다. Random Sampling은 같은 budget에서도 선택 결과에 따른 변동이 생긴다.

Motion이나 frame difference가 크다는 사실만으로 anomaly 여부를 판단하기도 어렵다. 카메라 움직임이나 정상적인 활동 때문에 불필요한 프레임을 선택할 수 있고, 정차나 느린 이상 행동처럼 움직임이 작은 상황을 놓칠 수 있다. 이 예들은 분석 가설이며, 실제 benchmark의 영상과 annotation에서 해당 사례를 확인해야 한다.

따라서 연구의 초점은 단순히 프레임 수를 줄이는 데서 나아가 **anomaly-relevant frame을 제한된 budget 안에서 선택하는 것**이다.

## 3. 확정된 방향과 미정 사항

| 항목 | 현재 계획 |
|---|---|
| Main baseline | VadCLIP |
| Main benchmarks | UCF-Crime, XD-Violence |
| 모델 사용 방식 | 데이터셋별 공개 pretrained checkpoint를 이용한 inference |
| 통제 원칙 | CLIP encoder, VadCLIP 가중치와 평가 규칙을 고정하고 sampler만 변경 |
| 기본 비교 방법 | Full / Uniform / Random / Simple Adaptive / Ours |
| Frame budget | 100%, 50%, 25%, 12.5% |
| 주요 성능 지표 | UCF-Crime: frame-level ROC-AUC, XD-Violence: frame-level AP |
| 우선 진행 대상 | UCF-Crime에서 재현과 raw-video pipeline을 먼저 완성 |
| 제안 방법 Ours | 아직 설계하지 않음. Baseline 분석 후 구체화 |
| 추가 모델 검증 | 시간이 허용되면 RTFM 등으로 확장 검토 |

Random 반복 횟수, adaptive score의 구체적인 정의, frame-to-score 복원 방식, window 처리 규칙과 재현 허용 오차는 아래 실험 원칙을 바탕으로 구현 전에 확정한다. 이 문서의 세부 구현 제안은 이미 수행한 실험이나 확정된 성과를 뜻하지 않는다.

## 4. VadCLIP baseline과 데이터 준비

### 4.1. Baseline 선정 이유

VadCLIP을 사용하면 `raw frame sampling → CLIP visual encoding → VadCLIP inference` 구조에서 sampling 위치와 비용 절감 대상을 명확하게 설정할 수 있다. 공식 저장소는 UCF-Crime과 XD-Violence의 CLIP feature 및 pretrained model 링크를 제공한다. [공식 저장소](https://github.com/nwpu-zxr/VadCLIP)

UCF-Crime에는 UCF-Crime checkpoint를, XD-Violence에는 XD-Violence checkpoint를 사용한다. 각 데이터셋 안에서 모든 sampler가 같은 checkpoint를 사용한다. CLIP의 image/text encoder와 VadCLIP의 학습된 구성 요소는 모두 고정한다.

### 4.2. Benchmark 구성

UCF-Crime을 먼저 대상으로 pipeline을 검증한 뒤 XD-Violence로 확장한다. 두 데이터셋을 함께 사용하여 영상 및 anomaly 특성이 달라져도 sampling 효과가 유지되는지 평가한다.

| Benchmark | 주요 평가 지표 | 논문 보고값: 재현 참고치 |
|---|---|---:|
| UCF-Crime | Frame-level ROC-AUC | 88.02% AUC |
| XD-Violence | Frame-level Average Precision(AP) | 84.51% AP |

위 수치는 VadCLIP 논문 보고값이며 우리 실험 결과가 아니다. 공개 checkpoint와 평가 branch 등 실제 구성의 일치를 확인한 뒤 재현 결과와 비교한다. [VadCLIP 논문](https://ojs.aaai.org/index.php/AAAI/article/view/28423)

### 4.3. 필요한 데이터

두 데이터셋에서 다음을 확보한다.

- 공식 test split에 해당하는 normal/anomaly raw videos
- Test 영상 목록과 temporal anomaly annotation
- Baseline 재현용 official test CLIP features
- 데이터셋별 pretrained checkpoint 및 설정 정보

기존 대화에서 확인한 UCF-Crime 배포 파일명 기준으로는 다음을 우선 준비한다. 실제 사용 대상은 공식 test 목록으로 필터링한다.

```text
Anomaly-Videos-Part-1.zip
Anomaly-Videos-Part-2.zip
Anomaly-Videos-Part-3.zip
Anomaly-Videos-Part-4.zip
Testing_Normal_Videos.zip
UCF_Crimes-Train-Test-Split.zip
Temporal_Anomaly_Annotation_for_Testing_Videos.txt
```

XD-Violence의 다운로드 구성과 파일명은 별도로 확인한다. 공개 모델 링크의 존재와 실제 다운로드·로딩 성공은 구분하여 기록한다.

현재의 고정 checkpoint inference에는 VadCLIP 재학습을 위한 train data가 필수는 아니다. 다만 sampler 학습이나 hyperparameter 선택이 필요해지면 test set과 분리된 train/development/validation data를 확보해야 한다.

Baseline 실행 준비 전에 raw test 데이터를 먼저 정리한다. 공식 test 목록을 선택 기준으로 삼고 annotation의 category로 anomaly/normal을 구분한다. Annotation 파일 전체를 anomaly 목록으로 간주하지 않는다.

```text
UCF-Crime/test/
├── Anomaly/<category>/*.mp4
├── Normal/*.mp4
├── annotations/
├── splits/{test,anomaly,normal}.txt
└── manifest.{json,csv}
```

이미 압축 해제된 영상이 있으면 재사용하고, test 폴더는 원본 영상으로 연결되는 상대 symlink로 구성할 수 있다. 이 경우 원본 extracted 폴더도 함께 보존한다. Manifest에는 공식 feature CSV의 평가 순서, 원본 경로, ZIP entry, 파일 식별값, 영상 metadata와 원본 annotation interval을 기록한다. 파일 존재·누락·중복·ZIP CRC/크기 일치와 metadata를 점검하고, 전체 디코딩 검증 여부는 별도로 기록한다. GT interval의 좌표 규칙과 feature/GT 배열 정렬은 이후 평가 구현 단계에서 검증한다.


## 5. Baseline reproduction

Sampling 실험을 시작하기 전에 재현을 두 단계로 나눈다.

### Stage 1. Official feature 기반 재현

```text
Official test CLIP features
          ↓
Dataset-specific pretrained VadCLIP
          ↓
Official scoring / evaluation
          ↓
AUC 또는 AP 확인
```

이 단계에서는 checkpoint, feature 형식, test 목록과 evaluation code가 함께 정상적으로 동작하는지 확인한다. 논문 수치와 차이가 있으면 score branch, checkpoint 버전, feature aggregation, annotation 정렬과 후처리를 점검한다. 실제 재학습 없이 공개 checkpoint의 평가 결과를 재현하는 단계다.

### Stage 2. Raw video 기반 reference 재현

```text
Raw test video
      ↓
Reference frame extraction / preprocessing
      ↓
동일한 CLIP encoder
      ↓
동일한 pretrained VadCLIP
      ↓
동일한 평가
```

공식 feature 추출 조건을 확인하여 CLIP 종류, FPS/stride, resize/crop, normalization, feature 차원 및 aggregation을 맞춘다. 대표 영상에서 frame 수·timestamp·feature 형태를 점검하고, 전체 test metric을 Stage 1과 비교한다.

허용할 재현 오차는 sampling 결과를 보기 전에 정한다. 차이가 설명되지 않는 상태에서는 sampling 성능을 해석하지 않는다. 공식 feature 추출 조건을 완전히 복원하지 못하면 그 제한을 기록하고, 검증된 raw-video reference를 이후 sampling 비교의 기준으로 삼는다.

## 6. Raw-video sampling pipeline

```text
Raw video
    ↓
Frame decoder + 원본 frame index / timestamp 유지
    ↓
Sampling strategy: Full / Uniform / Random / Adaptive / Ours
    ↓
Selected frames: 시간순 정렬
    ↓
고정된 전처리 + CLIP visual encoder
    ↓
고정된 pretrained VadCLIP
    ↓
선택 시점의 anomaly scores
    ↓
원본 annotation timeline으로 score 복원
    ↓
AUC / AP + efficiency + sampling 분석
```

**최종 효율 실험의 sampling은 CLIP visual encoder 이전에 수행한다.** 전체 프레임의 CLIP feature를 먼저 계산한 다음 feature만 선택하는 실험은 초기 탐색에는 사용할 수 있지만, visual encoding 비용 절감의 근거로 사용할 수 없다.

Sampler가 모든 프레임을 저해상도로 살펴보거나 별도 모델을 사용하는 경우 그 비용도 포함한다. 영상 압축 구조 때문에 선택 프레임만 필요해도 decoder가 중간 프레임을 처리할 수 있으므로, encoder 입력 감소가 decoding 비용 감소로 그대로 이어진다고 가정하지 않는다.

비균일 sampling에서는 인접한 입력 token 사이의 실제 시간 간격이 달라진다. 원본 timestamp를 보존하고, 모델의 positional encoding, window 구성, padding/mask 처리와 score 복원 규칙을 사전에 고정한다. 모델 구조와 가중치는 변경하지 않으며, sampling에 따라 발생하는 시간 정보 손실은 결과 해석에 포함한다.

초기 연구는 전체 영상에서 프레임을 선택하는 offline inference로 진행한다. 실시간·온라인 성능을 주장하려면 미래 프레임 사용 여부와 처리 지연을 별도로 검증해야 한다.

## 7. Sampling baselines

| 방법 | 정의 | 확인할 특성 |
|---|---|---|
| Full / Reference | 기준 후보 프레임을 모두 encoder에 입력 | 성능과 시간의 기준점 |
| Uniform | 전체 시간축에서 균등하게 K개 선택 | Coverage와 짧은 사건 누락 |
| Random | 후보 중 K개를 중복 없이 무작위 선택 후 시간순 정렬 | Seed에 따른 성능 변동 |
| Simple Adaptive | 저비용 frame difference 또는 motion score로 K개 선택 | 큰 정상 motion 편향과 작은 anomaly 누락 |
| Ours | 이상 탐지에 유용한 정보와 시간축 coverage를 고려하는 방법 | 같은 budget에서 성능 보존과 실제 비용 절감 |

Adaptive는 예를 들어 frame difference score 상위 K개를 선택하는 단순한 구현부터 시작한다. 점수 계산 해상도, 비교 간격, 동점 처리와 선택 규칙을 명시한다. Ours의 구체적인 구조는 아직 미정이다.

Random은 동일한 seed 목록으로 반복한다. 초기 제안은 5개 seed이며 평균과 표준편차를 보고한다. 다른 방법에 무작위 요소가 있으면 같은 반복 원칙을 적용한다.

## 8. Frame budget과 공정한 비교

영상 v의 reference 후보 프레임 수를 `N_v`, budget 비율을 `r`이라 할 때 선택 수는 다음과 같이 정의한다.

```text
K_v = min(N_v, max(1, ceil(r × N_v)))
```

| Budget | Reference 후보 대비 encoder 입력 수 |
|---|---:|
| 100% | N_v |
| 50% | 약 N_v / 2 |
| 25% | 약 N_v / 4 |
| 12.5% | 약 N_v / 8 |

`N_v`는 Stage 2에서 고정한 FPS/stride의 후보 프레임 수다. 모든 원본 프레임을 후보로 삼는 경우에만 100%가 원본 전체 프레임과 같다. Reference의 실제 추출 비율과 원본 frame 수를 함께 기록하여 분모를 명확히 한다.

각 영상에서 sampler마다 동일한 `K_v`를 적용한다. 짧은 영상의 반올림과 빈 영상 처리도 동일하게 한다. 비교는 데이터셋별로 `4개 budget × 모든 sampling strategy`를 수행한다. 100%에서는 모든 후보가 선택되어 reference와 같은 탐지 결과가 나오는지 확인한다.

**같은 frame budget이 같은 총 계산 비용을 뜻하지는 않는다.** Sampler overhead가 다르므로, 먼저 같은 K에서 성능을 비교하고 측정된 총 시간에 따른 성능–효율 곡선도 제시한다.

## 9. Evaluation metrics

### 9.1. Detection performance

- UCF-Crime: frame-level ROC-AUC를 주 지표로 사용한다.
- UCF-Crime 보조 지표: Ano-AUC도 기록한다. Manifest의 anomaly 영상 전체 프레임을 모아 ROC-AUC를 계산하며, 해당 영상 내부의 정상 프레임도 포함한다. Normal-only 영상은 제외한다. 영상별 AUC의 단순 평균이 아니다.
- UCF-Crime fine-grained 보조 평가: full 모델 평가에서 A-branch의 14개 클래스 점수를 정상 영상 포함 290개 전체에 대해 복원하여 temporal mAP@IoU 0.1/0.2/0.3/0.4/0.5 및 AVG를 기록한다. 원본 segment/class GT와 수정하지 않은 공개 evaluator를 사용하며 원본 UCF test의 excludeNormal=False 호출을 따른다. 사용자 요청으로 별도 저장-score 비교 대신 full 기본 평가를 수정했다. 이전 140개/13개 클래스 결과는 다른 protocol의 과거 기록으로 구분하고, 측정 전 논문 성능 재현을 주장하지 않는다.
- XD-Violence: frame-level AP를 주 지표로 사용한다.
- Normal 영상까지 포함하여 공식 평가의 영상 순서, GT 정렬 및 집계 방식을 유지한다.
- Sampling 결과도 원본 annotation timeline의 모든 평가 프레임에 score를 복원한다. 선택된 프레임만 평가하지 않는다.
- Score branch, branch 결합, smoothing 및 calibration을 sampler별로 바꾸지 않는다.

Score 복원은 예를 들어 nearest-neighbor 또는 linear interpolation 중 하나를 사전에 선택한다. 시작·끝 구간의 처리도 고정한다. 비균일 선택 위치에 단순히 일정 횟수의 score 반복을 적용하면 시간축이 어긋날 수 있으므로 원본 index/timestamp로 복원한다.

Raw-video reference 대비 `Δmetric = sampled metric − reference metric`을 percentage point(pp)로 보고한다. Official-feature 재현 수치와 raw-video reference 수치는 별도 행으로 기록한다.

### 9.2. Efficiency

| 지표 | 측정 내용 |
|---|---|
| Processed frame ratio | 실제 encoder 입력 프레임 수 / reference 후보 프레임 수 |
| Decoding time | 영상 읽기와 프레임 복호화 시간 |
| Sampling time | 점수 계산과 선택에 필요한 모든 비용 |
| Feature extraction time | 전처리 및 CLIP visual encoding 시간 |
| VAD inference time | Temporal modeling과 anomaly scoring 시간 |
| End-to-end time | 영상 입력부터 원본 시간축의 최종 score 출력까지 실제 경과 시간 |
| Speedup | Reference end-to-end time / 해당 방법 end-to-end time |
| Peak memory, FLOPs | 측정 가능하면 보조 지표로 기록 |

같은 장비, batch 정책, precision, decoder와 I/O 조건을 사용한다. Warm-up과 GPU 동기화를 적용하고 반복 측정한다. 개별 단계 시간과 전체 경과 시간을 함께 기록하며, 파이프라인에서 단계가 겹치면 단순 합과 전체 시간이 다를 수 있음을 명시한다.

Feature cache로 탐지 성능을 반복 평가할 수 있으나, 최종 raw-video 시간 측정은 feature 생성과 sampler 비용을 포함하는 조건에서 수행한다. 25% frame 입력을 곧바로 4배 speedup으로 표현하지 않는다.

### 9.3. Sampling 자체의 분석 지표

선택한 원본 frame index 집합을 `S`, anomaly 구간을 `I_j`, 전체 anomaly frame 집합을 `A`라고 정의한다.

```text
Event coverage = |{j : S ∩ I_j가 비어 있지 않음}| / 전체 anomaly interval 수
Anomaly frame recall = |S ∩ A| / |A|
Anomaly selection precision = |S ∩ A| / |S|
```

Event coverage는 각 annotation 구간에 선택 프레임이 하나라도 있는지를 측정한다. Annotation interval은 의미적 사건 전체와 다를 수 있으므로 구간 단위 지표라는 점을 명시한다. Frame recall과 selection precision은 anomaly를 포함한 영상에 대해 정의하고 집계 방식을 고정한다.

GT annotation은 평가·분석에만 사용하며 실제 sampler의 입력으로 사용하지 않는다. Reference 후보 자체가 놓친 구간도 보고하여, candidate 생성 단계와 sampler 단계의 누락을 구분한다. Event coverage가 높아도 탐지에 필요한 문맥이 부족하면 AUC/AP가 낮을 수 있다.

## 10. Failure analysis

| 방법 또는 원인 | 예상 실패 | 확인할 증거 |
|---|---|---|
| Uniform | 짧은 anomaly가 sampling 간격 사이에 위치 | 사건 길이, 선택 시점, 구간 누락률 |
| Random | Seed에 따라 anomaly 관측 여부가 달라짐 | Seed별 coverage와 metric 변동 |
| Motion / difference 기반 | 카메라 motion이나 정상 활동에 budget 집중 | Normal 구간 선택 비율과 false positive |
| Motion / difference 기반 | 정적·느린 anomaly를 적게 선택 | 저 motion anomaly 구간의 coverage |
| Adaptive / Ours | 한 구간에 선택이 몰려 다른 사건·문맥 누락 | 시간축 선택 분포와 여러 구간의 coverage |
| 공통 | 사건은 관측했지만 시간 문맥이 부족 | Coverage 유지에도 score/metric이 하락하는 사례 |
| 공통 | 비균일 입력과 checkpoint의 시간 모델링 조건 불일치 | 선택 간격, window 경계, reference score와의 차이 |
| 공통 | 잘못된 score 복원이나 annotation 정렬 | 원본 timestamp와 GT/예측 경계 비교 |
| 공통 | Sampler overhead로 실제 시간 절감 실패 | 단계별 시간과 end-to-end speedup |

대표 영상별로 GT interval, 선택 프레임, reference/sample anomaly score를 한 시간축에 표시한다. 사건 길이와 motion 수준별 성능을 함께 보고, sampler 성공 사례와 실패 사례를 모두 남긴다.

이 분석은 Ours의 설계 동기를 구체화하는 데 사용한다. 설계에 활용할 사례는 별도 development/validation data에서 분석하는 것을 우선한다. Test 결과를 이미 보고 방법을 수정한 경우 그 결과는 탐색적 분석으로 구분하고, 최종 검증에는 설계에 사용하지 않은 평가 데이터를 확보한다.

## 11. 실험 원칙

1. **모델 고정:** 데이터셋 안에서 encoder, checkpoint, prompt, score branch와 후처리를 동일하게 유지한다.
2. **Sampling 위치 고정:** 비용 절감의 대상인 visual encoder 이전에 프레임을 선택한다.
3. **Budget 고정:** 동일 영상·동일 비율에서 실제 선택 수를 동일하게 맞춘다.
4. **시간축 보존:** 선택 index/timestamp, 시간순 정렬, window와 score 복원 규칙을 기록한다.
5. **GT 사용 제한:** GT를 sampling 선택이나 test hyperparameter 최적화에 사용하지 않는다.
6. **개발과 평가 분리:** 학습·튜닝이 필요하면 별도 개발 데이터를 확보하고 최종 평가 전에 방법을 고정한다.
7. **전체 비용 측정:** Decoder, sampler, encoder, VAD와 score 복원 비용을 포함한다. 추가 encoder 조회나 중복 처리도 기록한다.
8. **재현성 확보:** 코드 버전, checkpoint 식별값, split, 전처리, seed, 장비와 실행 설정을 저장한다.
9. **주장 범위 유지:** 두 benchmark의 결과로 데이터셋 간 일관성을 평가하되, 다른 architecture로의 일반화는 추가 실험 후 판단한다.

## 12. 실험표와 산출물

아래 표는 25% budget의 기록 예시다. 다른 budget도 동일한 형식으로 작성하고, 실제 측정 전에는 값을 채우지 않는다.

| Dataset | Sampling | Budget | AUC 또는 AP (%) | Δ (pp) | End-to-end (s) | Speedup | Event coverage (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| UCF-Crime | Full / Reference | 100% | — | 0 | — | 1.00× | — |
| UCF-Crime | Uniform | 25% | — | — | — | — | — |
| UCF-Crime | Random | 25% | — | — | — | — | — |
| UCF-Crime | Adaptive | 25% | — | — | — | — | — |
| UCF-Crime | Ours | 25% | — | — | — | — | — |
| XD-Violence | Full / Reference | 100% | — | 0 | — | 1.00× | — |
| XD-Violence | Uniform | 25% | — | — | — | — | — |
| XD-Violence | Random | 25% | — | — | — | — | — |
| XD-Violence | Adaptive | 25% | — | — | — | — | — |
| XD-Violence | Ours | 25% | — | — | — | — | — |

Random 등의 확률적 방법은 평균 ± 표준편차를 기입한다. 결과와 함께 다음을 남긴다.

- 데이터셋별 official-feature 재현 결과와 raw-video reference 검증 기록
- Budget–성능 및 실제 처리 시간–성능 곡선
- 단계별 비용, 실제 encoder 입력 수와 실행 설정
- 영상별 선택 index/timestamp 및 anomaly scores
- 사건 길이·motion 수준별 분석과 대표 failure case
- 제안 방법을 구성한 뒤 각 요소의 효과와 추가 비용을 확인하는 ablation

## 13. 진행 순서

| 순서 | 작업 | 완료 기준 |
|---:|---|---|
| 1 | UCF-Crime test 데이터, annotation, official feature와 checkpoint 확보 | 영상 목록·GT·feature 대응과 모델 로딩 확인 |
| 2 | UCF-Crime official-feature baseline reproduction | AUC 계산 완료, 논문과의 차이를 기록·설명 |
| 3 | UCF-Crime raw-video reference 구축 | 전처리·시간축 검증, Stage 1과의 차이 점검 |
| 4 | Uniform / Random 적용 | 4개 budget에서 metric과 시간 측정 완료 |
| 5 | XD-Violence로 pipeline 확장 | 데이터셋별 checkpoint와 AP 평가 재현 |
| 6 | 두 benchmark의 공통 baseline 비교 | Simple Adaptive 포함, budget·평가 규칙 일치 |
| 7 | Failure analysis | 사건 누락, 문맥 손실, motion 편향과 overhead 분석 |
| 8 | Ours 설계 및 개발 | 별도 개발 데이터에서 설계·튜닝하고 방법 고정 |
| 9 | 두 benchmark 최종 검증 | 고정된 방법으로 성능–효율 비교와 ablation 완료 |
| 10 | 선택적 확장 | 다른 VAD architecture에서 효과 검증 |

**먼저 UCF-Crime raw test 데이터를 정리·검증한 뒤 pretrained VadCLIP baseline을 재현한다.** 이후 raw-video reference가 검증되면 sampling 비교를 시작한다.

## 14. 참고 자료와 확인 범위

- 대화 기록: [베이스라인 VAD 모델 추천](chatgpt-conversation://6abccc83-fd20-83ee-a342-03f3e913740b) — 연구 방향, 데이터 준비와 단계별 진행 순서의 근거.
- [VadCLIP 공식 저장소](https://github.com/nwpu-zxr/VadCLIP) — 데이터셋별 공개 feature/model 링크 및 실행 안내.
- [VadCLIP: Adapting Vision-Language Models for Weakly Supervised Video Anomaly Detection, AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/28423) — 논문과 보고 성능.

문서 작성 시 공식 자료의 benchmark, 공개 자원 안내와 논문 보고값을 확인했다. 데이터 다운로드, checkpoint 로딩, raw feature 추출 및 baseline 실험은 아직 수행하지 않았다.
