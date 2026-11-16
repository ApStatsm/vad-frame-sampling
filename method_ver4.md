# ver4 — 배치별 점수 손실 분해와 고정 대상 주변 프레임 교체

작성일: 2026-11-15 UTC. 작업 위치: `/workspace/hyeongjun/vad`.

작성 당시 코드·정적 검토를 완료했다. 신규 Python5개 AST, JSON1개 구문, 신규 코드·문서 whitespace 검사를 통과했고 사용자 테스트6개를 작성했다. **이후 사용자 실행의 A 분석과 B full 진단 성공을 확인했으며 실제 결과는 [evaluation_ver4.md](evaluation_ver4.md)에 있다.** Codex가 분석·추론을 실행한 것은 아니다. 테스트·smoke의 별도 성공은 full 완료만으로 추정하지 않는다. ver3의 실제 결과는 [evaluation_ver3.md](evaluation_ver3.md)에 있다.

## 1. 목적과 범위

ver3 짝 배치는 Uniform보다 AP가0.0587pp 높았으나 무작위 재배치 평균보다 낮았고, FPR과 전체 사건 포착도 개선하지 못했다. 이 결과만으로 주변 맥락의 효과를 판정할 수 없다. 배치 변경은 관측 위치·내용·복원을 함께 바꿨기 때문이다.

ver4는 두 진단을 수행한다.

1. **A: 배치별 손실 분해.** 각 ver3 배치 위치에 Full 점수만 남겨 복원하고, 실제 sparse 추론 결과와 비교한다. 모델을 새로 추론하지 않는다.
2. **B: 고정 대상의 주변 교체.** 대상 프레임의 feature·입력 슬롯·전체 token 수를 유지하고 양옆 두 프레임만 시간적으로 가까운 원본 후보로 바꿔 추론한다.

이는 기존 test 결과를 본 뒤 정한 탐색적 원인 진단이다. 최종 Ours나 일반화 benchmark가 아니며, raw decoding·CLIP 비용 절감 또는 speedup을 입증하지 않는다. Padding 제외 등 모델 계산 변경은 이번 범위에서 제외한다.

## 2. 입력과 공통 검증

기본 입력은 완료된 ver3 sweep이다.

```text
outputs/analyses/xd_context_ver3_full_20261116_022711/sweep_ver3.json
```

필요 파일:

- sweep이 참조하는 성공8회(Full, Uniform, paired, relocated seed0–4)의 원본 `summary.json`.
- 해당 run의 `scores/*.npz`, `frame_scores.npz`.
- 공용 raw Reference의 `manifest.json`과 B에서 사용할 `features/*.npz`.
- 기존 XD GT·공식 목록. B는 추가로 CLIP·XD VadCLIP checkpoint와 모델 실행 환경이 필요하다.

공통 검증기는 sweep 완료·오류 없음, summary8개 hash, 성공/full/800영상, 동일 cache·환경·모델 옵션·소스·comparison signature, 공식 순서·영상별 N/K·GT offset·평가 길이를 확인한다. GT는 공식 전체 시간축2,331,296프레임이며 manifest 길이와 일치해야 한다.

원본 score NPZ는 실제 선택 좌표가 사전 고정한 ver3 규칙과 일치하고, 확률이 유한한0–1 범위인지 검사한다. 읽은 파일 hash를 새 결과에 남긴다. 이는 현재 읽은 파일의 identity와 내부 일관성 검증이며 과거 NPZ의 서명된 원본성을 뜻하지 않는다.

## 3. A — Full 점수의 배치별 재복원

두 branch에서 모든 ver3 조건을 각각 분석한다.

| 조건 | 점수 생성 | 전체 시간축 평가 |
|---|---|---|
| Full | 모든 reference feature로 기존 추론한 직접 점수 | reference token마다16프레임 반복 |
| Layout-restored Full | 해당 배치가 선택한 원본 좌표의 Full 직접 점수만 사용 | 실제 선택 좌표 anchor·nearest 복원 후16회 반복 |
| Actual sparse | 해당 배치의 기존 sparse 추론 직접 점수 | 같은 anchor·복원 규칙 |

Full/sparse의 reference grid가 정확히 같아야 하며, 선택 좌표가 Full 후보에 존재하지 않으면 중단한다. 이전 구간 시작점이나 가장 가까운 다른 후보로 대신 대응시키지 않는다.

각 run·branch에서 Full AP, 재복원 AP, 실제 AP 및 두 단계 차이(pp)를 기록한다. 동일 재현율 FPR도 재복원·실제 조건에서 비교한다. 목표는 branch별 Full 점수 임계값0.5 재현율이며, 선형보간값과 실제 도달 가능한 첫 ROC 점의 FPR/재현율을 함께 기록한다. XD 주 지표 해석은 Branch2 AP로 한다.

모든 run에서 직접 점수의 전체 복원 결과가 저장된 `frame_scores.npz`와 일치해야 하고, AP가 기존 summary와 절대오차1e-10 이내로 재현돼야 한다. Full 직접 점수도 전체 배열과 대조한다. 검증 실패 시 성공 결과로 처리하지 않는다.

AP는 비선형 지표이므로 두 단계 차이를 독립적인 인과 요인의 기여율로 해석하지 않는다. 재복원 조건의 점수는 이미 Full 입력의 정보를 반영하므로 실행 가능한 저비용 sampler가 아니라 진단용 조건이다. Random5seed는 모두 포함하고 평균·표본 표준편차를 보고한다.

## 4. B — 대상 프레임·입력 위치·token 수를 고정한 개입

### 4.1 대상 선정

기준은 Uniform12.5% 시퀀스이며 reference row 좌표는 `U[p]=8*p`, K=`ceil(N/8)`이다. 선택 함수는 N만 입력받고 GT·feature·점수는 받지 않는다.

eligible 슬롯은 다음을 모두 만족하는 p다.

- `1 <= p < K-1`: 유효한 양옆 입력 슬롯이 있다.
- `1 <= p mod 64 <= 62`: 대상과 양옆이 같은 XD local attention64-token 구간 안에 있다. 따라서256-token 추론 window도 같다.

eligible 목록 길이를 M이라 할 때 목록 index `floor((M-1)/3)`, `floor(2*(M-1)/3)`의 슬롯을 선택한다. 중복이면 하나만 사용하므로 영상당 최대2개다. eligible이 없으면 사유를 기록하고 해당 영상의 개입은 생략한다. 성능이나 이상 라벨을 보고 대상을 바꾸지 않는다.

Full 모드는800개 영상 모두의 대상 가능 여부를 기록하지만, 전체 영상 시간축 AP를 새로 평가하는 실험은 아니다. 최대1,600개의 대상과 양 branch 최대3,200행이며 실제 개수는 실행 후 확인한다. 정상·이상 대상 비율을 사전에 균등화하지 않는다.

### 4.2 세 조건

각 영상에서 Uniform 기준 추론과 동일 입력 재추론을 수행한다. 각 대상에 대해서는 **독립적인 한 번의 주변 교체**를 수행한다. 두 대상의 교체를 한 시퀀스에 동시에 적용하지 않는다.

| 조건 | 대상 주변 row | 바뀌는 부분 |
|---|---|---|
| Baseline | `..., t-8, t, t+8, ...` | 없음 |
| Unchanged replay | Baseline과 완전히 동일 | 없음. 수치적 반복 안정성 대조 |
| Near neighbors | `..., t-1, t, t+1, ...` | 대상의 바로 양옆2개 feature만 교체 |

t는 대상의 reference row다. 원본 프레임 간격은 Baseline128, Near16프레임이며, XD24fps 기준 약5.33초와0.67초다. 두 시점은 같은 영상에서 가져오며 원본512D feature의 dtype/값을 바꾸지 않는다.

대상 feature와 압축된 입력 슬롯은 동일하다. 모든 다른 슬롯의 feature, K, token 순서, 대상 window·local attention 구간·positional embedding 위치, 선언 길이·padding 수, prompt·모델 가중치도 유지한다. 모델 계산 자체는 변경하지 않는다.

시간순·중복 없음·K·정확히 양옆2개 슬롯만 변경됐는지 검사한다. 대상이 첫 내부 슬롯인 짧은 영상에서는 첫/마지막 선택 프레임이 교체될 수 있다. B는 ver3의 양끝 고정 배치 benchmark가 아니라 대상 중심 개입이다.

### 4.3 재현과 안정성 확인

- checkpoint, GT·split·tokenizer, frozen 모델/평가/복원 소스와 모든 vendored Python hash를 ver3와 대조한다. 추가된 ver4 파일 때문에 전체 package hash가 달라지는 것은 허용하지만 새 hash도 보존한다.
- XD 모델 옵션은 `visual_length=256`, `attn_window=64`, model seed234를 포함해 archived Uniform과 일치해야 한다.
- 처리하는 영상의 feature cache hash가 ver3 기록과 일치해야 한다.
- 현재 Baseline과 archived Uniform의 양 branch **전체 유효 token 점수** 최대 절대차가1e-5 이하여야 한다.
- Baseline과 Unchanged replay의 최대 절대차가1e-6 이하여야 한다.
- Near의 token 수·window 수·선언 길이·실제 유효 길이는 Baseline과 일치해야 한다.

재현 또는 안정성 확인을 통과하지 못하면 실행을 실패로 기록한다. 허용오차를 결과에 맞춰 자동 확대하지 않는다. 현재·archived 실행 환경 차이는 별도로 확인하며, 허용오차 통과가 모든 환경의 동등성을 증명하지는 않는다.

### 4.4 분석 항목

대상 선정·추론 후 공식 GT로 정상/이상을 구분한다. 두 branch별로 다음을 저장한다.

- Baseline, Replay, Near의 대상 직접 점수와 `Near−Baseline`, `Replay−Baseline`.
- 같은 원본 위치의 기존 Full 직접 점수, Baseline·Near의 Full 대비 절대차, 절대차의 변화. 음수이면 Full 점수에 가까워진 것이다.
- GT별 대상 수·영상 수, 점수 변화 평균·중앙값·상승 비율.
- 영상별 평균 변화의 평균·표본 표준편차. 대상 프레임을 독립 표본으로 취급한 유의성 검정은 하지 않는다.

전체 시퀀스의 점수는 개입별 NPZ로 저장하지만, 서로 다른 개입에서 얻은 대상 점수를 합쳐 전체 시간축 AP를 만들지 않는다. Full 점수에 가까워지는 것이 GT 판별의 개선과 동일한 의미도 아니다.

## 5. 결과 해석

- A에서 재복원 AP도 낮으면 관측 위치·복원 손실을 고려해야 한다. 실제 점수로 바꾸며 추가 하락이 크면 sparse 입력에 대한 모델 반응을 조사할 근거가 된다.
- B에서 정상 대상의 점수가 낮아지고 이상 대상 점수는 유지되면, 이 고정 조건에서 가까운 주변 관측이 판별에 유리할 수 있다는 근거다.
- 모든 대상의 점수가 함께 낮아지면 정상·이상 분리를 개선한 것인지 추가로 확인해야 한다. Full에 가까워졌다는 이유만으로 탐지 성능 개선을 주장하지 않는다.
- 변화가 없거나 악화하면 이 두 이웃 교체가 도움이 되지 않은 결과로 보고한다. 더 넓은 맥락의 효과까지 부정하지 않는다.
- B는 대상·위치·token 수·padding을 통제하지만 이웃의 시간 간격과 내용은 함께 바뀐다. 의미적 맥락·attention·graph의 개별 기전은 식별하지 않는다.
- 이후 sampler로 발전시키려면 별도 개발 데이터에서 규칙을 고정하고 독립 평가·동일 조건 raw 시간 측정을 수행한다.

## 6. 코드와 실행 순서

- [공통 검증·기하 규칙](vadfs/context_ver4.py)
- [A 분석기](scripts/analyze/decompose_context_ver4.py)
- [B 고정 대상 추론기](scripts/probe_context_ver4.py)
- [공통 실행 진입점](scripts/run_context_ver4.py)
- [고정 설정](configs/experiments/xd_context_ver4.json)
- [사용자 테스트](tests/test_context_ver4.py)

기존 모델·ver3 실행/분석 코드는 이번 ver4를 위해 변경하지 않는다. ver4는 기존 helper와 고정 inference를 재사용하는 별도 경로다.

**ver3를 성공시킨 Python 환경을 활성화한 터미널**에서 아래 `python`을 사용한다. 별도 사용자 폴더의 Python 경로를 하드코딩하지 않는다. 아래 명령의 프로젝트 코드는 모두 hyeongjun 폴더에 있다.

### 6.1 사용자 테스트

```bash
cd /workspace/hyeongjun/vad
python -m unittest discover -s tests -p 'test_context_ver4.py' -v
```

geometry·attention 경계·대상 feature 보존·복원 tie/좌표 대응을 확인한다. 실제 checkpoint 추론 검증을 대신하지 않는다.

### 6.2 A: 기존 점수 분석 — 새 모델 추론 없음

```bash
python scripts/run_context_ver4.py \
  --stage decomposition \
  --sweep-ver3 outputs/analyses/xd_context_ver3_full_20261116_022711/sweep_ver3.json
```

A는 기존800영상 전체 결과를 분석하며 `--mode`와 GPU를 사용하지 않는다. 저장 점수 배열 처리·metric 계산은 이 명령을 사용자가 실행할 때 수행한다.

### 6.3 B: Smoke

사용 가능한 GPU를 선택한다. 아래0은 예시이며 GPU 예약·프로세스 중단을 자동 수행하지 않는다.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_context_ver4.py \
  --stage context --mode smoke --device cuda \
  --local configs/local/hj.xd.json \
  --sweep-ver3 outputs/analyses/xd_context_ver3_full_20261116_022711/sweep_ver3.json
```

기존 smoke 대표 영상 규칙을 사용한다. source8개 요약과 GT 정렬을 확인한 뒤 대표 영상에서만 추론한다. `summary_ver4.json`의 성공·오류와 재현 오차, 실제 대상 존재 여부를 확인한다. 정상·이상 모두 충분한 표본이 있다고 가정하지 않는다.

### 6.4 B: Full

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_context_ver4.py \
  --stage context --mode full --device cuda \
  --local configs/local/hj.xd.json \
  --sweep-ver3 outputs/analyses/xd_context_ver3_full_20261116_022711/sweep_ver3.json
```

한 명령으로 A와 B를 순차 실행하려면 `--stage all --mode full`을 사용한다. A가 실패하면 B를 시작하지 않는다. 자동 resume은 없으며 재실행은 새 폴더를 만든다. 중간 성공 영상·부분 기록은 남지만 최종 `success=true` 전에는 완료 결과로 사용하지 않는다.

## 7. 출력 및 사용자 확인 항목

| 경로 | 내용 |
|---|---|
| `outputs/analyses/xd_context_ver4_<stage>_<timestamp>/sweep_ver4.json` | 단계별 출력 위치·성공·오류 |
| `outputs/analyses/xd_decomposition_ver4_full_<timestamp>/` | A 결과 |
| `decomposition_runs_ver4.csv` | 8run×2branch의 단계별 AP/FPR, 총16행 |
| `decomposition_comparison_ver4.csv` | 방법별 평균·표본 표준편차, 총8행 |
| `summary_ver4.json` | 입력·소스 hash·완료 상태 |
| `outputs/runs/xd_context_probe_ver4_<mode>_<timestamp>/` | B 결과. local output_root가 다르면 해당 경로 |
| `target_scores_ver4.csv` | 대상별 좌표·GT·branch·조건별 점수 |
| `target_comparison_ver4.csv` | branch×GT 기술통계4행 |
| `scores/*_ver4.npz` | Baseline/Replay와 대상별 Near의 전체 직접 점수·선택 row |

사용자는 A의 성공·AP 재현, B full의 영상 기록800개·skip 사유·실제 대상 수·정상/이상 표본 수, archived/replay 허용오차를 먼저 확인한다. 그 다음 정상·이상 점수 변화와 Full 대비 절대차를 함께 해석한다. B에는 whole-timeline AP나 raw speedup이 없다.

사용자 실행 완료 결과와 한계는 [evaluation_ver4.md](evaluation_ver4.md)에 기록했다. 작성 시 정적 검토와 후속 사용자 실제 실행 검증을 구분한다.
