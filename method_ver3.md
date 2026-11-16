# ver3 — 동일 프레임 예산에서 시간적 배치와 국소 관측 비교

작성일: 2026-11-15. 작업 위치: `/workspace/hyeongjun/vad`.

작성 당시 구현·정적 검토만 완료했다. **후속 사용자 실행에서 full8회와 자동 분석 완료를 확인했으며 결과는 [evaluation_ver3.md](evaluation_ver3.md)에 있다.** 본 문서는 XD-Violence 12.5%의 탐색 프로토콜이다. 기존 ver1/ver2와 별도이며, 성능을 확인한 뒤 설정을 선택하는 최종 benchmark가 아니다.

## 1. 질문과 해석 범위

기존 분석에서 사건 포착률 향상이 AP 개선을 보장하지 않았고, XD Uniform12.5%의 실제 sparse 추론은 같은 위치에 Full 점수를 재복원한 조건보다 AP가 낮았다. 정상 선택 프레임의 점수 상승도 관찰됐다. 근거와 한계는 [실험 의의 보고서](EXPERIMENT_SIGNIFICANCE_REPORT.md)에 정리했다.

ver3의 질문은 **동일한 입력 수에서 일부 프레임을 가까운 짝으로 배치하면 Uniform보다 AP와 정상 점수 분리를 개선할 수 있는가?**다. GT·CLIP 특징·모델 점수는 선택에 사용하지 않는다. 배치 규칙만 바꾼다.

짝 배치는 국소 관측을 늘리는 대신 coverage와 최대 간격을 바꾼다. 따라서 성공해도 주변 맥락의 단일 인과 효과를 입증한 것은 아니다. 시간 간격, 장면 내용, positional embedding, window/padding의 개별 효과는 분리하지 않는다.

## 2. 고정 조건과 입력

- XD official test 800개, normal 포함. 주 지표는 Branch 2 frame-level AP다.
- 입력은 공용 raw Reference의 저장된 512D CLIP feature 원본 행이다. 정규화·평균·투영·학습 없이 원본 dtype/값을 그대로 모델에 전달한다.
- 확인된 cache 위치: `/workspace/vad/results/baselines/xd_raw_reference/`. `summary.json`, `manifest.json`, `features/*.npz`가 필요하다. 원본은 수정하지 않는다.
- XD checkpoint와 CLIP checkpoint, 공식 GT·split·prompt·모델 옵션은 기존 공통 실행기의 검증을 따른다. 로컬 경로는 `configs/local/hj.xd.json`을 사용한다.
- 기본 reference 좌표는 `16*j`, budget 분모는 reference 후보 수 N이다. K는 `ceil(N/8)`로 모든 sparse 조건에서 동일하다.
- 모델 seed234 고정. upstream compressed token 순서의 위치 embedding과 window/mask/padding 처리를 유지한다. 새로운 시간 embedding을 넣지 않는다.
- 캐시·원본 summary·manifest·checkpoint·소스 hash와 환경을 기록한다. frozen source 검증을 우회하지 않는다. 과거 평가 소스 차이가 허용되는 cache라도 이번 코드로 Full을 재평가하여 비교 기준을 만든다.

본 실험은 **cached feature 성능 탐색**이다. raw 영상 decoding·CLIP encoding을 실행하지 않으며 raw end-to-end speedup을 주장하지 않는다.

## 3. 사전 고정한 선택 규칙

모든 숫자는 별도 표시가 없으면 **reference feature row 좌표**다. 원본 프레임 좌표는 row×16이다.

### 3.1 Uniform — `uniform_ver3`

`U[i] = 8*i`, `i=0,...,K-1`. 기존 Uniform12.5%와 같은 `0,128,256,...` 원본 좌표다.

### 3.2 규칙적 짝 배치 — `paired_ver3`

Uniform 슬롯을 시간순으로 8개씩 겹치지 않게 묶는다. 완전한 묶음에서 zero-based 슬롯3·4만 변경한다. 나머지 슬롯과 불완전한 마지막 묶음은 그대로 둔다.

묶음 시작 row가 B이면:

```text
Uniform rows : B+[0,8,16,24,32,40,48,56]
Paired rows  : B+[0,8,16,27,28,40,48,56]
```

즉 원본 첫 묶음은 다음과 같다.

```text
Uniform frames: 0,128,256,384,512,640,768,896
Paired frames : 0,128,256,432,448,640,768,896
```

짝 간격은 원본16프레임(XD24fps 기준 약0.67초)이다. 짝 주변 간격은176·192프레임으로 커진다. 전체 영상의 첫·마지막 선택 좌표와 K는 유지한다. K<8인 영상은 변경하지 않으며, 실제 변경 묶음 수와 슬롯 수를 영상별 진단에 기록한다.

### 3.3 무작위 분산 재배치 — `relocated_ver3`

동일한 완전한 묶음의 슬롯3·4를 같은 구간 내부로 옮긴다. 원래 두 좌표를 L=B+24, R=B+32라 하면, 정수쌍 `(a,b)` 중 `L<a<b<R`, `b-a>=2`인 **15개 쌍**에서 균등하게 하나를 선택한다. 두 프레임 모두 이동하고 나머지 슬롯은 유지한다.

이 대조군은 같은 변경 슬롯·지원 구간에서 adjacent pair 대신 분산된 두 위치를 선택한다. 중심점·양옆 간격이 paired와 완전히 같지는 않으므로 순수한 인접성 효과만 분리하는 대조군은 아니다.

seed는 **0,1,2,3,4**로 고정한다. 영상별 seed는 기존 `video_seed(seed,'xd',filename)`를 사용하고, Python `random.Random`으로 각 묶음을 시간순으로 처리한다. 버전을 실행 환경에 기록한다. GT를 보거나 좋은 seed만 골라 보고하지 않는다.

## 4. 복원 좌표와 평가

모든 조건에서 **실제로 선택한 원본 프레임 좌표를 score anchor로 사용**한다. 기존 Random-window의 구간 시작점 anchor와 구분한다. 재배치 후에도 예전 슬롯 시작점에 점수를 붙이지 않는다.

선택 token의 binary·class 점수를 reference grid에 nearest 복원하고, 동률은 앞 관측으로 처리한다. 그 후 고정16프레임 반복으로 공식 annotation 시간축 전체를 평가한다. 이 규칙은 모든 조건에 동일하다.

### 평가 항목

1. 주 지표: Branch2 AP 및 Uniform 대비 pp 차이. 기존 공통 evaluator의 ROC-AUC·Ano-AUC·temporal mAP도 각 run summary에 남는다.
2. Full의 Branch2 임계값0.5 재현율을 목표로 한 FPR. 선형보간값과 처음 도달 가능한 실제 ROC 점의 FPR/재현율을 함께 기록한다. 운영 임계값 선택에 사용하지 않는다.
3. 전체 사건 및 1초 이하 사건 포착 건수. 실제 선택 프레임이 `[start,end)`에 한 장 이상 있으면 포착으로 정의한다. 공식 평가 길이에 clip한 interval을 사용하고 빈 구간은 제외한다. FPS는 cache 원본 summary에서 읽는다.
4. Uniform과 공통인 선택 프레임에서 두 branch의 직접 점수 변화. 정상·이상 GT별 평균·중앙값·상승 비율을 기록한다. 전체 시간축 복원 점수와 혼동하지 않는다. 반복 간 요약을 합쳐 프레임 독립 표본의 유의성 검정을 하지 않는다.

paired와 relocated는 같은 슬롯을 변경하고 원래 위치를 재사용하지 않으므로 Uniform과의 공통 선택 위치가 동일하다. 원래 위치가 유지된 프레임도 인접한 시퀀스가 달라질 수 있다. 영상별 K가 같아도 실제 선택 비율은 반올림으로12.5%와 조금 다를 수 있다.

## 5. 실행 구성

총8회다: cached Full100% 1회 + Uniform12.5% 1회 + paired12.5% 1회 + relocated12.5% 5seed. Full은 평가 기준이며 sparse 세 조건의 동일 K 비교에 포함하지 않는다.

새 파일:

- [선택기](vadfs/samplers/context_ver3.py): 좌표·seed만으로 선택.
- [순차 실행기](scripts/run_context_ver3.py): 8회 실행·중간 sweep 저장·full 완료 후 자동 비교.
- [분석기](scripts/analyze/compare_context_ver3.py): provenance·좌표·K·AP 재현을 확인한 후 비교 CSV 생성.
- [사용자 테스트](tests/test_context_ver3.py): 짧은 영상·ceil budget·끝 좌표·tail·seed 재현·원본 feature 행 보존·raw 경로 거부.
- `configs/experiments/xd_{reference,uniform,paired,relocated}_ver3.json`.

기존 공통 `vadfs/pipeline.py`와 `vadfs/clip_selection.py`에는 cached ver3 선택을 연결하는 분기만 추가했다. 기존 ver2나 모델·평가·timeline 함수는 변경하지 않는다. 공통 소스 hash는 달라지므로 새 run은 새 provenance를 기록한다.

### 5.1 사용자 테스트

```bash
cd /workspace/hyeongjun/vad
/workspace/sangmin/.venv/bin/python -m unittest discover -s tests -p 'test_context_ver3.py' -v
```

위 Python은 서버에 있는 기존 환경의 예다. 이 문서 작성 중 해당 환경의 import·GPU 사용 가능성은 검증하지 않았다. 사용자가 준비한 다른 Python 환경으로 교체할 수 있다.

### 5.2 Smoke — metric 확인용이 아닌 연결 검증

```bash
cd /workspace/hyeongjun/vad
CUDA_VISIBLE_DEVICES=0 /workspace/sangmin/.venv/bin/python scripts/run_context_ver3.py \
  --reference-run /workspace/vad/results/baselines/xd_raw_reference \
  --local configs/local/hj.xd.json --device cuda --mode smoke
```

기존 공통 smoke 규칙에 따라 대표 영상만 처리한다. 자동 성능 집계는 하지 않는다. 모든 run의 `success=true`, 좌표·선택 개수·진단을 먼저 확인한다. smoke 완료를 800영상 평가 완료로 해석하지 않는다.

### 5.3 Full — 사용자 확인 후 실행

```bash
cd /workspace/hyeongjun/vad
CUDA_VISIBLE_DEVICES=0 /workspace/sangmin/.venv/bin/python scripts/run_context_ver3.py \
  --reference-run /workspace/vad/results/baselines/xd_raw_reference \
  --local configs/local/hj.xd.json --device cuda --mode full
```

한 GPU에서 순차 실행한다. GPU 가용성 확인·예약·기존 프로세스 중단은 자동으로 하지 않는다. 새 출력 디렉터리를 생성하며 기존 결과를 덮어쓰지 않는다. 자동 resume은 없고 실패한 sweep의 성공 결과는 보존한다. 분석만 다시 하려면 아래 명령에 성공8회의 summary 경로를 모두 지정한다.

```bash
/workspace/sangmin/.venv/bin/python scripts/analyze/compare_context_ver3.py \
  <FULL_SUMMARY> <UNIFORM_SUMMARY> <PAIRED_SUMMARY> \
  <RELOCATED_SEED0_SUMMARY> <RELOCATED_SEED1_SUMMARY> <RELOCATED_SEED2_SUMMARY> \
  <RELOCATED_SEED3_SUMMARY> <RELOCATED_SEED4_SUMMARY> \
  --output outputs/analyses/xd_context_manual_ver3
```

## 6. 출력과 사용자 확인 항목

- 개별 run: `outputs/runs/xd_cached_raw_features_*/summary.json`, `scores/*.npz`, `frame_scores.npz`.
- sweep/비교: `outputs/analyses/xd_context_ver3_{smoke,full}_<timestamp>/`.
- `sweep_ver3.json`: 완료 여부·오류·원본 summary hash·각 실행 경로.
- `runs_ver3.csv`: 실행별 AP·FPR·사건 포착·총 입력 수·Uniform 대비 차이.
- `comparison_ver3.csv`: 방법별 평균·표본 표준편차. 단일 실행의 표준편차는 빈칸이다.
- `common_frame_changes_ver3.csv`: 공통 선택 프레임의 branch별 정상·이상 점수 변화.
- `analysis_ver3.json`: 비교 입력 hash·분석기 hash·집계와 해석 범위.

사용자는 8회 모두 성공/800영상인지, sparse 조건의 총 입력이 각각18,549장인지, Full이145,706장인지 확인한다. 이 수는 기존 XD Reference에 대한 기대값이며 실행 결과를 대신하지 않는다. 분석기는 영상별 좌표·K를 선택 규칙으로 재검산하고 저장된 전체 Branch2 점수·summary AP를 재현하지 못하면 실패한다.

새 Uniform과 과거 AP78.8278%, 새 Full과 과거 AP84.5041%의 차이도 확인한다. 같다고 미리 보장하지 않는다. 차이가 생기면 cache·모델·환경·평가 조건을 조사한 뒤 새 결과를 해석한다. 성능이 낮은 seed와 변경 불가능한 짧은 영상도 제외하지 않는다.

## 7. 결과 판단 원칙

- paired가 Uniform과 relocated 평균보다 개선되면 시간적 배치의 가능성을 보여주는 탐색 신호다. 차이의 크기와 random 변동을 함께 보고하며 단일 실행에서 유의성을 주장하지 않는다.
- 사건 포착이 낮아져도 AP/FPR가 개선되면 coverage와 모델 판별 품질의 차이를 별도로 보고한다.
- paired가 개선되지 않으면 이 고정 배치 규칙이 도움이 되지 않은 것으로 결론낸다. 주변 맥락 자체가 불필요하다고 일반화하지 않는다.
- test 결과에 맞춰 짝 수·간격을 바꾸면 새 탐색 실험으로 구분한다. 최종 방법은 별도 개발 데이터에서 설정을 고정한 뒤 평가한다.
- 효율 증거는 후속 raw 실험에서 decoding·선택·CLIP·VAD·복원 전체 시간을 같은 조건으로 측정해야 한다.

## 8. 현재 검증 상태

작성 중 기존 코드·설정·입출력 계약을 읽어 정적으로 검토했다. 신규4개 및 수정2개 Python의 AST 구문 검사, JSON4개 파싱, tracked diff의 whitespace 검사를 통과했다. 작성 중 Codex는 테스트, dry-run, 프로젝트 import, 데이터 처리, 분석기, 모델 추론을 실행하지 않았다. 이후 사용자가 실행한 full8회와 자동 비교의 성공 기록을 확인했다. 실제 성능과 한계는 [evaluation_ver3.md](evaluation_ver3.md)에 기록했다. 테스트·smoke의 별도 성공 여부는 이 full 완료 확인만으로 추정하지 않는다.
