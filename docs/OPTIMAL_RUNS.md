# 적합 없는 CLIP-feature D·A·E 실행 안내

2026-11-12. `method.md`의 고정 투영 방식으로 코드를 수정했다.
**AST/설정/호출 경로의 정적 검토만 수행했으며 테스트·모델 추론·성능 재현은 미실행이다.**
개발 영상·PCA artifact·표준화 적합·영상 재디코딩 없이 저장된 raw Reference CLIP feature를 사용한다.

## 준비된 입력

| 입력 | UCF | XD |
|---|---|---|
| local JSON | `configs/local/hj.ucf.json` | `configs/local/hj.xd.json` |
| raw Reference cache | `/workspace/vad/results/baselines/ucf_raw_reference` | `/workspace/vad/results/baselines/xd_raw_reference` |
| 주 지표 | Branch1 ROC-AUC | Branch2 AP |

위 cache 폴더의 `features/*.npz`가 각각 290개·800개 있고 현재 계정에서 읽을 수 있음을 파일 목록과
권한으로 확인했다. summary/manifest 및 두 VadCLIP checkpoint와 CLIP checkpoint도 현재 읽을 수 있다.
NPZ 내용·수치 동등성은 이번 작업에서 검사하지 않았으며 사용자 실행 시 검증한다.
정적 source hash 대조에서 기존 Reference의 evaluation.py/metrics.py는 현재 코드와 달랐다.
모델·시간축 복원·색 변환 소스, vendored 모델, GT·평가 입력·추출 설정은 일치한다.
기존 cache는 특징 입력으로만 재사용하며, 과거 평가 코드가 다르면 그 summary metric 대비 Δ는 생성하지 않는다.
기본 sweep에서 cached Reference 100%도 현재 평가 코드로 다시 계산해 비교표의 기준으로 쓴다.
모델·GT·복원·추출 조건이 다른 cache는 계속 거부한다.
기존 Reference 폴더의 manifest.json을 기본으로 사용하므로 별도의 prepare 실행은 필요하지 않다.
local JSON에 setup_manifest가 있으면 그 명시적 설정을 우선한다.

새 설정은 `configs/experiments/{ucf,xd}_clip_*.json`이다. 이전
`{dataset}_{d,a,e}_optimal_*.json`은 raw 저비용 특징/PCA 경로로 남아 있으므로 구분한다.
이전 PCA 적합 도구·sampling_artifact local 키는 이번 경로에서 사용하지 않는다.

## Python 환경과 사용자 검증

이 checkout의 소스를 직접 실행한다. 아래 Python 실행 파일은 기존 baseline에 사용한 공유 환경이다.
현재 존재·읽기·실행 권한은 확인했으며 이번 작업에서 import나 연산 검증은 하지 않았다.
기존 환경의 패키지를 변경하지 않는다. 개인 검증 환경이 있다면 VAD_PY만 그 경로로 바꾼다.

```bash
cd /workspace/hyeongjun/vad
export VAD_PY=/workspace/sangmin/.venv/bin/python
"$VAD_PY" -m unittest discover -s tests -v
```

테스트는 D/A 직접 목적값 gain, E 전수교환, 고정 투영 재현·직교성·QR 부호·K<8,
원본 dtype/값 유지, 100% 실제 K=N 우회, NaN/0 norm 거부, rank 부족 시 차원 유지,
cache FPS/좌표·dtype 검사, 설정의 raw 우회 거부 및 mocked pipeline을 포함한다.
테스트 통과는 실제 데이터 전체 성능 재현을 대신하지 않는다.

먼저 일부 영상에서 D/A/E의 100%·25% smoke를 확인한다.

```bash
CUDA_VISIBLE_DEVICES=0 "$VAD_PY" -u scripts/run_optimal.py \
  --dataset ucf --local configs/local/hj.ucf.json \
  --reference-run /workspace/vad/results/baselines/ucf_raw_reference \
  --methods d a e --budgets 100 25 --device cuda --mode smoke

CUDA_VISIBLE_DEVICES=1 "$VAD_PY" -u scripts/run_optimal.py \
  --dataset xd --local configs/local/hj.xd.json \
  --reference-run /workspace/vad/results/baselines/xd_raw_reference \
  --methods d a e --budgets 100 25 --device cuda --mode smoke
```

smoke는 소수 영상만 처리하며 전체 metric/CSV는 생성하지 않는다. summary의 success와
K, 원본 feature dtype, 전체 평가 길이, projection hash, 100% bypass, rank/차원/목적값을 확인한다.
특정 영상은 공통 진입점의 `--video 정확한_파일명`으로 지정할 수 있다.

## tmux에서 UCF와 XD 실행

아래 명령은 Windows PowerShell에서 SSH로 접속한 **서버 bash**에서 실행한다.
먼저 사용자 명령 `nvidia-smi`로 GPU 0과 1이 사용 가능한지 확인한다.
CUDA_VISIBLE_DEVICES는 GPU 하나만 노출하고 --device cuda는 그 GPU로 모델을 계산하게 한다.
서버 GPU 1을 지정해도 프로세스 안에서 cuda:0으로 표시될 수 있다.

UCF:

```bash
tmux new -s optimal-ucf
cd /workspace/hyeongjun/vad
export VAD_PY=/workspace/sangmin/.venv/bin/python
CUDA_VISIBLE_DEVICES=0 "$VAD_PY" -u scripts/run_optimal.py \
  --dataset ucf --local configs/local/hj.ucf.json \
  --reference-run /workspace/vad/results/baselines/ucf_raw_reference \
  --device cuda --mode full
```

Ctrl-b를 누르고 손을 뗀 뒤 d로 세션에서 나온다. XD:

```bash
tmux new -s optimal-xd
cd /workspace/hyeongjun/vad
export VAD_PY=/workspace/sangmin/.venv/bin/python
CUDA_VISIBLE_DEVICES=1 "$VAD_PY" -u scripts/run_optimal.py \
  --dataset xd --local configs/local/hj.xd.json \
  --reference-run /workspace/vad/results/baselines/xd_raw_reference \
  --device cuda --mode full
```

다시 확인하려면:

```bash
tmux ls
tmux attach -t optimal-ucf
# 또는
tmux attach -t optimal-xd
```

tmux는 SSH 단절 후에도 작업을 유지한다. 서버 종료/재부팅이나 추론 오류를 자동 복구하지 않는다.
이번 경로는 PCA artifact를 요구하지 않는다.

## 기본 실행 조합과 결과

기본 전체 실행은 데이터셋당 **37회**다. 각 GPU에서 하나씩 순차 실행한다.

| 방법 | Budget | 반복 |
|---|---|---|
| cached Reference | 100% | 1 |
| Uniform | 100/50/25/12.5% | 각 1 |
| Random-feature | 100/50/25/12.5% | 각 seed 0~4 |
| D/A/E | 각각 100/50/25/12.5% | 각 1 |

--methods로 일부 방법만 지정할 수 있다. --budgets 25로 제한해도 Reference를 요청했다면
Reference는 100%로 실행한다. 예를 들어 --methods reference uniform random d a e --budgets 25는
Reference 1회 및 25% 비교군 9회다.

- 개별 run: `outputs/runs/<run_id>/summary.json`, `scores/*.npz`, full의 `frame_scores.npz`.
- 고정 Q: 낮은 budget의 optimal run의 `projection.npz`; summary에 Q 바이트/hash·shape·생성 규칙 기록.
- sweep: `outputs/analyses/<dataset>_clip_optimal_<mode>_<시각>/sweep.json`.
- full 종료 후 자동 성능표: `outputs/analyses/<dataset>_sampling_comparison_<시각>/comparison.csv`와 summary.json.
  실제 CSV 경로는 터미널 `Comparison:`과 sweep의 comparison_csv에 기록한다.

비교표에는 seed 평균/표본 표준편차, ROC-AUC/AP/Ano-AUC, IoU별 mAP/평균 mAP,
입력 수, Reference 대비 drop, 같은 budget Uniform 대비 Δ가 들어간다.
일부 방법만 실행했으면 없는 Reference/Uniform 대비 차이는 null이다.
Random-feature는 같은 Reference 후보의 행을 비복원 추출한 것이며 기존 Random-window 실험과 다르다.
과거 source metric을 기준으로 삼지 않도록 기본 Reference 행을 포함해 실행한다.

기존 Random-window를 함께 보고 싶으면 완료 sweep과 해당 dataset의 기존 seed aggregate를
compare_sampling.py에 입력한다. 과거 source가 다르면 --allow-source-differences로 검토 사실을
명시해야 하며, 모델/GT/평가/전체 시간축/영상별 K가 다르면 계속 거부한다.
새 cache의 byte hash나 optimal 표현·Q hash가 다르면 같은 비교에 묵시적으로 섞지 않는다.

## 개별 조합 재실행

wrapper에는 자동 resume가 없다. 실패하면 지금까지 완료된 run 경로를 sweep.json에 남긴다.
미완료 조합만 --methods/--budgets로 좁히거나 공통 진입점을 사용한다.

```bash
CUDA_VISIBLE_DEVICES=0 "$VAD_PY" -u scripts/run_experiment.py \
  --config configs/experiments/ucf_clip_d_optimal_25.json \
  --local configs/local/hj.ucf.json \
  --cached-reference-run /workspace/vad/results/baselines/ucf_raw_reference \
  --device cuda --mode full
```

Random-feature 개별 실행은 ucf_clip_random_25.json 및 --seed 0 등으로 지정한다.
wrapper의 --reference-run은 내부적으로 공통 진입점의 **--cached-reference-run**으로 전달된다.
따라서 cache를 읽어 성능만 평가하고 raw 영상이나 CLIP image encoder를 실행하지 않는다.

## 결과 해석

선택용으로만 CLIP row L2 정규화와 PCG64(seed 234) 고정 직교 투영을 적용한다.
d=min(8,K)이며 학습·PCA·영상별 SVD는 하지 않는다. VadCLIP 입력은 원본 512D 행이다.
전체 CLIP feature를 미리 사용하므로 이번 결과는 feature 기반 성능 탐색이며
CLIP 계산 절감 또는 raw end-to-end speedup을 주장하지 않는다.
summary의 experiment_scope는 cached_feature_exploration, speedup_eligible은 false다.
cache 읽기/투영/최적화/VAD/복원 시간은 별도로 남기지만 raw 시간과 비교하지 않는다.
