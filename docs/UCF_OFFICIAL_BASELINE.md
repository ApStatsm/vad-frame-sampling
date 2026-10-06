# UCF-Crime official-feature baseline 실행 안내

2026-10-06. Codex는 코드 작성·정적 검토만 수행했다. 사용자 MPS smoke 7사례가 통과했으며 전체 평가는 아직 미실행이다.

**GitHub 통합 후 다음 실행:** 기존 MPS smoke는 당시 source hash의 검증 결과다. 공유 준비에서 원본 파일의 변경 표시와 evaluator의 연구/upstream 버전 기록을 수정했으므로 현재 버전의 `--mode smoke --device mps`를 다시 통과한 뒤 `--mode full --device mps`를 실행한다. 모델 계산·가중치·window·prompt·score 복원 규칙은 이 준비에서 바꾸지 않았다. 기존 raw 보고서는 보존하고 공유 요약은 `results/`에 둔다.

## 확인된 준비 결과

사용자가 실행한 `outputs/setup/ucf_baseline_input_check.json`의 결과:

- Feature 290개 모두 `[T,512]` 형식이며 finite. `__0`/`__5` shape 일치.
- Feature dtype는 286개 float16, 4개 float32. 모델은 visual tokens를 float32로 처리한다.
- 기존 GT 길이 1,109,888과 feature score 16배 반복 길이 일치.
- 공식 `[start:end]` slicing 규칙과 CSV 순서를 기준으로 영상별 GT를 대조한 결과 mismatch 0.
- UCF checkpoint CPU strict loading 성공(349 state entries, 14 classes).
- CUDA unavailable. 사용자 MPS smoke 보고서(`ucf_official_smoke_mps_20261006_202116_841716`)에서 7사례의 forward, finite score, 복원 길이 검사 통과를 확인했다.

이 확인은 official-feature timeline의 검증이다. Raw-video annotation base/끝점·tail의 처리 규칙은 raw reference 단계에서 별도 확정한다.

## 이번 실행에 필요한 입력

- `configs/ucf_resources.json`에 연결된 feature 폴더와 UCF/CLIP checkpoint.
- `outputs/setup/ucf_CLIP_rgbtest.local.csv`의 공식 test 290개 순서.
- `outputs/setup/ucf_resource_check.json`과 `ucf_baseline_input_check.json`의 통과 기록.
- 기존 `VadCLIP/list/gt_ucf.npy`와 test manifest.

## Step A — CPU 소규모 추론 점검

프로젝트 루트에서 사용자가 실행한다.

```sh
.venv/bin/python scripts/evaluate_ucf_official.py --mode smoke --device cpu
```

7개 실행 사례를 확인한다. 실제 anomaly 영상, 가장 짧은 normal 영상, 긴 영상과 그 긴 영상의 앞 255/256/257/512 feature token 구간이다. 경계 prefix는 window/padding 동작 점검을 위한 것이며 benchmark 성능으로 보고하지 않는다.

확인 항목:

- 두 branch의 유효 token score가 finite이고 범위가 [0,1]인지.
- Score 수가 실제 입력 token 수와 같고, 16배 복원 score와 대응 GT 구간 길이가 같은지.
- 256/512 경계에서 공식 split이 추가 padded window를 만드는 경우의 shape와 declared/physical length.

정수배의 추가 padded window와 temporal에 padding mask를 전달하지 않는 upstream 동작은 재현 단계에서 유지한다. Sampling pipeline의 공통 처리 정책은 이후 별도로 확정한다.

CPU forward를 가능하게 하기 위해 `DistanceAdj`의 CUDA 고정 위치만 모델 장치에 맞추었다. 거리·상수·exp 식과 가중치는 유지한다. `CLIPVAD`의 선택적 local CLIP path 인수도 사용한다. Source hash와 연구 Git commit/dirty 상태, `VadCLIP/UPSTREAM.json`의 원본 commit을 구분하여 실행 보고서에 남긴다.

## Step B — 전체 test 평가

동일 코드·입력·device의 smoke가 통과한 뒤에만 실행한다. 스크립트도 통과한 smoke 보고서와 현재 source/config/checkpoint 식별값을 대조한다.

```sh
.venv/bin/python scripts/evaluate_ucf_official.py --mode full --device mps
```

Normal 150개를 포함한 공식 test 290개 전체의 frame-level ROC-AUC와 AP를 두 branch별로 기록한다. UCF의 주 지표는 AUC이며 AP는 보조 출력이다. Detection mAP는 이번 실행 범위에 포함하지 않는다.

재현 규칙:

- 공식 `__5.npy` feature와 원본 CSV 순서 유지.
- 모델 옵션은 bundled `ucf_option.py`의 기본값. 256 token window, attention window 8, prompt prefix/postfix 10, classes 14.
- Prompt는 원본 순서: Normal, Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism.
- Branch 1: sigmoid(logits1), branch 2: 1 - softmax(logits2)[:,0].
- Branch 결합·smoothing 없이 각각 기록. 공식 feature score를 16회 반복하여 기존 GT 전체에 평가.
- 모델 가중치·전처리·window 정책을 sampler별로 변경하지 않는다. Sampling 비교의 주 branch는 이후 논문/공식 재현 조건 확인 후 sampling 결과를 보기 전에 고정한다.

기존 사용자가 통과한 smoke 장치는 MPS이므로 위 full 명령도 MPS를 사용한다. CPU full 평가를 선택하려면 Step A의 CPU smoke부터 별도로 통과해야 한다. 실패 시 CPU로 자동 fallback하지 않는다. MPS smoke 명령은 아래와 같다. 현재 GitHub 통합 버전은 source hash가 변경되어 새 smoke가 필요하며, 이후 코드·입력·장치가 동일한 경우에만 해당 통과 결과를 재사용한다.

```sh
.venv/bin/python scripts/evaluate_ucf_official.py --mode smoke --device mps
```

MPS를 선택한 평가 결과는 device·precision과 함께 별도 run으로 기록한다. 소규모 추론 성공을 전체 평가 성공이나 speedup으로 확장하지 않는다.

## 예상 출력과 사용자 확인

- 실행별 `outputs/baseline/ucf_official_<mode>_<device>_<timestamp>/summary.json`.
- 같은 run의 `scores/*.npz`: 영상/사례별 두 branch의 feature token score와 GT 구간. 원본 frame 위치는 official token당 16개 반복과 summary의 gt_offset으로 복원한다.
- Smoke의 최신 요약 복사본: `outputs/setup/ucf_smoke_check.json`.
- Full 성공 시 `frame_scores.npz`와 summary의 `metrics`.
- 실패 시 `summary.json`의 error, 같은 run의 `error.txt`에 traceback. 입력·GT·checkpoint는 수정하지 않는다.

Smoke 완료 후 `success: true`와 7개 사례 OK를 확인한다. Full 성공은 전체 frame 수 1,109,888, test 영상 290개와 두 branch의 metric 저장으로 확인한다.

시간 출력은 official feature cache에서의 실행 기록이다. 영상 decoding·CLIP visual encoding을 포함하지 않으므로 raw-video 효율이나 sampling speedup 근거로 사용하지 않는다. 이 단계의 warm-up/반복 측정은 성능 benchmark 목적이 아니며 원본 입력의 inference 정상 동작과 AUC 재현을 확인한다.
