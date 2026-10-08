# UCF-Crime official-feature baseline 실행 안내

2026-10-07. Codex는 코드 작성·정적 검토 및 기존 결과 읽기만 수행했다. 사용자 MPS full run `ucf_full_20261007_010204`가 성공했으며 branch1 AUC 88.02%/Ano-AUC 70.23%, branch2 AUC 85.69%, 원본 fine-grained AVG 6.68%다. [실험 결과표](EXPERIMENT_RESULTS.md)에 official-feature 재현과 미측정 raw sampling 표를 구분했다. 아래 명령은 실행 안내이며 동일 조건의 성공 run을 반복할 필요는 없다.

**실행 선택:** 사용자 지시에 따라 smoke는 선택적 소규모 점검이며 `--mode full --device mps`를 바로 실행할 수 있다. Full에서는 과거 smoke 보고서를 읽거나 현재 코드 hash와 비교하지 않는다. 자원·입력 audit와 checkpoint 점검, 실행별 source hash 기록은 유지한다. 기존 작은 JSON/CSV/텍스트 보고서는 `outputs/`의 원래 위치에서 공유하고 score 배열은 업로드에서 제외한다. 구조·형식은 루트 README에 정리했다.

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

## 선택 사항 — CPU 소규모 추론 점검

소규모 추론을 확인하고 싶을 때 프로젝트 루트에서 사용자가 실행한다. 전체 평가의 선행 조건은 아니다.

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

## 전체 test 평가

Smoke 실행 여부와 관계없이 바로 실행할 수 있다. 기존 입력 audit와 config/list/manifest/GT 일치, checkpoint hash, 장치 availability는 확인한다. 실행 중 feature shape/dtype·score 유효성·GT 길이 검사도 유지한다.

```sh
.venv/bin/python scripts/evaluate_ucf_official.py --mode full --device mps
```

Normal 150개를 포함한 공식 test 290개 전체의 frame-level ROC-AUC와 AP를 두 branch별로 기록한다. 추가로 manifest에서 is_anomaly=true인 140개 영상의 전체 프레임을 모아 Ano-AUC도 기록한다. 이상 영상 내부 정상 구간은 포함하며 normal-only 150개 영상은 제외한다. UCF의 주 지표는 전체 AUC이며 AP/Ano-AUC는 보조 출력이다. Full은 A-branch의 fine-grained temporal mAP도 아래 scope로 계산한다.

재현 규칙:

- 공식 `__5.npy` feature와 원본 CSV 순서 유지.
- 모델 옵션은 bundled `ucf_option.py`의 기본값. 256 token window, attention window 8, prompt prefix/postfix 10, classes 14.
- Prompt는 원본 순서: Normal, Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism.
- Branch 1: sigmoid(logits1), branch 2: 1 - softmax(logits2)[:,0].
- Branch 결합·smoothing 없이 각각 기록. 공식 feature score를 16회 반복하여 기존 GT 전체에 평가.
- 모델 가중치·전처리·window 정책을 sampler별로 변경하지 않는다. Sampling 비교의 주 branch는 이후 논문/공식 재현 조건 확인 후 sampling 결과를 보기 전에 고정한다.

위 full 명령은 MPS를 사용하며 CPU/CUDA에서는 `--device cpu` 또는 `--device cuda`를 지정한다. 실패 시 CPU로 자동 fallback하지 않는다. MPS에서 선택적으로 소규모 추론을 확인하려면 아래 명령을 사용한다. 과거 smoke의 검증 범위는 당시 코드·입력·장치에 한정되지만 그 보고서의 상태로 full 실행을 제한하지 않는다.

```sh
.venv/bin/python scripts/evaluate_ucf_official.py --mode smoke --device mps
```

MPS를 선택한 평가 결과는 device·precision과 함께 별도 run으로 기록한다. 소규모 추론 성공을 전체 평가 성공이나 speedup으로 확장하지 않는다.

## 예상 출력과 사용자 확인

- 실행별 `outputs/baseline/ucf_<mode>_<YYYYMMDD_HHMMSS>/summary.json`. 예: `ucf_full_20261006_234239`. 날짜·시간은 Asia/Seoul 기준이며 같은 이름이 이미 있으면 `_2`, `_3` 등을 붙인다. 장치와 입력 종류는 summary에 기록한다.
- 같은 run의 `scores/*.npz`: 영상/사례별 두 branch의 feature token score와 GT 구간. 원본 frame 위치는 official token당 16개 반복과 summary의 gt_offset으로 복원한다.
- 새 NPZ에는 `branch2_class_feature_scores`(`[T,14]` softmax probability)와 `class_labels`(prompt 순서의 문자열 14개)도 저장한다. `allow_pickle=False`로 읽을 수 있다. 과거 scalar score 파일에서 클래스별 점수를 복원할 수는 없다.
- Smoke의 최신 요약 복사본: `outputs/setup/ucf_smoke_check.json`.
- Full 성공 시 `frame_scores.npz`와 summary의 `metrics`.
- 각 branch의 metric key는 `roc_auc`, `ap`, `ano_auc`이며 값은 0~1이다. %로 표시하려면 100을 곱한다. `anomaly_videos`=140, `anomaly_evaluation_frames` 및 `ano_auc_scope`에 Ano-AUC 집계 범위를 기록한다.
- `metrics.fine_grained.map_by_iou`는 `0.1`~`0.5`의 다섯 temporal mAP, `mean_map`은 다섯 값의 평균이다. 모두 0~1이며 % 표시는 100을 곱한다. Fine-grained 점수는 branch2에서만 계산한다.
- 실패 시 짧은 오류 설명은 `summary.json`의 error에 기록하고 상세 traceback은 터미널에 출력한다. `error.txt`는 생성하지 않는다. 입력·GT·checkpoint는 수정하지 않는다.

Smoke 완료 후 `success: true`와 7개 사례 OK를 확인한다. Full 성공은 전체 frame 수 1,109,888, test 영상 290개와 두 branch의 metric 저장으로 확인한다.

시간 출력은 official feature cache에서의 실행 기록이다. 영상 decoding·CLIP visual encoding을 포함하지 않으므로 raw-video 효율이나 sampling speedup 근거로 사용하지 않는다. 이 단계의 warm-up/반복 측정은 성능 benchmark 목적이 아니며 원본 입력의 inference 정상 동작과 AUC 재현을 확인한다.

## 기존 결과에 Ano-AUC 추가 (기록)

`ucf_full_20261007_010204`의 Ano-AUC(branch1 70.23%, branch2 65.52%)는 저장 score로 summary에 덧붙인 일회성 스크립트 `update_ucf_metrics.py`로 계산했다. 현재 full evaluator가 Ano-AUC를 직접 계산하므로 이 스크립트는 2026-10-08 정리에서 제거했다. 원본은 `git show ucf-raw-reference-v1:scripts/update_ucf_metrics.py`로 확인한다.

## Fine-grained mAP의 평가 범위

Full 명령은 모델 inference부터 다시 수행하여 두 branch의 AUC/AP/Ano-AUC와 `metrics.fine_grained`를 새 run에 함께 저장한다. 별도 저장-score mAP 갱신 단계는 사용하지 않는다.

```sh
.venv/bin/python scripts/evaluate_ucf_official.py --mode full --device mps
```

- 평가 대상: 정상 영상을 포함한 공식 test 290개, Normal+13개 사건의 14개 클래스. 원본 `VadCLIP/src/ucf_test.py`의 `excludeNormal=False` 호출을 따른다.
- 점수: 이번 inference의 A-branch softmax `[T,14]`를 token당 16 frame으로 복원한다. Class 순서·shape·finite/확률 범위·정규화와 영상 순서/full scope를 검증한다.
- GT: 원본 `VadCLIP/list/gt_segment_ucf.npy`, `gt_label_ucf.npy`를 사용한다. 290개 순서·구간·category를 manifest 및 feature timeline과 대조한다. 원본 label 'A' 등의 표현을 remap하거나 endpoint를 이동·clipping하지 않는다. `unmapped_gt_labels_preserved`에 미대응 label을 남긴다.
- 계산: 수정하지 않은 `VadCLIP/src/utils/ucf_detectionMAP.py`를 직접 호출한다. 구간 threshold, video/segment confidence, NMS, AP, 14개 클래스 평균과 예측 없는 클래스에서의 전체 mAP 조기 0 반환까지 원본 동작을 유지한다.
- 출력: `protocol=original_ucf_test_excludeNormal_false`, `evaluation_videos=290`, `evaluation_classes` 14개, `map_by_iou`의 다섯 mAP와 `mean_map`. 원본 helper의 % 값은 100으로 나누어 0~1로 저장한다. Branch1에는 fine-grained mAP를 부여하지 않는다.
- 출처: summary의 `input_sha256`에 `official_gt_segments`, `official_gt_labels`, `official_fine_grained_source`, `fine_grained_source` hash를 기록한다. 원본 GT/source는 변경하지 않는다.

새 실행의 입력은 기존 config·official features·CLIP/UCF checkpoints·frame GT·manifest 및 위 segment/class GT다. 출력은 `outputs/baseline/ucf_full_<YYYYMMDD_HHMMSS>/summary.json`, `scores/*.npz`, `frame_scores.npz`다. `success=true`, `test_videos=290`, `evaluation_frames=1109888`, 두 branch의 AUC/AP/Ano-AUC와 fine-grained의 protocol·영상/class 범위·다섯 mAP/AVG를 확인한다. Smoke는 선택 사항이다. Cached-feature loop 시간은 최종 mAP 계산을 포함하지 않으며 raw end-to-end 효율이 아니다.

## 이전 13개 클래스 결과와 구분

사용자 run `ucf_full_20261007_004524`의 anomaly-only AVG=7.46%는 140개/13개 클래스 방식으로 계산된 기존 기록이다. AUC=88.02%/Ano-AUC=70.23%(branch1)는 확인했지만, 이 mAP를 논문의 AVG 6.68%와 직접 비교하여 모델 성능 개선으로 해석하지 않는다.

사용자 요청으로 기본 fine-grained를 원본 공개 코드 방식으로 수정하고, 앞서 준비했던 `--fine-grained`/`--official-fine-grained` 저장-score 갱신 옵션과 별도 13개 클래스 구현을 제거했다. 기존 summary/NPZ는 수정하지 않았고 새 모델 실행에서 새 run을 생성한다.

새 사용자 `ucf_full_20261007_010204`의 원본 방식 mAP 11.72/7.83/6.40/4.53/2.93%, AVG 6.68%를 확인했다. 논문 표와 소수점 둘째 자리까지 일치하며 기존 coarse metric도 같다. Codex는 기존 JSON만 읽었고 모델·GT 배열 로딩·metric 계산은 실행하지 않았다. 다음 raw-video 좌표/추출 검증과 CLIP 시간 측정은 [raw reference 안내](UCF_RAW_REFERENCE.md)를 따른다.
