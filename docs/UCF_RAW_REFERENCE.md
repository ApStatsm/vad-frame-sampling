# UCF-Crime raw-video reference

작성일: 2026-10-07. 갱신: 2026-10-08 — 사용자 full run `ucf_raw_full_20261008_150142`(branch1 AUC 87.19%, 공식 대비 −0.83pp)를 UCF sampling 비교의 기준으로 확정했다. 추출 방식 탐색 문서는 이 파일로 통합하고 상세 기록은 [`archive/`](archive/)로 옮겼다.

원본 영상 → 명시된 후보 프레임 → crop/CLIP image encoder → 고정 VadCLIP → 공식 평가 시간축의 점수/metric을 연결한다. CLIP forward와 전체 영상 처리 시간을 따로 측정한다. 이번 버전은 Full/Reference 100%만 제공한다. Uniform/Random은 reference 조건 검증 후 연결한다.

## 공식 저장소의 샘플링 설명 확인 — 2026-10-07

저자의 [Issue #1 답변](https://github.com/nwpu-zxr/VadCLIP/issues/1#issuecomment-1873717061)은 이전 I3D feature가 16프레임을 나타내므로 VadCLIP도 16프레임마다 feature를 샘플링한다고 설명한다. [같은 issue의 모델 답변](https://github.com/nwpu-zxr/VadCLIP/issues/1#issuecomment-1873627266)에서 standard CLIP ViT-B/16와 10-crop을 명시했다. 따라서 시간 단위 16프레임은 저자 설명으로 확인됐으며 score 반복만으로 추론한 값은 아니다.

저자의 [Issue #2 답변](https://github.com/nwpu-zxr/VadCLIP/issues/2#issuecomment-1880791613)은 crop 0=중앙, 1~4=모서리, 5~9=각 좌우 반전 대응을 확인한다. UCF test는 __5, XD test는 __0을 사용하며 저자는 해당 설정에서 결과가 좋았다고 설명한다. 따라서 UCF의 __5는 좌우 반전된 중앙 crop으로 확인됐다. 우리 sampler 연구에서는 crop을 고정한다.

아직 구간 내 정확한 frame offset/프레임 1장 사용 여부/16장 평균 여부/tail·decoder 조건은 완전한 저자 원본 코드로 확정되지 않았다. [Issue #14의 16장 평균 댓글](https://github.com/nwpu-zxr/VadCLIP/issues/14#issuecomment-2261922859)은 일반 사용자의 추정이고 저자 확인이 아니다. [Issue #29](https://github.com/nwpu-zxr/VadCLIP/issues/29)에는 한 장/16장 평균 모두 배포 feature와 다르다는 사용자 보고가 있다. 확인한 관련 댓글에서 원본 raw 추출 알고리즘의 추가 저자 답변은 찾지 못했다. 2026-10-08 수치 진단으로 offset 0 한 장 방식을 재구성했다(아래 절). mean_block은 비교용 가설로만 유지한다.

학습·테스트의 feature 처리도 별개다. [공개 dataset/tools 코드](https://github.com/nwpu-zxr/VadCLIP/blob/main/src/utils/tools.py)에서는 학습용 긴 feature sequence를 256개의 균등 시간 구간으로 나눠 평균하고 짧으면 zero-pad한다. Test는 전체 feature sequence를 256-token window로 나누고 유효한 전체 score를 16배 복원한다. 이 학습용 feature 평균은 raw 구간 16장 평균의 근거가 아니다.

## 현재 추출 조건의 상태

2026-10-08 진단으로 float16 cohort(test 286개)의 추출 방식을 **재구성했다.** 표본 block과 영상 2개 전체에서 공식 feature와 상대 L2 약 0.0015~0.0018로 일치하며, 이는 CPU/MPS 백엔드 간 차이와 같은 수준이다. 근거는 `outputs/setup/ucf_recipe_reconstruction_20261008.json`과 [실행 기록](archive/EXTRACTION_DIAGNOSTIC_RUNS.md)에 있다. 원저자 코드로 확인한 것은 아니며, 290개 전체 feature 대조는 full 실행의 `feature_agreement_by_official_dtype`로 확인한다. 가장 긴 test 영상 4개(float32, ceil(N/16))는 같은 방식으로 일치하지 않는다(상대 L2 약 0.25~0.37). 시간축이 아닌 이미지 처리 차이로 보이며 미해결이다. 이 4개는 frame-level 평가 시간축의 24.7%(273,728/1,109,888프레임, 대부분 normal)를 차지하므로, raw full의 AUC/AP가 공식 결과와 다르면 우선 이 영상들의 영향을 의심한다. 결과의 `official_extraction_reproduced`는 계속 false로 둔다.

| 설정 | 기본값 / 의미 |
|---|---|
| `extraction_status` | `reconstructed_sample_verified_float16_cohort`. 기각된 이전 provisional 설정(offset 8)은 `git show ucf-raw-reference-v1:configs/ucf_raw_reference_provisional_offset8.json` |
| `block_frames` | 16. 원본 decode counter 기준 [0,16), [16,32), …의 완전한 구간 |
| `temporal_mode` | `single_frame`. 각 구간에서 한 장을 encode하여 token 하나 생성. 16장 평균은 시간축 V자 오차 곡선과 장면 전환 사례로 배제 |
| `frame_offset` | 0. Zero-based frame 0,16,32,… (각 구간의 첫 프레임) |
| `color_conversion` | `x86_swscale_emulation`. ffmpeg CLI의 bit-exact yuv420p를 x86 libswscale 고속 경로(16-bit `pmulhw`, nearest chroma, BT.601 limited)로 변환. 이 Mac의 OpenCV(ARM/C 경로)는 약 1 grey level 어두워 상대 L2 약 0.042가 남는다. `opencv_bgr`은 이전 방식 |
| `image_geometry` | `shortside_fxfy`. `cv2.resize(fx=fy=256/240)` → 341×256 → `[16:240, 58:282]`. 공개 crop.py의 340×256 고정 resize(`published_340x256`)는 상대 L2 약 0.12 |
| `tail_policy` | `match_official_rows`. 공식 row가 ceil(N/16)인 경우(float32 4개)에만 꼬리 구간의 offset 프레임 1장을 encode한다. 공식 row 수(배열 길이)만 사용하며 GT label은 사용하지 않는다. float16 286개는 floor(N/16)로 꼬리를 버린다 |
| `crop_type` | 5. 중앙 crop 후 좌우 반전. 10개 crop 배치(중앙 `[16:240, 58:282]`, 모서리 4개, 5~9는 각 좌우 반전) 모두 Abuse028 표본 row에서 공식 `__0`~`__9`와 상대 L2 0.001~0.002로 일치 |
| CLIP 입력 | 기존 bundled CLIP의 RGB/PIL 전처리·정규화 사용 |
| `encoder_batch_size` | 32. 입력 이미지를 bounded batch로 처리 |
| `encoder_precision` | `float32`. 고정된 CLIP visual module의 별도 복사본을 사용; VadCLIP/text encoder의 원래 precision은 유지 |
| `feature_storage_dtype` | `official`. 영상별 기존 feature dtype(float16/float32)에 맞춰 저장하고 모델에 입력 |
| `feature_l2_normalize` | false. encode_image 결과를 추가 L2 정규화하지 않음 |
| Seed | 234. PyTorch/NumPy/Python seed 설정; 반복·장치 간 bitwise 동일성 보장은 미검증 |
| Warm-up | 합성 이미지 batch와 256-token feature를 각각 2회 처리; 측정에서 제외 |

`mean_block`은 완전한 구간의 16장 모두를 encode한 뒤 float32 feature 평균을 token 하나로 만든다. 이 모드의 batch size는 16의 배수여야 한다. 두 모드는 encoder 입력 수·100% 후보 분모가 다르므로 동일 budget/동일 효율 조건으로 섞어 비교하지 않는다. Crop/temporal mode/offset/precision/정규화는 test AUC를 최대화하기 위해 선택하지 않는다. 원본 코드·추출 기록과 입력 feature 정렬을 근거로 확인하고 sampling 비교 전에 고정한다.

## 재구성 근거와 기각한 가설 (요약)

통합 근거는 `outputs/setup/ucf_recipe_reconstruction_20261008.json`이다. 진단 스크립트(`verify_ucf_extraction.py`, `scan_ucf_crop_geometry.py`, `search_ucf_*`, `compare_ucf_decoders.py`, `test_ucf_*`)는 2026-10-08 정리에서 작업 트리에서 제거했다. `git show ucf-raw-reference-v1:scripts/<file>`로 확인한다. 진단 run 폴더는 로컬 `.local-backups/pre-refactor-20261008/outputs/setup/`에 있다(git 미추적).

| 가설 | 결과 (공식 feature 대비 상대 L2) | 판단 |
|---|---|---|
| 재구성 recipe(offset 0, x86 YUV→RGB, fx/fy, crop5) | 표본 0.0015~0.0018, full 290개 중 float16 286개 ≤ 0.005 | 채택 |
| OpenCV(ARM/C) YUV→RGB | 약 0.042 | 기각(약 1 grey level 차이) |
| 공개 crop.py 340×256 고정 resize | 약 0.12 | 기각 |
| 16장 평균(`mean_block`) | 시간축 V자 오차, 장면 전환 사례 불일치 | 기각 |
| floor((N−1)/16) 꼬리 규칙 | N%16==0 영상 20개 불일치 | 기각 |
| ffmpeg JPEG 프레임 저장 | 0.21~0.29 | 기각 |
| full/bilinear chroma, accurate_rnd | 0.18 / 0.074 | 기각 |
| float32 cohort 4개: JPEG·기하·시간 평균·blur·사진 보정·crop 순서 | 약 0.30 유지 또는 악화 | 미해결(다른 사본/파이프라인 추정) |

상세 실행 기록: [EXTRACTION_DIAGNOSTIC_RUNS.md](archive/EXTRACTION_DIAGNOSTIC_RUNS.md), [UCF_EXTRACTION_VERIFICATION.md](archive/UCF_EXTRACTION_VERIFICATION.md), [VADCLIP_EXTRACTION_AUDIT.md](archive/VADCLIP_EXTRACTION_AUDIT.md), [EXTRACTION_COMPLETION_AUDIT.md](archive/EXTRACTION_COMPLETION_AUDIT.md).

## 필요한 기존 입력

- `configs/ucf_resources.json`, 통과한 `outputs/setup/ucf_baseline_input_check.json`·`ucf_resource_check.json` 및 local feature CSV.
- `UCF-Crime/test/manifest.json`과 290개 raw 영상 symlink. 연결 대상인 extracted 폴더도 보존한다.
- 기존 CLIP/UCF pretrained checkpoints와 official features. 원본을 쓰거나 재생성하지 않는다.
- `VadCLIP/list/gt_ucf.npy`, full mAP용 `gt_segment_ucf.npy`·`gt_label_ucf.npy`.
- 기존 성공 official run의 `summary.json`과 `scores/*.npz`. 기본 비교 대상은 `ucf_full_20261007_010204`다. NPZ는 Git에서 제외되므로 다른 컴퓨터에서는 official full을 실행한 뒤 `--official-run`으로 새 run을 지정한다.

Raw 설정/명령은 effective 값과 함께 기록한다. Config/manifest/frame GT/checkpoint·원본 평가 코드가 기존 audit/run과 맞는지 검사하고, 각 원본 영상의 크기와 SHA-256도 대조한다. Baseline의 source hash가 달라졌다면 변경 이유를 확인하고 해당 코드로 official 재현을 먼저 갱신한다.

## 사용자 실행

프로젝트 루트에서 full을 직접 실행할 수 있다. 기존 smoke 결과는 필요하지 않다.

```sh
.venv/bin/python scripts/evaluate_ucf_raw.py --mode full --device mps
```

짧은 anomaly·짧은 normal·512 token 이상인 영상의 최소 길이 사례를 먼저 확인하려면 다음 선택적 smoke를 사용한다. 영상 전체를 decode하고 평가용 window 경계도 거치며, subset을 전체 benchmark로 평가하지 않는다.

```sh
.venv/bin/python scripts/evaluate_ucf_raw.py --mode smoke --device mps
```

특정 영상 하나를 확인하려면:

```sh
.venv/bin/python scripts/evaluate_ucf_raw.py --mode smoke --device mps --video Abuse028_x264.mp4
```

구간 전체 평균이라는 별도 추출 가설을 입력 대조로 확인할 때:

```sh
.venv/bin/python scripts/evaluate_ucf_raw.py --mode smoke --device mps --temporal-mode mean_block --video Abuse028_x264.mp4
```

`--video`는 여러 번 지정할 수 있지만 full에는 사용할 수 없다. `--frame-offset 0` 등으로 구간 내 위치를 명시할 수 있다. 다른 설정은 `configs/ucf_raw_reference.json`을 수정하거나 `--config`로 별도 설정을 전달한다. 비교 run 변경은 `--official-run outputs/baseline/<실제_run_이름>`으로 지정한다. CPU/CUDA는 device 인수로 변경한다.

## 시간 측정 범위

각 영상의 open부터 전체 decode·후보 선택·전처리·전송·CLIP·feature 집계·VadCLIP·score 복원까지 실제 경과 시간을 측정한다. `opencv_bgr`은 VideoCapture(FFmpeg backend), `x86_swscale_emulation`은 ffmpeg CLI의 yuv420p pipe로 seek 없이 순서대로 영상 전체를 읽는다(OpenCV는 FPS/header 확인에만 사용). 모든 프레임을 decode하므로 후보를 줄인다고 decoding 비용 감소를 가정하지 않는다. x86 색 변환 재현은 선택된 프레임에만 numpy로 수행하며 `color_conversion` 단계로 따로 기록한다. 이 비용은 실제 x86 swscale보다 크고 선택 프레임 수에 비례하므로, 효율 비교에서는 이 단계를 분리해 보고한다.

| 결과 key | 범위 |
|---|---|
| `timing_seconds.clip_forward` | 동기화된 CLIP image encoder forward만 |
| `decode` | 영상 open/read/close (ffmpeg pipe 포함) |
| `color_conversion` | 선택 프레임의 YUV/BGR→RGB 변환 |
| `selection` | 후보 index/시점 선택·수집 |
| `preprocess` | Resize/crop/RGB/PIL/CLIP transform/CPU batch 구성 |
| `clip_input_transfer` / `clip_output_transfer` | 입력 장치 전송 / float32 출력 CPU 전송 |
| `feature_aggregation` | 구간 평균·선택적 정규화·dtype 변환·feature/시간축 배열 구성 |
| `vad_total` | 기존 infer_feature의 tensor/window/mask 구성·전송·모델/두 score 계산·CPU 복사 |
| 영상별 `model_forward_seconds` | 기존 evaluator가 측정하는 VadCLIP forward 부분 |
| `score_restore` | token 점수를 원본 16-frame 구간으로 반복 |
| `end_to_end_seconds` | 성공 영상별 위 pipeline 경과 시간의 합. 각 단계 합산으로 추정하지 않음 |
| `processing_loop_wall_seconds` | 영상 hash·공개 입력 대조·GT 분석·저장까지 포함한 loop 경과 시간 |
| `metric_seconds` | Full의 최종 metric 계산, E2E에서 제외 |

초기 import/입력 검사/model 로딩/warm-up, 공개 feature·score 대조, GT 분석, metric 계산과 결과 serialization은 pipeline E2E 밖이다. 영상 hash를 timed decode 직전에 읽으므로 OS cache는 통제되지 않고 warm일 가능성이 있다. 이를 summary에 기록한다. 한 번의 측정이며 최종 speedup 수치는 계산하지 않는다. 추후 sampler 비교에서도 같은 장치/precision/batch/I/O/타이밍 경계를 사용하고 반복 조건을 확정한다. 기존 cached-feature loop 시간과 이 raw 시간을 나누어 speedup으로 사용하지 않는다.

## 출력과 확인할 항목

새 `outputs/baseline/ucf_raw_<smoke|full>_<YYYYMMDD_HHMMSS>/`에 저장한다. Asia/Seoul 시각, 충돌 시 `_2` 등을 추가한다. 기존 run은 갱신하지 않는다.

- `summary.json`: 설정·입력/source/checkpoint hash·seed/장비/precision·진행·시간·대조 결과. 실패 시 짧은 error와 stderr traceback, 별도 error.txt 없음.
- `features/*.npz`: `features` [T,512], `selected_frame_indices` [N], `selected_timestamps_seconds` [N], `decoder_timestamps_seconds` [N], `token_member_frame_indices` [T,1 또는 16], `token_start_frames`·`token_end_frames` [T]. 모두 object array 없이 `allow_pickle=False`로 읽을 수 있다.
- `scores/*.npz`: 기존 두 branch token score, 14개 class probability/labels 및 해당 공식 frame GT. Full의 `frame_scores.npz`에는 전체 두 scalar branch 점수.
- `videos[*].feature_comparison`: shape, cosine 평균/최소, 절대 오차, relative L2 error와 norm 대조. `feature_agreement_by_official_dtype`는 공식 dtype(float16/float32) cohort별 상대 L2 중앙값·최댓값과 0.005 이하 영상 수를 요약한다. `score_comparison`: 기존 token 점수와의 절대 오차.
- Full의 `metrics`: 기존 두 branch AUC/AP/Ano-AUC와 원본 290개/14개 클래스 fine-grained mAP/AVG. `official_comparison`은 raw−official을 pp로 기록한다. 재현 허용 오차는 아직 미정이므로 자동 pass/improvement 판단은 하지 않는다.
- `event_coverage`: published [start,end) interval에 선택 원본 frame이 하나 이상 들어가는 비율(0~1). 선택이 끝난 뒤 GT로 분석한다. 구간 clipping/좌표 이동 없음. Feature timeline/tail을 넘는 사건도 분모에 남는다.

Decode counter는 zero-based다. Timestamp=index/fps는 CFR 가정이다. `opencv_bgr`에서는 OpenCV decoder PTS도 저장해 단조성·nominal timestamp 오차를 남기고, ffmpeg pipe에서는 PTS를 얻지 않으므로 NaN으로 저장한다(`decoder_pts_source`). Manifest FPS/header 값과 decoder 값을 함께 확인한다. 원본 annotation 좌표의 base/endpoint 및 실제 CFR 확인은 아직 완료하지 않았으므로 raw coordinate 검증 완료라고 표시하지 않는다.

Decode EOF의 실제 frame 수가 header와 다르면 실패한다. `tail_policy`를 적용한 뒤에도 raw token 수가 audited official T와 다르면 추출 feature/선택 정보를 진단 파일로 저장하고 실패한다. 길이를 맞추기 위한 clipping/resampling은 하지 않는다. 같은 T가 확인되면 token당 16 frame으로 복원하여 **공식 평가 시간축 전체**를 평가한다. 공식 시간축 밖 마지막 raw tail은 별도로 기록한다.

사전 제안 허용 오차(branch1 AUC |Δ| ≤ 0.5pp)는 −0.83pp로 넘었고, 사용자가 float32 4개 미재현을 제한으로 기록한 뒤 raw reference를 기준으로 확정했다(2026-10-08). Uniform/Random은 이 reference의 CLIP 전 단계에 추가한다.

