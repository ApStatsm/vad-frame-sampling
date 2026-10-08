# UCF 공식 feature 추출 방식 검증

`scripts/verify_ucf_extraction.py`는 같은 영상 구간에서 얻은 CLIP feature를 공식 `.npy`와 비교한다. VadCLIP 탐지 모델, GT annotation, AUC/AP는 사용하지 않는다. 기존 raw reference 설정을 수정하거나 원본 입력을 덮어쓰지 않는다.

초기 작성 당시에는 정적 검토만 수행했다. 이후 사용자가 이 검증 goal에 한해 실행을 허용하여 실제 비교를 수행했다. 최신 수치·조건별 실행·남은 문제는 [실행 기록](EXTRACTION_DIAGNOSTIC_RUNS.md)에 있다. 원본 recipe는 여전히 미확정이다.

## 기본 실행

프로젝트 루트에서 Apple Silicon은 다음 명령을 사용한다.

```sh
.venv/bin/python scripts/verify_ucf_extraction.py --device mps
```

CPU는 `--device cpu`, NVIDIA GPU는 `--device cuda`로 바꾼다. 기본 precision은 모든 장치에서 float32이며 MPS/CUDA에서만 `--precision float16`을 지정할 수 있다. Precision별 실행은 별도 출력 폴더를 만든다. Batch size 기본값은 32다. 메모리가 부족하면 `--batch-size 16` 또는 더 작은 양수로 줄인다.

기본 영상은 manifest 순서의 첫 anomaly, 첫 normal, 첫 anomaly와 다른 category의 첫 영상이다. 현재 파일 기준 `Abuse028_x264.mp4`, `Normal_Videos_003_x264.mp4`, `Arrest001_x264.mp4`다. 영상별 기본 block index는 `0, 1, 2, 3, 8, 16, 32`다. 영상별 최대 112장을 CLIP에 넣고, 마지막 지정 block까지 최대 528프레임을 순차 디코딩한다. 해시 확인에는 선택한 영상 파일 전체를 읽는다.

Block index는 0부터 세며 block `b`는 프레임 `[16*b, 16*b+16)`이다. 모든 프레임은 0부터 순차 디코딩하고 seek를 사용하지 않는다. 기본 가설은 이 block을 공식 feature의 행 `b`에 대응시키는 것이다. 실제로 해당 대응이 맞는지는 비교 대상이며 확정된 원본 recipe로 취급하지 않는다. Prefix 비교이므로 마지막 불완전 구간과 전체 영상의 EOF/프레임 수는 검증하지 않는다.

## 입력과 후보

- 설정: `configs/ucf_resources.json`의 `features_root`, `clip_checkpoint`.
- 영상 목록/상대경로: `UCF-Crime/test/manifest.json`.
- 공식 feature: 원본 `VadCLIP/list/ucf_CLIP_rgbtest.csv`의 `UCFClipFeatures` 뒤 상대 폴더/파일명을 configured feature root에 붙인다. 학습/event 폴더에 같은 normal basename이 있으므로 파일명만으로 찾지 않는다. CSV의 옛 절대경로 prefix는 사용하지 않으며 중복된 test 목록·누락·root 밖 경로는 오류다. 오래된 `ucf_CLIP_rgbtest.local.csv`의 절대경로는 사용하지 않는다.
- CLIP: 프로젝트의 `VadCLIP/src/clip` 구현과 이미 확보한 local checkpoint를 사용한다. 모델 이름으로 로딩하지 않아 자동 다운로드가 없다.
- Crop: `VadCLIP/src/crop.py`의 `image_crop`을 그대로 호출한다. 기본 type 5는 resize 340×256 → center crop 224×224 → horizontal flip이다. 이어서 `clip.load`가 반환한 원본 preprocess를 적용한다.
- `single_offset_00`~`single_offset_15`: 각 16프레임 구간에서 해당 위치의 한 장 feature.
- `mean16`: 각 프레임을 각각 CLIP에 넣은 후 16개 feature의 float32 평균. 원본 픽셀의 평균이나 16프레임 묶음의 동시 temporal encoding이 아니다.
- `mean16_of_unit_frames`: 프레임별 feature를 먼저 L2 정규화한 뒤 평균하는 추가 가설.

위 18개 후보 각각에 대해 `raw`, `raw_cast_official`, `l2`, `l2_cast_official`을 비교한다. `raw`는 후보 벡터 그대로, `l2`는 최종 후보 벡터를 L2 정규화한 값이다. `*_cast_official`은 공식 파일의 dtype으로 저장할 때 생기는 양자화 차이를 검사한다. 공식 벡터는 비교 대상으로 그대로 보존한다. 공식 dtype이 float32이면 cast 전후가 같을 수 있고, 일부 후보/정규화 조합도 같을 수 있다. 같은 수치의 순위가 여러 개 나와도 서로 다른 추출 방식이 검증된 것으로 해석하지 않는다.

공식 `.npy`를 로드하거나 내용을 보고 프레임을 선택하지 않는다. 파일 길이는 지정 block/row의 유효 범위 확인에만 사용한다. 후보 프레임은 CLI에서 지정한 block의 16개 위치 전체이며, 모든 후보에 동일하다.

## 범위 확장과 정렬 점검

영상·구간을 직접 지정할 수 있다. 긴 영상의 뒤쪽 구간은 그 위치까지 순차 디코딩하므로 실행 시간이 늘어난다.

```sh
.venv/bin/python scripts/verify_ucf_extraction.py --device mps \
  --video Abuse030_x264.mp4 --video Normal_Videos_003_x264.mp4 \
  --blocks 0 1 2 3 8 16 32 64
```

한 token 정도의 정렬 차이를 검사하려면 다음 명령을 사용한다.

```sh
.venv/bin/python scripts/verify_ucf_extraction.py --device mps --row-shifts -1 0 1
```

`row_shift=s`는 영상 block `b`의 후보와 공식 행 `b+s`를 비교한다. 모든 shift가 공통으로 비교할 수 있는 block만 사용하므로 후보별 표본 수는 같다. 위 명령에서는 공식 행 -1이 없는 block 0이 제외되며 `skipped_requested_blocks`에 기록된다. 범위를 벗어난 block은 제외하고, 공통 block이 없으면 오류다. 공식 파일에는 존재하는 block인데 실제 영상에서 필요한 프레임을 읽지 못하면 오류로 종료한다.

다른 spatial crop은 `--crop-type N`으로 검사할 수 있다. 이 경우 공식 counterpart도 반드시 `__N.npy`를 사용한다. `__5.npy`에 임의의 다른 crop을 맞추는 기능이 아니다.

## 출력과 판단

매 실행은 `outputs/setup/ucf_extraction_verify_<timestamp>/` 아래에 별도 저장된다.

| 파일 | 내용 |
|---|---|
| `summary.json` | 설정, 입력/source SHA-256, environment/preprocess, 영상별 순위, 합산 순위, 오류/성공 여부 |
| `ranking.csv` | 동일 후보를 모든 영상·block에 적용한 합산 수치 비교 |
| `per_block.csv` | 영상·block·공식 행·offset/mean·정규화별 상세 수치 |
| `arrays/<video_stem>.npz` | block index, 16개 frame index/nominal time/decoder PTS, `[blocks,16,512]` frame features, shift별 공식 feature와 row index |

`success=true`는 비교 코드가 끝까지 실행됐다는 뜻이다. 원본 방식의 재현 성공을 뜻하지 않는다. `historical_recipe_confirmed`는 항상 false다.

순위는 전체 비교 벡터의 relative L2 error가 작은 순서다. 코사인 평균/최솟값, MAE, RMSE, 최대 절대 오차, norm, 완전 동일 element 비율과 허용 오차 통과 비율도 저장한다. 순위의 relative L2는 전체 block의 오차 제곱합/공식 벡터 제곱합의 제곱근이다. Norm이 큰 공식 벡터의 영향이 크다. Cosine 평균과 통과 비율은 block별 동일 가중치다. 영상별 표본 수가 다르면 영향도 달라지므로 영상별 순위 역시 확인한다.

기본 허용 오차는 `atol=1e-4`, `rtol=1e-3`이며 `abs(candidate-official) <= atol + rtol*abs(official)`를 512개 차원 전부 만족해야 그 block을 통과로 표시한다. CLI에서 바꿀 수 있지만 이것은 사전에 선언하는 수치 검사 기준일 뿐, 원본 재현의 공인 기준이 아니다. 결과가 나쁜 것을 통과시키려고 사후에 허용 오차를 늘리지 않는다.

판단할 항목:

1. 같은 고정 offset 또는 평균 방식이 여러 영상·block에서 일관되게 맞는가?
2. 높은 cosine뿐 아니라 raw numerical error와 norm도 작은가? 모든 비교 block에서 허용 오차를 통과하는가?
3. static/인접 프레임과 중복 변환 때문에 여러 후보가 동률인가? 동률이면 더 다양한 독립 구간을 비교한다.
4. 어떤 후보도 충분히 맞지 않으면 영상 버전, decoder, 전처리, CLIP checkpoint/precision, row 정렬을 확인한다. 순위 1이라는 이유로 원본 방식을 확정하지 않는다.

더 많은 독립 영상·구간에서도 일관된 수치 일치를 확인하면 해당 recipe를 지지하는 강한 근거가 된다. 원본 추출 이력과 마지막 구간 처리까지 직접 확인하려면 저자의 전체 추출 스크립트 또는 명시적인 답변이 필요하다.


## 2026-10-08 추가 — 이미지 전처리 진단

첫 MPS float32 실행의 best relative L2는 0.137517, 같은 조건의 float16은 0.137506이었다. ±1 official row 정렬 검사에서도 shift 0/offset 0이 합산 1위이며 relative L2=0.138268, tolerance pass=0%였다. 계산 precision 변경과 ±1 row shift만으로 일치를 얻지 못했다. 이 결과는 decoder와 이미지 전처리가 같다는 증거가 아니며, 특정 원인으로 확정하지 않는다.

새 `--preprocess-variants` 옵션을 지정하지 않으면 이전처럼 `published` 조건만 비교한다. 기존 출력 폴더와 baseline 설정은 보존한다. 전처리 진단을 함께 실행하려면 다음 명령을 사용한다.

```sh
.venv/bin/python scripts/verify_ucf_extraction.py   --device mps --precision float16 --blocks 1 8 32 --row-shifts -1 0 1   --preprocess-variants all
```

이 명령은 기존 대표 3개 영상에서 block 1/8/32의 16프레임을 현재 33개 전처리 조건으로 비교한다. 초기 sweep은 당시 등록된 13개였고 추가 가설은 별도 실행으로 검사했다. 각 조건에서도 row shift -1/0/1을 함께 검사하여 전처리 변경과 정렬 가설의 조합을 빠뜨리지 않는다. 최대 CLIP 입력은 3×3×16×33=4752장이다. 영상은 한 번만 순차 디코딩하지만 CLIP은 각 전처리 조건마다 실행한다. 모든 조건은 같은 frame/block/official row와 같은 모델·precision·허용 오차를 사용한다. 이 명령의 block 목록은 이전 실행과 다르므로 오차 개선은 같은 새 실행 안의 `published` 조건에 상대적으로 판단한다.

| `preprocess_variant` | 공개 코드 조건에서 바꾸는 부분 |
|---|---|
| `published` | 기존 `image_crop` 그대로, 비교 기준 |
| `bgr_as_rgb` | crop 후 RGB 채널을 다시 바꿔 BGR 배열을 RGB 이미지로 해석 |
| `opencv_cubic` | resize 보간을 OpenCV cubic으로 변경, 크기와 crop 유지 |
| `opencv_area` | resize 보간을 OpenCV area로 변경, 크기와 crop 유지 |
| `pil_bicubic` | 340×256 resize를 Pillow bicubic으로 수행, crop 유지 |
| `shortside_opencv_linear` | OpenCV linear로 짧은 변 256까지 비율 유지 resize 후 중앙 crop |
| `shortside_bicubic` | Pillow bicubic으로 짧은 변 256까지 비율 유지 resize 후 중앙 crop |
| `clip_direct` | 외부 resize/crop을 생략하고 RGB 원본을 CLIP preprocess에 직접 입력, type 5면 먼저 좌우 반전 |
| `jpeg_q95` | decode 직후 원본 해상도 BGR frame을 OpenCV JPEG quality 95로 메모리 안에서 저장·재읽기 후 공개 crop |
| `crop_x_minus1`, `crop_x_plus1` | 원래 중앙 crop에서 x 시작 위치만 ±1 pixel 이동 후 기존 flip 적용 |
| `crop_y_minus1`, `crop_y_plus1` | 원래 중앙 crop에서 y 시작 위치만 ±1 pixel 이동 후 기존 flip 적용 |

이 목록은 원본 영상-to-feature script가 공개되지 않은 상황에서 검사할 입력 처리 가설이다. 저자가 JPEG를 저장했다거나 BGR을 사용했다는 뜻이 아니다. `shortside_opencv_linear`는 published의 보간을 유지한 채 resize 기하를 비교하며, `shortside_bicubic`은 `pil_bicubic`과 resize 기하를 비교할 수 있고, `clip_direct`는 외부 crop 파이프라인 자체가 다른 가설이다. 하나의 전체 순위만으로 원인을 확정하지 않는다.

특정 조건만 비교할 수도 있다.

```sh
.venv/bin/python scripts/verify_ucf_extraction.py --device mps --precision float16   --preprocess-variants published opencv_cubic pil_bicubic
```

진단 전처리는 center crop type 0/5에서만 지원한다. 다른 crop type은 기본 `published` 비교만 지원한다. `--crop-type`을 바꾸면 공식 counterpart도 그 번호로 바뀌며, 이 옵션에서는 공식 파일을 그대로 둔 채 명시적인 처리 가설을 비교한다.

추가 출력:

- `ranking.csv`, `per_block.csv`, summary의 영상별/합산 순위에 `preprocess_variant`가 추가된다.
- `preprocess_ranking.csv`와 `summary.preprocess_ranking`에는 조건별로 가장 가까운 고정 temporal/normalization 후보 한 개가 저장된다. 모든 조건을 같은 block에서 비교한다.
- `summary.preprocess_recipes`는 각 조건의 구현상 의미를 기록한다.
- NPZ의 `frame_features`는 첫 전처리 조건의 `[blocks,16,512]` 배열이며 `frame_features_variant`가 해당 이름이다. `frame_features_by_preprocess`는 `[variants,blocks,16,512]`, `preprocess_variants`는 그 순서다. 기본 첫 조건은 published다.
- `encoder_input_images`는 모든 조건의 encoder 입력 수 합계이고 `encoder_input_images_per_variant`는 조건 하나의 입력 수다. decoder prefix count는 조건 수만큼 곱하지 않는다.

전처리 하나에서 충분히 가까운 일치가 나오면 다른 독립 영상·block에서도 같은 고정 조건으로 확인한다. 모든 조건이 일치하지 않으면 원본 영상 버전, 실제 원본 decoder/프레임 이미지 저장 경로, CLIP 구현·가중치 또는 공개되지 않은 처리가 남을 수 있다. 허용 오차를 느슨하게 해서 일치로 바꾸거나 rank 1을 원본 recipe로 채택하지 않는다. 각 조건의 실제 실행 결과는 실행 기록에 연결했다. 한 번의 joint sweep에서 모든 33개를 동시에 검사했다는 의미는 아니다.


## 현재 추가 진단 옵션

`shortside_opencv_scale`는 output 크기 지정과 달리 `fx=fy=256/min(H,W)`를 사용한다. 320×240 입력에서는 341×256 출력의 중앙 224 crop과 선택한 flip을 사용한다. 같은 출력 크기여도 pixel 좌표 계산이 달라 결과가 달랐다. 추가 float resize/rounding/kernel/preflip 조건도 `--help`와 `summary.preprocess_recipes`에 기록된다.

`scale_jpeg_raw95/100`은 fx/fy resize 이전의 JPEG 중간 저장, `scale_jpeg_resized95/100`은 resize 후 crop 이전의 JPEG 중간 저장 가설이다. 기존 `jpeg_q95`는 공개 crop 조건이므로 별도다. 모두 메모리 내부 왕복이고 원본 영상을 수정하지 않는다.

`--image-normalization clip|imagenet|symmetric|none`, `--image-channel-order rgb|bgr`, `--activation quick_gelu|gelu|gelu_tanh`, `--attention-backend default|legacy`는 명시적인 진단 옵션이다. 기본은 원본 CLIP normalization/RGB/QuickGELU/현재 attention이며 checkpoint weights는 고정이다. 대체 연산 검사는 원본 recipe 또는 sampler 성능 비교로 간주하지 않는다.

현재 가장 가까운 가설을 기존 대표 영상에서 다시 검사하는 명령:

```sh
.venv/bin/python scripts/verify_ucf_extraction.py --device mps --precision float16 \
  --blocks 1 8 32 --row-shifts -1 0 1 \
  --preprocess-variants published shortside_opencv_scale
```

`scan_ucf_crop_geometry.py`는 지정 offset 한 장만으로 resize/crop 위치·flip을 검사하며 temporal 평균 검증을 대신하지 않는다. `compare_ucf_decoders.py`는 지정된 RGB frame 픽셀을 대조한다. `search_ucf_feature_timeline.py --video Shooting032_x264.mp4 --precision float32`는 EOF까지 stride16로 encoding하여 질의 official row의 최근접 raw feature를 찾는다. 이 최근접 검색은 공식 feature를 진단 target으로 사용하는 것이며 sampling 알고리즘·GT 사용·VAD 탐지 평가가 아니다. 저장된 feature 전체에서 가까운 이웃을 찾았다는 이유로 원본 선택 방식이라고 확정하지 않는다.
