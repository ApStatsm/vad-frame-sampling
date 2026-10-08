> **Archive (2026-10-08 정리)**: 추출 방식 탐색 기록이다. 결론은 [UCF_RAW_REFERENCE.md](../UCF_RAW_REFERENCE.md)에 통합했다. 언급된 진단 스크립트는 `git show ucf-raw-reference-v1:scripts/<file>`로, 진단 run 폴더는 로컬 `.local-backups/pre-refactor-20261008/`에서 확인한다.

# 원본 추출 검증 실행 기록

2026-10-08: 사용자가 이 goal 동안에만 Codex의 실행을 명시적으로 허용했다. 범위는 원본 frame/feature 추출 방식 검증이다. 탐지 성능·sampling 비교는 실행하지 않았다.

> 저장 위치 안내: `outputs/setup/` 아래 실행별 폴더(아래 표의 링크 대상)는 2026-10-08부터 git에서 제외하고 로컬에만 둔다. 공유되는 요약은 `outputs/setup/ucf_recipe_reconstruction_20261008.json`, `ucf_extraction_findings_20261008.json` 등 상위 통합 파일과 이 문서다.

**원본 recipe는 여전히 미확정이다.** 저장된 공식 feature와 수치 일치한 후보가 없다. 아래 수치는 탐지 성능이 아닌 feature 벡터의 상대 L2 오차다.

## 현재 근거

- 대표 3개 영상의 같은 block 1/8/32에서 공개 crop의 오차는 약 0.139, 비율 유지한 명시적 크기 resize는 0.06889, `cv2.resize(..., None, fx=fy=256/min(H,W))`는 0.04200이었다. 이어서 중앙 224 crop·좌우 반전·원본 CLIP preprocess를 적용했다.
- 이 조건의 `single_offset_00`은 cosine 0.999121이지만 max error 0.115295, tolerance pass 0%다. `mean16/raw/shift0`의 상대 L2는 0.12467이다. `16k`번째 한 장은 우선 가설이며 확정된 원본 규칙이 아니다.
- 조건 선정에 사용하지 않은 test 영상 4개(Arson007/Burglary005/RoadAccidents001/Normal006)의 지정 구간에서도 같은 고정 한 장/offset0/shift0 조건이 가장 가까웠다. 합산 상대 L2 0.04220, mean16 0.08567, 모두 허용 오차 미통과다. Sampler 성능의 별도 validation set은 아니다.
- CPU/MPS·FP32/FP16 차이, OpenCV 4.7/5.0 비교, PIL/torch resize, BGR, normalization, GELU, legacy attention, JPEG 중간 저장 가설에서 수치 재현을 얻지 못했다. 이 조건들 밖의 모든 원인을 배제한 것은 아니다.
- 290개 공식 test feature 중 286개는 float16이고 4개는 float32다. float32 4개는 Shooting032, Normal924/935/940이다. 이 예외의 오차는 일반 7개 표본과 달리 훨씬 크며 같은 recipe로 처리된 것인지 알 수 없다.
- 공식 VadCLIP checkpoint와 사용한 CLIP checkpoint의 visual tensor 152개는 저장 dtype을 맞춘 뒤 모두 동일했다. 이것만으로 배포 feature 제작 때도 같은 구현·입력 처리를 사용했다고 확정하지 않는다.

## 비교 실행 전체 목록

각 행은 해당 실행 내부에서 가장 가까운 후보다. 영상/구간/모델 연산이 다른 행은 서로 동일 표본으로 간주하지 않는다. 설정·개별 영상 순위·입력/source hash는 연결된 summary에 보존된다.

| 실행 | 영상/표본 | 계산 조건 | best 후보 | 상대 L2 | cosine | pass |
|---|---|---|---|---:|---:|---:|
| [ucf_extraction_verify_20261008_104914](../../.local-backups/pre-refactor-20261008/outputs/setup/ucf_extraction_verify_20261008_104914/summary.json) | default 3; blocks 0,1,2,3,8,16,32; 21 rows | mps; float32; cv5.0.0 | published; single_offset_00/raw; shift 0 | 0.137516991 | 0.990608644 | 0.0% |
| [ucf_extraction_verify_20261008_105323](../../.local-backups/pre-refactor-20261008/outputs/setup/ucf_extraction_verify_20261008_105323/summary.json) | default 3; blocks 0,1,2,3,8,16,32; 21 rows | mps; float16; cv5.0.0 | published; single_offset_00/raw; shift 0 | 0.137506072 | 0.990608531 | 0.0% |
| [ucf_extraction_verify_20261008_105907](../../.local-backups/pre-refactor-20261008/outputs/setup/ucf_extraction_verify_20261008_105907/summary.json) | default 3; blocks 0,1,2,3,8,16,32; 18 rows | mps; float16; cv5.0.0 | published; single_offset_00/raw; shift 0 | 0.138268468 | 0.990565129 | 0.0% |
| [ucf_extraction_verify_20261008_110848](../../.local-backups/pre-refactor-20261008/outputs/setup/ucf_extraction_verify_20261008_110848/summary.json) | default 3; blocks 1,8,32; 9 rows | mps; float16; cv5.0.0 | shortside_opencv_linear; single_offset_00/raw; shift 0 | 0.068888790 | 0.997655343 | 0.0% |
| [ucf_geometry_scan_20261008_111156](../../.local-backups/pre-refactor-20261008/outputs/setup/ucf_geometry_scan_20261008_111156/summary.json) | default 3; blocks 1,8,32; 9 rows | mps; float16; cv5.0.0 | linear_w341_h256_x58_y16; single_offset_00/raw; shift 0 | 0.068888790 | 0.997655343 | 0.0% |
| [ucf_extraction_verify_20261008_111259](../../.local-backups/pre-refactor-20261008/outputs/setup/ucf_extraction_verify_20261008_111259/summary.json) | default 3; blocks 1,8,32; 9 rows | mps; float16; cv5.0.0 | shortside_opencv_linear; single_offset_00/raw; shift 0 | 0.068888790 | 0.997655343 | 0.0% |
| [ucf_geometry_scan_20261008_111454](../../.local-backups/pre-refactor-20261008/outputs/setup/ucf_geometry_scan_20261008_111454/summary.json) | default 3; blocks 1,8,32; 9 rows | cpu; float32; cv5.0.0 | linear_w341_h256_x58_y16; single_offset_00/raw; shift 0 | 0.068926592 | 0.997652130 | 0.0% |
| [ucf_extraction_verify_20261008_111631](../../.local-backups/pre-refactor-20261008/outputs/setup/ucf_extraction_verify_20261008_111631/summary.json) | default 3; blocks 1,8,32; 9 rows | mps; float16; cv5.0.0; norm=imagenet | shortside_opencv_linear; single_offset_00/raw; shift 0 | 0.144517807 | 0.989761553 | 0.0% |
| [ucf_extraction_verify_20261008_111905](../../.local-backups/pre-refactor-20261008/outputs/setup/ucf_extraction_verify_20261008_111905/summary.json) | default 3; blocks 1,8,32; 9 rows | mps; float16; cv5.0.0 | shortside_opencv_linear; single_offset_00/raw; shift 0 | 0.068888790 | 0.997655343 | 0.0% |
| [ucf_geometry_scan_20261008_112129](../../.local-backups/pre-refactor-20261008/outputs/setup/ucf_geometry_scan_20261008_112129/summary.json) | default 3; blocks 1,8,32; 9 rows | mps; float16; cv5.0.0 | av_bilinear_w341_h256_x58_y16; single_offset_00/raw; shift 0 | 0.178754626 | 0.984267421 | 0.0% |
| [ucf_geometry_scan_20261008_112208](../../.local-backups/pre-refactor-20261008/outputs/setup/ucf_geometry_scan_20261008_112208/summary.json) | default 3; blocks 1,8,32; 9 rows | mps; float16; cv5.0.0; no flip | opencv_w341_h256_x58_y16_no_flip; single_offset_00/raw; shift 0 | 0.191497346 | 0.981600245 | 0.0% |
| [ucf_extraction_verify_20261008_112516](../../.local-backups/pre-refactor-20261008/outputs/setup/ucf_extraction_verify_20261008_112516/summary.json) | default 3; blocks 1,8,32; 9 rows | mps; float16; cv5.0.0 | shortside_opencv_scale; single_offset_00/raw; shift 0 | 0.041997638 | 0.999121154 | 0.0% |
| [ucf_extraction_verify_20261008_112840](../../.local-backups/pre-refactor-20261008/outputs/setup/ucf_extraction_verify_20261008_112840/summary.json) | default 3; blocks 1,8,32; 9 rows | mps; float16; cv4.7.0 | shortside_opencv_scale; single_offset_00/raw; shift 0 | 0.041997638 | 0.999121154 | 0.0% |
| [ucf_extraction_verify_20261008_113127](../../.local-backups/pre-refactor-20261008/outputs/setup/ucf_extraction_verify_20261008_113127/summary.json) | default 3; blocks 1,8,32; 9 rows | mps; float16; cv5.0.0 | shortside_opencv_scale; single_offset_00/raw; shift 0 | 0.041997638 | 0.999121154 | 0.0% |
| [ucf_extraction_verify_20261008_113300](../../.local-backups/pre-refactor-20261008/outputs/setup/ucf_extraction_verify_20261008_113300/summary.json) | default 3; blocks 1,8,32; 9 rows | mps; float16; cv5.0.0 | shortside_opencv_scale; single_offset_00/raw; shift 0 | 0.041997638 | 0.999121154 | 0.0% |
| [ucf_geometry_scan_20261008_113521](../../.local-backups/pre-refactor-20261008/outputs/setup/ucf_geometry_scan_20261008_113521/summary.json) | default 3; blocks 1,8,32; 9 rows | mps; float16; cv5.0.0 | opencv_w320_h240_x48_y8; single_offset_00/raw; shift 0 | 0.361694981 | 0.934904459 | 0.0% |
| [ucf_geometry_scan_20261008_113703](../../.local-backups/pre-refactor-20261008/outputs/setup/ucf_geometry_scan_20261008_113703/summary.json) | default 3; blocks 1,8,32; 9 rows | mps; float16; cv5.0.0; fx/fy resize; bt709 | av_rgb_opencv_w341_h256_x58_y16; single_offset_00/raw; shift 0 | 0.066994776 | 0.997782238 | 0.0% |
| [ucf_geometry_scan_20261008_113734](../../.local-backups/pre-refactor-20261008/outputs/setup/ucf_geometry_scan_20261008_113734/summary.json) | default 3; blocks 1,8,32; 9 rows | mps; float16; cv5.0.0; fx/fy resize | opencv_w340_h255_x57_y16; single_offset_00/raw; shift 0 | 0.148160285 | 0.989242646 | 0.0% |
| [ucf_extraction_verify_20261008_113911](../../.local-backups/pre-refactor-20261008/outputs/setup/ucf_extraction_verify_20261008_113911/summary.json) | default 3; blocks 1,8,32; 9 rows | mps; float16; cv5.0.0 | shortside_opencv_scale; single_offset_00/raw; shift 0 | 0.041997638 | 0.999121154 | 0.0% |
| [ucf_extraction_verify_20261008_114026](../../.local-backups/pre-refactor-20261008/outputs/setup/ucf_extraction_verify_20261008_114026/summary.json) | Arson007, Burglary005, RoadAccidents001, Normal_Videos_006; blocks 1,8,32; 11 rows | mps; float16; cv5.0.0 | shortside_opencv_scale; single_offset_00/raw; shift 0 | 0.042197830 | 0.999102223 | 0.0% |
| [ucf_geometry_scan_20261008_114407](../../.local-backups/pre-refactor-20261008/outputs/setup/ucf_geometry_scan_20261008_114407/summary.json) | Shooting032, Normal_Videos_924; blocks 1,8,32; 6 rows | cpu; float32; cv5.0.0; fx/fy resize | opencv_w341_h256_x58_y16; single_offset_00/raw; shift 0 | 0.327502384 | 0.947675905 | 0.0% |
| [ucf_extraction_verify_20261008_114634](../../.local-backups/pre-refactor-20261008/outputs/setup/ucf_extraction_verify_20261008_114634/summary.json) | Shooting032, Normal_Videos_924; blocks 1,8,32; 6 rows | mps; float32; cv5.0.0 | published; single_offset_04/raw; shift 0 | 0.319832108 | 0.949874140 | 0.0% |
| [ucf_extraction_verify_20261008_114848](../../.local-backups/pre-refactor-20261008/outputs/setup/ucf_extraction_verify_20261008_114848/summary.json) | default 3; blocks 1,8,32; 9 rows | mps; float16; cv5.0.0; activation=gelu | shortside_opencv_scale; single_offset_15/raw; shift 1 | 0.271047864 | 0.964316117 | 0.0% |
| [ucf_geometry_scan_20261008_115055](../../.local-backups/pre-refactor-20261008/outputs/setup/ucf_geometry_scan_20261008_115055/summary.json) | Shooting032, Normal_Videos_924; blocks 1,8,32; 6 rows | mps; float32; cv5.0.0; fx/fy resize | opencv_w341_h256_x58_y16; single_offset_00/raw; shift 0 | 0.327502067 | 0.947676007 | 0.0% |
| [ucf_extraction_verify_20261008_115146](../../.local-backups/pre-refactor-20261008/outputs/setup/ucf_extraction_verify_20261008_115146/summary.json) | Shooting032, Normal_Videos_924; blocks 1,8,32; 6 rows | mps; float32; cv5.0.0; activation=gelu | shortside_opencv_scale; single_offset_00/raw; shift 0 | 0.374326816 | 0.933723751 | 0.0% |
| [ucf_extraction_verify_20261008_115428](../../.local-backups/pre-refactor-20261008/outputs/setup/ucf_extraction_verify_20261008_115428/summary.json) | Shooting032, Normal_Videos_924; blocks 1,8,32; 6 rows | mps; float32; cv5.0.0; channels=bgr | published; single_offset_00/raw; shift -1 | 0.364421387 | 0.933485415 | 0.0% |
| [ucf_extraction_verify_20261008_120016](../../.local-backups/pre-refactor-20261008/outputs/setup/ucf_extraction_verify_20261008_120016/summary.json) | Shooting032, Normal_Videos_924; blocks 1,8,32; 6 rows | mps; float32; cv5.0.0; norm=symmetric | shortside_opencv_scale; mean16/raw; shift -1 | 0.375048176 | 0.931642873 | 0.0% |
| [ucf_extraction_verify_20261008_121333](../../.local-backups/pre-refactor-20261008/outputs/setup/ucf_extraction_verify_20261008_121333/summary.json) | default 3; blocks 1,8,32; 9 rows | mps; float16; cv5.0.0; attention=legacy | shortside_opencv_scale; single_offset_00/raw; shift 0 | 0.042025961 | 0.999120041 | 0.0% |
| [ucf_extraction_verify_20261008_121854](../../.local-backups/pre-refactor-20261008/outputs/setup/ucf_extraction_verify_20261008_121854/summary.json) | default 3; blocks 1,8,32; 9 rows | mps; float16; cv5.0.0 | shortside_opencv_scale; single_offset_00/raw; shift 0 | 0.041997638 | 0.999121154 | 0.0% |
| [ucf_geometry_scan_20261008_122109](../../.local-backups/pre-refactor-20261008/outputs/setup/ucf_geometry_scan_20261008_122109/summary.json) | Shooting032, Normal_Videos_924, Normal_Videos_935, Normal_Videos_940; blocks 1,8,32; 12 rows | mps; float32; cv5.0.0; fx/fy resize; no flip | opencv_w341_h256_x58_y16_no_flip; single_offset_00/raw; shift 0 | 0.338909279 | 0.941157728 | 0.0% |
| [ucf_extraction_verify_20261008_122844](../../.local-backups/pre-refactor-20261008/outputs/setup/ucf_extraction_verify_20261008_122844/summary.json) | Normal_Videos_935, Normal_Videos_940; blocks 1,8,32; 6 rows | mps; float32; cv5.0.0 | shortside_opencv_scale; single_offset_03/raw; shift 0 | 0.255675732 | 0.967500920 | 0.0% |
| [ucf_extraction_verify_20261008_124443](../../.local-backups/pre-refactor-20261008/outputs/setup/ucf_extraction_verify_20261008_124443/summary.json) | Shooting032, Normal924/935/940; blocks1,8,32; 12 rows | mps; float32; cv5.0.0; norm=imagenet | published; mean16/raw; shift0 | 0.302447368 | 0.954107182 | 0.0% |

## 저장된 추가 대조

- [현재 OpenCV와 PyAV의 RGB 픽셀 대조](../../.local-backups/pre-refactor-20261008/outputs/setup/ucf_decoder_compare_20261008_111543_761068/summary.json): 검사한 9개 프레임 모두 동일. 역사적 decoder 전체의 동등성을 확인한 것은 아니다.
- [OpenCV 4.7 격리 환경 대조](../../.local-backups/pre-refactor-20261008/outputs/setup/ucf_decoder_compare_20261008_112850_808724/summary.json): 검사한 9개 프레임 모두 동일. 기존 project dependencies는 변경하지 않았다.
- [동일 geometry CPU/MPS feature 차이](../../.local-backups/pre-refactor-20261008/outputs/setup/ucf_geometry_scan_20261008_111454/backend_comparison.json): 상대 L2 약 0.001566~0.001693. 공식 feature와의 잔차보다 작다.
- [visual weights 대조](../../outputs/setup/ucf_visual_weight_comparison.json): 152개 tensor 동일.
- [예외 4개의 마지막 row·실제 float32 값 검사](../../outputs/setup/ucf_float32_tail_diagnostic.json): 단순 half 값의 dtype 승격이 아니며 마지막 row는 비영/이전 row와 다르다.
- [Shooting032 전체 영상 최근접 검색](../../.local-backups/pre-refactor-20261008/outputs/setup/ucf_timeline_search_20261008_120702_731073/summary.json): 21681프레임 EOF·stride16의 1356개 raw feature 검사. 초반 row들은 16k 프레임에 가장 가깝지만 모든 질의가 그렇지는 않고 수치 오차가 커서 recipe를 확정하지 않는다.
- [최근접 검색 수치 재검산](../../.local-backups/pre-refactor-20261008/outputs/setup/ucf_timeline_search_20261008_120702_731073/arithmetic_validation.json): NumPy matmul 경고 때문에 float64 einsum/직접 차이 및 독립 norm 계산으로 재검산했다. 전 행/후보 finite·rank 동일·차이 5e-15 이내. 영상/encoder 재실행 없음. 향후 코드는 같은 안전한 산술과 finite 검사를 사용한다.
- [공식 test 경로 검증](../../outputs/setup/ucf_feature_mapping_validation_20261008.json): train/event 폴더와 중복되는 normal basename 때문에 upstream CSV의 상대 폴더를 사용하도록 보완했다. crop0/5 각각 290개 대응, 기존 report의 86개 영상 항목 input hash 일치.

## 범위와 다음 작업

기존 raw reference offset8/원본 입력/GT/weights/공식 baseline 결과는 변경하지 않았다. 이 진단의 1위 조건을 raw reference로 확정하거나 VAD 효율 결과로 사용하지 않는다. 진단에 custom GELU/attention/이미지 normalization을 쓴 실행은 원본 고정 모델의 sampler 성능 비교와 분리한다.

다음은 예외 4개와 일반 cohort의 원본 입력·추출 pipeline 차이를 설명하고, 원본 frame 생성/전처리/마지막 불완전 구간을 직접 확인하는 일이다. 저자의 원본 전체 extractor 또는 제작 당시 입력 프레임과 runtime 정보는 아직 없다. 질의 초안은 있지만 이메일/GitHub 메시지는 발송하지 않았다. Goal은 active이며 완료로 표시하지 않는다.

## 전체 영상 EOF 대조

[전체 290개 검사](../../outputs/setup/ucf_all_full_decode_20261008.json)를 실제 순차 decode로 완료했다. 286개는 이번 검사에서 새로 읽고, 예외 4개는 [직전 전체 decode](../../outputs/setup/ucf_float32_full_decode_20261008.json)의 영상/공식 feature hash를 다시 검증해 재사용했다. 전부 decoded/header/manifest frame 수가 일치했다.

| 공식 dtype | 영상 수 | 실제 N에 대한 official T | 해석의 한계 |
|---|---:|---|---|
| float16 | 286 | 모든 파일에서 `floor(N/16)` | 마지막 불완전 구간을 feature로 만들지 않는 규칙과 부합하지만 저장 행만으로 원본 선택/평균을 확정할 수 없다. |
| float32 | 4 | 모든 파일에서 `ceil(N/16)` | 마지막 구간 처리가 다르다는 증거다. 한 장·평균·padding 중 무엇인지 확인되지 않았다. |

Feature 벡터의 수치 대조는 전체 290개에 수행한 것이 아니다. 일반 cohort 7개와 예외 cohort 4개의 지정 구간만 encoding했다. 전체 EOF 검사는 길이·대응 관계만 확인했다. 실측 decode 시간은 추출/VAD end-to-end speedup으로 사용하지 않는다.

[통합 findings JSON](../../outputs/setup/ucf_extraction_findings_20261008.json)에 source·동일 조건 single/mean 비교·전체 EOF 결과·미확정 상태를 연결했다.

## 추가 encoder/배포 ZIP 대조

- [Float32 예외 4개 ImageNet normalization](../../.local-backups/pre-refactor-20261008/outputs/setup/ucf_extraction_verify_20261008_124443/summary.json): 공개 crop/비율 유지 fx-fy/CLIP-only 조건, 모든16개 offset/mean과 ±1 row shift를 동일 구간에서 검사했다. Best relative L2≈0.30245, pass0%이며 수치 일치를 얻지 못했다. 이 결과는 예외 4개도 같은 방식으로 추출했다고 확정하지 않는다.
- [TorchScript/Python encoder 비교](../../.local-backups/pre-refactor-20261008/outputs/setup/ucf_jit_encoder_compare_20261008_124929_925900/summary.json): cached RGB 9개를 공개 crop/비율 유지 fx-fy의 두 조건으로 처리한18개 동일 tensor, CPU float32에서 두 경로가 모두 선언한 tolerance를 통과했다. 최대 상대 L2=3.72735e-6, 최대 절대 오차=1.07288e-5. 공식 feature와의 잔차는 남았다. CPU 경로의 이 결과를 역사적 CUDA runtime 검증으로 확대하지 않는다.
- 원본 `clip.load(jit=True)`는 현재 PyTorch에서 제거된 `Node['value']` API 때문에 실패했다. 해당 실패 run은 보존했다. 로컬 진단 코드에서 원본 CPU device/dtype patch의 대상과 값을 유지하면서 typed `node.s/node.i` access로 읽는 API만 바꿨다. JIT module의 freeze도 parameter별로 수행했다. 원본 clip.py/model.py와 checkpoint는 수정하지 않았다.
- [공식 feature ZIP metadata/NPY header 검사](../../outputs/setup/ucf_feature_archive_metadata_20261008.json): 19500개 모두 header 확인, 다른 비-directory 파일이나 코드 없음. 전체 float16 18530개는 ZIP 날짜2023-01-11~13, float32 970개는2023-02-26/28 또는03-01이다. Test 286개 float16도1월, 예외4개 float32는 모두2월28일이다. 로컬 test290개 전부 archive member와 CRC·크기가 일치하여 이 저장 구분이 로컬 dtype 변경에서 생긴 것은 아니다.
- ZIP 날짜는 수정 가능한 timezone 없는 metadata다. 여러 저장 date cohort가 존재한다는 직접 증거와 여러 extraction pipeline/run을 사용했을 가능성이라는 추론을 구분한다. 실제 제작 시점·원본 프레임·평균/offset·runtime을 확정하지 않는다. Fingerprint는 member metadata의 hash이며13GB archive 전체의SHA라고 표기하지 않는다.

## 공개 fork source 추가 조사와 완료 감사

[Source audit](../../outputs/setup/ucf_public_extractor_source_audit_20261008.json): fork38개 default tree, 변경 Python/README145개를 hash 대조하여 읽었다. Community raw extractor는 있으나 원본 제작 provenance 또는 strict 수치 일치를 확보하지 못했다. 이 턴에는 외부 코드 실행이나 추가 CLIP inference를 하지 않았다. 기존 cached 후보를 대조했고, 느슨한 snapshot threshold가 확인된 numerical mismatch도 통과시킨다는 것을 확인했다.

[완료 조건 감사](EXTRACTION_COMPLETION_AUDIT.md)에 요구별 증거/누락과 같은 원본 제작 자료 부재가 세 goal 턴에서 지속된 이력을 기록했다. 사용자가 추가 원저자 자료가 없다고 답했으며 현재는 직접 자료 또는 외부 상태 변화가 필요한 blocked 상태다. 원본 재현 완료로 표시하지 않는다.

## 2026-10-08 오후 — float16 cohort 추출 방식 재구성

사용자가 이 goal(원본 추출 방식 탐색) 동안 Claude의 실행을 허용했다. VAD 평가·sampling 실험은 하지 않았다. 통합 근거는 [ucf_recipe_reconstruction_20261008.json](../../outputs/setup/ucf_recipe_reconstruction_20261008.json)에 있다.

**결론: float16 cohort(test 286개, 배포 ZIP 1,853개 영상)의 feature는 "frame 16k 한 장 → x86 libswscale YUV→RGB → `cv2.resize(fx=fy=256/240)` → `[16:240, 58:282]` crop(+flip) → CLIP ViT-B/16"으로 재구성된다.** 표본 block과 영상 2개 전체에서 상대 L2 0.0015~0.0018로, CPU/MPS 백엔드 간 차이(약 0.0016)와 같은 수준이다. 원저자 코드로 확인한 것은 아니다. 가장 긴 영상 4개(float32)는 미해결이다.

| 단계 | 근거 | 결과 |
|---|---|---|
| 시간축 | 기존 `per_block.csv`(112516, 114026) 재집계 | 움직임이 큰 block에서 d=0일 때만 뾰족한 최소(예: Abuse028 b8 0.13/0.044/0.17). Burglary005 b8은 16k와 16k+1 사이 장면 전환에서 전환 이전 쪽과만 일치 → 16장 평균 배제 |
| 꼬리 | [전체 EOF](../../outputs/setup/ucf_all_full_decode_20261008.json) | N%16==0인 float16 영상 20개 모두 T=N/16 → floor(N/16) 확정(floor((N-1)/16) 배제) |
| dtype 분리 | 배포 ZIP 19,500개 NPY header | 1,950개 영상 중 T≤1246은 전부 float16, T≥1274는 전부 float32, 겹침 0. test float32 4개 = test에서 가장 긴 영상 4개 |
| ffmpeg 이미지 저장 | [ucf_ffmpeg_frames_20261008_133653](../../.local-backups/pre-refactor-20261008/outputs/setup/ucf_ffmpeg_frames_20261008_133653/summary.json) | JPEG(기본, q1~q10) 0.21~0.29로 기각. PNG는 OpenCV와 픽셀 동일 |
| YUV→RGB | [ucf_yuv_conversion_20261008_134025](../../.local-backups/pre-refactor-20261008/outputs/setup/ucf_yuv_conversion_20261008_134025/summary.json) | 현재 swscale = SIMD 끈 C 경로(0.0417). accurate_rnd 0.074, full chroma 0.18, 정밀 BT.601 nearest 0.036/0.032, **x86 SIMD 재현 0.0159**(float 렌더러) |
| x86 + 실제 cv2.resize | [ucf_affine_search_20261008_134157_x86_float16](../../.local-backups/pre-refactor-20261008/outputs/setup/ucf_affine_search_20261008_134157_x86_float16/summary.json) `sanity` | 7개 영상 block 1/8/32: fp16 **0.00148**, fp32 0.00159 |
| 마지막 row 포함 | `ucf_extraction_audit_sources_20261008/check_float32_cohort_x86.py` | Abuse028 row 1/8/32/64/86/87에서 d=0일 때 0.0012~0.0016, 이웃 프레임은 0.008~0.19 |
| 수정한 추출기 전체 영상 | `ucf_extraction_audit_sources_20261008/check_reconstructed_extractor.py` | Abuse028 88 row 0.00168(최소 cos 0.999996), Normal003 176 row 0.00182 |
| float32 cohort | 같은 스크립트 | Shooting032/Normal940: d=-2~+2에서 0.24~0.37로 평평 → 시간축이 아닌 이미지 처리 차이. 미해결 |

해석상 주의:
- x86 재현은 FFmpeg `ff_yuv2rgb_c_init_tables`의 BT.601 계수(y=9539, vr=13075, ub=16525, ug=-3209, vg=-6660)와 16-bit `pmulhw` 연산을 numpy로 옮긴 것이다. 실제 x86 바이너리로 대조하지는 않았다.
- float 렌더러(0.016)와 실제 `cv2.resize`(0.0015)의 차이는 OpenCV 고정소수점 resize의 반올림 때문이다. 연속 affine 탐색(`ucf_affine_search_*`)은 이 사실이 확인되어 중단했다(ABORTED.txt).
- 허용 오차(atol 1e-4, rtol 1e-3) 통과는 여전히 기대하지 않는다. 공식 파일의 float16 저장과 CUDA/MPS 연산 차이만으로도 그 수준을 넘는다.

## 2026-10-08 오후 — 10-crop 검증과 float32 cohort 추가 조사

- **10-crop 배치 확인:** Abuse028 row 1/8/32/64에서 341×256 fx/fy 이미지의 10개 crop이 공식 `__0`~`__9` 전부와 상대 L2 0.001~0.002로 일치한다(`ucf_extraction_audit_sources_20261008/float32_crop_struct.py`).
- **float32 cohort(가장 긴 4개)에서 기각한 가설:** [ucf_float32_cohort_20261008_144502](../../.local-backups/pre-refactor-20261008/outputs/setup/ucf_float32_cohort_20261008_144502/summary.json)와 `float32_*.py` 일회성 스크립트로 검사했다.

| 가설 | 결과 (기준: x86 + fx/fy 한 장 약 0.30) |
|---|---|
| JPEG 저장 후 읽기 (OpenCV/Pillow q50~95, ffmpeg 기본/q2/q5) | 0.36~0.48, 더 나쁨 |
| 기하 (340×256 고정, 짧은 변 224~288) | 0.25~0.47, 일관된 개선 없음 |
| 시간 평균 (16·32·80프레임, 중앙 정렬, unit 평균, 픽셀 평균 이미지) | 0.14~0.41, 한 장보다 일관되게 낫지 않음 |
| Blur / 저해상도 후 확대 | 0.31~0.64, 더 나쁨 |
| 밝기·대비·감마·채도·색 범위·BT.709·R/B 교환·흑백 | 잔차 방향과 cos ≤ 0.32, 최적 크기로 적용해도 감소 ≤ 0.02 |
| crop 번호 뒤바뀜 | 공식 `__k`는 우리 crop k와 가장 가까움 |

- **관찰:** 같은 영상 안에서는 잔차끼리 cos 0.30~0.48로 공통 방향이 있다(평균 잔차를 더하면 0.30 → 0.23~0.26). 반면 영상 간 평균 잔차는 cos 0.04~0.11이다. 파이프라인 전체에 걸린 변환이 아니라 영상별 차이라는 뜻이다. Shooting032 row 1354는 frame 21662/21663 사이 장면 전환에서 전환 이후 쪽과 맞아 큰 시간 drift는 없다.
- **작업 가설(미검증):** 이 4개는 로컬 파일과 다른 사본(다른 인코딩)이나 다른 파이프라인으로 처리되었을 가능성이 높다. 공식 시간축 길이(21696, 108000, 108000, 36032)는 우리 N보다 조금 큰 N'에 floor(N'/16)를 적용한 것과도 같다. 로컬 자료만으로는 더 좁힐 수단이 없다.
- **평가 영향:** 4개 영상은 frame-level 평가 시간축의 24.7%(273,728프레임, GT 양성 210개)를 차지한다. raw full과 공식 metric의 차이는 이 영상들에서 크게 나올 수 있다.
