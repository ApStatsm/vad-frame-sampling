> **Archive (2026-10-08 정리)**: 추출 방식 탐색 기록이다. 결론은 [UCF_RAW_REFERENCE.md](../UCF_RAW_REFERENCE.md)에 통합했다. 언급된 진단 스크립트는 `git show ucf-raw-reference-v1:scripts/<file>`로, 진단 run 폴더는 로컬 `.local-backups/pre-refactor-20261008/`에서 확인한다.

# VadCLIP 원본 추출 검증 — 완료 조건 감사

2026-10-08. 목표는 원본 raw video에서 프레임을 선택하고 feature를 만드는 실제 방식을 확인하는 것이다. 가장 가까운 후보나 자체 reference를 만드는 것으로 목표를 줄이지 않는다. **원본 recipe는 미확정이며 완료 조건을 통과하지 않았다.**

## 요구 조건과 현재 증거

| 요구 조건 | 확인한 실제 증거 | 판정 |
|---|---|---|
| Feature 한 행의 원본 encoder 입력 수: 한 장/16장 평균 | 원본 crop 예제는 전달받은 이미지별 encoding만 보여준다. 저자는16-frame feature sampling을 명시하지만 구간 내 입력 수는 명시하지 않는다. 일반7개 영상 지정 구간에서 offset0 한 장이 평균보다 가깝지만 numerical pass0%다. | 미확정. 원본 입력 수를 순위로 선언할 수 없다. |
| 원본 sampling 시작 index/offset/시간 대응 | offset0~15와 ±1 row shift를 검사했다. 근접 프레임이 높은 similarity를 가지며 불일치는 지속된다. Shooting032 전체 raw feature 최근접도 모든 질의에서16k가 최근접인 것은 아니다. | 미확정. |
| 원본 video/frame 버전·decoder·이미지 preprocessing | 현재 영상 hash/EOF 검증, 검사한 OpenCV/PyAV RGB9개 동일. 고정 resize→aspect/fx-fy resize에서 오차 감소하지만 원저자 입력과 직접 대조하지 못했다. | 일부 현재 입력 검증 완료; 역사적 원본 pipeline 미확정. |
| 실제 image encoder/checkpoint/precision | 저장 weights의 visual tensor152개 일치. 동일18개 CPU float32 이미지에서 saved JIT/Python 경로 tolerance pass. 여러 precision/attention/activation 가설 불일치. | 현재 encoder 검증 완료; 배포 feature 제작 당시 실행 환경까지 확인된 것은 아니다. |
| 전체 구간 수·마지막 불완전 구간 | 전체290개 EOF/헤더/manifest 일치. Float16 286개는floor(N/16), float32 4개는ceil(N/16). Tail row는 영벡터/단순 이전 row 복제가 아니다. | 길이 관계 확인. 마지막 row의 실제 원본 입력·padding/평균 규칙은 미확정. |
| 고정된 원본 recipe로 공식 feature 수치 재현 | 비교32개 실행, 일반7개/예외4개 지정 구간. Best 일반 표본 약relL2 0.042, pass0%. 전체290개 vector 재현은 완료되지 않았다. | 미충족. Cosine≈0.999만으로 통과시키지 않는다. |
| 실제 제작 extractor/확인 가능한 provenance | 공식 main/이력/댓글/논문/관련 저자 자료와 배포 ZIP 확인. ZIP19500개는 NPY만 존재. 추가로 fork38개의 default tree 및 변경 Python/README145개 확인. 원저자 제작 코드나 strict reproduction 근거를 확보하지 못했다. 사용자도 추가 자료가 없다고 답했다. | 직접 증거 누락. 조사 범위 밖 private/deleted/non-default 코드의 부재를 단정하지 않는다. |

위 요구 조건은 [수치 실행 기록](EXTRACTION_DIAGNOSTIC_RUNS.md), [공개 자료 조사](VADCLIP_EXTRACTION_AUDIT.md), [통합 findings](../../outputs/setup/ucf_extraction_findings_20261008.json), [새 fork source audit](../../outputs/setup/ucf_public_extractor_source_audit_20261008.json)에 연결된다. UCF 자료의 부분 검증을 XD-Violence 원본 recipe 확인으로 확대하지 않는다.

## 추가 공개 코드 조사 결과

38개 fork의 default branch tree는 모두 응답했고 truncated/error는 없었다. 변경된 Python/README145개를 내려받아 blob SHA를 확인했으며 decoding 표시가 있는 파일은 내용까지 읽었다. 다른 branch와 전체 인터넷을 모두 조사했다는 주장은 하지 않는다. 외부 코드는 import/실행하지 않았다.

| 공개 코드 | 실제 처리 | 원본 확정 근거로 사용할 수 없는 이유 |
|---|---|---|
| [Wedsonlin crop.py](https://github.com/Wedsonlin/VadCLIP/blob/main/src/crop.py) | ViT-B/16, 1-based counter의 stride16/shift8. Zero-based frame7+16k 한 장; fixed340×256 crop; half 저장. | 사용자 추가 code다. Shift8의 이름만으로 우리 zero-based offset8 또는 원본 offset이라고 하지 않는다. |
| [Wedsonlin snapshot test](https://github.com/Wedsonlin/VadCLIP/blob/main/tests/test_crop_video_features.py) | XD fixture/custom crop filename에 mean cosine>0.78, mean abs error<0.22 검사. | 원저자 snapshot provenance와 strict numerical agreement를 확인하지 못했다. 같은 frame/crop 가설의 기존 UCF 캐시는relL2≈0.248/pass0%인데도 영상별로 이 두 느슨한 threshold를 만족한다. 해당 fork 자체나 XD fixture를 실행한 결과로 표시하지 않는다. |
| [DrKolaman extractor](https://github.com/DrKolaman/VadCLIP/blob/main/src/generate_ucf_features.py) | ViT-B/32, 모든 raw frame 개별 encoding, CLIP-only preprocess. | 원본 ViT-B/16/16-frame 시간 단위와 다르다. |
| [Jiisou MobileCLIP extractor](https://github.com/Jiisou/VadCLIP/blob/main/src/extract_mobileclip_features.py) | 다른 backbone, 모든 segment frame encode→평균→최종 L2. | 원본 encoder·feature norm 조건과 다르다. |
| [jinmang2 wsad extractor](https://github.com/jinmang2/wsad/blob/dev/src/features/clip.py) | OpenCLIP Laion pretrained, per-frame→기본32 segment 평균. | 원본 OpenAI checkpoint/fullT recipe와 다르며 original feature 수치 parity 근거를 확인하지 못했다. |
| [VadCLIP-TTA enhanced extractor](https://github.com/yufengHenry/VadCLIP-TTA/blob/main/extract_enhanced_features_simple.py) | 기존 NPY features를 로드해 enhancement. | raw frame 생성/CLIP 최초 추출 코드가 아니다. |

Wedsonlin의 model source blob은 원본과 다르지만, docstring 제거 후 visual class/image method/build_model/transform AST는 동일했다. Image crop RGB 변환 순서도 검사한9개 cached frame에서 pixel 동일했다. Cached comparison의 계산 환경과 fork 실제 실행은 구분한다.

## Blocking audit

동일한 조건이 세 goal 턴에서 이어졌다: (1) 실제 수치·EOF 검증 후 원본 제작 code/runtime 부재, (2) JIT/ZIP 검증 후 원본 frame/preprocess 이력 부재, (3) 공개 fork source 추가 조사 후 같은 직접 제작 증거 부재. 마지막 턴에는 사용자가 추가 자료가 없음을 명시했다.

각 턴의 조사에서 실제 진전은 있었다. 그러나 현재 남은 원본 확인을 가능하게 하는 직접 증거 또는 수치 일치는 없고, 동일 실험을 다시 실행하거나 근거 없는 전처리 파라미터를 계속 sweep하는 것은 그 누락을 해결하지 못한다. 원본 script/실제 제작 입력·환경이 확보되거나 그러한 자료를 제공하는 외부 상태가 바뀌어야 이어서 확정할 수 있다. 목표를 완료로 표시하지 않으며 상태는 blocked로 기록한다.

## 이어서 검증할 최소 직접 자료

- 원저자가 배포 feature를 생성할 때 사용한 전체 raw-to-feature extractor와 runtime/encoder 조건.
- 특정 공식 vector(예: Abuse028__5.npy row1)에 실제 입력한 frame index·원본 frame 또는 encoder 입력 tensor와 대응 관계. 하나의 대조로 원인을 나눈 후 여러 독립 영상·구간과 tail까지 확대 검증한다.
- Float16/floor 그룹과 float32/ceil 그룹의 제작 경로가 달랐는지에 대한 설명 또는 각각의 extractor. ZIP 날짜는 actual extraction 날짜로 단정하지 않는다.

저자 질의 초안은 준비됐지만 발송하지 않았다. 외부 메시지 발송은 사용자 명시적 승인 없이 수행하지 않는다. 기존 raw reference offset8은 provisional로 보존하고 sampling/VAD 성능 결과를 새로 만들지 않았다.

## 2026-10-08 갱신 — blocked 해제, float16 cohort 재구성

Claude 세션에서 실행 허용을 받아 재개했다. 위 표의 "미확정" 항목 중 float16 cohort(test 286개)에 대해서는 다음이 수치로 판별됐다. 자세한 근거는 [실행 기록](EXTRACTION_DIAGNOSTIC_RUNS.md)의 마지막 절과 `outputs/setup/ucf_recipe_reconstruction_20261008.json`에 있다.

| 요구 조건 | 갱신된 판정 |
|---|---|
| Encoder 입력 수 | 구간당 한 장. 16장 평균 배제(V자 곡선, 장면 전환 사례) |
| 시작 index/offset | zero-based frame 16k(offset 0) |
| Decoder/전처리 | x86 libswscale YUV→RGB(OpenCV CAP_FFMPEG on x86) + `cv2.resize(fx=fy=256/240)` + `[16:240, 58:282]` + flip |
| Encoder/precision | ViT-B/16, fp16이 fp32보다 약간 가까움(0.00148 vs 0.00159) |
| 구간 수·꼬리 | float16: floor(N/16), 꼬리 버림. float32: ceil(N/16) |
| 수치 재현 | 표본·영상 2개 전체에서 상대 L2 약 0.0015~0.0018(백엔드 noise 수준). 290개 전체는 raw full 실행에서 확인 예정 |

남은 미해결: 가장 긴 영상 4개(float32 cohort)의 이미지 처리, 원저자 코드 확인. 모서리 crop(1~4, 6~9)은 검증하지 않았다.

추가(같은 날): 10개 crop 배치 모두 공식 파일과 일치함을 확인했다. float32 cohort는 JPEG·기하·시간 평균·blur·사진 보정·crop 순서 가설을 모두 기각했다. 잔차가 영상별로 달라 다른 사본/파이프라인으로 추정하며, 로컬 자료로는 더 확인할 수 없다(평가 시간축의 24.7%).
