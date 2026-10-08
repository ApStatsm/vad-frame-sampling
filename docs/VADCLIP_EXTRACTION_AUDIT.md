# VadCLIP 원본 feature 추출 방식 조사

조사일: 2026-10-08 (Asia/Seoul)

## 결론

> **2026-10-08 갱신:** 이후 수치 진단으로 float16 cohort(test 286개)의 방식을 재구성했다. 각 16프레임 구간의 첫 프레임(16k) 한 장을 x86 libswscale YUV→RGB, `cv2.resize(fx=fy=256/240)`, `[16:240, 58:282]` crop(+flip)으로 처리하면 상대 L2 약 0.0015로 일치한다. 아래 본문은 공개 자료 조사 당시의 판단이다. 최신 근거는 [실행 기록](EXTRACTION_DIAGNOSTIC_RUNS.md) 마지막 절에 있다.

공개 원본 자료에서 확인된 것은 standard CLIP ViT-B/16, 10-crop 및 16-frame 단위 feature sampling이다. **Feature 한 행이 단일 프레임 encoding인지, 16장의 encoding을 평균한 것인지는 확정하지 못했다.** 단일 프레임 가설과 16장 평균 가설을 모두 미확정으로 유지한다. 확인되지 않은 원본 encoder 입력 수로 효율 비교의 분모를 설정하지 않는다.

상단은 공개 자료 조사 결과다. 이후 실행된 수치 대조는 아래 추가 실험 절과 별도 실행 기록에서 구분한다. 원본 추출 방식 재현 완료를 뜻하지 않는다.

## 직접 확인한 근거

1. [저자 Issue #1 답변](https://github.com/nwpu-zxr/VadCLIP/issues/1#issuecomment-1873717061): 공개 REST API로 댓글 본문과 작성자 nwpu-zxr / author_association=OWNER를 확인했다. 저자는 “we also sample features every 16 frames”라고 답했다. 이 문장은 16-frame 시간 단위를 확인하지만, 구간 내 encoder 입력 수/선택 offset/평균 연산은 명시하지 않는다.
2. [CLIP 종류와 crop에 대한 저자 답변](https://github.com/nwpu-zxr/VadCLIP/issues/1#issuecomment-1873627266): standard ViT-B/16 및 10-crop을 명시한다. 10-crop과 시간축 16장은 별개다.
3. [공개 crop.py](https://github.com/nwpu-zxr/VadCLIP/blob/c41067f07d252efcda18008bea367886070c33b0/src/crop.py): 예제는 3개의 합성 이미지를 입력으로 만들고 각 이미지를 encode_image에 넣어 feature를 concatenate한다. Raw video read, 16-frame stride/offset 선택 또는 16장 feature 평균은 없다. 따라서 전달된 이미지 한 장을 encoding하는 연산은 확인되지만 배포 feature의 시간축 생성 규칙은 확인되지 않는다.
4. [추출 코드 추가 commit](https://github.com/nwpu-zxr/VadCLIP/commit/c41067f07d252efcda18008bea367886070c33b0): 변경 파일은 src/crop.py 하나다. 전체 7개 commit의 변경 파일 목록에서도 별도 raw temporal extraction 파일을 찾지 못했다.
5. [Issue #14의 16장 평균 설명](https://github.com/nwpu-zxr/VadCLIP/issues/14#issuecomment-2261922859)과 [Issue #17의 같은 설명](https://github.com/nwpu-zxr/VadCLIP/issues/17#issuecomment-2261930827)은 일반 사용자 zg-young(author_association=NONE)의 추정이다. 작성자도 불확실성을 명시하며 저자 확인은 없다. 이를 확정 근거로 쓰지 않는다.
6. [Issue #29](https://github.com/nwpu-zxr/VadCLIP/issues/29)는 한 장/16장 평균을 모두 시도했지만 배포 feature와 다르다는 사용자의 질문이다. 이번 조회의 댓글 1개는 다른 사용자의 질문이며 저자 답변은 없다. 사용자의 불일치 보고만으로 두 방법 모두 틀렸다고 확정할 수도 없다.
7. [Issue #35](https://github.com/nwpu-zxr/VadCLIP/issues/35)는 raw frame splitting 코드 요청이며 이번 조회의 댓글은 0개다. [Issue #33](https://github.com/nwpu-zxr/VadCLIP/issues/33)의 encoder 입력 수 질문도 댓글 0개다.
8. [논문 v1](https://arxiv.org/html/2308.11681v1), [v2](https://arxiv.org/html/2308.11681v2), [v3](https://arxiv.org/html/2308.11681v3) 및 [AAAI 게재본](https://ojs.aaai.org/index.php/AAAI/article/view/28423/28826)의 관련 방법·구현 설명에서 원본 16-frame 구간의 단일 프레임/평균 규칙을 확인하지 못했다. 모델의 시각 prompt feature 평균 및 학습용 feature 축소는 raw encoder 입력의 시간 평균과 구분한다.

## 확인 범위

- 공식 main: c41067f07d252efcda18008bea367886070c33b0.
- 공식 저장소의 recursive tree, 유일한 main branch, 빈 tag 목록, 전체 7개 commit 상세 변경 파일.
- Open/closed issue 36개, 전체 공개 issue 댓글 69개 및 그중 OWNER 댓글 9개. 페이지별 API와 전체 댓글 endpoint로 누락 여부를 대조했다.
- 관련 공개 저장소 Roc-Ng/VadCLIP, AVadCLIP, VAR, XDVioDet, XD-Violence의 recursive 파일 목록. 현재 공개 목록에서 원본 VadCLIP과 연결되는 완전한 시간축 feature 추출 파일을 추가로 찾지 못했다. 다른 연구의 알고리즘을 원본 recipe로 간주하지 않는다.
- 로컬 crop.py/dataset.py/tools.py 및 과거 사용자 probe JSON 2개 읽기. 두 probe 모두 success=false, videos=[]이며 feature 경로 FileNotFoundError로 끝났다. 실제 수치 대조 증거가 없다. 사용자가 취소한 probe 코드나 안내를 복원하지 않았다.

공개 자료가 없다는 결론은 위 확인 범위에 한정한다. 비공개 코드의 존재/내용은 알 수 없다.

## 원본 확정에 필요한 정보

원본 extraction 코드 또는 저자의 명시적인 확인으로 다음을 확보해야 한다.

- 구간당 image encoder 입력: 한 장 또는 16장 전체.
- 한 장이면 원본 frame offset과 시작 index.
- 여러 장이면 평균 전/후 정규화, 평균 대상 및 저장 dtype.
- Decoder/FPS 처리, 구간 경계, tail 처리 및 실제 preprocessing.

Feature shape=[T,512]와 T가 raw frame 수의 약 1/16이라는 사실만으로 단일 프레임/평균을 구분할 수 없다. 재현 대조에서 특정 가설의 수치가 충분히 일치하면 그 데이터에 대한 강한 근거를 얻을 수 있지만, 다른 preprocessing이 섞인 상태의 단순 상대 순위는 원본 알고리즘 확정이 아니다.

## 저자 질의 초안 — 발송하지 않음

For the released UCF-Crime and XD-Violence CLIP features, does each saved 512-dimensional row come from (A) encoding one selected frame in each 16-frame block, or (B) encoding all 16 frames individually and averaging their features? If A, which frame offset and starting index do you use? If B, are individual features normalized before averaging? Could you share the original video-to-feature extraction script, including preprocessing and handling of incomplete final blocks?

## 조사 파일과 실행 범위

GitHub API 원본 조회 JSON은 /private/tmp/vadclip_recipe_audit_20261008/에 저장했다. API 조회·파일 읽기만 수행했으며 raw 영상 decode, CLIP/VadCLIP inference, 성능 평가, 새로운 feature 대조는 실행하지 않았다. 연구 코드·설정·기존 결과는 변경하지 않았다. 이메일/GitHub 질문은 발송하지 않았다.

## 2026-10-08 추가 조사 — GitHub 외부 자료

### 새로 찾은 저자 측의 직접 자료

| 자료 | 자료 자체에서 확인한 내용 | 원본 VadCLIP에 대한 근거 수준 |
|---|---|---|
| [STPrompt, arXiv 2408.05905v1, §4.2](https://arxiv.org/html/2408.05905v1#S4.SS2) | UCF-Crime에서 16프레임 중 1장을 처리한다고 명시한다. 같은 문단은 frozen CLIP으로 영상 프레임 feature를 추출한다고 설명한다. | Peng Wu, Xuerong Zhou 등 원 논문과 겹치는 주요 저자의 후속 연구다. 원본 배포 feature 생성 규칙을 직접 기술한 문장은 아니다. |
| [Open-Vocabulary Video Anomaly Detection, arXiv 2311.07042v2, §4.1](https://arxiv.org/html/2311.07042v2#S4.SS1) | Implementation Details에서 각 영상의 16프레임 중 1장을 처리한다고 명시한다. | 원 논문의 주요 저자가 참여한 관련 연구의 단일 프레임 처리에 대한 직접 근거다. 원본 VadCLIP 추출 코드로 간주하지 않는다. |
| [AVadCLIP, arXiv 2504.04495v1, §3.2/§4.2](https://arxiv.org/html/2504.04495v1#S4.SS2) | §3.2는 sampled video frames의 CLIP encoding, §4.2는 XD-Violence에서 16프레임당 1장 선택을 명시한다. | Peng Wu, Guansong Pang 등 원 논문의 여러 저자가 참여한 후속 연구다. UCF 원본 배포 feature의 한 장/평균을 직접 확정하지 않는다. |
| [특허 CN118537775A, 중국어 원문, 단계 S101](https://patents.google.com/patent/CN118537775A/zh) | 16프레임마다 한 장을 뽑아 image encoder에 입력한다고 명시한다. 발명자는 吴鹏/Peng Wu, 周学荣/Xuerong Zhou, 王鹏/Peng Wang, 张艳宁/Yanning Zhang으로 원 논문의 저자들과 겹친다. | ST-VadCLIP이라는 시공간 확장 방법에 대한 문헌이다. 원본 AAAI VadCLIP의 released feature 제작 명세와 동일하다는 확인은 없다. 법적 상태에 대한 판단은 하지 않는다. |

### 추가 확인한 사이트와 제한

- [AAAI/Underline 원본 발표 페이지](https://underline.io/lecture/92052-vadclip-adapting-vision-language-models-for-weakly-supervised-video-anomaly-detection): 발표 페이지와 Slides 링크는 존재하나 이번 도구의 Slides 조회가 실패했다. 발표 슬라이드 내용까지 검토했다고 표시하지 않는다.
- [Guansong Pang의 저자 출판 목록](https://www.guansongpang.com/publications): 원 논문 항목은 arXiv 및 공식 저장소로 연결한다. 별도 raw extraction 설명을 확보하지 못했다.
- 영어·중국어 검색으로 Tencent Cloud의 논문 소개, CSDN/Zhihu 검색 결과, DeepWiki 및 관련 논문을 탐색했다. [Tencent 소개](https://cloud.tencent.com/developer/article/2374666)는 모델 구성과 결과 설명 위주이며 원본 frame offset/16장 평균/decoder 설정의 추가 원본 명세는 확보하지 못했다. 해설과 자동 생성 문서는 탐색 단서로만 취급하고 알고리즘 확정 근거로 쓰지 않는다.

### 판단 갱신

기존 결론인 **원본 배포 feature의 exact recipe 미확정**은 유지한다. 다만 저자 측 외부 자료에 명시적인 단일 프레임 선택 설명이 여러 건 있으므로, 원본 방식의 재현 가설로는 **16장 모두 encoding 후 평균보다 16프레임마다 한 장 선택을 우선 조사하는 것이 더 타당하다는 추론**을 얻었다. 이 우선순위는 원본 방식 확인 완료나 수치 재현 성공이 아니다.

선택 frame offset, crop pixel 처리, decoder/FPS, tail 및 저장 precision이 여전히 원본 배포 feature와 대조되지 않았다. 현재 provisional offset 8이나 mean_block 설정을 이 외부 조사만으로 자동 변경하지 않았다. 새로운 프로젝트 코드·feature 추출·모델 실행은 수행하지 않았다.


## 2026-10-08 추가 — 사용자 승인 goal의 실제 수치 검증

사용자가 “원본에서 어떻게 프레임 추출해서 피쳐 추출했는지 검증해줘. 이거 실행하는 동안에만 코드 알아서 실행해.”라고 허용하여 이 범위에서 실제 decoding/CLIP 추출/비교를 수행했다. 과거 절의 미실행 기록은 해당 조사 당시 상태다.

[실행별 증거](EXTRACTION_DIAGNOSTIC_RUNS.md)와 [통합 findings](../outputs/setup/ucf_extraction_findings_20261008.json)에 조건과 결과를 저장했다. 같은 구간에서 공개 crop의 상대 L2 약0.139가 aspect-preserving fx/fy linear resize에서 약0.042로 줄었다. 단일 offset0은 16장 평균보다 가까웠고 추가 test 영상 4개에서도 같은 고정 조건을 지지했다. 하지만 cosine 0.99912만으로 재현이라고 하지 않으며 선언한 수치 tolerance 통과는 0%다. 어떤 고정 후보도 수치 일치를 얻지 못했다.

전체 290개 영상의 EOF 프레임 수와 official T를 대조했다. 모두 실제 decoded/header frame 수가 일치했고 286개 float16은 floor(N/16), 4개 float32는 ceil(N/16)이었다. float32 4개는 Shooting032, Normal924/935/940이며 같은 pipeline으로 생성됐는지 확인되지 않았다. 벡터 encoding 대조는 일반 7개/예외 4개의 지정 구간에 한정된다.

Official model과 local CLIP의 visual weights 152개는 모두 동일했다. 검사한 OpenCV/PyAV RGB frame 9개도 pixel까지 일치했다. 현재 CPU/MPS·float16/float32·OpenCV 4.7/5.0·보간/rounding·채널/이미지 normalization·GELU/attention·JPEG 가설으로 남은 오차를 해결하지 못했다. 역사적 다른 runtime·입력 영상/프레임 제작·공개되지 않은 후처리까지 모두 배제한 것은 아니다.

최종 판단은 원본 recipe 미확정이다. `16k`번째 한 장은 수치/관련 연구가 지지하는 우선 가설이며 확정된 원본 명세는 아니다. 기존 raw offset8 가정을 자동 변경하거나 탐지/효율 실험으로 넘어가지 않았다. 저자 질의 초안은 발송하지 않았다.


### 추가 배포 파일·encoder 경로 확인

공식 ZIP의19500개 NPY header에서18530개 float16/970개 float32를 직접 확인했다. Float16 파일 ZIP 날짜는2023년1월, float32 파일은2~3월로 분리된다. 로컬 test290개 파일의 CRC·크기는 ZIP과 모두 같고, 예외4개 float32는 배포 ZIP에서도2023-02-28 날짜였다. 이 metadata는 여러 제작/저장 run 가능성을 지지하지만 변경 가능한 ZIP 날짜만으로 역사적 extraction 시점이나 별도 recipe를 단정하지 않는다. ZIP에는 NPY 외 extractor/문서 파일이 없었다.

현재 Python encoder와 체크포인트에 저장된 TorchScript encoder를 같은18개 전처리 tensor로 CPU float32에서 대조한 결과 두 경로는 tolerance를 통과했다(최대relL2≈3.7e-6). 따라서 검사한 CPU 조건의 forward 구현 차이는 현재 residual을 설명하지 못한다. 역사적 CUDA/프레임 제작 경로의 동일성을 증명한 것은 아니다. 예외4개 ImageNet normalization 추가검사도 불일치다. 최신 결과와 source는 [실행 기록](EXTRACTION_DIAGNOSTIC_RUNS.md)에 있다.


### 추가 fork source와 최종 미완료 감사

공개fork38개 default branch/변경Python·README145개까지확대조사했다. Communityraw extractor를찾았으나원저자제작코드/수치parity근거를확보하지못했다. 구현별조건차이와cachedhypothesis불일치는 [완료감사](EXTRACTION_COMPLETION_AUDIT.md)에기록했다. 추가원저자자료가없다는사용자답변및세goal턴간같은제작증거부재를확인해원본검증은blocked로기록한다. 이조사범위밖코드의부재를단정하거나원본확인완료라고표시하지않는다.
