# Step 1 — UCF-Crime 자원 준비

2026-10-06. 사용자가 자원 경로와 baseline 입력 점검을 실행하여 통과했다. 다음 inference 단계는 [UCF_OFFICIAL_BASELINE.md](UCF_OFFICIAL_BASELINE.md)를 따른다. Codex는 프로젝트 코드를 실행하지 않았다.

현재 업데이트: 사용자 경로 점검(19:59:32 KST)과 baseline 입력 점검(20:05:51 KST)이 통과했다. Feature 290개 매핑·형식·유효값과 __0/__5 shape, GT 1,109,888개 위치의 공식 시간축 정렬, CPU 모델 strict loading을 확인했다. CUDA false/MPS true availability가 기록되었다. Forward/inference는 다음 단계이며 raw-video 좌표 규칙은 별도 검증 대상이다.

## 먼저 확보할 입력

| 입력 | 용도 | 현재 확인 |
|---|---|---|
| 공식 UCF-Crime CLIP feature 배열 | Baseline 입력. 공식 CSV의 290개 test 파일 필요 | `data/features/official/ucf/UCFClipFeatures/` 존재. 19,500개 .npy 경로 확인 |
| UCF-Crime pretrained VadCLIP checkpoint | UCF용 학습 완료 모델 | `data/checkpoints/vadclip/model_ucf.pth` CPU strict loading 성공 |
| 완성된 CLIP ViT-B/16 가중치 | 모델 생성·CLIP 추출 | `data/checkpoints/clip/ViT-B-16.pt` 기대 hash 일치, CPU 모델 생성/로딩 성공 |

이미 다운로드한 자원이 있으면 실제 경로를 재사용한다. 없다면 [VadCLIP 공식 README의 Setup 표](https://github.com/nwpu-zxr/VadCLIP#setup)에서 UCF-Crime 행의 CLIP(feature)와 Model(VadCLIP checkpoint)을 받는다. 이 CLIP 열은 feature 배포이며 CLIP encoder 가중치와 다르다. 현재 README에 OneDrive·Baidu 링크가 등록된 것은 확인했지만 다운로드 성공·배포 내용·test-only 분리 제공 여부는 아직 확인하지 않았다. Baidu feature 코드는 `7yzp`, model 코드는 `kq5u`다.

CLIP ViT-B/16 가중치 URL과 기대 SHA-256은 프로젝트의 `VadCLIP/src/clip/clip.py`에 있다. `.part` 파일의 이름만 바꿔 완료 처리하지 않는다. 완성본은 기대 hash와 비교한다.

처음에는 사용자가 features와 UCF checkpoint 미확보를 확인하여 저장 폴더를 준비했다. 현재는 해당 자원 파일이 존재하는 것을 확인했다. 파일 존재는 무결성·평가 성공을 뜻하지 않는다. 원래 feature 파일명과 category 구조를 유지하고, `__0.npy`를 `__5.npy`로 임의 변경하지 않는다.

공식 README에 등록된 직접 링크:

- [UCF features — OneDrive](https://stuxidianeducn-my.sharepoint.com/:u:/g/personal/pengwu_stu_xidian_edu_cn/Ea86YOcp5z9KhRFDQm9a8zwBcGiGGg5BuBJtgmCVByazBQ?e=tqHLHt)
- [UCF pretrained model — OneDrive](https://stuxidianeducn-my.sharepoint.com/:u:/g/personal/pengwu_stu_xidian_edu_cn/Eaz6sn40RmlFmjELcNHW1IkBV7C0U5OrOaHcuLFzH2S0-Q?e=x8wtVe)
- [CLIP ViT-B/16 encoder 가중치](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt)

OneDrive 링크 두 개는 웹 도구에서 접근할 수 없어 다운로드 가능 여부를 확인하지 못했다. 링크 폐기나 권한 문제로 단정하지 않는다. 사용자 브라우저에서 열어보고 실패하면 공식 README의 Baidu 대체 링크를 확인한다. 이번 작업에서 다운로드·압축 해제는 수행하지 않았다.

## 경로 설정

`configs/ucf_resources.json`은 현재 확인한 세 자원 경로로 갱신했다. 다른 위치로 이동한 경우에만 수정한다.

```json
{
  "features_root": "data/features/official/ucf/UCFClipFeatures",
  "ucf_checkpoint": "data/checkpoints/vadclip/model_ucf.pth",
  "clip_checkpoint": "data/checkpoints/clip/ViT-B-16.pt"
}
```

파일명만 비교하면 normal test 50개가 `Normal_Videos_event/`와 `Testing_Normal_Videos_Anomaly/`에서 중복으로 발견된다. 원본 패키지를 변경하지 않고 **공식 CSV의 바로 위 폴더명 + 파일명**으로 선택하도록 점검 코드를 수정했다. Test normal은 공식 CSV가 지정한 `Testing_Normal_Videos_Anomaly/`의 파일을 사용한다. 다른 폴더의 같은 이름은 `ignored_same_name_features`에 기록하며 mapping 중복으로 처리하지 않는다. 배열 내용이 서로 같은지는 확인하지 않았다.

- `features_root`: 압축 해제된 `.npy` feature가 들어 있는 상위 폴더. 하위 폴더를 재귀 검색한다.
- `ucf_checkpoint`: UCF용 완성된 `.pth` 또는 `.pt` 파일.
- `clip_checkpoint`: 완성된 ViT-B/16 `.pt` 파일.

절대 경로를 권장한다. 상대 경로는 현재 터미널 위치가 아니라 프로젝트 루트 기준으로 해석한다. 아래는 경로 형식만 보여주는 예시이며 실제 존재를 확인한 경로가 아니다.

```json
{
  "features_root": "/실제/feature/폴더",
  "ucf_checkpoint": "/실제/model_ucf.pth",
  "clip_checkpoint": "/실제/ViT-B-16.pt"
}
```

## 사용자 실행

### UCF feature ZIP을 wget으로 다운로드

**사용자 실행 결과: 아래 SourceUrl 단독 명령은 HTTP 403을 반환했다.** 현재 환경의 권장 후속 명령은 다음이다. 실제 성공은 아직 미검증이다.

```sh
bash scripts/setup/download_ucf_features.sh
```

이 스크립트는 원래 공식 공유 링크에 먼저 요청하여 서버가 설정하는 session cookie를 임시 파일로 받는다. 같은 cookie를 download.aspx 요청에 전달한다. 브라우저에서 원래 공유 링크가 UCFClipFeatures.zip 페이지를 열고 다운로드 버튼을 표시하는 것은 읽기 전용으로 확인했다. 파일 경로 요청의 403이 공유 접근 정보 누락 때문인지는 아직 가설이며, JavaScript 또는 별도의 다운로드 token이 필요하면 이 방식도 실패할 수 있다.

- 입력: 스크립트에 기록한 공식 feature 공유 링크와 파일 경로. 사용자 계정 cookie를 브라우저에서 추출하지 않는다.
- 출력: `data/features/official/ucf/UCFClipFeatures.shared.zip`. 다운로드 중에는 별도 `.shared.zip.part` 파일을 사용하여 이전 실패 파일과 분리한다.
- 임시 cookie는 실행 종료 시 삭제한다. 쿠키·응답 header를 로그에 출력하지 않는다.
- HTTP 성공 뒤 ZIP 유형과 central directory를 확인한 경우에만 완료 이름으로 옮긴다. 전체 CRC·압축 해제·feature 매핑은 수행하지 않는다.
- 다시 403이면 브라우저의 다운로드 버튼으로 시작된 실제 다운로드 링크 또는 요청이 필요하다. 사용자 브라우저에서 다운로드 링크를 복사하여 로컬 wget에 사용하며, 링크만으로 접근되지 않으면 해당 요청의 cookie/인증 여부를 확인해야 한다. 인증 정보를 채팅에 전달할 필요는 없다.

Codex는 이 스크립트를 정적으로 검토했으며 실행하거나 다운로드를 시작하지 않았다.

아래는 **403이 보고된 이전 단독 명령**과 URL 구성 근거다.

사용자가 제공한 OneDrive 화면 URL의 `id`를 해석하면 파일 경로는 `Documents/VadCLIP-AAAI2024/features/UCF-Crime/UCFClipFeatures.zip`이다. Microsoft의 [SharePoint 다운로드 링크 안내](https://learn.microsoft.com/en-in/answers/questions/5443776/link-to-excel-file-in-sharepoint-to-download-only)의 `download.aspx?SourceUrl=` 형식으로 변환했다. 이 호스트의 실제 익명 다운로드 성공은 미검증이며, 브라우저에서 열리는 것만으로 wget의 접근이 보장되지는 않는다.

현재 로컬에 `/opt/homebrew/bin/wget`, `/usr/bin/file`, `/usr/bin/zipinfo`가 존재한다. 프로젝트 루트에서 사용자가 실행한다.

```sh
wget -c --server-response --timeout=60 --tries=5 \
  -O data/features/official/ucf/UCFClipFeatures.zip \
  'https://stuxidianeducn-my.sharepoint.com/personal/pengwu_stu_xidian_edu_cn/_layouts/15/download.aspx?SourceUrl=https%3A%2F%2Fstuxidianeducn-my.sharepoint.com%2Fpersonal%2Fpengwu_stu_xidian_edu_cn%2FDocuments%2FVadCLIP-AAAI2024%2Ffeatures%2FUCF-Crime%2FUCFClipFeatures.zip'
```

`-c`는 서버가 지원하면 중단된 다운로드를 이어받는다. `-O`는 저장 파일을 지정한다. 이 명령은 다운로드만 수행하며 feature 압축 해제·모델 평가를 실행하지 않는다.

저장 위치: 프로젝트의 `data/features/official/ucf/UCFClipFeatures.zip`. 다운로드가 끝나면 파일 유형과 ZIP central directory를 확인한다.

```sh
file data/features/official/ucf/UCFClipFeatures.zip
zipinfo -t data/features/official/ucf/UCFClipFeatures.zip
```

`file`에서 ZIP으로 식별되고 `zipinfo`가 archive 정보를 정상 출력하는지 확인한다. 이는 전체 member의 CRC/내용 검증은 아니다. HTML 문서가 저장되거나 HTTP 401/403이 나오면 해당 주소가 wget의 비로그인 세션에서 파일을 제공하지 못한 상태다. 결과를 전달받아 공유 링크·브라우저 다운로드 요청에 맞춰 후속 명령을 준비한다. 실패한 HTML 파일에는 `-c`로 재개하지 않는다.

다운로드·확인 명령은 이번 작업에서 실행하지 않았다. 정상 ZIP을 확보한 뒤 압축 해제와 feature 290개 대응 점검으로 이어간다.

### 확보한 자원 경로 점검

프로젝트 루트에서 실행한다.

```sh
.venv/bin/python scripts/setup/check_ucf_resources.py --write-local-list
```

이 실행은 영상 디코딩·inference 없이 파일 경로, 원본 CSV/manifest 순서·label, 공식 parent 폴더를 유지한 feature 누락·중복·빈 파일, checkpoint hash, 패키지 metadata를 점검한다. CLIP hash 검증은 파일 전체를 읽는다. Feature/GT 배열과 모델은 로딩하지 않는다.

예상 출력:

- `outputs/setup/ucf_resource_check.json`: 매 실행의 점검 보고서.
- `outputs/setup/ucf_CLIP_rgbtest.local.csv`: feature 매핑이 모두 유효할 때만 생성. 공식 CSV 순서·label을 보존하고 실제 절대 경로를 기록한다.
- 자원이 미설정/누락되면 원인을 출력하고 종료 코드 1을 반환한다. 빈 feature 폴더·checkpoint `null` 상태에서 이는 예상 동작이다.
- 이전 local CSV가 존재해도 이번 점검 실패 시 갱신하지 않는다. 보고서의 `local_csv_written_this_run`을 확인한다.

원본 CSV·GT·영상·checkpoint는 수정하지 않는다. 파생 보고서와 local CSV만 생성·갱신한다.

## 확인할 항목과 다음 작업

1. `matched_feature_count`가 290이고 `feature_mapping_ok`가 true인지 확인한다.
2. `missing_features`, `duplicate_features`, `empty_features`, `missing_test_videos`가 비어 있는지 확인한다.
3. `checkpoints.clip.inventory_ok`가 true인지 확인한다. UCF checkpoint의 hash 기록은 모델 호환성 검증을 뜻하지 않는다.
4. `path_inventory_complete`가 true인지 확인한다. 이것만으로 Step 1 전체 완료가 되지는 않는다.
5. 실행 출력과 보고서를 바탕으로 feature shape/길이/유효값, 기존 GT의 영상별 정렬, 모델 로딩과 실제 장치를 확인하는 다음 점검을 준비한다. 현재 점검 코드는 그 부분을 수행하지 않는다.

기존 GT 생성 스크립트의 `__0.npy` 조건과 현재 CSV의 `__5.npy` 대응이 미검증이므로 GT 생성 스크립트를 바로 실행하여 기존 GT를 덮어쓰지 않는다.

## 다음 사용자 실행 — Feature/GT와 모델 로딩 점검

자원 경로 점검은 통과했으므로 다시 실행할 필요 없다. 프로젝트 루트에서 다음 명령을 실행한다.

```sh
.venv/bin/python scripts/setup/check_ucf_baseline_inputs.py --check-model
```

필요 입력은 현재 설정한 features와 두 checkpoint, 생성된 local CSV, test manifest, 기존 `gt_ucf.npy`다. 새 스크립트는 다음을 확인한다.

1. Local CSV의 영상 순서·label·폴더명·파일명을 다시 대조한다.
2. Test feature 290개가 `[T, 512]` 형식이고 floating point·NaN/Inf 없음인지 확인한다.
3. 같은 폴더의 `__0`과 `__5` variant shape를 비교한다. 값의 동일성을 요구하지 않는다.
4. Score 16배 반복에 해당하는 전체 길이와 GT 길이를 비교한다. Manifest annotation을 공식 GT 생성 코드의 `[start:end]` 규칙으로 영상별 복원하여 기존 GT와 비교한다. 이 비교는 official-feature 시간축의 검증이며 raw-video 좌표 규칙의 확정을 뜻하지 않는다.
5. 데이터 점검이 통과하면 numpy/torch/모델 의존성을 import하고, 두 로컬 checkpoint를 사용하여 VadCLIP을 CPU에서 생성·strict loading한다. CUDA/MPS availability를 기록한다. Forward pass·AUC 계산·영상 디코딩은 수행하지 않는다.

`CLIPVAD` 생성자에 선택적인 `clip_checkpoint` 인수를 추가하여 준비한 로컬 CLIP 파일을 명시적으로 로딩할 수 있게 했다. 기존 호출의 기본값·모델 구조·forward·가중치 해석은 유지한다. 이 변경을 포함한 실행 성공은 사용자 결과 확인 전까지 미검증이다.

예상 출력은 `outputs/setup/ucf_baseline_input_check.json`이며 영상별 feature 형식·GT offset·불일치와 모델 로딩 결과를 포함한다. 오류가 있으면 보고서를 남기고 종료 코드 1을 반환한다. GT·입력 feature·checkpoint는 수정하지 않는다.

확인할 값은 `data_checks_passed: true`, `gt_check.annotation_alignment_under_official_rule: true`, `model_check.ok: true`, `issues: []`다. 통과하면 작은 입력의 inference/window/padding 검증과 official-feature 전체 AUC 평가 준비로 이어간다. 실제 평가 장치와 score reporting 정책도 그 전에 정한다.
