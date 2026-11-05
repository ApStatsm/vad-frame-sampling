# 실행 환경 준비

새 컴퓨터에서 이 저장소를 받아 기존 결과(official AUC 88.02%)를 재현할 때까지의 절차다. 모든 명령은 프로젝트 루트에서 실행한다.

## 1. Python 환경

기존 환경은 Python 3.10.0, PyTorch 2.8.0, macOS arm64(MPS)다. 새 환경 설치는 아직 검증하지 않았다.

```sh
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Raw 영상 처리에는 `ffmpeg`/`ffprobe` CLI가 필요하다(macOS: `brew install ffmpeg`). Official feature 평가만 할 때는 필요 없다.

## 2. 대용량 자원 확보

Git에는 들어 있지 않다. 모두 `data/` 아래에 둔다.

```text
data/
├── UCF-Crime/
│   ├── Anomaly-Videos-Part-{1,2,3,4}.zip, Testing_Normal_Videos.zip
│   ├── extracted/                    # 위 ZIP을 압축 해제한 폴더
│   └── test/                         # manifest·split은 Git에 있음, 영상 symlink는 3단계에서 생성
├── features/official/ucf/UCFClipFeatures/<배포 폴더>/<video>__<crop>.npy
└── checkpoints/
    ├── clip/ViT-B-16.pt
    └── vadclip/model_ucf.pth
```

| 자원 | 용도 | 받는 곳 |
|---|---|---|
| Official CLIP features (약 13GB ZIP) | Official baseline 입력, raw 추출 대조 | [VadCLIP README Setup 표](https://github.com/nwpu-zxr/VadCLIP#setup)의 UCF-Crime 행 "CLIP" 열 (OneDrive, Baidu 코드 `7yzp`) |
| `model_ucf.pth` | UCF용 pretrained VadCLIP | 같은 표의 "Model" 열 (OneDrive, Baidu 코드 `kq5u`) |
| `ViT-B-16.pt` | CLIP encoder | [OpenAI 배포 URL](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt) (`VadCLIP/src/clip/clip.py`와 같음) |
| UCF-Crime 원본 영상 ZIP 5개 | Raw-video 평가 | UCF-Crime 공식 배포처 ([UCF CRCV 프로젝트 페이지](https://www.crcv.ucf.edu/projects/real-world/)). 이 링크의 현재 동작은 확인하지 않았다 |

- Feature ZIP은 `bash scripts/setup/download_ucf_features.sh`로 받을 수 있다. 공유 링크의 session cookie를 받아 `data/features/official/ucf/UCFClipFeatures.shared.zip`에 저장한다. 이 컴퓨터의 feature ZIP이 이 이름이다. 브라우저 없이 바로 받는 `download.aspx` 단독 요청은 HTTP 403이 났다. 스크립트도 실패하면 브라우저에서 받는다.
- Feature ZIP은 `data/features/official/ucf/` 안에서 압축 해제한다. 원래 폴더 구조와 파일명을 바꾸지 않는다. Normal test 50개는 두 배포 폴더에 같은 이름으로 있는데, 점검 코드가 공식 CSV의 바로 위 폴더명으로 구분한다.
- `ViT-B-16.pt`는 완성 파일이어야 한다(`.part` 아님). SHA-256 `5806e77c…`가 점검에서 확인된다.
- 영상 ZIP은 `data/UCF-Crime/extracted/`에 압축 해제한다. ZIP 원본도 지우지 않는다(3단계에서 CRC를 대조한다).

**팀 안에서 받는 경우**: 상민의 `data/` 폴더를 symlink를 유지한 채 그대로 복사하면 다운로드와 3단계의 `prepare_ucf_test.py`를 건너뛸 수 있다(`rsync -a`처럼 symlink를 보존하는 방식). Official feature만 평가하려면 `features/`와 `checkpoints/`만 있어도 된다.

## 3. 입력 점검

경로는 `configs/ucf_resources.json`에 프로젝트 루트 기준 상대 경로로 들어 있다. 위 배치를 따랐다면 수정할 필요가 없다.

```sh
# (원본 영상을 직접 받은 경우만) test 290개 symlink·manifest 생성과 ZIP CRC·ffprobe 점검
.venv/bin/python scripts/setup/prepare_ucf_test.py

# 경로·feature 대응·checkpoint hash 점검, 이 컴퓨터용 feature 목록 생성
.venv/bin/python scripts/setup/check_ucf_resources.py --write-local-list

# Feature 배열·GT 정렬 점검과 CPU 모델 로딩
.venv/bin/python scripts/setup/check_ucf_baseline_inputs.py --check-model
```

| 단계 | 통과 기준 |
|---|---|
| `check_ucf_resources.py` | `Feature files matched: 290/290`, `Path inventory complete: True` |
| `check_ucf_baseline_inputs.py` | `Data checks passed: True`, `Model loading passed: True` |

결과는 `outputs/setup/ucf_resource_check.json`, `ucf_baseline_input_check.json`, `ucf_CLIP_rgbtest.local.csv`에 저장된다. **이 세 파일은 Git에 올라가 있지만 내용은 실행한 컴퓨터의 절대 경로와 hash다.** 각자 다시 만든 파일은 커밋하지 않는다.

## 4. Official baseline으로 재현 확인

```sh
.venv/bin/python scripts/evaluate_ucf_official.py --mode full --device mps
```

- MPS 기준 약 30초. `--device`는 `cpu` / `cuda` / `mps`이고, 실패해도 다른 장치로 자동 전환하지 않는다.
- 끝에 출력되는 `Report: outputs/baseline/<run ID>/summary.json`을 확인한다.
- 통과 기준: `success: true`, 290개 영상, branch1 AUC 0.8802 / AP 0.3356 / Ano-AUC 0.7023, branch2 AUC 0.8569, `fine_grained.mean_map` 0.0668.

선택 사항으로 `--mode smoke`는 7개 사례(짧은 anomaly·normal, 긴 영상과 256-token 경계 prefix)만 돌려 실행 오류를 빨리 확인한다. Benchmark 결과가 아니다.

## 5. Raw-video reference (선택)

원본 영상과 ffmpeg가 있어야 한다. MPS 기준 약 25분.

```sh
.venv/bin/python scripts/evaluate_ucf_raw.py --mode full --device mps \
  --official-run outputs/baseline/<4단계에서 만든 run ID>
```

`configs/ucf_raw_reference.json`의 기본 비교 run(`ucf_full_20261105_222638`)은 점수 NPZ가 Git에 없으므로, 다른 컴퓨터에서는 4단계에서 만든 run을 `--official-run`으로 지정한다. 통과 기준은 branch1 AUC 0.8719 근처이고, 추출 방식과 출력은 [UCF_RAW_REFERENCE.md](UCF_RAW_REFERENCE.md)에 있다. 다른 장치에서는 feature가 bit 단위로 같지 않을 수 있다(CPU와 MPS 사이 상대 L2 약 0.0015).
