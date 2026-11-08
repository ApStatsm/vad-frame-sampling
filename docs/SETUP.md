# 설치와 입력 준비

이 문서는 사용자 실행용입니다. 설치, import, 테스트와 모델 실행은 작성 과정에서 수행하지 않았습니다.

## 정상 동작하는 Vast PyTorch 템플릿

1. 서버 터미널에서 현재 Python/GPU/패키지를 확인합니다.

```bash
which python
nvidia-smi
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda); x=torch.ones(2,2,device='cuda'); print(x@x)"
python -m pip freeze > initial-environment.txt
```

2. 현재 프로젝트 폴더에서 공통 패키지를 설치합니다. `requirements.txt`는 torch를 재설치하지 않습니다.

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m pip check
```

`opencv-python`, `opencv-contrib-python`, headless 계열은 cv2 namespace를 공유합니다.
한 종류를 유지해야 합니다. 기존 환경에 다른 OpenCV 배포판이 있다면 설치 목록을 확인하고
프로젝트 전용 환경에서 정리하세요. 드라이버를 apt로 교체하지 마세요.

3. 시스템 ffmpeg가 없으면 Ubuntu에서 설치합니다.

```bash
apt-get update
apt-get install -y ffmpeg
```

4. 코드가 들어 있는 이 폴더를 유지한 채 실행합니다. `VadCLIP/`, `configs/`, 작은 GT 자원이 필요하므로
일반 wheel만 다른 위치로 복사하는 배포는 지원하지 않습니다.

## 새 Python 환경

Python 3.10~3.13을 대상으로 작성했고 실제 새 환경 설치는 미검증입니다.
RTX 5090은 Blackwell 지원 CUDA wheel과 호스트 드라이버가 필요합니다.
공식 PyTorch는 2.7에서 CUDA 12.8 Blackwell 지원을 도입했습니다.
설치 가능한 조합의 한 예로 torch 2.8.0 / torchvision 0.23.0 cu128 파일을 제공합니다.
기존에 성공한 2.14.1+cu130 환경을 이 예시로 무조건 교체할 필요는 없습니다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/torch-cu128.txt
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

CPU는 `requirements/torch-cpu.txt`, Mac은 `requirements/torch-macos.txt`를 선택합니다.
CLIP은 public packaging 모듈을 직접 import하므로 setuptools 57.4.0을 설치하지 않습니다.
`requirements/legacy-macos-py310.txt`는 과거 기록이며 새 서버용 설치 파일이 아닙니다.

## 데이터 경로

기본값을 사용하거나 `configs/local/me.json`에서 videos_root, features_root,
clip_checkpoint, vadclip_checkpoint, output_root, device를 바꿉니다.
데이터셋의 checkpoint 키는 공통적으로 `vadclip_checkpoint`입니다.

예시 위치:

```text
/workspace/shared-data/
  UCF-Crime/test/                 # 290개 MP4, 하위 폴더 허용
  features/official/ucf/UCFClipFeatures/
  checkpoints/clip/ViT-B-16.pt
  checkpoints/vadclip/model_ucf.pth
```

UCF feature는 원 배포의 부모 폴더와 파일명을 유지합니다. 동일 이름 normal feature가
다른 배포 폴더에 중복되므로 부모 폴더까지 대조합니다. XD는 upstream __0 test 목록을 사용합니다.
공개 feature ZIP이 train을 함께 포함해도 실제 준비·평가 대상은 test 목록으로 필터링합니다.

## ZIP 없이 테스트 영상만 준비

```bash
python scripts/setup/prepare_dataset.py --dataset configs/datasets/ucf.json --local configs/local/me.json --input both
```

- raw: 영상 290개의 SHA/크기를 기존 reference와 대조하고 고정 reference 길이·dtype·GT 정렬 확인.
- official: test feature 290개의 이름/부모/shape/dtype/finiteness/hash와 GT 정렬 확인. raw 영상은 없어도 됨.
- both: 두 입력 모두 준비.
- 준비 결과는 실행별 immutable manifest이며 `.local/setup/<dataset>/<setup_id>/manifest.json`에 저장.
- 실행기에 `--setup-manifest`를 주면 latest pointer와 독립적으로 정확한 준비 결과를 고정.
- 영상 내용을 새로 수정·재인코딩한 데이터는 기존 SHA와 달라 거부함. 단순 확장자/이름 변경으로 우회하지 않음.
- 전처리나 raw frame 좌표 규칙의 정당성까지 이 준비 단계가 검증하는 것은 아님.

원본 ZIP을 처음 검증하는 경우 기존 `scripts/setup/prepare_ucf_test.py`가 ZIP CRC 경로를 유지합니다.
이 스크립트는 작은 legacy data manifest를 재생성하므로 최초 검증용 별도 작업에서 사용하세요.
신규 실행기의 portable reference는 `configs/datasets/ucf_reference.json`으로 분리돼 있습니다.

## 출처

- PyTorch CUDA 12.8/Blackwell: https://pytorch.org/blog/pytorch-2-7/
- torch 2.8 / torchvision 0.23 설치 조합: https://pytorch.org/get-started/previous-versions/
- OpenCV 배포판과 namespace: https://pypi.org/project/opencv-python-headless/
- 기존 수치: `docs/EXPERIMENT_RESULTS.md`, 원본 commit의 summary 사본 `docs/reference_snapshot/outputs/`.
