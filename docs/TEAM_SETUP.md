# 팀원 환경 준비

2026-10-06. Git clone만으로 영상·feature·checkpoint·local audit가 준비되지는 않습니다. 기존 사용자 보고서는 과거 실행의 증거이며 새 컴퓨터의 통과 기록이 아닙니다.

## 1. 저장소와 환경

GitHub 저장소의 Code 메뉴에서 실제 주소를 복사하여 clone한 뒤 프로젝트 루트로 이동합니다. Python 3.10과 `ffprobe`를 준비합니다. 새 clone에서:

```sh
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

의존성 버전은 기존 패키지 metadata 기준이며 새 설치는 미검증입니다. 기존 `.venv`는 system-site-packages를 사용하지만 새 환경은 독립적으로 생성합니다. OS/장치별 PyTorch 설치가 다르면 사용하는 build와 버전을 실행 기록에 남깁니다. CPU/CUDA 실행 성공은 현재 MPS 사용자 결과와 별개로 확인해야 합니다.

## 2. 원본 입력 확보

- [UCF-Crime/README.md](../UCF-Crime/README.md)의 ZIP 5개와 extracted raw videos.
- [features/README.md](../features/README.md)의 official feature 패키지.
- [checkpoints/README.md](../checkpoints/README.md)의 CLIP/UCF pretrained weights.
- Git에 포함된 공식 test CSV·GT 및 annotation. 원본 파일은 임의 수정하지 않습니다.

다운로드 링크는 [UCF_RESOURCE_SETUP.md](UCF_RESOURCE_SETUP.md)를 참고합니다. 팀 공유 저장 공간을 사용하는 경우에도 데이터 배포 조건과 접근 권한은 원본 제공 기준으로 관리합니다.

## 3. 해당 컴퓨터의 입력 점검

`configs/ucf_resources.json`의 세 값은 프로젝트 상대 경로입니다. 동일 폴더 구조로 자원을 두면 그대로 사용할 수 있습니다. 다른 위치를 쓰면 Git에서 제외되는 `configs/ucf_resources.local.json`에 설정하고 아래 자원 점검·baseline 입력 점검·smoke·full 모두 같은 `--config configs/ucf_resources.local.json`을 전달합니다.

```sh
.venv/bin/python scripts/prepare_ucf_test.py
.venv/bin/python scripts/check_ucf_resources.py --write-local-list
.venv/bin/python scripts/check_ucf_baseline_inputs.py --check-model
```

첫 명령은 `ffprobe`와 영상/ZIP 읽기를 수행하고 로컬 manifest·test symlink를 만듭니다. 뒤 명령은 local feature CSV와 자원/배열/GT/CPU loading 점검 보고서를 만듭니다. 기존 로컬 사용자는 이 세 단계가 통과했고 입력이 그대로라면 GitHub 정리만을 이유로 반복할 필요가 없습니다.

다른 팀원의 local CSV나 audit를 `outputs/setup/`에 복사하여 통과 기록으로 사용하지 않습니다. 절대 경로와 hash가 해당 컴퓨터 입력에 맞아야 합니다.

## 4. 현재 버전 smoke와 full

공유 준비에서 source의 변경 표시와 Git 버전 기록이 바뀌었으므로 기존 사용자도 **현재 버전 smoke**를 먼저 수행합니다. MPS 환경 예시:

```sh
.venv/bin/python scripts/evaluate_ucf_official.py --mode smoke --device mps
.venv/bin/python scripts/evaluate_ucf_official.py --mode full --device mps
```

CPU 또는 CUDA 사용자는 두 명령 모두 같은 `--device cpu` 또는 `--device cuda`를 지정합니다. 장치가 없으면 자동 전환하지 않습니다.

Smoke에서 `outputs/setup/ucf_smoke_check.json`의 success=true, 장치 일치, 7사례 유효 score/길이를 확인합니다. Full은 통과한 smoke와 source/config/checkpoint hash를 대조한 다음 전체 평가를 수행합니다.

예상 full 출력: `outputs/baseline/ucf_official_full_<device>_<timestamp>/summary.json`, 영상별 `scores/*.npz`, `frame_scores.npz`. 성공 상태, 영상 290개, 평가 frame 1,109,888과 두 branch AUC/AP를 확인합니다. Full 실행이 끝나기 전에는 측정값을 채우지 않습니다.

## 5. 기록과 공유

코드·설정을 commit한 뒤 실행합니다. 원본 저장소의 기준 commit은 `VadCLIP/UPSTREAM.json`에서, 연구 코드 버전과 tracked dirty 상태는 연구 Git에서 기록합니다. Git이 없는 소스 사본은 research commit=null로 기록하고 source hash를 유지합니다.

작은 결과 요약은 `results/<run_id>/`, 원시 산출물은 `outputs/`나 팀 공유 저장 공간에 보관합니다. GitHub에 포함된 `results/`의 기존 MPS smoke는 이전 사용자 실행을 요약한 것이며 최신 코드의 실행 검증/전체 benchmark 결과가 아닙니다.
