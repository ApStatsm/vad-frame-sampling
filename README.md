# VAD Frame Sampling

고정된 pretrained **VadCLIP**에서 frame sampling만 변경하여 이상 탐지 성능과 실제 계산 비용을 비교하는 연구 프로젝트입니다. UCF-Crime을 먼저 검증하고 XD-Violence로 확장합니다.

## 현재 진행 상태 — 2026-10-06

| 단계 | 확인된 상태 |
|---|---|
| UCF raw test 정리 | 290개: anomaly 140, normal 150. ZIP CRC/크기·영상 metadata 점검 완료 |
| Official feature·GT·checkpoint | 사용자 자원/입력 점검 통과. GT 1,109,888, annotation mismatch 0. CPU strict loading 성공 |
| 소규모 inference | 사용자 MPS smoke 7사례 통과. 당시 소스 hash를 결과에 보존 |
| 전체 official-feature baseline | 아직 미실행. 실제 AUC/AP 없음 |
| Raw reference·sampling 비교·XD 확장 | 미착수 |
| Ours | 미설계. 별도 개발 데이터에서 설계 예정 |

GitHub 공유 준비에서 실행 버전 기록과 변경 표시를 보완했습니다. **기존 smoke는 이전 소스의 결과이므로 현재 코드의 smoke를 다시 통과한 후 full 평가를 실행해야 합니다.** 모델 계산·가중치·window·prompt·score 복원 규칙은 이번 공유 준비에서 변경하지 않았습니다. 새 코드·환경 설치는 정적 검토만 했습니다.

## 먼저 읽을 문서

- [연구 계획](VAD_Frame_Sampling_Research_Plan.md): 목표, 통제 조건, sampling·평가·효율 원칙.
- [진행 기록](PROGRESS.md): 실제 완료 상태, 환경, 증거, 미해결 문제.
- [로드맵](docs/PROJECT_ROADMAP.md): 단계별 작업과 완료 기준.
- [팀원 환경 준비](docs/TEAM_SETUP.md): clone 이후 필요한 입력과 실행 순서.
- [자원 준비](docs/UCF_RESOURCE_SETUP.md), [baseline 실행](docs/UCF_OFFICIAL_BASELINE.md).
- [협업 방법](CONTRIBUTING.md), [원본 코드와 변경 내역](VadCLIP/LOCAL_CHANGES.md).

## 파일 구성

```text
VadCLIP/        원본 VadCLIP 소스·평가 목록·GT와 연구용 수정
scripts/        데이터 준비, 자원/입력 점검, official-feature 평가
configs/        프로젝트 상대 경로를 사용하는 공유 설정
docs/           준비·실행·연구 진행 안내
results/        기존 사용자 실행의 작은 공유 요약
UCF-Crime/      작은 annotation·목록과 안내; 영상·manifest는 로컬 생성
features/       안내만 공유; official feature는 별도 확보
checkpoints/    안내만 공유; pretrained weights는 별도 확보
outputs/        실행별 보고서·local CSV·score·로그; Git 제외
```

원본 영상, feature, weights, 가상환경과 원시 outputs는 저장소에 포함하지 않습니다. 팀원은 자신의 컴퓨터에서 확보하고 local manifest/CSV/audit를 생성해야 합니다. 원본 GT 배열과 CLIP tokenizer 사전은 작고 필요한 평가 입력이므로 `VadCLIP/`에 보존합니다.

## 환경과 실행

기존 사용자 검증 환경은 Python 3.10.0, PyTorch 2.8.0, macOS arm64/MPS입니다. [requirements.txt](requirements.txt)는 설치된 패키지의 metadata에서 확인한 직접 의존성 버전이며, 새로운 환경의 설치 성공이나 CPU/CUDA 동등성을 검증한 목록은 아닙니다. Raw test 준비에는 `ffprobe`도 필요합니다.

새 clone에서 사용자가 실행합니다. 기존 작업 폴더의 `.venv`는 보존합니다.

```sh
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

[팀원 환경 준비](docs/TEAM_SETUP.md)에 따라 영상·feature·checkpoint를 확보하고 test manifest를 만든 다음, 프로젝트 루트에서 아래 순서로 실행합니다.

```sh
.venv/bin/python scripts/check_ucf_resources.py --write-local-list
.venv/bin/python scripts/check_ucf_baseline_inputs.py --check-model
.venv/bin/python scripts/evaluate_ucf_official.py --mode smoke --device mps
.venv/bin/python scripts/evaluate_ucf_official.py --mode full --device mps
```

macOS MPS 외의 환경에서는 smoke/full 모두 같은 `--device cpu` 또는 `--device cuda`를 지정합니다. 해당 장치에서의 성공은 별도 사용자 검증 대상입니다. 재료나 코드가 바뀌면 필요한 점검부터 다시 실행합니다.

예상 출력은 `outputs/setup/`의 점검 보고서/local CSV와 `outputs/baseline/<run_id>/summary.json`, scores입니다. Full 성공 여부, 영상 290개, 평가 frame 1,109,888, 두 branch AUC/AP를 확인합니다. 이 시간 기록은 cached-feature 실행이며 raw-video speedup을 뜻하지 않습니다.

## 연구·협업 원칙

- 같은 데이터셋에서 encoder·checkpoint·prompt·score branch·후처리를 고정합니다.
- 최종 sampling은 CLIP visual encoder 전에 수행하고, 원본 timeline 전체에서 평가합니다.
- GT는 평가/분석에만 사용하고 test 결과로 sampler를 튜닝하지 않습니다.
- 작업별 branch에서 작은 commit을 남기고 PR로 변경과 검증 상태를 검토합니다.
- 실험 실행 전 관련 코드·설정을 commit하고 run ID, 코드/입력 hash, device와 결과 위치를 남깁니다.
- Codex는 코드 작성·정적 검토를 담당합니다. 사용자 명시 요청 없이 프로젝트 코드·테스트·inference를 실행하지 않습니다. 자세한 규칙은 [AGENTS.md](AGENTS.md)를 따릅니다.

## 출처

원본: [nwpu-zxr/VadCLIP](https://github.com/nwpu-zxr/VadCLIP), 기준 commit `c41067f07d252efcda18008bea367886070c33b0`. `VadCLIP/`은 이 소스의 snapshot과 명시한 local 변경을 일반 하위 폴더로 포함합니다. 원본 [Apache-2.0 LICENSE](VadCLIP/LICENSE), README와 기존 출처 표기를 보존합니다. 이 프로젝트 전체에 새로운 라이선스를 지정한 것은 아닙니다.
