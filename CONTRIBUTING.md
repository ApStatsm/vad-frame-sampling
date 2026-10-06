# 팀 협업

## 일상 작업

`main`에는 공유 가능한 상태를 유지합니다. 코드가 있어도 실행 검증이 끝나지 않았다면 그 사실을 문서와 PR에 표시합니다. Issues에는 다음 작업과 담당자를, `PROGRESS.md`에는 실제 완료와 실험 근거를 기록합니다.

```sh
git switch main
git pull --ff-only
git switch -c feat/uniform-sampling
```

하나의 목적을 담은 변경을 준비하고 추가할 파일을 지정합니다. 아래 파일명은 예시이며 실제 변경에 맞춰 바꿉니다.

```sh
git add scripts/your_script.py docs/your_guide.md PROGRESS.md
git diff --cached --stat
git diff --cached
git commit -m "feat: add uniform frame sampling"
git push -u origin feat/uniform-sampling
```

GitHub에서 `main`으로 PR을 만들고 팀원 검토 후 병합합니다. 진행 중인 변경은 Draft PR로 공유할 수 있습니다. 작업 브랜치는 `feat/...`, `fix/...`, `docs/...`처럼 목적별로 만듭니다.

## PR에 기록할 내용

- 해결하는 문제와 변경 후 동작.
- 실제 실행한 검증과 정적 검토만 한 범위.
- 필요한 입력, 사용자 실행 명령, 예상 결과 위치.
- 실험 비교에 영향을 주는 checkpoint·prompt·window·timeline 규칙 변경 여부.

Codex는 별도 명시적 요청이 없으면 코드·테스트·dry-run·import·inference를 실행하지 않습니다. 사용자 결과를 받아 실행 검증 상태를 갱신합니다.

## 실험과 결과 공유

실험 전에 관련 코드와 설정을 commit하고 작업 트리를 정리합니다. 기록에는 research commit, dirty 상태, 고정된 upstream commit, 실행 source/config/checkpoint hash를 함께 남깁니다. Commit은 소스 버전이며, checkpoint hash는 모델 식별값입니다.

원시 outputs는 로컬 또는 팀 저장 공간에 보관합니다. Git에 올릴 요약은 `results/<run_id>/`에 두고 입력 유형, 장치, precision, seed, metric, 검증 범위와 원시 결과 위치를 기록합니다. 다른 컴퓨터의 절대 경로를 자신의 입력 설정이나 통과 보고서로 재사용하지 않습니다. Smoke 결과에는 AUC/AP를 만들어 넣지 않습니다.

주요 단계가 검증되면 해당 코드 commit에 태그를 붙입니다. 아래는 official-feature 재현 완료 후 사용하는 예시이며, 현재 완료된 단계는 아닙니다.

```sh
git tag -a ucf-official-baseline-v1 -m "Verified UCF official-feature baseline"
git push origin ucf-official-baseline-v1
```

## 원본 코드 유지

`VadCLIP/UPSTREAM.json`의 기준 commit과 `VadCLIP/LOCAL_CHANGES.md`를 보존합니다. 원본을 갱신할 때 변경 내역과 기준 commit을 함께 갱신하고 기존 local 수정이 유지되는지 검토합니다. 이 폴더에서 별도 `git init`을 하지 않습니다. 전체 프로젝트 루트의 Git이 소스를 관리합니다.

원본 LICENSE·저작권·출처를 보존하고 수정한 원본 파일에는 변경 표시를 남깁니다. 영상·feature·checkpoint·가상환경·개인 설정·인증 파일과 raw outputs는 커밋하지 않습니다.
