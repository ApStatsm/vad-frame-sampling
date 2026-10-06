# 공유할 실행 요약

이 폴더의 초기 두 요약은 **기존 사용자 실행 보고서에서 추린 기록**입니다. GitHub 준비 중 Codex가 프로젝트 코드나 inference를 실행한 결과가 아닙니다. 모델 source의 변경 표시와 연구 Git 버전 기록이 이후 보완되었으므로 최신 코드의 smoke 통과를 의미하지 않습니다.

| 요약 | 기록한 범위 | Benchmark metric |
|---|---|---|
| [setup_20261006](setup_20261006/summary.json) | Raw test 준비·자원 매핑·feature/GT 정렬·CPU strict loading | 없음 |
| [MPS smoke 7사례](ucf_official_smoke_mps_20261006_202116_841716/summary.json) | 실제 영상 3개와 255/256/257/512 token 경계 prefix. 두 branch 유효 score/길이 통과 | 없음 |

원본 보고서·score는 각 JSON의 프로젝트 상대 경로로 로컬 `outputs/`에 보존합니다. 공유 요약의 경로는 위치 안내이며 팀원의 실행용 입력이나 통과 audit를 대신하지 않습니다. 기존 run은 연구 Git 초기화 전 실행이므로 research commit=null, 당시 source hash를 보존합니다.

새 benchmark는 `results/<run_id>/summary.json`과 필요한 작은 표/그래프로 공유합니다. 코드 commit·dirty 상태, upstream commit, dataset/split, 입력 유형, 설정·seed, checkpoint/source hash, 장치·precision, metric, 시간 측정 범위, 원시 결과 위치, 미검증 사항을 남깁니다. 큰 score 배열은 포함하지 않습니다. Cached-feature 시간을 raw-video end-to-end 시간이나 sampling speedup으로 표현하지 않습니다.
