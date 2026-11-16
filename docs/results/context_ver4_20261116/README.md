# ver4 결과 보존

[결과 해석](../../../evaluation_ver4.md) · [방법·실행 안내](../../../method_ver4.md)

사용자 실행의 A/B 결과 CSV와 완료 sweep을 원본 그대로 복사했다. [sources.json](sources.json)에 원본 경로·원본 SHA-256과 사본 identity를 기록했다.

원본 위치:

- A: `outputs/analyses/xd_decomposition_ver4_full_20261116_030537/`.
- B: `outputs/runs/xd_context_probe_ver4_full_20261116_030706/`.

`summary_compact_ver4.json`은 대용량 배열·mapping을 제외한 사본이며 **원본 summary와 바이트가 다르다**. A는 `score_input_sha256`, B는 `videos`·`targets`를 제외했다. 각 사본의 `archive_note`와 sources.json에 원본 hash·생략 개수를 기록했다. 원본과 compact 사본 hash를 혼동하지 않는다. B의 대상별 수치는 원본 그대로 복사한 `probe/target_scores_ver4.csv`에도 보존했다.

A의 단계별 AP/FPR와 B의 기술통계·provenance·전체 성공 상태는 compact summary에 남는다. 영상별 재현 오차0·skip8개 확인은 서버 원본 summary에 근거한 결과 문서 기록이며, 해당 영상별 배열은 compact 사본에 없다. 상세 NPZ·feature와 전체 원본 summary는 서버에 보존한다. 원본 링크는 서버 절대 경로를 포함할 수 있으며 독립 실행에는 데이터·checkpoint·환경 준비가 필요하다.

원본 결과 생성 시 Git 작업 트리는 미커밋 상태였다. 이후 보존된 소스는 기록된 SHA-256으로 대조하며, 문서의 커밋 보존이 당시 실행 commit을 바꾸지는 않는다. 보존 과정에서 프로젝트 코드·분석·metric·추론은 실행하지 않았다.
