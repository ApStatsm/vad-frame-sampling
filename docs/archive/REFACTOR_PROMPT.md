# 작업 지시: 실험 코드 정리 및 리팩토링 (UCF raw reference 고정 이후)

## 0. 먼저 읽을 것

`AGENTS.md`, `VAD_Frame_Sampling_Research_Plan.md`, `PROGRESS.md`(특히 §3 유지할 결정의 **고정 source hash**와 마지막 3개 작업 기록)를 읽는다. AGENTS.md의 실행 역할 규칙이 이 작업에도 적용된다. **프로젝트 코드·테스트·dry-run·import 실행은 하지 않는다.** 검증은 사용자가 실행할 명령으로 전달한다.

## 1. 목적

추출 방식 탐색 과정에서 쌓인 진단 스크립트·설정·출력·문서를 정리하고, 앞으로 쓸 핵심 파이프라인(official 평가, raw reference 추출·평가, metric)만 남겨 **sampler(Uniform/Random/Adaptive/Ours)를 붙이기 쉬운 구조**로 리팩토링한다. 그리고 XD-Violence로 확장할 수 있게 한다.

**수치 동작은 바꾸지 않는다.** 리팩토링 후에도 raw reference(`ucf_raw_full_20261008_150142`, branch1 AUC 87.19%)와 official 재현(`ucf_full_20261007_010204`, 88.02%)이 같은 결과를 내야 한다.

## 2. 현재 상태 (정적 확인 결과)

### 2.1 scripts/ 분류

| 분류 | 파일 | 비고 |
|---|---|---|
| **핵심 (유지·리팩토링)** | `evaluate_ucf_official.py`, `evaluate_ucf_raw.py`, `ucf_raw_features.py`, `ucf_color.py`, `ucf_metrics.py`, `ucf_finegrained.py` | 5개는 PROGRESS §3에 hash로 고정됨. `evaluate_ucf_raw`는 `evaluate_ucf_official`에서 `ROOT, LABELS, infer_feature, project_path, sha256, synchronize`를 import함 → 공용 모듈로 분리 필요 |
| **분석 (유지)** | `analyze_ucf_raw_cohorts.py` | 미실행, 선택 작업. 저장 점수만 사용 |
| **일회성 setup (완료, 보존)** | `prepare_ucf_test.py`, `check_ucf_resources.py`, `check_ucf_baseline_inputs.py`, `download_ucf_features.sh` | 재현성을 위해 `scripts/setup/`으로 이동 |
| **일회성 패치 (완료)** | `update_ucf_metrics.py` | 기존 run에 Ano-AUC를 덧붙이는 용도. 지금은 평가기에 통합됨 → 제거 후보 |
| **추출 진단 (결론 남, 제거 후보)** | `verify_ucf_extraction.py`(708줄, 진단들의 공용 helper 허브), `scan_ucf_crop_geometry.py`, `search_ucf_affine_geometry.py`, `search_ucf_dense_alignment.py`(한 번도 실행 안 됨), `search_ucf_feature_timeline.py`, `compare_ucf_decoders.py`, `test_ucf_ffmpeg_frames.py`, `test_ucf_yuv_conversion.py`, `test_ucf_float32_cohort.py` | 핵심 파이프라인은 이 파일들을 import하지 않는다(확인 필요: grep으로 재확인). 결론은 `outputs/setup/ucf_recipe_reconstruction_20261008.json`과 docs에 요약되어 있음 |

### 2.2 그 밖의 정리 대상

- `configs/ucf_raw_reference_provisional_offset8.json`: 기각된 설정 → 제거 후보.
- `docs/` 9개: 추출 감사 관련 4개(`EXTRACTION_COMPLETION_AUDIT`, `EXTRACTION_DIAGNOSTIC_RUNS`, `UCF_EXTRACTION_VERIFICATION`, `VADCLIP_EXTRACTION_AUDIT`)가 겹친다. 하나의 결론 문서와 archive로 통합한다.
- `PROGRESS.md`: 614줄, 125KB. §1–8(현재 상태)만 남기고 §9 날짜별 작업 기록은 `docs/archive/progress_log_2026-10.md`로 옮긴다. (PROGRESS.md는 `.gitignore` 대상이라 git에 없다. 로컬 파일 이동만 하면 된다.)
- `outputs/setup/` 타임스탬프 진단 폴더 약 30개(~100MB, git 미추적): `ucf_extraction_verify_*`, `ucf_geometry_scan_*`, `ucf_decoder_compare_*`, `ucf_jit_encoder_compare_*`, `ucf_yuv_conversion_*`, `ucf_affine_search_*`, `ucf_timeline_search_*`, `ucf_ffmpeg_frames_*`, `ucf_probe_*` 등. 통합 JSON(`outputs/setup/*.json`)은 유지한다.
- `outputs/baseline/ucf_raw_full_20261008_144218/`, `..._144302/`: 경로 오류로 실패한 run(영상 0개 처리).
- `UCF-Crime/Temporal_Anomaly_Annotation_for_Testing_Videos.txt`: `UCF-Crime/test/annotations/`의 파일과 byte 동일하지만 `prepare_ucf_test.py`(20행)의 **원본 입력**이다. 유지한다(Phase 0 확인).
- `.local-backups/tools/gh`(53MB), 빈 `.github/`, `.DS_Store`.
- 알려진 버그: `evaluate_ucf_official.py` 271행 근처의 `audited['path']`가 옛 절대경로(`03_인공지능 프로젝트`)를 그대로 쓴다. raw 평가기의 `audited_path()`와 같은 처리가 필요하다(PROGRESS 2026-10-08 기록 참고).

## 3. 절대 제약

1. **원본 입력 불변**: `UCF-Crime/extracted/`, `UCF-Crime/test/`(manifest·splits·annotations), `features/`, `checkpoints/`, `VadCLIP/list/`(GT .npy 포함)는 수정·이동·삭제하지 않는다. `VadCLIP/src/`는 upstream 코드이므로 건드리지 않는다.
2. **수치 동작 보존**: decode, 색 변환, resize/crop, CLIP 호출, process_split/window/padding, 16배 반복, 두 score branch, metric 계산 순서는 그대로 둔다. PROGRESS §8의 알려진 이상 동작(정수배 길이의 추가 padded window, temporal mask 미전달)도 **이번에는 고치지 않는다.** 주석으로만 표시한다.
3. **삭제 전 승인**: 파일을 지우기 전에 Phase 0 목록을 사용자에게 보여 주고 승인을 받는다. git 미추적 파일(outputs 진단 폴더 등)은 `rm` 대신 `.local-backups/pre-refactor-20261008/`로 **이동**한다. 영구 삭제는 사용자가 직접 한다.
4. **추적 파일은 git으로 보존**: 정리 전 현재 commit에 tag `ucf-raw-reference-v1`을 단다. 그 뒤 진단 스크립트는 작업 트리에서 지운다. 문서에서는 `git show ucf-raw-reference-v1:scripts/<file>`로 참조한다.
5. 작업은 `refactor/cleanup` 브랜치에서 하고 Phase마다 commit한다.

## 4. 진행 단계 (각 Phase 끝에서 멈추고 사용자 확인)

### Phase 0 — 인벤토리와 동결 (코드 변경 없음)
- `git tag ucf-raw-reference-v1` (현재 HEAD). PROGRESS §3의 고정 hash 5개가 현재 파일과 같은지 `shasum -a 256`로 확인한다(파일 읽기만 하므로 허용 범위).
- 모든 tracked/untracked 파일을 **유지 / 이동 / git 삭제 / 로컬 백업 이동** 중 하나로 분류한 표를 만든다. 각 항목에 근거를 붙인다(누가 import하는지, 어느 문서가 참조하는지). `grep -rn`으로 scripts·docs·configs·README의 참조를 전부 확인한다.
- → **사용자 승인 대기**

### Phase 1 — 정리 (로직 변경 없음)
- 승인된 항목을 삭제하거나 이동한다. setup 스크립트는 `scripts/setup/`으로 옮긴다.
- 이동 때문에 깨지는 import와 경로 참조를 고친다. setup 스크립트의 `ROOT = Path(__file__).parents[1]` 같은 값은 한 단계 더 올라가야 한다.
- 문서 통합: 추출 관련 docs 4개 → `docs/ucf_raw_reference.md` 하나(최종 recipe, 근거 수치, 기각 가설 요약, float32 4개 미해결)와 `docs/archive/`. PROGRESS 작업 기록 → `docs/archive/progress_log_2026-10.md`.
- `.gitignore`의 failed run 예외 줄 등 더 이상 필요 없는 규칙을 정리한다.
- → commit, **사용자 확인**

### Phase 2 — 핵심 코드 리팩토링 (동작 보존)
아래와 같은 구조를 제안한다. 더 단순한 구조가 낫다고 판단되면 근거와 함께 대안을 제시한다.

```
vadfs/                      # 연구 코드 패키지 (scripts에서 import)
  paths.py                  # ROOT, project_path, audited_path, sha256
  provenance.py             # run 폴더 생성, 환경/hash/seed 기록, synchronize, 시간 측정
  data/ucf.py               # manifest·LABELS·GT·공식 feature index 로딩
  extract/decode.py         # ffmpeg yuv420p pipe decode, 16-frame block
  extract/color.py          # (ucf_color.py 이동)
  extract/encoder.py        # ReferenceImageEncoder, crop, resize
  vad/vadclip.py            # 모델 로딩, infer_feature, process_split/window (원본 동작 그대로)
  eval/metrics.py           # ucf_metrics + ucf_finegrained
  sampling/__init__.py      # 인터페이스 자리만: select(N_v, r, seed) -> 정렬된 index. 구현은 하지 않음
scripts/
  eval_official.py          # 얇은 CLI
  eval_raw.py               # 얇은 CLI (sampler 인자가 들어올 자리만 남겨 둠)
  analyze_raw_cohorts.py
  setup/...
configs/ucf/{resources.json, raw_reference.json}
```

- `evaluate_ucf_raw` → `evaluate_ucf_official` 간 import 같은 CLI끼리의 의존을 없앤다. 공용 함수는 패키지로 옮긴다.
- `sys.path.insert` 해킹은 VadCLIP/src 한 곳만 `vad/vadclip.py`에 둔다(또는 `pyproject.toml` + `pip install -e .`. 이 경우 사용자에게 설치 명령을 안내한다).
- `audited_path` 버그를 official 평가기에도 적용한다. **이것은 유일하게 허용하는 동작 변경이다.** 경로 해석만 바뀌고 수치에는 영향이 없다는 근거를 적는다.
- summary.json에 기록하는 source hash 목록을 새 파일 경로에 맞게 갱신한다. 출력 JSON의 key 이름과 구조는 유지한다. 기존 run과 비교 가능해야 하므로, 바꿔야 하면 변경표를 남긴다.
- `ucf_` 접두어는 dataset 모듈 이름으로 옮긴다. XD 지원은 구현하지 않고 `data/xd.py`를 추가할 수 있는 형태로만 둔다.
- → commit, **사용자 확인**

### Phase 3 — 동등성 검증 (사용자 실행)
사용자가 실행할 명령과 통과 기준을 작성해 전달한다. 필요하면 작은 검증 스크립트 `scripts/check_refactor_equivalence.py`를 작성한다. 이 스크립트는 저장된 결과만 읽는다.
1. **Metric 동등성(빠름)**: 두 기준 run의 저장 `frame_scores.npz`로 새 `eval/metrics.py`가 계산한 AUC/AP/Ano-AUC/fine-grained가 summary.json과 **완전히 같은지** 확인한다.
2. **추출 동등성(중간)**: 새 코드로 영상 몇 개(float16 짧은 영상, N%16==0 영상, float32 1개, Normal 1개)를 추출해 `outputs/baseline/ucf_raw_full_20261008_150142/features/`와 비교한다. 같은 장치(MPS)에서 bit 동일 또는 max abs diff ≤ 1e-6을 기준으로 한다.
3. **VAD 동등성**: 같은 영상들의 token score가 저장 `scores/`와 같은지 확인한다.
4. (선택) raw full 재실행 → branch1 AUC 87.19% 일치.
- 통과하면 PROGRESS §3의 고정 hash를 **새 파일 기준으로 다시 기록**한다. 이전 hash는 `ucf-raw-reference-v1` tag와 함께 남긴다.

### Phase 4 — 문서 마무리
- README: 새 구조, 설치, 실행 명령 3개(official 평가 / raw 평가 / cohort 분석)를 적는다.
- PROGRESS.md: 현재 상태, 고정 결정, 새 경로, 다음 작업(Uniform/Random 구현)을 갱신하고 작업 기록에 이 리팩토링을 추가한다.
- 이 파일(`REFACTOR_PROMPT.md`)은 작업이 끝나면 삭제하거나 `docs/archive/`로 옮긴다.

## 5. 보고 형식

Phase마다 (1) 변경 파일 목록, (2) 이동/삭제 목록, (3) 정적 검토 결과, (4) **실행하지 않은 검증**, (5) 사용자가 실행할 명령과 통과 기준을 보고한다. 정적 검토와 실제 실행 검증을 구분한다.
