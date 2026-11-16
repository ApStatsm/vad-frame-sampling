# Frame Sampling 성능 평가 — ver3

사용자 실행 완료 결과. 완료 폴더의 날짜는 2026-11-16 KST(2026-11-15 UTC)다. XD-Violence test 800개에서 저장된 raw Reference CLIP feature를 사용해 시간적 배치를 비교했다. 8회 모두 `success=true`, 자동 비교까지 완료됐다.

## XD-Violence — Frame-level AP (%)

| 방법 | Budget | 실행 수 | AP (%) | Uniform 대비 (pp) |
|---|---:|---:|---:|---:|
| Full / Reference | 100% | 1 | 84.5041 | 기준 외 |
| Uniform | 12.5% | 1 | 78.8278 | 0 |
| 규칙적 짝 배치 | 12.5% | 1 | 78.8865 | +0.0587 |
| 무작위 분산 재배치 | 12.5% | 5 | 78.9148 ± 0.0448 | +0.0870 |

무작위 재배치는 seed0–4의 평균 ± 표본 표준편차다. Full은 별도 기준이며 동일 예산 순위에 포함하지 않는다. AP 최고값이나 작은 차이는 통계적 유의성을 의미하지 않는다. UCF와 다른 budget은 ver3에서 실행하지 않았다.

## 동일 재현율 오탐률 및 사건 포착

목표 재현율은 Full Branch2 점수의 임계값0.5 재현율 **97.6499%**다. FPR은 ROC 선형보간값이며 운영 임계값 선택에 사용하지 않는다.

| 방법 | FPR (%)↓ | 전체 사건 포착 (/1,238) | 1초 이하 포착 (/27) | 선택 입력 수 |
|---|---:|---:|---:|---:|
| Full / Reference | 24.7452 | 1,236 | 25 | 145,706 |
| Uniform | 30.2283 | 957 | 2 | 18,549 |
| 짝 배치 | 30.8133 | 924 | 2 | 18,549 |
| 무작위 재배치 | 30.5397 ± 0.3003 | 940.6 ± 3.5777 | 1.0 ± 0.0 | 18,549 |

포착은 실제 선택 프레임이 정답 구간에 한 장 이상 있는 경우다. 모델의 탐지 성공과 구분한다. 평균 사건 수가 소수인 것은 5seed 평균이기 때문이다.

## 동일 위치의 점수 변화

짝 배치와 Uniform의 공통 정상 프레임11,316개에서 Branch2 직접 점수의 평균 변화는 **+0.0005134**, 이상 프레임3,319개에서는 **−0.0007354**였다. 두 집단 모두 중앙값은0이었다. 이는 정상 점수 상승을 해소했다는 근거가 되지 않는다. 복원된 전체 시간축 점수와 구분하며 프레임을 독립 표본으로 간주한 유의성 검정은 하지 않았다.

## 평가 조건과 해석

- [method_ver3.md](method_ver3.md)의 사전 고정 규칙을 사용했다. Uniform 슬롯8개 중 슬롯3·4만 가까운 reference row의 짝 또는 같은 구간 내부 무작위 비인접 쌍으로 바꾼다. 양끝·불완전 tail·영상별 K를 보존한다.
- 모델·prompt·전처리·feature 값과 dtype은 고정했다. 실제 선택 좌표에 점수를 두고 nearest 복원한 뒤 전체 공식 시간축에서 평가했다.
- Full과 Uniform AP는 기존84.5041%·78.8278%와 일치한다. 자동 비교기는 선택 규칙·저장 점수·AP 재현을 검증한 뒤 완료됐다.
- 짝 배치는 Uniform보다 AP가0.0587pp 높았지만 무작위 재배치 평균보다 낮았다. FPR은0.5850pp 높고 전체 사건 포착은33건 적었다. 1초 이하 포착은 동일했다.
- **이 고정 짝 배치의 뚜렷한 이점은 확인되지 않았다.** 이 결과로 주변 맥락이 불필요하다고 일반화할 수 없다. 배치 변경은 coverage·시간 간격·입력 내용도 함께 바꾼다.
- 기존 test 분석 이후 설계한 탐색 실험이며, cached 성능 비교다. raw-video speedup을 입증하지 않는다.
- 이 문서 작성 과정에서는 기존 JSON·CSV만 읽었으며 실험·분석·metric을 재실행하지 않았다.

## 원본 결과 및 재현 안내

- [완료 sweep](docs/results/context_ver3_20261116/sweep_ver3.json)
- [집계 비교 CSV](docs/results/context_ver3_20261116/comparison_ver3.csv)
- [실행별 결과](docs/results/context_ver3_20261116/runs_ver3.csv)
- [공통 프레임 점수 변화](docs/results/context_ver3_20261116/common_frame_changes_ver3.csv)
- [분석 입력 및 provenance](docs/results/context_ver3_20261116/analysis_ver3.json)
- [방법·실행 명령](method_ver3.md)

GitHub 열람용 사본과 원본 경로·SHA-256은 [보존 안내](docs/results/context_ver3_20261116/README.md)에 있다. 상세 NPZ와 개별 원본 run summary는 서버 outputs에 남아 있다.
