# 기업 고용·수익성 분석: 데이터 품질 감사와 예측 검증

[![checks](https://github.com/healthy27/data-final/actions/workflows/ci.yml/badge.svg)](https://github.com/healthy27/data-final/actions/workflows/ci.yml)

수업 Colab의 재무·직원수 분석을 원본 대조, 명시적 계정 매핑, 기업 간 교차검증을 갖춘 Python 프로젝트로 개선했습니다.
**5개 제공 파일을 감사하고, 125개 기업의 2021→2022년 탐색 분석을 실행했습니다.**

## 확인한 결과

- 대체 전 재무 결측 **1,134개**가 계정 구분 없이 연도 전체 중앙값으로 채워져 있었습니다.
- `자본과부채총계 → 부채총계` **86행**, `부채와자본총계 → 자본총계` **10행**을 확인했습니다.
- 동일 재무 행 **33개 중복**, 직원수 **1개 기업·연도 충돌**, 표준 계정 **429개 중복 그룹**을 기록했습니다.
- 한 기업은 기존 전처리·합산으로 **적자 부호가 흑자로 바뀌었습니다.** 대체 전 원본에서 정답을 다시 추출했습니다.
- 직원수 추가의 로지스틱 AP 변화는 평균 **+0.0011**로, 분할에 따라 개선과 악화가 섞였습니다.
- 단순한 전년 적자 유지 기준이 복잡한 모델보다 높은 정확도·적자 재현율을 보였습니다.

원본 자료의 통화·연결 범위·실제 수집 연도·공시 시점은 미확인입니다.
이번 결과는 단일 연도 전환에서의 기업 간 탐색 검증이며, 여러 미래 연도나 실전 투자 성능을 의미하지 않습니다.

![실데이터 모델 비교](reports/legacy/comparison.png)

## 성능과 해석

125개 기업 중 적자 25개(20%). 5-fold를 5회 반복한 전체 OOF 점수의 평균입니다.
모델 선택은 각 외부 학습 자료 안의 3-fold에서 AP로 수행했습니다. 양성=적자, 임계값=0.5 고정.

| 방법 | ROC-AUC | AP | 정확도 | 적자 재현율 |
| --- | ---: | ---: | ---: | ---: |
| 모두 흑자 예측에 해당하는 학습 비율 기준 | 0.5000 | 0.2000 | 80.00% | 0.0% |
| 전년 적자 상태 유지 | 0.7750 | 0.4816 | 85.60% | 64.0% |
| 재무 로지스틱 | 0.7510 | 0.5796 | 84.64% | 32.0% |
| 재무+직원수 로지스틱 | 0.7615 | 0.5807 | 84.96% | 34.4% |
| 내부 검증에서 모델·피처를 선택하는 절차 | 0.7712 | 0.5708 | 83.52% | 38.4% |

AP는 적자 기업을 높은 순위에 놓는 능력을 봅니다. 임계값을 적용한 재현율과는 다른 지표입니다.
복잡한 선택 절차가 전년 상태 기준보다 AP는 높았지만 정확도·재현율·ROC-AUC는 높지 않았습니다.
기존 Colab 95.24%는 같은 해의 380개 항목으로 같은 해 손익을 분류한 다른 실험이어서 직접 비교하지 않습니다.

**[문제·중요성·대안·장단점·해결·결과 전체 설명](docs/case_study.md)**

함께 볼 자료: [실데이터 분석 리포트](reports/legacy/report.md),
[포트폴리오·면접 문안](docs/portfolio.md), [실행한 노트북](notebooks/02_real_data_review.ipynb).

## API 키 없이 재현

Python 3.12와 [uv](https://docs.astral.sh/uv/getting-started/installation/)를 사용합니다.

```bash
git clone https://github.com/healthy27/data-final.git
cd data-final
uv sync --locked --no-editable
uv run --locked --no-editable data-final analyze-legacy data/reference/legacy_panel.csv
uv run --locked --no-editable pytest -q
```

출력은 `reports/runs/legacy/`입니다.
`data/reference/legacy_panel.csv`는 대체 전 제공 재무와 두 직원수 자료에서 추출한 재현용 입력입니다.
기업코드·통화·연결 기준을 추정해 채우지 않고 `legacy_unverified`로 표시했습니다.
원본 Excel·전체 CSV는 저장소에 복사하지 않았습니다.

원본 5개 파일이 있는 폴더에서 감사부터 다시 실행할 수 있습니다.

```bash
uv run --locked --no-editable data-final audit-legacy --source-dir /path/to/original-files
uv run --locked --no-editable data-final analyze-legacy reports/runs/source_audit/legacy_panel.csv
```

실제 분석과 별도로 파이프라인 동작을 확인하는 [합성 데이터 예제](reports/demo/report.md)도 유지했습니다.
`data-final demo`는 합성 데이터이며 실제 기업 분석 성과로 사용하지 않습니다.

## 새 DART 자료 수집

원본 Colab에 노출됐던 키는 재사용하지 말고 새 키를 환경변수 `DART_API_KEY`로 설정합니다.
`.env.example`은 형식 안내이며 자동으로 로드되지 않습니다.

```bash
uv run --locked --no-editable data-final collect \
  --universe configs/universe.example.csv --start-year 2018 --end-year 2024
uv run --locked --no-editable data-final validate data/processed/panel.csv
uv run --locked --no-editable data-final analyze data/processed/panel.csv --test-year 2024
```

새 수집 경로는 연간 OFS·기업 고유번호·표준 계정 ID를 사용하며 다른 연도로 대체하지 않습니다.
다년도 자료는 과거 연도에서 모델을 선택하고 마지막 연도를 평가합니다.
예시 기업 8개는 접속 확인용이며 대표 표본이 아닙니다. 충분한 클래스가 없으면 분석이 중단됩니다.
새 키를 사용한 실제 수집과 원문 검증은 아직 수행하지 않았습니다.

## 설계와 구조

```text
src/data_final/
  legacy.py            # 제공 원본 대조, 의미가 같은 계정만 추출, 출처 기록
  legacy_analysis.py   # 단일 전환의 기업 간 nested CV, 직원수 ablation, 민감도 분석
  dart.py              # 새 DART 수집, 표준 ID 매핑, 원본 캐시·감사
  data.py              # 검증된 새 자료의 계약·시차 변수, 합성 예제
  analysis.py          # 다년도 순차 검증, 기준 모델, 그래프·리포트
  cli.py               # 실행 명령
data/reference/        # 제공 자료에서 추출한 재현용 입력
reports/legacy/        # 실데이터 결과, 출처·제외 목록·OOF 예측
reports/demo/          # 합성 데이터 실행 예제
notebooks/             # 실행 결과를 설명하는 노트북
tests/                 # 누수·시점·정답·매핑·중복·기업 분할 회귀 테스트
docs/                  # 사례 연구, 데이터 사전, 방법론, 발표 가이드
uv.lock                # 의존성 고정
```

### 이 프로젝트에서 보여주는 역량

원본 파일을 대조해 오류의 위치와 영향을 정량화하고, 사용 가능한 자료의 한계에 맞춰 평가를 설계했습니다.
직원수 추가 효과와 단순 기준을 비교해 복잡한 모델의 효용을 검증했습니다.
결론의 범위·출처·실행 기록·자동 테스트를 함께 남겨 결과를 다시 확인할 수 있게 했습니다.

참고 문서: [전체 사례 연구](docs/case_study.md), [제공 자료 평가 설계](docs/legacy_methodology.md),
[새 DART 자료의 다년도 설계](docs/methodology.md), [원본 검토 기록](docs/original_review.md).

원본: [수업 Colab](https://colab.research.google.com/drive/127xObPPWqpLnHoqjdiXId0Dhkakf9tzL).
개선 구현에는 AI 코딩 도구를 활용했으며, 포트폴리오에서는 본인이 설명·검증한 범위를 구분합니다.
