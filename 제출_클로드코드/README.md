# 지역사회건강 우울 선별모형 — 코드·자료 일체

KCHS에서 개발하고 미국 NHANES에서 외적검증한 머신러닝 우울 선별모형(PHQ-9 ≥10)의
분석 코드, 결과, 원고 원본, 제출 파일을 담았다.

- 개발: Korea Community Health Survey 2017–2025, n=2,048,510
- 외적검증: NHANES 팬데믹 이전(n=7,888)·이후(n=5,210), 재적합 없음
- 성능: 내부 AUROC 0.836 (0.827–0.846), 외부 0.765 / 0.771
- 투고: JMIR, Algorithm Development and Validation Study

---

## 폴더 구성

| 폴더 | 내용 |
|---|---|
| `01_원고_생성및검증` | 원고·부록 원본(`docs/*.md`)과 생성·검사 스크립트 |
| `02_분석` | 조화(`src/`), 주분석, 탐색실험 |
| `03_결과` | 결과 CSV 81개, 그림 7장 |
| `04_제출본` | JMIR 제출 파일 일체 |
| `05_작업기록` | 수정 스크립트와 검토 기록 |

---

## 자료는 별도로 올린다

이 폴더에는 코드·결과·문서만 있다. 원자료와 처리자료는 따로 올린다.

분석이 실제로 읽는 것은 조화가 끝난 parquet 6개다.
`data/processed/` 아래 두면 주분석부터 그대로 돌아간다.

| 파일 | 내용 |
|---|---|
| `kchs_dev.parquet` | KCHS 개발표본 (훈련 500,000 + 홀드아웃 100,001) |
| `ext_kchs.parquet` | KCHS 전체 조화본 |
| `nhanes_pre.parquet` / `nhanes_post.parquet` | NHANES 두 주기 |
| `ext_pre.parquet` / `ext_post.parquet` | 외부검증용 조화본 |

원자료에서 다시 만들려면 `data/raw/kchs`, `data/raw/nhanes`를 두고
`02_분석/src/01_harmonize.py`부터 돌린다.

---

## 다시 돌리는 순서

원자료에서 시작한다면 `02_분석/src/01_harmonize.py`부터,
처리자료에서 시작한다면 주분석부터 돌린다.

```
python 70_primary_corrected.py     주분석 (부트스트랩 포함, 오래 걸림)
python 72_sensitivity_primary2.py  민감도
python 71_utility_primary2.py      임상유용성
python 73_figures_primary2.py      그림 7장
python 74_shap_cross.py            SHAP·교차일치도
```

**70 → 72 → 71 → 73 → 74 순서를 지킨다.** 70이 `primary2_sensitivity.csv`의 앞 네 행을
쓰고 72가 같은 파일을 전체로 다시 쓰므로, 72를 먼저 돌리면 뒷행이 사라진다.

문서를 다시 만들 때는 `01_원고_생성및검증`에서

```
python 61_sync_numbers.py          결과 CSV → 문서로 수치 주입
python 44_make_docx.py             원고 docx
python 45_make_appendix_docx.py    부록 docx
python 46_build_submission.py      제출본 폴더
python 60_verify_manuscript.py     검증 1,039건
python 92_crossref_audit.py        상호참조 검사
```

---

## 수치가 어긋나지 않게 하는 장치

수치를 손으로 옮겨 적다 보면 문서마다 세대가 갈린다. 실제로 그 문제가 반복돼서
다음 구조를 만들었다.

**`61_sync_numbers.py`** — 결과 CSV를 읽어 문서의 해당 구간을 다시 쓴다.
패턴이 맞지 않으면 조용히 넘어가지 않고 즉시 멈춘다.

**`60_verify_manuscript.py`** — 1,039건을 검사한다. CSV 값과 문서 문자열의 일치,
반올림 자릿수, 금지어, 대시, P값 표기, 표 칸 너비, 문서 신선도까지 본다.

**`92_crossref_audit.py`** — 절 이름·표·그림·참고문헌·약어가 실제로 존재하고
서로 맞는지 본다. 인용 어순, 표 번호 연속성, 편집 이력 표현 잔존도 검사한다.

문서를 고친 뒤에는 항상 이 셋을 돌린다.

---

## 작업 기록에 대해

`05_작업기록`에는 일회성 수정 스크립트 100여 개와 검토 기록이 있다.
각 스크립트 첫머리에 **무엇을 왜 고쳤는지**가 적혀 있다.

원고는 검토를 거치며 본문 11,563 → 7,006단어로 줄었고, 그 과정에서
사실이 빠지거나 표 참조가 어긋나는 일이 반복됐다. 검토에서 확정된 결함과
그 수정 내역이 `검토기록/*.json`에 남아 있다.

---

## 제출 전 채울 것

`04_제출본`의 다섯 자리는 비워 두었다.

| 위치 | 내용 |
|---|---|
| 원고 첫머리 | 저자명 |
| 원고 첫머리 | 교신저자 연락처 |
| 원고 윤리 절 | IRB 승인번호 |
| 부록1 윤리 절 | IRB 승인번호 |
| 커버레터 첫 줄 | 제출 날짜 |

문서 속성(작성자·설명)은 익명 심사를 위해 비워 두었다.
