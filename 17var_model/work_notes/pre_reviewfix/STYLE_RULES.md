# 원고 작성 규칙 - 누적 정리

GBD 원고(*The Lancet Healthy Longevity* 투고본) 작업 중 지시하신 규칙을 모두 모은 것이다.
각 항목에 **원문 지시**와 **현재 우울 ML 원고(JMIR)에서의 상태**를 함께 적는다.

상태 표기 - ✅ 적용됨(60_verify_manuscript.py 가 기계적으로 감시) · ⏸ 확인 필요 · ➖ 저널이 달라 재해석 필요

---

## A. 문장·표현

| # | 규칙 | 원문 지시 | 상태 |
|---|---|---|---|
| A1 | 인칭·지시 대명사 금지 - we, our, ours, us, they, their, them, themselves | "we, they 같은 대명사 사용 금지" | ✅ 원고·부록·모델카드 0건 |
| A2 | `because` 금지 → given that · since · owing to · in view of | (이번 지시) | ✅ 0건 |
| A3 | 결과의 `so` 금지 → thus · hence · therefore (목적절 `so that` 은 허용) | "'so' 라고 표현되있는 부분은 저급한 표현이기 때문에 'thus', 'hence', 'therefore' 등으로 변경" | ✅ 0건 |
| A4 | em dash(-) → 괄호. 단 수치·연도 범위(2019-2021)는 en dash 유지 | "- 이렇게 생긴 모양 … 이거 괄호로 써야하거든" | ✅ 산문 0건 (표 셀·굵은 라벨 구분자만 유지, 검증기가 감시) |
| A5 | 축약어는 첫 등장 때 1회만 풀어 쓰고, 이후에는 축약어만 | "축약어는 한번 처음 등장할 때 설명하고 그냥 축약어만 사용" | ✅ 미정의 12건 정의·ECE 중복정의 1건 제거 |
| A6 | 근거를 넘어서는 단정 금지 (예: "robust", "not an artefact of modeling") | GBD 정정 이력 - 자기 부록과 모순되는 단정을 세 번 고침 | ✅ 금지어 검사에 포함 |

## B. 인용·참고문헌

| # | 규칙 | 원문 지시 | 상태 |
|---|---|---|---|
| B1 | 참고문헌 번호 연속 나열 금지 → 범위 표기(`4-7`). 연속 2개는 콤마 | "14, 15, 16, 17 이런식으로 표현하기에는 너무 많다" | ✅ 등장순 재번호, 3개 이상만 범위 |
| B2 | 참고문헌은 PubMed 등재 SCI급 저널 논문만 | "래퍼런스는 항상 PubMed에서 SCI급 저널에 등재된 논문만" | ✅ 30개 중 28개 PMID 대조 완료 |
| B3 | 논문이 아닌 참고문헌은 제목과 링크를 적는다 | 동일 지시 | ✅ SHAP(NeurIPS)·Lumley(J Stat Softw) 2건에 URL 부기 |
| B4 | 참고문헌 정보는 Codex와 함께 교차검증 | "하단의 래퍼런스 리스트들은 무조건 정확한 정보를 담아야 하기 때문에 Codex와 함께 찾아서 검토" | ✅ PubMed 메타데이터로 대조(오류 1건 정정) |
| B5 | 참고문헌 개수 | Lancet HL 30개 기준 | ✅ 16 → **30개** 확장 |

## C. 조판·서식

| # | 규칙 | 원문 지시 | 상태 |
|---|---|---|---|
| C1 | 전체 폰트 Times New Roman · 본문 11pt · 표/그림 제목·캡션 10pt · 표 내용 10pt | 원문 그대로 | ➖ JMIR 서식 확인 후 적용 |
| C2 | 표 디자인: 색 채우기 없음, 헤더 아래와 표 맨 아래에만 1pt 선, 페이지 넘김 시 자동 잘림, 페이지마다 헤더 반복 금지 | 원문 그대로 | ➖ |
| C3 | 표·그림 제목은 짧게. 표기법(Figure 1 / table 1 / Table 1.)은 저널 예시 그대로 | 원문 그대로 | ⏸ |
| C4 | 문서 전체 앞뒤 공백 제거 | "워드파일 전체에 앞뒤로 공백 제거" | ⏸ |
| C5 | placeholder 는 건드리지 말 것 | "placeholder은 건드리지 말것" | ✅ `[to be inserted]` 등 유지 |
| C6 | 본문 내 그림 인용 형식은 저널 예시 그대로 (`(figure 2; appendix pp 35-73)`) | 원문 그대로 | ➖ |

## D. 구조·분량

| # | 규칙 | 원문 지시 | 상태 |
|---|---|---|---|
| D1 | Discussion 에 소제목을 두지 않는다 (저널 예시에 없으면 없앤다) | "예시논문에도 없는데 단순히 내가 discussion 작성방법을 줬다고 소제목을 단거잖아" | ➖ JMIR은 소제목 허용 |
| D2 | 뒷단 순서: Contributors → Declaration of interests → Data sharing → Acknowledgments | "contributors 다음에 declaration of interests가 나와야하는데" | ⏸ |
| D3 | 분량 한도에 너무 빡빡하게 붙이지 않는다 | "본문 한도가 3500단어인데 … 3000~3300 정도로" | ✅ 초록 425/450 |
| D4 | Limitations·Conclusion 을 별도 절로 둘지는 저널 예시를 따른다 | 원문 지시 | ⏸ |

## E. 작업 방식

| # | 규칙 | 원문 지시 |
|---|---|---|
| E1 | 저널의 게재 예시논문을 **고정 틀**로 삼아 구조·표기·깊이를 복제한다 | "이 저널의 규칙에 맞게 작성을 해야해 … 세세하게 분석해서" |
| E2 | 검증은 Codex와 병행한다 | 반복 지시 |
| E3 | 데이터 다운로드는 사용자가 직접 한다 | "데이터 다운은 내가 할거야. 너가 임의로 하지마" |
| E4 | 프로토콜/사전지정을 사후에 바꾸지 않는다 - 개정은 날짜와 함께 공개한다 | 본 연구 진행 중 확립 |

---

## 기계 감시 현황

`60_verify_manuscript.py` 가 매 실행 시 540개 항목을 검사하며, 그중 문체 규칙은 A1·A2·A3·A4·A5·A6 과 B1 이다.
위반이 생기면 실패로 보고되므로 재발하지 않는다. C1~C4·D1·D2·D4 는 JMIR 투고규정과 예시논문을 확인한 뒤 확정한다.

---

## 예시파일(JMIR) 대조로 확정한 서식 - 2026-08-03

`윤수 예시파일/JMIR Adolescent Depressed Mood and Suicidality.docx`(초안)와
`jmir-2024-1-e55913_청소년REF.pdf`(실제 게재본)를 함께 대조해 확정했다.
둘이 다를 때는 **게재본**을 따랐다.

| 항목 | 확정값 |
|---|---|
| 폰트·크기 | Times New Roman 11pt |
| 여백 | 사방 1인치 |
| 줄간격 | 본문 2.0 (부록 1.5) |
| 정렬 | 양쪽맞춤 |
| 본문 첫줄 들여쓰기 | 18pt (0.25") · 제목·약어목록은 들여쓰기 없음 |
| 문단 앞뒤 공백 | 0 |
| 제목 | 굵게 + 파랑 `#2E74B5` |
| 표 | 캡션은 표 **위**, `Table N.` 라벨만 굵게 · 주석은 표 **아래** 10pt · 위첨자 문자(a, b, c) 로 칸과 연결 |
| 표 테두리 | 표 위·아래와 머리글 아래에만 1pt · 좌우·내부 없음 · 머리글 반복 없음 |
| 그림 | 캡션은 그림 **위** 10pt · 첫 인용 문단 뒤에 배치 · 캡션 설명 40~80단어 |
| 그림 규격 | 최소 300×225px, 최대 1200×900px, 5MB 이하 (JMIR 규정) |
| 참고문헌 | `N.<탭>저자(6명+et al). 제목. 학술지약어. Mon DD, YYYY;권(호):쪽. [doi: ] [Medline: ]` · 내어쓰기 0.45" · 10pt · 줄간격 1.5 |
| 절 번호 | 쓰지 않는다 · 절 상호참조(§)도 쓰지 않는다 |
| 제목 순서 | Abstract → Introduction → Methods → Results → Discussion → Acknowledgments → Funding → Data Availability → Authors' Contributions → Conflicts of Interest → Use of Generative AI Tools → Multimedia Appendix 1·2 → References → Abbreviations |
| Discussion 소제목 | Principal Findings · Comparison With Prior Work · Plausible Mechanisms · Clinical and Policy Implications · Strengths and Limitations · Conclusions |
| 부록 | 2개 (보충 방법·결과 / TRIPOD+AI 체크리스트) |
| 분량 | 초록 ≤450단어 · 본문 ≤10,000단어 권장 |
