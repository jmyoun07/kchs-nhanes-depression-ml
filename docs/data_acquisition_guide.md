# 데이터 확보 절차 가이드 (연구자 직접 수행)

> ⚠️ 다운로드는 **연구자 본인이 직접** 수행한다. 받은 파일은 아래 지정 위치에 넣는다.
> 받은 뒤 `src/01_harmonize.py`부터 코드가 동작한다.

## A. KCHS (지역사회건강조사) — 신청·심의 필요 (가장 먼저 착수)

**포털:** https://chs.kdca.go.kr (지역사회건강조사)
**근거:** 질병관리청 예규 제124호(원시자료 공개 규정). 공개 다운로드가 아니라 **요청–심의** 방식.

**절차**
1. **PHQ-9 수집 연도 먼저 확인** — `원시자료 이용지침서` Part 4(변수설명)에서 PHQ-9(우울) 문항이 실린 연도 특정. (신청서에 연도를 써야 하므로 선행)
2. 포털 계정 생성 + **이용서약** 동의.
3. **원시자료 자료요청** 제출 — 대상 **연도**, 필요 **변수**(아래 목록), **연구 목적** 기재.
4. **심의 대기** (~2–3영업일).
5. 승인 후 **다운로드** (링크 유효기간: 요청일로부터 1년).
6. 함께 제공되는 **이용지침서 + 조사표(설문지)** 도 반드시 받아 둔다 (연도별 변수코드·PHQ-9 수집연도 확인용).

**요청할 변수(개념 → KCHS 항목, 연도별 실제 코드는 이용지침서로 확인):**
PHQ-9(우울 9문항), 나이, 성별, 교육수준, 월 가구소득, 혼인상태, 가구원수/독거, 현재흡연, 월간음주/고위험음주, 걷기/중등도 신체활동, 수면시간, 신장·체중(BMI), 주관적건강, 고혈압 의사진단, 당뇨 의사진단.
(+ 보조 국내모델용: 스트레스 인지, EQ-5D/EQ-VAS, 사회적 지지, 주관적 인지저하 — 선택)

**배치 위치:** `data/raw/kchs/`

---

## B. NHANES (미국) — 공개 다운로드 (신청 불필요)

**포털:** https://wwwn.cdc.gov/nchs/nhanes/
컴포넌트: Demographics / Questionnaire / Examination / Laboratory. 각 파일에 온라인 코드북 존재. **SAS transport(.XPT)** 로 받는다.

**받을 파일 (두 시기 × 10개 컴포넌트):**

| 컴포넌트 | pre (2017–2020.3) | post (2021.8–2023.8) | 용도 |
|---|---|---|---|
| Demographics | P_DEMO | DEMO_L | 나이·성별·교육·소득(PIR)·혼인·가구크기·인종 |
| Depression (PHQ-9) | P_DPQ | DPQ_L | 결과변수 |
| Smoking | P_SMQ | SMQ_L | 흡연 |
| Alcohol | P_ALQ | ALQ_L | 음주 |
| Phys. Activity | P_PAQ | PAQ_L | 신체활동 |
| Sleep | P_SLQ | SLQ_L | 수면 |
| Body Measures | P_BMX | BMX_L | BMI(실측) |
| 자가건강(General Health) | **P_HUQ** (HUQ010) | **HUQ_L** | 자가건강 5단계 |
| Current Health(참고) | P_HSQ | HSQ_L | HSQ590만(자가건강 아님) |
| Blood Pressure | P_BPQ | BPQ_L | 고혈압 진단 |
| Diabetes | P_DIQ | DIQ_L | 당뇨 진단 |

**병합:** 한 주기 내에서 **SEQN(응답자 일련번호)** 로 컴포넌트 병합 → strata/PSU + MEC/interview 가중치 적용(NHANES Analytic Guidelines). ⚠️ pre는 3.2년 결합가중치, post는 해당 주기 가중치 — 시기 간 데이터를 합치지 말고 **각각 별도 검증셋**으로 유지.

**배치 위치:** `data/raw/nhanes/` (파일명 그대로)

---

## C. 분석 착수 전 확인 목록 (원 프로토콜 Appendix A)

- [ ] KCHS에서 PHQ-9가 수집된 정확한 연도 (이용지침서 Part 4)
- [ ] Table 1 각 항목의 연도별 KCHS 원시 변수코드
- [ ] NHANES 시기별 우울증 사건수 산출 → 보정·DCA 정밀도 확인
- [ ] 음주·신체활동·소득의 공통 조작적 정의 확정
- [ ] NHANES 주기별 변수명 확인 (DMDMARTZ vs DMDMARTL; ALQ 항목)

> 위 확인이 끝나면 `docs/harmonization_dictionary`(변수 조화 사전)를 채우고 `src/01_harmonize.py`로 넘어간다.
