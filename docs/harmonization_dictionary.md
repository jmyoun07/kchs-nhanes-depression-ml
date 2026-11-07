# 변수 조화 사전 (KCHS ↔ NHANES) — 실측 기반

작성 2026-07-08 · 양 데이터의 실제 변수명·라벨을 읽어 확정. 값 코딩(0–3 등)은 실제 데이터/이용지침서로 재확인 예정.
개발셋 KCHS = **2017–2025**(주분석). NHANES 검증 = **pre(P_)** / **post(_L)**.

> ✅ **[2026-07-10 정정] 자가건강 NHANES 매핑 = `HSQ·HSD010` → `HUQ·HUQ010`** (General health, 1=Excellent…5=Poor; P_HUQ 결측0%, HUQ_L 확인). KCHS `qoa_01z1`과 동일 5단계·방향. → **§2.5 교집합 충족, 프로토콜 Table 1대로 조화모델에 포함**. 이로써 조화모델 = **Table 1 전체 14변수**(자가건강·독거·신체활동·음주 포함). 상세 [review/protocol_erratum.md](review/protocol_erratum.md). 아래 표의 "자가건강 → NHANES 없음/제외" 서술은 이 정정으로 대체됨.

## 조화 예측변수 매핑

| 개념 | KCHS 변수 (연도 주의) | NHANES pre `P_` | NHANES post `_L` | 조화 정의 | 상태 |
|---|---|---|---|---|---|
| **결과: 우울** | `mtb_07a1`~`mtb_07i1` (9문항) | `DPQ010`–`DPQ090` | `DPQ010`–`DPQ090` | 합 ≥10 | ✅ 코딩 0–3 확인 |
| 연령 | `age` | `RIDAGEYR` | `RIDAGEYR` | 연속+10세 구간 | ✅ 전연도 |
| 성별 | `sex` | `RIAGENDR` | `RIAGENDR` | 남/녀 | ✅ |
| 교육 | `sob_01z1`(+`sob_02z1`) | `DMDEDUC2` | `DMDEDUC2` | 4수준 | ✅ (코딩확인) |
| 소득 | `fma_14z1`(월)/`fma_13z1`(년) *2018+* | `INDFMPIR` | `INDFMPIR` | 설문내 5분위 | ⚠ 2017 코드상이 |
| 혼인 | `sod_02z3` *2020+* / 이전 別코드 | `DMDMARTZ` | `DMDMARTZ` | 유배우 vs 무 | ⚠ 2017–19 별코드 |
| 독거 | `mbhld_co`(가구원수) *2020+* | `DMDHHSIZ` | `DMDHHSIZ` | ==1(독거) | ⚠ 이전 별코드 |
| 인종 | (없음) | `RIDRETH3` | `RIDRETH3` | **공정성 하위군 전용** | NHANES only |
| 흡연 | `sma_03z2`(현재)+`sma_01z1`(평생) | `SMQ020`,`SMQ040` | `SMQ020`,`SMQ040` | never/former/current | ✅ |
| 음주 | `dra_01z1`,`drb_01z3`,`drb_03z1`,`drb_04/05z1` | `ALQ111`,`ALQ121`,`ALQ130` | 동일 | 현재음주+위험음주 | ⚠ 도구차이 |
| 신체활동 | `pha_07z1`(중등도),`phb_01z1`(걷기),`pha_04z1`(격렬) | `PAQ605`,`620`,`650`,`665` (GPAQ) | `PAD790Q`,`800`,`810Q`,`820` (약식 LTPA) | 유산소 권장 충족(y/n) | ⚠⚠ NHANES 전/후 도구 다름 |
| 수면 | `mtc_01z1`(2017–19)/`mtc_17z1`+`mtc_18z1`(2020–22,24,25) | `SLD012` | `SLD012` | 시간+단/정상/장 | ⚠ **KCHS 2023 없음** |
| BMI | `oba_bmi`(자가) / `oba_03z1`(몸무게)+신장 | `BMXBMI`(실측) | `BMXBMI` | 연속 BMI | ⚠ 자가(KCHS) vs 실측(NHANES) |
| 자가건강 | `qoa_01z1`(1–5) | `HUQ·HUQ010` | `HUQ·HUQ010` | 5단계 서열 | ✅ **포함**(HSQ→HUQ 정정) |
| 고혈압 | `hya_04z1` | `BPQ020` | `BPQ020` | 진단 y/n | ✅ |
| 당뇨 | `dia_04z1` | `DIQ010` | `DIQ010` | 진단 y/n | ✅ |

## 복합표본 설계 변수
| | KCHS | NHANES pre | NHANES post |
|---|---|---|---|
| 가중치 | `wt_p`(개인)/`wt_h`(가구) *2019+* | `WTINTPRP`/`WTMECPRP` | **`WTINT2YR`/`WTMEC2YR`** |
| 층화 | `kstrata` | `SDMVSTRA` | `SDMVSTRA` |
| 집락(PSU) | `SPOT_NO` *2021+* | `SDMVPSU` | `SDMVPSU` |

## ⚠ 프로토콜 Table 1과 달라진 점 (실측으로 확인)

1. **자가건강(self-rated health) → 조화모델에 포함(정정).** Table 1은 `HSQ·HSD010`으로 적었으나 실제 자가건강은 NHANES **`HUQ·HUQ010`**(General health 5단계, P_HUQ 결측0%·HUQ_L 확인)에 존재. KCHS `qoa_01z1`과 동일 5단계 → §2.5 교집합 충족 → 조화모델 포함. (초기엔 HSQ만 봐서 오판)
2. **신체활동 — NHANES 전/후 도구가 다름.** pre=`P_PAQ`(GPAQ 전체: 일/이동/여가), post=`PAQ_L`(약식 여가시간 LTPA만, `PAD790Q`/`PAD810Q`). `PAQ650`/`665`가 2021–2023엔 **없음** → "유산소 권장 충족"을 **시기별로 다르게 산출**해야 함(전이성 한계로 명시·민감도).
3. **수면 — KCHS 2023년만 없음**, 그리고 변수명이 연도별 상이(`mtc_01z1` ↔ `mtc_17z1`/`18z1`).
4. **KCHS 변수명 연도별 변화**(프로토콜이 경고한 사항): 소득·혼인·독거·가중치·PSU 등이 특히 **2019/2020 전후로 코드가 바뀜** → 개발셋(2017–2025)은 **연도별 크로스워크** 필요. (2020–2025는 비교적 안정적)
5. **NHANES 가중치명이 시기별 다름**: pre `WTINTPRP/WTMECPRP`(약 3.2년) vs post `WTINT2YR/WTMEC2YR`(2년).

## 실측 확정 사항 (2026-07-08 데이터 기반)
- **KCHS 신장=`oba_02z1`(키), 체중=`oba_03z1`, 자가BMI=`oba_bmi`**(2020·2023·2024). 2021·2022는 키+몸무게로 BMI 계산. (`oba_01z1`은 '본인 인지 체형'이라 신장 아님)
- **PHQ-9 KCHS는 1–4 코딩 → 1을 빼서 0–3으로 재코딩** 후 합산(NHANES는 이미 0–3).
- **조화모델에서 최종 제외된 2변수**(교집합 원칙):
  - `living_alone`: NHANES **pre(P_DEMO)에 DMDHHSIZ 없음** → pre 100% 결측 → 제외
  - `active`(신체활동): NHANES pre(GPAQ) vs post(약식 LTPA) 도구 상이 + KCHS 근사 → 제외(추후 별도)
- **자가건강**(`qoa_01z1` ↔ NHANES `HUQ010`)는 조화모델에 **포함**(정정, 위 §참조).
- **최종 조화 예측변수 11개**: age, sex, edu(4), income(설문내 5분위), partnered, smoking(3), current_drinker, sleep_hrs, bmi, htn, dm. (+race=NHANES 공정성 전용)
- 주개발셋 = **KCHS 2020–2024**(코드 검증 완료, n≈114만, 우울 3.6%). 2017–2019/2025는 크로스워크 확정 후 확장.
- 검증 유병률: NHANES **pre 9.2% → post 12.7%**(코로나 후 증가 반영).

## 다음 작업
1. 실제 데이터 값으로 **값 코딩 확인**(PHQ-9 0–3, 성별·교육·흡연 등) — 이용지침서 없이 데이터에서 직접.
2. 연도별 KCHS 크로스워크 확정 → `src/01_harmonize.py`.
3. 결과변수(PHQ-9≥10) 산출 + 적격성(성인·9문항 완결).
