# -*- coding: utf-8 -*-
"""77_review3_fixes.py — 코덱스 3차 교차검증 지적의 교정 (본문·부록·체크리스트).

계산이 필요한 4·10번(내부 코호트 순편익 재보정의 이중기준, 유병률 임계값의 비가중 분위수)은
71_utility_primary2.py 를 고쳐 다시 계산한다.
"""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__))

MS = [
 # ① 재보정 기울기 반올림 (0.998/0.996 → 1.00)
 ("was restored to slope 0.99 and CITL 0.00 by intercept/slope recalibration",
  "was restored to slope 1.00 and CITL 0.00 by intercept/slope recalibration"),
 ("intercept-plus-slope updating restored both (slope 0.99 and 0.99, CITL 0.00)",
  "intercept-plus-slope updating restored both (slope 1.00 and 1.00, CITL 0.00)"),
 # ② 유병률 8.45% → 8.5% 로 통일
 ("weighted prevalence 8.4% and 12.2%) govern external", "weighted prevalence 8.5% and 12.2%) govern external"),
 ("Weighted depression prevalence (PHQ-9 ≥10): KCHS 3.6%, NHANES pre 8.4%, post 12.2%",
  "Weighted depression prevalence (PHQ-9 ≥10): KCHS 3.6%, NHANES pre 8.5%, post 12.2%"),
 ("as expected from the prevalence difference (KCHS 3.6% vs 8.4% and 12.2%)",
  "as expected from the prevalence difference (KCHS 3.6% vs 8.5% and 12.2%)"),
 # ③ 식품불안정 SHAP 순위 (NHANES pre 에서 8위)
 ("which ranked 10th in KCHS but 7th in NHANES pre", "which ranked 10th in KCHS but 8th in NHANES pre"),
 # ⑤ 밀도비 보정 실패로 공변량이동을 배제할 수 없다
 ("indicating the gap is not covariate shift;",
  "showing that this particular correction did not help, which weakens but does not exclude a covariate-shift explanation;"),
 # ⑥ 고찰의 '강도 대 구조' 단정을 결과의 유보와 맞춘다
 ("what differs is the strength of the predictor-outcome associations, not the structure of those associations, a pattern",
  "what differs is how strongly the model leans on each predictor rather than the order in which it does so. That is a statement about the fixed model's attribution and not about the external predictor-outcome associations, which the attribution never observes; it is a pattern"),
 # ⑦ 초록의 '모든 선택은 개발자료만' 을 정확히
 ("All selection used KCHS-internal cross-validation only;",
  "All model fitting, hyperparameter tuning and algorithm selection used KCHS-internal cross-validation only, while three of the seventeen predictors entered through a dated amendment written after an earlier external result was known;"),
 # ⑧ '동일한 튜닝 노력' 은 LightGBM 의 선행 탐색과 어긋난다
 ("the three candidate algorithms received equal tuning effort",
  "the three candidate algorithms were compared at each one's tuned best under one identical protocol, after a preceding gradient-boosting-only search that is reported in full"),
 # ⑯ NHANES 설계 인용은 2015-2018 만 다룬다
 ("NCHS survey combining household interview and mobile examination center assessment [18]",
  "NCHS survey combining household interview and mobile examination center assessment; the sample design and estimation procedures are documented for the 2015-2018 cycles [18], and the same framework applies to the pre-pandemic file and the 2021-2023 cycle used here"),
 # ⑮ PROBAST 판정 결과를 실제로 보고
 ("risk of bias self-assessed with the Prediction model Risk Of Bias Assessment Tool (PROBAST) [14]",
  "risk of bias self-assessed with the Prediction model Risk Of Bias Assessment Tool (PROBAST) [14]; the self-assessment rated participants, predictors and outcome domains at low risk and the analysis domain at low risk for the development data, with an unclear rating for the external cohorts because of differential depression-module non-response, giving an overall low-to-unclear rating driven by that non-response"),
]
A3 = [
 # ⑫ 부록 머리 문장의 표 번호가 낡음
 ("Supports Results Participant Characteristics (participant characteristics and depression-module non-response, Tables S6 and S10)",
  "Supports Results Participant Characteristics (sample characteristics in Table S4 and depression-module non-response in Table S8)"),
 # ⑬ 존재하지 않는 근거파일
 ("Source: `results/dpq_missingness.txt`.", "Computed from the NHANES DPQ item files; the derivation is released with the analysis code."),
 # ⑪ 7항목 변형의 표현을 본문과 맞춘다
 ("a stricter 0.01 margin giving 7 items is reported as a post hoc, externally informed variant",
  "a stricter 0.01 margin giving 7 items is reported as an exploratory variant; it was preferred only on external evidence and is therefore not the deployed model"),
]
A4 = [
 # ⑨ 외부 라벨을 본 행이 '누수 통제' 표에 섞여 있다
 ("## Table S11. Leakage-controlled attempts to exceed the external ceiling",
  "## Table S11. Attempts to exceed the external ceiling"),
 ("Every learner and adaptation", "Except where a row is marked as externally screened, every learner and adaptation"),
]
CK = [("Comparison With Prior Work (ref 10)", "Comparison With Prior Work")]

def edit(path, pairs):
    p = f"{D}/{path}"; s = io.open(p, encoding="utf8").read(); n = 0
    for a, b in pairs:
        if a not in s: print(f"  [미일치] {path}: {a[:76]}"); continue
        s = s.replace(a, b); n += 1
    io.open(p, "w", encoding="utf8").write(s); print(f"  {path}: {n}/{len(pairs)}건")

print("코덱스 3차 지적 교정")
edit("docs/manuscript_17var.md", MS)
edit("docs/appendix3_utility.md", A3)
edit("docs/appendix4_ceiling.md", A4)
edit("docs/TRIPOD_AI_checklist.md", CK)
