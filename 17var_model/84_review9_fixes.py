# -*- coding: utf-8 -*-
"""84_review9_fixes.py — 코덱스 9차 교차검증 지적의 교정 (8차에서 새로 쓴 문단의 재검토 포함)."""
import io, os, sys
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__))

MS = [
 # ① 8차에서 새로 쓴 표본크기 문단이 표본을 잘못 서술했다
 ("The development sample is the full pooled KCHS 2017-2025 cohort and the external samples are the complete NHANES cycles, that is, all available data rather than a target computed in advance.",
  "The samples are those the data allow rather than a target computed in advance: development draws a 500,000-record training set and a 100,001-record hold-out from the 2,048,510 eligible pooled KCHS records, and each external cohort is every eligible respondent in that NHANES cycle."),
 # ② 구간은 AUROC·AUPRC 에만 붙는다
 ("The external cohorts are the binding constraint on precision, which is why interval widths, and not point estimates alone, are reported for every external quantity, and why subgroup estimates are presented with intervals and read as exploratory.",
  "The external cohorts are the binding constraint on precision, which is why discrimination is reported with confidence intervals rather than as point estimates alone, and why subgroup estimates are presented with intervals and read as exploratory. Intervals are computed for AUROC and AUPRC; calibration, operating characteristics and net benefit are reported as point estimates, as stated under Performance Evaluation."),
 # ③ 직렬화된 모형 객체는 존재하지 않고, 소득분위는 파이프라인 바깥에서 계산된다
 ("Analysis code, the variable-harmonization dictionary, the model card, the fitted LightGBM model object together with the preprocessing pipeline needed to score a new individual, and the protocol, statistical analysis plan and amendment documents",
  "Analysis code sufficient to refit the reported model exactly from the source data (fixed seed, fixed hyperparameters), the full preprocessing specification, including the household income quintile, which is derived within the target cohort before the fitted pipeline is applied and is therefore not reproducible from a single individual's record alone, the variable-harmonization dictionary, the model card, and the protocol, statistical analysis plan and amendment documents"),
 # ⑧ 세 단계 모두 계산했으나 표에 실린 것은 절편+기울기다
 ("Following the plan, recalibration was applied in three steps and all three are reported: intercept only, intercept plus slope (the two-parameter Platt-type update), and one-parameter temperature scaling with the intercept fixed at zero, each cross-fitted over five folds.",
  "Following the plan, recalibration was applied in three steps, each cross-fitted over five folds: intercept only, intercept plus slope (the two-parameter Platt-type update), and one-parameter temperature scaling with the intercept fixed at zero. Table 2 tabulates the intercept-plus-slope result, which is the recommended update; the other two steps are reported numerically in the Results."),
 # ⑧ 결과에 온도 조정 값을 실제로 적는다
 ("and temperature scaling was intermediate; intercept-plus-slope is therefore the recommended deployment update; H2 and H3 supported.",
  "and temperature scaling was intermediate, leaving the slope at 0.83 pre and 1.17 post with CITL +0.02 and -0.02; intercept-plus-slope is therefore the recommended deployment update; H2 and H3 supported."),
 # ⑨ 구간의 산출 방식
 ("with 1000 replicates", "with 1000 replicates, intervals taken as the 2.5th and 97.5th percentiles of the replicate distribution"),
 # ⑩ 이 연구는 디지털 중재의 이익을 보이지 않았다
 ("which places a low-burden self-report instrument within reach for digital triage, the delivery mode in which mobile interventions for depressive symptoms have shown benefit [40].",
  "which places a low-burden self-report instrument within reach for digital triage. Whether triage delivered this way improves depression outcomes was not evaluated here; mobile delivery of mental-health content has shown benefit in trials of intervention rather than of screening [40]."),
 # ⑪ 두 설계 선택은 그럴듯한 설명이지 입증된 설명이 아니다
 ("Two deliberate design choices explain why internal discrimination is lower (0.83):",
  "Two deliberate design choices plausibly account for the lower internal discrimination (0.83), although no ablation was run to attribute the difference quantitatively across studies that differ in population and outcome scale:"),
 # ⑫ 재보정의 '적은 자료로 충분' 은 검증되지 않았다
 ("intercept-plus-slope updating on modest local data was sufficient and is the recommended update.",
  "intercept-plus-slope updating restored calibration in both external cohorts, each using that cohort's full sample, and is the recommended update; the local sample size needed for a stable update was not examined here."),
 # ⑬ 이 문장은 앞 문장의 소득분위 결과에는 맞지 않는다
 ("The intervals are wide and mutually overlapping; hence these differences are not established, but a deployed screener",
  "The NHANES race and ethnicity intervals are wide and mutually overlapping, so those differences are not established; the highest-income KCHS estimate sits clear of the cohort interval and is the one flagged subgroup. Either way, a deployed screener"),
 # ⑭ 보정 기울기의 구간은 보고하지 않았다
 ("is derived [45], which is consistent with the width of the external intervals reported here.",
  "is derived [45]; no interval was computed for the calibration slope here, so that constraint is stated as a design consideration rather than as something the present intervals demonstrate."),
 # ⑮ 오라클 증분은 NHANES 안의 값이다
 ("and a development survey that captured multimorbidity and utilization could plausibly lift the transportable ceiling by about 0.01 AUROC.",
  "and a development survey that captured multimorbidity and utilization might raise the attainable ceiling, though the 0.013 increment observed here is an in-domain NHANES gain and is not itself evidence that the added signal would transport from Korea."),
 # ⑯ 표 S4 는 대부분 소수 첫째 자리다
 ("the accompanying missing shares are unweighted record proportions. Both, with the exact values, are given in Multimedia Appendix 1, Table S4.",
  "the accompanying missing shares are unweighted record proportions. Both are tabulated in Multimedia Appendix 1, Table S4."),
 # ⑱ 그림 7 의 임계값 범위
 ("Weighted net benefit of the model is plotted against threshold probability, with each dataset carrying its own screen-all and screen-none default strategies.",
  "Weighted net benefit of the model is plotted against threshold probability over the range 0.02 to 0.30, with each dataset carrying its own screen-all and screen-none default strategies."),
]
A2 = [
 # ④ 원자료 행수임을 분명히
 ("| KCHS 2017–2025 | 2,071,211 (all ages) | **2,048,510** |",
  "| KCHS 2017–2025 | 2,071,211 (before the ≥20 y and complete-PHQ-9 filters) | **2,048,510** |"),
]
CL = [
 # ⑤ 표본 크기를 맞춘 비교값은 0.81-0.82 다
 ("(fitting NHANES on its own labels reaches only ~0.77 across the learners tested, whereas the same predictors reach ~0.83 within Korea)",
  "(fitting NHANES on its own labels reaches only ~0.77 across the learners tested, whereas the same predictors reach 0.81-0.82 within Korea at matched sample size)"),
 # ⑥ 탐색적인 이유는 마진이 사후였기 때문이다
 ("a smaller-margin seven-item variant is reported and labelled exploratory, since it does not improve both external cohorts,",
  "a smaller-margin seven-item variant is reported and labelled exploratory, the 0.01 margin being post hoc while the plan fixed 0.02,"),
]
CK = [
 # ⑦ 커버레터와 같은 시제로
 ("the journal's official checklist form is uploaded separately with the submission.",
  "the journal's official checklist form will be uploaded with the submission."),
]

def edit(path, pairs):
    p = f"{D}/{path}"; s = io.open(p, encoding="utf8").read(); n = 0
    for a, b in pairs:
        if a not in s: print(f"  [미일치] {path}: {a[:76]}"); continue
        s = s.replace(a, b, 1); n += 1
    io.open(p, "w", encoding="utf8").write(s); print(f"  {path}: {n}/{len(pairs)}건")

print("코덱스 9차 지적 교정")
for f, pairs in [("docs/manuscript_17var.md", MS), ("docs/appendix2_crosswalk.md", A2),
                 ("docs/cover_letter.md", CL), ("docs/TRIPOD_AI_checklist.md", CK)]:
    edit(f, pairs)
