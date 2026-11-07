# -*- coding: utf-8 -*-
"""62_style_rules.py — 교수님 문체 규칙을 원고·부록에 적용(1회성 교정).

적용 규칙 (GBD 원고에서 확립된 것):
  R1  인칭·지시 대명사 금지: we / our / ours / us / they / their / them / themselves
  R2  because 금지 → given that · since · owing to · in view of 등 격식 표현
  R3  결과의 'so' 금지 → thus / hence / therefore  ('so that' 목적절은 유지)

한 번 실행해 문서를 고치고, 이후 재발은 60_verify_manuscript.py 가 잡는다.
생성식 문구는 61_sync_numbers.py 안에서 이미 규칙에 맞게 작성되어 있어야 한다.
"""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__)); DOCS = f"{D}/docs"

def edit(path, pairs, strict=True):
    p = f"{DOCS}/{path}"; s = io.open(p, encoding="utf8").read(); n = 0
    for a, b in pairs:
        if a not in s:
            if strict: raise SystemExit(f"[문체교정 실패] 원문 미일치 in {path}:\n  {a[:110]}")
            continue
        s = s.replace(a, b); n += 1
    io.open(p, "w", encoding="utf8").write(s); print(f"  {path}: {n}건 교정")

# ══════════════ 원고 ══════════════
edit("manuscript_17var.md", [
 # ── R1 대명사 ──
 ("only; whether they transport to another country", "only; whether such models transport to another country"),
 ("and of their behavioral correlates changed during the pandemic",
  "and of the associated behavioral characteristics changed during the pandemic"),
 ("We pursued four aims:", "This study pursued four aims:"),
 ("We hypothesized:", "The pre-specified hypotheses were:"),
 ("before any model using them was fitted", "before any model using those items was fitted"),
 ("After the primary analysis we screened all of them by comparing the direction of their adjusted odds ratios with depression across the three cohorts and then testing them in the model.",
  "After the primary analysis every candidate was screened by comparing the direction of its adjusted odds ratio with depression across the three cohorts and then testing it in the model."),
 ("We verified the implementation against", "The implementation was verified against"),
 ("are pooled into one pseudo-stratum so their variance contribution is not lost",
  "are pooled into one pseudo-stratum, ensuring that the variance contribution of those strata is retained"),
 ("adding respondents missing only one or two items using their person-mean",
  "adding respondents missing only one or two items using the person-mean"),
 ("We localized the ceiling with an **in-domain oracle**", "The ceiling was localized with an **in-domain oracle**"),
 ("an empirical benchmark over the learners we tried", "an empirical benchmark over the learners evaluated"),
 ("explain why our internal discrimination is lower (0.83): we excluded outcome-proximal predictors such as perceived stress and single-item depressed-mood questions, which inflate apparent performance but are near-tautological with the outcome; and we restricted the predictor set to variables transportable to a second country.",
  "explain why internal discrimination is lower (0.83): outcome-proximal predictors such as perceived stress and single-item depressed-mood questions (which inflate apparent performance but are near-tautological with the outcome) were excluded; and the predictor set was restricted to variables transportable to a second country."),
 ("without numeric decision margins, so judging them 'supported' involves interpretation",
  "without numeric decision margins; thus judging those two hypotheses 'supported' involves interpretation"),
 ("In a coverage analysis we extracted candidate domains available in NHANES but not harmonizable to KCHS—physician-diagnosed multimorbidity (asthma, arthritis, thyroid, chronic obstructive pulmonary disease, liver disease, and cancer, excluding circulatory conditions) and healthcare utilization—and found that adding them raised the NHANES in-domain oracle from 0.765 to 0.778.",
  "In a coverage analysis, candidate domains available in NHANES but not harmonizable to KCHS (physician-diagnosed multimorbidity — asthma, arthritis, thyroid, chronic obstructive pulmonary disease, liver disease, and cancer, excluding circulatory conditions — and healthcare utilization) were extracted; adding those domains raised the NHANES in-domain oracle from 0.765 to 0.778."),
 # ── R2 because ──
 ("uses current drinking alone because the NHANES quantity item was not carried through harmonization",
  "uses current drinking alone, given that the NHANES quantity item was not carried through harmonization"),
 ("is inactive in the deployed configuration because the subsampling frequency was left at its default of zero, so the fitted model uses all rows at every boosting iteration.",
  "is inactive in the deployed configuration, given that the subsampling frequency was left at its default of zero; the fitted model therefore uses all rows at every boosting iteration."),
 ("Rescaling is necessary because most NHANES strata contain only two primary sampling units",
  "Rescaling is necessary given that most NHANES strata contain only two primary sampling units"),
 ("crossed with the design stratum, because weights are normalized within year and stratum codes recur across years.",
  "crossed with the design stratum, owing to weight normalization within year and to the recurrence of stratum codes across years."),
 ("the income quintile collapsed to four levels in the post-pandemic cohort because the income-to-poverty ratio is top-coded at 5.00",
  "the income quintile collapsed to four levels in the post-pandemic cohort, owing to top-coding of the income-to-poverty ratio at 5.00"),
 ("(5-fold cross-fit); because each fold uses a separate monotone calibration map",
  "(5-fold cross-fit); since each fold uses a separate monotone calibration map"),
 # ── R3 'so' (결과) ── 목적절 'so that' 은 건드리지 않는다
 ("that share no cluster with it, so the internal estimate refers to new clusters",
  "that share no cluster with it; hence the internal estimate refers to new clusters"),
 ("as a post-hoc diagnostic; no item was added, so the predictor set is unchanged.",
  "as a post-hoc diagnostic; no item was added, and the predictor set is therefore unchanged."),
 ("and a within-survey income quintile, so no alternative definition of those two was available.",
  "and a within-survey income quintile; hence no alternative definition of those two was available."),
 ("of the weighted KCHS sample is older (up to 110), so the model learns splits above 80",
  "of the weighted KCHS sample is older (up to 110); hence the model learns splits above 80"),
 ("lacks the moderate-activity day count so the indicator falls back on the remaining components",
  "lacks the moderate-activity day count, and the indicator therefore falls back on the remaining components"),
 ("whereas NHANES asks about weekdays, so the later KCHS years sit on a different definition.",
  "whereas NHANES asks about weekdays; hence the later KCHS years sit on a different definition."),
 ("handled natively by the tree learner—so structurally missing binary items such as living-alone in NHANES pre are never imputed to a constant—while",
  "handled natively by the tree learner (structurally missing binary items such as living-alone in NHANES pre are therefore never imputed to a constant), while"),
 ("rather than as unordered categories, so they were median-imputed and standardized.",
  "rather than as unordered categories, and were median-imputed and standardized."),
 ("is top-coded at 5.00 for 23.8% of respondents, so the top two quintiles collapse",
  "is top-coded at 5.00 for 23.8% of respondents; hence the top two quintiles collapse"),
 ("and then calibration slope without constructing the tie band, so the plan's rule was checked retrospectively.",
  "and then calibration slope without constructing the tie band; the plan's rule was therefore checked retrospectively."),
 ("differ by 0.0046 and therefore fall inside the tie band, so the plan's tie-break applies",
  "differ by 0.0046 and therefore fall inside the tie band, to which the plan's tie-break applies"),
 ("and the parsimony criterion is not reached, so the retrospective check reproduces the model",
  "and the parsimony criterion is not reached; thus the retrospective check reproduces the model"),
 ("among Asian and Black respondents—so the external cohorts, and particularly the post-pandemic fairness estimates, should be read as complete-case results",
  "among Asian and Black respondents; hence the external cohorts, and particularly the post-pandemic fairness estimates, should be read as complete-case results"),
 ("the two cycles are statistically indistinguishable, so discrimination did not deteriorate across the pandemic.",
  "the two cycles are statistically indistinguishable; hence discrimination did not deteriorate across the pandemic."),
 ("and temperature scaling was intermediate — so intercept-plus-slope is the recommended deployment update",
  "and temperature scaling was intermediate; intercept-plus-slope is therefore the recommended deployment update"),
 ("is the offset-model calibration intercept, so positive values denote under-prediction",
  "is the offset-model calibration intercept; hence positive values denote under-prediction"),
 ("and its interval overlaps the transported estimates, so the comparison is approximate",
  "and its interval overlaps the transported estimates; thus the comparison is approximate"),
 ("with differential depression-module non-response (§3.1), so measurement and selection effects cannot be separated",
  "with differential depression-module non-response (§3.1); hence measurement and selection effects cannot be separated"),
 ("No smaller-margin variant is reported, so no external information entered the item-count decision.",
  "No smaller-margin variant is reported; hence no external information entered the item-count decision."),
 ("weighted R² 0.216/0.166/0.102), so the transportability pattern is not an artifact",
  "weighted R² 0.216/0.166/0.102); thus the transportability pattern is not an artifact"),
 ("and then applied unchanged to the hold-out and to both external cohorts, so the external operating characteristics are transported values",
  "and then applied unchanged to the hold-out and to both external cohorts; hence the external operating characteristics are transported values"),
])

# ══════════════ 부록 2 ══════════════
edit("appendix2_crosswalk.md", [
 ("The locked document numbers them food insecurity, employment, oral health; this table reorders them for readability without changing the set.",
  "The locked document lists these three as food insecurity, employment and oral health; the present table reorders the entries for readability without changing the set."),
 ("Pre-specified for exclusion because they re-measure the outcome construct rather than predict it:",
  "Pre-specified for exclusion, given that the following re-measure the outcome construct rather than predict it:"),
 ("| KCHS `mtb_07a1`–`mtb_07i1` | the PHQ-9 items themselves (the outcome) |",
  "| KCHS `mtb_07a1`–`mtb_07i1` | the PHQ-9 items constituting the outcome |"),
 ("rather than physician diagnosis, so they do not match the NHANES concept",
  "rather than physician diagnosis; hence those items do not match the NHANES concept"),
 ("in Multimedia Appendix 3 Table S11. They comprise:", "in Multimedia Appendix 3 Table S11. These comprise:"),
 ("were screened by comparing the direction of their adjusted associations with depression across the three cohorts.",
  "were screened by comparing the direction of the adjusted associations with depression across the three cohorts."),
 ("**External AUROC of 0.80 was explicitly not expected**, because earlier work had already localized the external gap",
  "**External AUROC of 0.80 was explicitly not expected**, since earlier work had already localized the external gap"),
 ("no item was added, because those with a consistent direction added no external discrimination",
  "no item was added, given that those with a consistent direction added no external discrimination"),
])

# ══════════════ 부록 3 ══════════════
edit("appendix3_utility.md", [
 ("## Table S11. Pre-specified selection rules and how they were applied",
  "## Table S11. Pre-specified selection rules and their disposition"),
 ("Other pre-specified rules and their disposition: the parsimony margin",
  "Other pre-specified rules were applied as follows: the parsimony margin"),
 ("but the harmonized variable uses current drinking alone because the NHANES quantity item was not carried through harmonization",
  "but the harmonized variable uses current drinking alone, given that the NHANES quantity item was not carried through harmonization"),
 ("and the prior-depression-diagnosis exclusion could not be run because KCHS records only self-reported BMI and no prior diagnosis",
  "and the prior-depression-diagnosis exclusion could not be run, given that KCHS records only self-reported BMI and no prior diagnosis"),
])

# ══════════════ 부록 4 ══════════════
edit("appendix4_ceiling.md", [
 ("what is quarantined to the oracle rows is their use for model *fitting*",
  "what is quarantined to the oracle rows is the use of those labels for model *fitting*"),
 ("held against every leakage-controlled attempt we made", "held against every leakage-controlled attempt made"),
 ("the single density-ratio correction we ran shows only that that procedure did not help",
  "the single density-ratio correction applied shows only that the procedure did not help"),
 ("both already included), so they cannot enter a KCHS-developed transportable model",
  "both already included); hence those domains cannot enter a KCHS-developed transportable model"),
 ("An in-sample (apparent) bound is not reported because it is strongly learner-dependent",
  "An in-sample (apparent) bound is not reported, given that it is strongly learner-dependent"),
])

# ══════════════ 모델카드 ══════════════
edit("model_card_17var.md", [
 ("differing measurement, development-survey coverage and complete-case selection cannot be separated from it in these data.",
  "differing measurement, development-survey coverage and complete-case selection cannot be separated from concept shift in these data."),
], strict=False)

print("\n완료. 이어서 python 61_sync_numbers.py && python 60_verify_manuscript.py")
