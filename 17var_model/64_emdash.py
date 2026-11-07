# -*- coding: utf-8 -*-
"""64_emdash.py — em dash 정리(규칙 A4, "상황에 맞게 · 강조용도는 쓰지 않는다").

  · 삽입구(— … —)  → 괄호
  · 정의·열거를 여는 단독 dash → 콜론 또는 세미콜론
  · 강조용 dash → 쉼표로 낮추거나 문장을 나눈다
  · 남기는 것: 표 셀의 '해당 없음' 표시, 소제목 라벨 구분자(**Development — KCHS.**), 수치 범위의 en dash
"""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__))

def edit(path, pairs):
    p = f"{D}/{path}"; s = io.open(p, encoding="utf8").read(); n = 0
    for a, b in pairs:
        if a not in s: raise SystemExit(f"[em dash 정리 실패] 원문 미일치 in {path}:\n  {a[:110]}")
        s = s.replace(a, b, 1); n += 1
    io.open(p, "w", encoding="utf8").write(s); print(f"  {path}: {n}건")

# ══════════════ 생성식(61) 안의 문구 ══════════════
edit("61_sync_numbers.py", [
 ('"Within KCHS a few intervals sit clear of the cohort interval — " + "; ".join(parts)\n                   + " — hence subgroup discrimination is not uniform in the development population. ")',
  '"Within KCHS a few intervals sit clear of the cohort interval: " + "; ".join(parts)\n                   + ". Subgroup discrimination is therefore not uniform in the development population. ")'),
 ('flag_txt = ("Applying the fairness flag set in the analysis plan — a subgroup AUROC differing from the cohort value by more than 0.05 — "',
  'flag_txt = ("Applying the fairness flag set in the analysis plan (a subgroup AUROC differing from the cohort value by more than 0.05) "'),
])

# ══════════════ 원고 ══════════════
edit("docs/manuscript_17var.md", [
 ("could scale at low marginal cost—an application squarely within the scope",
  "could scale at low marginal cost, an application squarely within the scope"),
 ("Whether recalibration—updating only the intercept and slope without re-estimating predictor effects [11,12]—restores",
  "Whether recalibration (updating only the intercept and slope without re-estimating predictor effects [11,12]) restores"),
 ("statistical analysis plan—including the four hypotheses—were written and dated",
  "statistical analysis plan, including the four hypotheses, were written and dated"),
 ("split at the **cluster level** — every survey-year/stratum/primary sampling unit (PSU) is assigned entirely to one side — giving",
  "split at the **cluster level** (every survey-year/stratum/primary sampling unit (PSU) is assigned entirely to one side), giving"),
 ("did not improve—and slightly reduced—external discrimination (§3.4).",
  "did not improve, and slightly reduced, external discrimination (§3.4)."),
 ("in the epidemiological literature — socioeconomic position [16], sleep duration [17], physical activity [18], body mass index [19], smoking [20] and living alone [21] — and the three predictors covered by the protocol amendment are documented in §2.5.1.",
  "in the epidemiological literature: socioeconomic position [16], sleep duration [17], physical activity [18], body mass index [19], smoking [20] and living alone [21]. The three predictors covered by the protocol amendment are documented in §2.5.1."),
 ("in the epidemiological literature — food insecurity (pooled odds ratio ≈1.4) [22]",
  "in the epidemiological literature: food insecurity (pooled odds ratio ≈1.4) [22]"),
 ("surfaced further harmonizable items — public-assistance receipt",
  "surfaced further harmonizable items: public-assistance receipt"),
 ("crossed with two representations—main effects only, versus main effects plus four candidate interaction terms, specified when the search was written but not named in any earlier document (age×self-rated health, self-rated health×oral health, age×employment, income×employment)—evaluated by 4-fold survey-weighted cross-validation in a 300,000-record subsample of the training set;",
  "crossed with two representations: main effects only, versus main effects plus four candidate interaction terms (age×self-rated health, self-rated health×oral health, age×employment, income×employment; specified when the search was written but not named in any earlier document). The comparison used 4-fold survey-weighted cross-validation in a 300,000-record subsample of the training set;"),
 ("and recovery quantified—updating calibration only, without re-estimating predictor effects [11,12].",
  "and recovery quantified (updating calibration only, without re-estimating predictor effects [11,12])."),
 ("The item count follows the analysis plan's margin — the smallest set within 0.02 AUROC of the full model, subject to a 5–10 item ceiling; a variant",
  "The item count follows the analysis plan's margin (the smallest set within 0.02 AUROC of the full model, subject to a 5–10 item ceiling); a variant"),
 ("evaluation samples, and—as pre-specified in the protocol—a continuous-score analysis",
  "evaluation samples, and, as pre-specified in the protocol, a continuous-score analysis"),
 ("(DMDHHSIZ absent → constant imputation) — corrected to preserve missingness",
  "(DMDHHSIZ absent → constant imputation), corrected to preserve missingness"),
 ("string-typed unit fields mis-parsed to NaN — corrected with explicit unit mapping",
  "string-typed unit fields mis-parsed to NaN, corrected with explicit unit mapping"),
 ("placing non-respondents in the top income quintile — corrected.",
  "placing non-respondents in the top income quintile, corrected."),
 ("The surveys differed markedly on measured covariates—e.g., mean BMI",
  "The surveys differed markedly on measured covariates: mean BMI"),
 ("every characteristic in Table S6 uses available-case denominators)—consistent with genuinely distinct populations and, for food insecurity, with the single-item-vs-module measurement difference.",
  "every characteristic in Table S6 uses available-case denominators). This pattern is consistent with genuinely distinct populations and, for food insecurity, with the single-item-vs-module measurement difference."),
 ("Non-response was differential—higher among respondents",
  "Non-response was differential: higher among respondents"),
 ("calibration slope 1.03, CITL +0.10 — **H1 met**.",
  "calibration slope 1.03, CITL +0.10; **H1 met**."),
 ("is therefore the recommended deployment update — **H2 and H3 supported**.",
  "is therefore the recommended deployment update; **H2 and H3 supported**."),
 ("from 0.5M to 2M examples did not raise—and slightly lowered—external discrimination",
  "from 0.5M to 2M examples did not raise, and slightly lowered, external discrimination"),
 ("entering the six remaining items together—alcohol frequency, alcohol quantity in two codings, binge drinking and two public-assistance indicators—slightly *reduced* it",
  "entering the six remaining items together (alcohol frequency, alcohol quantity in two codings, binge drinking and two public-assistance indicators) slightly *reduced* it"),
 ("**shift in the predictor–outcome relationship itself**—concept shift—although the surveys",
  "**shift in the predictor–outcome relationship itself** (concept shift), although the surveys"),
 ("Applying the margin set in the analysis plan — the smallest set within 0.02 AUROC of the full model, subject to a 5–10 item ceiling — selects **six items**",
  "Applying the margin set in the analysis plan (the smallest set within 0.02 AUROC of the full model, subject to a 5–10 item ceiling) selects **six items**"),
 ("for the full 17-predictor model fitted on the same data — a loss of 0.015 internally, 0.019 pre-pandemic and 0.012 post-pandemic, each within the plan's 0.02 tolerance — **H4 supported**.",
  "for the full 17-predictor model fitted on the same data (a loss of 0.015 internally, 0.019 pre-pandemic and 0.012 post-pandemic, each within the plan's 0.02 tolerance); **H4 supported**."),
 ("or entering all three indicators together — the alternative alcohol definitions listed in the analysis plan — changed discrimination",
  "or entering all three indicators together (the alternative alcohol definitions listed in the analysis plan) changed discrimination"),
 ("self-rated oral health and food insecurity — the three items whose inclusion is documented in the protocol amendment (§2.5.1) — lowers external discrimination",
  "self-rated oral health and food insecurity (the three items whose inclusion is documented in the protocol amendment, §2.5.1) lowers external discrimination"),
 ("retained clinically useful—though lower—discrimination (~0.77)",
  "retained clinically useful, though lower, discrimination (~0.77)"),
 ("with a test R² of 33%—both identifying perceived stress",
  "with a test R² of 33%; both identifying perceived stress"),
 ("this study provides what prior work did not—genuine cross-national",
  "this study provides what prior work did not: genuine cross-national"),
 ("which may bias the external estimates—especially the post-pandemic subgroup results—in an unknown direction",
  "which may bias the external estimates (especially the post-pandemic subgroup results) in an unknown direction"),
 ("was written while the study was under way — after an earlier external result was known, though before any model using the amended set was fitted;",
  "was written while the study was under way, after an earlier external result was known but before any model using the amended set was fitted;"),
 ("not harmonizable to KCHS (physician-diagnosed multimorbidity — asthma, arthritis, thyroid, chronic obstructive pulmonary disease, liver disease, and cancer, excluding circulatory conditions — and healthcare utilization) were extracted",
  "not harmonizable to KCHS were extracted: physician-diagnosed multimorbidity (asthma, arthritis, thyroid, chronic obstructive pulmonary disease, liver disease, and cancer, excluding circulatory conditions) and healthcare utilization"),
 ("is a property of the transportable predictor set—most consistent with shift in the predictor–outcome relationship, with residual contributions from differing measurement, development-survey coverage and complete-case selection—rather than a modeling failure.",
  "is a property of the transportable predictor set, most consistent with shift in the predictor–outcome relationship and with residual contributions from differing measurement, development-survey coverage and complete-case selection, rather than a modeling failure."),
 ("Figure 1: Weighted sample characteristics across cohorts—panel A",
  "Figure 1: Weighted sample characteristics across cohorts: panel A"),
 ("Figure 4: Sensitivity analyses—weighted AUROC across specifications",
  "Figure 4: Sensitivity analyses: weighted AUROC across specifications"),
])
print("\n완료. 이어서 python 61_sync_numbers.py && python 60_verify_manuscript.py")
