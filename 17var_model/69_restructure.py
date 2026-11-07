# -*- coding: utf-8 -*-
"""69_restructure.py — 게재본(J Med Internet Res 2024;26:e55913)에 맞춘 본문 재구성.

  A) Methods 소제목 14 → 6 (흡수된 제목은 굵은 러닝헤드로 문단 앞에 남긴다)
  B) Discussion 을 게재본 구성으로: Principal Findings / Comparison With Prior Work /
     Plausible Mechanisms for the External Ceiling / Clinical and Policy Implications /
     Strengths and Limitations / Conclusions
  C) Introduction 확장 (384 → 약 800단어)
  D) 수치 강조 굵게 제거 — 게재본에는 없다. 남기는 굵게는 초록 라벨·러닝헤드·표 라벨뿐
"""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__)); P = f"{D}/docs/manuscript_17var.md"
s = io.open(P, encoding="utf8").read()

# ══════════════ A. Methods 소제목 통합 ══════════════
RENAME = {"Study Design and Reporting Standards": "Study Design and Data Sources",
          "Participants": "Participants and Outcome",
          "Candidate Predictors and Harmonization": "Predictors and Harmonization",
          "Sample Size": "Model Development and Evaluation",
          "External Validation Strategy": "External Validation and Sensitivity Analyses"}
ABSORB = {"Data Sources": "Data sources.", "Outcome": "Outcome.", "Missing Data": "Missing data.",
          "Model Development": "Algorithms and selection.", "Performance Evaluation": "Performance evaluation.",
          "Interpretability, Fairness, and Deployment": "Interpretability, fairness, and deployment.",
          "Planned Sensitivity Analyses": "Planned sensitivity analyses.",
          "Data-Integrity Audit": "Data-integrity audit."}
for old, new in RENAME.items():
    assert f"### {old}\n" in s, f"제목 없음: {old}"
    s = s.replace(f"### {old}\n", f"### {new}\n", 1)
for old, label in ABSORB.items():
    pat = f"### {old}\n"
    assert pat in s, f"제목 없음: {old}"
    i = s.index(pat); j = i + len(pat)
    nxt = s[j:j + 3]
    s = s[:i] + (f"**{label}** " if nxt != "**" else "") + s[j:]
print(f"A) Methods 소제목 통합 — 유지 6개, 러닝헤드로 흡수 {len(ABSORB)}개")

# ══════════════ C. Introduction 확장 ══════════════
INTRO_ADD_1 = (" National health-examination and community-health surveys already collect the sociodemographic, behavioural "
    "and clinical items such a tool would need, and are administered to hundreds of thousands of respondents each year, "
    "which makes such surveys an attractive development base for screening algorithms intended to run at population scale.")
a = ("could scale at low marginal cost, an application squarely within the scope of internet-based health services.")
assert a in s; s = s.replace(a, a + INTRO_ADD_1, 1)

INTRO_ADD_2 = (" Reviews in other clinical domains have reached the same conclusion: among hundreds of prediction models "
    "published during the COVID-19 pandemic, almost none were validated in data independent of the development source and "
    "most carried a high risk of bias [7]. A systematic review of machine-learning depression models built on electronic "
    "health records likewise found reported discrimination clustered near 0.80, with generalizability and interpretability "
    "repeatedly identified as unresolved concerns [6].")
b = ("However, most published models are assessed only through internal validation [6,7].")
assert b in s; s = s.replace(b, b + INTRO_ADD_2, 1)

INTRO_NEW = ("\n\nTwo questions follow, and neither has been answered for survey-based depression screening. The first is "
    "empirical: how much discrimination is lost when a model developed in one national survey is applied, without refitting, "
    "to another country's survey, and does that loss change across a population-wide shock such as the pandemic? The second "
    "is interpretive: when external performance falls, is the shortfall a modelling failure that better algorithms or larger "
    "training sets could recover, or a property of the transportable predictor set itself, in which case additional modelling "
    "effort cannot close it? Separating the two requires more than a single external estimate. It requires an upper bound on "
    "what the same predictors can achieve inside the external population, and a systematic search over modelling choices in "
    "which no external label is ever used for selection.")
c = ("enabling temporal validation within the same external population.")
assert c in s; s = s.replace(c, c + INTRO_NEW, 1)
print("C) Introduction 확장")

# ══════════════ B. Discussion 재구성 ══════════════
PF_OLD = s[s.index("### Principal Findings"):s.index("### Comparison With Prior Work")]
PF_NEW = """### Principal Findings
A depression-screening ML model developed in Korea and externally validated in the US met its internal target (AUROC 0.837) and retained clinically useful, though lower, discrimination (~0.77) on genuinely external US data and across the COVID-19 shift, with calibration recoverable by lightweight recalibration. Further modeling did not raise external discrimination, and the plateau is best explained by a shift in how the same predictors relate to the outcome rather than by a deficiency of the algorithm.

Three findings carry the argument. First, external discrimination was 0.765 (95% CI 0.724-0.804) before the pandemic and 0.771 (0.739-0.801) after it; the two cycles are statistically indistinguishable; hence the model did not decay across the population-wide shock that motivated the temporal analysis. Second, calibration behaved exactly as transportability theory predicts and clinical deployment fears: the slope fell to 0.65 and 0.68 and calibration-in-the-large rose to +0.51 and +1.13, meaning the Korea-trained model systematically under-predicted absolute risk in a higher-prevalence population, and intercept-plus-slope updating restored both without re-estimating any predictor effect. Third, a six-item form selected entirely inside the development data lost only 0.015 internally and 0.019 and 0.012 externally, which places a low-burden self-report instrument within reach for digital triage.

Operating characteristics support that reading. With the threshold fixed once in the development data at 85% sensitivity and then applied unchanged, external sensitivity was 0.810 and 0.791 with a number needed to screen of 6.8 and 4.7, and decision-curve net benefit exceeded both screen-all and screen-none defaults in every dataset after local recalibration.

"""
s = s.replace(PF_OLD, PF_NEW)

MECH = """### Plausible Mechanisms for the External Ceiling
Three lines of evidence converge on concept shift as the dominant explanation for the ~0.77 external plateau. Leakage-controlled optimization across 2,199 catalogued runs, spanning feature engineering, hyperparameter search, heterogeneous ensembling, domain adaptation, alternative losses and deep tabular learners, never exceeded that level when model selection was restricted to development-internal criteria; apparent gains from externally tuned variants did not survive leakage control. Matching sample size did not close the gap either, and increasing the training set from 0.5 to 2 million records slightly lowered external discrimination, a pattern consistent with source-specific overfitting rather than an estimation-limited ceiling.

The decisive test was an in-domain oracle. Fitting NHANES on its own labels under cluster-separated cross-validation, a deliberate departure from the external-validation rule used for diagnosis only, reached just 0.764-0.776, whereas the same predictors reached 0.81-0.82 in equally sized Korean samples. The transported model therefore already performs at the level that the best achievable model on those predictors attains inside the external population, which locates the limit in the predictors rather than in the transfer.

Interpretability analysis points the same way. Predictor-importance rank order was highly concordant between surveys (Spearman rho 0.97 pre-pandemic and 0.99 post-pandemic), indicating that the two populations use the predictors in a similar order; what differs is the strength of the predictor-outcome associations, not the structure of those associations. Two caveats bound this conclusion. Several predictors are measured differently in the two surveys, and the external cohorts are complete-case samples with differential depression-module non-response; hence measurement and selection effects cannot be separated from concept shift with these data.

### Clinical and Policy Implications
The intended use is triage, not diagnosis. At population prevalence the positive predictive value is low, and the appropriate reading of a positive score is a prompt for confirmatory assessment rather than an action in itself. Framed that way, the numbers are favourable: at a triage-oriented operating point, 6.8 respondents in the pre-pandemic cohort and 4.7 in the post-pandemic cohort must be screened to identify one case, and net benefit exceeded both default strategies across the clinically relevant threshold range.

Two deployment requirements follow from the calibration results. Any implementation outside the development population must apply local recalibration, since discrimination transported but absolute risk did not; intercept-plus-slope updating on modest local data was sufficient and is the recommended update. Item reduction should also be re-evaluated whenever the deployment population differs materially from the development population, given that the ranking of predictors that supports a six-item form is itself population-dependent.

Equity requires explicit monitoring rather than a single fairness statement. Subgroup discrimination was lowest in Mexican American respondents in both NHANES cycles and, within the development population, in the highest income quintile, which is the one development subgroup that trips the analysis plan's 0.05 flag. The intervals are wide and mutually overlapping; hence these differences are not established, but a deployed screener should report subgroup performance continuously rather than assume the development-time estimate holds.

"""
i = s.index("### Strengths and Limitations")
s = s[:i] + MECH + s[i:]

SL_OLD = s[s.index("### Strengths and Limitations"):s.index("### Conclusions")]
SL_NEW = """### Strengths and Limitations
The principal strength is that the outcome is identical in both surveys. Depression is defined as a nine-item PHQ-9 sum of at least 10 in KCHS and in NHANES alike, which removes the outcome-harmonization error that usually confounds cross-national validation and leaves country, health system, racial and ethnic composition, and survey mode as the differences under test. Validation was also temporal as well as geographic, and recalibration was quantified rather than asserted. Evaluation was complex-survey weighted throughout, with intervals from a rescaled stratified cluster bootstrap validated against a closed-form design-based standard error. The predictor set was restricted to items both surveys measure identically, with every inclusion decision documented and dated; the three candidate algorithms received equal tuning effort; a data-integrity audit and subsequent peer review identified and corrected seven data-processing and harmonization defects before the primary analysis; and the optimization search was leakage-controlled, which rules out overfitting-driven optimism in the ceiling claim.

Several limitations bound the interpretation. The design is cross-sectional; hence the model screens current symptoms rather than predicting future onset. Alcohol, physical activity, income, body mass index and food insecurity are measured differently in the two surveys and were harmonized to conservatively comparable definitions. Depression-module non-response was 14.3% before the pandemic and 32.7% after it and was handled as complete-case; given that non-response was differential by education and by race and ethnicity, the external estimates, and particularly the post-pandemic subgroup results, may be biased in an unknown direction. Living alone is unavailable in the pre-pandemic NHANES cycle, and food insecurity is a single item in KCHS and unmeasured there in 2020 and 2021. Prior depression diagnosis is not recorded in KCHS, which precluded a diagnosis-exclusion sensitivity analysis.

Three further constraints concern the analysis itself. Development used a 500,000-record training set and a 100,001-record hold-out drawn from the 2,048,510 pooled records rather than the full pooled sample; a training-set size experiment showed this did not cost external discrimination, but internal estimates are hold-out rather than fully cross-validated. The external cycles are modest for a validation study, and the calibration slope is typically the binding precision constraint when the minimum sample size for external validation of a binary-outcome model is derived [30], which is consistent with the width of the external intervals reported here. Finally, the amendment that added three of the seventeen predictors was written while the study was under way, after an earlier external result was known but before any model using the amended set was fitted; it is dated, released unmodified (Multimedia Appendix 1), and its contribution is reported as a robustness analysis rather than a primary result. Two of the four hypotheses were stated qualitatively in the protocol, without numeric decision margins; thus judging those two 'supported' involves interpretation.

The transportable predictor set is also bounded by the development survey's content. In a coverage analysis, candidate domains available in NHANES but not harmonizable to KCHS were extracted: physician-diagnosed multimorbidity (asthma, arthritis, thyroid, chronic obstructive pulmonary disease, liver disease, and cancer, excluding circulatory conditions) and healthcare utilization; adding those domains raised the NHANES in-domain oracle from 0.765 to 0.778. KCHS, however, records only hypertension and diabetes among physician-diagnosed conditions, both already included, and does not collect a broader chronic-disease inventory, hospitalization, or visit counts; these domains therefore cannot enter a KCHS-developed transportable model. Part of the residual external gap is thus attributable to development-survey coverage rather than to concept shift alone, and a development survey that captured multimorbidity and utilization could plausibly lift the transportable ceiling by about 0.01 AUROC.

"""
s = s.replace(SL_OLD, SL_NEW)
print("B) Discussion 재구성 — 소제목 6개")

# ══════════════ D. 굵게 정리 ══════════════
KEEP = re.compile(r"^(Original Paper|Background:|Objective:|Methods:|Results:|Conclusions:|Keywords:|Table \d+\.|Figure legends\.)$")
def unbold(m):
    t = m.group(1)
    if KEEP.match(t.strip()): return m.group(0)
    if t.rstrip().endswith(".") and len(t) < 70 and t[0].isupper(): return m.group(0)   # 러닝헤드
    return t
before = len(re.findall(r"\*\*[^*]+\*\*", s))
s = re.sub(r"\*\*([^*]+)\*\*", unbold, s)
after = len(re.findall(r"\*\*[^*]+\*\*", s))
print(f"D) 굵게 {before} → {after}개 (초록 라벨·러닝헤드·표 라벨만 유지)")

io.open(P, "w", encoding="utf8").write(s)
w = lambda t: len(re.sub(r"\|[^\n]*\||[*#\[\]]", "", t).split())
print(f"\nIntroduction {w(s[s.index('## Introduction'):s.index('## Methods')])}단어 · "
      f"Discussion {w(s[s.index('## Discussion'):s.index('## Acknowledgments')])}단어")
