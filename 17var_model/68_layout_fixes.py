# -*- coding: utf-8 -*-
"""68_layout_fixes.py — 게재본(J Med Internet Res 2024;26:e55913)에 맞춘 배치·표기 수정.

  1) Table 1 을 Methods(Candidate Predictors and Harmonization)로 옮긴다 — 지금은 Discussion 뒤에 있어 번호 순서가 뒤집혀 있다
  2) 표 캡션은 'Table N.' 라벨만 굵게, 설명은 보통 두께 (게재본과 동일)
  3) 표·그림 캡션 설명을 게재본 수준(12~70단어)으로 보강
  4) Discussion 에 Conclusions 소제목 복원 (게재본에 있다)
  5) 키워드 12 → 8개
  6) 사전지정 가설 이름표(H1~H4)의 굵게 해제 — 게재본에는 이런 강조가 없다
"""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__))
P = f"{D}/docs/manuscript_17var.md"
s = io.open(P, encoding="utf8").read()

# ── 1. Table 1 블록을 Methods 로 이동 ──
m = re.search(r"\*\*Table 1\.[\s\S]*?\n\n(?=\*\*Figure legends)", s)
assert m, "Table 1 블록을 찾지 못함"
block = m.group(0)
s = s.replace(block, "")
anchor = ("Every harmonized predictor has an established association with depression in the epidemiological literature:")
i = s.index(anchor)
j = s.index("\n\n", i) + 2
s = s[:j] + block + s[j:]
print("1) Table 1 → Methods(Candidate Predictors and Harmonization) 로 이동")

# ── 2·3. 캡션: 라벨만 굵게 + 설명 보강 ──
CAP = {
 "Table 1": ("**Table 1.** Harmonized 17-predictor set shared by the Korea Community Health Survey (KCHS) and the US "
             "National Health and Nutrition Examination Survey (NHANES). All 17 predictors are defined identically in both "
             "surveys; the KCHS column gives one representative variable name per concept, and the full year-specific "
             "crosswalk is provided in Multimedia Appendix 1. Household income is entered as a within-survey quintile, and "
             "self-rated health and self-rated oral health are 5-level ordinal scores (1 best to 5 worst). Predictors 15 to 17 "
             "(employment, self-rated oral health, and household food insecurity) entered the set through the documented "
             "protocol amendment. Race and ethnicity is not a predictor and is used only for the NHANES subgroup analysis."),
 "Table 2": ("**Table 2.** Discrimination and calibration of the primary model (light gradient-boosting machine, 17 predictors) "
             "in the internal KCHS hold-out and in the two NHANES cycles, before and after recalibration. All estimates are "
             "complex-survey weighted. The internal hold-out shares no primary sampling unit with the training data, so the "
             "internal estimate refers to clusters the model never saw. The recalibrated columns apply intercept-and-slope "
             "updating out-of-sample by 5-fold cross-fitting; predictor effects are not re-estimated."),
}
for k, v in CAP.items():
    old = re.search(rf"\*\*{k}\.[^*]*\*\*[^\n]*", s)
    assert old, f"{k} 캡션 미일치"
    s = s.replace(old.group(0), v)
print("2·3) 표 캡션: 라벨만 굵게 + 설명 보강")

FIGCAP = {
 1: ("Weighted sample characteristics across the three cohorts. Panel A shows prevalences and proportions, panel B shows means. "
     "All estimates are complex-survey weighted and computed on available cases, that is, respondents with the item recorded; "
     "the share missing in each cohort and the exact values are given in Multimedia Appendix 1, Table S4. Self-rated health and "
     "self-rated oral health are scored 1 (best) to 5 (worst)."),
 2: ("Calibration of the primary model in the internal KCHS hold-out and in the two NHANES cycles. Panel A shows the model as "
     "transported, without any updating; panel B shows the same predictions after intercept-and-slope recalibration applied "
     "out-of-sample by 5-fold cross-fitting. Points are deciles of predicted risk, plotted as weighted mean predicted probability "
     "against weighted observed proportion; the dashed line is perfect calibration. Points below the diagonal indicate "
     "under-prediction of absolute risk."),
 3: ("Weighted receiver operating characteristic curves for the internal KCHS hold-out and the two NHANES cycles, using the "
     "transported model without refitting or recalibration. The area under each curve is given in the legend; the diagonal is "
     "chance performance."),
 4: ("Sensitivity analyses. Weighted area under the receiver operating characteristic curve (AUROC) for the internal hold-out and "
     "both NHANES cycles across specifications that alter modeling choices, the predictor set, the analytic population, or the "
     "outcome definition. The dotted vertical line marks the 0.80 internal target set in the protocol. Exact values are given in "
     "Multimedia Appendix 1, Table S5."),
 5: ("Global predictor importance of the 17-predictor model, measured as mean absolute Shapley additive explanations (SHAP) value "
     "and computed on a random subsample of the development fitting partition. Larger values indicate a greater average "
     "contribution to the predicted risk; the measure is unsigned and does not indicate direction of effect."),
 6: ("Subgroup discrimination with cluster-bootstrap intervals. Weighted AUROC and 95% CI by sex, age band, education and household "
     "income quintile in the KCHS hold-out, and by race and ethnicity in the two NHANES cycles. Intervals are domain estimates "
     "computed from replicate weights generated on the full cohort design. Dotted vertical lines mark the corresponding "
     "cohort-level AUROC."),
 7: ("Decision-curve analysis for the three datasets. Weighted net benefit of the model is plotted against threshold probability, "
     "with each dataset carrying its own screen-all and screen-none default strategies. Probabilities are recalibrated "
     "out-of-sample by 5-fold cross-fitting. The model is clinically useful over the range in which its curve lies above both defaults."),
}
leg = s.split("**Figure legends.**")[1].split("\n\n")[0]
new_leg = " ".join(f"Figure {n}: {t}" for n, t in FIGCAP.items())
new_leg += " Tables: Table 1 (harmonized predictor set) and Table 2 (discrimination and calibration)."
s = s.replace(leg, " " + new_leg)
print("3) 그림 캡션 7개 보강")

# ── 4. Conclusions 소제목 복원 ──
concl = "A survey-derived depression screener transports across country and pandemic with clinically useful discrimination"
assert concl in s, "결론 문단을 찾지 못함"
s = s.replace(concl, "### Conclusions\n" + concl, 1)
print("4) Discussion 에 Conclusions 소제목 복원")

# ── 5. 키워드 8개 ──
old_kw = re.search(r"\*\*Keywords:\*\*[^\n]*", s).group(0)
new_kw = ("**Keywords:** depression; mass screening; machine learning; external validation; transportability; "
          "COVID-19; mobile health; Republic of Korea.")
s = s.replace(old_kw, new_kw)
print("5) 키워드 12 → 8개")

# ── 6. H1~H4 굵게 해제 ──
s = re.sub(r"\*\*(H[1-4](?: met| and H3 supported| supported)?)\*\*", r"\1", s)
print("6) 가설 이름표 굵게 해제")

io.open(P, "w", encoding="utf8").write(s)
print("\n완료 — 이어서 61 → 44/45 → 46 → 60 실행")
