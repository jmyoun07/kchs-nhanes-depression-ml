# -*- coding: utf-8 -*-
"""그림 캡션을 예시파일 형식으로 다시 쓴다.

이미지에서 제목과 패널 부제목을 걷어냈으므로, 패널이 무엇인지는 캡션이 말한다.
예시파일과 같은 'A: ...; B: ...' 형식. 전체 분량도 줄인다.
"""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")
p = "docs/manuscript_17var.md"
s = io.open(p, encoding="utf8").read()

m = re.search(r"(^##\s+Figure Legends\s*\n+)(.*?)(?=\n##\s)", s, re.S | re.M)
assert m, "Figure Legends 절을 못 찾음"
old = m.group(2)

LEG = [
    "Figure 1: Weighted sample characteristics across the three cohorts. A: prevalences and "
    "proportions; B: means. Estimates are complex-survey weighted and computed on available cases, "
    "that is, respondents with the item recorded; the accompanying missing shares are unweighted "
    "record proportions, and both are tabulated in Multimedia Appendix 1, Table S4. Self-rated health "
    "and self-rated oral health are scored 1 (best) to 5 (worst).",

    "Figure 2: Calibration of the primary model in the internal KCHS hold-out and the two NHANES "
    "cycles. A: the model as transported, without any updating; B: the same predictions after "
    "intercept-and-slope recalibration applied out-of-sample by 5-fold cross-fitting, applied to all "
    "three cohorts so that the panels are comparable. Points are deciles of predicted risk, plotted as "
    "weighted mean predicted probability against weighted observed proportion; the dashed line is "
    "perfect calibration, and points above it indicate under-prediction of absolute risk. The deployed "
    "internal model is the one shown in A.",

    "Figure 3: Weighted receiver operating characteristic curves for the internal KCHS hold-out and the "
    "two NHANES cycles, using the transported model without refitting or recalibration. The area under "
    "each curve is given in the legend; the diagonal is chance performance.",

    "Figure 4: Weighted area under the receiver operating characteristic curve (AUROC) for the internal "
    "hold-out and both NHANES cycles across specifications that alter modeling choices, the predictor "
    "set, the analytic population, or the outcome definition. The dotted vertical line marks the 0.80 "
    "internal target set in the protocol; exact values are given in Multimedia Appendix 1, Table S5.",

    "Figure 5: Global predictor importance of the 17-predictor model, measured as mean absolute Shapley "
    "additive explanations (SHAP) value and computed on a random subsample of the development fitting "
    "partition. Larger values indicate a greater average contribution to the predicted risk; the "
    "measure is unsigned and does not indicate direction of effect.",

    "Figure 6: Weighted AUROC and 95% CI by sex, age band, education and household income quintile in "
    "the KCHS hold-out, and by race and ethnicity in the two NHANES cycles. Intervals are domain "
    "estimates computed from replicate weights generated on the full cohort design; dotted vertical "
    "lines mark the corresponding cohort-level AUROC.",

    "Figure 7: Weighted net benefit plotted against threshold probability over the range 0.02 to 0.30, "
    "with each dataset carrying its own screen-all and screen-none default strategies. A: internal KCHS "
    "hold-out, scored with the original probabilities; B: NHANES pre-pandemic; C: NHANES post-pandemic, "
    "the two external cohorts being scored with probabilities recalibrated out-of-sample by 5-fold "
    "cross-fitting grouped by primary sampling unit. The model is clinically useful over the range in "
    "which its curve lies above both defaults.",
]
new = " ".join(LEG) + " Tables: Table 1 (harmonized predictor set) and Table 2 (discrimination and calibration).\n"
s = s[: m.start(2)] + new + s[m.end(2):]
io.open(p, "w", encoding="utf8").write(s)

print(f"그림 캡션 {len(old.split())} -> {len(new.split())} 단어")
for l in LEG:
    print(f"  [{len(l.split()):>3}단어] {l[:58]}...")
