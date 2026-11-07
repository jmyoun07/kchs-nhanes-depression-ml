# -*- coding: utf-8 -*-
"""86_review11_fixes.py — 코덱스 11차 교차검증 지적의 교정.

핵심은 두 가지다.
 (1) 탐색 실행(Table S12·S13)의 수치는 저장 정밀도가 4자리라, 3자리로 다시 반올림하면
     0.7685 → 0.769 처럼 원값(0.768488)과 어긋난다. 저장된 4자리를 그대로 싣는다.
 (2) 절 이름·결함 개수·"시간적 검증"·"사전 선택" 같은 서술을 실제와 맞춘다.
"""
import io, os, re, sys, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__))

# ── ① 탐색 표는 저장된 4자리를 그대로 ───────────────────────────────────────
ME = pd.read_csv(f"{D}/results/methods_explore.csv")
DL = pd.read_csv(f"{D}/results/deeplearning_explore.csv")
S12_LABEL = ["Baseline (300k)", "Monotonic constraints (clinical direction)",
             "Transport-stable feature selection (PSI; internal-selected k=0)",
             "LOYO transfer-robust hyperparameters (re-selected baseline)",
             "KCHS per-year model ensemble", "Out-of-fold weight-of-evidence target encoding",
             "Missing-indicator augmentation", "Explicit imputation (no tree-native missing)",
             "Internal-CV-tuned configuration"]
S13_LABEL = ["**Gradient boosting (exploration-run reference)**", "Multilayer perceptron (internal-selected)",
             "TabNet", "Deep MLP (BatchNorm+Dropout, PyTorch)", "FT-Transformer"]
s12 = ("| Method | External pre | External post | Verdict |\n|---|---|---|---|\n" +
       "".join(f"| {lab} | {r.pre:.4f} | {r.post:.4f} | {r.verdict} |\n"
               for lab, (_, r) in zip(S12_LABEL, ME.iterrows())))
s13 = ("| Deep model | Internal | External pre | External post | Verdict |\n|---|---|---|---|---|\n" +
       "".join(f"| {lab} | {r.internal:.4f} | {r.pre:.4f} | {r.post:.4f} | {r.verdict} |\n"
               for lab, (_, r) in zip(S13_LABEL, DL.iterrows())))

def splice(path, start, stop, new):
    p = f"{D}/{path}"; s = io.open(p, encoding="utf8").read()
    i = s.index(start); j = s.index(stop, i)
    io.open(p, "w", encoding="utf8").write(s[:i] + new + s[j:])

splice("docs/appendix4_ceiling.md", "| Method | External pre | External post | Verdict |", "\n## Table S13", s12)
splice("docs/appendix4_ceiling.md", "| Deep model | Internal | External pre | External post | Verdict |", "\n## Table S14", s13)
print("  부록4: 표 S12·S13 을 저장 정밀도(4자리)로 재생성")

A4 = [
 ("An additional eight leakage-free strategies were run with an in-run same-sample baseline (pre 0.769, post 0.769).",
  "An additional eight leakage-free strategies were run with an in-run same-sample baseline (pre 0.7694, post 0.7685). Values in this table and in Table S13 are given to four decimals, the precision at which the exploration run log records them, so that they are not rounded a second time."),
 ("All fell below gradient boosting internally (0.837)", "All fell below the gradient-boosting reference internally (0.8370)"),
 ("The gradient-boosting reference row is the 300,000-record baseline that anchors Table S12, and its external values (0.769/0.769) therefore differ slightly from the primary model's 0.765/0.771;",
  "The gradient-boosting reference row is the 300,000-record baseline that anchors Table S12, and its external values (0.7694/0.7685) therefore differ from the primary model's 0.765/0.771;"),
 # ⑪ 없어진 절 이름
 ("(a deliberate External Validation Strategy violation, for diagnosis only)",
  "(a deliberate departure from the external-validation rule set out under External Validation and Sensitivity Analyses, for diagnosis only)"),
]
MS = [
 # ⑤ 2,199 은 등록부 전체 행수이고 그 안에 외부로 조정된 실행도 있다
 ("Leakage-controlled optimization (2,199 catalogued runs) did not exceed ~0.77 externally,",
  "A registry of 2,199 catalogued optimization runs contains no leakage-controlled configuration exceeding ~0.77 externally,"),
 ("Leakage-controlled optimization across 2,199 catalogued runs, spanning feature engineering, hyperparameter search, heterogeneous ensembling, domain adaptation, alternative losses and deep tabular learners, never exceeded that level when model selection was restricted to development-internal criteria;",
  "Optimization across a registry of 2,199 catalogued runs, spanning feature engineering, hyperparameter search, heterogeneous ensembling, domain adaptation, alternative losses and deep tabular learners, never exceeded that level once model selection was restricted to development-internal criteria; runs that were screened on external labels are marked as such in the registry and in Multimedia Appendix 1;"),
 # ⑥ 개발자료가 두 시기에 걸쳐 있어 엄밀한 의미의 시간적 외적검증은 아니다
 ("Temporal validation contrasted the National Center for Health Statistics (NCHS)-defined pre-pandemic (n=7,888) and post-pandemic (n=5,210) cycles.",
  "Performance was contrasted between the National Center for Health Statistics (NCHS)-defined pre-pandemic (n=7,888) and post-pandemic (n=5,210) cycles."),
 ("NHANES additionally provides NCHS-defined pre- and post-pandemic data, enabling temporal validation within the same external population.",
  "NHANES additionally provides NCHS-defined pre- and post-pandemic data, so that the same transported model can be evaluated in two periods of the same external population. Since KCHS development pools 2017-2025 and therefore spans both periods, this is a contrast of target periods rather than validation of a model frozen before the later cycle."),
 ("Validation was also temporal as well as geographic, and recalibration was quantified rather than asserted.",
  "Validation covered two target periods as well as a second country, and recalibration was quantified rather than asserted."),
 ("### Geographic and Temporal External Validation", "### External Validation Across Country and Pandemic Period"),
 # ⑦ 알고리즘 선택은 사후 확인이었다
 ("LightGBM was the primary model selected a priori by internal cross-validated AUROC.",
  "LightGBM was the primary model, chosen on internal cross-validated AUROC and calibration slope within the development data; the plan's tie band was reconstructed afterwards and reproduces the same choice."),
 # ⑧ 결함 개수를 표 S10 과 맞춘다
 ("Four harmonization defects were identified and corrected before the primary analysis rather than carried as limitations. Three follow from questionnaire changes inside KCHS:",
  "Six corrections were applied to the analytic data before the primary analysis rather than carried as limitations. Three follow from questionnaire changes inside KCHS:"),
]
A3 = [
 ("| Baseline (uncorrected data, before the seven corrections) |", "| Baseline (uncorrected data, before the six corrections below) |"),
 ("All rows use the corrected primary data and the cluster-level split (Candidate Predictors and Harmonization, Data-Integrity Audit).",
  "All rows use the corrected primary data and the cluster-level split (Predictors and Harmonization, Data-Integrity Audit)."),
 ("each cross-fitted by primary sampling unit (External Validation Strategy)", "each cross-fitted by primary sampling unit (External Validation and Sensitivity Analyses)"),
 ("are dated documents released with the code (Study Design and Reporting Standards).", "are dated documents released with the code (Study Design and Data Sources)."),
]
A2 = [
 ("Sections the Methods (outcome definition), the Methods (model development) and the Methods (external-validation strategy) of the protocol were therefore left unchanged;",
  "The protocol sections covering the outcome definition, model development and the external-validation strategy were therefore left unchanged;"),
 ("it is reported in Methods the Methods as a post-hoc diagnostic.",
  "it is reported under Predictors and Harmonization in the Methods as a post-hoc diagnostic."),
]
CK = [
 ("| 5b | Predictor definitions, timing, blinding | ✅ | Candidate Predictors and Harmonization, Table 1 |",
  "| 5b | Predictor definitions, timing, blinding | ✅ | Predictors and Harmonization, Table 1 |"),
 ("| 5c | Data preparation, harmonization, feature engineering | ✅ | Candidate Predictors and Harmonization,",
  "| 5c | Data preparation, harmonization, feature engineering | ✅ | Predictors and Harmonization,"),
 ("| 8 | Predictors handling in analysis | ✅ | Candidate Predictors and Harmonization, Model Development |",
  "| 8 | Predictors handling in analysis | ✅ | Predictors and Harmonization, Model Development |"),
 ("| 28 | Strengths and limitations | ✅ | Strengths and Limitations, Strengths and Limitations |",
  "| 28 | Strengths and limitations | ✅ | Strengths and Limitations |"),
]
MC = [
 # ⑨ 개정은 외부 결과를 안 뒤에 시작됐다 / 내부 지표는 hold-out 에서 나온다
 ("no external data used in any selection step",
  "no external data used in fitting, tuning or algorithm selection; the three-predictor amendment is the documented exception and was written after the earlier external result was known"),
 ("remaining clusters used for internal metrics", "internal metrics computed on that hold-out"),
]
CL = [
 ("an adversarial data-integrity audit and subsequent peer review identified seven data-processing and harmonization defects, all of which are corrected in the primary analysis and documented individually (Data-Integrity Audit, Multimedia Appendix 1 Table S10)",
  "an adversarial data-integrity audit and subsequent peer review identified seven data-processing and harmonization defects, all corrected in the primary analysis, each described in the Data-Integrity Audit paragraph, and the six that can be applied singly to the analytic data quantified one by one in Multimedia Appendix 1 Table S10"),
 ("this ~0.77 external ceiling withstood every leakage-controlled attempt to raise it",
  "this ~0.77 external ceiling withstood every leakage-controlled attempt made to raise it"),
]

def edit(path, pairs):
    p = f"{D}/{path}"; s = io.open(p, encoding="utf8").read(); n = 0
    for a, b in pairs:
        if a not in s: print(f"  [미일치] {path}: {a[:76]}"); continue
        s = s.replace(a, b, 1); n += 1
    io.open(p, "w", encoding="utf8").write(s); print(f"  {path}: {n}/{len(pairs)}건")

print("코덱스 11차 지적 교정")
for f, pairs in [("docs/manuscript_17var.md", MS), ("docs/appendix2_crosswalk.md", A2),
                 ("docs/appendix3_utility.md", A3), ("docs/appendix4_ceiling.md", A4),
                 ("docs/TRIPOD_AI_checklist.md", CK), ("docs/model_card_17var.md", MC),
                 ("docs/cover_letter.md", CL)]:
    edit(f, pairs)

# 본문의 §참조가 바뀌었으므로 표 S10 설명도 여섯 건 기준으로
p = f"{D}/docs/appendix3_utility.md"; s = io.open(p, encoding="utf8").read()
a = "Each row applies **one** correction to the analytic data and refits the primary model; all other settings are unchanged."
b = ("Each row applies **one** correction to the analytic data and refits the primary model; all other settings are unchanged. "
     "The two NHANES-side audit corrections (living alone fabricated to a constant, post-pandemic physical activity parsed to zero) "
     "are not shown as single-correction rows, since without them the affected variable is constant rather than merely mismeasured.")
if a in s: io.open(p, "w", encoding="utf8").write(s.replace(a, b, 1)); print("  부록3: 표 S10 주석 보강")
