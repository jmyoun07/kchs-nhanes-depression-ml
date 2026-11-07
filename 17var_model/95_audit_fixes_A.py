# -*- coding: utf-8 -*-
"""95_audit_fixes_A.py — 독립감사 지적 가운데 주장에 영향을 주는 것들.

 ① living_alone: 파이프라인에 대한 거짓 문장 삭제 + 실제 동작 기술
 ② DPQ 미응답: 면접 프레임 → MEC 추정 프레임으로 정정 (수치는 정본 CSV 에서)
 ③ 응답성향 IPW 결과를 실제로 보고 (이미 계산돼 있으나 문서에 없었다)
 ④ 105행의 낡은 '보정은 점추정' 문장 삭제 (초록·표 2·고찰과 정면 모순)
"""
import io, os, sys, pandas as pd
from decimal import Decimal, ROUND_HALF_UP
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__)); RES = f"{D}/results"
def R(v, nd=1): return f"{Decimal(str(v)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP):.{nd}f}"

NR = pd.read_csv(f"{RES}/primary2_dpq_nonresponse.csv")
def ov(period, col): return NR[(NR.period == period) & (NR.factor == "__overall__")][col].iloc[0]
def gv(period, grp, col): return NR[(NR.period == period) & (NR.group == grp)][col].iloc[0]
IPW = pd.read_csv(f"{RES}/primary2_nonresponse_ipw.csv").set_index("set")

pre_mec, post_mec = int(ov("pre", "n_mec")), int(ov("post", "n_mec"))
pre_lost = R((pre_mec - 7888) / pre_mec * 100, 1)
post_lost = R((post_mec - 5210) / post_mec * 100, 1)
print(f"MEC 프레임 손실률: pre {pre_lost}% (n={pre_mec:,}) · post {post_lost}% (n={post_mec:,})")

MS = [
 # ① 거짓 문장을 실제 동작으로 교체
 ("binary predictors were passed through with missingness preserved and handled natively by the tree learner (structurally missing binary items such as living-alone in NHANES pre are therefore never imputed to a constant), while",
  "binary predictors were passed through with missingness preserved and handled natively by the tree learner, while"),
 # ① 실제 동작을 명시 (Harmonization cautions 뒤에 붙인다)
 ("Living alone is the single retained item not present in both cycles, being absent from the pre-pandemic file.",
  "Living alone is the single retained item not present in both cycles, being absent from the pre-pandemic file. Its handling requires a caveat that the tree learner's native missing support does not remove. Living alone has no missing values anywhere in the KCHS training data, so the learner never observes a missing value for it and never learns a missing branch; at prediction time it therefore routes the entirely missing pre-pandemic column to the default branch, which is the same branch as \"not living alone\". The pre-pandemic cohort is thus scored as if no respondent lived alone. Because this differs between the two NHANES cycles, the pandemic-period contrast is repeated with living alone removed from the model, and that control is reported under Sensitivity Analyses."),
 # ② 미응답을 MEC 추정 프레임에서
 (f"DPQ item non-response was substantial in both NHANES cycles and much larger after the pandemic: among adults ≥20 years, {R(ov('pre','nr3_interview_pct'))}% of the pre-pandemic and {R(ov('post','nr3_interview_pct'))}% of the post-pandemic respondents left three or more PHQ-9 items unanswered, yielding the complete-case analytic samples of 7,888 and 5,210 (item-level detail in Multimedia Appendix 1, Table S8).",
  f"Depression-module non-response was appreciable in both NHANES cycles and larger after the pandemic. The relevant denominator is the mobile examination centre (MEC) frame, since the depression module is administered there and every weighted estimate in this study uses MEC weights: of {pre_mec:,} pre-pandemic and {post_mec:,} post-pandemic MEC-examined adults ≥20 years, {pre_lost}% and {post_lost}% respectively were lost to PHQ-9 item non-response, yielding the complete-case analytic samples of 7,888 and 5,210 (item-level detail in Multimedia Appendix 1, Table S8). Counted instead over all interviewed adults, the figures are {R(ov('pre','nr3_interview_pct'))}% and {R(ov('post','nr3_interview_pct'))}%, but that denominator conflates module non-response with never having been examined, which the MEC weights already account for."),
 # ② 하위군 수치도 같은 프레임으로
 (f"higher among respondents with less than high-school education ({R(gv('pre','<HS','nr3_interview_pct'))}% pre, {R(gv('post','<HS','nr3_interview_pct'))}% post)",
  f"higher among respondents with less than high-school education ({R(gv('pre','<HS','nr3_mec_pct'))}% pre, {R(gv('post','<HS','nr3_mec_pct'))}% post, MEC frame)"),
 # ④ 낡은 문장 삭제
 ("Intervals are computed for AUROC and AUPRC; calibration, operating characteristics and net benefit are reported as point estimates, as stated under Performance Evaluation.",
  "Interval coverage is set out under Performance Evaluation."),
]

# ③ IPW 결과를 민감도에 보고
IPW_TXT = (f"Response-propensity weighting gave the same answer: reweighting each external cohort by the inverse of "
           f"its education-specific depression-module response rate moved external AUROC by at most 0.002 "
           f"({IPW.loc['NHANES pre','AUROC_ipw']:.3f} vs {IPW.loc['NHANES pre','AUROC_base']:.3f} pre-pandemic and "
           f"{IPW.loc['NHANES post','AUROC_ipw']:.3f} vs {IPW.loc['NHANES post','AUROC_base']:.3f} post-pandemic), "
           f"so differential non-response by education does not by itself explain the external level. ")

def edit(path, pairs):
    p = f"{D}/{path}"; s = io.open(p, encoding="utf8").read(); n = 0
    for a, b in pairs:
        if a not in s: print(f"  [미일치] {path}: {a[:90]}"); continue
        s = s.replace(a, b, 1); n += 1
    io.open(p, "w", encoding="utf8").write(s); print(f"  {path}: {n}/{len(pairs)}건")

print("독립감사 교정 A (주장 관련)")
edit("docs/manuscript_17var.md", MS)

# IPW 문장은 민감도 절 끝에 붙인다
p = f"{D}/docs/manuscript_17var.md"; s = io.open(p, encoding="utf8").read()
anchor = "### Interpretability, Fairness, and Clinical Utility"
if "Response-propensity weighting gave the same answer" not in s and anchor in s:
    i = s.index(anchor)
    j = s.rfind("\n\n", 0, i)
    s = s[:j] + "\n\n" + IPW_TXT.strip() + s[j:]
    io.open(p, "w", encoding="utf8").write(s); print("  IPW 결과 문장 추가")
