# -*- coding: utf-8 -*-
"""61_sync_numbers.py - 정본 결과 파일 → 문서로 수치를 '주입'한다(손으로 옮기지 않는다).

재검증마다 수정이 반복된 근본 원인은 두 가지였다.
  (1) 재계산된 숫자를 여러 문서에 수작업으로 옮기면서 세대가 섞였다.
  (2) 이 스크립트가 패턴 미일치를 print 로만 알리고 넘어가, 점추정은 옛 값 그대로 두고
      신뢰구간만 새 값으로 바꾸는 '혼재'를 만들어냈다.

그래서 이제 (a) 자주 바뀌는 구간은 통째로 재생성하고, (b) 패턴이 맞지 않으면 즉시 예외를 던진다.

사용: python 61_sync_numbers.py && python 60_verify_manuscript.py
"""
import io, os, sys, re
import numpy as np, pandas as pd
from decimal import Decimal, ROUND_HALF_UP
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__)); RES = f"{D}/results"; DOCS = f"{D}/docs"
R = lambda v, nd=3: f"{Decimal(str(v)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP):.{nd}f}"
S2 = lambda v: R(v, 2)
def sgn(v, nd=2):
    """0 으로 반올림되는 값은 부호 없이 0.00 으로 - '-0.00' 같은 표기를 막는다."""
    t = R(v, nd)
    if float(t) == 0: return R(0, nd)
    return ("+" + t) if float(v) > 0 else t

PM   = pd.read_csv(f"{RES}/primary2_metrics.csv").set_index("set")
m2   = PM
gain = pd.read_csv(f"{RES}/primary2_gain.csv").set_index("set")
fair = pd.read_csv(f"{RES}/primary2_fairness.csv")
par  = pd.read_csv(f"{RES}/primary2_parsimonious.csv")
sens = pd.read_csv(f"{RES}/primary2_sensitivity.csv").set_index("cond")
chr_ = pd.read_csv(f"{RES}/primary2_characteristics.csv").set_index("characteristic")
calci = pd.read_csv(f"{RES}/primary2_calibration_ci.csv")
dauc  = pd.read_csv(f"{RES}/primary2_delta_auroc.csv").iloc[0]
NEST = "after intercept/slope recalibration (nested)"
def CV(setname, col):
    r = calci[(calci["set"] == setname) & (calci["probabilities"] == NEST)].iloc[0]
    return R(r[col], 2)
def CCI(setname, kind, what):
    if kind == "after intercept/slope recalibration": kind = NEST
    elif kind == "after intercept/slope recalibration (fixed)": kind = "after intercept/slope recalibration"
    r = calci[(calci["set"] == setname) & (calci["probabilities"] == kind)].iloc[0]
    if what == "slope": return f"{R(r.slope_lo,2)}-{R(r.slope_hi,2)}"
    return f"{sgn(r.CITL_lo)} to {sgn(r.CITL_hi)}"

cut5 = pd.read_csv(f"{RES}/primary2_cut5_prevalence.csv").set_index("set")
opd  = pd.read_csv(f"{RES}/primary2_operating.csv").set_index("set")
opr  = pd.read_csv(f"{RES}/primary2_operating_ref.csv")
tri  = pd.read_csv(f"{RES}/primary2_triage.csv")
nbm  = pd.read_csv(f"{RES}/primary2_netbenefit_models.csv")
shap = pd.read_csv(f"{RES}/primary2_shap.csv"); shap.columns = ["var", "val"]
ov   = pd.read_csv(f"{RES}/primary2_design.csv").iloc[0]
orc_all = pd.read_csv(f"{RES}/review4_oracle_grouped.csv")
orc  = orc_all[~orc_all.learner.str.contains("refit")]
orc_nest = orc_all[orc_all.learner.str.contains("refit")].iloc[0]
INT, PRE, POST = "Internal(hold-out)", "NHANES pre", "NHANES post"
KSTAR = int(ov.kstar)

def rd(path):  return io.open(f"{DOCS}/{path}", encoding="utf8").read()
def wr(path, s): io.open(f"{DOCS}/{path}", "w", encoding="utf8").write(s)

def patch(path, pairs):
    """정규식 치환. 미일치는 예외 - 조용히 넘어가면 세대 혼재가 다시 생긴다."""
    s = rd(path); n = 0
    for pat, new in pairs:
        s2, k = re.subn(pat, new.replace("\\", "\\\\"), s)
        if k == 0: raise SystemExit(f"[동기화 실패] 패턴 미일치 in {path}:\n  {pat}")
        n += k; s = s2
    wr(path, s); print(f"  {path}: {n}건 주입")

def splice(path, start, end, new):
    """start(포함)부터 end(제외)까지를 new 로 교체. 표 뒤 각주는 지키고 표만 갈아끼운다.

    예전에는 구간을 통째로 갈아엎어서, 표 아래 손으로 쓴 약어 각주가 동기화할 때마다
    사라졌다. 표(파이프로 시작하는 줄) 이후에 남아 있던 줄은 그대로 되돌려 붙인다.
    """
    s = rd(path)
    if start not in s or end not in s: raise SystemExit(f"[동기화 실패] 구간 앵커 없음 in {path}: {start[:50]!r}")
    i = s.index(start); j = s.index(end, i)
    NL = chr(10)
    seg = s[i:j].split(NL)
    k = len(seg)
    while k > 0 and not seg[k - 1].lstrip().startswith("|"): k -= 1   # 마지막 표 줄까지 되감는다
    # 표가 없는 산문 구간이면 지킬 각주도 없다. k==0 일 때 전부 지키면 옛 문단이
    # 그대로 남아 새 문단과 중복된다(실제로 한 번 그렇게 깨졌다).
    keep = [x for x in seg[k:] if x.strip()] if k > 0 else []
    tail = (NL + NL.join(keep) + NL) if keep else ""
    wr(path, s[:i] + new.rstrip(NL) + NL + tail + s[j:])
    print(f"  {path}: 구간 재생성 ({start[:38]}…{', 각주 ' + str(len(keep)) + '줄 유지' if keep else ''})")

def splice_table(path, header, new):
    """표 머리글 행부터 표가 끝나는 데까지만 바꾼다.

    '다음 표 이름' 같은 바깥 앵커를 쓰면 표 순서를 바꿀 때마다 전부 깨진다.
    표의 경계는 표 안에서 정해진다: 파이프로 시작하는 줄이 이어지는 동안이 표다.
    표 아래 각주는 이 구간 밖이므로 자동으로 보존된다.
    """
    s = rd(path)
    if header not in s:
        raise SystemExit(f"[동기화 실패] 표 머리글 없음 in {path}: {header[:50]!r}")
    i = s.index(header)
    rest = s[i:].split(chr(10))
    k = 0
    while k < len(rest) and rest[k].lstrip().startswith("|"): k += 1
    j = i + len(chr(10).join(rest[:k]))
    wr(path, s[:i] + new.rstrip(chr(10)) + s[j:])
    print(f"  {path}: 표 재생성 ({header[:38]}… {k}행)")

# ══════════════ [1] 원고 - 표 2 · §3.2 · §4.1 · 오라클 ══════════════
print("[1] 원고 - 표 2 · 내부성능 · 오라클")
def col(k, f, nd=3): return R(PM.loc[k, f], nd)
tab2 = (
 "| Metric | Internal hold-out | NHANES pre-pandemic | pre-pandemic, recalibrated | NHANES post-pandemic | post-pandemic, recalibrated |\n"
 "|---|---|---|---|---|---|\n"
 f"| AUROC (95% CI) | {col(INT,'AUROC')} ({col(INT,'CI_lo')}-{col(INT,'CI_hi')}) | {col(PRE,'AUROC')} ({col(PRE,'CI_lo')}-{col(PRE,'CI_hi')}) | {col(PRE,'recal_is_AUROC')} | {col(POST,'AUROC')} ({col(POST,'CI_lo')}-{col(POST,'CI_hi')}) | {col(POST,'recal_is_AUROC')} |\n"
 f"| AUPRC (95% CI) | {col(INT,'AUPRC')} ({col(INT,'AUPRC_lo')}-{col(INT,'AUPRC_hi')}) | {col(PRE,'AUPRC')} ({col(PRE,'AUPRC_lo')}-{col(PRE,'AUPRC_hi')}) | {col(PRE,'recal_AUPRC')} | {col(POST,'AUPRC')} ({col(POST,'AUPRC_lo')}-{col(POST,'AUPRC_hi')}) | {col(POST,'recal_AUPRC')} |\n"
 f"| Brier score | {col(INT,'Brier')} | {col(PRE,'Brier')} | {col(PRE,'recal_is_Brier')} | {col(POST,'Brier')} | {col(POST,'recal_is_Brier')} |\n"
 f"| ECE | {col(INT,'ECE')} | {col(PRE,'ECE')} | {col(PRE,'recal_ECE')} | {col(POST,'ECE')} | {col(POST,'recal_ECE')} |\n"
 f"| Calibration slope (95% CI) | {col(INT,'cal_slope',2)} | {col(PRE,'cal_slope',2)} ({CCI('NHANES pre','as transported','slope')}) | {col(PRE,'recal_is_slope',2)} ({CCI('NHANES pre','after intercept/slope recalibration','slope')}) | {col(POST,'cal_slope',2)} ({CCI('NHANES post','as transported','slope')}) | {col(POST,'recal_is_slope',2)} ({CCI('NHANES post','after intercept/slope recalibration','slope')}) |\n"
 f"| CITL (95% CI) | {sgn(PM.loc[INT,'CITL'])} | {sgn(PM.loc[PRE,'CITL'])} ({CCI('NHANES pre','as transported','CITL')}) | {sgn(PM.loc[PRE,'recal_is_CITL'])} ({CCI('NHANES pre','after intercept/slope recalibration','CITL')}) | {sgn(PM.loc[POST,'CITL'])} ({CCI('NHANES post','as transported','CITL')}) | {sgn(PM.loc[POST,'recal_is_CITL'])} ({CCI('NHANES post','after intercept/slope recalibration','CITL')}) |\n\n")
splice("manuscript_17var.md", "| Metric | Internal hold-out |", "CIs are from a rescaled stratified cluster bootstrap", tab2)

_dep = lambda c: R(chr_.loc["Depression PHQ-9 ≥10", c], 1)
patch("manuscript_17var.md", [  # 유병률 문장
 (r"was \d\.\d% in KCHS against \d\.\d% and \d{1,2}\.\d% in the two NHANES cycles",
  f'was {_dep("KCHS")}% in KCHS against {_dep("NHANES pre")}% and {_dep("NHANES post")}% in the two NHANES cycles'),
 (r"\(KCHS \d\.\d% vs \d\.\d% and \d{1,2}\.\d%\)",
  f'(KCHS {_dep("KCHS")}% vs {_dep("NHANES pre")}% and {_dep("NHANES post")}%)'),
 (r"weighted prevalence \d\.\d% and \d{1,2}\.\d%\) are the binding constraint",
  f'weighted prevalence {_dep("NHANES pre")}% and {_dep("NHANES post")}%) are the binding constraint'),
])

patch("manuscript_17var.md", [
 (r"Internal hold-out AUROC was 0\.\d{3} \(95% CI 0\.\d{3}-0\.\d{3}\), AUPRC 0\.\d{3}, Brier 0\.\d{3}, calibration slope \d\.\d{2}, CITL [+-]?\d\.\d{2}",
  f"Internal hold-out AUROC was {col(INT,'AUROC')} (95% CI {col(INT,'CI_lo')}-{col(INT,'CI_hi')}), AUPRC {col(INT,'AUPRC')}, "
  f"Brier {col(INT,'Brier')}, calibration slope {col(INT,'cal_slope',2)}, CITL {sgn(PM.loc[INT,'CITL'])}"),
 (r"met its internal target \(AUROC 0\.\d{3}\)", f"met its internal target (AUROC {col(INT,'AUROC')})"),
 # 초록도 정본에서 생성한다 (손으로 옮겨 적어 세대가 갈리던 자리)
 (r"was 0\.\d{3} \(95% confidence interval \[CI\] 0\.\d{3}-0\.\d{3}\), meeting the ≥0\.80 target",
  f"was {col(INT,'AUROC')} (95% confidence interval [CI] {col(INT,'CI_lo')}-{col(INT,'CI_hi')}), meeting the ≥0.80 target"),
 (r"indistinguishable \(pre 0\.\d{3}, 95% CI 0\.\d{3}-0\.\d{3}; post 0\.\d{3}, 0\.\d{3}-0\.\d{3}; difference [+-]0\.\d{3}, 95% CI [+-]0\.\d{3} to [+-]0\.\d{3}, computed post hoc\)",
  f"indistinguishable (pre {col(PRE,'AUROC')}, 95% CI {col(PRE,'CI_lo')}-{col(PRE,'CI_hi')}; post {col(POST,'AUROC')}, {col(POST,'CI_lo')}-{col(POST,'CI_hi')}; "
  f"difference {sgn(dauc.delta,3)}, 95% CI {sgn(dauc.CI_lo,3)} to {sgn(dauc.CI_hi,3)}, computed post hoc)"),
 # 결과 - 외부 판별력 헤드라인 (손글씨였던 자리)
 (r"between the two cycles: pre 0\.\d{3} \(95% CI 0\.\d{3}-0\.\d{3}\), post 0\.\d{3} \(0\.\d{3}-0\.\d{3}\) \(Table 2; Figures 2 and 3\); the difference is [+-]0\.\d{3} \(95% CI [+-]0\.\d{3} to [+-]0\.\d{3}; bootstrap P=\.\d{2}\)",
  f"between the two cycles: pre {col(PRE,'AUROC')} (95% CI {col(PRE,'CI_lo')}-{col(PRE,'CI_hi')}), post {col(POST,'AUROC')} ({col(POST,'CI_lo')}-{col(POST,'CI_hi')}) (Table 2; Figures 2 and 3); "
  f"the difference is {sgn(dauc.delta,3)} (95% CI {sgn(dauc.CI_lo,3)} to {sgn(dauc.CI_hi,3)}; bootstrap P={format(dauc.P, '.2f').lstrip('0')})"),
 # 고찰 - 같은 수치
 (r"[Ee]xternal discrimination was 0\.\d{3} \(95% CI 0\.\d{3}-0\.\d{3}\) before the pandemic and 0\.\d{3} \(0\.\d{3}-0\.\d{3}\) after it; the difference is [+-]0\.\d{3} \(95% CI [+-]0\.\d{3} to [+-]0\.\d{3}; bootstrap P=\.\d{2}\)",
  f"External discrimination was {col(PRE,'AUROC')} (95% CI {col(PRE,'CI_lo')}-{col(PRE,'CI_hi')}) before the pandemic and {col(POST,'AUROC')} ({col(POST,'CI_lo')}-{col(POST,'CI_hi')}) after it; "
  f"the difference is {sgn(dauc.delta,3)} (95% CI {sgn(dauc.CI_lo,3)} to {sgn(dauc.CI_hi,3)}; bootstrap P={format(dauc.P, '.2f').lstrip('0')})"),
 (r"The KCHS-transferred model \(pre 0\.\d{3}, post 0\.\d{3}\)",
  f"The KCHS-transferred model (pre {col(PRE,'AUROC')}, post {col(POST,'AUROC')})"),
 (r"reached only 0\.\d{3}-0\.\d{3} \(best learner 0\.\d{3}, 95% CI 0\.\d{3}-0\.\d{3}; widening to 0\.\d{3}-0\.\d{3}",
  f"reached only {R(orc.oracle_AUROC.min())}-{R(orc.oracle_AUROC.max())} (best learner {R(orc.oracle_AUROC.max())}, "
  f"95% CI {R(orc[orc.oracle_AUROC==orc.oracle_AUROC.max()].iloc[0].CI_lo)}-{R(orc[orc.oracle_AUROC==orc.oracle_AUROC.max()].iloc[0].CI_hi)}; "
  f"widening to {R(orc_nest.CI_lo)}-{R(orc_nest.CI_hi)}"),
 (r"reached only 0\.\d{3}-0\.\d{3}, whereas the same predictors",
  f"reached only {R(orc.oracle_AUROC.min())}-{R(orc.oracle_AUROC.max())}, whereas the same predictors"),
])

# ══════════════ [2] 원고 §3.6 - 가중학습 민감도 ══════════════
print("[2] 원고 §3.6 - 민감도 수치")
def tri3(cond): r = sens.loc[cond]; return f"{R(r.internal)}/{R(r.ext_pre)}/{R(r.ext_post)}"
_c5 = lambda k: R(cut5.loc[k, "prevalence"]*100, 1)
_g10 = PM.loc[INT, "AUROC"] - max(PM.loc[PRE, "AUROC"], PM.loc[POST, "AUROC"])
_i5, _p5, _s5 = sens.loc["Outcome PHQ-9 ≥5", ["internal", "ext_pre", "ext_post"]]
patch("manuscript_17var.md", [
 (r"weighted training \(0\.\d{3}/0\.\d{3}/0\.\d{3} internal/pre/post\)",
  f"weighted training ({tri3('Weighted training')} internal/pre/post)"),
 (r"missing-indicator imputation \(0\.\d{3}/0\.\d{3}/0\.\d{3}\)",
  f"missing-indicator imputation ({tri3('Missing-indicator imputation')})"),
 (r"single chained-equations imputation of the continuous predictors \(0\.\d{3}/0\.\d{3}/0\.\d{3}\)",
  f"single chained-equations imputation of the continuous predictors ({tri3('Single chained-equations imputation')})"),
 (r"aged ≥30 years \*improved\* discrimination \(0\.\d{3}/0\.\d{3}/0\.\d{3}\)",
  f"aged ≥30 years *improved* discrimination ({tri3('Age ≥30 (train and evaluate)')})"),
 (r"lowered discrimination throughout \(0\.\d{3}/0\.\d{3}/0\.\d{3}\) at a much higher weighted prevalence \(\d{2}\.\d%/\d{2}\.\d%/\d{2}\.\d%\)",
  f"lowered discrimination throughout ({tri3('Outcome PHQ-9 ≥5')}) at a much higher weighted prevalence "
  f"({_c5('internal')}%/{_c5('pre')}%/{_c5('post')}%)"),
 (r"the Korea-US gap narrows from about 0\.\d{2} AUROC at the ≥10 threshold to about 0\.\d{2} at ≥5 \(0\.\d{3} vs 0\.\d{3}-0\.\d{3}\)",
  f"the Korea-US gap narrows from about {R(_g10,2)} AUROC at the ≥10 threshold to about {R(_i5-max(_p5,_s5),2)} at ≥5 "
  f"({R(_i5)} vs {R(min(_p5,_s5))}-{R(max(_p5,_s5))})"),
])

# 부록3 Table S4 - 표본 특성(가용사례)
_rows = ["| Characteristic | KCHS | NHANES pre | NHANES post | Missing (KCHS / pre / post) |", "|---|---|---|---|---|",
         "| N (unweighted) | 2,048,510 | 7,888 | 5,210 | - |"]
for _k, _r in chr_.iterrows():
    # 연속변수만 소수 2자리, 나머지는 모두 비율이므로 % 를 붙인다
    _CONT = ("Age (years)", "BMI", "Sleep hours", "Self-rated health", "Oral health", "Household income quintile")
    _lab = _k
    _f = (lambda v: "-" if pd.isna(v) else R(v, 2)) if _k in _CONT else \
         (lambda v: "not collected" if pd.isna(v) else f"{R(v,1)}%")
    _rows.append(f"| {_lab} | {_f(_r['KCHS'])} | {_f(_r['NHANES pre'])} | {_f(_r['NHANES post'])} | "
                 f"{R(_r['KCHS missing%'],1)}% / {R(_r['NHANES pre missing%'],1)}% / {R(_r['NHANES post missing%'],1)}% |")
splice_table("appendix3_utility.md", "| Characteristic | KCHS | NHANES pre |", "\n".join(_rows) + "\n")
# ══════════════ [3] 원고 §3.7 - SHAP · 하위군 · 공정성 플래그 ══════════════
print("[3] 원고 §3.7 - SHAP · 하위군 · 공정성 플래그")
NAME = {"srh": "self-rated health", "age": "age", "sex": "sex", "sleep_hrs": "sleep duration",
        "oral_health": "self-rated oral health", "partnered": "partnership", "smoking": "smoking",
        "employed": "employment", "bmi": "BMI", "food_insecure": "food insecurity", "income_q": "household income quintile",
        "active": "physical activity", "edu": "education", "living_alone": "living alone",
        "current_drinker": "current drinking", "htn": "hypertension", "dm": "diabetes"}
top = shap.head(7)
fi = shap.reset_index(drop=True); fi_rank = int(fi.index[fi["var"] == "food_insecure"][0]) + 1
shap_txt = ("Global SHAP importance, computed in the development fitting partition, ranked the predictors: "
            + "; ".join(f"{NAME[r['var']]} ({R(r.val,2)})" for _, r in top.iterrows())
            + f". Food insecurity contributed modestly ({R(fi[fi['var']=='food_insecure'].iloc[0].val,2)}, ranked {fi_rank}th). "
            "The predictors carrying most of the signal are therefore the general-health, demographic and sleep items "
            "that both surveys measure identically (Figure 5). ")
splice("manuscript_17var.md", "Global SHAP importance, computed in the development fitting partition",
       "Rank ordering was highly concordant", shap_txt)

SUBLAB = {"Sex: female": "female", "Sex: male": "male", "Age 20-39": "adults aged 20-39 years",
          "Age 40-59": "adults aged 40-59 years", "Age 60+": "adults aged ≥60 years",
          "Education: college": "college-educated respondents", "Education: high school": "high-school-educated respondents",
          "Education: middle": "middle-school-educated respondents", "Education: <=elementary": "respondents with elementary education or less",
          "Income quintile 1": "the lowest income quintile", "Income quintile 2": "income quintile 2",
          "Income quintile 3": "income quintile 3", "Income quintile 4": "income quintile 4",
          "Income quintile 5": "the highest income quintile"}
def slab(sg):
    return SUBLAB.get(sg, str(sg).split(": ")[-1].replace("Non-Hispanic", "non-Hispanic")
                      .replace("Other/multiracial", "other/multiracial").replace("Other Hispanic", "other Hispanic"))
def row(coh, sub): return fair[(fair.cohort == coh) & (fair.subgroup == sub)].iloc[0]
def cite(coh, sub, first=False):
    r = row(coh, sub)
    return f"{R(r.AUROC)}, {'95% CI ' if first else ''}{R(r.CI_lo)}-{R(r.CI_hi)}"
kc = fair[fair.cohort == "KCHS hold-out"]
edu = kc[kc.subgroup.str.startswith("Education")]
inc = kc[kc.subgroup.str.startswith("Income")]
def listing(coh):
    d = fair[fair.cohort == coh].sort_values("AUROC")
    return ", ".join(f"{slab(r.subgroup)} {R(r.AUROC)} ({R(r.CI_lo)}-{R(r.CI_hi)}; n={int(r.n):,})" for _, r in d.iterrows())
COH = {"KCHS hold-out": PM.loc[INT, "AUROC"], "NHANES pre": PM.loc[PRE, "AUROC"], "NHANES post": PM.loc[POST, "AUROC"]}
flags = [(c, s.subgroup, s.AUROC - COH[c], s) for c in COH for _, s in fair[fair.cohort == c].iterrows()
         if abs(s.AUROC - COH[c]) > 0.05]
LAB = {"KCHS hold-out": "KCHS", "NHANES pre": "NHANES pre-pandemic", "NHANES post": "NHANES post-pandemic"}
flag_txt = ("Applying the fairness flag set in the analysis plan (a subgroup AUROC differing from the cohort value by more than 0.05) "
            + ("flags " + "; ".join(f"{slab(f[1])} in {LAB[f[0]]} ({sgn(f[2],3)})" for f in flags) + ". " if flags
               else "flags no subgroup. "))
CI_COH = {"KCHS hold-out": (PM.loc[INT, "CI_lo"], PM.loc[INT, "CI_hi"]),
          "NHANES pre": (PM.loc[PRE, "CI_lo"], PM.loc[PRE, "CI_hi"]),
          "NHANES post": (PM.loc[POST, "CI_lo"], PM.loc[POST, "CI_hi"])}
def sep(coh):
    lo, hi = CI_COH[coh]
    return [r for _, r in fair[fair.cohort == coh].iterrows() if r.CI_hi < lo or r.CI_lo > hi]
sep_k = sep("KCHS hold-out"); sep_n = sep("NHANES pre") + sep("NHANES post")
def _dir(rs, above):
    lo, hi = CI_COH["KCHS hold-out"]
    sel = [r for r in rs if (r.CI_lo > hi) if above] if above else [r for r in rs if r.CI_hi < lo]
    return ", ".join(f"{slab(r.subgroup)} ({R(r.CI_lo)}-{R(r.CI_hi)})" for r in sel)
if sep_k:
    _hi, _lo = _dir(sep_k, True), _dir(sep_k, False)
    parts = ([f"higher in {_hi}"] if _hi else []) + ([f"lower in {_lo}"] if _lo else [])
    overlap_txt = ("Within KCHS a few intervals sit clear of the cohort interval: " + "; ".join(parts)
                   + ". Subgroup discrimination is therefore not uniform in the development population. ")
else:
    overlap_txt = "Every KCHS subgroup interval overlaps the cohort interval. "
overlap_txt += ("In NHANES, by contrast, the intervals are wide and mutually overlapping, and every one overlaps the cohort-level interval; "
                if not sep_n else
                "In NHANES the following intervals sit clear of the cohort interval: "
                + ", ".join(f"{r.subgroup.split(': ')[-1].lower()} ({R(r.CI_lo)}-{R(r.CI_hi)})" for r in sep_n) + ", ")
overlap_txt += ("hence the race/ethnicity ordering should not be read as established differential performance; that the lowest point estimate "
                "falls in Mexican American respondents in both cycles nonetheless warrants monitoring as an equity caveat (Figure 6). ")

kc_sl, kc_ci = kc.cal_slope, kc.CITL
nh = fair[fair.cohort != "KCHS hold-out"]
nh_lo = nh[nh.cal_slope < 1]; nh_hi = nh[nh.cal_slope >= 1]
cal_txt = (f"Internal subgroup calibration was good (slope {R(kc_sl.min(),2)}-{R(kc_sl.max(),2)} across sex, age, education and income strata; "
           f"subgroup CITL {sgn(kc_ci.min())} to {sgn(kc_ci.max())}). In NHANES the subgroup calibration slopes were below 1 in "
           + (f"every subgroup ({R(nh_lo.cal_slope.min(),2)}-{R(nh_lo.cal_slope.max(),2)})" if len(nh_hi) == 0 else
              f"all but {'one' if len(nh_hi)==1 else len(nh_hi)} subgroup{'s' if len(nh_hi) > 1 else ''} ({R(nh_lo.cal_slope.min(),2)}-{R(nh_lo.cal_slope.max(),2)}; "
              + ", ".join(f"{slab(r.subgroup)} {r.cohort.replace('NHANES ','')}-pandemic {R(r.cal_slope,2)}" for _, r in nh_hi.iterrows()) + ")")
           + f", and subgroup CITL was positive in every group except {', '.join(f"{slab(r.subgroup)} {r.cohort.replace('NHANES ','')}-pandemic" for _, r in nh[nh.CITL < 0].iterrows()) or 'none'}"
           + f" (up to {sgn(nh.CITL.max())}), confirming that the external miscalibration is under-prediction of absolute risk and not only "
           "over-extreme ranking. Exact subgroup values with CIs are tabulated in Table S7 in Multimedia Appendix 1.")

sub_txt = (
 f"Subgroup discrimination in KCHS spanned {R(kc.AUROC.min())}-{R(kc.AUROC.max())} (weighted AUROC): "
 f"sex (female {cite('KCHS hold-out','Sex: female', True)}; male {cite('KCHS hold-out','Sex: male')}), "
 f"education ({R(edu.AUROC.min())}-{R(edu.AUROC.max())}), "
 f"age (20-39 years {cite('KCHS hold-out','Age 20-39')} < 40-59 years {cite('KCHS hold-out','Age 40-59')} < ≥60 years {cite('KCHS hold-out','Age 60+')}), "
 f"and household income quintile ({R(inc.AUROC.min())}-{R(inc.AUROC.max())}, lowest in the highest-income quintile). "
 f"In NHANES, discrimination was lower and far less precisely estimated. Pre-pandemic: {listing('NHANES pre')}. "
 f"Post-pandemic: {listing('NHANES post')}. " + flag_txt + overlap_txt + cal_txt + " ")
splice("manuscript_17var.md", "Subgroup discrimination in KCHS", "\n\nOn decision-curve analysis", sub_txt)

# ══════════════ [4] 원고 §3.7 - 운영점·순편익 서술 ══════════════
print("[4] 원고 §3.7 - 운영점 서술")
def o(k, f, nd=3): return R(opd.loc[k, f], nd)
patch("manuscript_17var.md", [
 (r"sensitivity/specificity were 0\.\d{3}/0\.\d{3} internally, 0\.\d{3}/0\.\d{3} in NHANES pre and 0\.\d{3}/0\.\d{3} in NHANES post, "
  r"with positive predictive value 0\.\d{3} and 0\.\d{3} externally \(number needed to screen \d\.\d and \d\.\d\) and negative predictive value 0\.\d{3} and 0\.\d{3}; "
  r"the model flagged \d{2}% of the pre-pandemic and \d{2}% of the post-pandemic population",
  f"sensitivity/specificity were {o(INT,'sens')}/{o(INT,'spec')} internally, {o(PRE,'sens')}/{o(PRE,'spec')} in NHANES pre and "
  f"{o(POST,'sens')}/{o(POST,'spec')} in NHANES post, with positive predictive value {o(PRE,'PPV')} and {o(POST,'PPV')} externally "
  f"(number needed to screen {R(opd.loc[PRE,'NNS'],1)} and {R(opd.loc[POST,'NNS'],1)}) and negative predictive value {o(PRE,'NPV')} and {o(POST,'NPV')}; "
  f"the model flagged {R(opd.loc[PRE,'flag']*100,0)}% of the pre-pandemic and {R(opd.loc[POST,'flag']*100,0)}% of the post-pandemic population"),
])


# ══════════════ [4b] 원고 - 경량모델 문단 ══════════════
print("[4b] 원고 - 경량모델 궤적·성능")
_pk = par.set_index("k")
_traj = "; ".join(f"{k} item{'s' if k > 1 else ''} {R(_pk.loc[k, 'selection_AUROC'])}" for k in [1, 3, 5, KSTAR, 8, 12, 17])
_items = ", ".join(NAME[v] for v in _pk.loc[KSTAR, "vars"].split(";"))
_WORD = {5: "five", 6: "six", 7: "seven", 8: "eight"}
_loss = lambda c: R(_pk.loc[17, c] - _pk.loc[KSTAR, c])
patch("manuscript_17var.md", [
 (r"On the selection set, discrimination rose steeply and then flattened \([^\n]*?17 items \d\.\d{3}\)",
  f"On the selection set, discrimination rose steeply and then flattened ({_traj})"),
 (r"selected \w+ items: [^\n]*?\. Evaluated afterwards on data that played no part in selection, this six-item form achieved "
  r"hold-out AUROC 0\.\d{3}, NHANES pre 0\.\d{3} and NHANES post 0\.\d{3}, against 0\.\d{3}/0\.\d{3}/0\.\d{3} "
  r"for the full 17-predictor model fitted on the same data \(a loss of 0\.\d{3} internally, 0\.\d{3} pre-pandemic and "
  r"0\.\d{3} post-pandemic, each within the plan's 0\.02 tolerance\)",
  f"selected {_WORD.get(KSTAR, KSTAR)} items: {_items}. Evaluated afterwards on data that played no part in selection, "
  f"this six-item form achieved hold-out AUROC {R(_pk.loc[KSTAR, 'holdout_AUROC'])}, NHANES pre {R(_pk.loc[KSTAR, 'ext_pre'])} "
  f"and NHANES post {R(_pk.loc[KSTAR, 'ext_post'])}, against {R(_pk.loc[17, 'holdout_AUROC'])}/{R(_pk.loc[17, 'ext_pre'])}/"
  f"{R(_pk.loc[17, 'ext_post'])} for the full 17-predictor model fitted on the same data (a loss of {_loss('holdout_AUROC')} "
  f"internally, {_loss('ext_pre')} pre-pandemic and {_loss('ext_post')} post-pandemic, each within the plan's 0.02 tolerance)"),
 (r"lost only 0\.\d{3} internally and 0\.\d{3} and 0\.\d{3} externally",
  f"lost only {_loss('holdout_AUROC')} internally and {_loss('ext_pre')} and {_loss('ext_post')} externally"),
])

# ══════════════ [4c] 원고 - 교차조사 SHAP 순위 일치도 ══════════════
_rho = pd.read_csv(f"{RES}/primary2_shap_rho.csv").set_index("comparison")["spearman_rho"]
patch("manuscript_17var.md", [
 (r"\(Spearman ρ=0\.\d{2} pre, 0\.\d{2} post\)",
  f"(Spearman ρ={R(_rho['KCHS vs NHANES pre'],2)} pre, {R(_rho['KCHS vs NHANES post'],2)} post)"),
 (r"\(Spearman ρ=0\.\d{2} pre-pandemic and 0\.\d{2} post-pandemic\)",
  f"(Spearman ρ={R(_rho['KCHS vs NHANES pre'],2)} pre-pandemic and {R(_rho['KCHS vs NHANES post'],2)} post-pandemic)"),
])

# ══════════════ [5] 모델카드 ══════════════
print("[5] 모델카드")
mc_metrics = (
 "| Setting | AUROC (95% CI) | Calibration slope | After recalibration |\n"
 "|---|---|---|---|\n"
 f"| Internal (KCHS hold-out) | {col(INT,'AUROC')} ({col(INT,'CI_lo')}-{col(INT,'CI_hi')}) | {col(INT,'cal_slope',2)} (CITL {sgn(PM.loc[INT,'CITL'])}) | - |\n"
 f"| NHANES pre-COVID | {col(PRE,'AUROC')} ({col(PRE,'CI_lo')}-{col(PRE,'CI_hi')}) | {col(PRE,'cal_slope',2)} (CITL {sgn(PM.loc[PRE,'CITL'])}) | slope→{col(PRE,'recal_is_slope',2)}, CITL→{S2(abs(PM.loc[PRE,'recal_is_CITL']))} |\n"
 f"| NHANES post-COVID | {col(POST,'AUROC')} ({col(POST,'CI_lo')}-{col(POST,'CI_hi')}) | {col(POST,'cal_slope',2)} (CITL {sgn(PM.loc[POST,'CITL'])}) | slope→{col(POST,'recal_is_slope',2)}, CITL→{S2(abs(PM.loc[POST,'recal_is_CITL']))} |\n")
pk = par.set_index("k")
items = ", ".join(NAME[v] for v in pk.loc[KSTAR, "vars"].split(";"))
p7 = pk.loc[7]
mc_pars = (f"- Parsimonious variant selected by the analysis plan's margin, entirely inside the training data: **{KSTAR} items** ({items}): "
           f"hold-out {R(pk.loc[KSTAR,'holdout_AUROC'])}, external {R(pk.loc[KSTAR,'ext_pre'])} (pre) and {R(pk.loc[KSTAR,'ext_post'])} (post). "
           f"A 7-item variant chosen under a stricter margin helps the pre-pandemic cohort ({R(p7.ext_pre)}) but not the post-pandemic one "
           f"({R(p7.ext_post)}); that comparison inspected the external cohorts and is therefore reported as exploratory.\n")
splice("model_card_17var.md", "| Setting | AUROC (95% CI)", "- CIs from a rescaled stratified cluster bootstrap", mc_metrics + mc_pars)

yo = opr[opr.threshold == "Youden (exact)"].set_index("set")
t85 = tri[(tri.model == f"{len(NAME) and 17}-predictor") & (tri.target == "sens 0.85")].set_index("set")
mc_op = (f"- Operating point (Youden, computed over all observed scores): sensitivity/specificity "
         f"{R(yo.loc['Internal','sens'])}/{R(yo.loc['Internal','spec'])} (internal), {R(yo.loc['NHANES pre','sens'])}/{R(yo.loc['NHANES pre','spec'])} (pre), "
         f"{R(yo.loc['NHANES post','sens'])}/{R(yo.loc['NHANES post','spec'])} (post); PPV low ({R(yo.PPV.min())}-{R(yo.PPV.max())}), NPV high ({R(yo.NPV.min())}-{R(yo.NPV.max())}). "
         f"At the single development-fixed threshold (sensitivity 0.85 in development) the number needed to screen is {R(opd.loc[PRE,'NNS'],1)} (pre) and {R(opd.loc[POST,'NNS'],1)} (post).\n"
         f"- Subgroup AUROC: KCHS {R(kc.AUROC.min())}-{R(kc.AUROC.max())}; NHANES race/ethnicity "
         f"{R(fair[fair.cohort!='KCHS hold-out'].AUROC.min())}-{R(fair[fair.cohort!='KCHS hold-out'].AUROC.max())} across both cycles, "
         "with wide and mutually overlapping CIs (Mexican American lowest in both cycles).\n")
splice("model_card_17var.md", "- Operating point (Youden", "\n## Training data", mc_op)
patch("model_card_17var.md", [
 (r"an in-domain oracle fitted directly to NHANES reached only 0\.\d{3}-0\.\d{3} against 0\.\d{2}-0\.\d{2} in equally sized Korean samples",
  f"an in-domain oracle fitted directly to NHANES reached only {R(orc.oracle_AUROC.min())}-{R(orc.oracle_AUROC.max())} against 0.81-0.82 in equally sized Korean samples"),
])

# ══════════════ [6] 부록3 - S1 · S4 · S5 · S7 · S8 · S9 ══════════════
print("[6] 부록3 - S1·S4·S5·S7·S8·S9")
r1 = ["| Set | Threshold rule | Probability cutoff | Sensitivity | Specificity | PPV | NPV |", "|---|---|---|---|---|---|---|"]
for _, r in opr.iterrows():
    r1.append(f"| {r['set']} | {r.threshold} | {r.value:.4f} | {R(r.sens)} | {R(r.spec)} | {R(r.PPV)} | {R(r.NPV)} |")
# 본문이 인용하는 '개발셋 확정 임계값' 행도 같은 표에 넣는다 (코덱스 2차 지적 14)
_LB = {"Development (threshold source)": "Development set", "Internal(hold-out)": "Internal",
       "NHANES pre": "NHANES pre", "NHANES post": "NHANES post"}
for _k, _r in opd.iterrows():
    r1.append(f"| {_LB.get(_k, _k)} | development-fixed (sens 0.85) | {_r.threshold:.4f} | "
              f"{R(_r.sens)} | {R(_r.spec)} | {R(_r.PPV)} | {R(_r.NPV)} |")
splice_table("appendix3_utility.md", "| Set | Threshold rule | Probability cutoff |", "\n".join(r1) + "\n")
r4 = ["| Model | Set | Target | Specificity | PPV | NPV | Flag rate | NNS |", "|---|---|---|---|---|---|---|---|"]
for _, r in tri.iterrows():
    r4.append(f"| {r.model} | {r['set']} | {r.target} | {R(r.spec)} | {R(r.PPV)} | {R(r.NPV)} | {R(r.flag)} | {r.NNS:.1f} |")
splice_table("appendix3_utility.md", "| Model | Set | Target |", "\n".join(r4) + "\n")
r5 = ["| Model | Set | Threshold | Model net benefit | Screen-all NB | vs screen-all | vs better default | Flag rate |", "|---|---|---|---|---|---|---|---|"]
for _, r in nbm.iterrows():
    f4 = lambda v: ("+" if v >= 0 else "") + f"{v:.4f}"
    r5.append(f"| {r.model} | {r['set']} | {r.threshold:.2f} | {f4(r.net_benefit)} | {f4(r.screen_all_NB)} | {f4(r.vs_screen_all)} | {f4(r.vs_best_default)} | {r.flag_rate:.3f} |")
splice_table("appendix3_utility.md", "| Model | Set | Threshold | Model net benefit |", "\n".join(r5) + "\n")
p85 = tri[(tri.target == "sens 0.85")]
g17 = p85[p85.model == "17-predictor"].set_index("set"); g6 = p85[p85.model == f"{KSTAR}-item"].set_index("set"); g7 = p85[p85.model == "7-item"].set_index("set")
interp = ("Tuned to a triage-appropriate 85% sensitivity, the full model flags roughly half of the external population "
          f"(pre {R(g17.loc['NHANES pre','flag']*100,0)}%, post {R(g17.loc['NHANES post','flag']*100,0)}%) at a PPV of {R(g17.loc['NHANES pre','PPV'])} (pre) and {R(g17.loc['NHANES post','PPV'])} (post), "
          f"ie, a number needed to screen of {g17.loc['NHANES pre','NNS']:.1f} and {g17.loc['NHANES post','NNS']:.1f}. The {KSTAR}-item deployment model is close behind "
          f"(PPV {R(g6.loc['NHANES pre','PPV'])} and {R(g6.loc['NHANES post','PPV'])}; NNS {g6.loc['NHANES pre','NNS']:.1f} and {g6.loc['NHANES post','NNS']:.1f}), as is the 7-item variant "
          f"(NNS {g7.loc['NHANES pre','NNS']:.1f} and {g7.loc['NHANES post','NNS']:.1f}). For the two external cohorts after local recalibration, and for the internal hold-out on its original probabilities, net benefit is positive and exceeds both default strategies at threshold "
          "probabilities of 0.10-0.15 in every dataset and for every model size, supporting use as a low-burden digital triage aid rather than a diagnostic test.\n")
splice("appendix3_utility.md", "Tuned to a triage-appropriate", "\n## Table S4.", interp)

rows = ["| Specification | Internal | NHANES pre | NHANES post |", "|---|---|---|---|"]
for _c, r in sens.iterrows(): rows.append(f"| {_c} | {R(r.internal)} | {R(r.ext_pre)} | {R(r.ext_post)} |")
splice_table("appendix3_utility.md", "| Specification | Internal | NHANES pre | NHANES post |", "\n".join(rows) + "\n")
rows = ["| Items | Predictor added | Selection-set AUROC | Hold-out AUROC | NHANES pre | NHANES post |", "|---|---|---|---|---|---|"]
for _, r in par.iterrows():
    b = "**" if int(r.k) == KSTAR else ""
    rows.append(f"| {b}{int(r.k)}{b} | {b}{r.added}{b} | {b}{r.selection_AUROC:.4f}{b} | {b}{r.holdout_AUROC:.4f}{b} | {b}{r.ext_pre:.4f}{b} | {b}{r.ext_post:.4f}{b} |")
splice_table("appendix3_utility.md", "| Items | Predictor added |", "\n".join(rows) + "\n")
rows = ["| Cohort | Subgroup | n | Weighted AUROC | 95% CI | Calibration slope | CITL | FPR | FNR |", "|---|---|---|---|---|---|---|---|---|"]
for _, r in fair.iterrows():
    lab = str(r.subgroup).split(": ")[-1] if ":" in str(r.subgroup) else str(r.subgroup)
    rows.append(f"| {r.cohort} | {lab} | {int(r.n):,} | {R(r.AUROC)} | {R(r.CI_lo)}-{R(r.CI_hi)} | {R(r.cal_slope)} | {sgn(r.CITL,3)} | {R(r.FPR)} | {R(r.FNR)} |")
splice_table("appendix3_utility.md", "| Cohort | Subgroup | n | Weighted AUROC", "\n".join(rows) + "\n")
patch("appendix3_utility.md", [
 (r"rule applied post hoc in Interpretability, Fairness, and Clinical Utility \| [^|]*\|",
  "rule applied post hoc in Interpretability, Fairness, and Clinical Utility | " + ("; ".join(f"{slab(f[1])} in {LAB[f[0]]} ({sgn(f[2],3)})" for f in flags) if flags else "no subgroup flagged") + " |"),
])

# ══════════════ [7] 부록4 - 오라클 표 · 전이 모델 수치 ══════════════
print("[7] 부록4 - 오라클")
lines = ["| Learner | Oracle AUROC (5-fold CV) | 95% CI |", "|---|---|---|"]
for _, r in orc.iterrows(): lines.append(f"| {r.learner} | {R(r.oracle_AUROC)} | {R(r.CI_lo)}-{R(r.CI_hi)} |")
_b = orc.sort_values("oracle_AUROC").iloc[-1]
lines.append(f"| {_b.learner}, refitted inside each bootstrap replicate | {R(_b.oracle_AUROC)} | {R(orc_nest.CI_lo)}-{R(orc_nest.CI_hi)} |")
splice_table("appendix4_ceiling.md", "| Learner | Oracle AUROC (5-fold CV) | 95% CI |", "\n".join(lines))
patch("appendix4_ceiling.md", [
 (r"The transported LightGBM model \(pre 0\.\d{3}, post 0\.\d{3}\)",
  f"The transported LightGBM model (pre {col(PRE,'AUROC')}, post {col(POST,'AUROC')})"),
])


# ══════════════ 부록3 표 S16 - 보정 구간 + 두 주기 AUROC 차 ══════════════
print("[8] 부록3 표 S16 - 보정 구간")
_s16 = ""
for _, _r in calci.iterrows():
    _s16 += (f"| {_r['set']} | {_r['probabilities']} | {R(_r.cal_slope,3)} ({R(_r.slope_lo,3)}-{R(_r.slope_hi,3)}) | "
             f"{sgn(_r.CITL)} ({sgn(_r.CITL_lo)} to {sgn(_r.CITL_hi)}) | {R(_r.ECE)} |\n")
_S16H = "| Cohort | Probabilities | Calibration slope (95% CI) | CITL (95% CI) | ECE |\n|---|---|---|---|---|\n"
splice_table("appendix4_ceiling.md", "| Cohort | Probabilities | Calibration slope", _S16H + _s16)
patch("appendix4_ceiling.md", [
 (r"External AUROC differed between cycles by [+-]0\.\d{3} \(95% CI [+-]0\.\d{3} to [+-]0\.\d{3}; bootstrap P=\.\d{2}\)",
  f"External AUROC differed between cycles by {sgn(dauc.delta,3)} (95% CI {sgn(dauc.CI_lo,3)} to {sgn(dauc.CI_hi,3)}; bootstrap P={format(dauc.P, '.2f').lstrip('0')})"),
])
patch("manuscript_17var.md", [
 (r"the difference is [+-]0\.\d{3} \(95% CI [+-]0\.\d{3} to [+-]0\.\d{3}; bootstrap P=\.\d{2}\)",
  f"the difference is {sgn(dauc.delta,3)} (95% CI {sgn(dauc.CI_lo,3)} to {sgn(dauc.CI_hi,3)}; bootstrap P={format(dauc.P, '.2f').lstrip('0')})"),
 (r"restored by intercept/slope recalibration to slope \d\.\d{2} \(95% CI \d\.\d{2}-\d\.\d{2}\) and \d\.\d{2} \(\d\.\d{2}-\d\.\d{2}\)",
  f"restored by intercept/slope recalibration to slope {CV('NHANES pre','cal_slope')} (95% CI {CCI('NHANES pre','after intercept/slope recalibration','slope')}) and "
  f"{CV('NHANES post','cal_slope')} ({CCI('NHANES post','after intercept/slope recalibration','slope')})"),
 (r"to slope \d\.\d{2} \(\d\.\d{2}-\d\.\d{2}\) and \d\.\d{2} \(\d\.\d{2}-\d\.\d{2}\), with the update refitted inside every replicate",
  f"to slope {CV('NHANES pre','cal_slope')} ({CCI('NHANES pre','after intercept/slope recalibration','slope')}) and "
  f"{CV('NHANES post','cal_slope')} ({CCI('NHANES post','after intercept/slope recalibration','slope')}), with the update refitted inside every replicate"),
 (r"gives visibly wider slope intervals \(\d\.\d{2}-\d\.\d{2} pre and \d\.\d{2}-\d\.\d{2} post",
  f"gives visibly wider slope intervals ({CCI('NHANES pre','after intercept/slope recalibration (fixed)','slope')} pre and {CCI('NHANES post','after intercept/slope recalibration (fixed)','slope')} post"),
])

print("\n완료. 이어서 python 60_verify_manuscript.py 로 확인하세요.")
