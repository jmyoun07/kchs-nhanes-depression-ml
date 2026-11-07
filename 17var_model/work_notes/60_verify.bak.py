# -*- coding: utf-8 -*-
"""60_verify_manuscript.py - 원고 수치의 기계 검증 (반복 실행해도 같은 결과가 나오게 하는 장치).

문제의식: 재계산으로 숫자가 바뀔 때마다 7개 문서에 손으로 옮겨 적어서 세대 혼재·출처 불일치가 반복됐다.
해결: "정본 결과 파일 → 기대 문자열"을 코드로 정의하고, 문서에 그 문자열이 실제로 있는지 기계적으로 확인한다.
      또한 폐기된(superseded) 결과 파일 이름이 어느 문서에도 인용되지 않는지 확인한다.

사용: python 60_verify_manuscript.py    → 실패 항목이 있으면 exit code 1
"""
import io, os, re, sys, glob
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
sys.stdout.reconfigure(encoding="utf-8")

def R(v, nd=3):
    """보고 관례에 맞춘 half-up 반올림. Python의 기본 서식은 이진 표현 탓에 0.8075→0.807처럼 내려간다."""
    q = Decimal(1).scaleb(-nd)
    return f"{Decimal(str(v)).quantize(q, rounding=ROUND_HALF_UP):.{nd}f}"

D = os.path.dirname(os.path.abspath(__file__))
RES = f"{D}/results"; DOCS = f"{D}/docs"
def rd(name): return pd.read_csv(f"{RES}/{name}")
def doc(name): return io.open(f"{DOCS}/{name}", encoding="utf8").read()

M   = doc("manuscript_17var.md")
A1  = doc("TRIPOD_AI_checklist.md"); A2 = doc("appendix2_crosswalk.md")
A3  = doc("appendix3_utility.md");   A4 = doc("appendix4_ceiling.md")
A5  = doc("appendix5_methods.md")
MC  = doc("model_card_17var.md");    CL = doc("cover_letter.md"); LG = doc("evidence_ledger.md")
ALLDOCS = {"manuscript": M, "appendix1": A1, "appendix2": A2, "appendix3": A3,
           "appendix4": A4, "appendix5": A5, "model_card": MC, "cover_letter": CL, "ledger": LG}

fails, checks = [], 0
def need(cond, label):
    global checks; checks += 1
    if not cond: fails.append(label)

def in_doc(token, where=("manuscript",), label=None):
    """token 문자열이 지정 문서들에 존재하는지."""
    lab = label or token
    for w in where:
        need(token in ALLDOCS[w], f"[{w}] missing: {lab}")

# ══════════════════════════════════════════════════════════════════════
# 1. 정본 파일 → 기대 문자열
# ══════════════════════════════════════════════════════════════════════
PM = rd("primary2_metrics.csv").set_index("set")
m2 = PM
g  = rd("primary2_gain.csv").set_index("set")
f  = rd("primary2_fairness.csv")
o  = rd("review4_oracle_grouped.csv"); o = o[~o.learner.str.contains("refit")]; best = o.sort_values("oracle_AUROC").iloc[-1]
onest = rd("review4_oracle_grouped.csv"); onest = onest[onest.learner.str.contains("refit")].iloc[0]
reg = rd("experiment_registry.csv")
pc = rd("primary2_parsimonious.csv").set_index("k")
opy = rd("primary2_operating.csv").set_index("set")
sh = rd("primary2_shap.csv").set_index(rd("primary2_shap.csv").columns[0]).iloc[:,0]
pv = rd("review2_prevalence.csv").set_index("cohort")
ov = rd("primary2_design.csv").iloc[0]
cs = rd("continuous_score.csv").set_index("set")
tb = rd("sap_tiebreak.csv").set_index("algorithm")
al = rd("sap_alcohol_alt.csv")
sen= rd("primary2_sensitivity.csv").set_index("cond")
ch = rd("primary2_characteristics.csv").set_index("characteristic")
nb = rd("primary2_netbenefit_models.csv")
to = rd("primary2_triage.csv")
opr= rd("primary2_operating_ref.csv")
bv = rd("bootstrap_validation.csv").set_index("cohort")

# 내부/외부 판별력·CI
in_doc(f"{R(PM.loc['Internal(hold-out)','AUROC'])} (95% CI {R(PM.loc['Internal(hold-out)','CI_lo'])}-{R(PM.loc['Internal(hold-out)','CI_hi'])})", ("manuscript",), "internal AUROC+CI")
in_doc(f"| AUROC (95% CI) | {R(PM.loc['Internal(hold-out)','AUROC'])} ({R(PM.loc['Internal(hold-out)','CI_lo'])}-{R(PM.loc['Internal(hold-out)','CI_hi'])}) | {R(m2.loc['NHANES pre','AUROC'])} ({R(m2.loc['NHANES pre','CI_lo'])}-{R(m2.loc['NHANES pre','CI_hi'])}) |",
       ("manuscript",), "Table 2 AUROC row")
in_doc(f"| AUPRC (95% CI) | {R(PM.loc['Internal(hold-out)','AUPRC'])} ({R(PM.loc['Internal(hold-out)','AUPRC_lo'])}-{R(PM.loc['Internal(hold-out)','AUPRC_hi'])}) |",
       ("manuscript",), "Table 2 AUPRC row")
in_doc(f"| CITL (95% CI) | +{R(m2.loc['Internal(hold-out)','CITL'],2)} | +{R(m2.loc['NHANES pre','CITL'],2)} (",
       ("manuscript",), "Table 2 CITL row")
# 모델카드도 같은 CI
in_doc(f"{R(PM.loc['Internal(hold-out)','AUROC'])} ({R(PM.loc['Internal(hold-out)','CI_lo'])}-{R(PM.loc['Internal(hold-out)','CI_hi'])})", ("model_card",), "model card internal CI")
in_doc(f"{R(m2.loc['NHANES pre','AUROC'])} ({R(m2.loc['NHANES pre','CI_lo'])}-{R(m2.loc['NHANES pre','CI_hi'])})", ("model_card",), "model card pre CI")
in_doc(f"{R(m2.loc['NHANES post','AUROC'])} ({R(m2.loc['NHANES post','CI_lo'])}-{R(m2.loc['NHANES post','CI_hi'])})", ("model_card",), "model card post CI")

# 14→17 이득
for st, lab in [("NHANES pre","pre"), ("NHANES post","post")]:
    r = g.loc[st]
    in_doc(f"Δ+{R(r.delta)}, 95% CI +{R(r.CI_lo)} to +{R(r.CI_hi)}", ("manuscript",), f"gain {lab}")

# 오라클
in_doc(f"{R(o.oracle_AUROC.min())}-{R(o.oracle_AUROC.max())} (best learner {R(best.oracle_AUROC)}, 95% CI {R(best.CI_lo)}-{R(best.CI_hi)}",
       ("manuscript",), "oracle range+CI")
in_doc(f"{R(onest.CI_lo)}-{R(onest.CI_hi)} when the oracle is refitted", ("manuscript",), "oracle nested CI")
in_doc(f"{int(reg['rows'].sum()):,} recorded runs across {len(reg)} experiment files", ("manuscript",), "experiment registry count")
for _, r in o.iterrows():
    in_doc(f"| {r.learner} | {R(r.oracle_AUROC)} | {R(r.CI_lo)}-{R(r.CI_hi)} |", ("appendix4",), f"oracle table {r.learner}")

# 경량모델
KS = int(rd("primary2_design.csv").iloc[0].kstar)
in_doc(f"hold-out AUROC {R(pc.loc[KS,'holdout_AUROC'])}, NHANES pre {R(pc.loc[KS,'ext_pre'])} and NHANES post {R(pc.loc[KS,'ext_post'])}", ("manuscript",), "deployment-model evaluation")
for k in pc.index:
    need(f"| {k} |" in A3 or f"| **{k}** |" in A3, f"[appendix3] Table S6 missing row k={k}")

# Youden (정확)
in_doc(f"{R(opy.loc['Internal(hold-out)','sens'])}/{R(opy.loc['Internal(hold-out)','spec'])} internally, {R(opy.loc['NHANES pre','sens'])}/{R(opy.loc['NHANES pre','spec'])} in NHANES pre and {R(opy.loc['NHANES post','sens'])}/{R(opy.loc['NHANES post','spec'])} in NHANES post",
       ("manuscript",), "development-fixed operating point")

# SHAP (학습셋 기준)
rank = {v: i+1 for i, v in enumerate(sh.sort_values(ascending=False).index)}
in_doc(f"self-rated health ({R(sh['srh'],2)})", ("manuscript",), "SHAP srh")
in_doc(f"({R(sh['food_insecure'],2)}, ranked {rank['food_insecure']}th)", ("manuscript",), "SHAP food insecurity rank")

# 하위군: 모든 CI가 원고 또는 Table S9에 정확히 존재
for _, r in f.iterrows():
    tok = f"{R(r.CI_lo)}-{R(r.CI_hi)}"
    need(tok in A3, f"[appendix3] Table S7 missing CI {r.cohort}/{r.subgroup} {tok}")
    # 본문에는 학력·소득분위를 뺀 모든 하위군 CI 가 값 그대로 있어야 한다(전값은 표 S7 에도).
    if not any(t in str(r.subgroup) for t in ("Education", "Income quintile")):
        need(tok in M, f"[manuscript] 하위군 CI 누락 {r.cohort}/{r.subgroup} {tok}")

# 유병률·중첩·연속점수·특성
in_doc(f"{R(pv.loc['KCHS','available_case_pct'],1)}% in KCHS vs {R(pv.loc['NHANES pre','available_case_pct'],1)}% and {R(pv.loc['NHANES post','available_case_pct'],1)}%",
       ("manuscript",), "food insecurity available-case")
in_doc("sharing no cluster", ("manuscript",), "cluster-independent hold-out")
in_doc(f"ρ={R(cs.loc['Internal(hold-out)','spearman_classifier'])} internally, {R(cs.loc['NHANES pre','spearman_classifier'])} pre-pandemic and {R(cs.loc['NHANES post','spearman_classifier'])} post-pandemic",
       ("manuscript",), "continuous-score rho")
in_doc(f"bootstrap {R(bv.loc['NHANES pre','bootstrap_SE'],5)} vs design {R(bv.loc['NHANES pre','design_SE'],5)}",
       ("appendix5",), "bootstrap validation pre")

# SAP 동률 판정
in_doc(f"({R(tb.loc['LightGBM','cal_slope'])} for LightGBM against {R(tb.loc['Random forest','cal_slope'])} for random forest)",
       ("appendix5",), "SAP tie-break slopes")

# 민감도 표(Appendix 3 Table S5): primary2_sensitivity 와 일치
sn2 = rd("primary2_sensitivity.csv")
for _, r in sn2.iterrows():
    need(f"| {R(r.internal)} | {R(r.ext_pre)} | {R(r.ext_post)} |" in A3, f"[appendix3] Table S5 row mismatch: {r['cond']}")

# 민감도 표(Table S5)와 §3.6 본문 - 모든 행이 정본과 일치하는지
for _c, _r in sen.iterrows():
    need(f"| {_c} | {R(_r['internal'])} | {R(_r['ext_pre'])} | {R(_r['ext_post'])} |" in A3,
         f"[appendix3] Table S5 row mismatch {_c}")
for _c in ["Weighted training", "Missing-indicator imputation", "Single chained-equations imputation",
           "Age ≥30 (train and evaluate)", "Outcome PHQ-9 ≥5"]:
    _r = sen.loc[_c]
    need(f"({R(_r['internal'])}/{R(_r['ext_pre'])}/{R(_r['ext_post'])}" in M, f"[manuscript] §3.6 missing {_c}")

# 특성 표(Table S4)
for k in ["Food insecure", "BMI", "Sleep hours"]:
    r = ch.loc[k]
    v = f"{R(r['KCHS'],1)}%" if k == "Food insecure" else f"{R(r['KCHS'],2)}"
    need(v in A3, f"[appendix3] Table S4 missing {k} = {v}")

# 순편익·운영지점 표
for _, r in nb.iterrows():
    need(f"| {r['net_benefit']:+.4f} | {r['screen_all_NB']:+.4f} |" in A3, f"[appendix3] Table S3 row mismatch {r['model']}/{r['set']}/{r['threshold']}")
for _, r in to.iterrows():
    need(f"| {R(r['spec'])} | {R(r['PPV'])} | {R(r['NPV'])} |" in A3, f"[appendix3] Table S2 row mismatch {r['model']}/{r['set']}/{r['target']}")
for _, r in opr.iterrows():
    need(f"| {r['value']:.4f} | {R(r['sens'])} | {R(r['spec'])} |" in A3, f"[appendix3] Table S1 row mismatch {r['set']}/{r['threshold']}")

# ── 하위군: 점추정과 신뢰구간이 같은 세대인지 (혼재 방지: 점추정이 CI 안에 있어야 한다) ──
for _, r in f.iterrows():
    need(r['CI_lo'] - 1e-9 <= r['AUROC'] <= r['CI_hi'] + 1e-9,
         f"[결과파일] 하위군 점추정이 CI 밖 {r['cohort']}/{r['subgroup']}")
    need(f"{R(r['AUROC'])} ({R(r['CI_lo'])}-{R(r['CI_hi'])}" in M or f"| {R(r['AUROC'])} | {R(r['CI_lo'])}-{R(r['CI_hi'])} |" in A3,
         f"[하위군] 원고/부록에 없음 {r['cohort']}/{r['subgroup']}")

# ── 공정성 플래그: 0.05 초과 하위군이 모두 문서에 적혀 있는지 ──
_COH = {'KCHS hold-out': PM.loc['Internal(hold-out)','AUROC'], 'NHANES pre': m2.loc['NHANES pre','AUROC'], 'NHANES post': m2.loc['NHANES post','AUROC']}
for _, r in f.iterrows():
    d = r['AUROC'] - _COH[r['cohort']]
    if abs(d) > 0.05:
        _d = ("+" + R(d)) if d >= 0 else R(d)
        need(f"({_d})" in M, f"[공정성] 플래그 하위군 미기재 {r['cohort']}/{r['subgroup']} ({d:+.3f})")

# ── SHAP 상위 순서가 원고 서술과 일치하는지 ──
_sh = sh.sort_values(ascending=False)
need(all(f"({R(_sh.iloc[i],2)})" in M for i in range(5)), "[SHAP] 원고 상위 5개 값 불일치")

# ── 모델카드 지표표가 정본과 일치하는지 ──
for _k, _lab in [('Internal(hold-out)','Internal (KCHS hold-out)'), ('NHANES pre','NHANES pre-COVID'), ('NHANES post','NHANES post-COVID')]:
    need(f"| {_lab} | {R(PM.loc[_k,'AUROC'])} ({R(PM.loc[_k,'CI_lo'])}-{R(PM.loc[_k,'CI_hi'])}) | {R(PM.loc[_k,'cal_slope'],2)} " in MC,
         f"[model_card] 지표 행 불일치 {_lab}")

# ══════════════════════════════════════════════════════════════════════
# 2. 폐기 파일이 어느 문서에도 인용되지 않는지
# ══════════════════════════════════════════════════════════════════════
SUPERSEDED = ["final_metrics.csv","final_parsimonious.csv","gain_14to17_bootstrap.csv","fairness.csv",
              "fairness_ci.csv","review2_fairness.csv","review3_fairness.csv","oracle_ci.csv",
              "table_s1_operating.csv","table3_characteristics.csv","audit_recompute_metrics.csv",
              "audit_recompute_parsimonious.csv","audit_recompute_sensitivity.csv","audit_gain_retuned.csv",
              "shap_importance.csv","review3_oracle_ci.csv",
              # 교정 전(pre-correction) 세대 - primary2_*.csv 로 대체됨
              "review2_metrics.csv","review3_metrics.csv","review2_gain.csv","review3_fairness_domain.csv",
              "review2_operating.csv","parsimonious_clean.csv","final_sensitivity.csv","sensitivity_full.csv",
              "table_s4_operating.csv","table_s4_netbenefit.csv","characteristics_availablecase.csv",
              "shap_importance_train.csv","income_fix_sensitivity.csv","age_topcode_sensitivity.csv"]
for name, txt in ALLDOCS.items():
    if name == "ledger":  # 근거대장은 '폐기됨'을 명시적으로 기록해도 됨
        continue
    for sup in SUPERSEDED:
        need(sup not in txt, f"[{name}] cites superseded file: {sup}")

# ══════════════════════════════════════════════════════════════════════
# 3. 구조·표현 규칙
# ══════════════════════════════════════════════════════════════════════
body = M.split("## Figure Legends")[0]
order = [int(n) for m in re.findall(r"Figures? (\d)(?: and (\d))?", body) for n in m if n]
need(order == [1,2,3,4,5,6,7], f"figure citation order = {order}, expected 1..7")
refs = sorted(set(int(x) for x in re.findall(r"Table S(\d+)", M)))
have = sorted(set(int(x) for x in re.findall(r"## Table S(\d+)", A3 + A4)))
need(set(refs) <= set(have), f"cited appendix tables {refs} not all present {have}")
cited = sorted(set(int(x) for m in re.findall(r"\[([\d,\s-]+)\]", M) for x in re.findall(r"\d+", m)))
_reflist = sorted(int(x) for x in re.findall(r"^(\d+)\. ", M.split("## References")[-1], re.M))
need(_reflist == list(range(1, len(_reflist) + 1)), f"참고문헌 번호가 연속이 아님: {_reflist}")
need(cited == _reflist, f"본문 인용 {cited} 과 참고문헌 목록 {_reflist} 불일치")
need(len(_reflist) >= 30, f"참고문헌 {len(_reflist)}개 (목표 30개 이상)")
abst = M.split("## Abstract")[1].split("---")[0]
words = len(re.sub(r"\*\*|\[|\]", "", abst).split())
need(words <= 450, f"abstract {words} words > 450")

BANNED = [
 ("locked before analysis", "amendment timing overclaim"),
 ("preserved as missing rather than imputed to a constant", "missingness overclaim"),
 ("maximum discrimination extractable", "oracle overclaim"),
 ("intrinsic to the transportable predictor set", "causal overclaim"),
 ("multiple imputation by chained equations", "MICE mislabel"),
 ("PSU-cluster bootstrap", "old bootstrap label"),
 ("[to confirm]", "unresolved placeholder"),
 ("pre-specified interaction terms", "interaction provenance"),
]
for name, txt in ALLDOCS.items():
    if name == "ledger": continue
    for tok, lab in BANNED:
        need(tok not in txt, f"[{name}] banned phrase ({lab}): {tok!r}")

# ══════════════════════════════════════════════════════════════════════
# 3b. 문체 규칙 (GBD 원고에서 확립) - 커버레터는 편지 문체이므로 제외
#   R1 인칭·지시 대명사 금지  R2 because 금지  R3 결과의 so 금지('so that' 목적절은 허용)
# ══════════════════════════════════════════════════════════════════════
_PRON = re.compile(r"\b(we|We|our|Our|ours|they|They|their|Their|them|Them|themselves)\b"
                   r"|(?<![A-Za-z0-9])us(?![A-Za-z])")
def _mask_quoted_checklist(t):
    """TRIPOD 표의 'Checklist item' 열은 BMJ 체크리스트 원문을 그대로 옮긴 것이다.

    남이 쓴 문장이라 우리 문체 규칙(대명사·because·so)을 적용할 수 없고, 고쳐서도 안 된다.
    예시파일도 원문 그대로 싣는다. 그 열만 가리고 나머지 열과 표 밖 산문은 그대로 검사한다.
    """
    out = []
    for ln in t.splitlines():
        c = ln.split("|")
        if ln.startswith("|") and len(c) >= 7:
            c[4] = " (reproduced checklist text) "
            out.append("|".join(c))
        else:
            out.append(ln)
    return chr(10).join(out)

_STYLE_DOCS = {"manuscript": M, "appendix1": _mask_quoted_checklist(A1), "appendix2": A2,
               "appendix3": A3, "appendix4": A4, "appendix5": A5, "model_card": MC, "ledger": LG, "cover_letter": CL}
def _ctx(t, a, b):
    return t[max(0, a - 60): b + 60].replace("\n", " ")
for _n, _t in _STYLE_DOCS.items():
    for _m in _PRON.finditer(_t):
        need(False, f"[{_n}] 금지 대명사 {_m.group(0)!r}: …{_ctx(_t, _m.start(), _m.end())}…")
    for _m in re.finditer(r"\bbecause\b", _t, re.I):
        need(False, f"[{_n}] because 사용: …{_ctx(_t, _m.start(), _m.end())}…")
    for _m in re.finditer(r"(?<![A-Za-z])so (?!that\b)", _t):
        need(False, f"[{_n}] 결과의 so: …{_ctx(_t, _m.start(), _m.end())}…")
need(True, "문체 규칙(대명사·because·so) 검사 완료")
# 대시류 전면 금지 - 예시논문(JMIR)은 -, -, - 를 한 번도 쓰지 않고 하이픈만 쓴다
import glob as _g3
for _f3 in _g3.glob(f"{DOCS}/*.md"):
    if "_audit" in _f3: continue
    _n3 = os.path.basename(_f3); _s3 = io.open(_f3, encoding="utf8").read()
    for _ch, _nm in (("\u2014", "em dash"), ("\u2013", "en dash"), ("\u2212", "minus sign")):
        _hit = [l.strip()[:100] for l in _s3.splitlines() if _ch in l]
        need(not _hit, f"[{_n3}] {_nm} 사용 (예시논문은 하이픈만): {_hit[:2]}")
need(True, "대시 표기 검사 완료")

# 그림 파일 존재 + docx 생성기 목록과 범례 일치
FIGS = ["characteristics.png","calibration.png","roc_curves.png","sensitivity.png","shap_importance.png","subgroup_forest.png","dca.png"]
for i, fn in enumerate(FIGS, 1):
    need(os.path.exists(f"{RES}/figures/{fn}"), f"missing figure file {fn}")
mk = io.open(f"{D}/44_make_docx.py", encoding="utf8").read()
for i, fn in enumerate(FIGS, 1):
    need(f'("Figure {i}", "{fn}"' in mk, f"44_make_docx.py figure {i} != {fn}")

# ══════════════════════════════════════════════════════════════════════
# 4. 신선도(staleness) - 산출물이 원천보다 오래됐는지
# ══════════════════════════════════════════════════════════════════════
def mtime(path): return os.path.getmtime(path) if os.path.exists(path) else 0
FIG_SRC = {
 "characteristics.png": ["primary2_characteristics.csv"],
 "sensitivity.png":     ["primary2_sensitivity.csv"],
 "shap_importance.png": ["primary2_shap.csv"],
 "subgroup_forest.png": ["primary2_fairness.csv"],
}
for fig, srcs in FIG_SRC.items():
    fm = mtime(f"{RES}/figures/{fig}")
    for sc in srcs:
        need(fm >= mtime(f"{RES}/{sc}") - 1, f"stale figure: {fig} older than {sc} - regenerate")
DOC_SRC = {
 "논문원고_우울ML_17변수.docx": ["44_make_docx.py", "docs/manuscript_17var.md"] + [f"results/figures/{k}" for k in
   ["characteristics.png","calibration.png","roc_curves.png","sensitivity.png","shap_importance.png","subgroup_forest.png","dca.png"]],
 "부록1_보충방법및결과.docx":  ["45_make_appendix_docx.py", "docs/appendix2_crosswalk.md", "docs/appendix3_utility.md", "docs/appendix4_ceiling.md", "docs/appendix5_methods.md"],
 "부록2_TRIPOD체크리스트.docx": ["45_make_appendix_docx.py", "docs/TRIPOD_AI_checklist.md"],
}
for docx, srcs in DOC_SRC.items():
    dm = mtime(f"{D}/{docx}")
    need(dm > 0, f"missing output: {docx}")
    for sc in srcs:
        need(dm >= mtime(f"{D}/{sc}") - 1, f"stale docx: {docx} older than {sc} - rerun 44/45_make_*.py")

# ══════════════════════════════════════════════════════════════════════
# 5. 변수명 실재성 - 표에 적은 변수가 실제 조화 코드에 존재하는지
# ══════════════════════════════════════════════════════════════════════
HARM = ""
for f_ in [f"{D}/../src/01_harmonize.py", f"{D}/../src/15_ext_harmonize.py"]:
    if os.path.exists(f_): HARM += io.open(f_, encoding="utf8").read()
# 일부 변수명은 f-string 으로 생성돼 코드에 문자열로 나타나지 않는다 - 생성 결과를 펼쳐서 함께 본다
if 'f"mtb_07{c}1"' in HARM: HARM += " " + " ".join(f"mtb_07{c}1" for c in "abcdefghi")
if 'f"DPQ0{i}0"' in HARM: HARM += " " + " ".join(f"DPQ0{i}0" for i in range(1, 10))
_i1 = M.index("**Table 1.**")
t1 = M[_i1:M.index("\n\n", M.index("|  | Race/ethnicity", _i1))]
for var in re.findall(r"\b([A-Z]{2,4}[0-9]{3}[A-Z]?|[a-z]{3}_[0-9]{2}[a-z][0-9]|oba_bmi|mbhld_co|RID[A-Z]+|DMD[A-Z]+|IND[A-Z]+|BMX[A-Z]+|SLD[0-9]+|HUQ[0-9]+|FSDAD|OCD150|OHQ845)\b", t1):
    need(var in HARM, f"[Table 1] variable not found in harmonization code: {var}")
FE_TRUE = ["age","sleep_hrs","bmi","income_q","srh","oral_health","partnered","living_alone","current_drinker",
           "active","htn","dm","employed","food_insecure","sex","edu","smoking"]
pipe = io.open(f"{D}/20_final_pipeline.py", encoding="utf8").read()
for v in FE_TRUE:
    need(f'"{v}"' in pipe, f"predictor {v} missing from pipeline feature list")
_pipe_fe = [v for _g in re.findall(r"^(?:NUM|BIN|CAT)=\[(.*?)\]", pipe, re.M) for v in re.findall(r'"([a-z_]+)"', _g)]
need(sorted(_pipe_fe) == sorted(FE_TRUE), f"[pipeline] 예측변수 목록 불일치: {sorted(set(_pipe_fe) ^ set(FE_TRUE))}")
need(len(_pipe_fe) == 17, f"[pipeline] 예측변수 {len(_pipe_fe)}개 (17개여야 한다)")

# ══════════════════════════════════════════════════════════════════════

# ── DPQ 미응답: MEC 추정 프레임 기준(표 S8 정의 = ≥3문항 결측) ─────────────
_nr = rd("primary2_dpq_nonresponse.csv")
_LBL = {"<HS": "less than high school", "HS": "high school", "College": "college or more",
        "Asian": "non-Hispanic Asian", "Black": "non-Hispanic Black", "White": "non-Hispanic White"}
def _mec(period, grp, col="nr3_mec_pct"):
    return _nr[(_nr.period == period) & (_nr.group == grp)][col].iloc[0]
for _g, _lab in _LBL.items():
    _row = f"| Non-response (MEC frame), {_lab} | {_mec('pre', _g):.1f}% | {_mec('post', _g):.1f}% |"
    need(_row in A3, f"[appendix3] 표 S8 미응답 행 불일치: {_row}")
need(f"({_mec('pre','<HS'):.1f}% pre, {_mec('post','<HS'):.1f}% post, MEC frame)" in M,
     "[manuscript] 저학력 DPQ 미응답률이 MEC 프레임 정본과 불일치")
_ovr = lambda p_, c: _nr[(_nr.period == p_) & (_nr.factor == "__overall__")][c].iloc[0]
for _p, _n in (("pre", 8544), ("post", 6064)):
    need(int(_ovr(_p, "n_mec")) == _n, f"[results] MEC 프레임 n 불일치 {_p}: {_ovr(_p,'n_mec')}")
    need(f"{int(_ovr(_p,'n_mec')):,}" in M, f"[manuscript] MEC 프레임 n 미기재 {_p}")
# 면접 프레임 값이 미응답률로 단독 제시되지 않아야 한다
for _stale in ("20.2% pre, 45.3% post", "| Non-response, less than high school |"):
    need(all(_stale not in t for t in ALLDOCS.values()), f"[전체] 면접 프레임 값 잔존: {_stale}")

# ── 새로 붙인 구간들이 정본과 맞는지 ─────────────────────────────────────────
_cc = rd("primary2_calibration_ci.csv"); _da = rd("primary2_delta_auroc.csv").iloc[0]
for _, _r in _cc.iterrows():
    _cell = f"| {_r['set']} | {_r['probabilities']} | {R(_r.cal_slope,3)} ({R(_r.slope_lo,3)}-{R(_r.slope_hi,3)}) |"
    need(_cell in A4, f"[appendix4] 표 S16 보정 구간 불일치: {_cell}")
_pre = _cc[(_cc["set"]=="NHANES pre") & (_cc.probabilities=="after intercept/slope recalibration")].iloc[0]
_post = _cc[(_cc["set"]=="NHANES post") & (_cc.probabilities=="after intercept/slope recalibration")].iloc[0]
need("| Calibration slope (95% CI) |" in M, "[manuscript] 표 2 에 보정 기울기 구간이 없다")
need(f"{R(_pre.slope_lo,2)}-{R(_pre.slope_hi,2)}" in M, "[manuscript] 재보정 기울기 구간(pre) 불일치")
need(f"{R(_post.slope_lo,2)}-{R(_post.slope_hi,2)}" in M, "[manuscript] 재보정 기울기 구간(post) 불일치")
_d = ("+" if _da.delta >= 0 else "-") + R(abs(_da.delta),3)
need(f"difference {_d}," in M, f"[manuscript] 사후-사전 AUROC 차 불일치: {_d}")


# ── 그림은 JMIR 규격(300dpi 태그, 300-1200 x 225-900 px, 5MB 이하)을 지켜야 한다 ──
from PIL import Image as _Im
for _i in range(1, 8):
    _p = f"{D}/results/figures/" + ["characteristics","calibration","roc_curves","sensitivity","shap_importance","subgroup_forest","dca"][_i-1] + ".png"
    if not os.path.exists(_p): need(False, f"[figures] 없음: {_p}"); continue
    _im = _Im.open(_p); _dpi = _im.info.get("dpi", (0, 0))[0]
    need(_dpi >= 299.9, f"[figures] {os.path.basename(_p)} dpi {_dpi} < 300")
    need(300 <= _im.width <= 1200 and 225 <= _im.height <= 900, f"[figures] {os.path.basename(_p)} 규격 밖 {_im.size}")
    need(os.path.getsize(_p) <= 5_000_000, f"[figures] {os.path.basename(_p)} 5MB 초과")


# ── 상호참조 검사(92_crossref_audit.py)를 여기서 함께 돌린다 ──
import subprocess as _sp
_r = _sp.run([sys.executable, f"{D}/92_crossref_audit.py"], capture_output=True, text=True, encoding="utf-8")
checks += 1
if _r.returncode != 0:
    fails += [l.strip()[6:] for l in (_r.stdout or "").splitlines() if l.strip().startswith("FAIL:")] or ["[crossref] 검사 실패"]


# ── 독립감사 이후 새로 들어간 수치들 ─────────────────────────────────────────
_ipw = rd("primary2_nonresponse_ipw.csv").set_index("set")
for _k, _lab in (("NHANES pre", "pre-pandemic"), ("NHANES post", "post-pandemic")):
    need(f"{R(_ipw.loc[_k,'AUROC_ipw'])} vs {R(_ipw.loc[_k,'AUROC_base'])} {_lab}" in M,
         f"[manuscript] 응답성향 IPW 값 불일치 ({_k})")

_par = rd("primary2_parsimonious.csv").set_index("k")
_thr = _par.loc[17, "selection_AUROC"] - 0.02
need(f"falls {_thr - _par.loc[5,'selection_AUROC']:.4f} below" in M, "[manuscript] k=5 여유값 불일치")
need(f"clears it by {_par.loc[6,'selection_AUROC'] - _thr:.4f}" in M, "[manuscript] k=6 여유값 불일치")
need(f"reaches {R(_par.loc[5,'ext_pre'])} pre-pandemic and {R(_par.loc[5,'ext_post'])} post-pandemic" in M,
     "[manuscript] k=5 외부값 불일치")

_sn = rd("primary2_sensitivity.csv").set_index("cond")
_d17 = _sn.loc["Primary (17 predictors, corrected)"]; _d16 = _sn.loc["Food insecurity excluded (16)"]
_d14 = _sn.loc["Three added predictors excluded (14)"]
_fp = (_d17.ext_pre - _d16.ext_pre) / (_d17.ext_pre - _d14.ext_pre) * 100
_fs = (_d17.ext_post - _d16.ext_post) / (_d17.ext_post - _d14.ext_post) * 100
need(f"accounts for {_fp:.0f}% of the 17-versus-14 gap pre-pandemic and {_fs:.0f}% post-pandemic" in M,
     "[manuscript] 식품불안정 기여 비중 불일치")

_la = rd("primary2_livingalone_control.csv").iloc[1]
need(f"external AUROC {R(_la.ext_pre)} pre-pandemic and {R(_la.ext_post)} post-pandemic" in M,
     "[manuscript] living_alone 통제 대조 외부값 불일치")
need(f"internal discrimination {R(_la.internal)}" in M, "[manuscript] living_alone 통제 대조 내부값 불일치")
need("never imputed to a constant" not in M, "[manuscript] 결측 상수대체에 대한 거짓 문장 잔존")

_cc2 = rd("primary2_calibration_ci.csv")
need((_cc2.probabilities == "after intercept/slope recalibration (nested)").sum() == 2,
     "[results] 중첩 재보정 구간 행이 없다")
need("ECE (95% CI)" not in A3 + A4, "[appendix3] ECE 구간이 표 S16 에 남아 있다")
need("for the calibration slope, CITL and ECE" not in M, "[manuscript] ECE 구간 약속이 남아 있다")
need(not re.search(r"\*?P\*?\s*=\s*0?\.001\b", M), "[manuscript] 1000회 절차 하한을 등호로 표기(P<.001 이어야 한다)")


# ── 표 S4 는 전 행을 정본과 대조한다 (예측변수 5개를 뒤늦게 넣은 표라 서식이 갈리기 쉽다) ──
_CH = rd("primary2_characteristics.csv").set_index("characteristic")
_CONT = ("Age (years)", "BMI", "Sleep hours", "Self-rated health", "Oral health", "Household income quintile")
for _k, _r in _CH.iterrows():
    _f = ((lambda v: "-" if pd.isna(v) else R(v, 2)) if _k in _CONT
          else (lambda v: "not collected" if pd.isna(v) else f"{R(v,1)}%"))
    _row = (f"| {_k} | {_f(_r['KCHS'])} | {_f(_r['NHANES pre'])} | {_f(_r['NHANES post'])} | "
            f"{R(_r['KCHS missing%'],1)}% / {R(_r['NHANES pre missing%'],1)}% / {R(_r['NHANES post missing%'],1)}% |")
    need(_row in A3, f"[appendix3] 표 S4 행 불일치: {_row[:90]}")
_edu = [k for k in _CH.index if str(k).startswith("Education:")]
need(len(_edu) == 4, "[results] 학력 4수준 행이 없다")
for _c in ("KCHS", "NHANES pre", "NHANES post"):
    need(abs(sum(_CH.loc[k, _c] for k in _edu) - 100) < 0.05, f"[results] 학력 비율 합이 100%가 아니다 ({_c})")
# 모든 예측변수가 표 S4 에 있어야 한다
for _v in ("Living alone", "Current drinker", "Meets physical-activity recommendation",
           "Household income quintile", "Education: college or more"):
    need(_v in _CH.index, f"[results] 표 S4 에 예측변수 누락: {_v}")
# P 값 표기는 AMA 방식 하나로
import glob as _g
for _f in _g.glob(f"{DOCS}/*.md"):
    if "_audit" in _f: continue
    _bad = re.findall(r"\b\bP\s*[<=>]\s*0\.\d+", io.open(_f, encoding="utf8").read())
    need(not _bad, f"[{os.path.basename(_f)}] P 값에 선행 0 (AMA 방식 아님): {_bad[:3]}")


# ── AMA/JMIR 표기: ie, / eg, · 미국식 철자 · 치환 손상 없음 ─────────────────
import glob as _g2
for _f2 in _g2.glob(f"{DOCS}/*.md"):
    if "_audit" in _f2: continue
    _n2 = os.path.basename(_f2); _s2 = io.open(_f2, encoding="utf8").read()
    need(not re.search(r"i\.e\.|e\.g\.", _s2), f"[{_n2}] i.e./e.g. 잔존 (AMA 표기는 ie, / eg,)")
    need("behavioural" not in _s2.lower(), f"[{_n2}] 영국식 철자 behavioural 잔존")
    need(not re.search(r",\s*,", _s2), f"[{_n2}] 이중 쉼표")
    for _m2 in re.finditer(r"\b(ie|eg),", _s2):
        _pre = _s2[max(0, _m2.start() - 3):_m2.start()]
        _post = _s2[_m2.end():_m2.end() + 2]
        need(_m2.start() == 0 or _s2[_m2.start() - 1] in " (\n",
             f"[{_n2}] '{_m2.group(0)}' 앞 문자 이상: {_s2[max(0,_m2.start()-40):_m2.end()+20]!r}")
        need(not _pre.strip().endswith((".", "!", "?")),
             f"[{_n2}] '{_m2.group(0)}' 가 문장 첫머리: {_s2[max(0,_m2.start()-40):_m2.end()+20]!r}")
        need(_post.startswith(" ") and not _post.startswith("  "),
             f"[{_n2}] '{_m2.group(0)}' 뒤 공백 이상: {_s2[max(0,_m2.start()-20):_m2.end()+30]!r}")


# ── 문단 내 굵은 라벨 금지 (예시논문은 제목·초록라벨·절제목·캡션만 굵게) ──
_KEEPB = {"Background:", "Objective:", "Methods:", "Results:", "Conclusions:", "Keywords:",
          "Table 1.", "Table 2.", "Original Paper"}
_bodyM = M.split("## References")[0]
for _m in re.finditer(r"\*\*([A-Z][^*]{2,70}?)\*\*", _bodyM):
    need(_m.group(1) in _KEEPB, f"[manuscript] 문단 내 굵은 라벨: {_m.group(1)!r}")


# ── 자기점검: 소스에 제어문자가 섞이면 정규식이 조용히 죽는다 ──────────────
for _self in ("60_verify_manuscript.py", "61_sync_numbers.py", "92_crossref_audit.py"):
    _src = io.open(f"{D}/{_self}", encoding="utf8").read()
    _ctl = sorted({hex(ord(c)) for c in _src if (ord(c) < 32 or ord(c) == 127 or ord(c) in (0xA0, 0x200B, 0x200C, 0x200D, 0x2028, 0x2029, 0xFEFF)) and c not in "\n\t"})
    need(not _ctl, f"[{_self}] 소스에 제어문자 {_ctl} — 정규식이 무력화된다")

print("="*88)
print(f"검증 항목 {checks}건 · 실패 {len(fails)}건")
print("="*88)
for x in fails: print("  FAIL:", x)
if not fails: print("  모든 항목 통과 - 원고 수치가 정본 결과 파일과 일치합니다.")
sys.exit(1 if fails else 0)
