# -*- coding: utf-8 -*-
"""71_utility_primary2.py — 임상유용성 표(S1·S4·S5)를 '교정된 주분석' 모델로 재계산.

Appendix 3 의 S1(Youden/유병률 운영점), S4(고민감도 triage 운영점), S5(순편익)은
정정 이전 세대(review2/table_s4_*)에서 만들어져 세대 혼재를 일으켰다.
이 스크립트는 70_primary_corrected.py 의 데이터 교정·군집분할·주모델을 그대로 재현한 뒤
세 표를 다시 만든다.  출력: results/primary2_operating_ref.csv, _triage.csv, _netbenefit_models.csv
"""
import os, sys, io, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__)); RES = f"{D}/results"

# ── 70 번 스크립트의 준비 구간(데이터 교정 → 군집분할 → 주모델 적합)을 그대로 실행 ──
src = io.open(f"{D}/70_primary_corrected.py", encoding="utf8").read()
head = src[: src.index('p_tr = model.predict_proba')]
g = {"__file__": f"{D}/70_primary_corrected.py", "__name__": "__prep__"}
exec(compile(head, "70_prep", "exec"), g)
K, P, S = g["K"], g["P"], g["S"]; FE = g["FE"]; tr, ho = g["tr"], g["ho"]; y = g["y"]
Who, Wp, Ws = g["Who"], g["Wp"], g["Ws"]; yho, yp, ys = g["yho"], g["yp"], g["ys"]
p_ho, p_pre, p_post = g["p_ho"], g["p_pre"], g["p_post"]
Pipeline, make_ct, lgb, LGB, SEED = g["Pipeline"], g["make_ct"], g["lgb"], g["LGB"], g["SEED"]
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

# ── 경량 변형(사전지정 6항목 · 사후 7항목)도 같은 파이프라인으로 적합 ──
par = pd.read_csv(f"{RES}/primary2_parsimonious.csv").set_index("k")
def fit_subset(k):
    fs = par.loc[k, "vars"].split(";")
    NU = [c for c in g["NUM"] if c in fs]; BI = [c for c in g["BIN"] if c in fs]; CA = [c for c in g["CAT"] if c in fs]
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    parts = []
    if NU: parts.append(("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), NU))
    if BI: parts.append(("bin", "passthrough", BI))
    if CA: parts.append(("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")), ("o", OneHotEncoder(handle_unknown="ignore", drop="first"))]), CA))
    m = Pipeline([("pre", ColumnTransformer(parts)), ("clf", lgb.LGBMClassifier(**LGB))]).fit(K.iloc[tr][fs], y[tr])
    return (m.predict_proba(K.iloc[ho][fs])[:, 1], m.predict_proba(P[fs])[:, 1], m.predict_proba(S[fs])[:, 1])
KSTAR = int(pd.read_csv(f"{RES}/primary2_design.csv").iloc[0].kstar)
print(f"[1] 경량 변형 적합 — 사전지정 {KSTAR}항목 · 사후 7항목", flush=True)
p6 = fit_subset(KSTAR); p7 = fit_subset(7)
MODELS = {f"{len(FE)}-predictor": (p_ho, p_pre, p_post), f"{KSTAR}-item": p6, "7-item": p7}
SETS = [("Internal", yho, Who, 0), ("NHANES pre", yp, Wp, 1), ("NHANES post", ys, Ws, 2)]
SETS_D = {0: K.iloc[ho], 1: P, 2: S}   # 세트별 원자료(PSU 그룹 산출용)

def at(yv, p, w, t):
    f = p >= t; W1 = w[yv == 1].sum(); W0 = w[yv == 0].sum()
    tp = w[(yv == 1) & f].sum(); fp = w[(yv == 0) & f].sum(); fn = W1 - tp; tn = W0 - fp
    return dict(sens=tp/W1, spec=tn/W0, PPV=tp/(tp+fp) if tp+fp > 0 else np.nan,
                NPV=tn/(tn+fn) if tn+fn > 0 else np.nan, flag=w[f].sum()/w.sum())

def thr_for_sens(yv, p, w, target):
    o = np.argsort(-p); W1 = w[yv == 1].sum()
    c = np.cumsum(w[o]*(yv[o] == 1))/W1
    i = int(np.searchsorted(c, target))
    return float(p[o][min(i, len(p)-1)])

# ── S1. Youden(관측 점수 전수) · 유병률 임계값 ──
print("[2] S1 운영점", flush=True)
r1 = []
for nm, yv, w, k in SETS:
    p = MODELS[f"{len(FE)}-predictor"][k]
    fpr, tpr, thr = roc_curve(yv, p, sample_weight=w)
    tY = float(thr[int(np.argmax(tpr - fpr))])
    prev = w[yv == 1].sum()/w.sum()
    # 유병률 임계값도 설계가중을 반영한 분위수로 잡는다(코덱스 지적 10)
    _o = np.argsort(p); _cw = np.cumsum(w[_o])/w.sum()
    tP = float(p[_o][int(np.searchsorted(_cw, 1-prev))])
    for lab, t in [("Youden (exact)", tY), ("prevalence", tP)]:
        r1.append(dict(set=nm, threshold=lab, value=round(t, 6), **{a: round(b, 6) for a, b in at(yv, p, w, t).items() if a != "flag"}))
pd.DataFrame(r1).to_csv(f"{RES}/primary2_operating_ref.csv", index=False, encoding="utf-8-sig")

# ── S4. 고민감도 triage 운영점 ──
print("[3] S4 triage 운영점", flush=True)
r4 = []
for mname, pv in MODELS.items():
    for nm, yv, w, k in SETS:
        p = pv[k]
        for tgt in (0.85, 0.90):
            t = thr_for_sens(yv, p, w, tgt); m = at(yv, p, w, t)
            r4.append(dict(model=mname, set=nm, target=f"sens {tgt:.2f}", spec=round(m["spec"], 6), PPV=round(m["PPV"], 6),
                           NPV=round(m["NPV"], 6), flag=round(m["flag"], 6), NNS=round(1/m["PPV"], 1)))
pd.DataFrame(r4).to_csv(f"{RES}/primary2_triage.csv", index=False, encoding="utf-8-sig")

# ── S5. 순편익(절편+기울기 재보정, 5-fold 교차적합) ──
print("[4] S5 순편익", flush=True)
def logit(p): p = np.clip(p, 1e-6, 1-1e-6); return np.log(p/(1-p))
def recal_is(yv, p, w, groups=None):
    """재보정은 PSU 단위로 교차적합한다(표 2 와 같은 기준, 코덱스 지적 11·13)."""
    lp = logit(p); oof = np.zeros_like(p)
    sp = (StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(lp, yv, groups=groups)
          if groups is not None else
          StratifiedKFold(5, shuffle=True, random_state=SEED).split(lp, yv))
    for a, b in sp:
        c = LogisticRegression(penalty=None, solver="lbfgs").fit(lp[a].reshape(-1, 1), yv[a], sample_weight=w[a])
        oof[b] = c.predict_proba(lp[b].reshape(-1, 1))[:, 1]
    return oof

def psu_groups(d):
    return (d["strata"].astype(str) + "|" + d["psu"].astype(str)).values
r5 = []
for mname, pv in MODELS.items():
    for nm, yv, w, k in SETS:
        # 내부 hold-out 은 재보정하지 않는다(자기 라벨로 재보정하면 낙관적, 70 과 같은 기준)
        pr = pv[k] if nm == "Internal" else recal_is(yv, pv[k], w, groups=psu_groups(SETS_D[k]))
        W = w.sum(); prev = w[yv == 1].sum()/W
        for pt in (0.10, 0.15):
            f = pr >= pt
            nb = (w[(yv == 1) & f].sum() - w[(yv == 0) & f].sum()*pt/(1-pt))/W
            nb_all = prev - (1-prev)*pt/(1-pt)
            r5.append(dict(model=mname, set=nm, threshold=f"{pt:.2f}", net_benefit=round(nb, 6),
                           screen_all_NB=round(nb_all, 6), vs_screen_all=round(nb-nb_all, 6),
                           vs_best_default=round(nb-max(nb_all, 0.0), 6), flag_rate=round(w[f].sum()/W, 6)))
pd.DataFrame(r5).to_csv(f"{RES}/primary2_netbenefit_models.csv", index=False, encoding="utf-8-sig")
print("\n저장: primary2_operating_ref.csv · primary2_triage.csv · primary2_netbenefit_models.csv\n완료.", flush=True)
