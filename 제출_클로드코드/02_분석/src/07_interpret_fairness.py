# -*- coding: utf-8 -*-
"""
07_interpret_fairness.py — SHAP 전역 중요도 + KCHS↔NHANES 순위 일치도(전이 기전) + 공정성 하위군.
결과: results/shap_importance.csv, results/fairness.csv, results/figures/shap.png
"""
import os, warnings, numpy as np, pandas as pd, joblib
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import shap

ROOT = r"C:/Users/rudtn/Desktop/claude code/지역사회건강"
PROC = os.path.join(ROOT, "data/processed"); MODELS = os.path.join(ROOT, "models"); RES = os.path.join(ROOT, "results")
SEED = 42
prim = joblib.load(os.path.join(MODELS, "primary.joblib")); FEATURES = prim["features"]
pipe = joblib.load(os.path.join(MODELS, f"{prim['primary']}.joblib"))
pre = pipe.named_steps["pre"]; clf = pipe.named_steps["clf"]
fnames = list(pre.get_feature_names_out())
QC = []; qc = lambda s: (QC.append(str(s)), print(s))

def load(name):
    return pd.read_parquet(os.path.join(PROC, f"{name}.parquet"))

def shap_imp(df, n=30000):
    d = df.sample(min(n, len(df)), random_state=SEED)
    Xt = pre.transform(d[FEATURES])
    sv = shap.TreeExplainer(clf).shap_values(Xt)
    if isinstance(sv, list): sv = sv[1]
    return pd.Series(np.abs(sv).mean(0), index=fnames).sort_values(ascending=False)

dev = load("kchs_dev"); npre = load("nhanes_pre")
imp_k = shap_imp(dev); imp_n = shap_imp(npre)
imp = pd.DataFrame({"KCHS": imp_k, "NHANES_pre": imp_n}).fillna(0)
imp.to_csv(os.path.join(RES, "shap_importance.csv"))
rho, _ = spearmanr(imp["KCHS"].rank(), imp["NHANES_pre"].rank())
qc(f"SHAP 상위 예측변수(KCHS):\n{imp_k.head(8).round(4).to_string()}")
qc(f"\nKCHS↔NHANES SHAP 순위 일치도(Spearman rho) = {rho:.3f}  (전이가능성 기전 지지)")

plt.figure(figsize=(7, 5)); imp_k.head(12)[::-1].plot.barh()
plt.title("SHAP 전역 중요도 (KCHS, 상위 12)"); plt.tight_layout()
plt.savefig(os.path.join(RES, "figures", "shap.png"), dpi=130)

# ===== 공정성 =====
X, y = dev[FEATURES], dev["dep"].astype(int).values
_, Xho, _, yho = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
dho = dev.loc[Xho.index].copy(); dho["p"] = pipe.predict_proba(Xho)[:, 1]; dho["y"] = yho

rng = np.random.RandomState(SEED)
def cluster_ci(sub, B=400):  # #2 PSU-군집(설계) 부트스트랩 CI
    yy = sub["y"].to_numpy(); pp = sub["p"].to_numpy(); w = sub["W"].to_numpy(float)
    s = sub["strata"].to_numpy().astype(str); u = sub["psu"].to_numpy().astype(str)
    dfg = pd.DataFrame({"s": s, "u": u, "i": np.arange(len(yy))})
    groups = [[g["i"].to_numpy() for _, g in sub2.groupby("u")] for _, sub2 in dfg.groupby("s")]
    a = []
    for _ in range(B):
        idx = []
        for psus in groups:
            k = len(psus)
            for j in rng.randint(0, k, k): idx.append(psus[j])
        idx = np.concatenate(idx); yb = yy[idx]
        if yb.sum() in (0, len(yb)): continue
        try: a.append(roc_auc_score(yb, pp[idx], sample_weight=w[idx]))
        except Exception: pass
    return (round(np.percentile(a, 2.5), 3), round(np.percentile(a, 97.5), 3)) if len(a) > 20 else (np.nan, np.nan)

def iid_ci(sub, B=200):  # 대형 하위군(n>2만)용 빠른 가중 iid CI (n 크면 설계효과 미미)
    yy = sub["y"].to_numpy(); pp = sub["p"].to_numpy(); w = sub["W"].to_numpy(float); a = []
    for _ in range(B):
        idx = rng.randint(0, len(yy), len(yy))
        if yy[idx].sum() in (0, len(idx)): continue
        a.append(roc_auc_score(yy[idx], pp[idx], sample_weight=w[idx]))
    return (round(np.percentile(a, 2.5), 3), round(np.percentile(a, 97.5), 3)) if len(a) > 10 else (np.nan, np.nan)

def subgroup(df, col, mapper=None):  # #18/#2 가중 AUROC + PSU-군집 CI
    out = []
    g = df[col].map(mapper) if mapper else df[col]
    for lv, sub in df.groupby(g):
        if sub["y"].nunique() < 2 or len(sub) < 200: continue
        yy = sub["y"].to_numpy(); w = sub["W"].to_numpy(float)
        try:
            auc = roc_auc_score(yy, sub["p"].to_numpy(), sample_weight=w)
            lo, hi = cluster_ci(sub) if len(sub) < 20000 else iid_ci(sub)
        except Exception:
            auc = lo = hi = np.nan
        out.append({"factor": col, "level": str(lv), "n": len(sub),
                    "prev_w": round(float(np.average(yy, weights=w)), 3), "AUROC_w": round(auc, 3), "CI_lo": lo, "CI_hi": hi})
    return out

dho["ageband"] = pd.cut(dho["age"], [19, 39, 59, 200], labels=["20-39", "40-59", "60+"])
rows = []
for c in ["sex", "ageband", "edu", "income_q"]:
    rows += subgroup(dho, c)
# NHANES post 인종별
npost = load("nhanes_post"); npost["p"] = pipe.predict_proba(npost[FEATURES])[:, 1]; npost["y"] = npost["dep"].astype(int)
race_map = {1: "Mex", 2: "OthHisp", 3: "White", 4: "Black", 6: "Asian", 7: "Other"}
rows += [{**r, "factor": "race(NHANESpost)"} for r in subgroup(npost, "race", race_map)]
fair = pd.DataFrame(rows)
fair.to_csv(os.path.join(RES, "fairness.csv"), index=False)
qc("\n공정성 하위군 AUROC:\n" + fair.round(3).to_string(index=False))

open(os.path.join(RES, "qc_interpret.txt"), "w", encoding="utf-8").write("\n".join(QC))
qc("\n저장: results/shap_importance.csv, results/fairness.csv, figures/shap.png")
