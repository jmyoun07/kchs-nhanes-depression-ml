# -*- coding: utf-8 -*-
"""
08_parsimonious.py (교차검증 #17) — 경량모델. 특징랭킹을 TRAIN에서만 산출(누수 제거),
비교군=실제 10변수 primary. 결과: results/parsimonious.csv
"""
import os, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb, shap

ROOT = r"C:/Users/rudtn/Desktop/claude code/지역사회건강"; PROC = os.path.join(ROOT, "data/processed"); RES = os.path.join(ROOT, "results")
SEED = 42
NUM = ["age", "sleep_hrs", "bmi", "income_q", "srh"]; BIN = ["partnered", "living_alone", "current_drinker", "active", "htn", "dm"]; CAT = ["sex", "edu", "smoking"]
ALL = NUM + BIN + CAT
dev = pd.read_parquet(os.path.join(PROC, "kchs_dev.parquet"))
y = dev["dep"].astype(int).values
tr, ho = train_test_split(np.arange(len(dev)), test_size=0.2, stratify=y, random_state=SEED)

def build(num, binf, catf):
    ct = ColumnTransformer(
        [("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num),
         ("bin", SimpleImputer(strategy="most_frequent"), binf),
         ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")), ("o", OneHotEncoder(handle_unknown="ignore", drop="first"))]), catf)])
    return Pipeline([("pre", ct), ("clf", lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31,
                     subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1))])

# ── TRAIN에서만 SHAP 랭킹(누수 제거) ──
full = build(NUM, BIN, CAT).fit(dev.iloc[tr][ALL], y[tr])
Xs = dev.iloc[tr][ALL].sample(min(30000, len(tr)), random_state=SEED)
Xt = full.named_steps["pre"].transform(Xs)
sv = shap.TreeExplainer(full.named_steps["clf"]).shap_values(Xt)
if isinstance(sv, list): sv = sv[1]
fnames = list(full.named_steps["pre"].get_feature_names_out())
imp = pd.Series(np.abs(sv).mean(0), index=fnames)
def base_feat(enc):
    core = enc.split("__", 1)[1]
    for f in ALL:
        if core == f or core.startswith(f + "_"): return f
    return core
order = imp.groupby(base_feat).sum().sort_values(ascending=False).index.tolist()
order = [f for f in order if f in ALL]
print("TRAIN SHAP 순위:", order)

# 비교군=실제 10변수 full
pf = full.predict_proba(dev.iloc[ho][ALL])[:, 1]; full_auc = roc_auc_score(y[ho], pf)
rows = []
for K in [3, 4, 5, 6, 8, 10, 14]:
    feats = order[:K]
    pipe = build([f for f in feats if f in NUM], [f for f in feats if f in BIN], [f for f in feats if f in CAT]).fit(dev.iloc[tr][feats], y[tr])
    p = pipe.predict_proba(dev.iloc[ho][feats])[:, 1]; auc = roc_auc_score(y[ho], p)
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    sl = LogisticRegression(penalty=None, solver="lbfgs").fit(lp.reshape(-1, 1), y[ho]).coef_[0, 0]
    rows.append({"K": K, "features": ", ".join(feats), "AUROC": round(auc, 4), "AUPRC": round(average_precision_score(y[ho], p), 4),
                 "Brier": round(brier_score_loss(y[ho], p), 4), "cal_slope": round(float(sl), 3), "ΔAUROC_vs_full10": round(auc - full_auc, 4)})
res = pd.DataFrame(rows); res.to_csv(os.path.join(RES, "parsimonious.csv"), index=False)
print(f"full(10변수) AUROC={full_auc:.4f}"); print(res.to_string(index=False))
rec = res[res["ΔAUROC_vs_full10"] > -0.02].iloc[0]
print(f"\n★ 권장 경량: 상위 {int(rec['K'])}변수 (ΔAUROC={rec['ΔAUROC_vs_full10']})")
