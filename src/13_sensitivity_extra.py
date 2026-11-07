# -*- coding: utf-8 -*-
"""13 (프로토콜 §2.9·§2.12 보완) — ECE, 연속점수 민감도, BMI drop, 약조화변수 drop.
결과: results/sensitivity_extra.csv"""
import os, sys, warnings, numpy as np, pandas as pd, joblib
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import lightgbm as lgb

ROOT = r"C:/Users/rudtn/Desktop/claude code/지역사회건강"; PROC = ROOT + "/data/processed"; RES = ROOT + "/results"; MODELS = ROOT + "/models"
SEED = 42
prim = joblib.load(os.path.join(MODELS, "primary.joblib")); FE = prim["features"]; model = joblib.load(os.path.join(MODELS, "lightgbm.joblib"))
dev = pd.read_parquet(PROC + "/kchs_dev.parquet"); pre = pd.read_parquet(PROC + "/nhanes_pre.parquet"); post = pd.read_parquet(PROC + "/nhanes_post.parquet")
yk = dev["dep"].astype(int).values
tr, ho = train_test_split(np.arange(len(dev)), test_size=0.2, stratify=yk, random_state=SEED)

def ece(y, p, w, bins=10):
    d = pd.DataFrame({"y": y, "p": p, "w": w}); d["b"] = pd.qcut(d["p"], bins, labels=False, duplicates="drop"); W = w.sum(); e = 0
    for _, g in d.groupby("b"):
        e += g["w"].sum() / W * abs(np.average(g["p"], weights=g["w"]) - np.average(g["y"], weights=g["w"]))
    return e

# ---- ECE (내부/pre/post) + 연속점수(예측 vs PHQ총점 상관) ----
sets = {"내부": (dev.iloc[ho], yk[ho]), "pre": (pre, pre["dep"].astype(int).values), "post": (post, post["dep"].astype(int).values)}
print("== §2.9 ECE + 연속점수 민감도 ==")
rows = []
for nm, (d, y) in sets.items():
    p = model.predict_proba(d[FE])[:, 1]; w = d["W"].to_numpy(float)
    e = ece(y, p, w); rho, _ = spearmanr(p, d["phq_sum"])
    rows.append({"지표": f"{nm}", "ECE": round(e, 4), "예측-PHQ총점 Spearman": round(rho, 3)})
    print(f"  {nm}: ECE={e:.4f}, 예측↔PHQ연속 Spearman={rho:.3f}")

# ---- §2.12 BMI drop / 약조화변수 drop (전이 영향) ----
NUM = ["age", "sleep_hrs", "bmi", "income_q", "srh"]; BIN = ["partnered", "living_alone", "current_drinker", "active", "htn", "dm"]; CAT = ["sex", "edu", "smoking"]
def build(num, binf, catf):
    ct = ColumnTransformer([("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num),
                            ("bin", SimpleImputer(strategy="most_frequent"), binf),
                            ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")), ("o", OneHotEncoder(handle_unknown="ignore", drop="first"))]), catf)])
    return Pipeline([("pre", ct), ("clf", lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1))])
def auc3(num, binf, catf):
    f = num + binf + catf; pipe = build(num, binf, catf).fit(dev.iloc[tr][f], yk[tr])
    r = {"내부": roc_auc_score(yk[ho], pipe.predict_proba(dev.iloc[ho][f])[:, 1], sample_weight=dev.iloc[ho]["W"].to_numpy(float))}
    for nm, d in [("pre", pre), ("post", post)]:
        r[nm] = roc_auc_score(d["dep"].astype(int).values, pipe.predict_proba(d[f])[:, 1], sample_weight=d["W"].to_numpy(float))
    return r
print("\n== §2.12 변수 제외 민감도 (내부/pre/post AUROC) ==")
S = [("전체 14변수", NUM, BIN, CAT),
     ("BMI 제외(자가vs실측 이슈)", [x for x in NUM if x != "bmi"], BIN, CAT),
     ("약조화변수 제외(BMI·PA·음주·독거)", [x for x in NUM if x != "bmi"], ["partnered", "htn", "dm"], CAT)]
for name, n, b, c in S:
    r = auc3(n, b, c); rows.append({"지표": name, "내부": round(r["내부"], 4), "pre": round(r["pre"], 4), "post": round(r["post"], 4)})
    print(f"  {name:<34} 내부 {r['내부']:.4f} | pre {r['pre']:.4f} | post {r['post']:.4f}")
pd.DataFrame(rows).to_csv(RES + "/sensitivity_extra.csv", index=False, encoding="utf-8-sig")
