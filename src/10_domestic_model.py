# -*- coding: utf-8 -*-
"""
10_domestic_model.py — KCHS 국내 전용 모델(보조): 조화 변수 + KCHS 강한 예측인자(스트레스·자가건강).
조화모델(전이가능) vs 국내모델(성능) 의 AUROC 비교. 결과: results/domestic_vs_harmonized.csv
"""
import os, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb

ROOT = r"C:/Users/rudtn/Desktop/claude code/지역사회건강"; PROC = os.path.join(ROOT, "data/processed"); RES = os.path.join(ROOT, "results")
SEED = 42
NUM = ["age", "sleep_hrs", "bmi", "income_q", "srh"]; BIN = ["partnered", "living_alone", "current_drinker", "active", "htn", "dm"]; CAT = ["sex", "edu", "smoking"]
dev = pd.read_parquet(os.path.join(PROC, "kchs_dev.parquet"))

def build(num, binf, catf):
    ct = ColumnTransformer(
        [("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num),
         ("bin", SimpleImputer(strategy="most_frequent"), binf),
         ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")), ("o", OneHotEncoder(handle_unknown="ignore", drop="first"))]), catf)])
    return Pipeline([("pre", ct), ("clf", lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31,
                     subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1))])

y = dev["dep"].astype(int).values
tr, ho = train_test_split(np.arange(len(dev)), test_size=0.2, stratify=y, random_state=SEED)
specs = {
    "조화모델(전이가능, 14변수·자가건강포함)": (NUM, BIN, CAT),
    "국내모델(+스트레스 인지, Table2)": (NUM + ["stress"], BIN, CAT),
}
rows = []
for name, (num, binf, catf) in specs.items():
    feats = num + binf + catf
    pipe = build(num, binf, catf).fit(dev.iloc[tr][feats], y[tr])
    p = pipe.predict_proba(dev.iloc[ho][feats])[:, 1]
    rows.append({"모델": name, "변수수": len(feats), "AUROC": round(roc_auc_score(y[ho], p), 4),
                 "AUPRC": round(average_precision_score(y[ho], p), 4)})
res = pd.DataFrame(rows)
res.to_csv(os.path.join(RES, "domestic_vs_harmonized.csv"), index=False, encoding="utf-8-sig")
print(res.to_string(index=False))
print(f"\nsrh 결측%={dev['srh'].isna().mean()*100:.1f}, stress 결측%={dev['stress'].isna().mean()*100:.1f}")
