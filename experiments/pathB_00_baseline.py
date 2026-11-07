# -*- coding: utf-8 -*-
"""pathB_00_baseline.py — 기준선 재현 검증 (내부 0.8336 / pre 0.7681 / post 0.7682)."""
import os, sys, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathB_common import *
from sklearn.pipeline import Pipeline
import lightgbm as lgb

t0 = time.time()
k, pre, post = load()
itr, iho = split(k)
y = k["dep"].astype(int).values
W = k["W"].to_numpy(float)
X = k[FEATURES]

pipe = Pipeline([("pre", make_pre()),
                 ("clf", lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31,
                                            subsample=0.8, colsample_bytree=0.8,
                                            random_state=SEED, n_jobs=-1, verbose=-1))])
pipe.fit(X.iloc[itr], y[itr])
a_int = wauc(y[iho], pipe.predict_proba(X.iloc[iho])[:, 1], W[iho])
res = {"internal": a_int}
for nm, d in [("pre", pre), ("post", post)]:
    p = pipe.predict_proba(d[FEATURES])[:, 1]
    res[nm] = wauc(d["dep"].astype(int).values, p, d["W"].to_numpy(float))
print(f"BASELINE internal={res['internal']:.4f} pre={res['pre']:.4f} post={res['post']:.4f} [{time.time()-t0:.0f}s]")
print("target      internal=0.8336 pre=0.7681 post=0.7682")
pd.DataFrame([dict(model="baseline_lgbm_17", **res)]).to_csv(
    os.path.join(RES, "pathB_baseline.csv"), index=False, encoding="utf-8-sig")
