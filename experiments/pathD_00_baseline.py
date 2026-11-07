# -*- coding: utf-8 -*-
"""pathD_00_baseline.py — 기준선 재현 확인 (내부 0.8336 / pre 0.7681 / post 0.7682)."""
import sys, os, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathD_common import *
from sklearn.pipeline import Pipeline
import lightgbm as lgb

t0 = time.time()
k, pre, post = load()
itr, iho = split(k)
y = k["dep"].astype(int).values
print(f"load+split {time.time()-t0:.0f}s | train={len(itr)} test={len(iho)}")

pipe = Pipeline([("pre", make_pre()), ("clf", lgb.LGBMClassifier(**BASE_LGB))])
pipe.fit(k[FEATURES].iloc[itr], y[itr])
r = evaluate(pipe, k, pre, post, iho)
print(f"BASELINE internal={r['internal']:.4f} pre={r['ext_pre']:.4f} post={r['ext_post']:.4f}")

TARGET = dict(internal=0.8336, ext_pre=0.7681, ext_post=0.7682)
ok = all(abs(r[a] - b) < 0.0005 for a, b in TARGET.items())
print("MATCH(3dp):", ok, {a: round(r[a] - b, 5) for a, b in TARGET.items()})

pd.DataFrame([dict(name="baseline_lgbm_17var", **r, protocol_compliant=True,
                   note="spec reproduction")]).to_csv(
    os.path.join(RES, "pathD_baseline.csv"), index=False, encoding="utf-8-sig")
print(f"done {time.time()-t0:.0f}s")
