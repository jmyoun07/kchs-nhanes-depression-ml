# -*- coding: utf-8 -*-
"""pathD2_00_baseline.py — 기준선 재현 + LOYO/지역CV 타이밍 캘리브레이션."""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathD2_common import *  # noqa

t0 = time.time()
b = Bench()
print(f"load+split {time.time()-t0:.0f}s  train={len(b.itr):,} holdout={len(b.iho):,}")
print(f"KCHS dep={b.y.mean():.4f} | pre dep={b.ypre.mean():.4f} post dep={b.ypost.mean():.4f}")

rows = []

# ---- (1) 스펙 기준선: 80/20 + income_q train-only ----
t = time.time()
pipe = Pipeline([("pre", make_pre()), ("clf", lgb.LGBMClassifier(**LGB_BASE))])
pipe.fit(b.X.iloc[b.itr], b.y[b.itr])
r = b.evaluate(pipe, "D0_baseline_80_20_incq_trainonly",
               note="spec reproduction (corrected split)")
d = {a: round(r[a] - v, 4) for a, v in TARGET.items()}
ok = all(abs(r[a] - v) < 0.0006 for a, v in TARGET.items())
print(fmt_ext(r) + f"  [{time.time()-t:.0f}s]")
print(f"  vs TARGET {TARGET} -> Δ={d}  MATCH(3dp)={ok}")
r["fit_seconds"] = round(time.time() - t, 1)
rows.append(r)

# ---- (2) 스펙 문구대로 500k/100k (참고: 재현 실패) ----
t = time.time()
k2, pre2, post2 = load()
y2 = k2["dep"].astype(int).values
itr2, iho2 = train_test_split(np.arange(len(k2)), train_size=500000,
                              test_size=100000, stratify=y2, random_state=SEED)
p2 = Pipeline([("pre", make_pre()), ("clf", lgb.LGBMClassifier(**LGB_BASE))])
p2.fit(k2[FEATURES].iloc[itr2], y2[itr2])
W2 = k2["W"].to_numpy(float)
r2 = dict(name="D0_literal_500k_100k",
          internal=wauc(y2[iho2], p2.predict_proba(k2[FEATURES].iloc[iho2])[:, 1], W2[iho2]),
          ext_pre=wauc(b.ypre, p2.predict_proba(b.Xpre)[:, 1], b.wpre),
          ext_post=wauc(b.ypost, p2.predict_proba(b.Xpost)[:, 1], b.wpost),
          note="literal spec wording; does NOT reproduce target",
          fit_seconds=round(time.time() - t, 1))
print(fmt_ext(r2))
rows.append(r2)
del k2, p2

# ---- (3) 타이밍 캘리브레이션: LOYO 1폴드 ----
for n in (200000, 400000):
    t = time.time()
    tr, va, yv = next(iter(loyo_folds(b)))
    tr = b.sub(tr, n)
    pp = Pipeline([("pre", make_pre()), ("clf", lgb.LGBMClassifier(**LGB_BASE))])
    pp.fit(b.X.iloc[tr], b.y[tr])
    s = wauc(b.y[va], pp.predict_proba(b.X.iloc[va])[:, 1], b.W[va])
    print(f"  LOYO fold year={yv} train_n={n:,} auc={s:.4f} [{time.time()-t:.0f}s]")

save(rows, "pathD2_00_baseline.csv")
print(f"total {time.time()-t0:.0f}s")
