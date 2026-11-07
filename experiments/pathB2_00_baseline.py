# -*- coding: utf-8 -*-
"""pathB2_00 — 기준선 재현(내부 0.8336 / pre 0.7681 / post 0.7682) + 타이밍 측정."""
import os, sys, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathB2_common import *
from sklearn.pipeline import Pipeline
import lightgbm as lgb

BASE_PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=31,
                   subsample=0.8, colsample_bytree=0.8,
                   random_state=SEED, n_jobs=-1, verbose=-1)

t0 = time.time()
k, pre, post = load()
kk, itr, iho = baseline_split(k)
y = kk["dep"].astype(int).values
W = kk["W"].to_numpy(float)
X = kk[FEATURES]
print(f"loaded n={len(kk):,} train={len(itr):,} holdout={len(iho):,} [{time.time()-t0:.0f}s]", flush=True)

t1 = time.time()
pipe = Pipeline([("pre", make_pre()),
                 ("clf", lgb.LGBMClassifier(**BASE_PARAMS))])
pipe.fit(X.iloc[itr], y[itr])
fit_s = time.time() - t1
a_int = wauc(y[iho], pipe.predict_proba(X.iloc[iho])[:, 1], W[iho])
e = ext_eval(pipe, pre, post)
print(f"BASELINE(full train {len(itr):,})  internal={a_int:.4f} "
      f"pre={e['ext_pre']:.4f} post={e['ext_post']:.4f}  fit={fit_s:.0f}s")
print("target                             internal=0.8336 pre=0.7681 post=0.7682")

rows = [dict(model="baseline_lgbm17_fulltrain", n_train=len(itr), internal=a_int,
             ext_pre=e["ext_pre"], ext_post=e["ext_post"], fit_sec=round(fit_s, 1))]

# --- CV 대리표본 타당성 확인: 학습 서브샘플 크기별 성능/시간 -------------------
for ntr in [200000, 400000, 600000]:
    rng = np.random.RandomState(SEED)
    sub = rng.choice(itr, size=ntr, replace=False)
    t1 = time.time()
    p2 = Pipeline([("pre", make_pre()), ("clf", lgb.LGBMClassifier(**BASE_PARAMS))])
    p2.fit(X.iloc[sub], y[sub])
    fs = time.time() - t1
    ai = wauc(y[iho], p2.predict_proba(X.iloc[iho])[:, 1], W[iho])
    ee = ext_eval(p2, pre, post)
    print(f"  sub n={ntr:,}  internal={ai:.4f} pre={ee['ext_pre']:.4f} "
          f"post={ee['ext_post']:.4f}  fit={fs:.0f}s", flush=True)
    rows.append(dict(model=f"baseline_lgbm17_sub{ntr}", n_train=ntr, internal=ai,
                     ext_pre=ee["ext_pre"], ext_post=ee["ext_post"], fit_sec=round(fs, 1)))

pd.DataFrame(rows).to_csv(os.path.join(RES, "pathB2_00_baseline.csv"),
                          index=False, encoding="utf-8-sig")
print(f"total {time.time()-t0:.0f}s -> results/pathB2_00_baseline.csv")
