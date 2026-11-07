# -*- coding: utf-8 -*-
"""
pathC_00_baseline.py — 기준선 재현 확인.
과제 명시 기준선: 내부 0.8336 / 외부 pre 0.7681 / post 0.7682
두 가지 분할(full02: test_size=0.2, 500k: train500000/test100000)을 모두 돌려 어느 쪽이 일치하는지 확정.
출력: results/pathC_baseline.csv
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from pathC_common import load, make_pre, wauc, FEATURES, RES, SEED

LGB_PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=31,
                  subsample=0.8, colsample_bytree=0.8,
                  random_state=SEED, n_jobs=-1, verbose=-1)

rows = []
for split in ["full02", "500k"]:
    t0 = time.time()
    d = load(split)
    pipe = Pipeline([("pre", make_pre(True)), ("clf", lgb.LGBMClassifier(**LGB_PARAMS))])
    pipe.fit(d.Xtr, d.ytr)
    a_int = wauc(d.yho, pipe.predict_proba(d.Xho)[:, 1], d.Who)
    a_pre = wauc(d.ypre, pipe.predict_proba(d.Xpre)[:, 1], d.wpre)
    a_post = wauc(d.ypost, pipe.predict_proba(d.Xpost)[:, 1], d.wpost)
    rows.append(dict(split=split, n_train=len(d.Xtr), n_test=len(d.Xho),
                     internal=a_int, ext_pre=a_pre, ext_post=a_post,
                     secs=time.time() - t0))
    print(f"[{split}] ntr={len(d.Xtr)} nho={len(d.Xho)} "
          f"내부={a_int:.4f} pre={a_pre:.4f} post={a_post:.4f} [{time.time()-t0:.0f}s]")
    sys.stdout.flush()

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, "pathC_baseline.csv"), index=False, encoding="utf-8-sig")
print(df.to_string())
