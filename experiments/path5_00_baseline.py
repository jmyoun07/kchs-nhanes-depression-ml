# -*- coding: utf-8 -*-
"""
path5_00_baseline.py — (1) 기준선 재현 확인 + native API 동치성 확인.

목표값: 내부 0.8310 / ext_pre 0.7740 / ext_post 0.7653
"""
import os, sys, time, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path5_common import *   # noqa
import lightgbm as lgb
from sklearn.pipeline import Pipeline

t0 = time.time()
k, pre, post = load()
y = k["dep"].astype(int).values
itr, iho = base_split(k)
print(f"split: train={len(itr)} holdout={len(iho)}  ({time.time()-t0:.1f}s)", flush=True)

# ---------------------------------------------------------------- A. 명세 그대로(sklearn pipeline)
ct = make_pre()
clf = lgb.LGBMClassifier(**SPEC)
pipe = Pipeline([("ct", ct), ("clf", clf)])
t = time.time()
pipe.fit(k.iloc[itr][FEATURES], y[itr])
print(f"fit {time.time()-t:.1f}s", flush=True)

pi = pipe.predict_proba(k.iloc[iho][FEATURES])[:, 1]
pp = pipe.predict_proba(pre[FEATURES])[:, 1]
po = pipe.predict_proba(post[FEATURES])[:, 1]
base = rec("A_baseline_spec_sklearn",
           wauc(y[iho], pi, k["W"].to_numpy(float)[iho]),
           wauc(pre["dep"], pp, pre["W"]),
           wauc(post["dep"], po, post["W"]),
           True, "과제 명세 코드 그대로(sklearn LGBMClassifier pipeline)")

TARGET = dict(internal=0.8310, ext_pre=0.7740, ext_post=0.7653)
ok = all(abs(round(base[kk], 4) - TARGET[kk]) < 5e-4 for kk in TARGET)
print("BASELINE REPRODUCED (3dp):", ok,
      {kk: round(base[kk], 4) for kk in TARGET}, flush=True)

# ---------------------------------------------------------------- B. native API 동치성
D = Data()
t = time.time()
b = train_native(D, "binary")
print(f"native fit {time.time()-t:.1f}s", flush=True)
r = D.evaluate_raw(predict_raw(b, D.Xho), predict_raw(b, D.Xpre), predict_raw(b, D.Xpost))
rec("B_baseline_native_binary", r["internal"], r["ext_pre"], r["ext_post"], True,
    "lgb.train(objective='binary') — 동일 설계행렬/파라미터. A와 동치여야 함", base=base)

# ---------------------------------------------------------------- C. 커스텀 objective로 동일 logloss 재현(동치성 2차 확인)
def logloss_obj(pred, ds):
    yv = ds.get_label()
    p = 1.0 / (1.0 + np.exp(-pred))
    return p - yv, np.maximum(p * (1.0 - p), 1e-12)

t = time.time()
b2 = train_native(D, None, fobj=logloss_obj,
                  params_extra=dict(boost_from_average=False))
print(f"custom-logloss fit {time.time()-t:.1f}s", flush=True)
r2 = D.evaluate_raw(predict_raw(b2, D.Xho), predict_raw(b2, D.Xpre), predict_raw(b2, D.Xpost))
rec("C_custom_logloss_noinit", r2["internal"], r2["ext_pre"], r2["ext_post"], True,
    "커스텀 objective=logloss (init=0). 커스텀 경로 sanity check", base=base)

# init_score를 기준선과 동일한 prior logit으로
prior = float(D.ytr.mean())
f0 = np.log(prior / (1 - prior))
t = time.time()
b3 = train_native(D, None, fobj=logloss_obj,
                  params_extra=dict(boost_from_average=False),
                  init_score=np.full(len(D.ytr), f0))
r3 = D.evaluate_raw(predict_raw(b3, D.Xho) + f0, predict_raw(b3, D.Xpre) + f0,
                    predict_raw(b3, D.Xpost) + f0)
rec("C2_custom_logloss_priorinit", r3["internal"], r3["ext_pre"], r3["ext_post"], True,
    "커스텀 objective=logloss + prior logit init. B와 동치여야 함", base=base)

dump("path5_00_baseline.csv")
print(f"total {time.time()-t0:.1f}s")
