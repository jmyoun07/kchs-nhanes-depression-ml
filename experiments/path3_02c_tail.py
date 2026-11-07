# -*- coding: utf-8 -*-
"""path3_02c_tail.py — 02b에서 read-only 버그로 못 돈 나머지: implausible->nan, drop rows, 밀도비 진단."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from path3_common import Bench, Recorder, RES, SEED, FEATURES, make_pre, wauc

B = Bench()
R = Recorder(os.path.join(RES, "path3_02c_tail.csv"))
N500 = 500000
rng0 = np.random.RandomState(SEED)
tr500 = rng0.choice(B.pool, N500, replace=False)
ref, _, _ = B.run("REF_pool500k_s42", tr500)
R.base = ref
R.add(ref, "0_REF", True)

RANGES = {"bmi": (13.0, 50.0), "sleep_hrs": (3.0, 13.0)}
Xk = B.k[FEATURES].copy(); Xp = B.pre[FEATURES].copy(); Xq = B.post[FEATURES].copy()
for c, (lo, hi) in RANGES.items():
    for D in (Xk, Xp, Xq):
        v = np.array(D[c].to_numpy(float), copy=True)
        v[(v < lo) | (v > hi)] = np.nan
        D[c] = v
r, _, _ = B.run("O_implausible_to_nan", tr500, X=Xk, Xho=Xk.iloc[B.ho], Xpre=Xp, Xpost=Xq,
                note="타당범위 밖은 결측화 후 median 대체")
R.add(r, "O_outlier", True)

m = np.ones(len(tr500), bool)
for c, (lo, hi) in RANGES.items():
    v = B.k[c].to_numpy(float)[tr500]
    m &= ~((v < lo) | (v > hi))
r, _, _ = B.run("O_drop_implausible_rows", tr500[m], note=f"타당범위 밖 학습행 제외 ({(~m).sum():,}행)")
R.add(r, "O_outlier", True)

# 밀도비 진단
R.log("\n########## G) 밀도비(공변량 이동) 가중 진단 ##########")
sub = rng0.choice(tr500, 60000, replace=False)
Xd = pd.concat([B.k[FEATURES].iloc[sub], B.pre[FEATURES]], axis=0)
yd = np.r_[np.zeros(len(sub)), np.ones(len(B.pre))]
dp = Pipeline([("pre", make_pre()),
               ("clf", lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                          num_leaves=31, random_state=SEED, n_jobs=-1, verbose=-1))])
oof = cross_val_predict(dp, Xd, yd, cv=5, method="predict_proba")[:, 1]
dauc = wauc(yd, oof, np.ones(len(yd)))
pk = oof[:len(sub)]
w = pk / np.maximum(1 - pk, 1e-12)
ess = (w.sum() ** 2) / max((w ** 2).sum(), 1e-30)
diag = dict(domain_auc_oof=dauc, kchs_p_max=float(pk.max()), kchs_p_median=float(np.median(pk)),
            kchs_p_q99=float(np.quantile(pk, .99)), frac_kchs_p_lt_1e2=float((pk < 1e-2).mean()),
            ess=float(ess), n_kchs=len(sub), ess_ratio=float(ess / len(sub)))
pd.DataFrame([diag]).to_csv(os.path.join(RES, "path3_02c_densratio_diag.csv"), index=False, encoding="utf-8-sig")
for kk, vv in diag.items():
    R.log(f"  {kk:20s} = {vv}")
R.log(f"\n완료: {R.out}")
