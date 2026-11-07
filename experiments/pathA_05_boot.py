# -*- coding: utf-8 -*-
"""경로A 단계5: 최종 후보들의 외부 AUROC 부트스트랩 CI + 기준선 대비 차이의 불확실성.
NHANES n(pre=7888, post=5210)에서 AUROC 표준오차가 크므로 '개선'이 잡음인지 판정한다."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathA_common import *
import lightgbm as lgb

OUT = os.path.join(RES, "pathA_bootstrap.csv")
B = 500
rng = np.random.default_rng(SEED)


def fit_pred(dk, dp, dq, num=NUM, binf=BIN, cat=CAT, model=None):
    feats = list(num) + list(binf) + list(cat)
    clf = model if model is not None else lgb.LGBMClassifier(**LGBM)
    pipe = Pipeline([("pre", make_prep(num, binf, cat,
                                       not isinstance(clf, lgb.LGBMClassifier))), ("clf", clf)])
    pipe.fit(dk[feats].iloc[ITR], yK[ITR])
    return pipe.predict_proba(dp[feats])[:, 1], pipe.predict_proba(dq[feats])[:, 1]


CAND = {}
CAND["baseline"] = fit_pred(K, PRE, POST)

Kr, Pr, Qr = rank_transform(NUM, weighted=True, normal=True)
CAND["rankN_all6"] = fit_pred(Kr, Pr, Qr)

Ks, Ps, Qs = rank_transform(["srh", "oral_health"], weighted=True, normal=False)
CAND["rank_srh_oh"] = fit_pred(Ks, Ps, Qs)

Kc, Pc, Qc = K.copy(), PRE.copy(), POST.copy()
for d in (Kc, Pc, Qc):
    d["srh_c"] = d["srh"].astype("Int64").astype(str)
    d["oh_c"] = d["oral_health"].astype("Int64").astype(str)
num_no = [c for c in NUM if c not in ("srh", "oral_health")]
CAND["srh_onehot"] = fit_pred(Kc, Pc, Qc, num_no, BIN, CAT + ["srh_c", "oh_c"])

ADD = dict(LGBM); ADD.update(max_depth=1, num_leaves=2, n_estimators=2000, learning_rate=0.05)
CAND["additive"] = fit_pred(K, PRE, POST, model=lgb.LGBMClassifier(**ADD))

# 평균 앙상블(표현 다양성)
CAND["ens_base_rank_add"] = tuple(
    np.mean([CAND["baseline"][i], CAND["rankN_all6"][i], CAND["additive"][i]], axis=0)
    for i in (0, 1))

rows = []
for setname, D, j in (("pre", PRE, 0), ("post", POST, 1)):
    y = D["dep"].astype(int).to_numpy(); w = D["W"].to_numpy(float); n = len(y)
    base = CAND["baseline"][j]
    idxs = [rng.integers(0, n, n) for _ in range(B)]
    for nm, pr in CAND.items():
        p = pr[j]
        pt = wauc(y, p, w)
        ds, aus = [], []
        for ii in idxs:
            if y[ii].sum() == 0 or y[ii].sum() == len(ii):
                continue
            a = wauc(y[ii], p[ii], w[ii]); b0 = wauc(y[ii], base[ii], w[ii])
            aus.append(a); ds.append(a - b0)
        rows.append(dict(set=setname, cand=nm, auc=pt,
                         lo=np.percentile(aus, 2.5), hi=np.percentile(aus, 97.5),
                         d_vs_base=np.mean(ds),
                         d_lo=np.percentile(ds, 2.5), d_hi=np.percentile(ds, 97.5)))
        print(rows[-1], flush=True)

pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
print("saved:", OUT)
