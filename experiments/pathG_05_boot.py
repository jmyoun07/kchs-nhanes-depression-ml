# -*- coding: utf-8 -*-
"""pathG_05: combo + 부트스트랩 CI (독립 실행, 경량)."""
import os, sys, time, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathG_common import *
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

T0 = time.time()
k, pre, post = load()
tr, ho = base_split(k)
y = k["dep"].astype(int).values
Wk = k["W"].to_numpy(float)
ypre, wpre = pre["dep"].astype(int).values, pre["W"].to_numpy(float)
ypost, wpost = post["dep"].astype(int).values, post["W"].to_numpy(float)
Ktr = k.iloc[tr][FEATURES].reset_index(drop=True)
Kho = k.iloc[ho][FEATURES].reset_index(drop=True)
Pre = pre[FEATURES].reset_index(drop=True)
Post = post[FEATURES].reset_index(drop=True)
ytr, yho, who = y[tr], y[ho], Wk[ho]

CT = make_ct()
Xtr = CT.fit_transform(Ktr).astype(np.float64)
Xho = CT.transform(Kho).astype(np.float64)
Xpre = CT.transform(Pre).astype(np.float64)
Xpost = CT.transform(Post).astype(np.float64)

PAR = dict(LGB_PARAMS); PAR["n_jobs"] = 4
m0 = lgb.LGBMClassifier(**PAR)
m0.fit(Xtr, ytr)
pp = m0.predict_proba(Xpre)[:, 1]
po = m0.predict_proba(Xpost)[:, 1]
print("base pre=%.5f post=%.5f" % (wauc(ypre, pp, wpre), wauc(ypost, po, wpost)), flush=True)


def boot_ci(df, p, n=3000, seed=SEED):
    yv = df["dep"].astype(int).values
    wv = df["W"].to_numpy(float)
    st = df["strata"].values; ps = df["psu"].values
    groups = {}
    for s in np.unique(st):
        sm = st == s
        groups[s] = [np.where(sm & (ps == u))[0] for u in np.unique(ps[sm])]
    keys = list(groups.keys())
    rs = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        chunks = []
        for s in keys:
            us = groups[s]
            if len(us) < 2:
                chunks.extend(us); continue
            pick = rs.randint(0, len(us), len(us))
            for j in pick:
                chunks.append(us[j])
        idx = np.concatenate(chunks)
        if len(np.unique(yv[idx])) < 2:
            continue
        out.append(roc_auc_score(yv[idx], p[idx], sample_weight=wv[idx]))
    out = np.array(out)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)),
            float(out.mean()), float(out.std()), len(out), float((out >= 0.80).mean()))


cis = []
for tag, df, p in [("pre", pre, pp), ("post", post, po)]:
    lo, hi, mn, sd, nb, p80 = boot_ci(df, p)
    row = {"set": tag, "wauc": round(wauc(df["dep"], p, df["W"]), 5),
           "boot_mean": round(mn, 5), "boot_sd": round(sd, 5),
           "ci95_lo": round(lo, 5), "ci95_hi": round(hi, 5),
           "p_ge_0.80": round(p80, 4), "n_boot": nb,
           "note": "기준선 17변수 전이모델, PSU/strata 복원추출 3000회"}
    cis.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)
pd.DataFrame(cis).to_csv(os.path.join(RES, "pathG_04_bootci.csv"),
                         index=False, encoding="utf-8-sig")
print("TOTAL %.1f min" % ((time.time() - T0) / 60))
