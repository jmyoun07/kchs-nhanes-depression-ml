# -*- coding: utf-8 -*-
"""pathD_00b_variants.py — 기준선 재현 후보 변형 탐색 (열 순서/income_q 처리)."""
import sys, os, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathD_common import *
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import lightgbm as lgb

k, pre, post = load()
itr, iho = split(k)
y = k["dep"].astype(int).values
TARGET = dict(internal=0.8336, ext_pre=0.7681, ext_post=0.7682)

# income_q train-only 재계산 버전
def income_q_trainonly(df, is_tr):
    iq = np.full(len(df), np.nan); yr = df["year"].values; inc = df["income_raw"].to_numpy(float)
    for yv in np.unique(yr):
        m = yr == yv; v = inc[m & is_tr]; v = v[~np.isnan(v)]
        if len(v) < 50: continue
        e = np.unique(np.quantile(v, [0, .2, .4, .6, .8, 1.0]))
        if len(e) < 3: continue
        ii = np.where(m)[0]
        b = np.clip(np.digitize(inc[ii], e[1:-1]), 0, len(e) - 2).astype(float)
        b[np.isnan(inc[ii])] = np.nan; iq[ii] = b
    return iq

is_tr = np.zeros(len(k), bool); is_tr[itr] = True
k2 = k.copy(); k2["income_q"] = income_q_trainonly(k2, is_tr)

NUM14 = ["age", "sleep_hrs", "bmi", "income_q", "srh"]
BIN14 = ["partnered", "living_alone", "current_drinker", "active", "htn", "dm"]

def ct_extra(extra):
    n = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    c = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                  ("oh", OneHotEncoder(handle_unknown="ignore", drop="first"))])
    b = [("num", n, NUM14), ("bin", "passthrough", BIN14), ("cat", c, CAT)]
    if extra: b.append(("cand", "passthrough", extra))
    return ColumnTransformer(b)

def run(tag, ct, feats, kdf):
    t0 = time.time()
    pipe = Pipeline([("pre", ct), ("clf", lgb.LGBMClassifier(**BASE_LGB))])
    pipe.fit(kdf[feats].iloc[itr], y[itr])
    W = kdf["W"].to_numpy(float)
    r = dict(internal=wauc(y[iho], pipe.predict_proba(kdf[feats].iloc[iho])[:, 1], W[iho]))
    for nm, d in [("ext_pre", pre), ("ext_post", post)]:
        r[nm] = wauc(d["dep"].astype(int).values, pipe.predict_proba(d[feats])[:, 1],
                     d["W"].to_numpy(float))
    dd = {a: round(r[a] - b, 4) for a, b in TARGET.items()}
    print(f"{tag:34s} int={r['internal']:.4f} pre={r['ext_pre']:.4f} post={r['ext_post']:.4f} "
          f"Δ={dd} [{time.time()-t0:.0f}s]")
    return dict(name=tag, **r)

rows = []
EX3 = ["employed", "oral_health", "food_insecure"]
rows.append(run("V1_14+extra3_pq_incq", ct_extra(EX3), NUM14 + BIN14 + CAT + EX3, k))
rows.append(run("V2_14+extra3_trainonly_incq", ct_extra(EX3), NUM14 + BIN14 + CAT + EX3, k2))
rows.append(run("V3_17grouped_trainonly_incq", make_pre(), FEATURES, k2))
# oral_health를 num, employed/food_insecure를 bin (기본) 이미 V0. 다른 순서:
NUMb = ["age", "sleep_hrs", "bmi", "income_q", "srh", "oral_health"]
BINb = ["partnered", "living_alone", "current_drinker", "active", "htn", "dm", "employed", "food_insecure"]
rows.append(run("V4_17grouped_pq_incq", make_pre(NUMb, BINb, CAT), NUMb + BINb + CAT, k))
pd.DataFrame(rows).to_csv(os.path.join(RES, "pathD_baseline_variants.csv"),
                          index=False, encoding="utf-8-sig")
