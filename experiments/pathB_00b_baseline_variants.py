# -*- coding: utf-8 -*-
"""pathB_00b — 기준선 불일치 원인 탐색: income_q 산출방식/split 방식 변형 대조."""
import os, sys, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathB_common import *
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import lightgbm as lgb

k, pre, post = load()
y = k["dep"].astype(int).values
W = k["W"].to_numpy(float)


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


def run(kk, itr, iho, tag):
    X = kk[FEATURES]
    pipe = Pipeline([("pre", make_pre()),
                     ("clf", lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31,
                                                subsample=0.8, colsample_bytree=0.8,
                                                random_state=SEED, n_jobs=-1, verbose=-1))])
    pipe.fit(X.iloc[itr], y[itr])
    r = {"model": tag, "internal": wauc(y[iho], pipe.predict_proba(X.iloc[iho])[:, 1], W[iho])}
    for nm, d in [("pre", pre), ("post", post)]:
        r[nm] = wauc(d["dep"].astype(int).values, pipe.predict_proba(d[FEATURES])[:, 1], d["W"].to_numpy(float))
    print(f"{tag:38s} internal={r['internal']:.4f} pre={r['pre']:.4f} post={r['post']:.4f}")
    return r

rows = []
# V1: 500k/100k split, parquet income_q (= 기준선 명세 그대로)
itr, iho = train_test_split(np.arange(len(k)), train_size=500000, test_size=100000, stratify=y, random_state=SEED)
rows.append(run(k, itr, iho, "V1 500k/100k, income_q as-is"))

# V2: 500k/100k split, income_q train-only 재산출
is_tr = np.zeros(len(k), bool); is_tr[itr] = True
kk = k.copy(); kk["income_q"] = income_q_trainonly(kk, is_tr)
rows.append(run(kk, itr, iho, "V2 500k/100k, income_q train-only"))

# V3: 80/20 split, income_q as-is
itr3, iho3 = train_test_split(np.arange(len(k)), test_size=0.2, stratify=y, random_state=SEED)
rows.append(run(k, itr3, iho3, "V3 80/20, income_q as-is"))

# V4: 80/20 split, income_q train-only
is_tr3 = np.zeros(len(k), bool); is_tr3[itr3] = True
kk3 = k.copy(); kk3["income_q"] = income_q_trainonly(kk3, is_tr3)
rows.append(run(kk3, itr3, iho3, "V4 80/20, income_q train-only"))

# V5: 500k/100k, oral_health 를 CAT 취급? -> NUM 순서 변경(맨 끝 대신 srh 앞)
import pathB_common as C
C.NUM = ["age", "sleep_hrs", "bmi", "income_q", "srh"]  # oral_health를 BIN 뒤 passthrough로
C.BIN = ["partnered", "living_alone", "current_drinker", "active", "htn", "dm", "employed", "food_insecure", "oral_health"]
globals()['make_pre'] = C.make_pre
rows.append(run(k, itr, iho, "V5 oral_health passthrough"))

pd.DataFrame(rows).to_csv(os.path.join(RES, "pathB_baseline_variants.csv"), index=False, encoding="utf-8-sig")
print("\ntarget: internal=0.8336 pre=0.7681 post=0.7682")
