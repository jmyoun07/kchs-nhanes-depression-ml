# -*- coding: utf-8 -*-
"""경로E-0b: script17 A_core3 배치(oral_health/employed/food_insecure = passthrough 'cand')를
정확 재현해 0.8336/0.7681/0.7682와의 잔차 원인 확인."""
import os, sys, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathE_common import *
from sklearn.model_selection import train_test_split

k, pre, post = load()
y = k["dep"].astype(int).values
itr, iho = train_test_split(np.arange(len(k)), test_size=0.2, stratify=y, random_state=SEED)
is_tr = np.zeros(len(k), bool); is_tr[itr] = True

def _income_q_trainonly(df, is_tr):
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

kk = k.copy(); kk["income_q"] = _income_q_trainonly(kk, is_tr)

NUM14 = ["age","sleep_hrs","bmi","income_q","srh"]
BIN14 = ["partnered","living_alone","current_drinker","active","htn","dm"]
CAT14 = ["sex","edu","smoking"]
EXTRA = ["employed","oral_health","food_insecure"]
feats = NUM14+BIN14+CAT14+EXTRA

npipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
cpipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                  ("oh", OneHotEncoder(handle_unknown="ignore", drop="first"))])
ct = ColumnTransformer([("num", npipe, NUM14), ("bin", "passthrough", BIN14),
                        ("cat", cpipe, CAT14), ("cand", "passthrough", EXTRA)])
pipe = Pipeline([("pre", ct), ("clf", lgb.LGBMClassifier(**LGB_PARAMS))])
t0=time.time(); pipe.fit(kk.iloc[itr][feats], y[itr])
W = k["W"].to_numpy(float)
pi = pipe.predict_proba(kk.iloc[iho][feats])[:,1]
pp = pipe.predict_proba(pre[feats])[:,1]
po = pipe.predict_proba(post[feats])[:,1]
print(f"script17-exact: internal={wauc(y[iho],pi,W[iho]):.6f} pre={wauc(pre.dep.astype(int),pp,pre.W):.6f} post={wauc(post.dep.astype(int),po,post.W):.6f} [{time.time()-t0:.0f}s]")
print("target        : internal=0.833600 pre=0.768146 post=0.768205")
