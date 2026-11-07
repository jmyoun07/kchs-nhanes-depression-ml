# -*- coding: utf-8 -*-
"""경로E-0: 기준선 재현. 두 가지 split 규약을 모두 시도해 0.8336/0.7681/0.7682 재현 확인."""
import os, sys, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathE_common import *
from sklearn.model_selection import train_test_split

k, pre, post = load()
y = k["dep"].astype(int).values
print("loaded", k.shape, pre.shape, post.shape)

rows = []

# --- 규약 A: train=500000 / test=100000 (과제 명세) ---
itrA, ihoA = split_idx(k, 500000, 100000, SEED)
t0 = time.time()
pipeA = Pipeline([("pre", make_pre()), ("clf", lgb.LGBMClassifier(**LGB_PARAMS))])
pipeA.fit(k.iloc[itrA][FEATURES], y[itrA])
iA, pA, oA = evaluate(pipeA, k, pre, post, ihoA)[:3]
print(f"A(500k/100k): internal={iA:.4f} pre={pA:.4f} post={oA:.4f}  [{time.time()-t0:.0f}s]")
rows.append(dict(spec="A_500k_100k", internal=iA, ext_pre=pA, ext_post=oA))

# --- 규약 B: 80/20 split, income_q 그대로 (script17 유사) ---
itrB, ihoB = train_test_split(np.arange(len(k)), test_size=0.2, stratify=y, random_state=SEED)
t0 = time.time()
pipeB = Pipeline([("pre", make_pre()), ("clf", lgb.LGBMClassifier(**LGB_PARAMS))])
pipeB.fit(k.iloc[itrB][FEATURES], y[itrB])
iB, pB, oB = evaluate(pipeB, k, pre, post, ihoB)[:3]
print(f"B(80/20 asis): internal={iB:.4f} pre={pB:.4f} post={oB:.4f}  [{time.time()-t0:.0f}s]")
rows.append(dict(spec="B_8020_incomeq_asis", internal=iB, ext_pre=pB, ext_post=oB))

# --- 규약 C: 80/20 split + income_q train-only 재계산 (script17 정확 재현) ---
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

is_tr = np.zeros(len(k), bool); is_tr[itrB] = True
kk = k.copy(); kk["income_q"] = _income_q_trainonly(kk, is_tr)
t0 = time.time()
pipeC = Pipeline([("pre", make_pre()), ("clf", lgb.LGBMClassifier(**LGB_PARAMS))])
pipeC.fit(kk.iloc[itrB][FEATURES], y[itrB])
iC, pC, oC = evaluate(pipeC, kk, pre, post, ihoB)[:3]
print(f"C(8020 trainonly-incq): internal={iC:.4f} pre={pC:.4f} post={oC:.4f}  [{time.time()-t0:.0f}s]")
rows.append(dict(spec="C_8020_incomeq_trainonly", internal=iC, ext_pre=pC, ext_post=oC))

df = pd.DataFrame(rows)
df["target_internal"] = 0.8336; df["target_pre"] = 0.7681; df["target_post"] = 0.7682
os.makedirs(RES, exist_ok=True)
df.to_csv(os.path.join(RES, "pathE_baseline_repro.csv"), index=False, encoding="utf-8-sig")
print(df.to_string())
