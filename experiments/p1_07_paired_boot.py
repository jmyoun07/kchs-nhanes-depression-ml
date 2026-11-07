# -*- coding: utf-8 -*-
"""경로1-07: 주 선택 규칙 C09 vs 기준선 — 짝지은 설계기반(층내 PSU 재표집) 부트스트랩.
전이-강건 선택의 외부 개선이 표본잡음 대비 유의한지 냉정히 판정."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_common import *
from p1_01_grid import G
import numpy as np, pandas as pd

SPEC = {n: s for n, s, _ in G}
k, pre, post = load()
y = k["dep"].astype(int).to_numpy()
itr, iho = base_split(len(k), y)

def fit(spec):
    p = dict(LGB_BASE); p.update({kk: vv for kk, vv in spec.get("params", {}).items() if kk != "n_jobs"})
    pipe = Pipeline([("pre", make_pre()), ("clf", lgb.LGBMClassifier(**p))])
    pipe.fit(k.iloc[itr][FEATURES], y[itr])
    return {"pre": pipe.predict_proba(pre[FEATURES])[:, 1], "post": pipe.predict_proba(post[FEATURES])[:, 1]}

Pb = fit({"params": {}})           # baseline
Pc = fit(SPEC["C09"])              # 주 선택 규칙 R5/R2/R3/R6 = C09
D = {"pre": pre, "post": post}

def paired_boot(d, pb, pc, nb=1000, seed=0):
    rng = np.random.default_rng(seed)
    st = d["strata"].astype(str).values; ps = d["psu"].astype(str).values
    yv = d["dep"].astype(int).values; wv = d["W"].values
    key = np.char.add(np.char.add(st, "_"), ps)
    uk, inv = np.unique(key, return_inverse=True)
    idx_by = [np.flatnonzero(inv == i) for i in range(len(uk))]
    clu_st = np.array([st[c[0]] for c in idx_by])
    ab = []; bb = []; diff = []
    for _ in range(nb):
        take = []
        for s in np.unique(clu_st):
            cs = np.flatnonzero(clu_st == s)
            for c in rng.choice(cs, size=len(cs), replace=True):
                take.append(idx_by[c])
        ii = np.concatenate(take)
        if len(np.unique(yv[ii])) < 2: continue
        a = roc_auc_score(yv[ii], pb[ii], sample_weight=wv[ii])
        b = roc_auc_score(yv[ii], pc[ii], sample_weight=wv[ii])
        ab.append(a); bb.append(b); diff.append(b - a)
    return np.array(ab), np.array(bb), np.array(diff)

rows = []
for nm in ("pre", "post"):
    a0 = wauc(D[nm]["dep"], Pb[nm], D[nm]["W"])
    c0 = wauc(D[nm]["dep"], Pc[nm], D[nm]["W"])
    ba, bc, bd = paired_boot(D[nm], Pb[nm], Pc[nm])
    lo, hi = np.percentile(bd, [2.5, 97.5])
    # p-value 양측 (부호 뒤집힘 비율 기반, 짝지은 차이)
    p_two = 2 * min((bd <= 0).mean(), (bd >= 0).mean())
    rows.append(dict(dataset=nm, baseline=round(a0,6), C09=round(c0,6),
        delta=round(c0-a0,6), delta_lo=round(float(lo),6), delta_hi=round(float(hi),6),
        delta_se=round(float(bd.std()),6), p_two_sided=round(float(p_two),4),
        n_to_reach_080_from_C09=round(0.80-c0,6),
        note="층내 PSU 재표집 짝지은 부트스트랩 %d회. §2.10 준수(NHANES 선택 미사용)"%len(bd)))
    print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

pd.DataFrame(rows).to_csv(os.path.join(RES, "p1_07_paired_boot.csv"), index=False, encoding="utf-8-sig")
print("saved", os.path.join(RES, "p1_07_paired_boot.csv"))
