# -*- coding: utf-8 -*-
"""경로1-05: 한계치(ceiling) 확정 진단.

(1) 기준선 외부 가중 AUROC의 설계기반(층내 PSU 재표집) 부트스트랩 신뢰구간
(2) ★비준수 오라클: NHANES 자체 라벨로 적합한 모델의 교차검증 가중 AUROC
    = 이 17변수가 NHANES 안에서 담을 수 있는 정보 상한
    (§2.10 위반이므로 진단 전용, protocol_compliant=False)
(3) (1)과 (2)의 비교로 '전이 손실'과 '변수 정보한계'를 분리
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_common import *
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

NB = 400
rows = []

k, pre, post = load()
y = k["dep"].astype(int).to_numpy()
itr, iho = base_split(len(k), y)

# ---- 기준선 모델 (명세 그대로) ----
pipe = Pipeline([("pre", make_pre()), ("clf", lgb.LGBMClassifier(**LGB_BASE))])
pipe.fit(k.iloc[itr][FEATURES], y[itr])
P = {"pre": pipe.predict_proba(pre[FEATURES])[:, 1],
     "post": pipe.predict_proba(post[FEATURES])[:, 1]}
D = {"pre": pre, "post": post}


def design_boot(d, p, nb=NB, seed=0):
    """층 내 PSU 복원추출 부트스트랩."""
    rng = np.random.default_rng(seed)
    st = d["strata"].astype(str).values
    ps = d["psu"].astype(str).values
    yv = d["dep"].astype(int).values
    wv = d["W"].values
    key = np.char.add(np.char.add(st, "_"), ps)
    uk, inv = np.unique(key, return_inverse=True)
    idx_by_clu = [np.flatnonzero(inv == i) for i in range(len(uk))]
    clu_str = np.array([st[c[0]] for c in idx_by_clu])
    out = []
    for b in range(nb):
        take = []
        for s in np.unique(clu_str):
            cs = np.flatnonzero(clu_str == s)
            pick = rng.choice(cs, size=len(cs), replace=True)
            for c in pick:
                take.append(idx_by_clu[c])
        ii = np.concatenate(take)
        if len(np.unique(yv[ii])) < 2:
            continue
        out.append(roc_auc_score(yv[ii], p[ii], sample_weight=wv[ii]))
    return np.array(out)


for nm in ("pre", "post"):
    d = D[nm]
    a = wauc(d["dep"], P[nm], d["W"])
    bs = design_boot(d, P[nm])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    rec(rows, name=f"baseline_ext_{nm}_bootCI", auc=round(float(a), 6),
        lo=round(float(lo), 6), hi=round(float(hi), 6), se=round(float(bs.std()), 6),
        protocol_compliant=True,
        note=f"층내 PSU 재표집 부트스트랩 {len(bs)}회. 0.80 도달 필요 증가분="
             f"{0.80-a:+.4f} (= {(0.80-a)/bs.std():.1f} SE)")

# ---- 오라클: NHANES 자체 적합 (§2.10 위반, 진단 전용) ----
def oracle(d, name, model="lgb"):
    X = d[FEATURES].copy()
    yv = d["dep"].astype(int).values
    wv = d["W"].values
    oof = np.zeros(len(d))
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for tr, te in skf.split(X, yv):
        if model == "lgb":
            m = Pipeline([("pre", make_pre()),
                          ("clf", lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                                     num_leaves=7, min_child_samples=100,
                                                     reg_lambda=10.0, random_state=SEED,
                                                     n_jobs=4, verbose=-1))])
        else:
            m = Pipeline([("pre", make_pre()),
                          ("imp2", SimpleImputer(strategy="most_frequent")),
                          ("clf", LogisticRegression(max_iter=2000, C=1.0))])
        m.fit(X.iloc[tr], yv[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    return wauc(yv, oof, wv), oof


for nm in ("pre", "post"):
    for mdl in ("lgb", "logit"):
        a, _ = oracle(D[nm], nm, mdl)
        rec(rows, name=f"ORACLE_{nm}_selffit_{mdl}_5foldCV", auc=round(float(a), 6),
            lo="", hi="", se="", protocol_compliant=False,
            note="★비준수(§2.10): NHANES 자체 라벨로 적합. 17변수 정보 상한 진단")

# pooled
pool = pd.concat([pre.assign(_src=0), post.assign(_src=1)], ignore_index=True)
for mdl in ("lgb", "logit"):
    a, oof = oracle(pool, "pooled", mdl)
    n0 = len(pre)
    rec(rows, name=f"ORACLE_pooled_selffit_{mdl}_5foldCV", auc=round(float(a), 6),
        lo=round(float(wauc(pre["dep"], oof[:n0], pre["W"])), 6),
        hi=round(float(wauc(post["dep"], oof[n0:], post["W"])), 6), se="",
        protocol_compliant=False,
        note="★비준수(§2.10): pre+post 통합 자체적합. lo칸=pre, hi칸=post 부분 AUROC")

dump(rows, "p1_05_ceiling.csv")
