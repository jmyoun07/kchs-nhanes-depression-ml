# -*- coding: utf-8 -*-
"""
pathD2_04_ensemble.py — 기법(4)(5) 확장: 환경(연도/지역) 앙상블 + 최악환경 최적화.

아이디어: 각 환경(연도 또는 지역클러스터)에서 개별 모델을 적합하고 예측을 평균하면
환경 특유(idiosyncratic) 신호가 상쇄되고 환경 불변(invariant) 신호만 남는다는 가정.
평가는 KCHS 내부 CV(LOYO)로만 선택. NHANES는 기록 전용.

추가: 최악환경 가중 앙상블 — 각 구성모델의 '최악 환경 성능'으로 가중.
"""
import os
import sys
import time
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathD2_engine import (data, build_fit, _pred, wauc, RES, SEED, LGB_BASE)

OUT = os.path.join(RES, "pathD2_ensemble.csv")
d = data()
pool = d.itr
ALLIX = np.arange(len(d.cols))
N_PER = 150000


def fit_on(idx, spec=None, seed=SEED):
    spec = dict(spec or {})
    spec.setdefault("family", "lgb")
    spec["_ix"] = ALLIX
    idx = np.sort(idx)
    return build_fit(d, np.asarray(d.Z[idx]), d.y[idx], spec)


def pmat(models, X):
    return np.column_stack([_pred(m, X) for m in models])


def agg(P, how):
    if how == "mean":
        return P.mean(1)
    if how == "rank":
        return np.column_stack([rankdata(P[:, j]) / len(P) for j in range(P.shape[1])]).mean(1)
    if how == "logit":
        q = np.clip(P, 1e-6, 1 - 1e-6)
        return np.log(q / (1 - q)).mean(1)
    raise ValueError(how)


rows = []
t0 = time.time()


def record(name, internal, pre, post, cv=None, note=""):
    r = dict(name=name, internal=internal, ext_pre=pre, ext_post=post,
             loyo_mean=None if cv is None else cv["mean"],
             loyo_min=None if cv is None else cv["min"],
             loyo_std=None if cv is None else cv["std"],
             note=note, seconds=round(time.time() - t0, 1))
    rows.append(r)
    print(f"{name:34s} 내부={internal:.4f} pre={pre:.4f} post={post:.4f}" +
          ("" if cv is None else f" | LOYOmean={cv['mean']:.4f} min={cv['min']:.4f}") +
          f" [{time.time()-t0:.0f}s]")
    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")


# ---------------- 1) 연도별 단일모델 앙상블 ----------------
rs = np.random.RandomState(SEED)
year_models = {}
for yv in d.years:
    ix = pool[d.year[pool] == yv]
    ix = rs.choice(ix, min(N_PER, len(ix)), replace=False)
    year_models[int(yv)] = fit_on(ix)
    print(f"  year model {yv} [{time.time()-t0:.0f}s]")

# --- 내부 CV(LOYO 앙상블): held-out 연도 y는 나머지 8개 모델 평균으로 예측 ---
for how in ("mean", "rank", "logit"):
    sc = []
    for yv in d.years:
        va = np.sort(pool[d.year[pool] == yv])
        ms = [m for k, m in year_models.items() if k != int(yv)]
        p = agg(pmat(ms, np.asarray(d.Z[va])), how)
        sc.append(wauc(d.y[va], p, d.W[va]))
    sc = np.array(sc)
    cv = dict(mean=sc.mean(), min=sc.min(), std=sc.std())
    ms = list(year_models.values())
    record(f"E1_year_ens_{how}",
           wauc(d.y[d.iho], agg(pmat(ms, np.asarray(d.Z[d.iho])), how), d.W[d.iho]),
           wauc(d.ypre, agg(pmat(ms, d.Zpre), how), d.wpre),
           wauc(d.ypost, agg(pmat(ms, d.Zpost), how), d.wpost),
           cv, "연도별 9모델 앙상블")

# ---------------- 2) 최악연도 가중 앙상블 ----------------
# 각 연도모델의 '타 연도에서의 최악 성능'을 KCHS 내부에서 측정 -> 가중치
worst = {}
for k, m in year_models.items():
    s = []
    for yv in d.years:
        if int(yv) == k:
            continue
        va = np.sort(rs.choice(pool[d.year[pool] == yv], 40000, replace=False))
        s.append(wauc(d.y[va], _pred(m, np.asarray(d.Z[va])), d.W[va]))
    worst[k] = min(s)
print("  per-year model worst-other-year AUROC:",
      {k: round(v, 4) for k, v in worst.items()})
wv = np.array([worst[int(y)] for y in d.years])
wt = np.exp((wv - wv.mean()) / (wv.std() + 1e-9))
wt = wt / wt.sum()
ms = list(year_models.values())
for how in ("mean",):
    P = pmat(ms, np.asarray(d.Z[d.iho]))
    record("E2_year_ens_worstweighted",
           wauc(d.y[d.iho], P @ wt, d.W[d.iho]),
           wauc(d.ypre, pmat(ms, d.Zpre) @ wt, d.wpre),
           wauc(d.ypost, pmat(ms, d.Zpost) @ wt, d.wpost),
           None, "최악연도성능 가중")

# ---------------- 3) 지역클러스터 앙상블 ----------------
cl = d.strata_cluster[pool]
sm = []
for c in range(20):
    ix = pool[cl == c]
    if len(ix) < 20000:
        continue
    sm.append(fit_on(rs.choice(ix, min(N_PER, len(ix)), replace=False)))
print(f"  {len(sm)} strata-cluster models [{time.time()-t0:.0f}s]")
for how in ("mean", "rank"):
    record(f"E3_strata_ens_{how}",
           wauc(d.y[d.iho], agg(pmat(sm, np.asarray(d.Z[d.iho])), how), d.W[d.iho]),
           wauc(d.ypre, agg(pmat(sm, d.Zpre), how), d.wpre),
           wauc(d.ypost, agg(pmat(sm, d.Zpost), how), d.wpost),
           None, f"{len(sm)}개 지역클러스터 앙상블")

# ---------------- 4) 시드 앙상블(대조군) ----------------
# 환경 앙상블의 이득이 '환경 다양성' 때문인지 '단순 앙상블 분산감소' 때문인지 분리
seed_models = []
for s in range(9):
    ix = np.random.RandomState(100 + s).choice(pool, N_PER, replace=False)
    sp = dict(params=dict(random_state=100 + s))
    seed_models.append(fit_on(ix, sp))
record("E4_seed_ens_mean(대조군)",
       wauc(d.y[d.iho], agg(pmat(seed_models, np.asarray(d.Z[d.iho])), "mean"), d.W[d.iho]),
       wauc(d.ypre, agg(pmat(seed_models, d.Zpre), "mean"), d.wpre),
       wauc(d.ypost, agg(pmat(seed_models, d.Zpost), "mean"), d.wpost),
       None, "무작위 부분표본 9모델(환경 다양성 없음)")

print("saved:", OUT, f"total {time.time()-t0:.0f}s")
