# -*- coding: utf-8 -*-
"""
path5_02_rank.py — 경로5 (1) AUROC 직접 최적화 근사(pairwise ranking).

  a) LightGBM native objective='lambdarank' / 'rank_xendcg' (이진 relevance)
     - 그룹: (i) 무작위 블록, (ii) 층화 블록(연령대×성×연도)
  b) 커스텀 RankNet(쌍 로지스틱) 목적함수 — AUROC = P(f+ > f-) 의 매끄러운 상계

전부 KCHS 학습데이터만 사용. NHANES 는 기록용 최종 평가.
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path5_common import *      # noqa
import path5_obj as O
import lightgbm as lgb

t0 = time.time()
D = Data()
prior = float(D.ytr.mean())
F0 = np.log(prior / (1 - prior))
INIT = np.full(len(D.ytr), F0)
n = len(D.ytr)
print(f"data ready {time.time()-t0:.1f}s n={n} pos={D.ytr.sum()}", flush=True)


def eval_booster(b, name, note, compliant=True, base=None, sec=None):
    r = D.evaluate_raw(predict_raw(b, D.Xho), predict_raw(b, D.Xpre),
                       predict_raw(b, D.Xpost))
    return rec(name, r["internal"], r["ext_pre"], r["ext_post"], compliant, note,
               base=base, extra=dict(sec=sec))


def run_native(name, obj, group, params_extra=None, note="", base=None, order=None):
    t = time.time()
    pe = dict(objective=obj, metric="None")
    if params_extra:
        pe.update(params_extra)
    X = D.Xtr if order is None else D.Xtr[order]
    yv = D.ytr if order is None else D.ytr[order]
    try:
        p = dict(NATIVE); p.update(pe)
        ds = lgb.Dataset(X, label=yv, group=group, free_raw_data=False)
        b = lgb.train(p, ds, num_boost_round=NROUND)
        return eval_booster(b, name, note, True, base, round(time.time() - t, 1))
    except Exception as e:
        print(f"!! {name} FAILED: {type(e).__name__}: {e}", flush=True)
        return rec(name, np.nan, np.nan, np.nan, True, f"FAILED: {e}", base=base)


def run_custom(name, fobj, note="", base=None, nround=NROUND):
    t = time.time()
    try:
        b = train_native(D, None, fobj=fobj,
                         params_extra=dict(boost_from_average=False),
                         nround=nround, init_score=INIT)
        return eval_booster(b, name, note, True, base, round(time.time() - t, 1))
    except Exception as e:
        print(f"!! {name} FAILED: {type(e).__name__}: {e}", flush=True)
        return rec(name, np.nan, np.nan, np.nan, True, f"FAILED: {e}", base=base)


def blocks(n_, g):
    """길이 n_ 을 크기 g 블록으로 (마지막 블록은 나머지)."""
    q, r = divmod(n_, g)
    grp = [g] * q + ([r] if r else [])
    return np.array(grp, dtype=np.int32)


BASE = rec("00_native_binary", 0.830969, 0.774013, 0.765262, True,
           "기준선(path5_00 재현값)")

# ---------------------------------------------------------------- a) 무작위 블록 그룹
# itr 자체가 train_test_split 로 셔플된 순서이므로 연속 블록 = 무작위 그룹
for g in [20, 50, 100, 200, 500, 1000]:
    grp = blocks(n, g)
    run_native(f"50_lambdarank_g{g}", "lambdarank", grp,
               params_extra=dict(lambdarank_truncation_level=min(g, 30),
                                 label_gain=[0, 1]),
               note=f"lambdarank, 무작위 그룹크기={g}, trunc={min(g,30)}", base=BASE)

for g in [50, 200, 1000]:
    grp = blocks(n, g)
    run_native(f"51_lambdarank_g{g}_fulltrunc", "lambdarank", grp,
               params_extra=dict(lambdarank_truncation_level=g, label_gain=[0, 1]),
               note=f"lambdarank, 그룹={g}, truncation=전체(모든 쌍)", base=BASE)

for g in [50, 200, 1000]:
    grp = blocks(n, g)
    run_native(f"52_rankxendcg_g{g}", "rank_xendcg", grp,
               params_extra=dict(label_gain=[0, 1]),
               note=f"rank_xendcg, 무작위 그룹크기={g}", base=BASE)

# ---------------------------------------------------------------- b) 층화 블록 그룹
# 그룹 내부에서만 순위를 매기므로 '그룹 간 수준차'는 손실에서 제거된다
# -> 연령/성/연도 효과를 제거한 조건부 랭킹 학습(전이 관점에서 유의미)
ktr = D.ktr
ageb = np.clip((ktr["age"].to_numpy(float) // 10).astype(int), 1, 8)
sexb = (ktr["sex"].astype(str).values == "M").astype(int)
yrb = ktr["year"].to_numpy(int)
strata_defs = {
    "age10": ageb,
    "age10_sex": ageb * 2 + sexb,
    "age10_sex_year": (ageb * 2 + sexb) * 100 + (yrb - yrb.min()),
}
for sname, sv in strata_defs.items():
    order = np.argsort(sv, kind="stable")
    sv_sorted = sv[order]
    # 층 내부를 다시 최대 gmax 크기 청크로 쪼갠다
    for gmax in [200, 1000]:
        _, cnts = np.unique(sv_sorted, return_counts=True)
        grp = []
        for c in cnts:
            q, r = divmod(int(c), gmax)
            grp += [gmax] * q
            if r:
                grp += [r]
        grp = np.array(grp, dtype=np.int32)
        run_native(f"53_lambdarank_strat[{sname}]_g{gmax}", "lambdarank", grp,
                   params_extra=dict(lambdarank_truncation_level=30, label_gain=[0, 1]),
                   note=f"층화그룹({sname}) 내 랭킹, 청크={gmax}", base=BASE, order=order)

# ---------------------------------------------------------------- c) 커스텀 RankNet
for K in [1, 5, 20]:
    run_custom(f"60_ranknet_K{K}", O.make_ranknet(D.ytr, n_neg_per_pos=K, seed=SEED),
               note=f"RankNet 쌍손실, 양성당 음성 {K}개 샘플", base=BASE)
for K, lam in [(5, 0.25), (5, 0.5), (5, 0.75)]:
    run_custom(f"61_ranknet_K{K}_ce{lam}",
               O.make_ranknet(D.ytr, n_neg_per_pos=K, lam_ce=lam, seed=SEED),
               note=f"RankNet + logloss 혼합(ce비중={lam})", base=BASE)

dump("path5_02_rank.csv")
print(f"TOTAL {time.time()-t0:.1f}s")
