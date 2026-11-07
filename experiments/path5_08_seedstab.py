# -*- coding: utf-8 -*-
"""
path5_08_seedstab.py — RankNet 혼합 후보의 외부 개선이 시드노이즈인지 판정(경량, 빠른 경로).

내부 CV(path5_07): ranknet_K5_ce0.5 = 0.830599 ≈ binary 0.830614  -> 내부로는 개선 아님.
그렇다면 단일홀드아웃에서 보였던 외부 pre/post 개선이 진짜인가?
  * logloss(대조)와 ranknet_ce0.5 를 각각 5시드 학습 -> 외부 pre/post 분포로 판정.
  * 두 방법 모두 커스텀 경로(공정 비교), 빠른 train_native 사용.
"""
import os, sys, time, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path5_common import *      # noqa
import path5_obj as O
import lightgbm as lgb

t0 = time.time()
D = Data()
BASE = dict(internal=0.830969, ext_pre=0.774013, ext_post=0.765262)
prior = float(D.ytr.mean()); F0 = np.log(prior / (1 - prior))
INIT = np.full(len(D.ytr), F0)
print(f"data ready {time.time()-t0:.1f}s", flush=True)


def fit(kind, seed, lam_ce=0.5, K=5):
    if kind == "logloss":
        fobj = O.make_logloss()
    else:
        fobj = O.make_ranknet(D.ytr, n_neg_per_pos=K, lam_ce=lam_ce, seed=seed)
    p = dict(NATIVE); p.update(objective=fobj, boost_from_average=False, seed=seed)
    ds = lgb.Dataset(D.Xtr, label=D.ytr, init_score=INIT, free_raw_data=False)
    b = lgb.train(p, ds, num_boost_round=NROUND)
    return (b.predict(D.Xho, raw_score=True), b.predict(D.Xpre, raw_score=True),
            b.predict(D.Xpost, raw_score=True))


rows = []
def ev(name, rh, rp, ro, note):
    r = D.evaluate_raw(rh, rp, ro)
    row = dict(name=name, internal=round(r["internal"], 6), ext_pre=round(r["ext_pre"], 6),
               ext_post=round(r["ext_post"], 6),
               d_pre=round(r["ext_pre"] - BASE["ext_pre"], 6),
               d_post=round(r["ext_post"] - BASE["ext_post"], 6),
               protocol_compliant=True, note=note)
    rows.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
    return r

SEEDS = [42, 7, 123, 2024, 99]
store = {"logloss": ([], [], []), "ranknet": ([], [], [])}
for sd in SEEDS:
    for kind in ["logloss", "ranknet"]:
        t = time.time()
        rh, rp, ro = fit(kind, sd)
        store[kind][0].append(rh); store[kind][1].append(rp); store[kind][2].append(ro)
        ev(f"{kind}_seed{sd}", rh, rp, ro, f"{kind} seed={sd} ({time.time()-t:.0f}s)")

for kind in ["logloss", "ranknet"]:
    h, p_, o = store[kind]
    ev(f"{kind}_seedens_raw", np.mean(h, 0), np.mean(p_, 0), np.mean(o, 0),
       f"{kind} 5시드 raw-score 평균 앙상블")

pd.DataFrame(rows).to_csv(os.path.join(RES, "path5_08_seedstab.csv"),
                          index=False, encoding="utf-8-sig")

def stat(name, arr):
    a = np.array(arr)
    return f"{name}: mean={a.mean():.5f} sd={a.std():.5f} [{a.min():.5f},{a.max():.5f}]"
print("\n=== SEED-STABILITY (5 seeds, single-model ext) ===", flush=True)
print(f"baseline_ref: pre={BASE['ext_pre']:.5f} post={BASE['ext_post']:.5f}", flush=True)
for kind in ["logloss", "ranknet"]:
    pre = [r["ext_pre"] for r in rows if r["name"].startswith(kind + "_seed") and "ens" not in r["name"]]
    post = [r["ext_post"] for r in rows if r["name"].startswith(kind + "_seed") and "ens" not in r["name"]]
    print(stat(f"{kind}_pre ", pre), flush=True)
    print(stat(f"{kind}_post", post), flush=True)
print(f"TOTAL {time.time()-t0:.1f}s")
