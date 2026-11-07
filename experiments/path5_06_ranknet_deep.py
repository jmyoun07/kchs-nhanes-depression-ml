# -*- coding: utf-8 -*-
"""
path5_06_ranknet_deep.py — 유망 후보 심화: RankNet(pairwise, AUROC 직접근사) + logloss 혼합.

스크리닝에서 ranknet_K5_ce0.5 가 내부/pre/post 를 동시에 소폭 개선(유일).
핵심 질문: 이 외부 개선(특히 post +0.005)이 '진짜'인가 아니면 음성표집 시드노이즈인가?

설계(§2.10):
  * (K, ce) 선택은 path5_05 내부 CV 결과로 확정(K=5, ce=0.5). 여기서 외부로 재선택하지 않음.
  * 선택설정을 5개 시드로 학습 -> 외부 pre/post 분포(평균±표준편차)로 안정성 판정.
  * 시드앙상블(raw-score 평균/순위평균)로 분산축소 효과 측정.
  * 순수 logloss(커스텀 경로) 대조군도 동일 5시드로 학습해 apple-to-apple 비교.
"""
import os, sys, time, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path5_common import *      # noqa
import path5_obj as O
import lightgbm as lgb

t0 = time.time()
D = Data()
print(f"data ready {time.time()-t0:.1f}s", flush=True)
BASE = dict(internal=0.830969, ext_pre=0.774013, ext_post=0.765262)
BEST_K, BEST_LAM = 5, 0.5     # path5_05 내부 CV로 확정된 설정


def train_ranknet(K, lam_ce, seed):
    pr = float(D.ytr.mean()); f0 = np.log(pr / (1 - pr))
    fobj = O.make_ranknet(D.ytr, n_neg_per_pos=K, lam_ce=lam_ce, seed=seed)
    p = dict(NATIVE); p.update(objective=fobj, boost_from_average=False, seed=seed)
    ds = lgb.Dataset(D.Xtr, label=D.ytr, init_score=np.full(len(D.ytr), f0),
                     free_raw_data=False)
    return lgb.train(p, ds, num_boost_round=NROUND)


rows = []
def ev(name, rh, rp, ro, note):
    r = D.evaluate_raw(rh, rp, ro)
    row = dict(name=name, internal=round(r["internal"], 6),
               ext_pre=round(r["ext_pre"], 6), ext_post=round(r["ext_post"], 6),
               d_internal=round(r["internal"] - BASE["internal"], 6),
               d_pre=round(r["ext_pre"] - BASE["ext_pre"], 6),
               d_post=round(r["ext_post"] - BASE["ext_post"], 6),
               protocol_compliant=True, note=note)
    rows.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
    return r


SEEDS = [42, 7, 123, 2024, 99]

# 대조: 순수 logloss(ce=1.0) 5시드
lh, lp, lo = [], [], []
for sd in SEEDS:
    b = train_ranknet(5, 1.0, sd)
    rh, rp, ro = (b.predict(D.Xho, raw_score=True), b.predict(D.Xpre, raw_score=True),
                  b.predict(D.Xpost, raw_score=True))
    lh.append(rh); lp.append(rp); lo.append(ro)
    ev(f"ctrl_logloss_seed{sd}", rh, rp, ro, "순수 logloss(커스텀경로) 시드안정성 대조")
    del b

# 선택설정 RankNet 혼합 5시드
sh, sp, so = [], [], []
for sd in SEEDS:
    b = train_ranknet(BEST_K, BEST_LAM, sd)
    rh, rp, ro = (b.predict(D.Xho, raw_score=True), b.predict(D.Xpre, raw_score=True),
                  b.predict(D.Xpost, raw_score=True))
    sh.append(rh); sp.append(rp); so.append(ro)
    ev(f"ranknet_K{BEST_K}_ce{BEST_LAM}_seed{sd}", rh, rp, ro, "선택설정 단일시드")
    del b

# 앙상블
ev("ctrl_logloss_seedens", np.mean(lh, 0), np.mean(lp, 0), np.mean(lo, 0),
   "logloss 5시드 raw 평균(대조 앙상블)")
ev(f"ranknet_ce{BEST_LAM}_seedens_raw", np.mean(sh, 0), np.mean(sp, 0), np.mean(so, 0),
   "선택설정 5시드 raw-score 평균 앙상블")

def rk(s):
    return np.mean([pd.Series(x).rank().to_numpy() for x in s], 0)
ev(f"ranknet_ce{BEST_LAM}_seedens_rank", rk(sh), rk(sp), rk(so),
   "선택설정 5시드 순위평균 앙상블")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, "path5_06_ranknet_deep.csv"), index=False, encoding="utf-8-sig")

# 안정성 요약
def summ(tag, pre_list, post_list):
    pa = np.array(pre_list); po = np.array(post_list)
    print(f"[{tag}] pre {pa.mean():.5f}±{pa.std():.5f} (min {pa.min():.5f} max {pa.max():.5f})  "
          f"post {po.mean():.5f}±{po.std():.5f} (min {po.min():.5f} max {po.max():.5f})", flush=True)
selpre = [r["ext_pre"] for r in rows if r["name"].startswith(f"ranknet_K{BEST_K}_ce{BEST_LAM}_seed")]
selpost = [r["ext_post"] for r in rows if r["name"].startswith(f"ranknet_K{BEST_K}_ce{BEST_LAM}_seed")]
ctlpre = [r["ext_pre"] for r in rows if r["name"].startswith("ctrl_logloss_seed") and "ens" not in r["name"]]
ctlpost = [r["ext_post"] for r in rows if r["name"].startswith("ctrl_logloss_seed") and "ens" not in r["name"]]
print("\n=== seed-stability summary ===", flush=True)
summ("baseline_ref", [BASE["ext_pre"]], [BASE["ext_post"]])
summ("logloss_custom", ctlpre, ctlpost)
summ(f"ranknet_ce{BEST_LAM}", selpre, selpost)
print(f"TOTAL {time.time()-t0:.1f}s")
