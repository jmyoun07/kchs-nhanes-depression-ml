# -*- coding: utf-8 -*-
"""
path5_07_final.py — 경로5 최종 확정 실험(결합).

A) 선택 정당화: 내부 5-fold 가중 AUROC CV 로 후보 목적함수 비교(외부 미사용, §2.10).
   - binary(대조) / ranknet_K5_ce0.5 / ranknet_K5_ce0.25 / focal_g2.0 / focal_g2.0_a0.9
B) 시드 안정성: 내부 CV 로 선택된 후보(ranknet_K5_ce0.5)와 logloss 대조를
   5개 시드로 학습 -> 외부 pre/post 평균±표준편차. 외부 개선이 진짜인지 판정.
C) 시드앙상블(raw 평균)로 분산축소 후 최종 외부 1회 평가.

주의: 커스텀 objective 는 Python 콜백이라 느리다 -> CV 는 후보만 선별해 수행.
"""
import os, sys, time, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path5_common import *      # noqa
import path5_obj as O
import lightgbm as lgb

t0 = time.time()
D = Data()
FOLDS = cv_folds(D.ytr, n_splits=5, seed=SEED)
BASE = dict(internal=0.830969, ext_pre=0.774013, ext_post=0.765262)
print(f"data ready {time.time()-t0:.1f}s", flush=True)


def make_obj(spec, y):
    k = spec["kind"]
    if k == "logloss": return O.make_logloss()
    if k == "focal":   return O.make_focal(gamma=spec["gamma"], alpha=spec.get("alpha"))
    if k == "ranknet": return O.make_ranknet(y, n_neg_per_pos=spec["K"],
                                             lam_ce=spec.get("lam_ce", 0.0), seed=spec.get("seed", SEED))
    raise ValueError(k)


def fit_custom(X, y, spec, seed=SEED):
    pr = float(y.mean()); f0 = np.log(pr / (1 - pr))
    sp = dict(spec); sp["seed"] = seed
    fobj = make_obj(sp, y)
    p = dict(NATIVE); p.update(objective=fobj, boost_from_average=False, seed=seed)
    ds = lgb.Dataset(X, label=y, init_score=np.full(len(y), f0), free_raw_data=False)
    return lgb.train(p, ds, num_boost_round=NROUND)


def fit_native(X, y, sp, seed=SEED):
    p = dict(NATIVE); p.update(sp); p["seed"] = seed
    ds = lgb.Dataset(X, label=y, free_raw_data=False)
    return lgb.train(p, ds, num_boost_round=NROUND)


# =============================================================== A) 내부 CV 선택
cv_rows = []
CV_CANDS = [
    ("binary_ctrl", "native", dict(objective="binary")),
    ("ranknet_K5_ce0.5", "custom", dict(kind="ranknet", K=5, lam_ce=0.5)),
    ("ranknet_K5_ce0.25", "custom", dict(kind="ranknet", K=5, lam_ce=0.25)),
    ("focal_g2.0", "custom", dict(kind="focal", gamma=2.0)),
    ("focal_g2.0_a0.9", "custom", dict(kind="focal", gamma=2.0, alpha=0.9)),
]
for nm, fam, sp in CV_CANDS:
    t = time.time(); sc = []
    for (a, b) in FOLDS:
        ya = D.ytr[a]
        bst = fit_native(D.Xtr[a], ya, sp) if fam == "native" else fit_custom(D.Xtr[a], ya, sp)
        pv = bst.predict(D.Xtr[b], raw_score=True)
        sc.append(wauc(D.ytr[b], pv, D.Wtr[b]))
        del bst
    r = dict(name=nm, cv_mean=round(float(np.mean(sc)), 6), cv_sd=round(float(np.std(sc)), 6),
             folds=[round(s, 6) for s in sc], sec=round(time.time() - t, 1))
    cv_rows.append(r); print("CV", json.dumps(r, ensure_ascii=False), flush=True)

cvdf = pd.DataFrame(cv_rows).sort_values("cv_mean", ascending=False)
cvdf.to_csv(os.path.join(RES, "path5_07_cv.csv"), index=False, encoding="utf-8-sig")
print(cvdf[["name", "cv_mean", "cv_sd"]].to_string(index=False), flush=True)
best_nonbase = cvdf[cvdf.name != "binary_ctrl"].iloc[0]
print(f"internal-CV best (non-baseline objective): {best_nonbase['name']} "
      f"cv={best_nonbase['cv_mean']} vs binary={cvdf[cvdf.name=='binary_ctrl'].iloc[0]['cv_mean']}",
      flush=True)

# =============================================================== B/C) 시드 안정성 + 앙상블
ext_rows = []
def ev(name, rh, rp, ro, note):
    r = D.evaluate_raw(rh, rp, ro)
    row = dict(name=name, internal=round(r["internal"], 6),
               ext_pre=round(r["ext_pre"], 6), ext_post=round(r["ext_post"], 6),
               d_internal=round(r["internal"] - BASE["internal"], 6),
               d_pre=round(r["ext_pre"] - BASE["ext_pre"], 6),
               d_post=round(r["ext_post"] - BASE["ext_post"], 6),
               protocol_compliant=True, note=note)
    ext_rows.append(row); print("EXT", json.dumps(row, ensure_ascii=False), flush=True)
    return r

SEEDS = [42, 7, 123, 2024, 99]
SEL = dict(kind="ranknet", K=5, lam_ce=0.5)

lh, lp, lo = [], [], []
sh, sp_, so = [], [], []
for sd in SEEDS:
    bl = fit_custom(D.Xtr, D.ytr, dict(kind="logloss"), seed=sd)
    ev(f"ctrl_logloss_seed{sd}", bl.predict(D.Xho, raw_score=True),
       bl.predict(D.Xpre, raw_score=True), bl.predict(D.Xpost, raw_score=True), "logloss 대조")
    lh.append(bl.predict(D.Xho, raw_score=True)); lp.append(bl.predict(D.Xpre, raw_score=True))
    lo.append(bl.predict(D.Xpost, raw_score=True)); del bl

    bs = fit_custom(D.Xtr, D.ytr, SEL, seed=sd)
    ev(f"ranknet_ce0.5_seed{sd}", bs.predict(D.Xho, raw_score=True),
       bs.predict(D.Xpre, raw_score=True), bs.predict(D.Xpost, raw_score=True), "선택설정 단일시드")
    sh.append(bs.predict(D.Xho, raw_score=True)); sp_.append(bs.predict(D.Xpre, raw_score=True))
    so.append(bs.predict(D.Xpost, raw_score=True)); del bs

ev("ctrl_logloss_seedens", np.mean(lh, 0), np.mean(lp, 0), np.mean(lo, 0), "logloss 5시드 raw평균")
ev("ranknet_ce0.5_seedens_raw", np.mean(sh, 0), np.mean(sp_, 0), np.mean(so, 0),
   "선택설정 5시드 raw-score 평균 앙상블")

pd.DataFrame(ext_rows).to_csv(os.path.join(RES, "path5_07_seedstab.csv"),
                              index=False, encoding="utf-8-sig")

def s(name, arr):
    a = np.array(arr)
    return f"{name}: mean={a.mean():.5f} sd={a.std():.5f} min={a.min():.5f} max={a.max():.5f}"
selpre = [r["ext_pre"] for r in ext_rows if r["name"].startswith("ranknet_ce0.5_seed")]
selpost = [r["ext_post"] for r in ext_rows if r["name"].startswith("ranknet_ce0.5_seed")]
ctlpre = [r["ext_pre"] for r in ext_rows if r["name"].startswith("ctrl_logloss_seed") and "ens" not in r["name"]]
ctlpost = [r["ext_post"] for r in ext_rows if r["name"].startswith("ctrl_logloss_seed") and "ens" not in r["name"]]
print("\n=== SEED-STABILITY (ext) ===", flush=True)
print(f"baseline_ref: pre={BASE['ext_pre']:.5f} post={BASE['ext_post']:.5f}", flush=True)
print(s("logloss_pre ", ctlpre), flush=True)
print(s("ranknet_pre ", selpre), flush=True)
print(s("logloss_post", ctlpost), flush=True)
print(s("ranknet_post", selpost), flush=True)
print(f"TOTAL {time.time()-t0:.1f}s")
