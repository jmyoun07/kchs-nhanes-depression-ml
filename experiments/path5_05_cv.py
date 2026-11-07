# -*- coding: utf-8 -*-
"""
path5_05_cv.py — 경로5 (3)(4) 모델선택: **KCHS 내부 5-fold CV 만으로** 목적함수를 고른다.

  * 500k 학습표본을 StratifiedKFold(5) 로 나눠 fold 내부 가중 AUROC 평균을 계산.
  * NHANES 는 이 스크립트에서 전혀 사용하지 않는다(§2.10).
  * 최종 선택된 후보만 path5_06 에서 외부 1회 평가.

스크리닝(내부 홀드아웃)에서 내부 성능 상위 + 각 계열 대표를 후보로 삼는다.
"""
import os, sys, time, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path5_common import *      # noqa
import path5_obj as O
import lightgbm as lgb

t0 = time.time()
D = Data()
n = len(D.ytr)
print(f"data ready {time.time()-t0:.1f}s n={n}", flush=True)

FOLDS = cv_folds(D.ytr, n_splits=5, seed=SEED)


def make_obj(spec, y_fold):
    kind = spec["kind"]
    if kind == "logloss":
        return O.make_logloss()
    if kind == "focal":
        return O.make_focal(gamma=spec["gamma"], alpha=spec.get("alpha"))
    if kind == "ls":
        return O.make_label_smoothing(eps=spec["eps"])
    if kind == "gce":
        return O.make_gce(q=spec["q"])
    if kind == "gradclip":
        return O.make_gradclip(c=spec["c"])
    if kind == "ranknet":
        return O.make_ranknet(y_fold, n_neg_per_pos=spec["K"],
                              lam_ce=spec.get("lam_ce", 0.0), seed=SEED)
    if kind == "brier":
        return O.make_brier()
    raise ValueError(kind)


CANDS = [
    ("C_logloss_ctrl",        dict(kind="logloss")),
    ("focal_g0.5",            dict(kind="focal", gamma=0.5)),
    ("focal_g1.0",            dict(kind="focal", gamma=1.0)),
    ("focal_g2.0",            dict(kind="focal", gamma=2.0)),
    ("focal_g2.0_a0.9",       dict(kind="focal", gamma=2.0, alpha=0.9)),
    ("ls_e0.01",              dict(kind="ls", eps=0.01)),
    ("ls_e0.02",              dict(kind="ls", eps=0.02)),
    ("gradclip_c0.05",        dict(kind="gradclip", c=0.05)),
    ("ranknet_K5_ce0.5",      dict(kind="ranknet", K=5, lam_ce=0.5)),
    ("ranknet_K5_ce0.25",     dict(kind="ranknet", K=5, lam_ce=0.25)),
    ("gce_q0.1",              dict(kind="gce", q=0.1)),
]

NATIVE_CANDS = [
    ("N_binary_baseline", dict(objective="binary")),
    ("N_lambdarank_g100", dict(objective="lambdarank", group=100,
                               lambdarank_truncation_level=30, label_gain=[0, 1])),
    ("N_lambdarank_g1000_full", dict(objective="lambdarank", group=1000,
                                     lambdarank_truncation_level=1000, label_gain=[0, 1])),
    ("N_scale_pos_weight_2", dict(objective="binary", scale_pos_weight=2)),
    ("N_cross_entropy", dict(objective="cross_entropy")),
]

rows = []


def cv_custom(name, spec):
    t = time.time()
    sc = []
    for fi, (a, b) in enumerate(FOLDS):
        ya = D.ytr[a]
        pr = float(ya.mean()); f0 = np.log(pr / (1 - pr))
        fobj = make_obj(spec, ya)
        p = dict(NATIVE); p.update(objective=fobj, boost_from_average=False)
        ds = lgb.Dataset(D.Xtr[a], label=ya, init_score=np.full(len(a), f0),
                         free_raw_data=False)
        bst = lgb.train(p, ds, num_boost_round=NROUND)
        pv = bst.predict(D.Xtr[b], raw_score=True)
        sc.append(wauc(D.ytr[b], pv, D.Wtr[b]))
        del bst, ds
    r = dict(name=name, cv_mean=round(float(np.mean(sc)), 6),
             cv_sd=round(float(np.std(sc)), 6),
             folds=[round(s, 6) for s in sc], sec=round(time.time() - t, 1),
             family="custom", spec=json.dumps(spec))
    rows.append(r); print(json.dumps(r, ensure_ascii=False), flush=True)
    return r


def cv_native(name, spec):
    t = time.time()
    sc = []
    sp = dict(spec)
    g = sp.pop("group", None)
    for fi, (a, b) in enumerate(FOLDS):
        p = dict(NATIVE); p.update(sp); p["metric"] = "None"
        grp = None
        if g:
            q, r_ = divmod(len(a), g)
            grp = np.array([g] * q + ([r_] if r_ else []), dtype=np.int32)
        ds = lgb.Dataset(D.Xtr[a], label=D.ytr[a], group=grp, free_raw_data=False)
        bst = lgb.train(p, ds, num_boost_round=NROUND)
        pv = bst.predict(D.Xtr[b], raw_score=True)
        sc.append(wauc(D.ytr[b], pv, D.Wtr[b]))
        del bst, ds
    r = dict(name=name, cv_mean=round(float(np.mean(sc)), 6),
             cv_sd=round(float(np.std(sc)), 6),
             folds=[round(s, 6) for s in sc], sec=round(time.time() - t, 1),
             family="native", spec=json.dumps(spec))
    rows.append(r); print(json.dumps(r, ensure_ascii=False), flush=True)
    return r


for nm, sp in NATIVE_CANDS:
    try:
        cv_native(nm, sp)
    except Exception as e:
        print(f"!! {nm} FAILED {e}", flush=True)
for nm, sp in CANDS:
    try:
        cv_custom(nm, sp)
    except Exception as e:
        print(f"!! {nm} FAILED {e}", flush=True)

df = pd.DataFrame(rows).sort_values("cv_mean", ascending=False)
os.makedirs(RES, exist_ok=True)
df.to_csv(os.path.join(RES, "path5_05_cv.csv"), index=False, encoding="utf-8-sig")
print(df[["name", "cv_mean", "cv_sd"]].to_string(index=False), flush=True)
print("SELECTED(by internal CV only):", df.iloc[0]["name"], flush=True)
print(f"TOTAL {time.time()-t0:.1f}s")
