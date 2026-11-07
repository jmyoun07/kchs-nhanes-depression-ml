# -*- coding: utf-8 -*-
"""
path5_04_extra.py — 경로5 보조: 손실의 클래스가중/기울기 스케일 변형(불균형 대응 계열).

focal 과 같은 목적을 다른 방식으로 건드리는 저비용 native 옵션들.
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path5_common import *      # noqa
import path5_obj as O

t0 = time.time()
D = Data()
prior = float(D.ytr.mean())
print(f"data ready {time.time()-t0:.1f}s prior={prior:.4f}", flush=True)


def run(name, native_obj="binary", params_extra=None, note="", base=None, wtr=None):
    t = time.time()
    try:
        b = train_native(D, native_obj, params_extra=params_extra, wtr=wtr)
        r = D.evaluate_raw(predict_raw(b, D.Xho), predict_raw(b, D.Xpre),
                           predict_raw(b, D.Xpost))
        return rec(name, r["internal"], r["ext_pre"], r["ext_post"], True, note,
                   base=base, extra=dict(sec=round(time.time() - t, 1)))
    except Exception as e:
        print(f"!! {name} FAILED: {type(e).__name__}: {e}", flush=True)
        return rec(name, np.nan, np.nan, np.nan, True, f"FAILED: {e}", base=base)


BASE = run("00_native_binary", note="기준선(=명세 재현)")

# ---- 클래스 불균형 대응(손실 가중)
run("90_is_unbalance", params_extra=dict(is_unbalance=True),
    note="is_unbalance=True (양성 손실을 n_neg/n_pos 로 가중)", base=BASE)
for s in [2, 5, 10, 26.7]:
    run(f"91_scale_pos_weight_{s}", params_extra=dict(scale_pos_weight=s),
        note=f"scale_pos_weight={s}", base=BASE)

# ---- 손실 기울기(sigmoid) 스케일
for sg in [0.5, 0.75, 1.5, 2.0]:
    run(f"92_sigmoid_{sg}", params_extra=dict(sigmoid=sg),
        note=f"binary objective sigmoid={sg} (손실 기울기 스케일)", base=BASE)

# ---- 초기값
run("93_no_boost_from_average", params_extra=dict(boost_from_average=False),
    note="boost_from_average=False", base=BASE)

# ---- 언더샘플링(=암묵적 손실 재가중)
rng = np.random.RandomState(SEED)
P = np.where(D.ytr > 0.5)[0]
N = np.where(D.ytr <= 0.5)[0]
for ratio in [1, 3, 10]:
    sel = np.concatenate([P, rng.choice(N, size=min(len(N), ratio * len(P)), replace=False)])
    sel.sort()
    t = time.time()
    b = train_native(D, "binary", Xtr=D.Xtr[sel], ytr=D.ytr[sel])
    r = D.evaluate_raw(predict_raw(b, D.Xho), predict_raw(b, D.Xpre), predict_raw(b, D.Xpost))
    rec(f"94_undersample_1to{ratio}", r["internal"], r["ext_pre"], r["ext_post"], True,
        f"음성 언더샘플링 1:{ratio} (n={len(sel)})", base=BASE,
        extra=dict(sec=round(time.time() - t, 1)))

dump("path5_04_extra.csv")
print(f"TOTAL {time.time()-t0:.1f}s")
