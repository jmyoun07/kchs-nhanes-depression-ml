# -*- coding: utf-8 -*-
"""
path5_01_screen.py — 경로5 (2) 목적함수 실측 스크리닝.

고정: split(500k/100k), 설계행렬, 트리 하이퍼파라미터(num_leaves=31, lr=0.03, 600 rounds).
변경: objective(손실)만.

기록만 하고 선택은 하지 않는다(선택은 path5_05 의 KCHS 내부 CV로).
"""
import os, sys, time, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path5_common import *      # noqa
import path5_obj as O
import lightgbm as lgb

t0 = time.time()
D = Data()
print(f"data ready {time.time()-t0:.1f}s  Xtr={D.Xtr.shape}  pos={D.ytr.sum()}", flush=True)

prior = float(D.ytr.mean())
F0 = np.log(prior / (1 - prior))
INIT = np.full(len(D.ytr), F0)


def run(name, fobj=None, native_obj=None, params_extra=None, note="",
        compliant=True, nround=NROUND, use_init=True, base=None):
    t = time.time()
    pe = dict(params_extra or {})
    if fobj is not None:
        pe.setdefault("boost_from_average", False)
        init = INIT if use_init else None
    else:
        init = None
    try:
        b = train_native(D, native_obj, fobj=fobj, params_extra=pe, nround=nround,
                         init_score=init)
        r = D.evaluate_raw(predict_raw(b, D.Xho), predict_raw(b, D.Xpre),
                           predict_raw(b, D.Xpost))
        return rec(name, r["internal"], r["ext_pre"], r["ext_post"], compliant, note,
                   base=base, extra=dict(sec=round(time.time() - t, 1)))
    except Exception as e:
        print(f"!! {name} FAILED: {type(e).__name__}: {e}", flush=True)
        return rec(name, np.nan, np.nan, np.nan, compliant, f"FAILED: {e}", base=base)


# ---------------------------------------------------------------- 대조군
BASE_NATIVE = run("00_native_binary", native_obj="binary",
                  note="기준선(=명세 재현). 모든 비교의 절대 기준")
CTRL = run("01_custom_logloss_ctrl", fobj=O.make_logloss(),
           note="커스텀 objective 경로의 대조군(logloss). 커스텀군은 이 값과 비교",
           base=BASE_NATIVE)

# ---------------------------------------------------------------- (2) focal loss
for g in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]:
    run(f"10_focal_g{g}", fobj=O.make_focal(gamma=g),
        note=f"focal loss gamma={g}, alpha 없음", base=CTRL)
for g in [1.0, 2.0]:
    run(f"11_focal_g{g}_hnorm", fobj=O.make_focal(gamma=g, hess_norm=True),
        note=f"focal gamma={g} + 총헤시안 정규화(logloss와 동일 스케일)", base=CTRL)
for g, a in [(1.0, 0.75), (2.0, 0.75), (2.0, 0.9), (2.0, 0.25)]:
    run(f"12_focal_g{g}_a{a}", fobj=O.make_focal(gamma=g, alpha=a),
        note=f"focal gamma={g} alpha={a} (클래스 가중)", base=CTRL)

# ---------------------------------------------------------------- (3) label smoothing
for e in [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]:
    run(f"20_labelsmooth_e{e}", fobj=O.make_label_smoothing(eps=e),
        note=f"label smoothing eps={e}", base=CTRL)

# ---------------------------------------------------------------- (5) robust / Huber 계열
for q in [0.1, 0.3, 0.5, 0.7, 1.0]:
    run(f"30_gce_q{q}", fobj=O.make_gce(q=q),
        note=f"Generalized Cross Entropy q={q} (q→0=CE, 1=MAE) 라벨잡음 강건", base=CTRL)
run("31_savage", fobj=O.make_savage(), note="Savage loss(유계 손실)", base=CTRL)
for c in [0.05, 0.1, 0.3, 0.5, 0.9]:
    run(f"32_gradclip_c{c}", fobj=O.make_gradclip(c=c),
        note=f"Huber화 로지스틱: |grad| 절단 c={c}", base=CTRL)
run("33_brier", fobj=O.make_brier(), note="Brier(제곱)손실", base=CTRL)

# ---------------------------------------------------------------- native 대안 목적함수
run("40_cross_entropy", native_obj="cross_entropy",
    note="LightGBM native cross_entropy", base=BASE_NATIVE)
run("41_xentlambda", native_obj="cross_entropy_lambda",
    note="LightGBM native cross_entropy_lambda(=XE^lambda)", base=BASE_NATIVE)
run("42_regression_l2", native_obj="regression",
    note="이진 라벨에 L2 회귀(=Brier, 링크 없음)", base=BASE_NATIVE)
run("43_regression_huber", native_obj="huber", params_extra=dict(alpha=0.9),
    note="native Huber 회귀 손실(이진 라벨)", base=BASE_NATIVE)
run("44_regression_fair", native_obj="fair",
    note="native Fair 손실(robust 회귀)", base=BASE_NATIVE)

dump("path5_01_screen.csv")
print(f"TOTAL {time.time()-t0:.1f}s")
