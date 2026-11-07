# -*- coding: utf-8 -*-
"""
path5_01b_rest.py — path5_01 이 외부 요인(프로세스 강제종료)으로 중단된 뒤 남은 변형 재실행.
  * 남은 항목: gradclip c>=0.3, brier, native 대안 목적함수(cross_entropy 등), gce q0.5/0.7
  * 앞서 로그로만 남은 결과는 path5_01_salvage.py 로 CSV 복원.
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path5_common import *      # noqa
import path5_obj as O

t0 = time.time()
D = Data()
prior = float(D.ytr.mean())
F0 = np.log(prior / (1 - prior))
INIT = np.full(len(D.ytr), F0)
print(f"data ready {time.time()-t0:.1f}s", flush=True)

BASE = rec("00_native_binary", 0.830969, 0.774013, 0.765262, True, "기준선(재현값)")
CTRL = rec("01_custom_logloss_ctrl", 0.830588, 0.772075, 0.763515, True,
           "커스텀 대조군(path5_01 재현값)", base=BASE)


def run(name, fobj=None, native_obj=None, params_extra=None, note="", base=None):
    t = time.time()
    pe = dict(params_extra or {})
    init = None
    if fobj is not None:
        pe.setdefault("boost_from_average", False)
        init = INIT
    try:
        b = train_native(D, native_obj, fobj=fobj, params_extra=pe, init_score=init)
        r = D.evaluate_raw(predict_raw(b, D.Xho), predict_raw(b, D.Xpre),
                           predict_raw(b, D.Xpost))
        return rec(name, r["internal"], r["ext_pre"], r["ext_post"], True, note,
                   base=base, extra=dict(sec=round(time.time() - t, 1)))
    except Exception as e:
        print(f"!! {name} FAILED: {type(e).__name__}: {e}", flush=True)
        return rec(name, np.nan, np.nan, np.nan, True, f"FAILED: {e}", base=base)


for q in [0.5, 0.7]:
    run(f"30_gce_q{q}", fobj=O.make_gce(q=q), note=f"GCE q={q}", base=CTRL)
for c in [0.3, 0.5, 0.9]:
    run(f"32_gradclip_c{c}", fobj=O.make_gradclip(c=c),
        note=f"Huber화 로지스틱: |grad| 절단 c={c}", base=CTRL)
run("33_brier", fobj=O.make_brier(), note="Brier(제곱)손실", base=CTRL)

run("40_cross_entropy", native_obj="cross_entropy",
    note="LightGBM native cross_entropy", base=BASE)
run("41_xentlambda", native_obj="cross_entropy_lambda",
    note="LightGBM native cross_entropy_lambda", base=BASE)
run("42_regression_l2", native_obj="regression",
    note="이진 라벨 L2 회귀(링크 없음)", base=BASE)
run("43_regression_huber", native_obj="huber", params_extra=dict(alpha=0.9),
    note="native Huber 회귀 손실", base=BASE)
run("44_regression_fair", native_obj="fair", note="native Fair 손실", base=BASE)

dump("path5_01b_rest.csv")
print(f"TOTAL {time.time()-t0:.1f}s")
