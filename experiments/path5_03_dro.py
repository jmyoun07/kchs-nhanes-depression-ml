# -*- coding: utf-8 -*-
"""
path5_03_dro.py — 경로5 (4) 분포강건 최적화(group DRO) + (6) 전이 정규화.

group DRO: 그룹별 평균 손실이 큰 그룹의 가중을 지수적으로 올려 '최악 그룹 손실'을 최소화.
          그룹은 KCHS 내부 변수(연도/연령대/성/교육/소득)로만 정의 -> 외부데이터 미사용.
전이 정규화: 그룹 간 예측분포(1차/2차 적률) 차이를 벌점으로 추가 -> 그룹불변 스코어 유도.
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
ktr = D.ktr
print(f"data ready {time.time()-t0:.1f}s", flush=True)


def codes(*arrs):
    key = None
    for a in arrs:
        a = pd.Series(np.asarray(a)).astype("category").cat.codes.to_numpy()
        key = a if key is None else key * (a.max() + 1) + a
    c = pd.Series(key).astype("category").cat.codes.to_numpy()
    return c.astype(np.int64), int(c.max() + 1)


age = ktr["age"].to_numpy(float)
ageb = np.clip((age // 10).astype(int), 1, 8)
sexv = ktr["sex"].astype(str).values
yrv = ktr["year"].to_numpy(int)
eduv = ktr["edu"].astype(str).values
incv = np.nan_to_num(ktr["income_q"].to_numpy(float), nan=-1).astype(int)

GROUPS = {
    "year": codes(yrv),
    "age10": codes(ageb),
    "age10_sex": codes(ageb, sexv),
    "year_age10": codes(yrv, ageb),
    "year_age10_sex": codes(yrv, ageb, sexv),
    "age10_sex_edu": codes(ageb, sexv, eduv),
    "age10_sex_edu_inc": codes(ageb, sexv, eduv, incv),
}
for kk, (g, ng) in GROUPS.items():
    print(f"  group {kk}: {ng} groups, min size {np.bincount(g).min()}", flush=True)


def run(name, fobj, note, base=None, compliant=True):
    t = time.time()
    try:
        b = train_native(D, None, fobj=fobj,
                         params_extra=dict(boost_from_average=False),
                         init_score=INIT)
        r = D.evaluate_raw(predict_raw(b, D.Xho), predict_raw(b, D.Xpre),
                           predict_raw(b, D.Xpost))
        return rec(name, r["internal"], r["ext_pre"], r["ext_post"], compliant, note,
                   base=base, extra=dict(sec=round(time.time() - t, 1)))
    except Exception as e:
        print(f"!! {name} FAILED: {type(e).__name__}: {e}", flush=True)
        return rec(name, np.nan, np.nan, np.nan, compliant, f"FAILED: {e}", base=base)


BASE = rec("00_native_binary", 0.830969, 0.774013, 0.765262, True, "기준선(재현값)")
CTRL = run("01_custom_logloss_ctrl", O.make_logloss(),
           "커스텀 경로 대조군(logloss)", base=BASE)

# ---------------------------------------------------------------- group DRO
for gname in ["year", "age10", "age10_sex", "year_age10", "age10_sex_edu"]:
    g, ng = GROUPS[gname]
    for eta in [0.01, 0.1, 1.0]:
        run(f"70_dro[{gname}]_eta{eta}", O.make_group_dro(g, ng, eta=eta),
            f"group DRO(온라인 지수가중), 그룹={gname}({ng}개), eta={eta}", base=CTRL)

# 큰 그룹수(세밀) + 강한 eta
g, ng = GROUPS["age10_sex_edu_inc"]
for eta in [0.1, 1.0]:
    run(f"71_dro[fine]_eta{eta}", O.make_group_dro(g, ng, eta=eta),
        f"group DRO 세밀그룹({ng}개), eta={eta}", base=CTRL)

# ---------------------------------------------------------------- 전이 정규화
for gname in ["year", "age10_sex", "age10_sex_edu"]:
    g, ng = GROUPS[gname]
    for lam in [0.1, 1.0, 10.0]:
        run(f"80_transreg[{gname}]_lam{lam}", O.make_transfer_reg(g, ng, lam=lam),
            f"전이정규화: 그룹({gname}) 예측평균 정렬 벌점 lam={lam}", base=CTRL)
g, ng = GROUPS["age10_sex"]
for lam in [1.0, 10.0]:
    run(f"81_transreg_m2[age10_sex]_lam{lam}", O.make_transfer_reg(g, ng, lam=lam, moment=2),
        f"전이정규화 2차적률(평균+분산) lam={lam}", base=CTRL)

dump("path5_03_dro.csv")
print(f"TOTAL {time.time()-t0:.1f}s")
