# -*- coding: utf-8 -*-
"""
경로6 STAGE7 — 구조 제약(가법성) 및 부드러운 전문가 혼합(soft mixture-of-experts).

H1  가법(GAM형) 부스팅: max_depth=1 → 상호작용 0차. 구조가 단순할수록 전이 안정?
H2  깊이 2/3 제한 부스팅 (저차 상호작용만)
H3  연령 soft-gating MoE: 하드 분할 대신 연령에 따른 부드러운 전문가 가중
모든 선택은 KCHS 내부 OOF (§2.10 준수).
"""
import time, numpy as np, pandas as pd
from path6_common import *
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

rows = []
T0 = time.time()
CACHE = os.path.join(RES, "_p6_cache"); os.makedirs(CACHE, exist_ok=True)


def cached(name, fn):
    p = os.path.join(CACHE, name + ".npz")
    if os.path.exists(p):
        z = np.load(p); return [z[k] for k in sorted(z.files)]
    out = fn(); np.savez(p, **{f"a{i}": v for i, v in enumerate(out)}); return out


def oof_of(clf_fn, folds=3):
    o = np.zeros(len(XTR))
    for tr, va in StratifiedKFold(folds, shuffle=True, random_state=SEED).split(XTR, YTR):
        (z,), _ = fit_predict(XTR.iloc[tr].reset_index(drop=True), YTR[tr],
                              [XTR.iloc[va].reset_index(drop=True)], clf=clf_fn())
        o[va] = z
    return o


# ---------------------------------------------------------------- H1/H2 깊이 제한
specs = [("H1_gam_depth1", dict(max_depth=1, num_leaves=2, n_estimators=3000, learning_rate=0.05)),
         ("H2_depth2", dict(max_depth=2, num_leaves=4, n_estimators=2000, learning_rate=0.05)),
         ("H2_depth3", dict(max_depth=3, num_leaves=8, n_estimators=1500, learning_rate=0.05)),
         ("H2_depth4", dict(max_depth=4, num_leaves=16, n_estimators=1000, learning_rate=0.04)),
         ("H0_baseline", dict())]
oofv = {}
for nm, kw in specs:
    def _f(kw=kw):
        (a, b, c), _ = fit_predict(XTR, YTR, [XTE, XP, XQ], clf=new_lgbm(**kw))
        return [a, b, c]
    a, b, c = cached("struct_" + nm, _f)

    def _o(kw=kw):
        return [oof_of(lambda kw=kw: new_lgbm(**kw))]
    (o,) = cached("struct_oof_" + nm, _o)
    oofv[nm] = wauc(YTR, o, WTR)
    rows.append(report(nm, a, b, c, note=f"KCHS OOF={oofv[nm]:.5f} | {kw}"))
    print(rows[-1], flush=True)
best = max(oofv, key=oofv.get)
print("H 계열 KCHS OOF 최우수:", best, round(oofv[best], 5))

# ---------------------------------------------------------------- H3 soft MoE (연령)
def soft_w(age, centers, tau):
    D = np.stack([-((age - c) ** 2) / (2 * tau ** 2) for c in centers], 1)
    D -= D.max(1, keepdims=True)
    E = np.exp(D)
    return E / E.sum(1, keepdims=True)


centers = [30.0, 50.0, 70.0]
for tau in [8.0, 12.0, 20.0]:
    def _f(tau=tau):
        Wtr = soft_w(AGE_TR, centers, tau)
        outs = [np.zeros(len(XTE)), np.zeros(len(XP)), np.zeros(len(XQ))]
        Wl = [soft_w(AGE_TE, centers, tau), soft_w(AGE_P, centers, tau),
              soft_w(AGE_Q, centers, tau)]
        for j in range(len(centers)):
            sw = Wtr[:, j]
            keep = sw > 0.02
            (a, b, c), _ = fit_predict(XTR.loc[keep].reset_index(drop=True), YTR[keep],
                                       [XTE, XP, XQ], wtr=sw[keep])
            for t, s in zip(outs, [a, b, c]):
                pass
            outs[0] += Wl[0][:, j] * logit(a)
            outs[1] += Wl[1][:, j] * logit(b)
            outs[2] += Wl[2][:, j] * logit(c)
        return outs
    a, b, c = cached(f"moe_tau{tau}", _f)
    rows.append(report(f"H3_softMoE_age_tau{tau:.0f}", a, b, c,
                       note="연령 soft-gating 전문가혼합 (하드 층화의 부드러운 판)"))
    print(rows[-1], flush=True)

# ---------------------------------------------------------------- H4 순위목표 학습 (7)
# LightGBM lambdarank 는 query 당 10000행 제한 → 5000행 그룹으로 분할
def _rank():
    prep = make_prep(NUM, BIN, CAT, bin_impute=True)
    A = np.asarray(prep.fit_transform(XTR[FEATURES]), float)
    Bs = [np.asarray(prep.transform(X[FEATURES]), float) for X in (XTE, XP, XQ)]
    gsz = 5000
    grp = [gsz] * (len(A) // gsz)
    if len(A) % gsz: grp.append(len(A) % gsz)
    ds = lgb.Dataset(A, label=YTR, group=grp)
    pr = dict(objective="lambdarank", lambdarank_truncation_level=200,
              learning_rate=0.03, num_leaves=31, bagging_fraction=0.8,
              bagging_freq=1, feature_fraction=0.8, seed=SEED, verbose=-1,
              label_gain=[0, 1], num_threads=-1)
    bst = lgb.train(pr, ds, num_boost_round=600)
    return [bst.predict(b) for b in Bs]

try:
    a, b, c = cached("rank_obj", _rank)
    rows.append(report("H4_lambdarank_g5000", a, b, c,
                       note="(7) 순위목표 학습: 확률 대신 쌍별 순위 최적화 (query=5000행)"))
    print(rows[-1], flush=True)
except Exception as e:
    print("lambdarank failed:", e)

save(rows, "path6_07_structure.csv")
print("TOTAL", round(time.time() - T0, 1), "s")
