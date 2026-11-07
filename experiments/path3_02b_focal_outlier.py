# -*- coding: utf-8 -*-
"""
path3_02b_focal_outlier.py — path3_02에서 focal loss 래퍼 버그로 중단된 구간 재실행.
  I2) focal loss (수정된 래퍼)
  O)  이상치 처리
  G)  밀도비 가중이 왜 무력한지 진단 (positivity/overlap 위반)
출력: results/path3_02b_focal_outlier.csv, results/path3_02b_densratio_diag.csv
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from path3_common import Bench, Recorder, RES, SEED, FEATURES, NUM, make_pre, wauc

B = Bench()
R = Recorder(os.path.join(RES, "path3_02b_focal_outlier.csv"))
N500 = 500000
rng0 = np.random.RandomState(SEED)
tr500 = rng0.choice(B.pool, N500, replace=False)

ref, _, _ = B.run("REF_pool500k_s42", tr500, note="크기통제 기준(=path3_02 REF)")
R.base = ref
R.add(ref, "0_REF", True)
REF = ref

# ============================================================
# I2) focal loss (수정된 래퍼: 부스터 raw score 직접 사용)
# ============================================================
R.log("\n########## I2) focal loss ##########")


class FocalLGBM(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=0.25, gamma=2.0, **kw):
        self.alpha = alpha
        self.gamma = gamma
        self.kw = kw

    def _obj(self):
        a, g = self.alpha, self.gamma

        def fl(x, t):
            p = 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
            p = np.clip(p, 1e-9, 1 - 1e-9)
            pt = t * p + (1 - t) * (1 - p)
            at = a * t + (1 - a) * (1 - t)
            return -at * (1 - pt) ** g * np.log(pt)

        def obj(y_true, y_pred):
            h = 1e-4
            f1 = fl(y_pred + h, y_true); f0 = fl(y_pred - h, y_true); fc = fl(y_pred, y_true)
            grad = (f1 - f0) / (2 * h)
            hess = np.maximum((f1 - 2 * fc + f0) / (h * h), 1e-6)
            return grad, hess
        return obj

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        self.m_ = lgb.LGBMClassifier(objective=self._obj(), **self.kw)
        self.m_.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        raw = self.m_.booster_.predict(X)          # 커스텀 목적함수 → raw score
        p = 1.0 / (1.0 + np.exp(-np.clip(raw, -30, 30)))
        return np.vstack([1 - p, p]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


KW = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, subsample=0.8,
          colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1)
for a, g in [(0.25, 1.0), (0.25, 2.0), (0.5, 2.0), (0.75, 2.0), (0.9, 2.0)]:
    est = FocalLGBM(alpha=a, gamma=g, **KW)
    r, _, _ = B.run(f"I_focal_a{a}_g{g}", tr500, est=est, note=f"focal alpha={a} gamma={g}")
    R.add(r, "I_imbalance", True)

# ============================================================
# O) 이상치 처리
# ============================================================
R.log("\n########## O) 이상치/극단값 처리 (cutpoint는 KCHS train에서만) ##########")


def wins_frames(cols, lo_q, hi_q):
    lo = {c: float(np.nanquantile(B.k[c].to_numpy(float)[tr500], lo_q)) for c in cols}
    hi = {c: float(np.nanquantile(B.k[c].to_numpy(float)[tr500], hi_q)) for c in cols}
    Xk = B.k[FEATURES].copy(); Xp = B.pre[FEATURES].copy(); Xq = B.post[FEATURES].copy()
    for c in cols:
        for D in (Xk, Xp, Xq):
            D[c] = np.clip(D[c].to_numpy(float), lo[c], hi[c])
    return Xk, Xp, Xq, lo, hi


for cols, lq, hq, tag in [(["bmi", "sleep_hrs"], .01, .99, "O_wins_bmisleep_1_99"),
                          (["bmi", "sleep_hrs"], .005, .995, "O_wins_bmisleep_05_995"),
                          (["bmi", "sleep_hrs"], .025, .975, "O_wins_bmisleep_25_975"),
                          (NUM, .01, .99, "O_wins_allnum_1_99"),
                          (NUM, .025, .975, "O_wins_allnum_25_975")]:
    Xk, Xp, Xq, lo, hi = wins_frames(cols, lq, hq)
    cut = {c: (round(lo[c], 1), round(hi[c], 1)) for c in cols}
    r, _, _ = B.run(tag, tr500, X=Xk, Xho=Xk.iloc[B.ho], Xpre=Xp, Xpost=Xq,
                    note=f"winsorize {cols} [{lq},{hq}] cut={cut}")
    R.add(r, "O_outlier", True)

RANGES = {"bmi": (13.0, 50.0), "sleep_hrs": (3.0, 13.0)}
Xk = B.k[FEATURES].copy(); Xp = B.pre[FEATURES].copy(); Xq = B.post[FEATURES].copy()
for c, (lo, hi) in RANGES.items():
    for D in (Xk, Xp, Xq):
        D[c] = np.clip(D[c].to_numpy(float), lo, hi)
r, _, _ = B.run("O_clip_plausible", tr500, X=Xk, Xho=Xk.iloc[B.ho], Xpre=Xp, Xpost=Xq,
                note="bmi[13,50] sleep[3,13] 타당범위 clip")
R.add(r, "O_outlier", True)

Xk = B.k[FEATURES].copy(); Xp = B.pre[FEATURES].copy(); Xq = B.post[FEATURES].copy()
for c, (lo, hi) in RANGES.items():
    for D in (Xk, Xp, Xq):
        v = D[c].to_numpy(float)
        v[(v < lo) | (v > hi)] = np.nan
        D[c] = v
r, _, _ = B.run("O_implausible_to_nan", tr500, X=Xk, Xho=Xk.iloc[B.ho], Xpre=Xp, Xpost=Xq,
                note="타당범위 밖은 결측화 후 median 대체")
R.add(r, "O_outlier", True)

# 극단값 행 자체를 학습에서 제외
m = np.ones(len(tr500), bool)
for c, (lo, hi) in RANGES.items():
    v = B.k[c].to_numpy(float)[tr500]
    m &= ~((v < lo) | (v > hi))
r, _, _ = B.run("O_drop_implausible_rows", tr500[m],
                note=f"타당범위 밖 학습행 제외 ({(~m).sum():,}행 제거)")
R.add(r, "O_outlier", True)

# ============================================================
# G) 밀도비 가중 무력화 진단
# ============================================================
R.log("\n########## G) 밀도비(공변량 이동) 가중 진단 ##########")
sub = rng0.choice(tr500, 60000, replace=False)
Xd = pd.concat([B.k[FEATURES].iloc[sub], B.pre[FEATURES]], axis=0)
yd = np.r_[np.zeros(len(sub)), np.ones(len(B.pre))]
dp = Pipeline([("pre", make_pre()),
               ("clf", lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                          num_leaves=31, random_state=SEED,
                                          n_jobs=-1, verbose=-1))])
oof = cross_val_predict(dp, Xd, yd, cv=5, method="predict_proba")[:, 1]
dauc = wauc(yd, oof, np.ones(len(yd)))
pk = oof[:len(sub)]
w = pk / np.maximum(1 - pk, 1e-12)
ess = (w.sum() ** 2) / max((w ** 2).sum(), 1e-30)
diag = dict(domain_auc_oof=dauc,
            kchs_p_max=float(pk.max()), kchs_p_median=float(np.median(pk)),
            kchs_p_q99=float(np.quantile(pk, .99)),
            frac_kchs_p_lt_1e3=float((pk < 1e-3).mean()),
            frac_kchs_p_lt_1e2=float((pk < 1e-2).mean()),
            w_median=float(np.median(w)), w_q99=float(np.quantile(w, .99)),
            w_max=float(w.max()), ess=float(ess), n_kchs=len(sub),
            ess_ratio=float(ess / len(sub)),
            nhanes_p_min=float(oof[len(sub):].min()),
            nhanes_p_median=float(np.median(oof[len(sub):])))
pd.DataFrame([diag]).to_csv(os.path.join(RES, "path3_02b_densratio_diag.csv"),
                            index=False, encoding="utf-8-sig")
for kk, vv in diag.items():
    R.log(f"  {kk:24s} = {vv}")
R.log("  → OOF 도메인 AUC가 1에 가까울수록 두 모집단의 공변량 지지집합이 겹치지 않음"
      "(positivity 위반) → 밀도비 재가중은 정의상 무력/불안정")

R.log(f"\n완료. 저장: {R.out}")
