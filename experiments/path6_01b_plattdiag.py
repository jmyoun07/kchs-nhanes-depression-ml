# -*- coding: utf-8 -*-
"""STAGE1 부속 — S2_age_R2_platt 의 이상값(0.673) 원인 진단 (소규모 부분표본)."""
import numpy as np, pandas as pd
from path6_common import *
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression as LR
from sklearn.isotonic import IsotonicRegression

sub = train_test_split(np.arange(len(XTR)), train_size=150_000,
                       stratify=YTR, random_state=1)[0]
Xs = XTR.iloc[sub].reset_index(drop=True); ys = YTR[sub]; ws = WTR[sub]
gs = age_band_idx(Xs["age"].to_numpy(float))
gte = age_band_idx(AGE_TE)

recs = []
pte_raw = np.zeros(len(XTE)); pte_pl_w = np.zeros(len(XTE)); pte_pl_u = np.zeros(len(XTE))
pte_iso = np.zeros(len(XTE))
for L in [0, 1, 2]:
    m = gs == L
    Xl = Xs.loc[m].reset_index(drop=True); yl = ys[m]; wl = ws[m]
    mte = gte == L
    (a,), _ = fit_predict(Xl, yl, [XTE.loc[mte]])
    oo = np.zeros(m.sum())
    for tr, va in StratifiedKFold(3, shuffle=True, random_state=SEED).split(Xl, yl):
        (z,), _ = fit_predict(Xl.iloc[tr].reset_index(drop=True), yl[tr],
                              [Xl.iloc[va].reset_index(drop=True)])
        oo[va] = z
    pte_raw[mte] = a
    plw = LR(max_iter=1000).fit(logit(oo).reshape(-1, 1), yl, sample_weight=wl)
    plu = LR(max_iter=1000).fit(logit(oo).reshape(-1, 1), yl)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6
                             ).fit(oo, yl, sample_weight=wl)
    pte_pl_w[mte] = plw.predict_proba(logit(a).reshape(-1, 1))[:, 1]
    pte_pl_u[mte] = plu.predict_proba(logit(a).reshape(-1, 1))[:, 1]
    pte_iso[mte] = iso.predict(a)
    recs.append(dict(band=L, n=int(m.sum()),
                     prev_unw=round(float(yl.mean()), 4),
                     prev_w=round(float(np.average(yl, weights=wl)), 4),
                     W_mean=round(float(wl.mean()), 1),
                     platt_slope_w=round(float(plw.coef_[0, 0]), 4),
                     platt_int_w=round(float(plw.intercept_[0]), 4),
                     platt_slope_u=round(float(plu.coef_[0, 0]), 4),
                     platt_int_u=round(float(plu.intercept_[0]), 4),
                     mean_raw=round(float(a.mean()), 4),
                     mean_platt_w=round(float(pte_pl_w[mte].mean()), 4),
                     mean_platt_u=round(float(pte_pl_u[mte].mean()), 4),
                     mean_iso=round(float(pte_iso[mte].mean()), 4)))
D = pd.DataFrame(recs)
print(D.to_string(index=False))
print("\ninternal wAUROC  raw=%.5f  platt_w=%.5f  platt_unw=%.5f  iso=%.5f" %
      (wauc(YTE, pte_raw, WTE), wauc(YTE, pte_pl_w, WTE),
       wauc(YTE, pte_pl_u, WTE), wauc(YTE, pte_iso, WTE)))
D.to_csv(os.path.join(RES, "path6_01b_plattdiag.csv"), index=False, encoding="utf-8-sig")
