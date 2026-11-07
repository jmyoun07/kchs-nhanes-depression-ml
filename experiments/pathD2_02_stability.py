# -*- coding: utf-8 -*-
"""
pathD2_02_stability.py — 기법(3)(5): 연도·지역 간 특징 안정성 분석.

KCHS 내부만 사용. 산출:
  - 연도별/지역클러스터별 표준화 로지스틱 계수 -> 부호 일관성, 변동계수(CV)
  - 부트스트랩(연도별) LGBM gain 중요도의 순위 안정성
  - 단조제약 부호(mono sign) 후보 도출
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathD2_engine import data, RES, SEED
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

t0 = time.time()
d = data()
pool = d.itr
rs = np.random.RandomState(SEED)

N_PER = 120000


def fit_logit(idx):
    X = np.nan_to_num(np.asarray(d.Z[np.sort(idx)]), nan=0.0)
    y = d.y[np.sort(idx)]
    m = LogisticRegression(C=1.0, max_iter=300, solver="lbfgs", n_jobs=5)
    m.fit(X, y)
    return m.coef_[0]


def fit_gain(idx):
    idx = np.sort(idx)
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                           random_state=SEED, n_jobs=5, verbose=-1)
    m.fit(np.asarray(d.Z[idx]), d.y[idx])
    g = m.booster_.feature_importance("gain")
    return g / g.sum()


# ---------- 1) 연도별 ----------
coef_y, gain_y = {}, {}
for yv in d.years:
    ix = pool[d.year[pool] == yv]
    ix = rs.choice(ix, min(N_PER, len(ix)), replace=False)
    coef_y[int(yv)] = fit_logit(ix)
    gain_y[int(yv)] = fit_gain(ix)
    print(f"  year {yv} done [{time.time()-t0:.0f}s]")

# ---------- 2) 지역클러스터별 (50개 중 12개 표본) ----------
coef_s = {}
cl = d.strata_cluster[pool]
for c in range(12):
    ix = pool[cl == c]
    if len(ix) < 20000:
        continue
    ix = rs.choice(ix, min(N_PER, len(ix)), replace=False)
    coef_s[c] = fit_logit(ix)
print(f"  strata clusters done [{time.time()-t0:.0f}s]")

# ---------- 3) 연도별 부트스트랩 (안정성 선택) ----------
B = 5
coef_b = []
for yv in d.years:
    ix0 = pool[d.year[pool] == yv]
    for bnum in range(B):
        ix = np.random.RandomState(1000 * bnum + int(yv)).choice(
            ix0, min(60000, len(ix0)), replace=True)
        coef_b.append(fit_logit(ix))
coef_b = np.array(coef_b)
print(f"  bootstrap done [{time.time()-t0:.0f}s]")

# ---------- 요약 ----------
CY = np.array([coef_y[int(v)] for v in d.years])
CS = np.array([coef_s[c] for c in sorted(coef_s)])
GY = np.array([gain_y[int(v)] for v in d.years])

rows = []
for j, c in enumerate(d.cols):
    m_y, s_y = CY[:, j].mean(), CY[:, j].std()
    m_s, s_s = CS[:, j].mean(), CS[:, j].std()
    m_b, s_b = coef_b[:, j].mean(), coef_b[:, j].std()
    rows.append(dict(
        col=c,
        coef_year_mean=m_y, coef_year_sd=s_y,
        coef_year_cv=abs(s_y / m_y) if abs(m_y) > 1e-9 else np.nan,
        sign_consistency_year=float(np.mean(np.sign(CY[:, j]) == np.sign(m_y))),
        coef_strata_sd=s_s,
        coef_strata_cv=abs(s_s / m_s) if abs(m_s) > 1e-9 else np.nan,
        sign_consistency_strata=float(np.mean(np.sign(CS[:, j]) == np.sign(m_s))),
        boot_sd=s_b,
        boot_sign_consistency=float(np.mean(np.sign(coef_b[:, j]) == np.sign(m_b))),
        # 안정성 t비: |평균| / (연도간 sd) — 클수록 안정
        stability_t=abs(m_y) / (s_y + 1e-9),
        gain_mean=GY[:, j].mean(),
        gain_cv=GY[:, j].std() / (GY[:, j].mean() + 1e-12),
        mono_sign=int(np.sign(m_y)),
    ))

df = pd.DataFrame(rows).sort_values("stability_t", ascending=False)
p = os.path.join(RES, "pathD2_feature_stability.csv")
df.to_csv(p, index=False, encoding="utf-8-sig")
print(df[["col", "coef_year_mean", "coef_year_sd", "stability_t",
          "sign_consistency_year", "sign_consistency_strata",
          "gain_mean", "gain_cv"]].to_string(index=False))
print("saved:", p, f"[{time.time()-t0:.0f}s]")
