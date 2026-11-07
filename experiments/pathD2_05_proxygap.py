# -*- coding: utf-8 -*-
"""
pathD2_05_proxygap.py — 진단: KCHS 내부 프록시(연도/지역) 이동량 vs 실제 한→미 이동량.

경로D의 전제는 "연도·지역 간 일반화가 국가 간 전이의 대리지표"라는 것이다.
그 전제가 성립하려면 프록시 환경 간 이동이 목표 이동과 **같은 종류·비슷한 크기**여야 한다.
여기서는 세 가지를 측정한다:
  (1) 공변량 이동: 표준화 평균차(SMD) 최대값
  (2) 도메인 판별 AUC: 환경 A vs B를 공변량만으로 구분 가능한 정도
  (3) 개념 이동(concept shift): 환경별 로지스틱 계수 벡터 간 거리

주의: NHANES 공변량은 **측정(진단)에만** 사용되며 학습·선택에는 일절 쓰이지 않는다.
      (§2.10 준수. 이 스크립트는 모델을 만들지 않는다.)
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
NS = 6000


def samp(idx, n=NS):
    return np.sort(rs.choice(idx, min(n, len(idx)), replace=False))


def Zof(idx):
    return np.nan_to_num(np.asarray(d.Z[idx]), nan=0.0)


def smd_max(A, B):
    ma, mb = A.mean(0), B.mean(0)
    sd = np.sqrt((A.var(0) + B.var(0)) / 2) + 1e-9
    v = np.abs(ma - mb) / sd
    return float(v.max()), float(v.mean())


def domain_auc(A, B):
    X = np.vstack([A, B])
    yy = np.r_[np.zeros(len(A)), np.ones(len(B))]
    tr = rs.rand(len(yy)) < 0.7
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                           random_state=SEED, n_jobs=4, verbose=-1)
    m.fit(X[tr], yy[tr])
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(yy[~tr], m.predict_proba(X[~tr])[:, 1]))


def coefs(idx):
    X = Zof(idx)
    m = LogisticRegression(C=1.0, max_iter=300, solver="lbfgs")
    m.fit(X, d.y[idx])
    return m.coef_[0]


def coefs_ext(which):
    Zx = d.Zpre if which == "pre" else d.Zpost
    yx = d.ypre if which == "pre" else d.ypost
    X = np.nan_to_num(Zx, nan=0.0)
    m = LogisticRegression(C=1.0, max_iter=300, solver="lbfgs")
    m.fit(X, yx)
    return m.coef_[0]


rows = []

# ---- (1)(2) 공변량 이동 ----
# KCHS 연도쌍 (최대 이동)
yr_pairs = []
for i, a in enumerate(d.years):
    for bq in d.years[i + 1:]:
        A = Zof(samp(pool[d.year[pool] == a]))
        B = Zof(samp(pool[d.year[pool] == bq]))
        s, sm = smd_max(A, B)
        yr_pairs.append((f"{a}v{bq}", s, sm))
yr_pairs.sort(key=lambda x: -x[1])
worst_yr = yr_pairs[0]
A = Zof(samp(pool[d.year[pool] == int(worst_yr[0].split("v")[0])]))
B = Zof(samp(pool[d.year[pool] == int(worst_yr[0].split("v")[1])]))
rows.append(dict(comparison=f"KCHS year {worst_yr[0]} (최악 연도쌍)",
                 smd_max=worst_yr[1], smd_mean=worst_yr[2],
                 domain_auc=domain_auc(A, B)))
print("  year pairs done", time.time() - t0)

# KCHS 지역클러스터쌍
cl = d.strata_cluster[pool]
sc = []
for i in range(8):
    for j in range(i + 1, 8):
        A = Zof(samp(pool[cl == i]))
        B = Zof(samp(pool[cl == j]))
        s, sm = smd_max(A, B)
        sc.append((f"c{i}vc{j}", s, sm, A, B))
sc.sort(key=lambda x: -x[1])
w = sc[0]
rows.append(dict(comparison=f"KCHS strata cluster {w[0]} (최악 지역쌍)",
                 smd_max=w[1], smd_mean=w[2], domain_auc=domain_auc(w[3], w[4])))
print("  strata pairs done", time.time() - t0)

# KCHS vs NHANES
Kall = Zof(samp(pool, 12000))
for nm, Zx in [("pre", d.Zpre), ("post", d.Zpost)]:
    X = np.nan_to_num(Zx, nan=0.0)
    s, sm = smd_max(Kall, X)
    rows.append(dict(comparison=f"KCHS vs NHANES {nm} (목표 이동)",
                     smd_max=s, smd_mean=sm, domain_auc=domain_auc(Kall, X)))
print("  KCHS-NHANES done", time.time() - t0)

df1 = pd.DataFrame(rows)
print("\n=== 공변량 이동 ===")
print(df1.round(4).to_string(index=False))

# ---- (3) 개념 이동: 계수 벡터 거리 ----
cy = {int(v): coefs(samp(pool[d.year[pool] == v], 60000)) for v in d.years}
cc = {i: coefs(samp(pool[cl == i], 60000)) for i in range(8)}
cpre, cpost = coefs_ext("pre"), coefs_ext("post")
ck = coefs(samp(pool, 120000))


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


r2 = []
yv = np.array(list(cy.values()))
r2.append(dict(comparison="KCHS 연도 간 (최악쌍)",
               cosine=min(cos(yv[i], yv[j]) for i in range(9) for j in range(i + 1, 9)),
               l2=max(np.linalg.norm(yv[i] - yv[j]) for i in range(9) for j in range(i + 1, 9))))
cv_ = np.array(list(cc.values()))
r2.append(dict(comparison="KCHS 지역 간 (최악쌍)",
               cosine=min(cos(cv_[i], cv_[j]) for i in range(8) for j in range(i + 1, 8)),
               l2=max(np.linalg.norm(cv_[i] - cv_[j]) for i in range(8) for j in range(i + 1, 8))))
r2.append(dict(comparison="KCHS vs NHANES pre", cosine=cos(ck, cpre),
               l2=float(np.linalg.norm(ck - cpre))))
r2.append(dict(comparison="KCHS vs NHANES post", cosine=cos(ck, cpost),
               l2=float(np.linalg.norm(ck - cpost))))
df2 = pd.DataFrame(r2)
print("\n=== 개념 이동(로지스틱 계수 벡터) ===")
print(df2.round(4).to_string(index=False))

# 변수별 계수 비교표
cmp = pd.DataFrame(dict(col=d.cols, KCHS=ck, NHANES_pre=cpre, NHANES_post=cpost))
cmp["sign_flip_pre"] = np.sign(cmp.KCHS) != np.sign(cmp.NHANES_pre)
cmp["sign_flip_post"] = np.sign(cmp.KCHS) != np.sign(cmp.NHANES_post)
print("\n=== 변수별 계수(표준화) ===")
print(cmp.round(3).to_string(index=False))

p1 = os.path.join(RES, "pathD2_proxygap_covariate.csv")
p2 = os.path.join(RES, "pathD2_proxygap_concept.csv")
p3 = os.path.join(RES, "pathD2_coef_compare.csv")
df1.to_csv(p1, index=False, encoding="utf-8-sig")
df2.to_csv(p2, index=False, encoding="utf-8-sig")
cmp.to_csv(p3, index=False, encoding="utf-8-sig")
print("saved:", p1, p2, p3, f"[{time.time()-t0:.0f}s]")
