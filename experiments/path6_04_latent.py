# -*- coding: utf-8 -*-
"""
경로6 STAGE4 — (6) 차원축소 후 학습: PCA / 오토인코더 / 도메인불변 부분공간.

L1  PCA-k (KCHS 학습셋에서 적합)                                  §2.10 준수
L2  PCA-k (KCHS+NHANES 공변량 통합 적합)                          회색: 대상 공변량 사용
L3  도메인분리 성분 제거 (KCHS vs NHANES 판별방향 직교화)          회색: 대상 공변량 사용
L4  KCHS 연도간 불변 부분공간 (연도판별 방향 제거) — 대상 미사용   §2.10 준수 (전이 프록시)
L5  오토인코더 잠재표현 (MLP)                                      §2.10 준수
k 는 KCHS 내부 OOF 로만 선택.
"""
import time, numpy as np, pandas as pd
from path6_common import *
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

rows = []
T0 = time.time()

prep = make_prep(NUM, BIN, CAT, bin_impute=True)
A_tr = np.asarray(prep.fit_transform(XTR[FEATURES]), float)
A_te = np.asarray(prep.transform(XTE[FEATURES]), float)
A_p = np.asarray(prep.transform(XP[FEATURES]), float)
A_q = np.asarray(prep.transform(XQ[FEATURES]), float)
print("design matrix", A_tr.shape)

# 이진/더미도 표준화해야 PCA 가 의미있음
from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(A_tr)
A_tr, A_te, A_p, A_q = sc.transform(A_tr), sc.transform(A_te), sc.transform(A_p), sc.transform(A_q)
D = A_tr.shape[1]


def lgb_on(Z_tr, Z_list, y=YTR):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(**LGBM)
    m.fit(Z_tr, y)
    return [m.predict_proba(z)[:, 1] for z in Z_list]


def oof_on(Z_tr, y=YTR, folds=3):
    import lightgbm as lgb
    o = np.zeros(len(Z_tr))
    for tr, va in StratifiedKFold(folds, shuffle=True, random_state=SEED).split(Z_tr, y):
        m = lgb.LGBMClassifier(**LGBM); m.fit(Z_tr[tr], y[tr])
        o[va] = m.predict_proba(Z_tr[va])[:, 1]
    return o


# ---------------------------------------------------------------- L1 PCA (KCHS)
pca = PCA(n_components=D, random_state=SEED).fit(A_tr)
ev = np.cumsum(pca.explained_variance_ratio_)
print("PCA cum-var:", np.round(ev[:15], 3))
Z_tr_f, Z_te_f = pca.transform(A_tr), pca.transform(A_te)
Z_p_f, Z_q_f = pca.transform(A_p), pca.transform(A_q)

ks = [3, 5, 8, 12, 16, D]
oof_by_k = {}
for k in ks:
    o = oof_on(Z_tr_f[:, :k]); oof_by_k[k] = wauc(YTR, o, WTR)
    print(f"  L1 PCA k={k}  KCHS OOF={oof_by_k[k]:.5f}")
best_k = max(oof_by_k, key=oof_by_k.get)
print("L1 selected k =", best_k)
for k in ks:
    a, b, c = lgb_on(Z_tr_f[:, :k], [Z_te_f[:, :k], Z_p_f[:, :k], Z_q_f[:, :k]])
    rows.append(report(f"L1_pcaKCHS_k{k}", a, b, c, compliant=(k == best_k),
                       note=("★KCHS OOF 선택 k" if k == best_k else "★진단전용: k 외부선택은 위반")
                            + f" | OOF={oof_by_k[k]:.5f}"))
    print(rows[-1])

# ---------------------------------------------------------------- L2 PCA (KCHS+NHANES 공변량)
Apool = np.vstack([A_tr[np.random.RandomState(0).choice(len(A_tr), 50000, replace=False)],
                   A_p, A_q])
pca2 = PCA(n_components=D, random_state=SEED).fit(Apool)
Z2 = [pca2.transform(x) for x in (A_tr, A_te, A_p, A_q)]
for k in [8, 12, D]:
    a, b, c = lgb_on(Z2[0][:, :k], [Z2[1][:, :k], Z2[2][:, :k], Z2[3][:, :k]])
    rows.append(report(f"L2_pcaPOOL_k{k}", a, b, c, compliant=False,
                       note="★회색: NHANES 공변량으로 PCA 적합(라벨 미사용). 배포시 대상분포 필요"))
    print(rows[-1])

# ---------------------------------------------------------------- L3 도메인분리 방향 제거
sub = np.random.RandomState(1).choice(len(A_tr), 60000, replace=False)
Xd = np.vstack([A_tr[sub], A_p, A_q])
yd = np.r_[np.zeros(len(sub)), np.ones(len(A_p) + len(A_q))]
dlr = LogisticRegression(max_iter=3000).fit(Xd, yd)
w = dlr.coef_.ravel(); w = w / np.linalg.norm(w)
print("domain AUC:", round(wauc(yd, dlr.decision_function(Xd), np.ones(len(yd))), 4))


def deflate(X, vecs):
    Xo = X.copy()
    for v in vecs:
        Xo = Xo - np.outer(Xo @ v, v)
    return Xo

for nv in [1, 2]:
    vecs = [w]
    if nv == 2:
        Rtr = deflate(A_tr[sub], [w]); Rp = deflate(np.vstack([A_p, A_q]), [w])
        d2 = LogisticRegression(max_iter=3000).fit(np.vstack([Rtr, Rp]), yd)
        v2 = d2.coef_.ravel(); v2 -= (v2 @ w) * w; v2 /= np.linalg.norm(v2)
        vecs.append(v2)
    a, b, c = lgb_on(deflate(A_tr, vecs),
                     [deflate(A_te, vecs), deflate(A_p, vecs), deflate(A_q, vecs)])
    rows.append(report(f"L3_domain_deflate_{nv}", a, b, c, compliant=False,
                       note="★회색: KCHS-NHANES 판별방향 제거(대상 공변량 사용, 라벨 미사용)"))
    print(rows[-1])

# ---------------------------------------------------------------- L4 KCHS 연도 판별방향 제거 (준수)
yr = K["year"].to_numpy()[ITR]
early = np.isin(yr, [2017, 2018, 2019]).astype(int)
late = np.isin(yr, [2022, 2023, 2024, 2025]).astype(int)
m4 = (early + late) > 0
ylr = LogisticRegression(max_iter=3000).fit(A_tr[m4], late[m4])
wy = ylr.coef_.ravel(); wy /= np.linalg.norm(wy)
for nv in [1]:
    a, b, c = lgb_on(deflate(A_tr, [wy]),
                     [deflate(A_te, [wy]), deflate(A_p, [wy]), deflate(A_q, [wy])])
    rows.append(report(f"L4_yeardir_deflate_{nv}", a, b, c, compliant=True,
                       note="KCHS 연도(초기vs후기) 판별방향 제거 — 대상 미사용, 전이 프록시"))
    print(rows[-1])

# ---------------------------------------------------------------- L5 오토인코더
try:
    from sklearn.neural_network import MLPRegressor
    ss = np.random.RandomState(2).choice(len(A_tr), 120000, replace=False)
    for h in [6, 10]:
        ae = MLPRegressor(hidden_layer_sizes=(32, h, 32), activation="tanh",
                          max_iter=60, random_state=SEED, early_stopping=True)
        ae.fit(A_tr[ss], A_tr[ss])

        def encode(X):
            Z = np.tanh(X @ ae.coefs_[0] + ae.intercepts_[0])
            return np.tanh(Z @ ae.coefs_[1] + ae.intercepts_[1])
        Ztr, Zte, Zp, Zq = encode(A_tr), encode(A_te), encode(A_p), encode(A_q)
        o = oof_on(Ztr)
        a, b, c = lgb_on(Ztr, [Zte, Zp, Zq])
        rows.append(report(f"L5_autoenc_h{h}", a, b, c,
                           note=f"오토인코더 잠재 {h}차원 | KCHS OOF={wauc(YTR,o,WTR):.5f}"))
        print(rows[-1])
except Exception as e:
    print("AE failed:", e)

save(rows, "path6_04_latent.csv")
print("TOTAL", round(time.time() - T0, 1), "s")
