# -*- coding: utf-8 -*-
"""
경로6 STAGE9 — 최종 후보의 기준선 대비 ΔAUROC 에 대한 복합표본(설계기반) 쌍대 부트스트랩.

NHANES 설계(strata 내 PSU 재추출)로 B회 재표본 → 동일 표본에서 두 모델의 가중 AUROC 차이를
계산하여 Δ의 95% 백분위 신뢰구간을 얻는다. 쌍대이므로 절대 AUROC CI(±0.024)보다 훨씬 좁다.
"""
import numpy as np, pandas as pd, os, time
from path6_common import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

T0 = time.time()
CACHE = os.path.join(RES, "_p6_cache")
z = np.load(os.path.join(CACHE, "full_scores.npz"))
g_te, g_pre, g_post, l_te, l_pre, l_post = [z[k] for k in sorted(z.files)]

# 마할라노비스 거리 재구성 (STAGE5 와 동일)
prep = make_prep(NUM, BIN, CAT, bin_impute=True)
A_tr = np.asarray(prep.fit_transform(XTR[FEATURES]), float)
sc = StandardScaler().fit(A_tr); A_tr = sc.transform(A_tr)
mu = A_tr.mean(0)
COV = np.cov(A_tr[np.random.RandomState(0).choice(len(A_tr), 120000, replace=False)], rowvar=False)
IC = np.linalg.pinv(COV + 1e-6 * np.eye(COV.shape[0]))


def mah(X):
    A = sc.transform(np.asarray(prep.transform(X[FEATURES]), float))
    D = A - mu
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", D, IC, D), 0))

d_tr = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", A_tr - mu, IC, A_tr - mu), 0))
d_p, d_q = mah(XP), mah(XQ)
C_GATE, S_GATE, A_GATE = float(np.percentile(d_tr, 50)), 0.5, 0.3   # STAGE5 프록시 선택값


def gate(d, gg, ll):
    a = A_GATE / (1 + np.exp(-(d - C_GATE) / S_GATE))
    return (1 - a) * logit(gg) + a * logit(ll)


CANDS = {
    "BASE_lgbm": (g_pre, g_post),
    "G1_gate": (gate(d_p, g_pre, l_pre), gate(d_q, g_post, l_post)),
    "G2_fixed_a0.4": (0.6 * logit(g_pre) + 0.4 * logit(l_pre),
                      0.6 * logit(g_post) + 0.4 * logit(l_post)),
    "B5_blend_a0.9": (0.9 * logit(g_pre) + 0.1 * logit(l_pre),
                      0.9 * logit(g_post) + 0.1 * logit(l_post)),
    "LOGIT_only": (l_pre, l_post),
}
for k, (a, b) in CANDS.items():
    print(f"{k:16s} pre={wauc(YP,a,WP):.6f}  post={wauc(YQ,b,WQ):.6f}")

B = 2000
rows = []
for setname, df, y, w, sc_map in [("pre", PRE, YP, WP, 0), ("post", POST, YQ, WQ, 1)]:
    st = df["strata"].astype(str).to_numpy()
    psu = df["psu"].astype(str).to_numpy()
    key = np.array([a + "|" + b for a, b in zip(st, psu)])
    ukey, kidx = np.unique(key, return_inverse=True)
    ustr = np.array([u.split("|")[0] for u in ukey])
    groups = {s: np.where(ustr == s)[0] for s in np.unique(ustr)}
    memb = [np.where(kidx == j)[0] for j in range(len(ukey))]
    rs = np.random.RandomState(SEED)
    base_scores = CANDS["BASE_lgbm"][sc_map]
    deltas = {k: [] for k in CANDS if k != "BASE_lgbm"}
    for b in range(B):
        pick = []
        for s, js in groups.items():
            n = len(js)
            sel = js[rs.randint(0, n, n)] if n > 1 else js
            for j in sel:
                pick.append(memb[j])
        idx = np.concatenate(pick)
        yb, wb = y[idx], w[idx]
        if yb.sum() < 5 or (1 - yb).sum() < 5:
            continue
        a0 = wauc(yb, base_scores[idx], wb)
        for k in deltas:
            deltas[k].append(wauc(yb, CANDS[k][sc_map][idx], wb) - a0)
    for k, v in deltas.items():
        v = np.array(v)
        rows.append(dict(model=k, set=setname, n_boot=len(v),
                         delta=round(float(wauc(y, CANDS[k][sc_map], w) -
                                           wauc(y, base_scores, w)), 6),
                         lo=round(float(np.percentile(v, 2.5)), 6),
                         hi=round(float(np.percentile(v, 97.5)), 6),
                         p_gt0=round(float((v > 0).mean()), 4),
                         sig=("유의" if np.percentile(v, 2.5) > 0 or
                              np.percentile(v, 97.5) < 0 else "n.s.")))
        print(rows[-1], flush=True)

D = pd.DataFrame(rows)
D.to_csv(os.path.join(RES, "path6_09_paired_boot.csv"), index=False, encoding="utf-8-sig")
print("saved. TOTAL", round(time.time() - T0, 1), "s")
