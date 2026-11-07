# -*- coding: utf-8 -*-
"""
경로6 STAGE8 (심화) — 외삽안전 부분선형화 (partial linearization).

동기: STAGE5 진단에서 NHANES 개인의 마할라노비스 거리 중앙값이 KCHS 93 분위였다.
      특히 BMI(23.7→29.8)처럼 대상 분포가 KCHS 학습 지지영역을 크게 벗어나면
      트리는 경계에서 '상수'로 외삽하여 정보가 소실된다.
      → 이동이 큰 연속변수는 '선형항'으로만 들어가게 하고(매끄러운 외삽),
        나머지는 트리에 맡기는 구조 하이브리드.

구현: LR(17변수 전체) 로짓을 init_score 로 고정 → LGBM 은 지정 부분집합만 사용해 잔차 학습.
선택: KCHS 꼬리 홀드아웃(마할라노비스 상위 20%) 프록시. NHANES 완전 미접촉 (§2.10 준수).
"""
import time, numpy as np, pandas as pd
from path6_common import *
from sklearn.linear_model import LogisticRegression
import lightgbm as lgbm_mod

rows = []
T0 = time.time()
CACHE = os.path.join(RES, "_p6_cache"); os.makedirs(CACHE, exist_ok=True)


def cached(name, fn):
    p = os.path.join(CACHE, name + ".npz")
    if os.path.exists(p):
        z = np.load(p); return [z[k] for k in sorted(z.files)]
    out = fn(); np.savez(p, **{f"a{i}": v for i, v in enumerate(out)}); return out


# 지지영역 이탈도 (KCHS 학습셋만 사용)
prep0 = make_prep(NUM, BIN, CAT, bin_impute=True)
A0 = np.asarray(prep0.fit_transform(XTR[FEATURES]), float)
from sklearn.preprocessing import StandardScaler
sc0 = StandardScaler().fit(A0); A0s = sc0.transform(A0)
mu = A0s.mean(0)
COV = np.cov(A0s[np.random.RandomState(0).choice(len(A0s), 120000, replace=False)], rowvar=False)
IC = np.linalg.pinv(COV + 1e-6 * np.eye(COV.shape[0]))
d_tr = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", A0s - mu, IC, A0s - mu), 0))
thr = np.percentile(d_tr, 80)
CORE, TAIL = d_tr <= thr, d_tr > thr
print("core/tail", CORE.sum(), TAIL.sum(), flush=True)


def hybrid(tree_feats, Xtr, ytr, Xlist, gamma=1.0):
    """LR(전체 17변수) init_score + LGBM(tree_feats 만) 잔차."""
    pr = make_prep(NUM, BIN, CAT, bin_impute=True)
    A = pr.fit_transform(Xtr[FEATURES]); Bs = [pr.transform(X[FEATURES]) for X in Xlist]
    lr = LogisticRegression(C=1.0, max_iter=3000).fit(A, ytr)
    b_tr = lr.decision_function(A); b_s = [lr.decision_function(b) for b in Bs]
    if not tree_feats:
        return [b for b in b_s]
    tn = [c for c in NUM if c in tree_feats]
    tb = [c for c in BIN if c in tree_feats]
    tc = [c for c in CAT if c in tree_feats]
    pr2 = make_prep(tn, tb, tc, bin_impute=True)
    f2 = tn + tb + tc
    A2 = pr2.fit_transform(Xtr[f2]); B2 = [pr2.transform(X[f2]) for X in Xlist]
    ds = lgbm_mod.Dataset(A2, label=ytr, init_score=b_tr)
    p = dict(objective="binary", learning_rate=0.03, num_leaves=31,
             bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8,
             seed=SEED, verbose=-1, num_threads=-1)
    bst = lgbm_mod.train(p, ds, num_boost_round=600)
    return [bb + gamma * bst.predict(b2, raw_score=True) for bb, b2 in zip(b_s, B2)]


# 후보 구조: 트리가 쓸 수 있는 변수 집합
HIGH_SHIFT = ["bmi", "sleep_hrs", "age"]      # 대상 분포가 KCHS 지지영역을 크게 벗어남
CANDS = {
    "C0_tree_all": FEATURES,
    "C1_no_bmi": [f for f in FEATURES if f != "bmi"],
    "C2_no_bmi_sleep": [f for f in FEATURES if f not in ("bmi", "sleep_hrs")],
    "C3_no_highshift": [f for f in FEATURES if f not in HIGH_SHIFT],
    "C4_tree_only_srh_oh": ["srh", "oral_health", "food_insecure", "employed",
                            "sex", "edu", "smoking"],
    "C5_tree_none_pureLR": [],
}

# ---- 1) 프록시 평가 (core 학습 → tail 평가), §2.10 준수
proxy = {}
for nm, tf in CANDS.items():
    def _f(tf=tf, nm=nm):
        return hybrid(tf, XTR.loc[CORE].reset_index(drop=True), YTR[CORE],
                      [XTR.loc[TAIL].reset_index(drop=True)])
    (s,) = cached("pl_proxy_" + nm, _f)
    proxy[nm] = wauc(YTR[TAIL], s, WTR[TAIL])
    print(f"  proxy[{nm}] = {proxy[nm]:.5f}", flush=True)
best = max(proxy, key=proxy.get)
print("프록시 최우수 구조:", best, round(proxy[best], 5), flush=True)

# ---- 2) 전체 학습 → 최종 외부 평가
for nm, tf in CANDS.items():
    def _f(tf=tf, nm=nm):
        return hybrid(tf, XTR, YTR, [XTE, XP, XQ])
    a, b, c = cached("pl_full_" + nm, _f)
    rows.append(report(f"P8_{nm}", a, b, c, compliant=(nm == best),
                       note=(f"★KCHS 꼬리프록시 선택 구조 (proxy={proxy[nm]:.5f})" if nm == best
                             else f"★진단전용: 구조 외부선택은 §2.10 위반 (proxy={proxy[nm]:.5f})")
                            + f" | tree_feats={len(tf)}"))
    print(rows[-1], flush=True)

# ---- 3) 선택 구조 + 밀도게이트 α 결합 (STAGE5 게이트를 하이브리드에 적용)
def _lr_only():
    return hybrid([], XTR, YTR, [XTE, XP, XQ])
l_te, l_pre, l_post = cached("pl_full_C5_tree_none_pureLR", _lr_only)
bt, bp, bq = cached("pl_full_" + best, lambda: hybrid(CANDS[best], XTR, YTR, [XTE, XP, XQ]))


def mah(X):
    A = np.asarray(prep0.transform(X[FEATURES]), float)
    A = sc0.transform(A); D = A - mu
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", D, IC, D), 0))

d_te, d_p, d_q = mah(XTE), mah(XP), mah(XQ)

# 게이트 파라미터는 core→tail 프록시로 선택
(ps,) = cached("pl_proxy_" + best, lambda: None) if False else (None,)
z = np.load(os.path.join(CACHE, "pl_proxy_" + best + ".npz")); ps = z[sorted(z.files)[0]]
z = np.load(os.path.join(CACHE, "pl_proxy_C5_tree_none_pureLR.npz")); pl_ = z[sorted(z.files)[0]]
dt = d_tr[TAIL]
grid = []
for c in np.percentile(d_tr, [50, 70, 80, 90]):
    for s in [0.5, 1.0, 2.0, 4.0]:
        for amax in [0.2, 0.3, 0.5, 0.7, 1.0]:
            a = amax / (1 + np.exp(-(dt - c) / s))
            grid.append((float(c), s, amax,
                         round(wauc(YTR[TAIL], (1 - a) * ps + a * pl_, WTR[TAIL]), 6)))
G = pd.DataFrame(grid, columns=["c", "s", "amax", "proxy_wauc"]).sort_values(
    "proxy_wauc", ascending=False)
G.to_csv(os.path.join(RES, "path6_08_gate_grid.csv"), index=False, encoding="utf-8-sig")
print(G.head(5).to_string(index=False), flush=True)
bc, bs, ba = G.iloc[0][["c", "s", "amax"]]
gf = lambda d, h, l: (1 - ba / (1 + np.exp(-(d - bc) / bs))) * h + \
                     (ba / (1 + np.exp(-(d - bc) / bs))) * l
rows.append(report(f"P8_{best}_GATED", gf(d_te, bt, l_te), gf(d_p, bp, l_pre),
                   gf(d_q, bq, l_post),
                   note=f"부분선형 하이브리드 + 밀도게이트(c={bc:.2f},s={bs},amax={ba}) "
                        f"전과정 KCHS 프록시 선택 (proxy={G.iloc[0].proxy_wauc:.5f})"))
print(str(rows[-1]).encode("ascii", "replace").decode(), flush=True)

save(rows, "path6_08_partlin.csv")
print("TOTAL", round(time.time() - T0, 1), "s")
