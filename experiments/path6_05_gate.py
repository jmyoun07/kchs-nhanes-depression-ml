# -*- coding: utf-8 -*-
"""
경로6 STAGE5 — (5)의 개별화 확장: '개인별 α' 밀도게이트 선형-트리 블렌딩.

동기: KCHS→NHANES 공변량 이동이 매우 크다(BMI 23.7→29.8, domain AUC 0.97).
트리는 학습 지지영역 밖에서 상수로 외삽하고, 선형모형은 매끄럽게 외삽한다.
따라서 '개인이 KCHS 지지영역에서 얼마나 벗어났는가' d(x) 에 따라
  logit = (1-α(x))·logit_LGBM + α(x)·logit_LR ,  α(x)=sigmoid((d(x)-c)/s)
로 개별화한다.

★§2.10 준수 설계: c,s 는 KCHS 내부 '꼬리 홀드아웃'(외삽 프록시)으로만 선택.
   NHANES 라벨은 물론 NHANES 공변량도 사용하지 않는다.
"""
import time, numpy as np, pandas as pd
from path6_common import *
from sklearn.linear_model import LogisticRegression
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


def _fit_all():
    (a, b, c), _ = fit_predict(XTR, YTR, [XTE, XP, XQ])
    (d, e, f), _ = fit_predict(XTR, YTR, [XTE, XP, XQ],
                               clf=LogisticRegression(C=1.0, max_iter=3000))
    return [a, b, c, d, e, f]

g_te, g_pre, g_post, l_te, l_pre, l_post = cached("full_scores", _fit_all)

# ---------------------------------------------------------------- 지지영역 이탈도 d(x)
prep = make_prep(NUM, BIN, CAT, bin_impute=True)
A_tr = np.asarray(prep.fit_transform(XTR[FEATURES]), float)
A_te = np.asarray(prep.transform(XTE[FEATURES]), float)
A_p = np.asarray(prep.transform(XP[FEATURES]), float)
A_q = np.asarray(prep.transform(XQ[FEATURES]), float)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(A_tr)
A_tr, A_te, A_p, A_q = (sc.transform(x) for x in (A_tr, A_te, A_p, A_q))

mu = A_tr.mean(0)
COV = np.cov(A_tr[np.random.RandomState(0).choice(len(A_tr), 120000, replace=False)], rowvar=False)
IC = np.linalg.pinv(COV + 1e-6 * np.eye(COV.shape[0]))


def mah(X):
    d = X - mu
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, IC, d), 0))


d_tr, d_te, d_p, d_q = mah(A_tr), mah(A_te), mah(A_p), mah(A_q)
print("Mahalanobis  KCHS med=%.2f p90=%.2f | pre med=%.2f p90=%.2f | post med=%.2f" %
      (np.median(d_tr), np.percentile(d_tr, 90), np.median(d_p),
       np.percentile(d_p, 90), np.median(d_q)))
print("NHANES pre 의 KCHS 분위수 위치: %.3f / post %.3f" %
      ((d_tr[:, None] < np.median(d_p)).mean(), (d_tr[:, None] < np.median(d_q)).mean()))

pd.DataFrame(dict(set=["kchs_tr", "kchs_te", "nh_pre", "nh_post"],
                  d_p25=[np.percentile(x, 25) for x in (d_tr, d_te, d_p, d_q)],
                  d_med=[np.median(x) for x in (d_tr, d_te, d_p, d_q)],
                  d_p75=[np.percentile(x, 75) for x in (d_tr, d_te, d_p, d_q)],
                  d_p95=[np.percentile(x, 95) for x in (d_tr, d_te, d_p, d_q)])
             ).to_csv(os.path.join(RES, "path6_05_support_diag.csv"), index=False,
                      encoding="utf-8-sig")

# ---------------------------------------------------------------- KCHS 꼬리 홀드아웃 (외삽 프록시)
# core = d 하위 80%, tail = 상위 20%. core 로만 학습 → tail 평가. (§2.10 준수: 대상 미사용)
thr = np.percentile(d_tr, 80)
core = d_tr <= thr; tail = ~core
print("core", core.sum(), "tail", tail.sum())


def _proxy():
    Xc = XTR.loc[core].reset_index(drop=True); yc = YTR[core]
    Xt = XTR.loc[tail].reset_index(drop=True)
    (gt,), _ = fit_predict(Xc, yc, [Xt])
    (lt,), _ = fit_predict(Xc, yc, [Xt], clf=LogisticRegression(C=1.0, max_iter=3000))
    return [gt, lt]

pg, pl = cached("gate_proxy", _proxy)
yt, wt, dt = YTR[tail], WTR[tail], d_tr[tail]
print("proxy tail: lgbm=%.5f  logit=%.5f" % (wauc(yt, pg, wt), wauc(yt, pl, wt)))

# 고정 α 프록시 곡선
fx = [(round(a, 2), round(wauc(yt, (1 - a) * logit(pg) + a * logit(pl), wt), 6))
      for a in np.arange(0, 1.01, 0.1)]
print("proxy fixed-α curve:", fx)
best_fix = max(fx, key=lambda t: t[1])[0]

# 게이트 파라미터 격자 (프록시로 선택)
grid = []
for c in np.percentile(d_tr, [50, 70, 80, 90, 95]):
    for s in [0.5, 1.0, 2.0, 4.0]:
        for amax in [0.3, 0.5, 0.7, 1.0]:
            a = amax / (1 + np.exp(-(dt - c) / s))
            grid.append((float(c), s, amax,
                         round(wauc(yt, (1 - a) * logit(pg) + a * logit(pl), wt), 6)))
G = pd.DataFrame(grid, columns=["c", "s", "amax", "proxy_wauc"]).sort_values(
    "proxy_wauc", ascending=False)
G.to_csv(os.path.join(RES, "path6_05_gate_grid.csv"), index=False, encoding="utf-8-sig")
print(G.head(8).to_string())
bc, bs, ba = G.iloc[0][["c", "s", "amax"]]
print("selected gate c=%.2f s=%.2f amax=%.2f (proxy=%.5f)" % (bc, bs, ba, G.iloc[0].proxy_wauc))

rows.append(report("G0_lgbm", g_te, g_pre, g_post, note="기준선"))
rows.append(report("G0_logit", l_te, l_pre, l_post, note="단순 로지스틱"))


def gated(c, s, amax, tag, compliant, note):
    def f(d, gg, ll):
        a = amax / (1 + np.exp(-(d - c) / s))
        return (1 - a) * logit(gg) + a * logit(ll)
    return report(tag, f(d_te, g_te, l_te), f(d_p, g_pre, l_pre), f(d_q, g_post, l_post),
                  compliant=compliant, note=note)


rows.append(gated(bc, bs, ba, f"G1_gate_c{bc:.1f}_s{bs}_a{ba}", True,
                  f"밀도게이트 개별화 α (KCHS 꼬리홀드아웃 프록시 선택, proxy={G.iloc[0].proxy_wauc:.5f})"))
print(rows[-1])
# 고정 α (프록시 선택)
rows.append(report(f"G2_fixed_a{best_fix:.1f}",
                   (1 - best_fix) * logit(g_te) + best_fix * logit(l_te),
                   (1 - best_fix) * logit(g_pre) + best_fix * logit(l_pre),
                   (1 - best_fix) * logit(g_post) + best_fix * logit(l_post),
                   note=f"고정 α={best_fix} (KCHS 꼬리홀드아웃 프록시 선택)"))
print(rows[-1])

# 진단: 게이트 파라미터 스윕의 외부곡선
for c, s, amax in [(np.percentile(d_tr, 50), 2.0, 1.0),
                   (np.percentile(d_tr, 80), 2.0, 1.0),
                   (np.percentile(d_tr, 90), 1.0, 1.0),
                   (np.percentile(d_tr, 70), 4.0, 0.7)]:
    rows.append(gated(c, s, amax, f"DIAG_gate_c{c:.1f}_s{s}_a{amax}", False,
                      "★진단전용: 게이트 파라미터 외부곡선"))

# ---------------------------------------------------------------- (7) 순위목표 학습 (calibration/rank-aware)
def rank_obj_model():
    A = np.asarray(prep.transform(XTR[FEATURES]), float)
    A = sc.transform(A)
    tr = lgb.Dataset(A, label=YTR, group=[len(A)])
    pr = dict(objective="lambdarank", lambdarank_truncation_level=100,
              learning_rate=0.03, num_leaves=31, bagging_fraction=0.8,
              bagging_freq=1, feature_fraction=0.8, seed=SEED, verbose=-1,
              label_gain=[0, 1], num_threads=-1)
    bst = lgb.train(pr, tr, num_boost_round=600)
    return [bst.predict(sc.transform(np.asarray(prep.transform(X[FEATURES]), float)))
            for X in (XTE, XP, XQ)]

try:
    a, b, c = rank_obj_model()
    rows.append(report("G3_lambdarank", a, b, c,
                       note="(7) 순위목표(lambdarank) 학습 — 확률 대신 순위 직접 최적화"))
    print(rows[-1])
except Exception as e:
    print("lambdarank failed:", e)

save(rows, "path6_05_gate.csv")
print("TOTAL", round(time.time() - T0, 1), "s")
