# -*- coding: utf-8 -*-
"""
경로6 STAGE10 (신규) — 밀도게이트의 '외삽 목표모형' 재선택.

동기: 기존 게이트(G1)는 고Mahalanobis(=NHANES가 위치한) 영역에서 딥트리 로짓을
      선형(LR) 로짓 쪽으로 개별화 이동시켜 이득을 봤다(post 유의). 그러나 ORACLE
      진단에서 NHANES 자체 최적 학습기는 '얕은 LGBM'(pooled 0.7786)이었다.
      → 외삽 목표를 LR 대신 '얕은 트리'로 두면 더 나을 수 있는가?를 실측.

★§2.10 준수: 게이트 파라미터(c,s,amax)와 '목표모형(LR vs shallow)'을 모두
   KCHS 내부 '꼬리 홀드아웃'(core 하위80%로 학습→tail 상위20% 평가) 프록시로만 선택.
   NHANES 라벨/공변량 일절 미사용. 최종 NHANES는 1회 평가.
"""
import time, numpy as np, pandas as pd
from path6_common import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

rows = []
T0 = time.time()
CACHE = os.path.join(RES, "_p6_cache"); os.makedirs(CACHE, exist_ok=True)


def cached(name, fn):
    p = os.path.join(CACHE, name + ".npz")
    if os.path.exists(p):
        z = np.load(p); return [z[k] for k in sorted(z.files)]
    out = fn(); np.savez(p, **{f"a{i}": v for i, v in enumerate(out)}); return out


# ---- 전역 딥LGBM + LR (기존 캐시 재사용) ----
def _fit_all():
    (a, b, c), _ = fit_predict(XTR, YTR, [XTE, XP, XQ])
    (d, e, f), _ = fit_predict(XTR, YTR, [XTE, XP, XQ],
                               clf=LogisticRegression(C=1.0, max_iter=3000))
    return [a, b, c, d, e, f]

g_te, g_pre, g_post, l_te, l_pre, l_post = cached("full_scores", _fit_all)

# ---- 전역 얕은 LGBM (depth2, H2_depth2 최적 OOF 설정) ----
SHALLOW = dict(max_depth=2, num_leaves=4, n_estimators=2000, learning_rate=0.05)

def _fit_shallow():
    (a, b, c), _ = fit_predict(XTR, YTR, [XTE, XP, XQ], clf=new_lgbm(**SHALLOW))
    return [a, b, c]

s_te, s_pre, s_post = cached("shallow_scores", _fit_shallow)
print("shallow global ext:", round(wauc(YP, s_pre, WP), 5), round(wauc(YQ, s_post, WQ), 5))

# ---- Mahalanobis 지지영역 이탈도 d(x) (게이트 축) ----
prep = make_prep(NUM, BIN, CAT, bin_impute=True)
A_tr = np.asarray(prep.fit_transform(XTR[FEATURES]), float)
A_te = np.asarray(prep.transform(XTE[FEATURES]), float)
A_p = np.asarray(prep.transform(XP[FEATURES]), float)
A_q = np.asarray(prep.transform(XQ[FEATURES]), float)
sc = StandardScaler().fit(A_tr)
A_tr, A_te, A_p, A_q = (sc.transform(x) for x in (A_tr, A_te, A_p, A_q))
mu = A_tr.mean(0)
COV = np.cov(A_tr[np.random.RandomState(0).choice(len(A_tr), 120000, replace=False)], rowvar=False)
IC = np.linalg.pinv(COV + 1e-6 * np.eye(COV.shape[0]))
def mah(X):
    d = X - mu
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, IC, d), 0))
d_tr, d_te, d_p, d_q = mah(A_tr), mah(A_te), mah(A_p), mah(A_q)

# ---- KCHS 꼬리 홀드아웃 프록시: core(하위80%) 학습 → tail(상위20%) 평가 ----
thr = np.percentile(d_tr, 80)
core = d_tr <= thr; tail = ~core

def _proxy_scores():
    Xc = XTR.loc[core].reset_index(drop=True); yc = YTR[core]
    Xt = XTR.loc[tail].reset_index(drop=True)
    (gt,), _ = fit_predict(Xc, yc, [Xt])                                  # deep
    (lt,), _ = fit_predict(Xc, yc, [Xt], clf=LogisticRegression(C=1.0, max_iter=3000))  # LR
    (st,), _ = fit_predict(Xc, yc, [Xt], clf=new_lgbm(**SHALLOW))         # shallow
    return [gt, lt, st]

pg, pl, ps = cached("gate_proxy_shallow", _proxy_scores)
yt, wt, dt = YTR[tail], WTR[tail], d_tr[tail]
print("proxy tail wauc: deep=%.5f logit=%.5f shallow=%.5f" %
      (wauc(yt, pg, wt), wauc(yt, pl, wt), wauc(yt, ps, wt)))

# ---- 게이트 격자 탐색 (목표모형 = LR 또는 shallow), 프록시로만 선택 ----
cand = np.percentile(d_tr, [50, 70, 80, 90, 95])
def gate_grid(proxy_target):
    G = []
    for c in cand:
        for s in [0.5, 1.0, 2.0, 4.0]:
            for amax in [0.3, 0.5, 0.7, 1.0]:
                a = amax / (1 + np.exp(-(dt - c) / s))
                blended = (1 - a) * logit(pg) + a * logit(proxy_target)
                G.append((float(c), s, amax, round(wauc(yt, blended, wt), 6)))
    return pd.DataFrame(G, columns=["c", "s", "amax", "proxy"]).sort_values("proxy", ascending=False)

def apply_gate(c, s, amax, tgt_te, tgt_pre, tgt_post, tag, note):
    def f(d, gg, tt):
        a = amax / (1 + np.exp(-(d - c) / s))
        return (1 - a) * logit(gg) + a * logit(tt)
    return report(tag, f(d_te, g_te, tgt_te), f(d_p, g_pre, tgt_pre), f(d_q, g_post, tgt_post),
                  compliant=True, note=note)

rows.append(report("G0_lgbm", g_te, g_pre, g_post, note="기준선 딥LGBM"))

# 목표=LR (기존 재확인)
GL = gate_grid(pl); cL, sL, aL = GL.iloc[0][["c", "s", "amax"]]
rows.append(apply_gate(cL, sL, aL, l_te, l_pre, l_post,
            f"G10L_gate2LR_c{cL:.1f}_s{sL}_a{aL}",
            f"게이트 목표=LR (프록시={GL.iloc[0].proxy:.5f})"))
print(rows[-1])

# 목표=shallow (신규)
GS = gate_grid(ps); cS, sS, aS = GS.iloc[0][["c", "s", "amax"]]
rows.append(apply_gate(cS, sS, aS, s_te, s_pre, s_post,
            f"G10S_gate2shallow_c{cS:.1f}_s{sS}_a{aS}",
            f"★신규 게이트 목표=얕은트리 (프록시={GS.iloc[0].proxy:.5f})"))
print(rows[-1])

# 프록시가 목표모형을 스스로 선택 (LR vs shallow)
choose = "shallow" if GS.iloc[0].proxy > GL.iloc[0].proxy else "LR"
print("프록시 선택 목표모형:", choose,
      "| LR proxy=%.5f shallow proxy=%.5f" % (GL.iloc[0].proxy, GS.iloc[0].proxy))

# 진단전용: 목표=shallow 의 외부 α 곡선 (고정 α), §2.10 위반이므로 표시만
for a in [0.2, 0.4, 0.6, 0.8]:
    rows.append(report(f"DIAG_fixed_gate2shallow_a{a}",
                (1 - a) * logit(g_te) + a * logit(s_te),
                (1 - a) * logit(g_pre) + a * logit(s_pre),
                (1 - a) * logit(g_post) + a * logit(s_post),
                compliant=False, note="★진단전용: 얕은트리 고정혼합 외부곡선(α 외부선택은 위반)"))

save(rows, "path6_10_gate_shallow.csv")
print("TOTAL", round(time.time() - T0, 1), "s")
