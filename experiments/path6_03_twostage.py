# -*- coding: utf-8 -*-
"""
경로6 STAGE3 — (1) 2단계 모델: 1단계 위험군 분류 → 층별 전문모델.

결합규칙이 관건이다. AUROC 는 전집단 단일 순위이므로 층 간 순서를 보존해야 한다.
  T1 stage2_only : 층 전문모델 확률 그대로
  T2 coarse_fine : logit(stage1) + β·(층내 중심화된 logit(stage2))  ← 전역순서 보존 + 국소보정
  T3 blend       : α·logit(stage1) + (1-α)·logit(stage2)
β·α·층수는 모두 KCHS 학습셋 OOF 로만 선택 (§2.10 준수).
"""
import time, numpy as np, pandas as pd
from path6_common import *
from sklearn.model_selection import StratifiedKFold

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
    from sklearn.linear_model import LogisticRegression
    (d, e, f), _ = fit_predict(XTR, YTR, [XTE, XP, XQ],
                               clf=LogisticRegression(C=1.0, max_iter=3000))
    return [a, b, c, d, e, f]

g_te, g_pre, g_post, l_te, l_pre, l_post = cached("full_scores", _fit_all)


def _oof():
    from sklearn.linear_model import LogisticRegression
    og = np.zeros(len(XTR)); ol = np.zeros(len(XTR))
    for tr, va in StratifiedKFold(3, shuffle=True, random_state=SEED).split(XTR, YTR):
        A = XTR.iloc[tr].reset_index(drop=True); B = XTR.iloc[va].reset_index(drop=True)
        (z,), _ = fit_predict(A, YTR[tr], [B]); og[va] = z
        (z,), _ = fit_predict(A, YTR[tr], [B], clf=LogisticRegression(C=1.0, max_iter=3000))
        ol[va] = z
    return [og, ol]

oof_g, oof_l = cached("oof_g_l", _oof)
rows.append(report("T0_stage1_only", g_te, g_pre, g_post, note="1단계 전역모델 = 기준선"))
print(rows[-1])


def run_twostage(nq, tag):
    """nq: 위험층 개수 (stage1 OOF 점수 분위로 분할)."""
    qs = np.quantile(oof_g, np.linspace(0, 1, nq + 1)[1:-1])
    a_tr = np.searchsorted(qs, oof_g)
    a_te = np.searchsorted(qs, g_te)
    a_p = np.searchsorted(qs, g_pre)
    a_q = np.searchsorted(qs, g_post)
    print(f"[{tag}] 층 분포 tr={np.bincount(a_tr)} te={np.bincount(a_te,minlength=nq)} "
          f"pre={np.bincount(a_p,minlength=nq)} post={np.bincount(a_q,minlength=nq)}")

    s2_te = np.zeros(len(XTE)); s2_p = np.zeros(len(XP)); s2_q = np.zeros(len(XQ))
    s2_oof = np.zeros(len(XTR))
    t0 = time.time()
    for L in range(nq):
        m = a_tr == L
        if m.sum() < 5000:
            print("  small stratum", L, m.sum())
        Xl = XTR.loc[m].reset_index(drop=True); yl = YTR[m]
        if yl.sum() < 50 or (1 - yl).sum() < 50:
            print("  degenerate stratum", L); s2_te[a_te == L] = g_te[a_te == L]
            s2_p[a_p == L] = g_pre[a_p == L]; s2_q[a_q == L] = g_post[a_q == L]
            s2_oof[m] = oof_g[m]; continue
        (b1, b2, b3), _ = fit_predict(Xl, yl, [XTE.loc[a_te == L], XP.loc[a_p == L], XQ.loc[a_q == L]])
        s2_te[a_te == L] = b1; s2_p[a_p == L] = b2; s2_q[a_q == L] = b3
        oo = np.zeros(m.sum())
        for tr, va in StratifiedKFold(3, shuffle=True, random_state=SEED).split(Xl, yl):
            (z,), _ = fit_predict(Xl.iloc[tr].reset_index(drop=True), yl[tr],
                                  [Xl.iloc[va].reset_index(drop=True)])
            oo[va] = z
        s2_oof[np.where(m)[0]] = oo
    secs = time.time() - t0

    out = [report(f"{tag}_T1_stage2only", s2_te, s2_p, s2_q,
                  note=f"{nq}개 위험층 전문모델 확률 그대로", secs=secs)]

    # T2 coarse-fine: 층내 중심화
    def center(v, a):
        z = logit(v).copy()
        for L in np.unique(a):
            m = a == L
            if m.sum(): z[m] -= z[m].mean()
        return z
    c_oof = center(s2_oof, a_tr); c_te = center(s2_te, a_te)
    c_p = center(s2_p, a_p); c_q = center(s2_q, a_q)
    bt = [(round(b, 2), round(wauc(YTR, logit(oof_g) + b * c_oof, WTR), 6))
          for b in np.arange(0, 1.51, 0.1)]
    bb = max(bt, key=lambda t: t[1])[0]
    print(f"[{tag}] β curve:", bt, "->", bb)
    out.append(report(f"{tag}_T2_coarsefine_b{bb:.1f}",
                      logit(g_te) + bb * c_te, logit(g_pre) + bb * c_p,
                      logit(g_post) + bb * c_q,
                      note=f"전역로짓 + β×층내중심화 stage2, β={bb} (KCHS OOF)"))
    for b2 in [0.3, 0.6, 1.0]:
        out.append(report(f"DIAG_{tag}_T2_b{b2:.1f}",
                          logit(g_te) + b2 * c_te, logit(g_pre) + b2 * c_p,
                          logit(g_post) + b2 * c_q, compliant=False, note="★진단전용 β곡선"))

    # T3 blend
    at = [(round(a, 2), round(wauc(YTR, a * logit(oof_g) + (1 - a) * logit(s2_oof), WTR), 6))
          for a in np.arange(0, 1.01, 0.1)]
    ba = max(at, key=lambda t: t[1])[0]
    print(f"[{tag}] α curve:", at, "->", ba)
    out.append(report(f"{tag}_T3_blend_a{ba:.1f}",
                      ba * logit(g_te) + (1 - ba) * logit(s2_te),
                      ba * logit(g_pre) + (1 - ba) * logit(s2_p),
                      ba * logit(g_post) + (1 - ba) * logit(s2_q),
                      note=f"stage1·stage2 로짓혼합 α={ba} (KCHS OOF)"))
    return out


for nq, tag in [(2, "T_2strata"), (3, "T_3strata"), (5, "T_5strata")]:
    o = run_twostage(nq, tag)
    rows += o
    for r in o: print(r)

save(rows, "path6_03_twostage.csv")
print("TOTAL", round(time.time() - T0, 1), "s")
