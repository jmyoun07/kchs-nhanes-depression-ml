# -*- coding: utf-8 -*-
"""
경로6 STAGE2 — (4) 잔차학습 (LR base + tree residual), (5) 선형-트리 로짓 블렌딩.

α·shrink 는 KCHS 학습셋 OOF(3-fold) 가중 AUROC 로만 결정 (§2.10 준수).
외부 전곡선은 '진단 전용'으로만 기록한다.
"""
import time, numpy as np, pandas as pd
from path6_common import *
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

rows = []
T0 = time.time()
CACHE = os.path.join(RES, "_p6_cache")
os.makedirs(CACHE, exist_ok=True)


def cached(name, fn):
    p = os.path.join(CACHE, name + ".npz")
    if os.path.exists(p):
        z = np.load(p)
        return [z[k] for k in sorted(z.files)]
    out = fn()
    np.savez(p, **{f"a{i}": v for i, v in enumerate(out)})
    return out


PREP_NUM = NUM  # 선형모형은 결측 대체 필요
def lr_model(C=1.0):
    return LogisticRegression(C=C, max_iter=3000, solver="lbfgs")


# ---------------------------------------------------------------- 기본 점수 (전체적합)
def _fit_all():
    (a, b, c), _ = fit_predict(XTR, YTR, [XTE, XP, XQ])                       # LGBM
    (d, e, f), _ = fit_predict(XTR, YTR, [XTE, XP, XQ], clf=lr_model())       # LR
    return [a, b, c, d, e, f]

g_te, g_pre, g_post, l_te, l_pre, l_post = cached("full_scores", _fit_all)
rows.append(report("B0_lgbm", g_te, g_pre, g_post, note="기준선 LGBM"))
rows.append(report("B0_logit", l_te, l_pre, l_post, note="단순 로지스틱 17변수"))
for r in rows: print(r)


# ---------------------------------------------------------------- OOF (α 선택용)
def _oof():
    og = np.zeros(len(XTR)); ol = np.zeros(len(XTR))
    for tr, va in StratifiedKFold(3, shuffle=True, random_state=SEED).split(XTR, YTR):
        A = XTR.iloc[tr].reset_index(drop=True); B = XTR.iloc[va].reset_index(drop=True)
        (z,), _ = fit_predict(A, YTR[tr], [B]); og[va] = z
        (z,), _ = fit_predict(A, YTR[tr], [B], clf=lr_model()); ol[va] = z
    return [og, ol]

oof_g, oof_l = cached("oof_g_l", _oof)
print("OOF wauc  lgbm=%.5f  logit=%.5f" % (wauc(YTR, oof_g, WTR), wauc(YTR, oof_l, WTR)))


# ---------------------------------------------------------------- (5) 로짓 블렌딩
tab = []
for al in np.arange(0, 1.001, 0.05):
    s = al * logit(oof_g) + (1 - al) * logit(oof_l)
    tab.append((round(al, 2), round(wauc(YTR, s, WTR), 6)))
best_al = max(tab, key=lambda t: t[1])[0]
print("blend α curve (KCHS OOF):", tab)
print("selected α =", best_al)

for al in [best_al] + [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0]:
    bt = al * logit(g_te) + (1 - al) * logit(l_te)
    bp = al * logit(g_pre) + (1 - al) * logit(l_pre)
    bq = al * logit(g_post) + (1 - al) * logit(l_post)
    sel = (al == best_al)
    rows.append(report(f"B5_blend_a{al:.2f}", bt, bp, bq,
                       compliant=sel,
                       note=("★KCHS OOF 선택값" if sel else "★진단전용: 외부로 α 고르면 §2.10 위반")))
    print(rows[-1])

# 순위평균 블렌딩(로짓 대신 rank)
for al in [0.5, best_al]:
    bt = al * zrank(g_te) + (1 - al) * zrank(l_te)
    bp = al * zrank(g_pre) + (1 - al) * zrank(l_pre)
    bq = al * zrank(g_post) + (1 - al) * zrank(l_post)
    rows.append(report(f"B5r_rankblend_a{al:.2f}", bt, bp, bq, compliant=(al == best_al),
                       note="순위평균 블렌딩 (대상 내 순위 사용 → transductive 주의)" ))
    print(rows[-1])


# ---------------------------------------------------------------- (4) 잔차학습
# LR 로짓을 init_score 로 주고 LGBM 이 잔차만 학습. shrink γ 로 트리 기여 축소.
import lightgbm as lgbm_mod

def residual_scores(depth_params, tag, Xtr, ytr, Xlist, lr_C=1.0):
    prep = make_prep(NUM, BIN, CAT, bin_impute=True)
    A = prep.fit_transform(Xtr[FEATURES])
    Bs = [prep.transform(X[FEATURES]) for X in Xlist]
    lr = lr_model(lr_C).fit(A, ytr)
    base_tr = lr.decision_function(A)
    base_s = [lr.decision_function(b) for b in Bs]
    ds = lgbm_mod.Dataset(A, label=ytr, init_score=base_tr)
    pr = dict(objective="binary", learning_rate=0.03, num_leaves=31,
              bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8,
              seed=SEED, verbose=-1, num_threads=-1)
    pr.update(depth_params)
    bst = lgbm_mod.train(pr, ds, num_boost_round=600)
    res_s = [bst.predict(b, raw_score=True) for b in Bs]
    return base_s, res_s


def _resid_full():
    bs, rs = residual_scores({}, "d31", XTR, YTR, [XTE, XP, XQ])
    return bs + rs

rb_te, rb_pre, rb_post, rr_te, rr_pre, rr_post = cached("resid_full", _resid_full)


def _resid_oof():
    ob = np.zeros(len(XTR)); orr = np.zeros(len(XTR))
    for tr, va in StratifiedKFold(3, shuffle=True, random_state=SEED).split(XTR, YTR):
        A = XTR.iloc[tr].reset_index(drop=True); B = XTR.iloc[va].reset_index(drop=True)
        bs, rs = residual_scores({}, "d31", A, YTR[tr], [B])
        ob[va] = bs[0]; orr[va] = rs[0]
    return [ob, orr]

oob, oorr = cached("resid_oof", _resid_oof)

gtab = []
for gm in np.arange(0, 1.001, 0.1):
    s = oob + gm * oorr
    gtab.append((round(gm, 2), round(wauc(YTR, s, WTR), 6)))
best_gm = max(gtab, key=lambda t: t[1])[0]
print("residual shrink γ curve (KCHS OOF):", gtab, "-> γ =", best_gm)

for gm in [best_gm] + [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    st = rb_te + gm * rr_te; sp = rb_pre + gm * rr_pre; sq = rb_post + gm * rr_post
    sel = (gm == best_gm)
    rows.append(report(f"B4_resid_g{gm:.1f}", st, sp, sq, compliant=sel,
                       note=("★KCHS OOF 선택 γ" if sel else "★진단전용: γ 외부선택은 §2.10 위반")))
    print(rows[-1])

# 얕은 잔차 트리 (전이 안정성 ↑ 기대)
for dp, nm in [({"num_leaves": 7, "max_depth": 3}, "shallow_d3"),
               ({"num_leaves": 15, "max_depth": 4}, "shallow_d4")]:
    def _f(dp=dp):
        bs, rs = residual_scores(dp, nm, XTR, YTR, [XTE, XP, XQ])
        return bs + rs
    b1, b2, b3, s1, s2, s3 = cached("resid_" + nm, _f)

    def _fo(dp=dp):
        ob = np.zeros(len(XTR)); orr = np.zeros(len(XTR))
        for tr, va in StratifiedKFold(3, shuffle=True, random_state=SEED).split(XTR, YTR):
            A = XTR.iloc[tr].reset_index(drop=True); B = XTR.iloc[va].reset_index(drop=True)
            bs, rs = residual_scores(dp, nm, A, YTR[tr], [B])
            ob[va] = bs[0]; orr[va] = rs[0]
        return [ob, orr]
    o1, o2 = cached("resid_oof_" + nm, _fo)
    gt = [(round(g, 2), round(wauc(YTR, o1 + g * o2, WTR), 6)) for g in np.arange(0, 1.001, 0.1)]
    bg = max(gt, key=lambda t: t[1])[0]
    print(f"[{nm}] γ curve:", gt, "->", bg)
    rows.append(report(f"B4_resid_{nm}_g{bg:.1f}", b1 + bg * s1, b2 + bg * s2, b3 + bg * s3,
                       note=f"얕은 잔차트리 {nm}, γ={bg} (KCHS OOF 선택)"))
    print(rows[-1])
    for g2 in [0.3, 0.6, 1.0]:
        rows.append(report(f"DIAG_B4_{nm}_g{g2:.1f}", b1 + g2 * s1, b2 + g2 * s2, b3 + g2 * s3,
                           compliant=False, note="★진단전용 γ 곡선"))

save(rows, "path6_02_blend.csv")
pd.DataFrame(tab, columns=["alpha", "kchs_oof_wauc"]).to_csv(
    os.path.join(RES, "path6_02_alpha_curve.csv"), index=False, encoding="utf-8-sig")
print("TOTAL", round(time.time() - T0, 1), "s")
