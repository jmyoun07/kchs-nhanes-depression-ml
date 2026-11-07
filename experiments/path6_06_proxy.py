# -*- coding: utf-8 -*-
"""
경로6 STAGE6 — §2.10 준수 '이동 프록시'로 구조/혼합비를 선택할 수 있는가.

문제의식: STAGE2 에서 선형-트리 로짓혼합의 외부 최적 α 는 0.6~0.8 부근인데
KCHS 내부 iid OOF 는 α=0.9 를 고른다. iid CV 는 분포이동을 볼 수 없기 때문이다.
NHANES 를 쓰지 않고도 이동에 강한 α 를 고를 수 있는 KCHS 내부 프록시가 있는가?

프록시 4종 (모두 KCHS 내부, NHANES 완전 미접촉 → §2.10 준수)
  PX1 iid OOF          : 기존 기준 (이동 없음)
  PX2 LOYO             : 연도 하나를 빼고 학습 → 그 연도 평가 (시간 이동)
  PX3 region holdout   : strata 군집 하나를 빼고 학습 → 그 군집 평가 (지역 이동)
  PX4 age holdout      : 40세 이상으로 학습 → 20-39세 평가 (연령구성 이동)
  PX5 tail holdout     : 마할라노비스 하위 80%로 학습 → 상위 20% 평가 (외삽)
각 프록시가 고른 α 를 최종모델에 적용해 외부 성능을 1회 보고한다.
"""
import time, numpy as np, pandas as pd
from path6_common import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

T0 = time.time()
CACHE = os.path.join(RES, "_p6_cache"); os.makedirs(CACHE, exist_ok=True)
ALPHAS = np.round(np.arange(0, 1.001, 0.05), 2)


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


def _oof():
    og = np.zeros(len(XTR)); ol = np.zeros(len(XTR))
    for tr, va in StratifiedKFold(3, shuffle=True, random_state=SEED).split(XTR, YTR):
        A = XTR.iloc[tr].reset_index(drop=True); B = XTR.iloc[va].reset_index(drop=True)
        (z,), _ = fit_predict(A, YTR[tr], [B]); og[va] = z
        (z,), _ = fit_predict(A, YTR[tr], [B], clf=LogisticRegression(C=1.0, max_iter=3000))
        ol[va] = z
    return [og, ol]

oof_g, oof_l = cached("oof_g_l", _oof)

YR = YEAR_TR
ST = STRATA_TR
reg = (pd.Series(ST).astype(str).str[:2].astype(int) % 5).to_numpy()

# 계산량 절감: 프록시 적합은 학습셋 300k 부분표본으로 (상대 순위 판단 목적)
_rs = np.random.RandomState(SEED)
SUB = np.sort(_rs.choice(len(XTR), 300_000, replace=False))
SUBM = np.zeros(len(XTR), bool); SUBM[SUB] = True


def alpha_curve(y, w, pg, pl):
    return np.array([wauc(y, a * logit(pg) + (1 - a) * logit(pl), w) for a in ALPHAS])


def holdout_curve(mask_tr, mask_va, tag):
    """mask_tr/mask_va: XTR 내부 불리언. 반환 α곡선(가중 AUROC)."""
    mask_tr = mask_tr & SUBM
    mask_va = mask_va & SUBM

    def _f():
        A = XTR.loc[mask_tr].reset_index(drop=True); ya = YTR[mask_tr]
        B = XTR.loc[mask_va].reset_index(drop=True)
        (pg,), _ = fit_predict(A, ya, [B])
        (pl,), _ = fit_predict(A, ya, [B], clf=LogisticRegression(C=1.0, max_iter=3000))
        return [pg, pl]
    pg, pl = cached("px6_" + tag, _f)
    return alpha_curve(YTR[mask_va], WTR[mask_va], pg, pl)


curves = {}

# PX1 iid OOF
curves["PX1_iid_OOF"] = alpha_curve(YTR, WTR, oof_g, oof_l)
print("PX1 done", round(time.time() - T0, 1))

# PX2 LOYO
loyo = []
for yv in sorted(set(YR.tolist())):
    c = holdout_curve(YR != yv, YR == yv, f"loyo{yv}")
    loyo.append(c)
    print(f"  LOYO {yv}: best α={ALPHAS[c.argmax()]:.2f}  auc={c.max():.5f}", flush=True)
loyo = np.vstack(loyo)
curves["PX2_LOYO_mean"] = loyo.mean(0)
curves["PX2_LOYO_min"] = loyo.min(0)

# PX3 region holdout
regc = []
for rv in sorted(set(reg.tolist())):
    c = holdout_curve(reg != rv, reg == rv, f"reg{rv}")
    regc.append(c)
    print(f"  REG {rv}: best α={ALPHAS[c.argmax()]:.2f}  auc={c.max():.5f}", flush=True)
regc = np.vstack(regc)
curves["PX3_region_mean"] = regc.mean(0)
curves["PX3_region_min"] = regc.min(0)

# PX4 age holdout (40+ 학습 → 20-39 평가)
curves["PX4_age_holdout"] = holdout_curve(AGE_TR >= 40, AGE_TR < 40, "age40")
print("PX4 done", flush=True)

# PX4b 역방향 (20-59 학습 → 60+ 평가)
curves["PX4b_age_old"] = holdout_curve(AGE_TR < 60, AGE_TR >= 60, "age60")

# PX5 tail holdout (마할라노비스)
prep = make_prep(NUM, BIN, CAT, bin_impute=True)
A_tr = np.asarray(prep.fit_transform(XTR[FEATURES]), float)
from sklearn.preprocessing import StandardScaler
scl = StandardScaler().fit(A_tr); A_tr = scl.transform(A_tr)
mu = A_tr.mean(0)
COV = np.cov(A_tr[np.random.RandomState(0).choice(len(A_tr), 120000, replace=False)], rowvar=False)
IC = np.linalg.pinv(COV + 1e-6 * np.eye(COV.shape[0]))
d_tr = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", A_tr - mu, IC, A_tr - mu), 0))
thr = np.percentile(d_tr, 80)
curves["PX5_tail_holdout"] = holdout_curve(d_tr <= thr, d_tr > thr, "tail80")
print("PX5 done", flush=True)

# ---------------------------------------------------------------- 외부 α곡선 (진단 전용)
ext_pre_curve = np.array([wauc(YP, a * logit(g_pre) + (1 - a) * logit(l_pre), WP) for a in ALPHAS])
ext_post_curve = np.array([wauc(YQ, a * logit(g_post) + (1 - a) * logit(l_post), WQ) for a in ALPHAS])
int_curve = np.array([wauc(YTE, a * logit(g_te) + (1 - a) * logit(l_te), WTE) for a in ALPHAS])

C = pd.DataFrame({"alpha": ALPHAS})
for k, v in curves.items():
    C[k] = np.round(v, 6)
C["EXT_pre(진단)"] = np.round(ext_pre_curve, 6)
C["EXT_post(진단)"] = np.round(ext_post_curve, 6)
C["INT_holdout"] = np.round(int_curve, 6)
C.to_csv(os.path.join(RES, "path6_06_proxy_alpha_curves.csv"), index=False, encoding="utf-8-sig")
print(C.to_string())

# ---------------------------------------------------------------- 프록시가 고른 α → 외부 1회 평가
rows = [report("PX_base_lgbm", g_te, g_pre, g_post, note="α=1.0 기준선")]
for k, v in curves.items():
    a = float(ALPHAS[np.argmax(v)])
    rows.append(report(f"{k}_pick_a{a:.2f}",
                       a * logit(g_te) + (1 - a) * logit(l_te),
                       a * logit(g_pre) + (1 - a) * logit(l_pre),
                       a * logit(g_post) + (1 - a) * logit(l_post),
                       compliant=True,
                       note=f"{k} 프록시가 고른 α={a:.2f} (KCHS 내부만 사용)"))
    print(rows[-1], flush=True)

sm = pd.DataFrame(dict(proxy=list(curves.keys()),
                       picked_alpha=[float(ALPHAS[np.argmax(v)]) for v in curves.values()]))
sm["ext_pre_optimal_alpha"] = float(ALPHAS[ext_pre_curve.argmax()])
sm["ext_post_optimal_alpha"] = float(ALPHAS[ext_post_curve.argmax()])
sm.to_csv(os.path.join(RES, "path6_06_proxy_pick.csv"), index=False, encoding="utf-8-sig")
print(sm.to_string())

save(rows, "path6_06_proxy.csv")
print("TOTAL", round(time.time() - T0, 1), "s")
