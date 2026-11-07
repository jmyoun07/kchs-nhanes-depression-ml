# -*- coding: utf-8 -*-
"""pathG_01: 경로2 - 공변량 분포만 쓰는 도메인 적응.

(1) 중요도 가중 (density-ratio importance weighting)
(2) CORAL / 특징 정렬
(3) 표본 트리밍 (propensity overlap)
(4) 불변 특징 다운웨이트
(5) 조합
+ 대조군: NHANES를 전혀 안 쓰는 프로토콜 준수 강건화(무작위 특징 마스킹)
"""
import os, sys, time, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathG_common import *
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import scipy.linalg as sla

T0 = time.time()
k, pre, post = load()
tr, ho = base_split(k)
y = k["dep"].astype(int).values
Wk = k["W"].to_numpy(float)
ypre, wpre = pre["dep"].astype(int).values, pre["W"].to_numpy(float)
ypost, wpost = post["dep"].astype(int).values, post["W"].to_numpy(float)

Ktr = k.iloc[tr][FEATURES].reset_index(drop=True)
Kho = k.iloc[ho][FEATURES].reset_index(drop=True)
Pre = pre[FEATURES].reset_index(drop=True)
Post = post[FEATURES].reset_index(drop=True)
ytr, yho = y[tr], y[ho]
who = Wk[ho]

# ---------- 공통 설계행렬 (기준선 CT를 학습셋에 적합) ----------
CT = make_ct()
Xtr = CT.fit_transform(Ktr).astype(np.float64)
Xho = CT.transform(Kho).astype(np.float64)
Xpre = CT.transform(Pre).astype(np.float64)
Xpost = CT.transform(Post).astype(np.float64)
FN = list(CT.get_feature_names_out())
print("design matrix", Xtr.shape, flush=True)

BASE = None


def run_arr(name, xtr, xho, xpre, xpost, sw=None, params=None, compliant=False,
            note="", ytr_=None, extra=None):
    t = time.time()
    p = dict(LGB_PARAMS)
    if params:
        p.update(params)
    m = lgb.LGBMClassifier(**p)
    m.fit(xtr, ytr if ytr_ is None else ytr_, sample_weight=sw)
    r = rec(name,
            wauc(yho, m.predict_proba(xho)[:, 1], who),
            wauc(ypre, m.predict_proba(xpre)[:, 1], wpre),
            wauc(ypost, m.predict_proba(xpost)[:, 1], wpost),
            compliant, note, base=BASE, extra=extra, secs=time.time() - t)
    return r, m


# ================= 0. 기준선(설계행렬 경로) =================
r0, m0 = run_arr("G01_base", Xtr, Xho, Xpre, Xpost, compliant=True,
                 note="기준선 재현(설계행렬 경로). NHANES 미사용")
BASE = (r0["internal"], r0["ext_pre"], r0["ext_post"])

# ================= 진단: living_alone 구조적 결측 =================
LA = FN.index("bin__living_alone")
print("living_alone col idx", LA,
      "pre nan frac=%.3f" % np.isnan(Xpre[:, LA]).mean(),
      "post nan frac=%.3f" % np.isnan(Xpost[:, LA]).mean(), flush=True)


# ================= 도메인 분류기 =================
def domain_ratio(Xt_target, drop_cols=(), n_splits=3, lgb_small=True, seed=SEED):
    """KCHS train(소스) 대비 target(NHANES)의 밀도비 w=p(T)/p(S) 를 교차적합으로 추정.
    반환: (w_source, auc, e_source)"""
    keep = [i for i in range(Xtr.shape[1]) if i not in drop_cols]
    Xs, Xt = Xtr[:, keep], Xt_target[:, keep]
    ns, nt = len(Xs), len(Xt)
    e = np.zeros(ns)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dummy = np.zeros(ns)
    aucs = []
    for f, (a, b) in enumerate(skf.split(Xs, dummy)):
        XX = np.vstack([Xs[a], Xt])
        yy = np.r_[np.zeros(len(a)), np.ones(nt)]
        ww = np.r_[np.ones(len(a)), np.full(nt, len(a) / nt)]
        if lgb_small:
            dm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                    num_leaves=31, subsample=0.8,
                                    colsample_bytree=0.8, random_state=seed,
                                    n_jobs=-1, verbose=-1)
        else:
            dm = LogisticRegression(max_iter=2000, C=1.0)
        dm.fit(np.nan_to_num(XX, nan=0.0), yy, sample_weight=ww)
        e[b] = dm.predict_proba(np.nan_to_num(Xs[b], nan=0.0))[:, 1]
        et = dm.predict_proba(np.nan_to_num(Xt, nan=0.0))[:, 1]
        from sklearn.metrics import roc_auc_score as _ra
        aucs.append(_ra(np.r_[np.zeros(len(b)), np.ones(nt)], np.r_[e[b], et]))
    w = e / np.clip(1 - e, 1e-9, None)          # 클래스 균형 가중을 썼으므로 prior 보정 불필요
    return w, float(np.mean(aucs)), e


def ess(w):
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / np.sum(w ** 2))


DROP_LA = (LA,)   # pre가 100% 결측이므로 도메인분류기에서 제외(제외 안하면 AUC=1.0 붕괴)

TARGETS = {
    "pre":   Xpre,
    "post":  Xpost,
    "both":  np.vstack([Xpre, Xpost]),
}

RATIOS = {}
for tag, Xt in TARGETS.items():
    for dl, dtag in [(DROP_LA, "noLA"), ((), "withLA")]:
        for small, mtag in [(True, "lgb"), (False, "logit")]:
            key = f"{tag}_{dtag}_{mtag}"
            w, auc, e = domain_ratio(Xt, drop_cols=dl, lgb_small=small)
            RATIOS[key] = (w, auc, e)
            print(f"[dom] {key:22s} AUC={auc:.4f} w mean={w.mean():.3f} "
                  f"p99={np.quantile(w,.99):.2f} max={w.max():.1f} "
                  f"ESS={ess(w):,.0f}", flush=True)

pd.DataFrame([dict(key=kk, dom_auc=v[1], w_mean=float(v[0].mean()),
                   w_p99=float(np.quantile(v[0], .99)), w_max=float(v[0].max()),
                   ess=ess(v[0])) for kk, v in RATIOS.items()]).to_csv(
    os.path.join(RES, "pathG_domain_clf.csv"), index=False, encoding="utf-8-sig")

# ================= (1) 중요도 가중 =================
def clipw(w, lo, hi, norm=True):
    v = np.clip(w, lo, hi)
    return v / v.mean() if norm else v


for key in ["pre_noLA_lgb", "post_noLA_lgb", "both_noLA_lgb",
            "both_noLA_logit", "pre_noLA_logit", "both_withLA_lgb"]:
    w, auc, e = RATIOS[key]
    wn = w / w.mean()
    variants = [
        ("clip1_10", clipw(wn, 1.0, 10.0)),
        ("clip1_20", clipw(wn, 1.0, 20.0)),
        ("clip.1_10", clipw(wn, 0.1, 10.0)),
        ("clip.05_20", clipw(wn, 0.05, 20.0)),
        ("clipq01_99", clipw(wn, np.quantile(wn, .01), np.quantile(wn, .99))),
        ("sqrt", clipw(np.sqrt(np.clip(wn, 1e-6, None)), 0.05, 20.0)),
        ("raw", wn),
    ]
    for vt, ww in variants:
        run_arr(f"G01_IW_{key}_{vt}", Xtr, Xho, Xpre, Xpost, sw=ww,
                compliant=False,
                note=(f"중요도가중. 도메인분류기 AUC={auc:.4f}, ESS={ess(ww):,.0f}. "
                      f"NHANES {key.split('_')[0]} 공변량 사용 → §2.10 회색지대"),
                extra=dict(dom_auc=round(auc, 4), ess=round(ess(ww))))
    dump("pathG_01_da.csv")

# ================= (3) 표본 트리밍 =================
for key in ["both_noLA_lgb", "pre_noLA_lgb"]:
    w, auc, e = RATIOS[key]
    for q in [0.10, 0.25, 0.50, 0.75]:
        thr = np.quantile(e, q)
        m = e >= thr
        run_arr(f"G01_TRIM_{key}_top{int((1-q)*100)}", Xtr[m], Xho, Xpre, Xpost,
                ytr_=ytr[m], compliant=False,
                note=(f"propensity overlap 상위 {(1-q)*100:.0f}%로 학습표본 제한 "
                      f"n={m.sum():,}. NHANES 공변량 사용 → 회색지대"),
                extra=dict(dom_auc=round(auc, 4), train_n=int(m.sum())))
    dump("pathG_01_da.csv")

# ================= (2) CORAL / 특징 정렬 =================
def _stats(X, eps=1e-6):
    Z = np.nan_to_num(X, nan=0.0)
    mu = Z.mean(0)
    C = np.cov(Z, rowvar=False) + eps * np.eye(Z.shape[1])
    return mu, C, Z


def _sqrtm(C):
    v, Q = np.linalg.eigh(C)
    v = np.clip(v, 1e-10, None)
    return Q @ np.diag(np.sqrt(v)) @ Q.T


def _isqrtm(C):
    v, Q = np.linalg.eigh(C)
    v = np.clip(v, 1e-10, None)
    return Q @ np.diag(1 / np.sqrt(v)) @ Q.T


for ttag, Xt in [("pre", Xpre), ("post", Xpost), ("both", np.vstack([Xpre, Xpost]))]:
    mus, Cs, Zs = _stats(Xtr)
    mut, Ct, Zt = _stats(Xt)
    A = _isqrtm(Cs) @ _sqrtm(Ct)
    # A) source -> target 로 정렬 후 학습, 타깃은 원본
    Xs2 = (Zs - mus) @ A + mut
    Xho2 = (np.nan_to_num(Xho, nan=0.0) - mus) @ A + mut
    run_arr(f"G01_CORAL_s2t_{ttag}", Xs2, Xho2,
            np.nan_to_num(Xpre, nan=0.0), np.nan_to_num(Xpost, nan=0.0),
            compliant=False,
            note=f"CORAL: KCHS를 NHANES {ttag} 공분산/평균에 정렬 후 학습. 회색지대")
    # B) target -> source 로 정렬(기준선 모델 재사용 가능하나 여기선 동일 학습)
    B = _isqrtm(Ct) @ _sqrtm(Cs)
    Xpre_b = (np.nan_to_num(Xpre, nan=0.0) - mut) @ B + mus
    Xpost_b = (np.nan_to_num(Xpost, nan=0.0) - mut) @ B + mus
    run_arr(f"G01_CORAL_t2s_{ttag}", np.nan_to_num(Xtr, nan=0.0),
            np.nan_to_num(Xho, nan=0.0), Xpre_b, Xpost_b, compliant=False,
            note=f"CORAL: NHANES(pre/post)를 KCHS 공분산/평균에 정렬 후 예측(기준 {ttag}). 회색지대")
    # C) 평균만 정렬 (mean shift)
    Xpre_c = np.nan_to_num(Xpre, nan=0.0) - mut + mus
    Xpost_c = np.nan_to_num(Xpost, nan=0.0) - mut + mus
    run_arr(f"G01_MEANALIGN_t2s_{ttag}", np.nan_to_num(Xtr, nan=0.0),
            np.nan_to_num(Xho, nan=0.0), Xpre_c, Xpost_c, compliant=False,
            note=f"평균만 정렬(NHANES {ttag} → KCHS). 회색지대")
    dump("pathG_01_da.csv")

# ================= (4) 불변 특징 다운웨이트 =================
def shift_scores():
    """설계행렬 열별 표준화 평균차 |d| (NHANES pre+post vs KCHS)."""
    Xt = np.vstack([Xpre, Xpost])
    s = []
    for j in range(Xtr.shape[1]):
        a = Xtr[:, j][~np.isnan(Xtr[:, j])]
        b = Xt[:, j][~np.isnan(Xt[:, j])]
        if len(b) == 0:
            s.append(np.nan); continue
        sd = np.sqrt(0.5 * (a.var() + b.var())) + 1e-9
        s.append(abs(a.mean() - b.mean()) / sd)
    return np.array(s)


SS = shift_scores()
ssdf = pd.DataFrame(dict(feature=FN, smd=SS)).sort_values("smd", ascending=False)
print(ssdf.to_string(), flush=True)
ssdf.to_csv(os.path.join(RES, "pathG_shift_smd.csv"), index=False, encoding="utf-8-sig")
S = np.nan_to_num(SS, nan=np.nanmax(SS))

for a in [0.5, 1.0, 2.0, 5.0]:
    fc = 1.0 / (1.0 + a * S)
    run_arr(f"G01_FCONTRI_inv_a{a}", Xtr, Xho, Xpre, Xpost,
            params=dict(feature_contri=list(fc)), compliant=False,
            note=f"불변특징 선호: feature_contri=1/(1+{a}*SMD). NHANES 공변량 분포 사용 → 회색지대")
for a in [1.0, 2.0]:
    fc = np.exp(-a * S)
    run_arr(f"G01_FCONTRI_exp_a{a}", Xtr, Xho, Xpre, Xpost,
            params=dict(feature_contri=list(fc)), compliant=False,
            note=f"불변특징 선호: feature_contri=exp(-{a}*SMD). 회색지대")
# 상위 shift 특징 하드 다운웨이트
order = np.argsort(-S)
for kk in [1, 2, 3]:
    fc = np.ones(len(S)); fc[order[:kk]] = 0.1
    run_arr(f"G01_FCONTRI_top{kk}_0.1", Xtr, Xho, Xpre, Xpost,
            params=dict(feature_contri=list(fc)), compliant=False,
            note=f"최대 shift {kk}개({[FN[i] for i in order[:kk]]}) contri=0.1. 회색지대")
dump("pathG_01_da.csv")

# living_alone 다운웨이트/제거 (pre에서 100% 결측이라는 구조 사실 이용)
fc = np.ones(len(S)); fc[LA] = 0.0
run_arr("G01_FCONTRI_dropLA", Xtr, Xho, Xpre, Xpost,
        params=dict(feature_contri=list(fc)), compliant=False,
        note="living_alone contri=0 (NHANES pre에서 100% 결측이라는 사실 이용). "
             "17변수 중 1개 무력화 → 변수고정 제약과도 충돌")
# 무작위 마스킹 강건화(프로토콜 준수: NHANES 전혀 미사용)
rng = np.random.RandomState(SEED)
for pmask in [0.05, 0.10, 0.20]:
    Xa = Xtr.copy()
    M = rng.rand(*Xa.shape) < pmask
    Xa[M] = np.nan
    run_arr(f"G01_MASKAUG_p{pmask}", Xa, Xho, Xpre, Xpost, compliant=True,
            note=f"학습 시 무작위 특징 마스킹 {pmask:.0%}(배포 결측 강건화). NHANES 미사용 → 준수")
dump("pathG_01_da.csv")

print("TOTAL %.1f min" % ((time.time() - T0) / 60))
