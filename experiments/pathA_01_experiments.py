# -*- coding: utf-8 -*-
"""
경로A 실험 본체 — 특징 표현 최적화 (17변수 고정).

실험군
 E1  baseline_17                    기준선
 E2  rank_cont_all                  연속·순서형 전부 within-survey 백분위 순위
 E3  rank_srh_oral                  srh/oral_health만 within-survey 순위
 E4  rank_srh_oral_income           srh/oral/income_q 순위 (income_q는 이미 설문내 분위)
 E5  srh_oral_binary                srh/oral 상위2 이진화
 E6  srh_oral_onehot                srh/oral 순서형→더미
 E7  nonlin_feats                   age/bmi/sleep 비선형 파생(편차·제곱·구간)
 E8  interactions_lgbm              명시적 상호작용 항 추가
 E9  logit_baseline                 로지스틱(선형) 기준
 E10 logit_spline                   로지스틱 + RCS 스플라인
 E11 logit_spline_interact          로지스틱 + 스플라인 + 상호작용
 E12 logit_spline_rank              로지스틱 + 스플라인 + 순위변환
 E13 rank+nonlin 조합 등 상위 조합

모두 KCHS로만 학습. NHANES 라벨 미사용.
"""
from __future__ import annotations
import os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathA_feature_repr import (K, PRE, POST, yK, WK, NUM, BIN, CAT, LGBM,
                                make_prep, wauc, SEED, RES, itr, ite)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

OUTCSV = os.path.join(RES, "pathA_experiments.csv")
ROWS = []
LOG = []


def log(s):
    LOG.append(str(s)); print(s, flush=True)


# ------------------------------------------------------------------ transforms
def pct_rank(series_ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    """ref 분포에서의 midrank 백분위(0~1). NaN 유지."""
    v = np.sort(series_ref[~np.isnan(series_ref)])
    lo = np.searchsorted(v, x, side="left")
    hi = np.searchsorted(v, x, side="right")
    r = (lo + hi) / (2.0 * len(v))
    r = np.asarray(r, float)
    r[np.isnan(x)] = np.nan
    return r


def apply_rank(dfs, cols):
    """dfs: [(name, full_df_for_ref, df_to_transform), ...] — 각 설문 자체 분포로 순위화."""
    out = []
    for ref, d in dfs:
        dd = d.copy()
        for c in cols:
            dd[c] = pct_rank(ref[c].to_numpy(float), d[c].to_numpy(float))
        out.append(dd)
    return out


def rcs(x, knots):
    """restricted cubic spline 기저 (Harrell). x 1차항 제외한 K-2개 반환."""
    k = np.asarray(knots, float)
    Kn = len(k)
    den = (k[-1] - k[0]) ** 2
    pos = lambda a: np.where(a > 0, a ** 3, 0.0)
    out = []
    for j in range(Kn - 2):
        t = (pos(x - k[j])
             - pos(x - k[Kn - 2]) * (k[-1] - k[j]) / (k[-1] - k[Kn - 2])
             + pos(x - k[-1]) * (k[Kn - 2] - k[j]) / (k[-1] - k[Kn - 2]))
        out.append(t / den)
    return out


# KCHS train 분포로 매듭 결정 (외부 미접촉)
KTR = K.iloc[itr]
KNOTS = {}
for c in ["age", "bmi", "sleep_hrs"]:
    v = KTR[c].to_numpy(float)
    v = v[~np.isnan(v)]
    KNOTS[c] = np.quantile(v, [0.05, 0.275, 0.50, 0.725, 0.95])
log(f"RCS knots (KCHS train): { {k: np.round(v,2).tolist() for k,v in KNOTS.items()} }")


def add_spline(d, cols=("age", "bmi", "sleep_hrs")):
    dd = d.copy()
    new = []
    for c in cols:
        x = d[c].to_numpy(float)
        for j, b in enumerate(rcs(x, KNOTS[c])):
            nm = f"{c}_sp{j+1}"
            dd[nm] = b
            new.append(nm)
    return dd, new


def add_nonlin(d):
    """비선형 파생: 수면 U자(편차), BMI U자(편차), 연령 제곱."""
    dd = d.copy()
    sl = d["sleep_hrs"].to_numpy(float)
    bm = d["bmi"].to_numpy(float)
    ag = d["age"].to_numpy(float)
    dd["sleep_dev"] = np.abs(sl - 7.0)
    dd["sleep_short"] = (sl < 6).astype(float); dd.loc[np.isnan(sl), "sleep_short"] = np.nan
    dd["sleep_long"] = (sl > 9).astype(float); dd.loc[np.isnan(sl), "sleep_long"] = np.nan
    dd["bmi_dev"] = np.abs(bm - 22.0)
    dd["bmi_under"] = (bm < 18.5).astype(float); dd.loc[np.isnan(bm), "bmi_under"] = np.nan
    dd["bmi_obese"] = (bm >= 30).astype(float); dd.loc[np.isnan(bm), "bmi_obese"] = np.nan
    dd["age_sq"] = ag ** 2
    new = ["sleep_dev", "bmi_dev", "age_sq"]
    newbin = ["sleep_short", "sleep_long", "bmi_under", "bmi_obese"]
    return dd, new, newbin


def add_interact(d):
    """명시적 상호작용 항 (수치화된 형태)."""
    dd = d.copy()
    male = (d["sex"].astype(str) == "M").astype(float)
    ag = d["age"].to_numpy(float)
    srh = d["srh"].to_numpy(float)
    orh = d["oral_health"].to_numpy(float)
    emp = d["employed"].to_numpy(float)
    inc = d["income_q"].to_numpy(float)
    la = d["living_alone"].to_numpy(float)
    fi = d["food_insecure"].to_numpy(float)
    dd["ix_age_male"] = ag * male
    dd["ix_srh_age"] = srh * ag
    dd["ix_srh_oral"] = srh * orh
    dd["ix_emp_age"] = emp * ag
    dd["ix_srh_male"] = srh * male
    dd["ix_inc_srh"] = inc * srh
    dd["ix_alone_age"] = la * ag
    dd["ix_fi_inc"] = fi * inc
    new = ["ix_age_male", "ix_srh_age", "ix_srh_oral", "ix_emp_age",
           "ix_srh_male", "ix_inc_srh", "ix_alone_age", "ix_fi_inc"]
    return dd, new


def add_srh_binary(d):
    dd = d.copy()
    srh = d["srh"].to_numpy(float)
    orh = d["oral_health"].to_numpy(float)
    dd["srh_bad"] = (srh >= 4).astype(float); dd.loc[np.isnan(srh), "srh_bad"] = np.nan
    dd["srh_good"] = (srh <= 2).astype(float); dd.loc[np.isnan(srh), "srh_good"] = np.nan
    dd["oral_bad"] = (orh >= 4).astype(float); dd.loc[np.isnan(orh), "oral_bad"] = np.nan
    dd["oral_good"] = (orh <= 2).astype(float); dd.loc[np.isnan(orh), "oral_good"] = np.nan
    return dd, ["srh_bad", "srh_good", "oral_bad", "oral_good"]


def add_srh_cat(d):
    dd = d.copy()
    dd["srh_c"] = d["srh"].astype("Int64").astype(str).replace("<NA>", np.nan)
    dd["oral_c"] = d["oral_health"].astype("Int64").astype(str).replace("<NA>", np.nan)
    return dd, ["srh_c", "oral_c"]


# ------------------------------------------------------------------ runner
def make_model(kind):
    if kind == "lgbm":
        return lgb.LGBMClassifier(**LGBM)
    if kind == "logit":
        return LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", n_jobs=-1)
    if kind == "logit_l2strong":
        return LogisticRegression(max_iter=2000, C=0.05, solver="lbfgs", n_jobs=-1)
    raise ValueError(kind)


def run(tag, dk, dpre, dpost, num, binf, cat, kind="lgbm",
        compliant=True, note="", cv=False):
    feats = list(num) + list(binf) + list(cat)
    t0 = time.time()
    pipe = Pipeline([("pre", make_prep(num, binf, cat)), ("clf", make_model(kind))])
    Xk = dk[feats]
    # 로지스틱은 이진형 NaN 처리 필요 → 이진도 median 대체
    if kind.startswith("logit"):
        pipe = Pipeline([("pre", make_prep(num + binf, [], cat)), ("clf", make_model(kind))])
        feats = list(num) + list(binf) + list(cat)
        Xk = dk[feats]
    pipe.fit(Xk.iloc[itr], yK[itr])
    a = wauc(yK[ite], pipe.predict_proba(Xk.iloc[ite])[:, 1], WK[ite])
    p = wauc(dpre["dep"], pipe.predict_proba(dpre[feats])[:, 1], dpre["W"])
    o = wauc(dpost["dep"], pipe.predict_proba(dpost[feats])[:, 1], dpost["W"])
    cvm = np.nan
    if cv:
        cvm = internal_cv(dk, num, binf, cat, kind)
    r = dict(name=tag, internal=round(a, 4), ext_pre=round(p, 4), ext_post=round(o, 4),
             internal_cv=cvm, kind=kind, n_feats=len(feats),
             protocol_compliant=compliant, note=note,
             seconds=round(time.time() - t0, 1))
    ROWS.append(r)
    pd.DataFrame(ROWS).to_csv(OUTCSV, index=False, encoding="utf-8-sig")
    log(f"{tag:32s} int={a:.4f} cv={cvm if isinstance(cvm,float) else cvm} "
        f"pre={p:.4f} post={o:.4f} ok={compliant} [{r['seconds']}s]")
    return r


def internal_cv(dk, num, binf, cat, kind, folds=3, nsub=200_000):
    """KCHS train 내부 3-fold 가중 AUROC (모델 선택 전용)."""
    feats = list(num) + list(binf) + list(cat)
    rng = np.random.RandomState(SEED)
    sub = rng.choice(itr, size=min(nsub, len(itr)), replace=False)
    X = dk[feats].iloc[sub].reset_index(drop=True)
    y = yK[sub]; w = WK[sub]
    if kind.startswith("logit"):
        prep_args = (num + binf, [], cat)
    else:
        prep_args = (num, binf, cat)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    sc = []
    for tr, va in skf.split(X, y):
        pipe = Pipeline([("pre", make_prep(*prep_args)), ("clf", make_model(kind))])
        pipe.fit(X.iloc[tr], y[tr])
        sc.append(wauc(y[va], pipe.predict_proba(X.iloc[va])[:, 1], w[va]))
    return round(float(np.mean(sc)), 4)


# ================================================================== EXPERIMENTS
if __name__ == "__main__":
    RANKC_ALL = ["age", "sleep_hrs", "bmi", "income_q", "srh", "oral_health"]

    # E1 baseline
    run("E1_baseline_17", K, PRE, POST, NUM, BIN, CAT, "lgbm", True,
        "protocol baseline", cv=True)

    # E2 전체 연속·순서형 within-survey 순위
    k2, p2, o2 = apply_rank([(K, K), (PRE, PRE), (POST, POST)], RANKC_ALL)
    run("E2_rank_all", k2, p2, o2, NUM, BIN, CAT, "lgbm", True,
        "within-survey percentile rank on 6 numeric; NHANES covariates only (no labels)", cv=True)

    # E3 srh/oral만
    k3, p3, o3 = apply_rank([(K, K), (PRE, PRE), (POST, POST)], ["srh", "oral_health"])
    run("E3_rank_srh_oral", k3, p3, o3, NUM, BIN, CAT, "lgbm", True,
        "rank on srh+oral_health only", cv=True)

    # E4 srh/oral/income_q
    k4, p4, o4 = apply_rank([(K, K), (PRE, PRE), (POST, POST)],
                            ["srh", "oral_health", "income_q"])
    run("E4_rank_srh_oral_inc", k4, p4, o4, NUM, BIN, CAT, "lgbm", True,
        "rank on srh+oral+income_q", cv=True)

    # E5 srh/oral 이진화
    k5, n5 = add_srh_binary(K); p5, _ = add_srh_binary(PRE); o5, _ = add_srh_binary(POST)
    num5 = [c for c in NUM if c not in ("srh", "oral_health")]
    run("E5_srh_oral_binary", k5, p5, o5, num5, BIN + n5, CAT, "lgbm", True,
        "srh/oral replaced by good/bad binaries", cv=True)

    # E6 srh/oral 원핫
    k6, n6 = add_srh_cat(K); p6, _ = add_srh_cat(PRE); o6, _ = add_srh_cat(POST)
    run("E6_srh_oral_onehot", k6, p6, o6, num5, BIN, CAT + n6, "lgbm", True,
        "srh/oral as one-hot categories", cv=True)

    # E7 비선형 파생 (LGBM)
    k7, n7, nb7 = add_nonlin(K); p7, _, _ = add_nonlin(PRE); o7, _, _ = add_nonlin(POST)
    run("E7_nonlin_lgbm", k7, p7, o7, NUM + n7, BIN + nb7, CAT, "lgbm", True,
        "explicit U-shape/threshold derivatives", cv=True)

    # E8 상호작용 (LGBM)
    k8, n8 = add_interact(K); p8, _ = add_interact(PRE); o8, _ = add_interact(POST)
    run("E8_interact_lgbm", k8, p8, o8, NUM + n8, BIN, CAT, "lgbm", True,
        "explicit interaction terms", cv=True)

    # E9 로지스틱 선형
    run("E9_logit_linear", K, PRE, POST, NUM, BIN, CAT, "logit", True,
        "linear logistic baseline", cv=True)

    # E10 로지스틱 + 스플라인
    k10, s10 = add_spline(K); p10, _ = add_spline(PRE); o10, _ = add_spline(POST)
    run("E10_logit_spline", k10, p10, o10, NUM + s10, BIN, CAT, "logit", True,
        "logit + RCS(age,bmi,sleep) knots from KCHS train", cv=True)

    # E11 로지스틱 + 스플라인 + 상호작용
    k11, i11 = add_interact(k10); p11, _ = add_interact(p10); o11, _ = add_interact(o10)
    run("E11_logit_spline_interact", k11, p11, o11, NUM + s10 + i11, BIN, CAT, "logit", True,
        "logit + spline + interactions", cv=True)

    # E12 로지스틱 + 스플라인 + 순위변환
    kr, pr, orr = apply_rank([(K, K), (PRE, PRE), (POST, POST)], RANKC_ALL)
    k12, s12 = add_spline(kr); p12, _ = add_spline(pr); o12, _ = add_spline(orr)
    run("E12_logit_spline_rank", k12, p12, o12, NUM + s12, BIN, CAT, "logit", True,
        "logit + rank + spline", cv=True)

    # E13 rank + 비선형 (LGBM)
    k13, n13, nb13 = add_nonlin(kr); p13, _, _ = add_nonlin(pr); o13, _, _ = add_nonlin(orr)
    run("E13_rank_nonlin_lgbm", k13, p13, o13, NUM + n13, BIN + nb13, CAT, "lgbm", True,
        "rank + nonlinear derivatives", cv=True)

    # E14 rank + 상호작용 (LGBM)
    k14, n14 = add_interact(kr); p14, _ = add_interact(pr); o14, _ = add_interact(orr)
    run("E14_rank_interact_lgbm", k14, p14, o14, NUM + n14, BIN, CAT, "lgbm", True,
        "rank + interactions", cv=True)

    open(os.path.join(RES, "pathA_experiments_log.txt"), "w", encoding="utf-8").write("\n".join(LOG))
    log("\n=== done ===")
    log(pd.DataFrame(ROWS).to_string())
