# -*- coding: utf-8 -*-
"""
경로6 공통 모듈 — 모델 구조·개별화 (two-stage / stratified / residual / blend / latent / calibration-aware).

§2.10 준수 원칙
 - 학습은 KCHS만. NHANES 라벨(dep)은 학습/선택에 절대 미사용.
 - 모델·하이퍼파라미터 선택은 KCHS 내부 CV(또는 KCHS 내부 LOYO/지역 프록시)로만.
 - NHANES는 최종 1회 평가.
 - 기준선은 과제 명세 그대로: train_test_split(train=500000, test=100000, stratify=y, seed=42)
"""
from __future__ import annotations
import os, time, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

ROOT = r"C:/Users/rudtn/Desktop/claude code/지역사회건강"
PROC = os.path.join(ROOT, "data/processed")
RES = os.path.join(ROOT, "results")
SEED = 42
N_TRAIN, N_TEST = 500_000, 100_000

NUM = ["age", "sleep_hrs", "bmi", "income_q", "srh", "oral_health"]
BIN = ["partnered", "living_alone", "current_drinker", "active",
       "htn", "dm", "employed", "food_insecure"]
CAT = ["sex", "edu", "smoking"]
FEATURES = NUM + BIN + CAT

LGBM = dict(n_estimators=600, learning_rate=0.03, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, n_jobs=-1, verbose=-1)


_KCOLS = FEATURES + ["dep", "W", "year", "strata"]


def load():
    # 메모리 절약: 필요한 열만 읽는다 (값은 동일하므로 기준선 재현에 영향 없음)
    K = pd.read_parquet(os.path.join(PROC, "ext_kchs.parquet"), columns=_KCOLS)
    P = pd.read_parquet(os.path.join(PROC, "ext_pre.parquet"))
    Q = pd.read_parquet(os.path.join(PROC, "ext_post.parquet"))
    return K, P, Q


import gc
K, PRE, POST = load()
yK = K["dep"].astype(int).to_numpy()
WK = K["W"].to_numpy(float)
_idx = np.arange(len(K))
ITR, ITE = train_test_split(_idx, train_size=N_TRAIN, test_size=N_TEST,
                            stratify=yK, random_state=SEED)
# 주의: 기준선 재현을 위해 정렬하지 않은 원 순서를 그대로 사용
YTR, YTE = yK[ITR], yK[ITE]
WTR, WTE = WK[ITR], WK[ITE]

XTR = K[FEATURES].iloc[ITR].reset_index(drop=True)
XTE = K[FEATURES].iloc[ITE].reset_index(drop=True)
XP = PRE[FEATURES].reset_index(drop=True)
XQ = POST[FEATURES].reset_index(drop=True)
# 설계변수는 학습셋 부분만 남기고 대용량 원본은 해제
YEAR_TR = K["year"].to_numpy()[ITR]
STRATA_TR = K["strata"].to_numpy()[ITR]
del K
gc.collect()
YP = PRE["dep"].astype(int).to_numpy(); WP = PRE["W"].to_numpy(float)
YQ = POST["dep"].astype(int).to_numpy(); WQ = POST["W"].to_numpy(float)

AGE_TR = XTR["age"].to_numpy(float)
AGE_TE = XTE["age"].to_numpy(float)
AGE_P = XP["age"].to_numpy(float)
AGE_Q = XQ["age"].to_numpy(float)
SEX_TR = XTR["sex"].to_numpy()
SEX_TE = XTE["sex"].to_numpy()
SEX_P = XP["sex"].to_numpy()
SEX_Q = XQ["sex"].to_numpy()


def wauc(y, p, w):
    return roc_auc_score(np.asarray(y).astype(int), np.asarray(p, float),
                         sample_weight=np.asarray(w, float))


def make_prep(num=NUM, binf=BIN, cat=CAT, bin_impute=False):
    nump = Pipeline([("i", SimpleImputer(strategy="median")),
                     ("s", StandardScaler())])
    catp = Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                     ("o", OneHotEncoder(handle_unknown="ignore", drop="first"))])
    binp = (Pipeline([("i", SimpleImputer(strategy="most_frequent"))])
            if bin_impute else "passthrough")
    blocks = []
    if num:
        blocks.append(("num", nump, list(num)))
    if binf:
        blocks.append(("bin", binp, list(binf)))
    if cat:
        blocks.append(("cat", catp, list(cat)))
    return ColumnTransformer(blocks)


def new_lgbm(**kw):
    p = dict(LGBM); p.update(kw)
    return lgb.LGBMClassifier(**p)


def fit_predict(Xtr, ytr, Xs, clf=None, bin_impute=None, wtr=None,
                num=NUM, binf=BIN, cat=CAT):
    """Xs: list of DataFrames to predict. returns list of prob arrays + fitted pipe"""
    if clf is None:
        clf = new_lgbm()
    if bin_impute is None:
        bin_impute = not isinstance(clf, lgb.LGBMClassifier)
    feats = list(num) + list(binf) + list(cat)
    pipe = Pipeline([("pre", make_prep(num, binf, cat, bin_impute)), ("clf", clf)])
    if wtr is not None:
        pipe.fit(Xtr[feats], ytr, clf__sample_weight=wtr)
    else:
        pipe.fit(Xtr[feats], ytr)
    outs = [pipe.predict_proba(X[feats])[:, 1] for X in Xs]
    return outs, pipe


def report(name, p_te, p_pre, p_post, note="", compliant=True, secs=None):
    return dict(name=name,
                internal=round(float(wauc(YTE, p_te, WTE)), 6),
                ext_pre=round(float(wauc(YP, p_pre, WP)), 6),
                ext_post=round(float(wauc(YQ, p_post, WQ)), 6),
                protocol_compliant=bool(compliant),
                seconds=None if secs is None else round(secs, 1),
                note=note)


def save(rows, fname):
    df = pd.DataFrame(rows)
    path = os.path.join(RES, fname)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print("saved ->", path)
    return df


# ---------------------------------------------------------------- 내부 CV (모델선택 전용)
def kchs_cv(build_and_score, folds=4, seed=SEED, sub=250_000, verbose=True):
    """KCHS 학습셋 내부 K-fold 가중 AUROC.
    build_and_score(Xtr, ytr, wtr, Xva) -> score array on Xva
    NHANES 전혀 미접촉 → §2.10 준수.
    """
    ii = np.arange(len(XTR)) if (sub is None or sub >= len(XTR)) else \
        train_test_split(np.arange(len(XTR)), train_size=sub,
                         stratify=YTR, random_state=seed)[0]
    Xs = XTR.iloc[ii].reset_index(drop=True)
    ys = YTR[ii]; ws = WTR[ii]
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    sc = []
    for tr, va in skf.split(Xs, ys):
        s = build_and_score(Xs.iloc[tr].reset_index(drop=True), ys[tr], ws[tr],
                            Xs.iloc[va].reset_index(drop=True))
        sc.append(wauc(ys[va], s, ws[va]))
    if verbose:
        print("   cv folds:", [round(x, 5) for x in sc])
    return float(np.mean(sc)), float(np.std(sc))


def zrank(x):
    """순위 -> [0,1]"""
    x = np.asarray(x, float)
    o = x.argsort(kind="mergesort")
    r = np.empty(len(x), float)
    r[o] = np.arange(len(x), dtype=float)
    return r / max(len(x) - 1, 1)


def logit(p, eps=1e-7):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


AGE_BANDS = [(20, 40, "20-39"), (40, 60, "40-59"), (60, 999, "60+")]


def age_band_idx(age):
    out = np.full(len(age), -1, int)
    for i, (lo, hi, _) in enumerate(AGE_BANDS):
        out[(age >= lo) & (age < hi)] = i
    return out
