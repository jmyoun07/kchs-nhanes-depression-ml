# -*- coding: utf-8 -*-
"""
pathC_common.py — 경로C(앙상블·확률보정) 공통 로더/전처리/평가 유틸.

§2.10 준수 원칙:
  - 학습(fit)은 KCHS(ext_kchs)만 사용. NHANES 라벨(dep)은 어떤 fit에도 들어가지 않는다.
  - 모델선택/가중치/하이퍼파라미터는 KCHS 내부(hold-out 또는 CV)로만 결정.
  - NHANES는 predict만 수행 → 가중 AUROC 산출.
확정 17 예측변수 고정.
"""
import os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score

ROOT = r"C:/Users/rudtn/Desktop/claude code/지역사회건강"
PROC = os.path.join(ROOT, "data/processed")
RES = os.path.join(ROOT, "results")
SEED = 42

# ---- 확정 17 예측변수 (추가·교체 금지) ----
NUM = ["age", "sleep_hrs", "bmi", "income_q", "srh", "oral_health"]
BIN = ["partnered", "living_alone", "current_drinker", "active",
       "htn", "dm", "employed", "food_insecure"]
CAT = ["sex", "edu", "smoking"]
FEATURES = NUM + BIN + CAT
assert len(FEATURES) == 17, len(FEATURES)


def _income_q_trainonly(df, is_tr):
    """연도별 소득 5분위 cutpoint를 train행에서만 학습(누수 차단)."""
    iq = np.full(len(df), np.nan)
    yr = df["year"].values
    inc = df["income_raw"].to_numpy(float)
    for yv in np.unique(yr):
        m = yr == yv
        v = inc[m & is_tr]; v = v[~np.isnan(v)]
        if len(v) < 50:
            continue
        e = np.unique(np.quantile(v, [0, .2, .4, .6, .8, 1.0]))
        if len(e) < 3:
            continue
        ii = np.where(m)[0]
        b = np.clip(np.digitize(inc[ii], e[1:-1]), 0, len(e) - 2).astype(float)
        b[np.isnan(inc[ii])] = np.nan
        iq[ii] = b
    return iq


class Data:
    pass


def load(split="full02", seed=SEED):
    """split='full02': test_size=0.2 (기존 산출물 재현용)
       split='500k'  : train_size=500000, test_size=100000 (과제 명세)"""
    k = pd.read_parquet(os.path.join(PROC, "ext_kchs.parquet"))
    pre = pd.read_parquet(os.path.join(PROC, "ext_pre.parquet"))
    post = pd.read_parquet(os.path.join(PROC, "ext_post.parquet"))
    y = k["dep"].astype(int).values
    W = k["W"].to_numpy(float)
    idx = np.arange(len(k))
    if split == "full02":
        itr, iho = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
    elif split == "500k":
        itr, iho = train_test_split(idx, train_size=500000, test_size=100000,
                                    stratify=y, random_state=seed)
    else:
        raise ValueError(split)
    is_tr = np.zeros(len(k), bool); is_tr[itr] = True
    kk = k.copy()
    kk["income_q"] = _income_q_trainonly(kk, is_tr)

    d = Data()
    d.k = kk; d.pre = pre; d.post = post
    d.y = y; d.W = W; d.itr = itr; d.iho = iho
    d.X = kk[FEATURES]
    d.Xtr = d.X.iloc[itr]; d.ytr = y[itr]; d.Wtr = W[itr]
    d.Xho = d.X.iloc[iho]; d.yho = y[iho]; d.Who = W[iho]
    d.Xpre = pre[FEATURES]; d.ypre = pre["dep"].astype(int).values
    d.wpre = pre["W"].to_numpy(float)
    d.Xpost = post[FEATURES]; d.ypost = post["dep"].astype(int).values
    d.wpost = post["W"].to_numpy(float)
    return d


def make_pre(native_nan=True):
    """기준선 전처리: 수치형 median+표준화 / 이진형 passthrough(NaN유지) / 범주형 최빈값+OneHot(drop=first)"""
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    binp = "passthrough" if native_nan else SimpleImputer(strategy="most_frequent",
                                                         add_indicator=True)
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore", drop="first"))])
    return ColumnTransformer([("num", num, NUM), ("bin", binp, BIN), ("cat", cat, CAT)])


def wauc(y, p, w):
    return float(roc_auc_score(y, p, sample_weight=w))


def rank01(p):
    """순위를 [0,1]로 정규화 (동점 평균순위). 척도 차이에 강건한 앙상블용."""
    p = np.asarray(p, float)
    order = p.argsort(kind="mergesort")
    r = np.empty(len(p), float)
    r[order] = np.arange(len(p), dtype=float)
    # 동점 평균처리
    s = pd.Series(p).rank(method="average").to_numpy()
    return (s - 1) / max(len(p) - 1, 1)
