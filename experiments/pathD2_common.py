# -*- coding: utf-8 -*-
"""
pathD2_common.py — 경로D(전이 지향 학습, transportability) 공통 하네스.

설계 원칙(§2.10 준수):
  - 학습·모델선택은 **KCHS 라벨/내부정보만** 사용.
  - NHANES(pre/post)는 재적합 없이 predict만 하고, 선택에 절대 사용하지 않음.
  - 외부 수치는 "기록만" 하며, 어떤 선택 규칙에도 입력되지 않는다.

기준선 정정(중요):
  스펙에 적힌 train_test_split(train=500000,test=100000)로는 스펙 수치가 재현되지 않는다.
  실측: 500k/100k → 내부 0.8310 / pre 0.7740 / post 0.7653.
  test_size=0.2(80/20) + income_q train-only 분위 → 내부 0.8335 / pre 0.7682 / post 0.7680
  으로 스펙(0.8336/0.7681/0.7682)과 3자리 일치. src/03_train.py가 실제로 80/20을 쓴다.
  따라서 본 경로의 기준선은 80/20 + income_q train-only로 고정한다.
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import lightgbm as lgb

ROOT = r"C:/Users/rudtn/Desktop/claude code/지역사회건강"
PROC = os.path.join(ROOT, "data/processed")
RES = os.path.join(ROOT, "results")
SEED = 42

# ---- 확정 17 예측변수 (추가/교체 금지) ----
NUM = ["age", "sleep_hrs", "bmi", "income_q", "srh", "oral_health"]
BIN = ["partnered", "living_alone", "current_drinker", "active", "htn", "dm",
       "employed", "food_insecure"]
CAT = ["sex", "edu", "smoking"]
FEATURES = NUM + BIN + CAT
assert len(FEATURES) == 17

LGB_BASE = dict(n_estimators=600, learning_rate=0.03, num_leaves=31,
                subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                n_jobs=-1, verbose=-1)

TARGET = dict(internal=0.8336, ext_pre=0.7681, ext_post=0.7682)


# ------------------------------------------------------------------ data
def load():
    k = pd.read_parquet(os.path.join(PROC, "ext_kchs.parquet"))
    pre = pd.read_parquet(os.path.join(PROC, "ext_pre.parquet"))
    post = pd.read_parquet(os.path.join(PROC, "ext_post.parquet"))
    return k, pre, post


def income_q_trainonly(df, is_tr):
    """연도별 소득 5분위 cutpoint를 train 행에서만 학습(누수 제거)."""
    iq = np.full(len(df), np.nan)
    yr = df["year"].values
    inc = df["income_raw"].to_numpy(float)
    for yv in np.unique(yr):
        m = yr == yv
        v = inc[m & is_tr]
        v = v[~np.isnan(v)]
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


def make_pre(num=NUM, binv=BIN, cat=CAT):
    n = Pipeline([("imp", SimpleImputer(strategy="median")),
                  ("sc", StandardScaler())])
    c = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                  ("oh", OneHotEncoder(handle_unknown="ignore", drop="first"))])
    return ColumnTransformer([("num", n, list(num)),
                              ("bin", "passthrough", list(binv)),
                              ("cat", c, list(cat))])


def wauc(y, p, w):
    return roc_auc_score(y, p, sample_weight=w)


# ------------------------------------------------------------------ bench
class Bench:
    """80/20 고정 홀드아웃 + 고정 외부 평가셋 + 연도/지역 CV 유틸."""

    def __init__(self, seed=SEED):
        k, pre, post = load()
        y = k["dep"].astype(int).values
        itr, iho = train_test_split(np.arange(len(k)), test_size=0.2,
                                    stratify=y, random_state=seed)
        is_tr = np.zeros(len(k), bool)
        is_tr[itr] = True
        k = k.copy()
        k["income_q"] = income_q_trainonly(k, is_tr)

        self.k = k
        self.pre, self.post = pre, post
        self.y = y
        self.W = k["W"].to_numpy(float)
        self.year = k["year"].to_numpy(int)
        self.strata = k["strata"].astype(str).to_numpy()
        self.itr, self.iho = itr, iho
        self.X = k[FEATURES]

        self.ypre = pre["dep"].astype(int).values
        self.wpre = pre["W"].to_numpy(float)
        self.Xpre = pre[FEATURES]
        self.ypost = post["dep"].astype(int).values
        self.wpost = post["W"].to_numpy(float)
        self.Xpost = post[FEATURES]

        self.years = np.unique(self.year)

    # -- 학습 인덱스 유틸 --
    def sub(self, idx, n, seed=SEED):
        if n is None or n >= len(idx):
            return idx
        return np.random.RandomState(seed).choice(idx, size=n, replace=False)

    # -- 외부 포함 최종 평가(기록용, 선택 금지) --
    def evaluate(self, model, tag, note="", proba=True, Xtf=None):
        """Xtf: (df)->df 변환기(특징표현 실험용). None이면 원 17변수."""
        def px(df):
            d = Xtf(df) if Xtf is not None else df
            return model.predict_proba(d)[:, 1] if proba else model.predict(d)
        r = dict(name=tag)
        r["internal"] = wauc(self.y[self.iho], px(self.X.iloc[self.iho]),
                             self.W[self.iho])
        r["ext_pre"] = wauc(self.ypre, px(self.Xpre), self.wpre)
        r["ext_post"] = wauc(self.ypost, px(self.Xpost), self.wpost)
        r["note"] = note
        return r


# ------------------------------------------------------------------ CV
def loyo_folds(bench, train_pool=None):
    """Leave-One-Year-Out: (train_idx, valid_idx, year) 생성.
    train_pool을 주면 그 안에서만 학습(홀드아웃 오염 방지)."""
    pool = bench.itr if train_pool is None else train_pool
    yr = bench.year[pool]
    for yv in bench.years:
        m = yr == yv
        yield pool[~m], pool[m], yv


def strata_folds(bench, n_splits=5, seed=SEED, train_pool=None):
    """지역(strata) 기반 GroupKFold — 지역 간 일반화 측정."""
    pool = bench.itr if train_pool is None else train_pool
    g = bench.strata[pool]
    uq = np.unique(g)
    rs = np.random.RandomState(seed)
    perm = rs.permutation(len(uq))
    assign = {uq[i]: perm[j] % n_splits for j, i in enumerate(range(len(uq)))}
    fold_of = np.array([assign[x] for x in g])
    for f in range(n_splits):
        m = fold_of == f
        yield pool[~m], pool[m], f


def cv_score(bench, fit_fn, folds, train_n=None, seed=SEED, tag=""):
    """fit_fn(Xtr_df, ytr, wtr) -> predict_proba 가능한 객체.
    각 fold의 **가중 AUROC**를 반환. 선택은 이 값들로만 한다(KCHS 내부)."""
    scores, labels = [], []
    for tr, va, lab in folds:
        tr = bench.sub(tr, train_n, seed=seed)
        m = fit_fn(bench.X.iloc[tr], bench.y[tr], bench.W[tr])
        p = m.predict_proba(bench.X.iloc[va])[:, 1]
        scores.append(wauc(bench.y[va], p, bench.W[va]))
        labels.append(lab)
    s = np.asarray(scores, float)
    return dict(tag=tag, mean=s.mean(), std=s.std(ddof=0), min=s.min(),
                max=s.max(), gap=s.max() - s.min(),
                scores={str(l): float(v) for l, v in zip(labels, s)})


def fmt_cv(d):
    return (f"{d['tag']:38s} mean={d['mean']:.4f} std={d['std']:.4f} "
            f"min={d['min']:.4f} gap={d['gap']:.4f}")


def fmt_ext(r, base=None):
    s = (f"{r['name']:38s} 내부={r['internal']:.4f} "
         f"pre={r['ext_pre']:.4f} post={r['ext_post']:.4f}")
    if base:
        s += (f"  Δ={r['internal']-base['internal']:+.4f}/"
              f"{r['ext_pre']-base['ext_pre']:+.4f}/"
              f"{r['ext_post']-base['ext_post']:+.4f}")
    return s


def save(rows, name):
    p = os.path.join(RES, name)
    pd.DataFrame(rows).to_csv(p, index=False, encoding="utf-8-sig")
    print("saved:", p)
    return p
