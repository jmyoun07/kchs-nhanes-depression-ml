# -*- coding: utf-8 -*-
"""
pathD_common.py — 경로D(전이지향 학습) 공통 유틸.
17변수 고정. KCHS 라벨만 학습에 사용. NHANES는 재적합 없이 적용(§2.10).
"""
import os, sys, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

ROOT = r"C:/Users/rudtn/Desktop/claude code/지역사회건강"
PROC = os.path.join(ROOT, "data/processed")
RES = os.path.join(ROOT, "results")
SEED = 42

NUM = ["age", "sleep_hrs", "bmi", "income_q", "srh", "oral_health"]
BIN = ["partnered", "living_alone", "current_drinker", "active", "htn", "dm",
       "employed", "food_insecure"]
CAT = ["sex", "edu", "smoking"]
FEATURES = NUM + BIN + CAT
assert len(FEATURES) == 17

BASE_LGB = dict(n_estimators=600, learning_rate=0.03, num_leaves=31,
                subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                n_jobs=-1, verbose=-1)


def load():
    k = pd.read_parquet(os.path.join(PROC, "ext_kchs.parquet"))
    pre = pd.read_parquet(os.path.join(PROC, "ext_pre.parquet"))
    post = pd.read_parquet(os.path.join(PROC, "ext_post.parquet"))
    return k, pre, post


def make_pre(num=NUM, binv=BIN, cat=CAT):
    n = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    c = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                  ("oh", OneHotEncoder(handle_unknown="ignore", drop="first"))])
    return ColumnTransformer([("num", n, list(num)),
                              ("bin", "passthrough", list(binv)),
                              ("cat", c, list(cat))])


def split(k):
    """기준선 분할: train=500000, test=100000, stratify=dep, seed=42"""
    y = k["dep"].astype(int).values
    itr, iho = train_test_split(np.arange(len(k)), train_size=500000,
                                test_size=100000, stratify=y, random_state=SEED)
    return itr, iho


def wauc(y, p, w):
    return roc_auc_score(y, p, sample_weight=w)


def evaluate(pipe, k, pre, post, iho, feats=FEATURES):
    y = k["dep"].astype(int).values
    W = k["W"].to_numpy(float)
    out = {}
    out["internal"] = wauc(y[iho], pipe.predict_proba(k[feats].iloc[iho])[:, 1], W[iho])
    for nm, d in [("ext_pre", pre), ("ext_post", post)]:
        out[nm] = wauc(d["dep"].astype(int).values,
                       pipe.predict_proba(d[feats])[:, 1], d["W"].to_numpy(float))
    return out
