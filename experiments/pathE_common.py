# -*- coding: utf-8 -*-
"""경로E 공통 하네스: 데이터 로드, 기준선 파이프라인, 가중 AUROC."""
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
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")
SEED = 42

NUM = ["age", "sleep_hrs", "bmi", "income_q", "srh", "oral_health"]
BIN = ["partnered", "living_alone", "current_drinker", "active",
       "htn", "dm", "employed", "food_insecure"]
CAT = ["sex", "edu", "smoking"]
FEATURES = NUM + BIN + CAT

LGB_PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=31,
                  subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                  n_jobs=-1, verbose=-1)


def load():
    k = pd.read_parquet(os.path.join(PROC, "ext_kchs.parquet"))
    pre = pd.read_parquet(os.path.join(PROC, "ext_pre.parquet"))
    post = pd.read_parquet(os.path.join(PROC, "ext_post.parquet"))
    return k, pre, post


def make_pre(num=NUM, bin_=BIN, cat=CAT):
    npipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())])
    cpipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore", drop="first"))])
    return ColumnTransformer([("num", npipe, list(num)),
                              ("bin", "passthrough", list(bin_)),
                              ("cat", cpipe, list(cat))])


def split_idx(k, train_n=500000, test_n=100000, seed=SEED):
    y = k["dep"].astype(int).values
    itr, iho = train_test_split(np.arange(len(k)), train_size=train_n,
                                test_size=test_n, stratify=y, random_state=seed)
    return itr, iho


def wauc(y, p, w):
    return roc_auc_score(y, p, sample_weight=w)


def evaluate(pipe, k, pre, post, iho, feats=FEATURES):
    y = k["dep"].astype(int).values
    W = k["W"].to_numpy(float)
    p_int = pipe.predict_proba(k.iloc[iho][feats])[:, 1]
    p_pre = pipe.predict_proba(pre[feats])[:, 1]
    p_post = pipe.predict_proba(post[feats])[:, 1]
    return (wauc(y[iho], p_int, W[iho]),
            wauc(pre["dep"].astype(int).values, p_pre, pre["W"].to_numpy(float)),
            wauc(post["dep"].astype(int).values, p_post, post["W"].to_numpy(float)),
            p_int, p_pre, p_post)
