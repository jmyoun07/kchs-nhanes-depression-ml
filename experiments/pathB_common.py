# -*- coding: utf-8 -*-
"""
pathB_common.py — 경로B(알고리즘·하이퍼파라미터 대탐색) 공용 모듈.
§2.10 준수: 모든 모델 선택은 KCHS 내부 CV로만. NHANES는 최종 평가 1회.
"""
import os, sys, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score

ROOT = r"C:/Users/rudtn/Desktop/claude code/지역사회건강"
PROC = os.path.join(ROOT, "data/processed")
RES = os.path.join(ROOT, "results")
SEED = 42

NUM = ["age", "sleep_hrs", "bmi", "income_q", "srh", "oral_health"]
BIN = ["partnered", "living_alone", "current_drinker", "active", "htn", "dm",
       "employed", "food_insecure"]
CAT = ["sex", "edu", "smoking"]
FEATURES = NUM + BIN + CAT          # 17개 확정 예측변수


def load():
    k = pd.read_parquet(os.path.join(PROC, "ext_kchs.parquet"))
    pre = pd.read_parquet(os.path.join(PROC, "ext_pre.parquet"))
    post = pd.read_parquet(os.path.join(PROC, "ext_post.parquet"))
    return k, pre, post


def split(k):
    """기준선과 동일: train=500000, test=100000, stratify=dep, random_state=42"""
    y = k["dep"].astype(int).values
    itr, iho = train_test_split(np.arange(len(k)), train_size=500000, test_size=100000,
                                stratify=y, random_state=SEED)
    return itr, iho


def make_pre():
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore", drop="first"))])
    return ColumnTransformer([("num", num, NUM), ("bin", "passthrough", BIN), ("cat", cat, CAT)])


def feat_names():
    """make_pre 출력 컬럼 순서 (OneHot drop=first). KCHS 전체 레벨 기준."""
    names = list(NUM) + list(BIN)
    lv = {"sex": ["F", "M"], "edu": ["college", "elem_low", "high", "middle"],
          "smoking": ["current", "former", "never"]}
    for c in CAT:
        names += [f"{c}={L}" for L in lv[c][1:]]
    return names


def wauc(y, p, w):
    return roc_auc_score(y, p, sample_weight=w)
