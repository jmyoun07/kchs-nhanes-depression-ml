# -*- coding: utf-8 -*-
"""pathG_common.py - 경로2(공변량 분포만 쓰는 도메인 적응) 공통 하네스.

기준선은 과제 명세 그대로:
  train_test_split(np.arange(len(kchs)), train_size=500000, test_size=100000,
                   stratify=y, random_state=42)
평가는 복합표본 가중 AUROC.
NHANES 라벨(dep)은 학습/모델선택에 절대 사용하지 않는다 (§2.10).
NHANES **공변량**을 쓰는 기법은 protocol_compliant=False 로 표기하고 note에 근거를 남긴다.
"""
import os, json, time, warnings
import numpy as np
import pandas as pd

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
assert len(FEATURES) == 17

LGB_PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=31,
                  subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                  n_jobs=-1, verbose=-1)

_CACHE = {}


def load():
    if "d" not in _CACHE:
        k = pd.read_parquet(os.path.join(PROC, "ext_kchs.parquet"))
        pre = pd.read_parquet(os.path.join(PROC, "ext_pre.parquet"))
        post = pd.read_parquet(os.path.join(PROC, "ext_post.parquet"))
        _CACHE["d"] = (k, pre, post)
    return _CACHE["d"]


def make_ct(num=None, bin_=None, cat=None):
    num = NUM if num is None else list(num)
    bin_ = BIN if bin_ is None else list(bin_)
    cat = CAT if cat is None else list(cat)
    return ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")),
                          ("s", StandardScaler())]), num),
        ("bin", "passthrough", bin_),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", OneHotEncoder(handle_unknown="ignore",
                                              drop="first"))]), cat),
    ])


def base_split(k):
    y = k["dep"].astype(int).values
    tr, ho = train_test_split(np.arange(len(k)), train_size=500000,
                              test_size=100000, stratify=y, random_state=SEED)
    return tr, ho


def wauc(y, p, w):
    return roc_auc_score(np.asarray(y).astype(int), np.asarray(p, float),
                         sample_weight=np.asarray(w, float))


ROWS = []


def rec(name, internal, ext_pre, ext_post, compliant, note="", base=None,
        extra=None, secs=None):
    r = dict(name=name,
             internal=round(float(internal), 6),
             ext_pre=round(float(ext_pre), 6),
             ext_post=round(float(ext_post), 6),
             protocol_compliant=bool(compliant),
             note=note)
    if base is not None:
        r["d_internal"] = round(float(internal) - base[0], 6)
        r["d_pre"] = round(float(ext_pre) - base[1], 6)
        r["d_post"] = round(float(ext_post) - base[2], 6)
    if secs is not None:
        r["secs"] = round(float(secs), 1)
    if extra:
        r.update(extra)
    ROWS.append(r)
    print(json.dumps(r, ensure_ascii=False), flush=True)
    return r


def dump(fname):
    os.makedirs(RES, exist_ok=True)
    p = os.path.join(RES, fname)
    pd.DataFrame(ROWS).to_csv(p, index=False, encoding="utf-8-sig")
    print("SAVED", p, flush=True)
    return p
