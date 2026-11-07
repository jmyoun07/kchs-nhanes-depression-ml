# -*- coding: utf-8 -*-
"""경로E(공변량 이동 보정) 공통 하네스.

- 기준선: 과제 명세 그대로 (train_test_split(train=500000, test=100000, stratify=dep, seed=42))
- 평가: 가중 AUROC (roc_auc_score(y, p, sample_weight=W))
- NHANES 라벨은 절대 학습/선택에 쓰지 않는다. (§2.10)
"""
import os, sys, time, json, warnings
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, StratifiedKFold
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


def base_split(k, train_n=500000, test_n=100000, seed=SEED):
    y = k["dep"].astype(int).values
    itr, iho = train_test_split(np.arange(len(k)), train_size=train_n,
                                test_size=test_n, stratify=y, random_state=seed)
    return itr, iho


def wauc(y, p, w):
    return roc_auc_score(np.asarray(y).astype(int), p, sample_weight=np.asarray(w, float))


def fit_eval(k, pre, post, itr, iho, feats=FEATURES, num=NUM, bin_=BIN, cat=CAT,
             sw=None, params=None, kdf=None, predf=None, postf=None):
    """kdf/predf/postf: 전처리(변형)된 데이터프레임. 없으면 원본 사용."""
    kd = k if kdf is None else kdf
    pd_ = pre if predf is None else predf
    po_ = post if postf is None else postf
    p = dict(LGB_PARAMS)
    if params:
        p.update(params)
    pipe = Pipeline([("pre", make_pre(num, bin_, cat)),
                     ("clf", lgb.LGBMClassifier(**p))])
    y = k["dep"].astype(int).values
    kw = dict()
    if sw is not None:
        kw["clf__sample_weight"] = np.asarray(sw, float)
    pipe.fit(kd.iloc[itr][feats], y[itr], **kw)
    pi = pipe.predict_proba(kd.iloc[iho][feats])[:, 1]
    pp = pipe.predict_proba(pd_[feats])[:, 1]
    ps = pipe.predict_proba(po_[feats])[:, 1]
    return (wauc(y[iho], pi, k["W"].to_numpy(float)[iho]),
            wauc(pre["dep"], pp, pre["W"]),
            wauc(post["dep"], ps, post["W"]),
            pipe)


ROWS = []


def rec(name, internal, ext_pre, ext_post, compliant, note="", base=None):
    r = dict(name=name, internal=round(float(internal), 6),
             ext_pre=round(float(ext_pre), 6), ext_post=round(float(ext_post), 6),
             protocol_compliant=bool(compliant), note=note)
    if base is not None:
        r["delta_pre"] = round(float(ext_pre) - base[1], 6)
        r["delta_post"] = round(float(ext_post) - base[2], 6)
    ROWS.append(r)
    print(json.dumps(r, ensure_ascii=False), flush=True)
    return r


def dump(path):
    df = pd.DataFrame(ROWS)
    os.makedirs(RES, exist_ok=True)
    df.to_csv(os.path.join(RES, path), index=False, encoding="utf-8-sig")
    print("saved", os.path.join(RES, path))
    return df
