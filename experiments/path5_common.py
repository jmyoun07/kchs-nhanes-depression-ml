# -*- coding: utf-8 -*-
"""
path5_common.py — 경로5(목적함수 변경) 공용 하네스.

설계 원칙(§2.10 강제):
  * 기준선 = 과제 명세 그대로:
      train_test_split(np.arange(len(kchs)), train_size=500000, test_size=100000,
                       stratify=y, random_state=42)
  * 전처리 = 명세 ColumnTransformer 그대로.
  * 모든 목적함수 변형은 **동일한 설계행렬 / 동일 split / 동일 트리 하이퍼파라미터**에서
    objective(손실)만 바꾼다.
  * 모델·기법 선택은 KCHS 내부(홀드아웃 + 5-fold CV)로만. NHANES는 최종 기록용.
  * NHANES 라벨은 학습·선택 어디에도 들어가지 않는다.
"""
import os, sys, json, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
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
NJOBS = 22

NUM = ["age", "sleep_hrs", "bmi", "income_q", "srh", "oral_health"]
BIN = ["partnered", "living_alone", "current_drinker", "active",
       "htn", "dm", "employed", "food_insecure"]
CAT = ["sex", "edu", "smoking"]
FEATURES = NUM + BIN + CAT
assert len(FEATURES) == 17

# 명세 LGBMClassifier 파라미터
SPEC = dict(n_estimators=600, learning_rate=0.03, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=SEED,
            n_jobs=-1, verbose=-1)

# 위와 동치인 native(lgb.train) 파라미터.
#  주의: sklearn 래퍼는 subsample_freq 기본 0 -> bagging 비활성.
#        따라서 native 에서도 bagging_freq=0 (bagging_fraction 무효)로 둔다.
NATIVE = dict(learning_rate=0.03, num_leaves=31,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=0,
              min_child_samples=20, min_child_weight=1e-3,
              min_split_gain=0.0, reg_alpha=0.0, reg_lambda=0.0,
              max_depth=-1, subsample_for_bin=200000,
              seed=SEED, num_threads=NJOBS, verbose=-1,
              deterministic=True, force_row_wise=True)
NROUND = 600


# ---------------------------------------------------------------- data
def load():
    k = pd.read_parquet(os.path.join(PROC, "ext_kchs.parquet"))
    pre = pd.read_parquet(os.path.join(PROC, "ext_pre.parquet"))
    post = pd.read_parquet(os.path.join(PROC, "ext_post.parquet"))
    return k, pre, post


def make_pre():
    num = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())])
    cat = Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                    ("o", OneHotEncoder(handle_unknown="ignore", drop="first"))])
    return ColumnTransformer([("num", num, NUM), ("bin", "passthrough", BIN),
                              ("cat", cat, CAT)])


def base_split(k):
    y = k["dep"].astype(int).values
    itr, iho = train_test_split(np.arange(len(k)), train_size=500000,
                                test_size=100000, stratify=y, random_state=SEED)
    return itr, iho


def wauc(y, p, w):
    return float(roc_auc_score(np.asarray(y).astype(int), np.asarray(p, float),
                               sample_weight=np.asarray(w, float)))


class Data:
    """설계행렬을 한 번만 만들어 재사용한다(모든 실험이 동일 입력을 보게)."""

    GRP_COLS = ["age", "sex", "year", "edu", "income_q"]

    def __init__(self, keep_full=False, dtype=np.float64):
        import gc
        k, pre, post = load()
        itr, iho = base_split(k)
        self.itr, self.iho = itr, iho
        y = k["dep"].astype(int).values
        W = k["W"].to_numpy(float)

        ct = make_pre()
        Xtr_df = k.iloc[itr][FEATURES]
        self.ct = ct.fit(Xtr_df)
        self.Xtr = np.asarray(ct.transform(Xtr_df), dtype=dtype)
        self.Xho = np.asarray(ct.transform(k.iloc[iho][FEATURES]), dtype=dtype)
        self.Xpre = np.asarray(ct.transform(pre[FEATURES]), dtype=dtype)
        self.Xpost = np.asarray(ct.transform(post[FEATURES]), dtype=dtype)
        self.ytr = y[itr].astype(np.float64)
        self.Wtr = W[itr]
        self.yho = y[iho]
        self.Who = W[iho]
        self.ypre = pre["dep"].astype(int).values
        self.wpre = pre["W"].to_numpy(float)
        self.ypost = post["dep"].astype(int).values
        self.wpost = post["W"].to_numpy(float)
        self.ktr = k.iloc[itr][self.GRP_COLS].copy()   # 그룹 정의용(연도/연령대 등)
        if keep_full:
            self.k, self.pre_df, self.post_df, self.y, self.W = k, pre, post, y, W
        else:                      # 메모리 절약: 대용량 원본 해제
            del k, pre, post, Xtr_df, y, W
            gc.collect()

    def feat_names(self):
        return [str(c) for c in self.ct.get_feature_names_out()]

    def evaluate_raw(self, f_ho, f_pre, f_post):
        return dict(internal=wauc(self.yho, f_ho, self.Who),
                    ext_pre=wauc(self.ypre, f_pre, self.wpre),
                    ext_post=wauc(self.ypost, f_post, self.wpost))


# ---------------------------------------------------------------- native trainer
def train_native(D, objective, fobj=None, params_extra=None, nround=NROUND,
                 Xtr=None, ytr=None, wtr=None, init_score=None, group=None):
    """objective: 문자열이면 native objective, fobj가 주어지면 커스텀."""
    p = dict(NATIVE)
    if params_extra:
        p.update(params_extra)
    X = D.Xtr if Xtr is None else Xtr
    yv = D.ytr if ytr is None else ytr
    if fobj is None:
        p["objective"] = objective
    else:
        p["objective"] = fobj
    ds = lgb.Dataset(X, label=yv, weight=wtr, free_raw_data=False,
                     init_score=init_score, group=group)
    booster = lgb.train(p, ds, num_boost_round=nround)
    return booster


def predict_raw(booster, X):
    return booster.predict(X, raw_score=True)


# ---------------------------------------------------------------- result logging
ROWS = []


def rec(name, internal, ext_pre, ext_post, compliant=True, note="",
        base=None, extra=None, quiet=False):
    r = dict(name=name, internal=round(float(internal), 6),
             ext_pre=round(float(ext_pre), 6), ext_post=round(float(ext_post), 6),
             protocol_compliant=bool(compliant), note=note)
    if base is not None:
        r["d_internal"] = round(float(internal) - base["internal"], 6)
        r["d_pre"] = round(float(ext_pre) - base["ext_pre"], 6)
        r["d_post"] = round(float(ext_post) - base["ext_post"], 6)
    if extra:
        r.update(extra)
    ROWS.append(r)
    if not quiet:
        print(json.dumps(r, ensure_ascii=False), flush=True)
    return r


def dump(fname, rows=None):
    df = pd.DataFrame(rows if rows is not None else ROWS)
    os.makedirs(RES, exist_ok=True)
    path = os.path.join(RES, fname)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print("saved", path, flush=True)
    return df


# ---------------------------------------------------------------- CV (KCHS only)
def cv_folds(y, n_splits=5, seed=SEED):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))
