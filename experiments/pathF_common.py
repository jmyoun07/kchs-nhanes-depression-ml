# -*- coding: utf-8 -*-
"""
pathF_common.py — 경로F(타깃·표본 정의 재검토) 공용 하네스.

핵심 설계:
- KCHS 내부 홀드아웃(iho)은 **모든 실험에서 고정**(20+, dep=PHQ9>=10, 전 연도).
  → 학습표본 정의를 바꿔도 내부 성능이 사과-대-사과로 비교됨.
- NHANES(pre/post)는 재적합 없이 predict만. 라벨은 평가에만 사용(§2.10).
- 모델 선택은 KCHS 내부(CV 또는 고정 홀드아웃)로만. 외부는 기록만 하고 선택에 쓰지 않음.
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


def load():
    k = pd.read_parquet(os.path.join(PROC, "ext_kchs.parquet"))
    pre = pd.read_parquet(os.path.join(PROC, "ext_pre.parquet"))
    post = pd.read_parquet(os.path.join(PROC, "ext_post.parquet"))
    return k, pre, post


def income_q_trainonly(df, is_tr):
    """연도별 소득 5분위 cutpoint를 train 행에서만 학습 (누수 제거)."""
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


def make_pre():
    """전처리: 수치형 median대체+표준화 / 이진 passthrough(NaN유지) / 범주형 최빈값+OneHot(drop=first)"""
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore", drop="first"))])
    return ColumnTransformer([("num", num, NUM), ("bin", "passthrough", BIN), ("cat", cat, CAT)])


def make_pipe(estimator=None, **lgb_kw):
    params = dict(LGB_BASE)
    params.update(lgb_kw)
    est = estimator if estimator is not None else lgb.LGBMClassifier(**params)
    return Pipeline([("pre", make_pre()), ("clf", est)])


def wauc(y, p, w):
    return roc_auc_score(y, p, sample_weight=w)


class Bench:
    """고정 내부 홀드아웃 + 고정 외부 평가셋."""

    def __init__(self, holdout_frac=0.2, seed=SEED):
        k, pre, post = load()
        y = k["dep"].astype(int).values
        itr, iho = train_test_split(np.arange(len(k)), test_size=holdout_frac,
                                    stratify=y, random_state=seed)
        is_tr = np.zeros(len(k), bool); is_tr[itr] = True
        k = k.copy()
        k["income_q"] = income_q_trainonly(k, is_tr)
        self.k, self.pre, self.post = k, pre, post
        self.y, self.W = y, k["W"].to_numpy(float)
        self.itr, self.iho = itr, iho
        self.ypre = pre["dep"].astype(int).values
        self.wpre = pre["W"].to_numpy(float)
        self.ypost = post["dep"].astype(int).values
        self.wpost = post["W"].to_numpy(float)

    # ---- 학습표본 선택자 ----
    def train_idx(self, mask_fn=None, n=None, seed=SEED):
        idx = self.itr
        if mask_fn is not None:
            m = mask_fn(self.k.iloc[idx])
            idx = idx[np.asarray(m, bool)]
        if n is not None and n < len(idx):
            idx = np.random.RandomState(seed).choice(idx, size=n, replace=False)
        return idx

    def evaluate(self, pipe, tag, tr_n, extra=None, is_regressor=False, note=""):
        X = self.k[FEATURES]
        if is_regressor:
            pi = pipe.predict(X.iloc[self.iho])
            pp = pipe.predict(self.pre[FEATURES])
            po = pipe.predict(self.post[FEATURES])
        else:
            pi = pipe.predict_proba(X.iloc[self.iho])[:, 1]
            pp = pipe.predict_proba(self.pre[FEATURES])[:, 1]
            po = pipe.predict_proba(self.post[FEATURES])[:, 1]
        r = dict(name=tag, train_n=int(tr_n),
                 internal=wauc(self.y[self.iho], pi, self.W[self.iho]),
                 ext_pre=wauc(self.ypre, pp, self.wpre),
                 ext_post=wauc(self.ypost, po, self.wpost),
                 note=note)
        if extra:
            r.update(extra)
        return r, (pi, pp, po)


def fmt(r, base=None):
    s = (f"{r['name']:34s} n={r['train_n']:>9,d} "
         f"내부={r['internal']:.4f} pre={r['ext_pre']:.4f} post={r['ext_post']:.4f}")
    if base:
        s += (f"  Δ내부={r['internal']-base['internal']:+.4f} "
              f"Δpre={r['ext_pre']-base['ext_pre']:+.4f} Δpost={r['ext_post']-base['ext_post']:+.4f}")
    return s
