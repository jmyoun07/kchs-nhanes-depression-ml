# -*- coding: utf-8 -*-
"""
path3_common.py — 경로3(학습 표본·설계 재구성) 공용 하네스.

설계 원칙(§2.10 준수):
  - 학습 및 모델선택은 KCHS 라벨/내부정보만 사용.
  - NHANES(pre/post)는 재적합 없이 predict만. 외부 수치는 '기록만' 하고
    어떤 선택 규칙의 입력도 되지 않는다.
  - 회색지대(외부 공변량 분포 사용) 기법은 protocol_compliant=False로 표기.

기준선(과제 지정, 그대로 재현):
  tr, ho = train_test_split(arange(n), train_size=500000, test_size=100000,
                            stratify=y, random_state=42)
  → 내부 0.8310 / 외부 pre 0.7740 / post 0.7653

비교 설계:
  - 내부 홀드아웃 ho(100,000)는 **모든 실험에서 고정** → 사과-대-사과.
  - 학습표본은 POOL = (전체 - ho) = 1,948,510 에서만 뽑는다.
    기준선 tr(500k) ⊂ POOL.
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

TARGET = dict(internal=0.8310, ext_pre=0.7740, ext_post=0.7653)


def load():
    k = pd.read_parquet(os.path.join(PROC, "ext_kchs.parquet"))
    pre = pd.read_parquet(os.path.join(PROC, "ext_pre.parquet"))
    post = pd.read_parquet(os.path.join(PROC, "ext_post.parquet"))
    return k, pre, post


def make_pre(num=None, binv=None, cat=None):
    n = Pipeline([("i", SimpleImputer(strategy="median")),
                  ("s", StandardScaler())])
    c = Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                  ("o", OneHotEncoder(handle_unknown="ignore", drop="first"))])
    return ColumnTransformer([("num", n, list(NUM if num is None else num)),
                              ("bin", "passthrough", list(BIN if binv is None else binv)),
                              ("cat", c, list(CAT if cat is None else cat))])


def wauc(y, p, w):
    return roc_auc_score(y, p, sample_weight=w)


class Bench:
    """기준선 500k/100k 분할 + 고정 내부 홀드아웃 + 고정 외부 평가셋."""

    def __init__(self, seed=SEED):
        k, pre, post = load()
        y = k["dep"].astype(int).values
        tr, ho = train_test_split(np.arange(len(k)), train_size=500000,
                                  test_size=100000, stratify=y,
                                  random_state=seed)
        self.k, self.pre, self.post = k, pre, post
        self.y = y
        self.W = k["W"].to_numpy(float)
        self.year = k["year"].to_numpy(int)
        self.age = k["age"].to_numpy(float)
        self.strata = k["strata"].astype(str).to_numpy()
        self.tr, self.ho = tr, ho
        # POOL = 홀드아웃 제외 전부 (학습표본은 여기서만 뽑는다)
        mask = np.ones(len(k), bool)
        mask[ho] = False
        self.pool = np.where(mask)[0]
        self.X = k[FEATURES]
        self.Xho = self.X.iloc[ho]
        self.yho = y[ho]
        self.who = self.W[ho]
        self.Xpre = pre[FEATURES]
        self.ypre = pre["dep"].astype(int).values
        self.wpre = pre["W"].to_numpy(float)
        self.Xpost = post[FEATURES]
        self.ypost = post["dep"].astype(int).values
        self.wpost = post["W"].to_numpy(float)

    # ---------------------------------------------------------- fit/eval
    def run(self, tag, tr_idx, sw=None, est=None, X=None, Xho=None,
            Xpre=None, Xpost=None, ylab=None, note="", **lgbkw):
        """tr_idx로 학습 → 고정 ho + NHANES pre/post 평가. 외부는 기록만."""
        t0 = time.time()
        X = self.X if X is None else X
        Xho = self.Xho if Xho is None else Xho
        Xpre = self.Xpre if Xpre is None else Xpre
        Xpost = self.Xpost if Xpost is None else Xpost
        if est is None:
            p = dict(LGB_BASE)
            p.update(lgbkw)
            est = lgb.LGBMClassifier(**p)
        pipe = Pipeline([("pre", make_pre()), ("clf", est)])
        yv = (self.y if ylab is None else ylab)[tr_idx]
        if sw is not None:
            pipe.fit(X.iloc[tr_idx], yv, clf__sample_weight=sw)
        else:
            pipe.fit(X.iloc[tr_idx], yv)
        pi = pipe.predict_proba(Xho)[:, 1]
        pp = pipe.predict_proba(Xpre)[:, 1]
        po = pipe.predict_proba(Xpost)[:, 1]
        r = dict(name=tag, train_n=int(len(tr_idx)),
                 internal=wauc(self.yho, pi, self.who),
                 ext_pre=wauc(self.ypre, pp, self.wpre),
                 ext_post=wauc(self.ypost, po, self.wpost),
                 secs=round(time.time() - t0, 1), note=note)
        return r, (pi, pp, po), pipe


class Recorder:
    def __init__(self, out_csv):
        self.out = out_csv
        self.rows = []
        self.base = None

    def log(self, s):
        print(s)
        sys.stdout.flush()

    def add(self, r, block="", compliant=True, base=None, extra=""):
        r = dict(r)
        r["block"] = block
        r["protocol_compliant"] = bool(compliant)
        b = self.base if base is None else base
        if b is not None:
            r["d_internal"] = r["internal"] - b["internal"]
            r["d_pre"] = r["ext_pre"] - b["ext_pre"]
            r["d_post"] = r["ext_post"] - b["ext_post"]
        else:
            r["d_internal"] = r["d_pre"] = r["d_post"] = 0.0
        if extra:
            r["note"] = (str(r.get("note", "")) + " | " + extra).strip(" |")
        self.rows.append(r)
        pd.DataFrame(self.rows).to_csv(self.out, index=False, encoding="utf-8-sig")
        s = (f"[{block}] {r['name']:34s} n={r['train_n']:>9,d} "
             f"int={r['internal']:.4f} pre={r['ext_pre']:.4f} post={r['ext_post']:.4f}")
        if b is not None:
            s += f"  D={r['d_internal']:+.4f}/{r['d_pre']:+.4f}/{r['d_post']:+.4f}"
        if not r["protocol_compliant"]:
            s += "  [GRAY]"
        self.log(s)
        return r
