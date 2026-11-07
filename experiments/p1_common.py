# -*- coding: utf-8 -*-
"""경로1 공통 하네스 — KCHS 내부 정보만으로 '전이 강건 설정'을 고르는 절차.

원칙(§2.10): NHANES 라벨은 설정 선택에 절대 사용하지 않는다.
  - 모든 선택은 KCHS 내부 CV(LOYO 연도 / 지역블록)로만 수행.
  - NHANES 평가는 사전 선언된 선택규칙이 뽑은 설정에 대해 최종 1회.

기준선(literal spec):
  train_test_split(np.arange(len(kchs)), train_size=500000, test_size=100000,
                   stratify=y, random_state=42)
  → 내부 0.8310 / pre 0.7740 / post 0.7653
"""
import os
import sys
import json
import time
import warnings

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
CACHE = os.path.join(ROOT, "experiments", "_p1cache")
SEED = 42

NUM = ["age", "sleep_hrs", "bmi", "income_q", "srh", "oral_health"]
BIN = ["partnered", "living_alone", "current_drinker", "active",
       "htn", "dm", "employed", "food_insecure"]
CAT = ["sex", "edu", "smoking"]
FEATURES = NUM + BIN + CAT

LGB_BASE = dict(n_estimators=600, learning_rate=0.03, num_leaves=31,
                subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                n_jobs=-1, verbose=-1)


# ------------------------------------------------------------------ data
def load():
    k = pd.read_parquet(os.path.join(PROC, "ext_kchs.parquet"))
    pre = pd.read_parquet(os.path.join(PROC, "ext_pre.parquet"))
    post = pd.read_parquet(os.path.join(PROC, "ext_post.parquet"))
    return k, pre, post


def make_pre():
    npipe = Pipeline([("i", SimpleImputer(strategy="median")),
                      ("s", StandardScaler())])
    cpipe = Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                      ("o", OneHotEncoder(handle_unknown="ignore", drop="first"))])
    return ColumnTransformer([("num", npipe, NUM),
                              ("bin", "passthrough", BIN),
                              ("cat", cpipe, CAT)])


def wauc(y, p, w):
    return roc_auc_score(np.asarray(y).astype(int), np.asarray(p, float),
                         sample_weight=np.asarray(w, float))


def base_split(n, y, train_n=500000, test_n=100000, seed=SEED):
    return train_test_split(np.arange(n), train_size=train_n, test_size=test_n,
                            stratify=y, random_state=seed)


# --------------------------------------------------- cached design matrix
def build_cache(force=False):
    """전처리기를 KCHS 전체에 1회 적합 → 설계행렬 캐시.

    CV 스크리닝 전용 근사(모든 설정에 동일 적용). 최종 NHANES 평가는
    기준선 파이프라인(폴드 내 적합)을 그대로 쓴다.
    """
    os.makedirs(CACHE, exist_ok=True)
    fx = os.path.join(CACHE, "Xk.npy")
    if os.path.exists(fx) and not force:
        return
    k, pre, post = load()
    ct = make_pre()
    Xk = ct.fit_transform(k[FEATURES]).astype(np.float32)
    Xp = ct.transform(pre[FEATURES]).astype(np.float32)
    Xs = ct.transform(post[FEATURES]).astype(np.float32)
    np.save(fx, Xk)
    np.save(os.path.join(CACHE, "Xpre.npy"), Xp)
    np.save(os.path.join(CACHE, "Xpost.npy"), Xs)
    np.save(os.path.join(CACHE, "yk.npy"), k["dep"].astype(int).to_numpy())
    np.save(os.path.join(CACHE, "Wk.npy"), k["W"].to_numpy(float))
    np.save(os.path.join(CACHE, "year.npy"), k["year"].to_numpy(int))
    np.save(os.path.join(CACHE, "strata.npy"), k["strata"].astype(int).to_numpy())
    meta = dict(cols=[str(c) for c in ct.get_feature_names_out()],
                n=int(len(k)), npre=int(len(pre)), npost=int(len(post)))
    json.dump(meta, open(os.path.join(CACHE, "meta.json"), "w"), ensure_ascii=False)
    print("cache built", Xk.shape, Xp.shape, Xs.shape)


class Cache:
    def __init__(self):
        build_cache()
        self.X = np.load(os.path.join(CACHE, "Xk.npy"))
        self.Xpre = np.load(os.path.join(CACHE, "Xpre.npy"))
        self.Xpost = np.load(os.path.join(CACHE, "Xpost.npy"))
        self.y = np.load(os.path.join(CACHE, "yk.npy"))
        self.W = np.load(os.path.join(CACHE, "Wk.npy"))
        self.year = np.load(os.path.join(CACHE, "year.npy"))
        self.strata = np.load(os.path.join(CACHE, "strata.npy"))
        self.meta = json.load(open(os.path.join(CACHE, "meta.json")))
        self.cols = self.meta["cols"]

    # ---- fold definitions (KCHS 내부 정보만 사용) ----
    def year_folds(self):
        return [(int(v), np.flatnonzero(self.year == v)) for v in np.unique(self.year)]

    def region_folds(self, k=5):
        """층(strata) 코드 연속 블록 = 행정순 지역 대리. n이 균등하도록 분할."""
        order = np.argsort(self.strata, kind="stable")
        s_sorted = self.strata[order]
        cuts = np.linspace(0, len(order), k + 1).astype(int)
        folds = []
        for i in range(k):
            lo, hi = cuts[i], cuts[i + 1]
            # 층 경계에서 자르기(같은 층이 두 폴드에 걸치지 않도록)
            if hi < len(order):
                v = s_sorted[hi - 1]
                while hi < len(order) and s_sorted[hi] == v:
                    hi += 1
                cuts[i + 1] = hi
            folds.append((f"R{i}", order[cuts[i]:cuts[i + 1]]))
        return folds


# ------------------------------------------------------------------ model
def make_clf(spec):
    p = dict(LGB_BASE)
    p.update(spec.get("params", {}))
    if spec.get("mono"):
        cols = spec["_cols"]
        mono = []
        for c in cols:
            hit = 0
            for kk, vv in spec["mono"].items():
                if c.endswith("__" + kk) or c == kk:
                    hit = vv
            mono.append(hit)
        p["monotone_constraints"] = mono
    return lgb.LGBMClassifier(**p)


def fit_predict(spec, Xtr, ytr, Xevs, wtr=None):
    spec = dict(spec)
    clf = make_clf(spec)
    if wtr is None:
        clf.fit(Xtr, ytr)
    else:
        clf.fit(Xtr, ytr, sample_weight=wtr)
    return [clf.predict_proba(Xe)[:, 1] for Xe in Xevs], clf


def subsample(idx, n, rng):
    if n is None or n >= len(idx):
        return idx
    return rng.choice(idx, size=n, replace=False)


_FOLDCACHE = {}


def fold_index(C, mode, train_n, k_region=5, eval_n=None, seed=SEED):
    """폴드별 (train_idx, eval_idx). 모든 설정에서 동일 인덱스 재사용(대응비교)."""
    key = (mode, train_n, k_region, eval_n, seed)
    if key in _FOLDCACHE:
        return _FOLDCACHE[key]
    folds = C.year_folds() if mode == "year" else C.region_folds(k_region)
    rng = np.random.default_rng(seed)
    out = []
    allidx = np.arange(len(C.y))
    for name, te in folds:
        m = np.ones(len(C.y), bool)
        m[te] = False
        tr = subsample(allidx[m], train_n, rng)
        ev = subsample(te, eval_n, rng) if eval_n else te
        out.append((str(name), np.sort(tr), np.sort(ev)))
    _FOLDCACHE[key] = out
    return out


def cv_run(C, spec, mode, train_n, seed=SEED, k_region=5, eval_n=None, verbose=False):
    """mode: 'year' | 'region'. 반환: per-fold 가중 AUROC dict + 요약."""
    spec = dict(spec)
    spec["_cols"] = C.cols
    out = {}
    for name, tr, te in fold_index(C, mode, train_n, k_region, eval_n, seed):
        t = time.time()
        w = C.W[tr] if spec.get("sw") == "survey" else None
        (p,), _ = fit_predict(spec, C.X[tr], C.y[tr], [C.X[te]], wtr=w)
        out[str(name)] = round(float(wauc(C.y[te], p, C.W[te])), 5)
        if verbose:
            print(f"    {mode}:{name} {out[str(name)]:.5f} ({time.time()-t:.0f}s)", flush=True)
    v = np.array(list(out.values()), float)
    return dict(per_fold=out, mean=float(v.mean()), min=float(v.min()),
                max=float(v.max()), std=float(v.std(ddof=0)))


ROWS = []


def rec(rows, **kw):
    rows.append(kw)
    print(json.dumps(kw, ensure_ascii=False, default=str), flush=True)


def dump(rows, path):
    os.makedirs(RES, exist_ok=True)
    df = pd.DataFrame(rows)
    fp = os.path.join(RES, path)
    df.to_csv(fp, index=False, encoding="utf-8-sig")
    print("saved", fp, flush=True)
    return df
