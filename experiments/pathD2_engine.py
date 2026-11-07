# -*- coding: utf-8 -*-
"""
pathD2_engine.py — 경로D 실험 엔진.

핵심 규약(§2.10):
  * 모든 모델 선택 지표는 KCHS 내부 CV(LOYO=연도, GroupKFold=지역strata)에서만 나온다.
  * NHANES는 predict만, 기록용. 어떤 선택 함수에도 외부 성능이 들어가지 않는다.
"""
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
ROOT = r"C:/Users/rudtn/Desktop/claude code/지역사회건강"
RES = os.path.join(ROOT, "results")
SEED = 42

LGB_BASE = dict(n_estimators=600, learning_rate=0.03, num_leaves=31,
                subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                n_jobs=5, verbose=-1)

_D = None


class D:
    """프로세스별 memmap 데이터."""

    def __init__(self):
        m = np.load(os.path.join(CACHE, "meta.npz"), allow_pickle=False)
        self.Z = np.load(os.path.join(CACHE, "Z.npy"), mmap_mode="r")
        self.Zpre = np.load(os.path.join(CACHE, "Zpre.npy"))
        self.Zpost = np.load(os.path.join(CACHE, "Zpost.npy"))
        self.y = m["y"]
        self.W = m["W"]
        self.year = m["year"]
        self.strata = m["strata"]
        self.itr = m["itr"]
        self.iho = m["iho"]
        self.ypre = m["ypre"]
        self.wpre = m["wpre"]
        self.ypost = m["ypost"]
        self.wpost = m["wpost"]
        self.cols = list(m["cols"])
        self.features = list(m["features"])
        self.years = np.unique(self.year)
        # 원변수 -> 설계열 인덱스
        self.fmap = {}
        for f in self.features:
            ix = [i for i, c in enumerate(self.cols)
                  if c == f or c.startswith(f + "_")]
            self.fmap[f] = ix
        # 지역 클러스터(strata를 K개 환경으로 축약) — 지역 DRO/그룹CV용
        uq, inv = np.unique(self.strata, return_inverse=True)
        rs = np.random.RandomState(SEED)
        self.strata_cluster = rs.randint(0, 50, size=len(uq))[inv]
        self.strata_fold5 = rs.randint(0, 5, size=len(uq))[inv]


def data():
    global _D
    if _D is None:
        _D = D()
    return _D


def wauc(y, p, w):
    return roc_auc_score(y, p, sample_weight=w)


# ------------------------------------------------------------- 모델 빌더
def _mono_vector(d, spec):
    """단조 제약 벡터. spec['mono'] = dict(feature -> +1/-1)."""
    v = np.zeros(len(d.cols), int)
    for f, s in spec.get("mono", {}).items():
        for i in d.fmap.get(f, []):
            v[i] = s
    return list(v)


def _cols_for(d, spec):
    """spec['keep'] (원변수 리스트)에 해당하는 설계열 인덱스."""
    keep = spec.get("keep")
    if keep is None:
        return np.arange(len(d.cols))
    ix = []
    for f in keep:
        ix.extend(d.fmap[f])
    return np.array(sorted(set(ix)))


def _sample_weight(d, idx, spec):
    """학습 표본가중. 'none'|'survey'(W)|배열."""
    sw = spec.get("sw", "none")
    if sw == "none":
        return None
    if sw == "survey":
        w = d.W[idx]
        return w / w.mean()
    if isinstance(sw, np.ndarray):
        return sw
    raise ValueError(sw)


def build_fit(d, Xtr, ytr, spec, wtr=None):
    fam = spec.get("family", "lgb")
    if fam == "lgb":
        p = dict(LGB_BASE)
        p.update(spec.get("params", {}))
        if spec.get("mono"):
            p["monotone_constraints"] = _mono_vector(d, spec)
            p["monotone_constraints_method"] = spec.get("mono_method", "basic")
        m = lgb.LGBMClassifier(**p)
        m.fit(Xtr, ytr, sample_weight=wtr,
              feature_name=[d.cols[i] for i in spec["_ix"]])
        return m
    if fam == "logit":
        X = np.nan_to_num(Xtr, nan=0.0)
        p = dict(C=1.0, max_iter=200, solver="lbfgs", n_jobs=5)
        p.update(spec.get("params", {}))
        m = LogisticRegression(**p)
        m.fit(X, ytr, sample_weight=wtr)
        m._nanfix = True
        return m
    raise ValueError(fam)


def _pred(m, X):
    if getattr(m, "_nanfix", False):
        X = np.nan_to_num(X, nan=0.0)
    return m.predict_proba(X)[:, 1]


# ------------------------------------------------------------- DRO
def dro_weights(d, idx, spec, groups, seed=SEED):
    """Group-DRO 근사: 초기 모델로 그룹별 손실 산출 -> 최악 그룹 지수 상향 가중 -> 반복.
    KCHS 라벨만 사용. 그룹은 연도 또는 지역클러스터."""
    rounds = spec.get("dro_rounds", 2)
    eta = spec.get("dro_eta", 1.0)
    n_sub = spec.get("dro_fit_n", 150000)
    ix = spec["_ix"]
    g = groups
    uq = np.unique(g)
    q = np.ones(len(uq)) / len(uq)
    w = np.ones(len(idx))
    rs = np.random.RandomState(seed)
    for _ in range(rounds):
        s = rs.choice(len(idx), size=min(n_sub, len(idx)), replace=False)
        base = dict(spec)
        base.pop("sw", None)
        m = build_fit(d, np.asarray(d.Z[idx[s]][:, ix]), d.y[idx[s]], base,
                      wtr=w[s] / w[s].mean())
        p = np.clip(_pred(m, np.asarray(d.Z[idx][:, ix])), 1e-6, 1 - 1e-6)
        yy = d.y[idx]
        loss = -(yy * np.log(p) + (1 - yy) * np.log(1 - p))
        gl = np.array([loss[g == u].mean() for u in uq])
        gl = (gl - gl.mean()) / (gl.std() + 1e-9)
        q = q * np.exp(eta * gl)
        q = q / q.sum()
        w = (q * len(uq))[np.searchsorted(uq, g)]
    return w / w.mean()


# ------------------------------------------------------------- 폴드 실행
def run_fold(spec, tr, va, tag):
    d = data()
    ix = _cols_for(d, spec)
    spec = dict(spec)
    spec["_ix"] = ix
    if spec.get("train_n"):
        rs = np.random.RandomState(spec.get("seed", SEED))
        if len(tr) > spec["train_n"]:
            tr = rs.choice(tr, size=spec["train_n"], replace=False)
    tr = np.sort(tr)
    Xtr = np.asarray(d.Z[tr][:, ix])
    ytr = d.y[tr]

    sw = spec.get("sw", "none")
    if isinstance(sw, str) and sw.startswith("dro"):
        gsel = spec.get("dro_groups", "year")
        g = d.year[tr] if gsel == "year" else d.strata_cluster[tr]
        wtr = dro_weights(d, tr, spec, g)
    else:
        wtr = _sample_weight(d, tr, spec)

    m = build_fit(d, Xtr, ytr, spec, wtr=wtr)
    va = np.sort(va)
    p = _pred(m, np.asarray(d.Z[va][:, ix]))
    return float(wauc(d.y[va], p, d.W[va])), tag


def cv_eval(spec, scheme="loyo", n_splits=5, seed=SEED, n_workers=4):
    """KCHS 내부 CV. scheme: 'loyo'(연도) | 'strata'(지역 GroupKFold)."""
    from joblib import Parallel, delayed
    d = data()
    pool = d.itr
    jobs = []
    if scheme == "loyo":
        yr = d.year[pool]
        for yv in d.years:
            m = yr == yv
            jobs.append((pool[~m], pool[m], int(yv)))
    elif scheme == "strata":
        f = d.strata_fold5[pool]
        for k in range(n_splits):
            m = f == k
            jobs.append((pool[~m], pool[m], f"s{k}"))
    else:
        raise ValueError(scheme)
    out = Parallel(n_jobs=n_workers, backend="loky")(
        delayed(run_fold)(spec, tr, va, tag) for tr, va, tag in jobs)
    s = np.array([o[0] for o in out], float)
    return dict(mean=float(s.mean()), std=float(s.std()),
                min=float(s.min()), max=float(s.max()),
                gap=float(s.max() - s.min()),
                per_fold={str(o[1]): round(o[0], 5) for o in out})


def final_eval(spec, tag, train_n=None, seed=SEED):
    """전체 train pool로 적합 후 내부 홀드아웃 + NHANES(기록용) 평가."""
    d = data()
    ix = _cols_for(d, spec)
    spec = dict(spec)
    spec["_ix"] = ix
    tr = d.itr
    if train_n and len(tr) > train_n:
        tr = np.sort(np.random.RandomState(seed).choice(tr, train_n, replace=False))
    Xtr = np.asarray(d.Z[tr][:, ix])
    sw = spec.get("sw", "none")
    if isinstance(sw, str) and sw.startswith("dro"):
        g = d.year[tr] if spec.get("dro_groups", "year") == "year" else d.strata_cluster[tr]
        wtr = dro_weights(d, tr, spec, g)
    else:
        wtr = _sample_weight(d, tr, spec)
    m = build_fit(d, Xtr, d.y[tr], spec, wtr=wtr)
    r = dict(name=tag)
    r["internal"] = float(wauc(d.y[d.iho], _pred(m, np.asarray(d.Z[d.iho][:, ix])),
                               d.W[d.iho]))
    r["ext_pre"] = float(wauc(d.ypre, _pred(m, d.Zpre[:, ix]), d.wpre))
    r["ext_post"] = float(wauc(d.ypost, _pred(m, d.Zpost[:, ix]), d.wpost))
    return r, m


def line(tag, cv=None, ext=None):
    s = f"{tag:36s}"
    if cv:
        s += f" CVmean={cv['mean']:.4f} CVmin={cv['min']:.4f} CVstd={cv['std']:.4f}"
    if ext:
        s += (f" | 내부={ext['internal']:.4f} pre={ext['ext_pre']:.4f}"
              f" post={ext['ext_post']:.4f}")
    return s
