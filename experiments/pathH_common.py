# -*- coding: utf-8 -*-
"""
pathH_common.py — 경로4(다양성 기반 앙상블) 공용 하네스.

설계 원칙(§2.10 엄수):
  - fit 은 KCHS 만. NHANES 라벨(dep)은 어떤 fit/모델선택/가중치결정에도 사용하지 않는다.
  - 모든 모델선택·앙상블가중치는 KCHS train(500k) 내부 5-fold OOF 로만 결정.
  - 내부 holdout(100k) 과 NHANES pre/post 는 '평가 전용'.
  - 확정 17 예측변수 고정.

기준선 정의(과제 명세 문자 그대로):
  tr, ho = train_test_split(np.arange(len(kchs)), train_size=500000,
                            test_size=100000, stratify=y, random_state=42)
  income_q 는 parquet 값 그대로 사용(as-is).
  → 목표: 내부 0.8310 / pre 0.7740 / post 0.7653
"""
import os, sys, time, json, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score

ROOT = r"C:/Users/rudtn/Desktop/claude code/지역사회건강"
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache", "pathH")
os.makedirs(CACHE, exist_ok=True)
os.makedirs(RES, exist_ok=True)
SEED = 42
NFOLD = 5

NUM = ["age", "sleep_hrs", "bmi", "income_q", "srh", "oral_health"]
BIN = ["partnered", "living_alone", "current_drinker", "active",
       "htn", "dm", "employed", "food_insecure"]
CAT = ["sex", "edu", "smoking"]
FEATURES = NUM + BIN + CAT
assert len(FEATURES) == 17

LGB_BASE = dict(n_estimators=600, learning_rate=0.03, num_leaves=31,
                subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                n_jobs=-1, verbose=-1)


class Data:
    pass


_C = {}


def load():
    if "d" in _C:
        return _C["d"]
    k = pd.read_parquet(os.path.join(PROC, "ext_kchs.parquet"))
    pre = pd.read_parquet(os.path.join(PROC, "ext_pre.parquet"))
    post = pd.read_parquet(os.path.join(PROC, "ext_post.parquet"))
    y = k["dep"].astype(int).values
    W = k["W"].to_numpy(float)
    tr, ho = train_test_split(np.arange(len(k)), train_size=500000,
                              test_size=100000, stratify=y, random_state=SEED)
    d = Data()
    d.X = k[FEATURES].reset_index(drop=True)
    d.y, d.W, d.tr, d.ho = y, W, tr, ho
    d.Xtr = d.X.iloc[tr].reset_index(drop=True)
    d.ytr, d.Wtr = y[tr], W[tr]
    d.year_tr = k["year"].values[tr]
    d.strata_tr = k["strata"].astype(str).values[tr]
    d.Xho = d.X.iloc[ho].reset_index(drop=True)
    d.yho, d.Who = y[ho], W[ho]
    d.Xpre = pre[FEATURES].reset_index(drop=True)
    d.ypre = pre["dep"].astype(int).values
    d.wpre = pre["W"].to_numpy(float)
    d.Xpost = post[FEATURES].reset_index(drop=True)
    d.ypost = post["dep"].astype(int).values
    d.wpost = post["W"].to_numpy(float)
    skf = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=SEED)
    d.folds = list(skf.split(np.zeros(len(d.ytr)), d.ytr))
    _C["d"] = d
    return d


# ------------------------------------------------------- preprocessing
def make_pre(cols=None, bin_impute=False):
    cols = FEATURES if cols is None else list(cols)
    num = [c for c in NUM if c in cols]
    binc = [c for c in BIN if c in cols]
    cat = [c for c in CAT if c in cols]
    blocks = []
    if num:
        blocks.append(("num", Pipeline([("i", SimpleImputer(strategy="median")),
                                        ("s", StandardScaler())]), num))
    if binc:
        bp = (Pipeline([("i", SimpleImputer(strategy="most_frequent"))])
              if bin_impute else "passthrough")
        blocks.append(("bin", bp, binc))
    if cat:
        blocks.append(("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                                        ("o", OneHotEncoder(handle_unknown="ignore",
                                                            drop="first"))]), cat))
    return ColumnTransformer(blocks)


def make_pipe(est, cols=None, bin_impute=False):
    return Pipeline([("pre", make_pre(cols, bin_impute)), ("clf", est)])


# ------------------------------------------------------------- metrics
def wauc(y, p, w):
    return float(roc_auc_score(np.asarray(y).astype(int), np.asarray(p, float),
                               sample_weight=np.asarray(w, float)))


def rank01(p):
    s = pd.Series(np.asarray(p, float)).rank(method="average").to_numpy()
    return (s - 1) / max(len(s) - 1, 1)


def logit(p, eps=1e-7):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


# --------------------------------------------------------------- cache
def cpath(name):
    return os.path.join(CACHE, name + ".npz")


def have(name):
    return os.path.exists(cpath(name))


def save_preds(name, oof, ho, pre, post, secs=0.0, note=""):
    np.savez_compressed(cpath(name),
                        oof=np.asarray(oof, np.float32),
                        ho=np.asarray(ho, np.float32),
                        pre=np.asarray(pre, np.float32),
                        post=np.asarray(post, np.float32),
                        secs=np.array([float(secs)]), note=np.array([note]))


def load_preds(name):
    z = np.load(cpath(name), allow_pickle=True)
    return dict(oof=z["oof"].astype(float), ho=z["ho"].astype(float),
                pre=z["pre"].astype(float), post=z["post"].astype(float),
                secs=float(z["secs"][0]), note=str(z["note"][0]))


def list_cached():
    return sorted(os.path.splitext(f)[0] for f in os.listdir(CACHE) if f.endswith(".npz"))


def metrics(name, oof, ho, pre, post, secs=0.0, note=""):
    d = load()
    return dict(name=name,
                oof=wauc(d.ytr, oof, d.Wtr),
                internal=wauc(d.yho, ho, d.Who),
                ext_pre=wauc(d.ypre, pre, d.wpre),
                ext_post=wauc(d.ypost, post, d.wpost),
                secs=round(float(secs), 1), note=note)


def show(r, base=None):
    s = (f"{r['name']:30s} OOF={r['oof']:.4f} 내부={r['internal']:.4f} "
         f"pre={r['ext_pre']:.4f} post={r['ext_post']:.4f}")
    if base:
        s += (f"  Δpre={r['ext_pre']-base['ext_pre']:+.4f} "
              f"Δpost={r['ext_post']-base['ext_post']:+.4f}")
    print(s, flush=True)
    return s


# ------------------------------------------------------- model runner
def run_model(name, make_est=None, cols=None, bin_impute=False, force=False, note="",
              pipe_fn=None):
    """5-fold OOF(500k) + 전체 500k 재적합 → holdout/pre/post 예측. npz 캐시.
    pipe_fn 을 주면 전체 파이프라인을 그 함수로 생성(전처리까지 커스텀)."""
    if have(name) and not force:
        z = load_preds(name)
        r = metrics(name, z["oof"], z["ho"], z["pre"], z["post"], z["secs"], z["note"])
        show(r); return r
    d = load()
    build = pipe_fn if pipe_fn is not None else \
        (lambda: make_pipe(make_est(), cols, bin_impute))
    t0 = time.time()
    oof = np.zeros(len(d.ytr))
    for fi, (ia, ib) in enumerate(d.folds):
        pipe = build()
        pipe.fit(d.Xtr.iloc[ia], d.ytr[ia])
        oof[ib] = pipe.predict_proba(d.Xtr.iloc[ib])[:, 1]
        print(f"    [{name}] fold{fi} {time.time()-t0:.0f}s", flush=True)
    pipe = build()
    pipe.fit(d.Xtr, d.ytr)
    f = lambda X: pipe.predict_proba(X)[:, 1]
    ho, pr, po = f(d.Xho), f(d.Xpre), f(d.Xpost)
    secs = time.time() - t0
    save_preds(name, oof, ho, pr, po, secs, note)
    r = metrics(name, oof, ho, pr, po, secs, note)
    show(r); return r


def run_group_experts(name, make_est, groups_tr, cols=None, bin_impute=False,
                      force=False, note="", combine="rank", min_n=2000):
    """그룹(연도/지역)별 전문가 모델을 학습해 평균 결합.
    OOF: 각 fold 의 train 부분에서 그룹별 전문가를 학습 → held-out fold 예측 평균.
    combine in {rank, prob, logit}."""
    if have(name) and not force:
        z = load_preds(name)
        r = metrics(name, z["oof"], z["ho"], z["pre"], z["post"], z["secs"], z["note"])
        show(r); return r
    d = load()
    g = np.asarray(groups_tr)
    uq = [u for u in np.unique(g) if (g == u).sum() >= min_n]
    t0 = time.time()

    def _comb(mats):
        M = np.vstack(mats)
        if combine == "rank":
            M = np.vstack([rank01(M[i]) for i in range(M.shape[0])])
        elif combine == "logit":
            M = np.vstack([logit(M[i]) for i in range(M.shape[0])])
        return M.mean(0)

    oof = np.zeros(len(d.ytr))
    for fi, (ia, ib) in enumerate(d.folds):
        ps = []
        for u in uq:
            sel = ia[g[ia] == u]
            if len(sel) < min_n:
                continue
            pipe = make_pipe(make_est(), cols, bin_impute)
            pipe.fit(d.Xtr.iloc[sel], d.ytr[sel])
            ps.append(pipe.predict_proba(d.Xtr.iloc[ib])[:, 1])
        oof[ib] = _comb(ps)
        print(f"    [{name}] fold{fi} experts={len(ps)} {time.time()-t0:.0f}s", flush=True)
    Ph, Pp, Po = [], [], []
    for u in uq:
        sel = np.where(g == u)[0]
        pipe = make_pipe(make_est(), cols, bin_impute)
        pipe.fit(d.Xtr.iloc[sel], d.ytr[sel])
        f = lambda X: pipe.predict_proba(X)[:, 1]
        Ph.append(f(d.Xho)); Pp.append(f(d.Xpre)); Po.append(f(d.Xpost))
    ho, pr, po = _comb(Ph), _comb(Pp), _comb(Po)
    secs = time.time() - t0
    save_preds(name, oof, ho, pr, po, secs, note)
    r = metrics(name, oof, ho, pr, po, secs, note)
    show(r); return r
