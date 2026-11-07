# -*- coding: utf-8 -*-
"""
경로A 공통 모듈 — 특징 표현(feature representation) 최적화.

§2.10 준수 원칙
 - 학습은 KCHS만. NHANES 라벨(dep)은 학습/선택에 절대 미사용.
 - 모델·하이퍼파라미터 선택은 KCHS 내부 CV로만.
 - within-survey rank 변환은 NHANES '공변량 분포'만 사용(라벨 미사용) → §2.10 위반 아님.
   단 배포 시 대상집단 공변량 분포(또는 그 추정치)가 필요하다는 제약이 생김.
"""
from __future__ import annotations
import os, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

ROOT = r"C:/Users/rudtn/Desktop/claude code/지역사회건강"
PROC = os.path.join(ROOT, "data/processed")
RES = os.path.join(ROOT, "results")
SEED = 42
N_TRAIN, N_TEST = 500_000, 100_000

NUM = ["age", "sleep_hrs", "bmi", "income_q", "srh", "oral_health"]
BIN = ["partnered", "living_alone", "current_drinker", "active",
       "htn", "dm", "employed", "food_insecure"]
CAT = ["sex", "edu", "smoking"]
FEATURES = NUM + BIN + CAT

LGBM = dict(n_estimators=600, learning_rate=0.03, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, n_jobs=-1, verbose=-1)


# ------------------------------------------------------------------ data
def load():
    K = pd.read_parquet(os.path.join(PROC, "ext_kchs.parquet"))
    P = pd.read_parquet(os.path.join(PROC, "ext_pre.parquet"))
    Q = pd.read_parquet(os.path.join(PROC, "ext_post.parquet"))
    return K, P, Q


K, PRE, POST = load()
yK = K["dep"].astype(int).to_numpy()
WK = K["W"].to_numpy(float)
_idx = np.arange(len(K))
ITR, ITE = train_test_split(_idx, train_size=N_TRAIN, test_size=N_TEST,
                            stratify=yK, random_state=SEED)
ITR = np.sort(ITR); ITE = np.sort(ITE)


def wauc(y, p, w):
    return roc_auc_score(np.asarray(y).astype(int), np.asarray(p, float),
                         sample_weight=np.asarray(w, float))


# ------------------------------------------------------------------ preprocessing
def make_prep(num, binf, cat, bin_impute=False):
    nump = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler())])
    catp = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                     ("oh", OneHotEncoder(handle_unknown="ignore", drop="first"))])
    binp = (Pipeline([("imp", SimpleImputer(strategy="most_frequent"))])
            if bin_impute else "passthrough")
    blocks = []
    if num:
        blocks.append(("num", nump, list(num)))
    if binf:
        blocks.append(("bin", binp, list(binf)))
    if cat:
        blocks.append(("cat", catp, list(cat)))
    return ColumnTransformer(blocks)


# ------------------------------------------------------------------ rank / quantile transforms
def wpct(ref_v, ref_w, x):
    """ref 분포(가중) 안에서 x의 midrank 백분위 (0~1). NaN in x -> NaN."""
    ref_v = np.asarray(ref_v, float)
    ref_w = np.asarray(ref_w, float)
    m = ~np.isnan(ref_v) & ~np.isnan(ref_w)
    v, w = ref_v[m], ref_w[m]
    o = np.argsort(v, kind="mergesort")
    v, w = v[o], w[o]
    cw = np.cumsum(w)
    tot = cw[-1]
    x = np.asarray(x, float)
    xs = np.where(np.isnan(x), np.inf, x)          # NaN 임시 처리
    lo = np.searchsorted(v, xs, side="left")
    hi = np.searchsorted(v, xs, side="right")
    Flo = np.where(lo > 0, cw[np.clip(lo - 1, 0, None)], 0.0)
    Fhi = np.where(hi > 0, cw[np.clip(hi - 1, 0, None)], 0.0)
    r = (Flo + Fhi) / (2.0 * tot)
    r[np.isnan(x)] = np.nan
    return r


def _norm(p):
    from scipy.special import ndtri
    q = np.clip(p, 1e-6, 1 - 1e-6)
    out = ndtri(q)
    out[np.isnan(p)] = np.nan
    return out


def rank_transform(cols, weighted=True, normal=False, kchs_ref="train",
                   post_ref="own"):
    """각 설문 '자기 분포' 기준으로 cols를 백분위로 치환한 3개 DataFrame 반환.

    kchs_ref : 'train' (train 행만 참조; test 누출 방지) | 'all'
    post_ref : 'own' (post 자체 분포) | 'pre' (pre 분포를 재사용; 배포현실성↑)
    """
    Kc, Pc, Qc = K.copy(), PRE.copy(), POST.copy()
    refK = K.iloc[ITR] if kchs_ref == "train" else K
    wk = refK["W"].to_numpy(float) if weighted else np.ones(len(refK))
    wp = PRE["W"].to_numpy(float) if weighted else np.ones(len(PRE))
    wq = POST["W"].to_numpy(float) if weighted else np.ones(len(POST))
    for c in cols:
        rk = wpct(refK[c].to_numpy(float), wk, K[c].to_numpy(float))
        rp = wpct(PRE[c].to_numpy(float), wp, PRE[c].to_numpy(float))
        if post_ref == "own":
            rq = wpct(POST[c].to_numpy(float), wq, POST[c].to_numpy(float))
        else:
            rq = wpct(PRE[c].to_numpy(float), wp, POST[c].to_numpy(float))
        if normal:
            rk, rp, rq = _norm(rk), _norm(rp), _norm(rq)
        Kc[c], Pc[c], Qc[c] = rk, rp, rq
    return Kc, Pc, Qc


# ------------------------------------------------------------------ splines
def rcs(x, knots):
    """Harrell restricted cubic spline basis. 반환: (n, len(knots)-1)"""
    x = np.asarray(x, float)
    k = np.asarray(knots, float)
    nk = len(k)
    den = (k[-1] - k[0]) ** 2
    cols = [x]
    pos = lambda a: np.where(a > 0, a ** 3, 0.0)
    for j in range(nk - 2):
        t = (pos(x - k[j])
             - pos(x - k[nk - 2]) * (k[-1] - k[j]) / (k[-1] - k[nk - 2])
             + pos(x - k[-1]) * (k[nk - 2] - k[j]) / (k[-1] - k[nk - 2]))
        cols.append(t / den)
    return np.column_stack(cols)


def add_rcs(dfs, col, knots, drop_orig=True):
    """dfs=(K,P,Q) 각각에 동일 knot로 스플라인 기저 추가. 새 컬럼명 리스트 반환."""
    names = None
    out = []
    for d in dfs:
        B = rcs(d[col].to_numpy(float), knots)
        names = [f"{col}_rcs{i}" for i in range(B.shape[1])]
        dd = d.copy()
        for i, nm in enumerate(names):
            dd[nm] = B[:, i]
        if drop_orig:
            pass
        out.append(dd)
    return out, names


# ------------------------------------------------------------------ evaluation
def evaluate(dk, dp, dq, num, binf, cat, model=None, tag="", bin_impute=None):
    feats = list(num) + list(binf) + list(cat)
    clf = model if model is not None else lgb.LGBMClassifier(**LGBM)
    if bin_impute is None:
        bin_impute = not isinstance(clf, lgb.LGBMClassifier)
    pipe = Pipeline([("pre", make_prep(num, binf, cat, bin_impute)), ("clf", clf)])
    t0 = time.time()
    pipe.fit(dk[feats].iloc[ITR], yK[ITR])
    out = {"name": tag}
    out["internal"] = wauc(yK[ITE], pipe.predict_proba(dk[feats].iloc[ITE])[:, 1], WK[ITE])
    out["ext_pre"] = wauc(dp["dep"], pipe.predict_proba(dp[feats])[:, 1], dp["W"])
    out["ext_post"] = wauc(dq["dep"], pipe.predict_proba(dq[feats])[:, 1], dq["W"])
    out["seconds"] = round(time.time() - t0, 1)
    return out, pipe


def cv_internal(dk, num, binf, cat, model_fn=None, folds=3, seed=SEED, sub=200_000,
                bin_impute=None):
    """KCHS 내부 가중 AUROC CV (모델 선택 전용, NHANES 미접촉)."""
    feats = list(num) + list(binf) + list(cat)
    ii = ITR if sub is None or sub >= len(ITR) else \
        train_test_split(ITR, train_size=sub, stratify=yK[ITR], random_state=seed)[0]
    X = dk[feats].iloc[ii].reset_index(drop=True)
    y = yK[ii]; w = WK[ii]
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    sc = []
    for tr, va in skf.split(X, y):
        clf = model_fn() if model_fn else lgb.LGBMClassifier(**LGBM)
        bi = (not isinstance(clf, lgb.LGBMClassifier)) if bin_impute is None else bin_impute
        pipe = Pipeline([("pre", make_prep(num, binf, cat, bi)), ("clf", clf)])
        pipe.fit(X.iloc[tr], y[tr])
        sc.append(wauc(y[va], pipe.predict_proba(X.iloc[va])[:, 1], w[va]))
    return float(np.mean(sc)), float(np.std(sc))
