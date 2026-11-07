# -*- coding: utf-8 -*-
"""
경로A — 특징 표현(feature representation) 최적화.
17변수 고정, 표현만 바꿔 외부(NHANES pre/post) 가중 AUROC 개선을 시도.

§2.10 준수 원칙:
 - 모델 학습은 KCHS만. NHANES 라벨(dep)은 절대 학습/선택에 사용하지 않음.
 - 모델·하이퍼파라미터 선택은 KCHS 내부 CV로만.
 - NHANES는 최종 평가 1회.
 - within-survey rank 변환은 NHANES '공변량 분포'만 사용 (라벨 미사용) → 준수.
   단 배포 시 대상집단 분포가 필요하다는 제약을 note에 기록.

출력: results/pathA_feature_repr.csv
"""
from __future__ import annotations
import os, sys, time, json, warnings
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

LOG = []
def log(s):
    LOG.append(str(s)); print(s, flush=True)


# ---------------------------------------------------------------- data
K = pd.read_parquet(os.path.join(PROC, "ext_kchs.parquet"))
PRE = pd.read_parquet(os.path.join(PROC, "ext_pre.parquet"))
POST = pd.read_parquet(os.path.join(PROC, "ext_post.parquet"))
log(f"load kchs={K.shape} pre={PRE.shape} post={POST.shape}")

yK = K["dep"].astype(int).to_numpy()
WK = K["W"].to_numpy(float)
idx_all = np.arange(len(K))
itr, ite = train_test_split(idx_all, train_size=N_TRAIN, test_size=N_TEST,
                            stratify=yK, random_state=SEED)
itr = np.sort(itr); ite = np.sort(ite)
log(f"split train={len(itr)} test={len(ite)}")


# ---------------------------------------------------------------- preprocessor
def make_prep(num, binf, cat):
    nump = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler())])
    catp = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                     ("oh", OneHotEncoder(handle_unknown="ignore", drop="first"))])
    blocks = []
    if num:
        blocks.append(("num", nump, num))
    if binf:
        blocks.append(("bin", "passthrough", binf))
    if cat:
        blocks.append(("cat", catp, cat))
    return ColumnTransformer(blocks)


def wauc(y, p, w):
    return roc_auc_score(np.asarray(y).astype(int), p, sample_weight=np.asarray(w, float))


def evaluate(dk, dpre, dpost, num, binf, cat, model=None, tag=""):
    """dk/dpre/dpost: 이미 변환된 전체 DataFrame. 학습=KCHS train, 평가=KCHS test + NHANES."""
    feats = list(num) + list(binf) + list(cat)
    clf = model if model is not None else lgb.LGBMClassifier(**LGBM)
    pipe = Pipeline([("pre", make_prep(num, binf, cat)), ("clf", clf)])
    X = dk[feats]
    t0 = time.time()
    pipe.fit(X.iloc[itr], yK[itr])
    a_int = wauc(yK[ite], pipe.predict_proba(X.iloc[ite])[:, 1], WK[ite])
    out = {"name": tag, "internal": a_int}
    for nm, d in (("ext_pre", dpre), ("ext_post", dpost)):
        p = pipe.predict_proba(d[feats])[:, 1]
        out[nm] = wauc(d["dep"], p, d["W"])
    out["seconds"] = round(time.time() - t0, 1)
    return out, pipe


def cv_internal(dk, num, binf, cat, model_fn=None, folds=3, seed=SEED):
    """KCHS 내부 가중 AUROC CV (모델 선택 전용, NHANES 미접촉)."""
    feats = list(num) + list(binf) + list(cat)
    X = dk[feats].iloc[itr].reset_index(drop=True)
    y = yK[itr]
    w = WK[itr]
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = []
    for tr, va in skf.split(X, y):
        clf = model_fn() if model_fn else lgb.LGBMClassifier(**LGBM)
        pipe = Pipeline([("pre", make_prep(num, binf, cat)), ("clf", clf)])
        pipe.fit(X.iloc[tr], y[tr])
        scores.append(wauc(y[va], pipe.predict_proba(X.iloc[va])[:, 1], w[va]))
    return float(np.mean(scores)), float(np.std(scores))


ROWS = []
def record(res, compliant, note=""):
    r = dict(res)
    r["protocol_compliant"] = bool(compliant)
    r["note"] = note
    ROWS.append(r)
    log(f"{r['name']:38s} int={r['internal']:.4f} pre={r['ext_pre']:.4f} "
        f"post={r['ext_post']:.4f} ok={compliant} [{r['seconds']}s] {note}")
    pd.DataFrame(ROWS).to_csv(os.path.join(RES, "pathA_feature_repr.csv"),
                              index=False, encoding="utf-8-sig")
    return r


# ================================================================ 0) BASELINE
if os.environ.get("PATHA_RUN_BASELINE") == "1":
    log("\n=== 0) 기준선 재현 ===")
    base, base_pipe = evaluate(K, PRE, POST, NUM, BIN, CAT, tag="baseline_17")
    record(base, True, "protocol baseline")


# ================================================================ helpers
def add_rank(df, ref, cols, suffix="_r"):
    """ref 분포 기준 within-survey 백분위 순위(0~1). ref는 같은 설문의 전체 데이터."""
    out = df.copy()
    for c in cols:
        v = ref[c].to_numpy(float)
        v = v[~np.isnan(v)]
        x = df[c].to_numpy(float)
        # midrank 백분위: P(ref < x) + 0.5*P(ref == x)
        lo = np.searchsorted(np.sort(v), x, side="left")
        hi = np.searchsorted(np.sort(v), x, side="right")
        r = (lo + hi) / (2.0 * len(v))
        r[np.isnan(x)] = np.nan
        out[c + suffix] = r
    return out


def spline_basis(df, col, knots, prefix=None):
    """자연스러운 restricted cubic spline (Harrell 형식) 기저 생성."""
    p = prefix or col
    x = df[col].to_numpy(float)
    k = np.asarray(knots, float)
    K = len(k)
    out = {}
    out[f"{p}_s0"] = x
    denom = (k[-1] - k[0]) ** 2
    for j in range(K - 2):
        def pos(a):
            return np.where(a > 0, a ** 3, 0.0)
        t = (pos(x - k[j])
             - pos(x - k[K - 2]) * (k[-1] - k[j]) / (k[-1] - k[K - 2])
             + pos(x - k[-1]) * (k[K - 2] - k[j]) / (k[-1] - k[K - 2]))
        out[f"{p}_s{j+1}"] = t / denom
    d = df.copy()
    for kk, vv in out.items():
        d[kk] = vv
    return d, list(out.keys())


if __name__ == "__main__":
    pass
