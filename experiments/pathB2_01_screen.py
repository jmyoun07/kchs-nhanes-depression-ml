# -*- coding: utf-8 -*-
"""
pathB2_01 — 1단계 대탐색(coarse screen).

§2.10: 오직 KCHS 내부 검증(학습부분을 다시 80/20으로 쪼갠 내부 valid)만 사용.
NHANES는 이 스크립트에서 단 한 번도 로드하지 않는다(코드로 확인 가능).

탐색 대상
  A. LightGBM 하이퍼파라미터 랜덤/격자 탐색 (num_leaves 7~63, min_child_samples 100~5000,
     reg_alpha/lambda 0~50, max_depth 3~8, lr 0.01~0.1, n_estimators 300~2000,
     subsample/colsample 0.5~1.0)
  B. 단조성 제약(monotone_constraints) 세트
  C. XGBoost / CatBoost / HistGradientBoosting
  D. 선형계열 강정규화(L1/L2/ElasticNet, 다양한 C)
"""
import os, sys, time, json, itertools, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathB2_common import (ROOT, RES, SEED, NUM, BIN, CAT, FEATURES, load,
                           baseline_split, make_pre, feat_names, mono_vector,
                           MONO_SETS, wauc)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

OUT = os.path.join(RES, "pathB2_01_screen.csv")
NJ = 12                       # 동일 머신에서 타 경로 작업이 병행 중 -> 코어 과점 방지
N_SCR_TR = 500_000            # 스크리닝용 학습 표본(순위 매기기 목적; 최종은 전체로 재학습)
N_SCR_VA = 250_000

# ---------------------------------------------------------------- KCHS만 로드
t0 = time.time()
k, _pre_unused, _post_unused = load()
del _pre_unused, _post_unused           # 외부데이터는 즉시 폐기(사용 불가)
kk, itr, iho = baseline_split(k)
y = kk["dep"].astype(int).values
W = kk["W"].to_numpy(float)
X = kk[FEATURES]

# 학습부분(itr)을 다시 내부 train/valid 로 분할 — 최종 holdout(iho)은 건드리지 않음
str_, sva = train_test_split(itr, test_size=0.2, stratify=y[itr], random_state=SEED)
rng0 = np.random.RandomState(SEED)
str_ = rng0.choice(str_, size=min(N_SCR_TR, len(str_)), replace=False)
sva = rng0.choice(sva, size=min(N_SCR_VA, len(sva)), replace=False)

# 전처리는 screen-train 에서 1회만 적합(설정마다 반복하지 않음; 누수 없음)
PRE = make_pre().fit(X.iloc[str_])


def dense(A):
    return np.asarray(A.todense() if hasattr(A, "todense") else A, dtype=np.float32)


Xtr = dense(PRE.transform(X.iloc[str_]))
Xva = dense(PRE.transform(X.iloc[sva]))
assert Xtr.shape[1] == len(feat_names()), (Xtr.shape, len(feat_names()))
ytr = y[str_]
yva, Wva = y[sva], W[sva]
del kk, X, k
import gc; gc.collect()
print(f"screen: train={Xtr.shape} valid={Xva.shape}  [{time.time()-t0:.0f}s]", flush=True)

rows = []
if os.path.exists(OUT):
    rows = pd.read_csv(OUT).to_dict("records")
done = {r["name"] for r in rows}


def run(name, family, make_model, meta):
    if name in done:
        return
    t1 = time.time()
    try:
        m = make_model()
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xva)[:, 1]
        s = wauc(yva, p, Wva)
    except Exception as ex:
        print(f"  !! {name}: {type(ex).__name__}: {ex}", flush=True)
        return
    dt = time.time() - t1
    rows.append(dict(name=name, family=family, valid_wauc=s, sec=round(dt, 1),
                     params=json.dumps(meta, sort_keys=True)))
    print(f"  {name:44s} {s:.5f}  [{dt:.0f}s]", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")


def lgbm(**kw):
    p = dict(random_state=SEED, n_jobs=NJ, verbose=-1)
    p.update(kw)
    return lambda: lgb.LGBMClassifier(**p)


# =========================================================== A. LGBM 기준 + 축별 1D 스윕
BASE = dict(n_estimators=600, learning_rate=0.03, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8)
print("[A] LGBM baseline + 1D sweeps", flush=True)
run("lgb_base", "lgbm", lgbm(**BASE), BASE)

SWEEPS = {
    "num_leaves":        [7, 15, 23, 31, 47, 63],
    "min_child_samples": [100, 300, 1000, 2500, 5000],
    "reg_alpha":         [0.5, 2, 10, 50],
    "reg_lambda":        [0.5, 2, 10, 50],
    "max_depth":         [3, 4, 5, 6, 8],
    "learning_rate":     [0.01, 0.05, 0.1],
    "n_estimators":      [300, 1000, 2000],
    "subsample":         [0.5, 0.6, 1.0],
    "colsample_bytree":  [0.5, 0.6, 1.0],
}
for key, vals in SWEEPS.items():
    for v in vals:
        p = dict(BASE); p[key] = v
        if key == "subsample":
            p["subsample_freq"] = 1
        if p == BASE:
            continue
        run(f"lgb_{key}={v}", "lgbm_1d", lgbm(**p), p)

# =========================================================== A2. 강정규화·얕은 트리 조합(가설)
print("[A2] LGBM strong-regularization / shallow combos", flush=True)
COMBOS = [
    dict(num_leaves=7,  max_depth=3, min_child_samples=2000, reg_alpha=10, reg_lambda=10,
         learning_rate=0.05, n_estimators=1200),
    dict(num_leaves=7,  max_depth=3, min_child_samples=5000, reg_alpha=50, reg_lambda=50,
         learning_rate=0.05, n_estimators=1500),
    dict(num_leaves=15, max_depth=4, min_child_samples=2000, reg_alpha=10, reg_lambda=10,
         learning_rate=0.03, n_estimators=1200),
    dict(num_leaves=15, max_depth=4, min_child_samples=1000, reg_alpha=5,  reg_lambda=20,
         learning_rate=0.05, n_estimators=800),
    dict(num_leaves=31, max_depth=5, min_child_samples=2500, reg_alpha=20, reg_lambda=20,
         learning_rate=0.03, n_estimators=1000),
    dict(num_leaves=15, max_depth=4, min_child_samples=3000, reg_alpha=25, reg_lambda=25,
         learning_rate=0.02, n_estimators=2000),
    dict(num_leaves=63, max_depth=8, min_child_samples=100,  reg_alpha=0,  reg_lambda=0,
         learning_rate=0.03, n_estimators=600),
]
for i, c in enumerate(COMBOS):
    p = dict(subsample=0.8, colsample_bytree=0.8, subsample_freq=1); p.update(c)
    run(f"lgb_combo{i}", "lgbm_combo", lgbm(**p), p)

# =========================================================== A3. 랜덤 서치
print("[A3] LGBM random search", flush=True)
rng = np.random.RandomState(SEED)
N_RAND = 45
for i in range(N_RAND):
    p = dict(
        num_leaves=int(rng.choice([7, 11, 15, 23, 31, 47, 63])),
        max_depth=int(rng.choice([3, 4, 5, 6, 8, -1])),
        min_child_samples=int(rng.choice([100, 300, 600, 1000, 2000, 3500, 5000])),
        reg_alpha=float(rng.choice([0, 0.5, 2, 5, 10, 25, 50])),
        reg_lambda=float(rng.choice([0, 0.5, 2, 5, 10, 25, 50])),
        learning_rate=float(rng.choice([0.01, 0.02, 0.03, 0.05, 0.1])),
        n_estimators=int(rng.choice([300, 600, 1000, 1500, 2000])),
        subsample=float(rng.choice([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])),
        colsample_bytree=float(rng.choice([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])),
        subsample_freq=1,
    )
    run(f"lgb_rand{i:02d}", "lgbm_rand", lgbm(**p), p)

# =========================================================== B. 단조성 제약
print("[B] monotone constraints", flush=True)
for mname, spec in MONO_SETS.items():
    if mname == "mono_none":
        continue
    mv = mono_vector(spec)
    for tag, base in [("base", BASE),
                      ("reg", dict(num_leaves=15, max_depth=4, min_child_samples=2000,
                                   reg_alpha=10, reg_lambda=10, learning_rate=0.03,
                                   n_estimators=1200, subsample=0.8,
                                   colsample_bytree=0.8, subsample_freq=1))]:
        p = dict(base); p["monotone_constraints"] = mv
        meta = dict(base); meta["mono"] = mname
        run(f"lgb_{mname}_{tag}", "lgbm_mono", lgbm(**p), meta)

# =========================================================== C. 타 알고리즘
print("[C] XGBoost / CatBoost / HistGB", flush=True)


def xgbm(**kw):
    p = dict(n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.8,
             colsample_bytree=0.8, tree_method="hist", eval_metric="auc",
             random_state=SEED, n_jobs=NJ)
    p.update(kw)
    return lambda: xgb.XGBClassifier(**p)


XGB_GRID = [
    dict(),
    dict(max_depth=3), dict(max_depth=4), dict(max_depth=8),
    dict(max_depth=4, min_child_weight=100, reg_alpha=5, reg_lambda=20, n_estimators=1200),
    dict(max_depth=3, min_child_weight=500, reg_alpha=20, reg_lambda=50,
         learning_rate=0.05, n_estimators=1500),
    dict(max_depth=5, min_child_weight=50, reg_lambda=10, n_estimators=1000,
         learning_rate=0.05),
    dict(max_depth=6, n_estimators=1500, learning_rate=0.02, reg_lambda=5),
]
for i, g in enumerate(XGB_GRID):
    run(f"xgb_{i}", "xgb", xgbm(**g), g)

# XGB 단조 제약
for mname in ["mono_srh_oral_inc", "mono_core", "mono_all17"]:
    mv = tuple(mono_vector(MONO_SETS[mname]))
    run(f"xgb_{mname}", "xgb_mono",
        xgbm(max_depth=4, min_child_weight=100, reg_alpha=5, reg_lambda=20,
             n_estimators=1200, monotone_constraints=mv),
        dict(mono=mname, max_depth=4))


def cbm(**kw):
    p = dict(iterations=800, learning_rate=0.05, depth=6, l2_leaf_reg=3,
             random_seed=SEED, verbose=0, thread_count=NJ, allow_writing_files=False)
    p.update(kw)
    return lambda: CatBoostClassifier(**p)


CB_GRID = [dict(), dict(depth=4), dict(depth=8),
           dict(depth=4, l2_leaf_reg=30, iterations=1500),
           dict(depth=3, l2_leaf_reg=50, iterations=2000, learning_rate=0.05)]
for i, g in enumerate(CB_GRID):
    run(f"cat_{i}", "catboost", cbm(**g), g)


def hgb(**kw):
    p = dict(max_iter=600, learning_rate=0.03, max_leaf_nodes=31,
             random_state=SEED)
    p.update(kw)
    return lambda: HistGradientBoostingClassifier(**p)


HGB_GRID = [dict(),
            dict(max_leaf_nodes=15, min_samples_leaf=2000, l2_regularization=10,
                 max_iter=1200),
            dict(max_leaf_nodes=7, max_depth=3, min_samples_leaf=5000,
                 l2_regularization=50, learning_rate=0.05, max_iter=1500),
            dict(max_leaf_nodes=63, max_iter=1000)]
for i, g in enumerate(HGB_GRID):
    run(f"hgb_{i}", "histgb", hgb(**g), g)

# HGB 단조
for mname in ["mono_srh_oral_inc", "mono_core"]:
    run(f"hgb_{mname}", "histgb_mono",
        hgb(max_leaf_nodes=15, min_samples_leaf=2000, l2_regularization=10,
            max_iter=1200, monotonic_cst=mono_vector(MONO_SETS[mname])),
        dict(mono=mname))

# =========================================================== D. 선형계열
print("[D] regularized linear", flush=True)
for C in [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 1.0]:
    run(f"logit_l2_C={C}", "linear",
        lambda C=C: LogisticRegression(C=C, penalty="l2", max_iter=2000, solver="lbfgs"),
        dict(penalty="l2", C=C))
for C in [0.001, 0.01, 0.1, 1.0]:
    run(f"logit_l1_C={C}", "linear",
        lambda C=C: LogisticRegression(C=C, penalty="l1", max_iter=3000, solver="saga", n_jobs=NJ),
        dict(penalty="l1", C=C))
for C in [0.01, 0.1, 1.0]:
    for l1r in [0.2, 0.5, 0.8]:
        run(f"logit_en_C={C}_l1r={l1r}", "linear",
            lambda C=C, l1r=l1r: LogisticRegression(C=C, penalty="elasticnet", l1_ratio=l1r, max_iter=3000, solver="saga", n_jobs=NJ),
            dict(penalty="elasticnet", C=C, l1_ratio=l1r))

pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"\nDONE {len(rows)} configs, {time.time()-t0:.0f}s -> {OUT}")
d = pd.DataFrame(rows).sort_values("valid_wauc", ascending=False)
print(d.head(25)[["name", "family", "valid_wauc"]].to_string(index=False))
