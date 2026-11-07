# -*- coding: utf-8 -*-
"""
pathH_01_alg.py — 기저모델 라이브러리 [알고리즘 다양성 + 깊이 다양성].

각 모델: KCHS train(500k) 5-fold OOF + 전체 500k 재적합 → holdout/pre/post 예측 캐시.
NHANES 는 predict 만. 라벨 미사용.
출력: results/pathH_01_alg.csv, 캐시: experiments/cache/pathH/*.npz
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from pathH_common import *
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import (HistGradientBoostingClassifier, RandomForestClassifier,
                              ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import SplineTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

ROWS = []
OUT = os.path.join(RES, "pathH_01_alg.csv")


def go(name, **kw):
    try:
        r = run_model(name, **kw)
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}", flush=True)
        return
    ROWS.append(r)
    pd.DataFrame(ROWS).to_csv(OUT, index=False, encoding="utf-8-sig")


L = lambda **kw: (lambda: lgb.LGBMClassifier(**{**LGB_BASE, **kw}))

# ---------------- LightGBM 계열: 깊이 다양성 ----------------
go("lgbm_base", make_est=L(), note="명세 기준선 파라미터")
go("lgbm_d2", make_est=L(num_leaves=4, max_depth=2, n_estimators=1500, learning_rate=0.05),
   note="매우 얕음(depth2)")
go("lgbm_d3", make_est=L(num_leaves=8, max_depth=3, n_estimators=1200, learning_rate=0.05),
   note="얕음(depth3)")
go("lgbm_stump", make_est=L(num_leaves=2, max_depth=1, n_estimators=2500, learning_rate=0.05),
   note="stump=가법모형(상호작용 0차)")
go("lgbm_d4", make_est=L(num_leaves=16, max_depth=4, n_estimators=900, learning_rate=0.04),
   note="중간깊이")
go("lgbm_deep", make_est=L(num_leaves=127, max_depth=-1, n_estimators=600),
   note="깊음")
go("lgbm_goss", make_est=L(boosting_type="goss", subsample=1.0), note="GOSS")
go("lgbm_reg", make_est=L(reg_lambda=10.0, min_child_samples=200), note="정규화")

# ---------------- XGBoost ----------------
X = lambda **kw: (lambda: xgb.XGBClassifier(**{
    **dict(n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.8,
           colsample_bytree=0.8, min_child_weight=10, tree_method="hist",
           n_jobs=-1, random_state=SEED, eval_metric="logloss"), **kw}))
go("xgb_base", make_est=X(), note="XGBoost hist depth6")
go("xgb_d3", make_est=X(max_depth=3, n_estimators=1200, learning_rate=0.05),
   note="XGBoost depth3")
go("xgb_d8", make_est=X(max_depth=8, min_child_weight=50), note="XGBoost depth8")

# ---------------- CatBoost ----------------
C = lambda **kw: (lambda: CatBoostClassifier(**{
    **dict(iterations=600, learning_rate=0.05, depth=6, verbose=0,
           thread_count=-1, allow_writing_files=False, random_seed=SEED), **kw}))
go("cat_base", make_est=C(), note="CatBoost depth6 (대칭트리)")
go("cat_d4", make_est=C(depth=4, iterations=1000), note="CatBoost depth4")

# ---------------- sklearn HistGB ----------------
go("histgb", make_est=lambda: HistGradientBoostingClassifier(
    max_iter=400, learning_rate=0.05, max_leaf_nodes=31, random_state=SEED),
   note="sklearn HistGB")

# ---------------- Bagging 계열 (결측 대체 필요) ----------------
go("rf", make_est=lambda: RandomForestClassifier(
    n_estimators=300, min_samples_leaf=50, max_features="sqrt",
    n_jobs=8, random_state=SEED), bin_impute=True, note="RandomForest")
go("et", make_est=lambda: ExtraTreesClassifier(
    n_estimators=300, min_samples_leaf=50, max_features="sqrt",
    n_jobs=8, random_state=SEED), bin_impute=True, note="ExtraTrees")

# ---------------- 선형/신경망/생성모형 ----------------
go("logit_l2", make_est=lambda: LogisticRegression(C=1.0, max_iter=2000),
   bin_impute=True, note="L2 로지스틱")
go("logit_l2_strong", make_est=lambda: LogisticRegression(C=0.05, max_iter=2000),
   bin_impute=True, note="강한 L2")


def spline_pipe():
    num = Pipeline([("i", SimpleImputer(strategy="median")),
                    ("sp", SplineTransformer(n_knots=6, degree=3, include_bias=False))])
    ct = ColumnTransformer([
        ("num", num, NUM),
        ("bin", SimpleImputer(strategy="most_frequent"), BIN),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", __import__("sklearn.preprocessing", fromlist=["x"])
                           .OneHotEncoder(handle_unknown="ignore", drop="first"))]), CAT)])
    return Pipeline([("pre", ct), ("clf", LogisticRegression(C=1.0, max_iter=3000))])


go("logit_spline", pipe_fn=spline_pipe, note="스플라인 기저 로지스틱(가법·비선형)")

go("mlp32", make_est=lambda: MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-3,
                                           max_iter=60, early_stopping=True,
                                           n_iter_no_change=5, random_state=SEED),
   bin_impute=True, note="얕은 MLP(32)")
go("mlp64_32", make_est=lambda: MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3,
                                              max_iter=60, early_stopping=True,
                                              n_iter_no_change=5, random_state=SEED),
   bin_impute=True, note="MLP(64,32)")
go("gnb", make_est=lambda: GaussianNB(), bin_impute=True, note="Gaussian NB(약한 학습기)")

pd.DataFrame(ROWS).to_csv(OUT, index=False, encoding="utf-8-sig")
print("SAVED", OUT)
