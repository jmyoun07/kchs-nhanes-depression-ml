# -*- coding: utf-8 -*-
"""pathC_timing.py — 기저모델 학습시간 추정(서브샘플 200k)으로 전체 실행계획 산정."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from pathC_common import load, make_pre, wauc

d = load("full02")
n = 200000
rng = np.random.RandomState(0)
sub = rng.choice(len(d.Xtr), n, replace=False)
Xs, ys = d.Xtr.iloc[sub], d.ytr[sub]

cands = {
    "lgbm": Pipeline([("pre", make_pre(True)),
                      ("clf", lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03,
                                                 num_leaves=31, subsample=0.8,
                                                 colsample_bytree=0.8, random_state=42,
                                                 n_jobs=-1, verbose=-1))]),
    "rf": Pipeline([("pre", make_pre(True)),
                    ("clf", RandomForestClassifier(n_estimators=200, max_depth=14,
                                                   min_samples_leaf=100, n_jobs=-1,
                                                   random_state=42))]),
    "et": Pipeline([("pre", make_pre(False)),
                    ("clf", ExtraTreesClassifier(n_estimators=200, max_depth=16,
                                                 min_samples_leaf=100, n_jobs=-1,
                                                 random_state=42))]),
    "histgb": Pipeline([("pre", make_pre(True)),
                        ("clf", HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                                               max_leaf_nodes=31,
                                                               l2_regularization=1.0,
                                                               random_state=42))]),
    "enet": Pipeline([("pre", make_pre(False)),
                      ("clf", LogisticRegression(penalty="elasticnet", l1_ratio=0.5, C=1.0,
                                                 solver="saga", max_iter=200, n_jobs=-1))]),
}
for nm, m in cands.items():
    t0 = time.time()
    m.fit(Xs, ys)
    fit_s = time.time() - t0
    t1 = time.time()
    p = m.predict_proba(d.Xho)[:, 1]
    pr_s = time.time() - t1
    print(f"{nm:8s} fit200k={fit_s:6.1f}s pred410k={pr_s:5.1f}s "
          f"내부(sub학습)={wauc(d.yho, p, d.Who):.4f}  전체train추정={fit_s*8.2:6.0f}s")
    sys.stdout.flush()
