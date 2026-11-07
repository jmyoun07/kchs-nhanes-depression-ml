# -*- coding: utf-8 -*-
"""pathH_01b: 메모리 오류로 실패한 모델 재시도(histgb) + 추가 알고리즘."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from pathH_common import *
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV

ROWS = []
OUT = os.path.join(RES, "pathH_01b_retry.csv")

def go(name, **kw):
    try:
        r = run_model(name, **kw); ROWS.append(r)
        pd.DataFrame(ROWS).to_csv(OUT, index=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}", flush=True)

go("histgb", make_est=lambda: HistGradientBoostingClassifier(
    max_iter=400, learning_rate=0.05, max_leaf_nodes=31, random_state=SEED),
   note="sklearn HistGB")
go("histgb_d3", make_est=lambda: HistGradientBoostingClassifier(
    max_iter=800, learning_rate=0.05, max_leaf_nodes=8, max_depth=3,
    l2_regularization=1.0, random_state=SEED), note="sklearn HistGB depth3")
go("lda", make_est=lambda: LinearDiscriminantAnalysis(), bin_impute=True,
   note="LDA(생성적 선형)")
go("sgd_huber", make_est=lambda: CalibratedClassifierCV(
    SGDClassifier(loss="modified_huber", alpha=1e-5, max_iter=30, random_state=SEED),
    cv=3, method="sigmoid"), bin_impute=True, note="SGD modified-huber")
print("DONE")
