# -*- coding: utf-8 -*-
"""
pathH_02_subspace.py — 기저모델 라이브러리 [서브스페이스 다양성].

17변수 중 일부만 쓰는 LightGBM 들. 변수 자체는 확정 17개 안에서만 사용(추가 없음).
  (a) drop-one : 각 변수 1개씩 제외한 16변수 모델 → 특정 변수 과의존 진단 겸용
  (b) random subspace : 무작위 11개 / 8개 변수 모델
  (c) 도메인 블록 : 인구·사회경제 / 건강상태 / 건강행태
출력: results/pathH_02_subspace.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from pathH_common import *
import lightgbm as lgb

ROWS = []
OUT = os.path.join(RES, "pathH_02_subspace.csv")
L = lambda **kw: (lambda: lgb.LGBMClassifier(**{**LGB_BASE, **kw}))


def go(name, cols, note=""):
    try:
        r = run_model(name, make_est=L(), cols=cols, note=note + f" |k={len(cols)}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}", flush=True)
        return
    r["cols"] = ",".join(cols)
    ROWS.append(r)
    pd.DataFrame(ROWS).to_csv(OUT, index=False, encoding="utf-8-sig")


# (a) drop-one
for f in FEATURES:
    go(f"drop_{f}", [c for c in FEATURES if c != f], note=f"{f} 제외")

# (b) random subspace
rng = np.random.RandomState(SEED)
for i in range(10):
    cols = sorted(rng.choice(FEATURES, size=11, replace=False).tolist())
    go(f"rsub11_{i}", cols, note="무작위 부분공간 11/17")
for i in range(8):
    cols = sorted(rng.choice(FEATURES, size=8, replace=False).tolist())
    go(f"rsub8_{i}", cols, note="무작위 부분공간 8/17")

# (c) 도메인 블록
BLOCKS = {
    "blk_demog": ["age", "sex", "edu", "income_q", "partnered", "living_alone", "employed"],
    "blk_health": ["srh", "oral_health", "bmi", "htn", "dm", "sleep_hrs"],
    "blk_behav": ["smoking", "current_drinker", "active", "sleep_hrs", "bmi", "food_insecure"],
    "blk_socio": ["income_q", "edu", "employed", "food_insecure", "living_alone",
                  "partnered", "age", "sex"],
    "blk_nosubj": [c for c in FEATURES if c not in ("srh", "oral_health")],
    "blk_subjonly": ["srh", "oral_health", "age", "sex"],
}
for n, cols in BLOCKS.items():
    go(n, cols, note="도메인 블록")

pd.DataFrame(ROWS).to_csv(OUT, index=False, encoding="utf-8-sig")
print("SAVED", OUT)
