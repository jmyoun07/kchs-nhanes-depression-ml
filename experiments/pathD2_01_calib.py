# -*- coding: utf-8 -*-
"""pathD2_01_calib.py — 캐시 설계행렬 동등성 검증 + 속도 캘리브레이션."""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathD2_common import *  # noqa
from pathD2_cache import Cached, loyo, strata_kfold

t0 = time.time()
C = Cached()
b = C.b
print(f"cache built {time.time()-t0:.0f}s  cols={len(C.cols)} -> {C.cols}")
print("Z", C.Z.shape, "Zpre", C.Zpre.shape, "Zpost", C.Zpost.shape)

# 동등성: 캐시 행렬로 기준선을 다시 학습해 파이프라인 결과와 비교
t = time.time()
m = lgb.LGBMClassifier(**LGB_BASE)
m.fit(C.Z[b.itr], b.y[b.itr])
r = dict(name="cached_baseline",
         internal=wauc(b.y[b.iho], m.predict_proba(C.Z[b.iho])[:, 1], b.W[b.iho]),
         ext_pre=wauc(b.ypre, m.predict_proba(C.Zpre)[:, 1], b.wpre),
         ext_post=wauc(b.ypost, m.predict_proba(C.Zpost)[:, 1], b.wpost))
print(fmt_ext(r) + f"  [fit {time.time()-t:.0f}s]")
print("  (파이프라인 기준선: 0.8335 / 0.7682 / 0.7680)")

# LOYO 속도
for n in (150000, 300000):
    t = time.time()
    tr, va, yv = next(iter(loyo(b)))
    tr = b.sub(tr, n)
    mm = lgb.LGBMClassifier(**LGB_BASE)
    mm.fit(C.Z[tr], b.y[tr])
    s = wauc(b.y[va], mm.predict_proba(C.Z[va])[:, 1], b.W[va])
    print(f"  LOYO y={yv} n={n:,} auc={s:.4f} [{time.time()-t:.1f}s]")

# strata fold 크기
sizes = [(len(tr), len(va)) for tr, va, f in strata_kfold(b, 5)]
print("strata folds (train,valid):", sizes)
print(f"total {time.time()-t0:.0f}s")
