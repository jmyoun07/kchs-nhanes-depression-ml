# -*- coding: utf-8 -*-
"""pathD2_build_cache.py — 설계행렬/라벨/설계변수를 디스크에 1회 저장(병렬 워커용 memmap)."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathD2_common import Bench, make_pre, FEATURES
from pathD2_cache import CACHE

t0 = time.time()
b = Bench()
ct = make_pre()
ct.fit(b.X.iloc[b.itr])
cols = [c.split("__", 1)[1] for c in ct.get_feature_names_out()]

np.save(os.path.join(CACHE, "Z.npy"), ct.transform(b.X).astype(np.float32))
np.save(os.path.join(CACHE, "Zpre.npy"), ct.transform(b.Xpre).astype(np.float32))
np.save(os.path.join(CACHE, "Zpost.npy"), ct.transform(b.Xpost).astype(np.float32))
np.savez(os.path.join(CACHE, "meta.npz"),
         y=b.y, W=b.W, year=b.year,
         strata=b.strata.astype("U8"),
         itr=b.itr, iho=b.iho,
         ypre=b.ypre, wpre=b.wpre, ypost=b.ypost, wpost=b.wpost,
         cols=np.array(cols, dtype="U32"),
         features=np.array(FEATURES, dtype="U32"))
print("cols:", cols)
print(f"cache written to {CACHE} in {time.time()-t0:.0f}s")
