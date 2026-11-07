# -*- coding: utf-8 -*-
"""경로6 STAGE0 — 기준선 재현 확인 (내부 0.8310 / pre 0.7740 / post 0.7653)"""
import time, numpy as np
from path6_common import *

t0 = time.time()
(pte, ppre, ppost), pipe = fit_predict(XTR, YTR, [XTE, XP, XQ])
r = report("BASE_lgbm17_literal500k", pte, ppre, ppost,
           note="명세 그대로: train=500000,test=100000,stratify,seed42",
           secs=time.time() - t0)
print(r)
tgt = dict(internal=0.8310, ext_pre=0.7740, ext_post=0.7653)
ok = all(abs(round(r[k], 4) - v) < 1e-4 for k, v in tgt.items())
print("MATCH(3dp):", ok, {k: round(r[k], 4) for k in tgt})

np.save(os.path.join(RES, "_p6_base_te.npy"), pte)
np.save(os.path.join(RES, "_p6_base_pre.npy"), ppre)
np.save(os.path.join(RES, "_p6_base_post.npy"), ppost)
save([r], "path6_00_baseline.csv")
print("elapsed", round(time.time() - t0, 1))
