# -*- coding: utf-8 -*-
"""pathH_00: 기준선 재현(과제 명세 코드 그대로) + 타이밍 측정."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from pathH_common import *
import lightgbm as lgb

d = load()
print("split:", len(d.tr), len(d.ho), "prev tr=%.5f ho=%.5f" % (d.ytr.mean(), d.yho.mean()))

t0 = time.time()
pipe = make_pipe(lgb.LGBMClassifier(**LGB_BASE))
pipe.fit(d.Xtr, d.ytr)
fit_s = time.time() - t0
f = lambda X: pipe.predict_proba(X)[:, 1]
ho, pr, po = f(d.Xho), f(d.Xpre), f(d.Xpost)
r = dict(name="H00_baseline_literal",
         oof=np.nan,
         internal=wauc(d.yho, ho, d.Who),
         ext_pre=wauc(d.ypre, pr, d.wpre),
         ext_post=wauc(d.ypost, po, d.wpost),
         secs=round(fit_s, 1))
print(r)
print("목표: internal 0.8310 / pre 0.7740 / post 0.7653")
print("일치:", [round(r[k], 4) for k in ("internal", "ext_pre", "ext_post")])

pd.DataFrame([r]).to_csv(os.path.join(RES, "pathH_00_baseline.csv"),
                         index=False, encoding="utf-8-sig")
print("1회 fit(500k) = %.1fs" % fit_s)
