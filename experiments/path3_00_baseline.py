# -*- coding: utf-8 -*-
"""path3_00_baseline.py — 과제 지정 기준선 정확 재현 확인."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from path3_common import Bench, Recorder, RES, TARGET

B = Bench()
R = Recorder(os.path.join(RES, "path3_00_baseline.csv"))
R.log(f"kchs={B.k.shape} pre={B.pre.shape} post={B.post.shape}")
R.log(f"tr={len(B.tr):,} ho={len(B.ho):,} pool={len(B.pool):,}")
R.log(f"dep rate: train={B.y[B.tr].mean():.5f} ho={B.yho.mean():.5f} "
      f"pre={B.ypre.mean():.4f} post={B.ypost.mean():.4f}")

r, _, _ = B.run("BASE_spec_500k", B.tr, note="과제지정 기준선 코드 그대로")
R.base = r
R.add(r, "0_BASE", True)

R.log("\n--- 목표치 대조 (소수 3자리) ---")
for kk, tv in TARGET.items():
    got = r[{"internal": "internal", "ext_pre": "ext_pre", "ext_post": "ext_post"}[kk]]
    ok = round(got, 3) == round(tv, 3)
    R.log(f"  {kk:9s} 목표={tv:.4f} 실측={got:.4f} 3자리일치={ok}")
R.log(f"\n저장: {R.out}")
