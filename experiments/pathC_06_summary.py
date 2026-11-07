# -*- coding: utf-8 -*-
"""
pathC_06_summary.py — 경로C 전체 결과 통합 + 시드 변동성(노이즈 바닥) 정량화.

시드 변동성은 '얼마나 작은 개선이 노이즈인지'의 기준이므로 반드시 함께 보고한다.
출력: results/pathC_summary.csv, results/pathC_seed_variance.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from pathC_common import RES

BASE = dict(internal=0.8335354, ext_pre=0.7681989, ext_post=0.7680069)

# ---- 시드 변동성 ----
bm = os.path.join(RES, "pathC_base_models.csv")
seedrows = None
if os.path.exists(bm):
    d = pd.read_csv(bm)
    s = d[d.model.str.startswith("lgbm_s")]
    if len(s) > 1:
        seedrows = pd.DataFrame([dict(
            metric=c, n_seeds=len(s), mean=s[c].mean(), sd=s[c].std(ddof=1),
            min=s[c].min(), max=s[c].max(), range=s[c].max() - s[c].min())
            for c in ["internal", "ext_pre", "ext_post"]])
        seedrows.to_csv(os.path.join(RES, "pathC_seed_variance.csv"),
                        index=False, encoding="utf-8-sig")
        print("=== 단일 LGBM 시드 변동성 (기준선 하이퍼파라미터, 시드만 변경) ===")
        print(seedrows.to_string(index=False))
        print()

# ---- 결과 통합 ----
frames = []
spec = [("pathC_base_models.csv", "C1_base_model", "model"),
        ("pathC_oof_models.csv", "C2_oof_model", "model"),
        ("pathC_ensembles.csv", None, "method"),
        ("pathC_bagging.csv", "C4_bagging", "method"),
        ("pathC_oracle.csv", "C5_oracle", "what")]
for fn, fam, key in spec:
    p = os.path.join(RES, fn)
    if not os.path.exists(p):
        print(f"(없음) {fn}")
        continue
    d = pd.read_csv(p)
    d["source"] = fn
    d["exp"] = d[key]
    d["family"] = d["family"] if ("family" in d.columns and fam is None) else fam
    if "protocol_compliant" not in d.columns:
        d["protocol_compliant"] = True
    frames.append(d[["source", "family", "exp", "internal", "ext_pre",
                     "ext_post", "protocol_compliant"]])
if frames:
    S = pd.concat(frames, ignore_index=True)
    S["d_pre"] = S.ext_pre - BASE["ext_pre"]
    S["d_post"] = S.ext_post - BASE["ext_post"]
    S["d_int"] = S.internal - BASE["internal"]
    S["mean_ext"] = (S.ext_pre + S.ext_post) / 2
    S = S.sort_values("mean_ext", ascending=False)
    S.to_csv(os.path.join(RES, "pathC_summary.csv"), index=False, encoding="utf-8-sig")
    print("=== 상위 25 (외부 pre/post 평균 기준) ===")
    print(S.head(25).to_string(index=False))
    print("\n=== §2.10 준수 실험만, 상위 15 ===")
    print(S[S.protocol_compliant == True].head(15).to_string(index=False))
    print("\n저장:", os.path.join(RES, "pathC_summary.csv"), S.shape)
