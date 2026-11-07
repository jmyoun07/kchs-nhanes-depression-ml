# -*- coding: utf-8 -*-
"""경로A 진단: 변수별 단변량 가중 AUROC / 분포 비교로 concept shift vs covariate shift 분해.
※ NHANES 라벨을 쓰는 '진단'이므로 모델 선택에는 사용하지 않는다(§2.10 보고용 사후 진단)."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathA_common import *

rows = []
TR = K.iloc[ITR]
for c in NUM + BIN:
    r = {"var": c}
    for nm, d in (("kchs", TR), ("pre", PRE), ("post", POST)):
        v = d[c].to_numpy(float); y = d["dep"].astype(int).to_numpy(); w = d["W"].to_numpy(float)
        m = ~np.isnan(v) & ~np.isnan(w)
        if m.sum() == 0 or len(set(y[m])) < 2:
            r[f"auc_{nm}"] = np.nan; r[f"mean_{nm}"] = np.nan; continue
        try:
            r[f"auc_{nm}"] = wauc(y[m], v[m], w[m])
        except Exception:
            r[f"auc_{nm}"] = np.nan
        r[f"mean_{nm}"] = float(np.average(v[m], weights=w[m]))
    r["d_auc_pre"] = r["auc_pre"] - r["auc_kchs"]
    r["d_auc_post"] = r["auc_post"] - r["auc_kchs"]
    rows.append(r)

for c in CAT:
    for lv in sorted(set(K[c].dropna().unique())):
        r = {"var": f"{c}={lv}"}
        for nm, d in (("kchs", TR), ("pre", PRE), ("post", POST)):
            v = (d[c] == lv).astype(float).to_numpy()
            y = d["dep"].astype(int).to_numpy(); w = d["W"].to_numpy(float)
            r[f"auc_{nm}"] = wauc(y, v, w)
            r[f"mean_{nm}"] = float(np.average(v, weights=w))
        r["d_auc_pre"] = r["auc_pre"] - r["auc_kchs"]
        r["d_auc_post"] = r["auc_post"] - r["auc_kchs"]
        rows.append(r)

df = pd.DataFrame(rows)
p = os.path.join(RES, "pathA_shift_diag.csv")
df.to_csv(p, index=False, encoding="utf-8-sig")
print(df.round(4).to_string(index=False))
print("saved:", p)
