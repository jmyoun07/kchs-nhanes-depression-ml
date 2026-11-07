# -*- coding: utf-8 -*-
"""pathG_00: 기준선 재현 + 도메인 진단."""
import os, sys, time, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathG_common import *
from sklearn.pipeline import Pipeline
import lightgbm as lgb

t0 = time.time()
k, pre, post = load()
tr, ho = base_split(k)
y = k["dep"].astype(int).values
print("split sizes", len(tr), len(ho), "prev tr=%.5f ho=%.5f" % (y[tr].mean(), y[ho].mean()))

pipe = Pipeline([("pre", make_ct()), ("clf", lgb.LGBMClassifier(**LGB_PARAMS))])
pipe.fit(k.iloc[tr][FEATURES], y[tr])
pi = pipe.predict_proba(k.iloc[ho][FEATURES])[:, 1]
pp = pipe.predict_proba(pre[FEATURES])[:, 1]
po = pipe.predict_proba(post[FEATURES])[:, 1]
b = (wauc(y[ho], pi, k["W"].to_numpy(float)[ho]),
     wauc(pre["dep"], pp, pre["W"]),
     wauc(post["dep"], po, post["W"]))
rec("G00_baseline_literal", *b, True,
    "과제 명세 코드 그대로(500k/100k, seed42). 목표 0.8310/0.7740/0.7653",
    secs=time.time() - t0)

# 비가중 AUROC도 참고
print("unweighted: int=%.4f pre=%.4f post=%.4f" % (
    roc_auc_score(y[ho], pi), roc_auc_score(pre["dep"], pp), roc_auc_score(post["dep"], po)))

# ---- 진단: 도메인별 결측/분포 ----
diag = []
for c in FEATURES:
    diag.append(dict(feature=c,
                     kchs_miss=round(float(k[c].isna().mean()), 5),
                     pre_miss=round(float(pre[c].isna().mean()), 5),
                     post_miss=round(float(post[c].isna().mean()), 5)))
dd = pd.DataFrame(diag)
print(dd.to_string())
dd.to_csv(os.path.join(RES, "pathG_missingness.csv"), index=False, encoding="utf-8-sig")

# 변수 중요도(기준선 모델)
imp = pipe.named_steps["clf"].feature_importances_
names = pipe.named_steps["pre"].get_feature_names_out()
gi = pd.DataFrame(dict(feature=names, gain=imp)).sort_values("gain", ascending=False)
print(gi.head(25).to_string())
gi.to_csv(os.path.join(RES, "pathG_baseline_importance.csv"), index=False, encoding="utf-8-sig")

np.save(os.path.join(RES, "pathG_base_pred_ho.npy"), pi)
dump("pathG_00_baseline.csv")
print("total %.1fs" % (time.time() - t0))
