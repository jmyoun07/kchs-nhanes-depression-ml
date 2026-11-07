# -*- coding: utf-8 -*-
"""
pathB2_02 — 2단계: 1단계 상위 후보에 대한 KCHS 5-fold CV(가중 AUROC).

§2.10: 여기서도 NHANES는 로드하지 않는다. 최종 선택은 이 CV 점수로만 결정한다.
후보 선정 규칙(사전 확정):
  * 1단계 valid_wauc 전체 상위 8개
  * 각 family(lgbm/xgb/catboost/histgb/linear/mono)별 1위 (중복 제거)
  * 기준선(lgb_base) 반드시 포함
  * 가설검정용: 최고 성능의 '강정규화·얕은트리' 구성 1개, 최고 성능 단조제약 1개
"""
import os, sys, time, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathB2_common import (RES, SEED, FEATURES, load, baseline_split, make_pre,
                           mono_vector, MONO_SETS, wauc, cv_indices, build_model)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

NJ = 22
SCREEN = os.path.join(RES, "pathB2_01_screen.csv")
OUT = os.path.join(RES, "pathB2_02_cv.csv")

sc = pd.read_csv(SCREEN)
sc["p"] = sc["params"].apply(json.loads)

cands = []
seen = set()


def add(name):
    if name in seen:
        return
    r = sc[sc["name"] == name]
    if len(r) == 0:
        return
    seen.add(name)
    cands.append(r.iloc[0])


add("lgb_base")
for n in sc.sort_values("valid_wauc", ascending=False)["name"].head(8):
    add(n)
for fam, g in sc.groupby("family"):
    add(g.sort_values("valid_wauc", ascending=False)["name"].iloc[0])
# 가설 arm: 강정규화·얕은트리 / 단조제약 각각 1위
sr = sc[sc["name"].str.startswith("lgb_combo")]
if len(sr):
    add(sr.sort_values("valid_wauc", ascending=False)["name"].iloc[0])
mo = sc[sc["family"].str.contains("mono")]
if len(mo):
    add(mo.sort_values("valid_wauc", ascending=False)["name"].iloc[0])

print(f"CV candidates ({len(cands)}): {[c['name'] for c in cands]}", flush=True)


# ------------------------------------------------------------------ model factory
def build(name, fam, p):
    return build_model(fam, p, NJ)


# ------------------------------------------------------------------ KCHS만 로드
t0 = time.time()
k, _a, _b = load(); del _a, _b
kk, itr, iho = baseline_split(k)
y = kk["dep"].astype(int).values
W = kk["W"].to_numpy(float)
X = kk[FEATURES].iloc[itr].reset_index(drop=True)
ytr = y[itr]; Wtr = W[itr]
folds = cv_indices(ytr, 5, SEED)
print(f"CV on KCHS train n={len(X):,}  [{time.time()-t0:.0f}s]", flush=True)

rows = []
if os.path.exists(OUT):
    rows = pd.read_csv(OUT).to_dict("records")
donec = {r["name"] for r in rows}

for c in cands:
    if c["name"] in donec:
        continue
    mk = build(c["name"], c["family"], c["p"])
    t1 = time.time(); s = []
    for f, (a, b) in enumerate(folds):
        m = mk(); m.fit(X.iloc[a], ytr[a])
        s.append(wauc(ytr[b], m.predict_proba(X.iloc[b])[:, 1], Wtr[b]))
    rows.append(dict(name=c["name"], family=c["family"], cv_mean=float(np.mean(s)),
                     cv_std=float(np.std(s)), screen_wauc=c["valid_wauc"],
                     sec=round(time.time() - t1, 1), params=c["params"]))
    print(f"  {c['name']:44s} cv={np.mean(s):.5f}±{np.std(s):.5f} "
          f"[{time.time()-t1:.0f}s]", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")

d = pd.DataFrame(rows).sort_values("cv_mean", ascending=False)
d.to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"\nDONE {time.time()-t0:.0f}s -> {OUT}")
print(d[["name", "family", "cv_mean", "cv_std", "screen_wauc"]].to_string(index=False))
