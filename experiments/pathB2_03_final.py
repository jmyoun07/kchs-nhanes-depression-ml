# -*- coding: utf-8 -*-
"""
pathB2_03 — 3단계: 사전 확정 후보에 대한 최종 외부평가(NHANES 1회, 재적합 없음).

§2.10 준수 설계
  * 선택(=primary model)은 2단계 KCHS 5-fold CV 1위로 **이미 확정**되어 있다.
  * 나머지 후보는 사전 규칙(각 family 1위, 기준선, 단조제약 1위, 강정규화 1위)으로
    뽑은 '사전지정 부차분석'이며, NHANES 성적을 보고 고른 것이 아니다.
  * NHANES는 학습에 절대 사용하지 않는다(fit은 KCHS train만).

추가로 진단용 상한(oracle) 참고치를 계산한다. 이는 프로토콜 위반(NHANES 재적합)이므로
'참고 진단'으로만 보고하며 어떤 선택에도 사용하지 않는다.
"""
import os, sys, time, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathB2_common import (RES, SEED, FEATURES, load, baseline_split, make_pre,
                           mono_vector, MONO_SETS, wauc, ext_eval, cv_indices)
from pathB2_common import build_model


def build(name, fam, p):
    return build_model(fam, p, 22)
import warnings; warnings.filterwarnings("ignore")

CVF = os.path.join(RES, "pathB2_02_cv.csv")
OUT = os.path.join(RES, "pathB2_03_final_external.csv")

cv = pd.read_csv(CVF).sort_values("cv_mean", ascending=False).reset_index(drop=True)
cv["p"] = cv["params"].apply(json.loads)

# --------- 사전 확정 후보 선정 규칙(외부성적 미열람 상태에서 확정) ---------------
sel, seen = [], set()


def add(row, role):
    if row["name"] in seen:
        return
    seen.add(row["name"])
    sel.append((row, role))


add(cv.iloc[0], "PRIMARY (KCHS CV 1위)")
for fam, g in cv.groupby("family"):
    add(g.sort_values("cv_mean", ascending=False).iloc[0], f"family-best:{fam}")
b = cv[cv["name"] == "lgb_base"]
if len(b):
    add(b.iloc[0], "baseline")
m = cv[cv["family"].str.contains("mono")]
if len(m):
    add(m.sort_values("cv_mean", ascending=False).iloc[0], "hypothesis:monotone")
print(f"final candidates ({len(sel)}):")
for r, role in sel:
    print(f"  {r['name']:42s} cv={r['cv_mean']:.5f}  [{role}]")

# ------------------------------------------------------------------ 데이터
t0 = time.time()
k, pre, post = load()
kk, itr, iho = baseline_split(k)
y = kk["dep"].astype(int).values
W = kk["W"].to_numpy(float)
X = kk[FEATURES]

rows = []
for r, role in sel:
    t1 = time.time()
    mdl = build(r["name"], r["family"], r["p"])()
    mdl.fit(X.iloc[itr], y[itr])                 # KCHS train only
    ai = wauc(y[iho], mdl.predict_proba(X.iloc[iho])[:, 1], W[iho])
    e = ext_eval(mdl, pre, post)                 # NHANES: 예측만, 재적합 없음
    rows.append(dict(name=r["name"], role=role, family=r["family"],
                     cv_mean=r["cv_mean"], internal=ai,
                     ext_pre=e["ext_pre"], ext_post=e["ext_post"],
                     protocol_compliant=True, sec=round(time.time() - t1, 1),
                     params=r["params"]))
    print(f"  {r['name']:42s} int={ai:.4f} pre={e['ext_pre']:.4f} "
          f"post={e['ext_post']:.4f} [{time.time()-t1:.0f}s]", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")

# ------------------------------------------------------------------ 진단용 상한(프로토콜 위반)
print("\n[진단 전용 / 프로토콜 위반 — 선택에 미사용] NHANES 자체 재적합 상한", flush=True)
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
for nm, d in [("ext_pre", pre), ("ext_post", post)]:
    yy = d["dep"].astype(int).values
    ww = d["W"].to_numpy(float)
    XX = d[FEATURES].reset_index(drop=True)
    oof = np.zeros(len(d))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(XX, yy):
        mm = Pipeline([("pre", make_pre()),
                       ("clf", lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03,
                                                  num_leaves=15, min_child_samples=50,
                                                  subsample=0.8, colsample_bytree=0.8,
                                                  random_state=SEED, n_jobs=22, verbose=-1))])
        mm.fit(XX.iloc[a], yy[a])
        oof[b] = mm.predict_proba(XX.iloc[b])[:, 1]
    s = wauc(yy, oof, ww)
    rows.append(dict(name=f"ORACLE_refit_on_{nm}", role="DIAGNOSTIC ONLY (protocol violation)",
                     family="oracle", cv_mean=np.nan, internal=np.nan,
                     ext_pre=s if nm == "ext_pre" else np.nan,
                     ext_post=s if nm == "ext_post" else np.nan,
                     protocol_compliant=False, sec=np.nan, params="{}"))
    print(f"  oracle 5-fold-CV within {nm}: {s:.4f}", flush=True)

pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"\nDONE {time.time()-t0:.0f}s -> {OUT}")
