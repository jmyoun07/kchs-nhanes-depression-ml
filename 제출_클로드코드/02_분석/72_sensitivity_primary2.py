# -*- coding: utf-8 -*-
"""72_sensitivity_primary2.py — 나머지 민감도 사양을 '교정된 주분석' 데이터로 재계산.

§3.6 의 결측지시자·연쇄방정식 대체·연령 ≥30·절단점 ≥5 는 정정 이전 세대(final_sensitivity.csv,
sensitivity_full.csv)에서 계산된 값이어서 주분석과 세대가 달랐다.  여기서는 70_primary_corrected.py
와 동일한 교정·군집분할 위에서 다시 계산해 primary2_sensitivity.csv 에 합친다.
"""
import os, io, sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__)); RES = f"{D}/results"
src = io.open(f"{D}/70_primary_corrected.py", encoding="utf8").read()
g = {"__file__": f"{D}/70_primary_corrected.py", "__name__": "__prep__"}
exec(compile(src[: src.index("p_tr = model.predict_proba")], "70_prep", "exec"), g)
K, P, S, FE, NUM, BIN, CAT = g["K"], g["P"], g["S"], g["FE"], g["NUM"], g["BIN"], g["CAT"]
tr, ho, y, SEED, LGB = g["tr"], g["ho"], g["y"], g["SEED"], g["LGB"]
Who, Wp, Ws, yho, yp, ys = g["Who"], g["Wp"], g["Ws"], g["yho"], g["yp"], g["ys"]
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

def ct(kind="median"):
    num = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())])
    if kind == "indicator":
        num = Pipeline([("i", SimpleImputer(strategy="median", add_indicator=True)), ("s", StandardScaler())])
    elif kind == "mice":
        num = Pipeline([("i", IterativeImputer(max_iter=5, random_state=SEED)), ("s", StandardScaler())])
    return ColumnTransformer([("num", num, NUM), ("bin", "passthrough", BIN),
                              ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                                                ("o", OneHotEncoder(handle_unknown="ignore", drop="first"))]), CAT)])

def run(label, kind="median", tr_idx=None, ho_idx=None, yv=None, ext=None):
    tri = tr if tr_idx is None else tr_idx
    hoi = ho if ho_idx is None else ho_idx
    yy = y if yv is None else yv
    m = Pipeline([("pre", ct(kind)), ("clf", lgb.LGBMClassifier(**LGB))]).fit(K.iloc[tri][FE], yy[tri])
    if ext is None: ext = [(P, yp, Wp), (S, ys, Ws)]
    i = roc_auc_score(yy[hoi], m.predict_proba(K.iloc[hoi][FE])[:, 1], sample_weight=K.iloc[hoi]["W"].to_numpy(float))
    a, b = [roc_auc_score(yv_, m.predict_proba(d[FE])[:, 1], sample_weight=w_) for d, yv_, w_ in ext]
    print(f"   {label:<48} 내부 {i:.4f} pre {a:.4f} post {b:.4f}", flush=True)
    return dict(cond=label, internal=round(i, 6), ext_pre=round(a, 6), ext_post=round(b, 6))

rows = []
print("[민감도 — 교정 데이터]", flush=True)
rows.append(run("Missing-indicator imputation", "indicator"))
rows.append(run("Single chained-equations imputation", "mice"))

age = pd.to_numeric(K["age"], errors="coerce").to_numpy(float)
tr30 = tr[age[tr] >= 30]; ho30 = ho[age[ho] >= 30]
P30 = P[pd.to_numeric(P["age"], errors="coerce") >= 30]; S30 = S[pd.to_numeric(S["age"], errors="coerce") >= 30]
e30 = [(P30, P30["dep"].astype(int).values, P30["W"].to_numpy(float)),
       (S30, S30["dep"].astype(int).values, S30["W"].to_numpy(float))]
rows.append(run("Age ≥30 (train and evaluate)", tr_idx=tr30, ho_idx=ho30, ext=e30))
rows.append(run("Age ≥30 (train only)", tr_idx=tr30))

y5 = (pd.to_numeric(K["phq_sum"], errors="coerce") >= 5).astype(int).to_numpy()
e5 = [(P, (pd.to_numeric(P["phq_sum"], errors="coerce") >= 5).astype(int).to_numpy(), Wp),
      (S, (pd.to_numeric(S["phq_sum"], errors="coerce") >= 5).astype(int).to_numpy(), Ws)]
rows.append(run("Outcome PHQ-9 ≥5", yv=y5, ext=e5))
prev = {"internal": np.average(y5[ho], weights=Who), "pre": np.average(e5[0][1], weights=Wp), "post": np.average(e5[1][1], weights=Ws)}
print("   PHQ-9 ≥5 가중 유병률:", {k: f"{v*100:.1f}%" for k, v in prev.items()}, flush=True)
pd.DataFrame([dict(set=k, prevalence=round(float(v), 4)) for k, v in prev.items()]).to_csv(
    f"{RES}/primary2_cut5_prevalence.csv", index=False, encoding="utf-8-sig")

old = pd.read_csv(f"{RES}/primary2_sensitivity.csv")
new = pd.concat([old[~old["cond"].isin([r["cond"] for r in rows])], pd.DataFrame(rows)], ignore_index=True)
new.to_csv(f"{RES}/primary2_sensitivity.csv", index=False, encoding="utf-8-sig")
print("\n저장: primary2_sensitivity.csv (+ primary2_cut5_prevalence.csv)\n", new.to_string(), flush=True)
