# -*- coding: utf-8 -*-
"""경로A 단계2: 스플라인/구간화 · 명시적 상호작용 · srh/oral_health 인코딩 변형.

선형모델(로지스틱)에서 특히 유효하므로 LGBM과 로지스틱 양쪽을 본다.
모든 인코딩 파라미터(노트 위치, 타깃인코딩 값)는 KCHS train에서만 추정 → §2.10 준수.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathA_common import *
from sklearn.linear_model import LogisticRegression

OUT = os.path.join(RES, "pathA_spline_int.csv")
ROWS = []


def rec(res, compliant, note=""):
    r = dict(res); r["protocol_compliant"] = bool(compliant); r["note"] = note
    ROWS.append(r)
    pd.DataFrame(ROWS).to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"{r['name']:38s} int={r['internal']:.4f} pre={r['ext_pre']:.4f} "
          f"post={r['ext_post']:.4f} ok={compliant} [{r['seconds']}s] {note}", flush=True)
    return r


def LR():
    return LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")


TRK = K.iloc[ITR]

# ============================================================ 0) 선형 기준선
r, _ = evaluate(K, PRE, POST, NUM, BIN, CAT, model=LR(), tag="B00_logit_linear")
rec(r, True, "로지스틱 기준선(17변수 선형)")

# ============================================================ 1) 스플라인
KNOTS = {}
for c in ["age", "bmi", "sleep_hrs"]:
    v = TRK[c].dropna().to_numpy(float)
    KNOTS[c] = list(np.quantile(v, [0.05, 0.275, 0.5, 0.725, 0.95]))
print("knots(train-only):", {k: [round(x, 2) for x in v] for k, v in KNOTS.items()}, flush=True)


def build_spline(dfs, cols):
    out = list(dfs); new = []
    for c in cols:
        out, nm = add_rcs(out, c, KNOTS[c])
        new += nm
    return out, new


(Ks, Ps, Qs), SP = build_spline((K, PRE, POST), ["age", "bmi", "sleep_hrs"])
num_sp = [c for c in NUM if c not in ("age", "bmi", "sleep_hrs")] + SP

r, _ = evaluate(Ks, Ps, Qs, num_sp, BIN, CAT, model=LR(), tag="B01_logit_rcs3")
rec(r, True, "age/bmi/sleep RCS5 (knot=KCHS train 분위)")
r, _ = evaluate(Ks, Ps, Qs, num_sp, BIN, CAT, tag="B02_lgbm_rcs3")
rec(r, True, "LGBM+RCS (트리는 단조변환 불변 → 무효과 예상)")

# ============================================================ 2) 상호작용 (선형모델)
def add_inter(dfs, pairs):
    outs = []
    names = [f"ix_{a}_{b}" for a, b in pairs]
    for d in dfs:
        dd = d.copy()
        for (a, b), nm in zip(pairs, names):
            va = dd[a] if dd[a].dtype != object else (dd[a] == dd[a].dropna().unique()[0]).astype(float)
            vb = dd[b] if dd[b].dtype != object else (dd[b] == dd[b].dropna().unique()[0]).astype(float)
            dd[nm] = va.astype(float) * vb.astype(float)
        outs.append(dd)
    return outs, names


# sex를 숫자로 (M=1)
def sexnum(dfs):
    outs = []
    for d in dfs:
        dd = d.copy(); dd["sex_m"] = (dd["sex"] == "M").astype(float); outs.append(dd)
    return outs


Kx, Px, Qx = sexnum((Ks, Ps, Qs))
PAIRS = [("age", "sex_m"), ("srh", "age"), ("srh", "oral_health"),
         ("employed", "age"), ("srh", "sex_m"), ("living_alone", "age"),
         ("bmi", "sex_m"), ("income_q", "age")]
(Kx, Px, Qx), IX = add_inter((Kx, Px, Qx), PAIRS)

r, _ = evaluate(Kx, Px, Qx, num_sp + IX, BIN, CAT, model=LR(), tag="B03_logit_rcs_inter")
rec(r, True, "RCS + 명시적 상호작용 8개")
r, _ = evaluate(Kx, Px, Qx, NUM + IX, BIN, CAT, model=LR(), tag="B04_logit_inter_only")
rec(r, True, "선형 + 상호작용만")
r, _ = evaluate(Kx, Px, Qx, num_sp + IX, BIN, CAT, tag="B05_lgbm_rcs_inter")
rec(r, True, "LGBM + RCS + 상호작용")

# ============================================================ 3) srh / oral_health 인코딩 변형
# 3a) one-hot (범주형 취급)
Kc, Pc, Qc = K.copy(), PRE.copy(), POST.copy()
for d in (Kc, Pc, Qc):
    d["srh_c"] = d["srh"].astype("Int64").astype(str)
    d["oh_c"] = d["oral_health"].astype("Int64").astype(str)
num_no = [c for c in NUM if c not in ("srh", "oral_health")]
r, _ = evaluate(Kc, Pc, Qc, num_no, BIN, CAT + ["srh_c", "oh_c"], model=LR(), tag="B06_logit_srh_onehot")
rec(r, True, "srh/oral_health One-Hot")
r, _ = evaluate(Kc, Pc, Qc, num_no, BIN, CAT + ["srh_c", "oh_c"], tag="B07_lgbm_srh_onehot")
rec(r, True, "srh/oral_health One-Hot + LGBM")

# 3b) 이진화 (원값 절단 vs 설문내 분위 절단)
Kb, Pb, Qb = K.copy(), PRE.copy(), POST.copy()
for d in (Kb, Pb, Qb):
    d["srh_bad"] = (d["srh"] >= 4).astype(float)
    d["oh_bad"] = (d["oral_health"] >= 4).astype(float)
r, _ = evaluate(Kb, Pb, Qb, num_no, BIN + ["srh_bad", "oh_bad"], CAT, tag="B08_lgbm_srh_bin_raw")
rec(r, True, "srh/oh 원값 절단 이진화(>=4)")

# 설문내 분위 절단: 각 설문에서 상위 p% 를 'bad'로. p는 KCHS train 기준 비율로 고정.
pK = float(np.average((K.iloc[ITR]["srh"] >= 4).astype(float),
                      weights=K.iloc[ITR]["W"]))
pO = float(np.average((K.iloc[ITR]["oral_health"] >= 4).astype(float),
                      weights=K.iloc[ITR]["W"]))
Kq, Pq, Qq = K.copy(), PRE.copy(), POST.copy()
for d, src in ((Kq, K), (Pq, PRE), (Qq, POST)):
    for col, tgt, p in (("srh", "srh_bad", pK), ("oral_health", "oh_bad", pO)):
        rk = wpct(src[col].to_numpy(float), src["W"].to_numpy(float), src[col].to_numpy(float))
        d[tgt] = (rk >= (1 - p)).astype(float)
r, _ = evaluate(Kq, Pq, Qq, num_no, BIN + ["srh_bad", "oh_bad"], CAT, tag="B09_lgbm_srh_bin_pctmatch")
rec(r, True, f"설문내 분위절단 이진화(KCHS 비율 srh={pK:.3f} oh={pO:.3f} 매칭)")

# 3c) 타깃 인코딩 (KCHS train 라벨만 사용 → 준수)
Kt, Pt, Qt = K.copy(), PRE.copy(), POST.copy()
tr = K.iloc[ITR]
for col, nm in (("srh", "srh_te"), ("oral_health", "oh_te")):
    g = pd.DataFrame({col: tr[col].to_numpy(float), "y": yK[ITR], "w": WK[ITR]})
    m = g.dropna().groupby(col).apply(lambda z: np.average(z["y"], weights=z["w"]))
    for d in (Kt, Pt, Qt):
        d[nm] = d[col].map(m).astype(float)
r, _ = evaluate(Kt, Pt, Qt, num_no + ["srh_te", "oh_te"], BIN, CAT, model=LR(), tag="B10_logit_srh_target")
rec(r, True, "srh/oh 타깃인코딩(KCHS train 라벨만)")
r, _ = evaluate(Kt, Pt, Qt, num_no + ["srh_te", "oh_te"], BIN, CAT, tag="B11_lgbm_srh_target")
rec(r, True, "srh/oh 타깃인코딩 + LGBM")

# ============================================================ 4) 연속변수 구간화(설문내 분위 10분위)
Kd, Pd, Qd = K.copy(), PRE.copy(), POST.copy()
for d, src in ((Kd, K), (Pd, PRE), (Qd, POST)):
    for c in ["age", "bmi", "sleep_hrs"]:
        rk = wpct(src[c].to_numpy(float), src["W"].to_numpy(float), src[c].to_numpy(float))
        d[c] = np.floor(np.clip(rk, 0, 0.999) * 10)
r, _ = evaluate(Kd, Pd, Qd, NUM, BIN, CAT, tag="B12_lgbm_decile_phys")
rec(r, True, "age/bmi/sleep 설문내 10분위 구간화")

print("saved:", OUT)
