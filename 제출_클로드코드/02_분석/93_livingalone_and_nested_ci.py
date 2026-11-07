# -*- coding: utf-8 -*-
"""93_livingalone_and_nested_ci.py — 독립감사가 지적한 두 가지 재계산.

(1) living_alone 통제 대조.
    living_alone 은 KCHS 0% 결측 / NHANES 사전 100% 결측 / 사후 0% 결측이다.
    학습자료에 결측이 하나도 없으므로 LightGBM 은 결측 분기를 배우지 못하고,
    예측 시 NaN 을 0 가지(=동거)로 보낸다 — 사실상 상수 대체다.
    따라서 사전/사후 대조가 '시기 차이'와 '변수 유무 차이'로 뒤섞여 있다.
    living_alone 을 뺀 16변수 모형으로 같은 대조를 다시 계산해 교란 크기를 보인다.

(2) 재보정 구간의 중첩(nested) 재계산.
    90 번은 재보정 지도를 전체 코호트에서 한 번 적합한 뒤 고정하고 재표본을 돌렸다.
    그러면 H3 가 다루는 바로 그 갱신의 불확실성이 구간에서 빠진다.
    오라클(표 S15)에서는 이미 중첩 방식을 쓰고 있으므로 여기서도 맞춘다.

출력: results/primary2_livingalone_control.csv, results/primary2_calibration_ci.csv(갱신)
"""
import io, os, sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__)); RES = f"{D}/results"

src = io.open(f"{D}/70_primary_corrected.py", encoding="utf8").read()
g = {"__file__": f"{D}/70_primary_corrected.py", "__name__": "__prep__"}
exec(compile(src[: src.index("mrows=[]")], "70_prep", "exec"), g)
K, P, S, FE = g["K"], g["P"], g["S"], g["FE"]
tr, ho = g["tr"], g["ho"]; y = g["y"]
yho, yp, ys = g["yho"], g["yp"], g["ys"]; Who, Wp, Ws = g["Who"], g["Wp"], g["Ws"]
p_pre, p_post = g["p_pre"], g["p_post"]; REPS = g["REPS"]; SEED = g["SEED"]
LGB = g["LGB"]
cal_slope, citl, ece, recal = g["cal_slope"], g["citl"], g["ece"], g["recal"]
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
import lightgbm as lgb
grp = lambda d: (d["strata"].astype(str) + "|" + d["psu"].astype(str)).values
AUC = lambda yv, p, w: roc_auc_score(yv, p, sample_weight=w)

def boot(yv, p, reps, fn):
    out = []
    for w in reps:
        m = w > 0
        if m.sum() < 10 or yv[m].min() == yv[m].max(): continue
        try: out.append(fn(yv, p, w))
        except Exception: pass
    return np.array(out)

def diff_ci(a_pre, a_post, obs, seed=SEED + 1, Kp=20000):
    rs = np.random.RandomState(seed)
    d = a_post[rs.randint(0, len(a_post), Kp)] - a_pre[rs.randint(0, len(a_pre), Kp)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(lo), float(hi), float(max(p, 1 / Kp))

# ══════════════ (1) living_alone 통제 대조 ══════════════
print("[1] living_alone 결측 구조 확인", flush=True)
for nm, d in [("KCHS", K), ("NHANES pre", P), ("NHANES post", S)]:
    print(f"    {nm:<12} living_alone 결측 {pd.to_numeric(d['living_alone'], errors='coerce').isna().mean()*100:.1f}%", flush=True)

rows = []
a_pre17 = boot(yp, p_pre, REPS["NHANES pre"], AUC); a_post17 = boot(ys, p_post, REPS["NHANES post"], AUC)
o17 = AUC(ys, p_post, Ws) - AUC(yp, p_pre, Wp)
lo17, hi17, p17 = diff_ci(a_pre17, a_post17, o17)
rows.append(dict(predictors="17 (primary)", n_predictors=17, internal=np.nan,
                 ext_pre=round(AUC(yp, p_pre, Wp), 6), ext_post=round(AUC(ys, p_post, Ws), 6),
                 delta=round(o17, 6), CI_lo=round(lo17, 6), CI_hi=round(hi17, 6), P=round(p17, 4)))
print(f"    17변수  pre {rows[-1]['ext_pre']:.4f} post {rows[-1]['ext_post']:.4f} · 차 {o17:+.4f} [{lo17:+.4f},{hi17:+.4f}]", flush=True)

FE16 = [c for c in FE if c != "living_alone"]
NUM, BIN, CAT = g["NUM"], g["BIN"], g["CAT"]
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
N16 = [c for c in NUM if c in FE16]; B16 = [c for c in BIN if c in FE16]; C16 = [c for c in CAT if c in FE16]
ct16 = ColumnTransformer([("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), N16),
                          ("bin", "passthrough", B16),
                          ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                                            ("o", OneHotEncoder(handle_unknown="ignore", drop="first"))]), C16)])
print("[2] 16변수(living_alone 제외) 모형 적합", flush=True)
m16 = Pipeline([("pre", ct16), ("clf", lgb.LGBMClassifier(**LGB))]).fit(K.iloc[tr][N16 + B16 + C16], y[tr])
q_pre = m16.predict_proba(P[N16 + B16 + C16])[:, 1]; q_post = m16.predict_proba(S[N16 + B16 + C16])[:, 1]
q_ho = m16.predict_proba(K.iloc[ho][N16 + B16 + C16])[:, 1]
a_pre16 = boot(yp, q_pre, REPS["NHANES pre"], AUC); a_post16 = boot(ys, q_post, REPS["NHANES post"], AUC)
o16 = AUC(ys, q_post, Ws) - AUC(yp, q_pre, Wp)
lo16, hi16, p16 = diff_ci(a_pre16, a_post16, o16)
rows.append(dict(predictors="16 (living alone removed)", n_predictors=16,
                 internal=round(AUC(yho, q_ho, Who), 6),
                 ext_pre=round(AUC(yp, q_pre, Wp), 6), ext_post=round(AUC(ys, q_post, Ws), 6),
                 delta=round(o16, 6), CI_lo=round(lo16, 6), CI_hi=round(hi16, 6), P=round(p16, 4)))
print(f"    16변수  내부 {rows[-1]['internal']:.4f} pre {rows[-1]['ext_pre']:.4f} post {rows[-1]['ext_post']:.4f} "
      f"· 차 {o16:+.4f} [{lo16:+.4f},{hi16:+.4f}]", flush=True)
pd.DataFrame(rows).to_csv(f"{RES}/primary2_livingalone_control.csv", index=False, encoding="utf-8-sig")

# ══════════════ (2) 중첩 재보정 구간 ══════════════
print("[3] 재보정 구간을 중첩(재표본마다 갱신 재적합)으로 다시 계산", flush=True)
out = []
for nm, d, yv, p, w in [("NHANES pre", P, yp, p_pre, Wp), ("NHANES post", S, ys, p_post, Ws)]:
    R = REPS[nm]; G = grp(d)
    pr_fixed = recal(yv, p, w, "slope", groups=G)
    for lab, fixed in [("as transported", p), ("after intercept/slope recalibration", pr_fixed)]:
        sl = boot(yv, fixed, R, cal_slope); ct_ = boot(yv, fixed, R, citl); ec = boot(yv, fixed, R, ece)
        rec = dict(set=nm, probabilities=lab,
                   cal_slope=round(cal_slope(yv, fixed, w), 6),
                   slope_lo=round(float(np.percentile(sl, 2.5)), 6), slope_hi=round(float(np.percentile(sl, 97.5)), 6),
                   CITL=round(citl(yv, fixed, w), 6),
                   CITL_lo=round(float(np.percentile(ct_, 2.5)), 6), CITL_hi=round(float(np.percentile(ct_, 97.5)), 6),
                   ECE=round(ece(yv, fixed, w), 6), nested="no")
        out.append(rec)
    # 중첩: 재표본마다 갱신을 다시 적합한다
    sl_n, ct_n = [], []
    for b, wb in enumerate(R):
        m = wb > 0
        if m.sum() < 10 or yv[m].min() == yv[m].max(): continue
        try:
            prb = recal(yv, p, wb, "slope", groups=G)
            sl_n.append(cal_slope(yv, prb, wb)); ct_n.append(citl(yv, prb, wb))
        except Exception: pass
        if (b + 1) % 250 == 0: print(f"      {nm} 중첩 {b+1}/{len(R)}", flush=True)
    sl_n = np.array(sl_n); ct_n = np.array(ct_n)
    out.append(dict(set=nm, probabilities="after intercept/slope recalibration (nested)",
                    cal_slope=round(cal_slope(yv, pr_fixed, w), 6),
                    slope_lo=round(float(np.percentile(sl_n, 2.5)), 6), slope_hi=round(float(np.percentile(sl_n, 97.5)), 6),
                    CITL=round(citl(yv, pr_fixed, w), 6),
                    CITL_lo=round(float(np.percentile(ct_n, 2.5)), 6), CITL_hi=round(float(np.percentile(ct_n, 97.5)), 6),
                    ECE=round(ece(yv, pr_fixed, w), 6), nested="yes"))
    print(f"    {nm} 중첩 기울기 [{out[-1]['slope_lo']:.3f},{out[-1]['slope_hi']:.3f}] "
          f"(고정 [{out[-3]['slope_lo']:.3f},{out[-3]['slope_hi']:.3f}])", flush=True)
pd.DataFrame(out).to_csv(f"{RES}/primary2_calibration_ci.csv", index=False, encoding="utf-8-sig")
print("저장: primary2_livingalone_control.csv · primary2_calibration_ci.csv")
print("주의: ECE 는 비음수·상향편의 통계라 백분위 구간이 점추정을 포함하지 않을 수 있어 구간을 싣지 않는다.")
