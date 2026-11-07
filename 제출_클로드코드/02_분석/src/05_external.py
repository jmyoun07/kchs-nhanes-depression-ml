# -*- coding: utf-8 -*-
"""
05_external.py (교차검증 Batch3) — 외부검증 + 재보정.
- 재보정: (a) 교차적합(5-fold, out-of-sample), (b) #5 전이검증 pre적합→post적용.
- #6 재보정 후 작동특성(임계 재산출) 보고. #12 CITL(offset) 별도. #7 보정곡선에 재보정curve.
- 복합표본 가중 평가.
결과: results/external_metrics.csv, results/figures/calibration.png
"""
import os, warnings, numpy as np, pandas as pd, joblib
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
from scipy.optimize import brentq

ROOT = r"C:/Users/rudtn/Desktop/claude code/지역사회건강"
PROC = os.path.join(ROOT, "data/processed"); MODELS = os.path.join(ROOT, "models"); RES = os.path.join(ROOT, "results")
os.makedirs(os.path.join(RES, "figures"), exist_ok=True); SEED = 42
prim = joblib.load(os.path.join(MODELS, "primary.joblib")); FEATURES = prim["features"]
model = joblib.load(os.path.join(MODELS, f"{prim['primary']}.joblib"))
QC = []; qc = lambda s: (QC.append(str(s)), print(s))
load = lambda n: pd.read_parquet(os.path.join(PROC, f"{n}.parquet"))
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

def cal_slope(y, p, w):
    c = LogisticRegression(penalty=None, solver="lbfgs").fit(logit(p).reshape(-1, 1), y, sample_weight=w)
    return float(c.coef_[0, 0])
def citl(y, p, w):  # #12 calibration-in-the-large (slope=1 고정, offset)
    lp = logit(p)
    try: return float(brentq(lambda a: np.sum(w * (y - 1 / (1 + np.exp(-(a + lp))))), -10, 10))
    except Exception: return np.nan
def thr_at_sens(y, p, w, t=0.85):
    fpr, tpr, thr = roc_curve(y, p, sample_weight=w); idx = np.where(tpr >= t)[0]
    return float(thr[idx[0]]) if len(idx) else 0.5
def opchar(y, p, w, thr):
    yp = (p >= thr).astype(int)
    tp = np.sum(w * ((yp == 1) & (y == 1))); fp = np.sum(w * ((yp == 1) & (y == 0)))
    tn = np.sum(w * ((yp == 0) & (y == 0))); fn = np.sum(w * ((yp == 0) & (y == 1)))
    return dict(sens=tp / (tp + fn), spec=tn / (tn + fp), ppv=tp / (tp + fp) if tp + fp else np.nan, npv=tn / (tn + fn) if tn + fn else np.nan)
def bm(y, p, w):
    return dict(n=len(y), obs=float(np.average(y, weights=w)), pred=float(np.average(p, weights=w)),
                AUROC=roc_auc_score(y, p, sample_weight=w), AUPRC=average_precision_score(y, p, sample_weight=w),
                Brier=brier_score_loss(y, p, sample_weight=w), cal_slope=cal_slope(y, p, w), citl=citl(y, p, w))
def crossfit_recal(y, p, w, k=5):
    pr = np.empty_like(p, dtype=float)
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=SEED).split(p.reshape(-1, 1), y):
        lr = LogisticRegression(penalty=None, solver="lbfgs").fit(logit(p[tr]).reshape(-1, 1), y[tr], sample_weight=w[tr])
        pr[te] = lr.predict_proba(logit(p[te]).reshape(-1, 1))[:, 1]
    return pr

# 내부 hold-out 재현 → 임계값(가중 민감도≥0.85)
dev = load("kchs_dev"); y = dev["dep"].astype(int).values
_, Xho, _, yho, _, Who = train_test_split(dev[FEATURES], y, dev["W"].to_numpy(float), test_size=0.2, stratify=y, random_state=SEED)
pho = model.predict_proba(Xho)[:, 1]; yho = np.asarray(yho, int); Who = np.asarray(Who, float)
THR = thr_at_sens(yho, pho, Who); qc(f"내부 가중 민감도≥0.85 임계값 = {THR:.4f}")

sets = {"Internal": (yho, pho, Who)}
for nm, key in [("NHANES pre", "nhanes_pre"), ("NHANES post", "nhanes_post")]:
    d = load(key); sets[nm] = (d["dep"].to_numpy(float).astype(int), model.predict_proba(d[FEATURES])[:, 1], d["W"].to_numpy(float))

rows = []; recal_curves = {}
for nm, (yy, pp, ww) in sets.items():
    m = bm(yy, pp, ww); m.update(opchar(yy, pp, ww, THR)); m.update(set=nm, stage="원본"); rows.append(m)
    if nm.startswith("NHANES"):
        pr = crossfit_recal(yy, pp, ww); thr2 = thr_at_sens(yy, pr, ww)
        m2 = bm(yy, pr, ww); m2.update(opchar(yy, pr, ww, thr2)); m2.update(set=nm, stage="재보정(교차적합)"); rows.append(m2)
        recal_curves[nm] = pr

# #5 전이검증: pre로 재보정 적합 → post에 적용
yp, ppre, wpre = sets["NHANES pre"]; ypo, ppo, wpo = sets["NHANES post"]
lr_pre = LogisticRegression(penalty=None, solver="lbfgs").fit(logit(ppre).reshape(-1, 1), yp, sample_weight=wpre)
ppo_tr = lr_pre.predict_proba(logit(ppo).reshape(-1, 1))[:, 1]
thr3 = thr_at_sens(ypo, ppo_tr, wpo)
m3 = bm(ypo, ppo_tr, wpo); m3.update(opchar(ypo, ppo_tr, wpo, thr3)); m3.update(set="NHANES post", stage="재보정(pre적합→post전이)"); rows.append(m3)

res = pd.DataFrame(rows)[["set", "stage", "n", "obs", "pred", "AUROC", "AUPRC", "Brier", "cal_slope", "citl", "sens", "spec", "ppv", "npv"]]
res.to_csv(os.path.join(RES, "external_metrics.csv"), index=False)
qc("\n" + res.round(4).to_string(index=False))

# #7 보정곡선(원본+재보정)
def cc(y, p, w, b=10):
    d = pd.DataFrame({"y": y, "p": p, "w": w}); d["q"] = pd.qcut(d["p"], b, labels=False, duplicates="drop")
    g = d.groupby("q").apply(lambda t: pd.Series({"pm": np.average(t.p, weights=t.w), "om": np.average(t.y, weights=t.w)}))
    return g["pm"], g["om"]
plt.figure(figsize=(6, 6)); plt.plot([0, .3], [0, .3], "k--", lw=1, label="이상적")
for nm, (yy, pp, ww) in sets.items():
    pm, om = cc(yy, pp, ww); plt.plot(pm, om, "o-", ms=4, label=nm)
for nm, pr in recal_curves.items():
    yy, _, ww = sets[nm]; pm, om = cc(yy, pr, ww); plt.plot(pm, om, "s--", ms=3, alpha=.7, label=f"{nm} 재보정")
plt.xlabel("예측 확률(가중)"); plt.ylabel("관측 유병률(가중)"); plt.title("보정 곡선 (원본·재보정, 가중)")
plt.legend(fontsize=7); plt.xlim(0, .3); plt.ylim(0, .3); plt.tight_layout(); plt.savefig(os.path.join(RES, "figures", "calibration.png"), dpi=130)
open(os.path.join(RES, "qc_external.txt"), "w", encoding="utf-8").write("\n".join(QC))
