# -*- coding: utf-8 -*-
"""
06_decision_curve.py (교차검증 수정본) — DCA는 원본(비재보정) 확률+복합표본 가중.
외부셋 AUROC CI는 PSU-군집 부트스트랩(설계 반영). 내부는 가중 iid 부트스트랩.
결과: results/figures/dca.png, results/bootstrap_ci.csv
"""
import os, warnings, numpy as np, pandas as pd, joblib
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
ROOT = r"C:/Users/rudtn/Desktop/claude code/지역사회건강"
PROC = os.path.join(ROOT, "data/processed"); MODELS = os.path.join(ROOT, "models"); RES = os.path.join(ROOT, "results")
SEED = 42; rng = np.random.RandomState(SEED)
prim = joblib.load(os.path.join(MODELS, "primary.joblib")); FEATURES = prim["features"]
model = joblib.load(os.path.join(MODELS, f"{prim['primary']}.joblib"))
load = lambda n: pd.read_parquet(os.path.join(PROC, f"{n}.parquet"))

dev = load("kchs_dev"); y = dev["dep"].astype(int).values
_, Xho, _, yho, _, Who, _, Sho, _, Uho = train_test_split(
    dev[FEATURES], y, dev["W"].to_numpy(dtype=float), dev["strata"].to_numpy().astype(str), dev["psu"].to_numpy().astype(str),
    test_size=0.2, stratify=y, random_state=SEED)
sets = {"Internal": (np.asarray(yho, int), model.predict_proba(Xho)[:, 1], np.asarray(Who, float), np.asarray(Sho, str), np.asarray(Uho, str))}
for nm, key in [("NHANES pre", "nhanes_pre"), ("NHANES post", "nhanes_post")]:
    d = load(key)
    sets[nm] = (d["dep"].to_numpy(dtype=float).astype(int), model.predict_proba(d[FEATURES])[:, 1],
                d["W"].to_numpy(dtype=float), d["strata"].to_numpy().astype(str), d["psu"].to_numpy().astype(str))  # 원본확률

# ===== DCA (가중, 원본확률) =====
pts = np.linspace(0.02, 0.30, 29)
plt.figure(figsize=(7, 5))
for nm, (yy, pp, ww, ss, uu) in sets.items():
    n = ww.sum(); prev = np.average(yy, weights=ww); nb = []
    for pt in pts:
        yp = pp >= pt
        tp = np.sum(ww * (yp & (yy == 1))); fp = np.sum(ww * (yp & (yy == 0)))
        nb.append(tp / n - fp / n * (pt / (1 - pt)))
    line, = plt.plot(pts, nb, "-", label=f"모델: {nm}")
    ta = [prev - (1 - prev) * (pt / (1 - pt)) for pt in pts]  # #8 코호트별 treat-all
    plt.plot(pts, ta, ":", color=line.get_color(), lw=1, alpha=.6, label=f"전체스크리닝: {nm}")
plt.axhline(0, color="gray", lw=0.8, label="무스크리닝")
plt.xlabel("임계확률 pt"); plt.ylabel("순편익(가중)"); plt.title("의사결정곡선분석 (원본확률·가중)")
plt.legend(fontsize=8); plt.ylim(-0.02, 0.13); plt.tight_layout(); plt.savefig(os.path.join(RES, "figures", "dca.png"), dpi=130)

# ===== 부트스트랩 CI =====
def cluster_boot(y, p, w, s, u, B=400):
    # #9/#10 설계기반: 층 내 PSU 복원추출. lonely-PSU(단일PSU 층)는 결정적 유지(분산 0 기여) → 개수 보고.
    pos = np.arange(len(y)); dfg = pd.DataFrame({"s": s, "u": u, "i": pos})
    groups = [[g["i"].to_numpy() for _, g in sub.groupby("u")] for _, sub in dfg.groupby("s")]
    lonely = sum(1 for gg in groups if len(gg) == 1)
    a = []
    for _ in range(B):
        idx = []
        for psus in groups:
            k = len(psus)
            for j in rng.randint(0, k, k): idx.append(psus[j])
        idx = np.concatenate(idx); yy = y[idx]
        if yy.sum() == 0 or yy.sum() == len(yy): continue
        a.append(roc_auc_score(yy, p[idx], sample_weight=w[idx]))
    lo, hi = np.percentile(a, [2.5, 97.5])
    return lo, hi, lonely, len(groups)

rows = []
for nm, (yy, pp, ww, ss, uu) in sets.items():
    auc = roc_auc_score(yy, pp, sample_weight=ww)
    B = 80 if nm == "Internal" else 400   # 내부는 층 많고 n 커서 B 축소
    lo, hi, lonely, nstr = cluster_boot(yy, pp, ww, ss, uu, B=B)
    rows.append({"set": nm, "AUROC_weighted": round(auc, 4), "CI95_low": round(lo, 4), "CI95_high": round(hi, 4),
                 "method": "PSU군집(설계)", "n_strata": nstr, "lonely_PSU_strata": lonely})
pd.DataFrame(rows).to_csv(os.path.join(RES, "bootstrap_ci.csv"), index=False)
print(pd.DataFrame(rows).to_string(index=False))
