# -*- coding: utf-8 -*-
"""73_figures_primary2.py - 표본특성과 그림 7종을 '교정된 주분석'으로 다시 만든다.

그림과 Table S6 은 정정 이전 세대(21/24/38/50 번 스크립트, characteristics_availablecase.csv)에서
만들어져 본문 수치와 세대가 달랐다.  이 스크립트가 유일한 생성 경로가 된다.

출력: results/primary2_characteristics.csv, results/figures/*.png
"""
import os, io, sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = os.path.dirname(os.path.abspath(__file__)); RES = f"{D}/results"; FIG = f"{RES}/figures"
os.makedirs(FIG, exist_ok=True)
CK, CPRE, CPOST = "#2E5B9A", "#c0392b", "#27ae60"

# ── JMIR 그림 규격: 최소 300x225, 최대 1200x900 픽셀, 5MB 이하 ──
# 300dpi 로 저장한 뒤 규격 안으로 축소한다(규격 내 최대 해상도를 얻기 위해).
JMIR_MAX = (1200, 900)
def panel_title(ax, text):
    """패널마다 무엇을 그린 것인지 작은 제목을 단다(예시논문 Figure 3 방식).

    그림 전체 제목은 넣지 않는다. 그건 캡션이 말한다.
    """
    ax.set_title(text, fontsize=9.5, pad=6)


def save_fig(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    from PIL import Image
    im = Image.open(path)
    if im.width > JMIR_MAX[0] or im.height > JMIR_MAX[1]:
        r = min(JMIR_MAX[0] / im.width, JMIR_MAX[1] / im.height)
        im = im.resize((max(1, int(im.width * r)), max(1, int(im.height * r))), Image.LANCZOS)
        im.save(path, optimize=True, dpi=(300, 300))   # 축소하면 dpi 태그가 사라진다 - 다시 넣는다
        im = Image.open(path)
    assert im.width >= 300 and im.height >= 225, f"{path}: 최소 규격 미달 {im.size}"
    assert im.info.get("dpi", (0, 0))[0] >= 299.9, f"{path}: dpi 태그 누락 {im.info.get('dpi')}"



src = io.open(f"{D}/70_primary_corrected.py", encoding="utf8").read()
g = {"__file__": f"{D}/70_primary_corrected.py", "__name__": "__prep__"}
exec(compile(src[: src.index("p_tr = model.predict_proba")], "70_prep", "exec"), g)
K, P, S, FE = g["K"], g["P"], g["S"], g["FE"]; ho = g["ho"]
Who, Wp, Ws, yho, yp, ys = g["Who"], g["Wp"], g["Ws"], g["yho"], g["yp"], g["ys"]
p_ho, p_pre, p_post = g["p_ho"], g["p_pre"], g["p_post"]
KH = K.iloc[ho]
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression

CHAR_K = (K, K["W"].to_numpy(float))
SETS = [("KCHS hold-out", KH, yho, p_ho, Who, CK), ("NHANES pre", P, yp, p_pre, Wp, CPRE), ("NHANES post", S, ys, p_post, Ws, CPOST)]

# ══════════════ 1. 표본 특성 (가용사례, 가중) ══════════════
PCTV = [("Depression PHQ-9 ≥10", "dep"), ("Female", None), ("Current smoker", None), ("Partnered", "partnered"),
        ("Living alone", "living_alone"), ("Employed", "employed"), ("Food insecure", "food_insecure"),
        ("Current drinker", "current_drinker"), ("Meets physical-activity recommendation", "active"),
        ("Hypertension", "htn"), ("Diabetes", "dm"),
        # 학력은 4수준이라 수준별 비율로 싣는다 (표 S4 에 예측변수가 빠져 있다는 지적)
        ("Education: elementary or less", None), ("Education: middle school", None),
        ("Education: high school", None), ("Education: college or more", None)]
CONTV = [("Age (years)", "age"), ("BMI", "bmi"), ("Sleep hours", "sleep_hrs"), ("Self-rated health", "srh"),
         ("Oral health", "oral_health"), ("Household income quintile", "income_q")]
def wmean(d, col, w, pct=False):
    v = pd.to_numeric(d[col], errors="coerce").to_numpy(float); m = ~np.isnan(v)
    if m.sum() == 0: return np.nan, 100.0
    return float(np.average(v[m], weights=w[m])) * (100 if pct else 1), float(100*(1-m.mean()))
def binvec(d, label):
    if label == "Female":
        sx = d["sex"].astype("string")
        return sx.map({"F": 1.0, "M": 0.0, "female": 1.0, "male": 0.0}).astype(float)
    if label == "Current smoker":
        sm = d["smoking"].astype("string")
        return (sm == "current").astype(float).where(sm.notna())
    if label.startswith("Education: "):
        lv = {"elementary or less": "elem_low", "middle school": "middle",
              "high school": "high", "college or more": "college"}[label.split(": ", 1)[1]]
        ed = d["edu"].astype("string")
        return (ed == lv).astype(float).where(ed.notna())
    return None
rows = []
for lab, col in PCTV:
    vals = {}
    for nm, d, _y, _p, w, _c in SETS:
        if nm == "KCHS hold-out": d, w = CHAR_K
        s = binvec(d, lab) if col is None else pd.to_numeric(d[col], errors="coerce")
        v = s.to_numpy(float); m = ~np.isnan(v)
        # 전량 결측(예: NHANES 사전의 living alone)이면 비율은 정의되지 않는다
        vals[nm] = ((np.nan if m.sum() == 0 else round(float(np.average(v[m], weights=w[m]))*100, 6)),
                    round(float(100*(1-m.mean())), 6))
    rows.append(dict(characteristic=lab, KCHS=vals["KCHS hold-out"][0], **{"KCHS missing%": vals["KCHS hold-out"][1]},
                     **{"NHANES pre": vals["NHANES pre"][0], "NHANES pre missing%": vals["NHANES pre"][1],
                        "NHANES post": vals["NHANES post"][0], "NHANES post missing%": vals["NHANES post"][1]}))
for lab, col in CONTV:
    vals = {nm: (wmean(*CHAR_K[:1], col, CHAR_K[1]) if nm == "KCHS hold-out" else wmean(d, col, w)) for nm, d, _y, _p, w, _c in SETS}
    rows.append(dict(characteristic=lab, KCHS=round(vals["KCHS hold-out"][0], 6), **{"KCHS missing%": round(vals["KCHS hold-out"][1], 6)},
                     **{"NHANES pre": round(vals["NHANES pre"][0], 6), "NHANES pre missing%": round(vals["NHANES pre"][1], 6),
                        "NHANES post": round(vals["NHANES post"][0], 6), "NHANES post missing%": round(vals["NHANES post"][1], 6)}))
CH = pd.DataFrame(rows)
CH.to_csv(f"{RES}/primary2_characteristics.csv", index=False, encoding="utf-8-sig")
print("[1] 표본특성 재계산", flush=True)

# ══════════════ Figure 1. 표본 특성 ══════════════
_g = lambda k: (float(CH.set_index("characteristic").loc[k, "KCHS"]), float(CH.set_index("characteristic").loc[k, "NHANES pre"]), float(CH.set_index("characteristic").loc[k, "NHANES post"]))
pct = [("Depression (PHQ-9≥10)",)+_g("Depression PHQ-9 ≥10"), ("Female",)+_g("Female"), ("Current smoker",)+_g("Current smoker"),
       ("Partnered",)+_g("Partnered"), ("Employed",)+_g("Employed"), ("Food insecure",)+_g("Food insecure"),
       ("Hypertension",)+_g("Hypertension"), ("Diabetes",)+_g("Diabetes")]
cont = [("Age (years)",)+_g("Age (years)"), ("BMI",)+_g("BMI"), ("Sleep (hours)",)+_g("Sleep hours"),
        ("Self-rated health\n(1 best-5 worst)",)+_g("Self-rated health"), ("Oral health\n(1 best-5 worst)",)+_g("Oral health")]
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [1, 0.8]})
def grouped(ax, data, title, xlabel):
    y = np.arange(len(data)); h = 0.26
    ax.barh(y+h, [d[1] for d in data], h, color=CK, label="KCHS")
    ax.barh(y,   [d[2] for d in data], h, color=CPRE, label="NHANES pre")
    ax.barh(y-h, [d[3] for d in data], h, color=CPOST, label="NHANES post")
    ax.set_yticks(y); ax.set_yticklabels([d[0] for d in data], fontsize=8.5); ax.invert_yaxis()
    panel_title(ax, title); ax.set_xlabel(xlabel, fontsize=9)
    ax.grid(axis="x", alpha=0.25); ax.tick_params(labelsize=8)
grouped(axes[0], pct, "Prevalence and proportion", "Weighted %")
grouped(axes[1], cont, "Means", "Weighted mean")
axes[0].legend(fontsize=8.5, loc="lower right", framealpha=0.9)
fig.tight_layout(); save_fig(fig, f"{FIG}/characteristics.png")

# ══════════════ Figure 2. 보정곡선 ══════════════
def logit(p): p = np.clip(p, 1e-6, 1-1e-6); return np.log(p/(1-p))
def recal_is(yv, p, w, groups=None):
    """재보정은 PSU 단위로 교차적합한다(표 2 와 같은 기준, 코덱스 지적 11·13)."""
    lp = logit(p); oof = np.zeros_like(p)
    sp = (StratifiedGroupKFold(5, shuffle=True, random_state=g["SEED"]).split(lp, yv, groups=groups)
          if groups is not None else
          StratifiedKFold(5, shuffle=True, random_state=g["SEED"]).split(lp, yv))
    for a, b in sp:
        c = LogisticRegression(penalty=None, solver="lbfgs").fit(lp[a].reshape(-1, 1), yv[a], sample_weight=w[a])
        oof[b] = c.predict_proba(lp[b].reshape(-1, 1))[:, 1]
    return oof

def psu_groups(d):
    return (d["strata"].astype(str) + "|" + d["psu"].astype(str)).values
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
for ax, (title, use_recal) in zip(axes, [("As transported", False),
                                         ("After intercept and slope recalibration", True)]):
    for nm, d, yv, p, w, c in SETS:
        q = recal_is(yv, p, w, groups=psu_groups(d)) if use_recal else p
        edges = np.quantile(q, np.linspace(0, 1, 11)); edges[0] -= 1e-9
        xs, os_ = [], []
        for i in range(10):
            m = (q > edges[i]) & (q <= edges[i+1])
            if m.sum() < 5: continue
            xs.append(np.average(q[m], weights=w[m])); os_.append(np.average(yv[m], weights=w[m]))
        ax.plot(xs, os_, "o-", color=c, ms=4, lw=1.4, label=nm)
    lim = 0.45
    ax.plot([0, lim], [0, lim], ls="--", color="#888", lw=0.9)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.grid(alpha=0.25)
    panel_title(ax, title); ax.set_xlabel("Predicted probability", fontsize=9)
axes[0].set_ylabel("Observed proportion", fontsize=9); axes[0].legend(fontsize=8.5, framealpha=0.9)
fig.tight_layout(); save_fig(fig, f"{FIG}/calibration.png")

# ══════════════ Figure 3. ROC ══════════════
fig, ax = plt.subplots(figsize=(6.8, 5.0))
for nm, d, yv, p, w, c in SETS:
    fpr, tpr, _ = roc_curve(yv, p, sample_weight=w)
    ax.plot(fpr, tpr, color=c, lw=1.6, label=f"{nm} (AUROC {roc_auc_score(yv,p,sample_weight=w):.3f})")
ax.plot([0, 1], [0, 1], ls="--", color="#888", lw=0.9)
ax.set_xlabel("1 - specificity", fontsize=9.5); ax.set_ylabel("Sensitivity", fontsize=9.5)
ax.grid(alpha=0.25)
ax.legend(fontsize=8.5, loc="lower right", framealpha=0.9)
fig.tight_layout(); save_fig(fig, f"{FIG}/roc_curves.png")

# ══════════════ Figure 4. 민감도 ══════════════
sn = pd.read_csv(f"{RES}/primary2_sensitivity.csv")
fig, ax = plt.subplots(figsize=(8.4, 4.3))
y = np.arange(len(sn))[::-1]
for i, (_, r) in zip(y, sn.iterrows()):
    ax.plot([r.ext_pre, r.ext_post, r.internal], [i, i, i], color="#ccc", lw=0.8, zorder=1)
ax.scatter(sn.internal, y, color=CK, label="Internal", zorder=3, s=42)
ax.scatter(sn.ext_pre, y, color=CPRE, label="NHANES pre", zorder=3, s=42, marker="s")
ax.scatter(sn.ext_post, y, color=CPOST, label="NHANES post", zorder=3, s=42, marker="^")
ax.axvline(0.80, color="#888", ls=":", lw=0.9, alpha=0.7); ax.text(0.801, 0.15, "0.80 target", fontsize=7.5, color="#666")
ax.set_yticks(y); ax.set_yticklabels(sn["cond"], fontsize=8.5)
ax.set_xlabel("Weighted AUROC", fontsize=9.5); ax.set_xlim(0.70, 0.87); ax.grid(axis="x", alpha=0.25)
ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
fig.tight_layout(); save_fig(fig, f"{FIG}/sensitivity.png")

# ══════════════ Figure 5. SHAP ══════════════
NAME = {"srh": "Self-rated health", "age": "Age", "sex": "Sex", "sleep_hrs": "Sleep duration", "oral_health": "Oral health",
        "partnered": "Partnership", "smoking": "Smoking", "employed": "Employment", "bmi": "BMI",
        "food_insecure": "Food insecurity", "income_q": "Income quintile", "active": "Physical activity", "edu": "Education",
        "living_alone": "Living alone", "current_drinker": "Current drinking", "htn": "Hypertension", "dm": "Diabetes"}
sh = pd.read_csv(f"{RES}/primary2_shap.csv"); sh.columns = ["var", "val"]; sh = sh.sort_values("val")
fig, ax = plt.subplots(figsize=(8.2, 4.0))
ax.barh([NAME.get(v, v) for v in sh["var"]], sh.val, color=CK)
ax.set_xlabel("Mean |SHAP value|", fontsize=9.5); ax.grid(axis="x", alpha=0.25); ax.tick_params(labelsize=8.5)
fig.tight_layout(); save_fig(fig, f"{FIG}/shap_importance.png")

# ══════════════ Figure 6. 하위군 포레스트 ══════════════
fa = pd.read_csv(f"{RES}/primary2_fairness.csv")
COL = {"KCHS hold-out": CK, "NHANES pre": CPRE, "NHANES post": CPOST}
fa = fa.sort_values(["cohort", "AUROC"])
fig, ax = plt.subplots(figsize=(8.0, 0.185*len(fa)+1.6))
y = np.arange(len(fa))[::-1]
for i, (_, r) in zip(y, fa.iterrows()):
    ax.plot([r.CI_lo, r.CI_hi], [i, i], color=COL[r.cohort], lw=1.5, alpha=0.85)
    ax.plot(r.AUROC, i, "o", color=COL[r.cohort], ms=4.5)
for nm, v in [("KCHS hold-out", g["yho"]), ]:
    pass
for nm, c in COL.items():
    ax.axvline({"KCHS hold-out": roc_auc_score(yho, p_ho, sample_weight=Who),
                "NHANES pre": roc_auc_score(yp, p_pre, sample_weight=Wp),
                "NHANES post": roc_auc_score(ys, p_post, sample_weight=Ws)}[nm], color=c, ls=":", lw=0.9, alpha=0.6)
ax.set_yticks(y); ax.set_yticklabels([f"{r.cohort} - {str(r.subgroup).split(': ')[-1]}" for _, r in fa.iterrows()], fontsize=6.4)
ax.set_xlabel("Weighted AUROC (95% CI)", fontsize=9.5); ax.grid(axis="x", alpha=0.25)
fig.tight_layout(); save_fig(fig, f"{FIG}/subgroup_forest.png")

# ══════════════ Figure 7. 결정곡선 ══════════════
fig, axes = plt.subplots(1, 3, figsize=(13, 6.6), sharey=True)
for ax, (nm, d, yv, p, w, c) in zip(axes, SETS):
    # 내부 hold-out 은 재보정하지 않는다(자기 라벨 재보정은 낙관적, 표 S3·70 과 같은 기준)
    pr = p if nm.startswith("KCHS") else recal_is(yv, p, w, groups=psu_groups(d))
    W = w.sum(); prev = w[yv == 1].sum()/W
    ts = np.linspace(0.02, 0.30, 40)
    nb = [(w[(yv == 1) & (pr >= t)].sum() - w[(yv == 0) & (pr >= t)].sum()*t/(1-t))/W for t in ts]
    ax.plot(ts, nb, color=c, lw=1.8,
            label=("Model (original)" if nm.startswith("KCHS") else "Model (recalibrated)"))
    ax.plot(ts, [prev - (1-prev)*t/(1-t) for t in ts], color="#888", lw=1.0, ls="--", label="Screen all")
    ax.axhline(0, color="#444", lw=0.9, ls=":", label="Screen none")
    panel_title(ax, nm); ax.set_xlabel("Threshold probability", fontsize=9)
    ax.grid(alpha=0.25); ax.set_ylim(-0.05, max(0.10, prev*1.1))
    # 범례는 패널마다 그린다 - 내부와 외부의 모형 곡선이 서로 다른 것이라서
    ax.legend(fontsize=8.5, framealpha=0.9)
axes[0].set_ylabel("Net benefit", fontsize=9)
fig.tight_layout(); save_fig(fig, f"{FIG}/dca.png")

print("[2] 그림 7종 재생성 완료:", ", ".join(sorted(os.listdir(FIG))), flush=True)
