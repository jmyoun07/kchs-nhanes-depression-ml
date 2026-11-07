# -*- coding: utf-8 -*-
"""50_figures_update.py — 심사 지적 반영 그림 갱신.

Figure 6 (하위군): NHANES post 추가 + 95% CI 오차막대 (기존엔 pre만·CI 없음)
Figure 7 (의사결정곡선): 코호트별 treat-all 기준선 (기존엔 pre 것 하나만 그려놓고 본문은 'cohort-specific'이라 서술)
"""
import sys, warnings, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import lightgbm as lgb

D = r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model"; FIG = f"{D}/results/figures"; RES = f"{D}/results"; SEED = 42
K, PRE, POST = "#2E5B9A", "#c0392b", "#27ae60"

# ── Figure 6: 하위군 forest (CI 포함, pre+post) ──
f = pd.read_csv(f"{RES}/review3_fairness_domain.csv")
f["label"] = f.apply(lambda r: f"{'KCHS' if r.cohort.startswith('KCHS') else ('pre' if 'pre' in r.cohort else 'post')}: "
                                f"{str(r.subgroup).split(': ')[-1]}", axis=1)
f = f.iloc[::-1].reset_index(drop=True)
cols = [K if r.cohort.startswith("KCHS") else (PRE if "pre" in r.cohort else POST) for r in f.itertuples()]
fig, ax = plt.subplots(figsize=(8.2, max(5, 0.32*len(f))))
ys = np.arange(len(f))
ax.hlines(ys, f.CI_lo, f.CI_hi, color=cols, lw=1.6, alpha=0.75)
ax.scatter(f.AUROC, ys, color=cols, s=34, zorder=3)
ax.axvline(0.832, color=K, ls=":", lw=0.9, alpha=0.65)
ax.axvline(0.772, color=PRE, ls=":", lw=0.9, alpha=0.65)
ax.axvline(0.769, color=POST, ls=":", lw=0.9, alpha=0.65)
ax.set_yticks(ys); ax.set_yticklabels([f"{r.label} (n={int(r.n):,})" for r in f.itertuples()], fontsize=8)
ax.set_xlabel("Weighted AUROC (95% CI)", fontsize=9.5); ax.set_xlim(0.55, 1.0); ax.grid(axis="x", alpha=0.25)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0],[0],color=K,marker="o",ls="",label="KCHS hold-out"),
                   Line2D([0],[0],color=PRE,marker="o",ls="",label="NHANES pre"),
                   Line2D([0],[0],color=POST,marker="o",ls="",label="NHANES post")],
          fontsize=8, loc="lower right", framealpha=0.9)
ax.set_title("Subgroup discrimination with 95% confidence intervals", fontsize=10.5, fontweight="bold")
fig.tight_layout(); fig.savefig(f"{FIG}/subgroup_forest.png", dpi=140, bbox_inches="tight"); plt.close(fig)
print("→ subgroup_forest.png (CI + post 포함)", flush=True)

# ── Figure 7: DCA, 코호트별 treat-all ──
kchs = pd.read_parquet(f"{D}/data/ext_kchs.parquet"); pre = pd.read_parquet(f"{D}/data/ext_pre.parquet"); post = pd.read_parquet(f"{D}/data/ext_post.parquet")
NUM=["age","sleep_hrs","bmi","income_q","srh","oral_health"]
BIN=["partnered","living_alone","current_drinker","active","htn","dm","employed","food_insecure"]
CAT=["sex","edu","smoking"]; FE=NUM+BIN+CAT
y=kchs["dep"].astype(int).values
LGB=dict(num_leaves=31,min_child_samples=1000,learning_rate=0.01,n_estimators=900,reg_alpha=1.0,
         reg_lambda=20.0,subsample=0.6,colsample_bytree=0.6,random_state=SEED,n_jobs=-1,verbose=-1)
ct=ColumnTransformer([("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),NUM),
                      ("bin","passthrough",BIN),
                      ("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore",drop="first"))]),CAT)])
tr,ho=train_test_split(np.arange(len(kchs)),train_size=500000,test_size=100000,stratify=y,random_state=SEED)
m=Pipeline([("pre",ct),("clf",lgb.LGBMClassifier(**LGB))]).fit(kchs.iloc[tr][FE],y[tr])
ts=np.linspace(0.01,0.30,60)
fig,ax=plt.subplots(figsize=(7.2,5.2))
for nm,d,c in [("Internal",kchs.iloc[ho],K),("NHANES pre",pre,PRE),("NHANES post",post,POST)]:
    p=m.predict_proba(d[FE])[:,1]; yv=d["dep"].astype(int).values; w=d["W"].to_numpy(float); W=w.sum()
    prev=np.average(yv,weights=w)
    nb=[(w[(yv==1)&(p>=t)].sum()/W)-(w[(yv==0)&(p>=t)].sum()/W)*(t/(1-t)) for t in ts]
    nball=[prev-(1-prev)*(t/(1-t)) for t in ts]
    ax.plot(ts,nb,color=c,lw=1.9,label=f"Model — {nm}")
    ax.plot(ts,nball,color=c,lw=1.0,ls="--",alpha=0.55,label=f"Screen all — {nm}")
ax.axhline(0,color="k",lw=0.8,label="Screen none")
ax.set_xlabel("Threshold probability",fontsize=9.5); ax.set_ylabel("Net benefit",fontsize=9.5)
ax.set_ylim(-0.06,0.14); ax.grid(alpha=0.25)
ax.legend(fontsize=7.6,ncol=2,loc="upper right",framealpha=0.9)
ax.set_title("Decision curves with cohort-specific default strategies",fontsize=10.5,fontweight="bold")
fig.tight_layout(); fig.savefig(f"{FIG}/dca.png",dpi=140,bbox_inches="tight"); plt.close(fig)
print("→ dca.png (코호트별 treat-all)", flush=True)
print("완료.", flush=True)
