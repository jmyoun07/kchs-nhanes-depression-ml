# -*- coding: utf-8 -*-
"""추가 그림 — ROC 곡선(내부/pre/post) + 하위군 AUROC forest plot."""
import os, sys, warnings, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, roc_curve
import lightgbm as lgb
D=r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model"; FIG=f"{D}/results/figures"; SEED=42
kchs=pd.read_parquet(f"{D}/data/ext_kchs.parquet"); pre=pd.read_parquet(f"{D}/data/ext_pre.parquet"); post=pd.read_parquet(f"{D}/data/ext_post.parquet")
NUM=["age","sleep_hrs","bmi","income_q","srh","oral_health"]; BIN=["partnered","living_alone","current_drinker","active","htn","dm","employed","food_insecure"]; CAT=["sex","edu","smoking"]; FE=NUM+BIN+CAT
y=kchs["dep"].astype(int).values
LGB=dict(num_leaves=31,min_child_samples=1000,learning_rate=0.01,n_estimators=900,reg_alpha=1.0,reg_lambda=20.0,subsample=0.6,colsample_bytree=0.6,random_state=SEED,n_jobs=-1,verbose=-1)
ct=ColumnTransformer([("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),NUM),("bin","passthrough",BIN),("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore",drop="first"))]),CAT)])
tr,ho=train_test_split(np.arange(len(kchs)),train_size=500000,test_size=100000,stratify=y,random_state=SEED)
model=Pipeline([("pre",ct),("clf",lgb.LGBMClassifier(**LGB))]).fit(kchs.iloc[tr][FE],y[tr])
def P(d): return model.predict_proba(d[FE])[:,1]

# ── ROC 곡선 ──
plt.figure(figsize=(6,6))
for nm,d,cl in [("Internal (0.832)",kchs.iloc[ho],"#2E5B9A"),("NHANES pre (0.772)",pre,"#c0392b"),("NHANES post (0.769)",post,"#27ae60")]:
    yv=d["dep"].astype(int).values; w=d["W"].to_numpy(float); p=P(d)
    fpr,tpr,_=roc_curve(yv,p,sample_weight=w)
    plt.plot(fpr,tpr,label=nm,color=cl,lw=1.8)
plt.plot([0,1],[0,1],"k--",alpha=.4,lw=.8)
plt.xlabel("1 − Specificity"); plt.ylabel("Sensitivity"); plt.title("ROC — 17-predictor model (weighted)")
plt.legend(loc="lower right",fontsize=9); plt.tight_layout(); plt.savefig(f"{FIG}/roc_curves.png",dpi=130); plt.close()
print("  → figures/roc_curves.png",flush=True)

# ── 하위군 AUROC forest plot (KCHS hold-out + NHANES 인종) ──
dho=kchs.iloc[ho].copy(); dho["_p"]=P(dho); age=pd.to_numeric(dho["age"])
prp=pre.copy(); prp["_p"]=P(pre)
RACE={1:"Mexican American",2:"Other Hispanic",3:"NH White",4:"NH Black",6:"NH Asian",7:"Other/Multi"}
rows=[]
def add(d,mask,label):
    dd=d[mask]
    if dd["dep"].nunique()<2 or len(dd)<150: return
    a=roc_auc_score(dd["dep"].astype(int),dd["_p"],sample_weight=dd["W"].to_numpy(float)); rows.append((label,a,len(dd)))
add(dho,dho["sex"]=="F","KCHS: Female"); add(dho,dho["sex"]=="M","KCHS: Male")
add(dho,age<40,"KCHS: Age 20–39"); add(dho,(age>=40)&(age<60),"KCHS: Age 40–59"); add(dho,age>=60,"KCHS: Age ≥60")
for e,lab in [("college","KCHS: College"),("high","KCHS: High school"),("middle","KCHS: Middle"),("elem_low","KCHS: ≤Elementary")]:
    add(dho,dho["edu"]==e,lab)
for r,lab in RACE.items(): add(prp,prp["race"]==r,f"NHANES pre: {lab}")
rows=rows[::-1]
plt.figure(figsize=(7,max(4,0.35*len(rows))))
ys=np.arange(len(rows)); vals=[r[1] for r in rows]; labs=[f"{r[0]} (n={r[2]:,})" for r in rows]
cols=["#2E5B9A" if r[0].startswith("KCHS") else "#c0392b" for r in rows]
plt.scatter(vals,ys,color=cols,zorder=3)
for i,v in enumerate(vals): plt.plot([0.6,v],[i,i],color="#ccc",zorder=1)
plt.axvline(0.772,color="#c0392b",ls=":",alpha=.6,label="NHANES pre overall 0.772")
plt.axvline(0.832,color="#2E5B9A",ls=":",alpha=.6,label="KCHS internal 0.832")
plt.yticks(ys,labs,fontsize=8); plt.xlabel("Weighted AUROC"); plt.xlim(0.6,0.9)
plt.title("Subgroup discrimination (AUROC)"); plt.legend(fontsize=8,loc="lower right"); plt.tight_layout()
plt.savefig(f"{FIG}/subgroup_forest.png",dpi=130); plt.close()
print("  → figures/subgroup_forest.png",flush=True)
print("완료.")
