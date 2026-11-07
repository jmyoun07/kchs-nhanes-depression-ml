# -*- coding: utf-8 -*-
"""17변수 모델 — 그림·표 생성 (SHAP·공정성·DCA·기저특성·보정곡선)."""
import os, sys, warnings, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
ROOT=r"C:/Users/rudtn/Desktop/claude code/지역사회건강"; D=f"{ROOT}/17var_model"
RES=f"{D}/results"; FIG=f"{D}/results/figures"; os.makedirs(FIG,exist_ok=True)
kchs=pd.read_parquet(f"{D}/data/ext_kchs.parquet"); pre=pd.read_parquet(f"{D}/data/ext_pre.parquet"); post=pd.read_parquet(f"{D}/data/ext_post.parquet")
NUM=["age","sleep_hrs","bmi","income_q","srh","oral_health"]
BIN=["partnered","living_alone","current_drinker","active","htn","dm","employed","food_insecure"]
CAT=["sex","edu","smoking"]; FE=NUM+BIN+CAT
y=kchs["dep"].astype(int).values
LGB=dict(num_leaves=31,min_child_samples=1000,learning_rate=0.01,n_estimators=900,reg_alpha=1.0,reg_lambda=20.0,subsample=0.6,colsample_bytree=0.6,random_state=42,n_jobs=-1,verbose=-1)
ct=ColumnTransformer([("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),NUM),
    ("bin","passthrough",BIN),("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore",drop="first"))]),CAT)])
tr,ho=train_test_split(np.arange(len(kchs)),train_size=500000,test_size=100000,stratify=y,random_state=42)
model=Pipeline([("pre",ct),("clf",lgb.LGBMClassifier(**LGB))]).fit(kchs.iloc[tr][FE],y[tr])
def P(d): return model.predict_proba(d[FE])[:,1]
def wauc(d,p=None):
    if p is None: p=P(d)
    return roc_auc_score(d["dep"].astype(int).values,p,sample_weight=d["W"].to_numpy(float))

# ── Table 3: 기저특성 (가중) ──
print("[1] Table 3 기저특성",flush=True)
def wmean(d,c):
    v=pd.to_numeric(d[c],errors="coerce"); W=d["W"].to_numpy(float); m=v.notna()
    return np.average(v[m],weights=W[m])
def wprop(d,cond):
    W=d["W"].to_numpy(float); return np.average(cond.astype(float),weights=W)
rows=[]
for nm,d in [("KCHS",kchs),("NHANES pre",pre),("NHANES post",post)]:
    W=d["W"].to_numpy(float)
    rows.append(dict(특성="N(비가중)",값={"KCHS":f"{len(d):,}","NHANES pre":f"{len(d):,}","NHANES post":f"{len(d):,}"}[nm]))
    row={"set":nm,"N":len(d),"우울%":wprop(d,d["dep"].astype(int)==1)*100,"연령":wmean(d,"age"),
         "여성%":wprop(d,d["sex"]=="F")*100,"현재흡연%":wprop(d,d["smoking"]=="current")*100,
         "유배우%":wprop(d,pd.to_numeric(d["partnered"],errors="coerce")==1)*100,"BMI":wmean(d,"bmi"),
         "수면":wmean(d,"sleep_hrs"),"자가건강":wmean(d,"srh"),"구강건강":wmean(d,"oral_health"),
         "고용%":wprop(d,pd.to_numeric(d["employed"],errors="coerce")==1)*100,
         "식품불안정%":wprop(d,pd.to_numeric(d["food_insecure"],errors="coerce")==1)*100,
         "고혈압%":wprop(d,pd.to_numeric(d["htn"],errors="coerce")==1)*100,"당뇨%":wprop(d,pd.to_numeric(d["dm"],errors="coerce")==1)*100}
    rows.append(row)
t3=pd.DataFrame([r for r in rows if "set" in r]).round(2); t3.to_csv(f"{RES}/table3_characteristics.csv",index=False,encoding="utf-8-sig")
print(t3.to_string(index=False),flush=True)

# ── SHAP 중요도 ──
print("\n[2] SHAP 중요도",flush=True)
try:
    import shap
    samp=np.random.RandomState(42).choice(ho,20000,replace=False)
    Xs=ct.transform(kchs.iloc[samp][FE])
    booster=model.named_steps["clf"]
    expl=shap.TreeExplainer(booster); sv=expl.shap_values(Xs)
    if isinstance(sv,list): sv=sv[1]
    featnames=NUM+BIN+list(model.named_steps["pre"].named_transformers_["cat"].named_steps["o"].get_feature_names_out(CAT))
    msv=np.abs(sv).mean(0)
    imp={}
    for i,fn in enumerate(featnames):
        base=fn.split("_")[0] if fn.startswith(("sex","edu","smoking")) else fn
        # map onehot back to CAT var
        orig=next((c for c in CAT if fn.startswith(c+"_")), fn)
        imp[orig]=imp.get(orig,0)+msv[i]
    imp=pd.Series(imp).sort_values(ascending=False)
    imp.to_csv(f"{RES}/shap_importance.csv",encoding="utf-8-sig")
    print(imp.round(4).to_string(),flush=True)
    plt.figure(figsize=(6,5)); imp.iloc[::-1].plot.barh(color="#2E5B9A"); plt.xlabel("mean |SHAP|"); plt.title("SHAP importance (17-var, KCHS)"); plt.tight_layout(); plt.savefig(f"{FIG}/shap_importance.png",dpi=120); plt.close()
    print("  → figures/shap_importance.png",flush=True)
except Exception as e:
    print("  SHAP 실패, LightGBM gain으로 대체:",str(e)[:60],flush=True)
    booster=model.named_steps["clf"]; featnames=NUM+BIN+list(model.named_steps["pre"].named_transformers_["cat"].named_steps["o"].get_feature_names_out(CAT))
    gains=booster.booster_.feature_importance(importance_type="gain")
    imp={}
    for fn,g in zip(featnames,gains):
        orig=next((c for c in CAT if fn.startswith(c+"_")), fn); imp[orig]=imp.get(orig,0)+g
    imp=pd.Series(imp).sort_values(ascending=False); imp=imp/imp.sum()
    imp.to_csv(f"{RES}/shap_importance.csv",encoding="utf-8-sig"); print(imp.round(4).to_string(),flush=True)

# ── 공정성 (가중 하위군 AUROC) ──
print("\n[3] 공정성 하위군 AUROC",flush=True)
frows=[]
pk=P(kchs.iloc[ho]); dho=kchs.iloc[ho].copy(); dho["_p"]=pk
def subauc(d,mask,label):
    dd=d[mask]
    if dd["dep"].nunique()<2 or len(dd)<200: return
    a=roc_auc_score(dd["dep"].astype(int),dd["_p"],sample_weight=dd["W"].to_numpy(float))
    frows.append(dict(축=label,n=len(dd),AUROC=round(a,3))); print(f"   {label:<16} n={len(dd):>6} AUROC={a:.3f}",flush=True)
subauc(dho,dho["sex"]=="F","성별:여"); subauc(dho,dho["sex"]=="M","성별:남")
age=pd.to_numeric(dho["age"]); subauc(dho,age<40,"연령:20-39"); subauc(dho,(age>=40)&(age<60),"연령:40-59"); subauc(dho,age>=60,"연령:60+")
for e in dho["edu"].dropna().unique(): subauc(dho,dho["edu"]==e,f"교육:{e}")
# NHANES 인종
prp=pre.copy(); prp["_p"]=P(pre)
for r in sorted(prp["race"].dropna().unique()):
    subauc(prp,prp["race"]==r,f"인종(pre):{int(r)}")
pd.DataFrame(frows).to_csv(f"{RES}/fairness.csv",index=False,encoding="utf-8-sig")

# ── DCA (net benefit, 가중) ──
print("\n[4] DCA 결정곡선",flush=True)
def net_benefit(d,p,ts):
    yv=d["dep"].astype(int).values; W=d["W"].to_numpy(float); N=W.sum(); prev=np.average(yv,weights=W)
    nb=[];nball=[]
    for t in ts:
        pred=p>=t; tp=W[(pred)&(yv==1)].sum(); fp=W[(pred)&(yv==0)].sum()
        nb.append(tp/N-(fp/N)*(t/(1-t))); nball.append(prev-(1-prev)*(t/(1-t)))
    return np.array(nb),np.array(nball)
ts=np.linspace(0.02,0.4,39)
plt.figure(figsize=(7,5))
for nm,d,cl in [("Internal",kchs.iloc[ho],"#2E5B9A"),("NHANES pre","p","#c0392b"),("NHANES post","s","#27ae60")]:
    dd={"Internal":kchs.iloc[ho],"NHANES pre":pre,"NHANES post":post}[nm]
    nb,nball=net_benefit(dd,P(dd),ts)
    plt.plot(ts,nb,label=nm,color=cl)
plt.plot(ts,net_benefit(pre,P(pre),ts)[1],"--",color="gray",label="treat-all(pre)",alpha=.5)
plt.axhline(0,color="k",lw=.7); plt.xlabel("Threshold probability"); plt.ylabel("Net benefit"); plt.title("Decision curve (17-var)"); plt.legend(); plt.ylim(-0.02,0.13); plt.tight_layout(); plt.savefig(f"{FIG}/dca.png",dpi=120); plt.close()
print("  → figures/dca.png",flush=True)

# ── 보정곡선 ──
print("\n[5] 보정곡선",flush=True)
plt.figure(figsize=(6,6))
for nm,d,cl in [("Internal",kchs.iloc[ho],"#2E5B9A"),("NHANES pre",pre,"#c0392b"),("NHANES post",post,"#27ae60")]:
    p=P(d); yv=d["dep"].astype(int).values; W=d["W"].to_numpy(float)
    bins=np.quantile(p,np.linspace(0,1,11)); bins[-1]+=1e-6; idx=np.digitize(p,bins[1:-1])
    xs=[];ys=[]
    for b in range(10):
        m=idx==b
        if m.sum()>10: xs.append(np.average(p[m],weights=W[m])); ys.append(np.average(yv[m],weights=W[m]))
    plt.plot(xs,ys,"o-",label=nm,color=cl,ms=4)
plt.plot([0,0.5],[0,0.5],"k--",alpha=.5); plt.xlabel("Predicted"); plt.ylabel("Observed"); plt.title("Calibration (17-var, original)"); plt.legend(); plt.tight_layout(); plt.savefig(f"{FIG}/calibration.png",dpi=120); plt.close()
print("  → figures/calibration.png\n완료.",flush=True)
