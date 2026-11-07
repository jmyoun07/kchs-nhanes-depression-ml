# -*- coding: utf-8 -*-
"""추가 지표 — NHANES SHAP·전이일치도 + ECE + AUPRC 부트스트랩 CI + 하위군 보정."""
import os, sys, warnings, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr
import lightgbm as lgb
D=r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model"; RES=f"{D}/results"; SEED=42
kchs=pd.read_parquet(f"{D}/data/ext_kchs.parquet"); pre=pd.read_parquet(f"{D}/data/ext_pre.parquet"); post=pd.read_parquet(f"{D}/data/ext_post.parquet")
NUM=["age","sleep_hrs","bmi","income_q","srh","oral_health"]; BIN=["partnered","living_alone","current_drinker","active","htn","dm","employed","food_insecure"]; CAT=["sex","edu","smoking"]; FE=NUM+BIN+CAT
y=kchs["dep"].astype(int).values
LGB=dict(num_leaves=31,min_child_samples=1000,learning_rate=0.01,n_estimators=900,reg_alpha=1.0,reg_lambda=20.0,subsample=0.6,colsample_bytree=0.6,random_state=SEED,n_jobs=-1,verbose=-1)
ct=ColumnTransformer([("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),NUM),("bin","passthrough",BIN),("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore",drop="first"))]),CAT)])
tr,ho=train_test_split(np.arange(len(kchs)),train_size=500000,test_size=100000,stratify=y,random_state=SEED)
model=Pipeline([("pre",ct),("clf",lgb.LGBMClassifier(**LGB))]).fit(kchs.iloc[tr][FE],y[tr])
pp=model.named_steps["pre"]; booster=model.named_steps["clf"]
featn=NUM+BIN+list(pp.named_transformers_["cat"].named_steps["o"].get_feature_names_out(CAT))
def P(d): return model.predict_proba(d[FE])[:,1]
def W(d): return d["W"].to_numpy(float)

# ── SHAP 교차 전이 일치도 ──
print("[1] SHAP 교차 전이 일치도",flush=True)
import shap
def shap_imp(d,n=6000):
    idx=np.random.RandomState(SEED).choice(len(d),min(n,len(d)),replace=False)
    X=pp.transform(d.iloc[idx][FE]); sv=shap.TreeExplainer(booster).shap_values(X)
    if isinstance(sv,list): sv=sv[1]
    msv=np.abs(sv).mean(0); imp={}
    for fn,v in zip(featn,msv):
        orig=next((c for c in CAT if fn.startswith(c+"_")),fn); imp[orig]=imp.get(orig,0)+v
    return pd.Series(imp)
ik=shap_imp(kchs.iloc[ho]); ipre=shap_imp(pre); ipost=shap_imp(post)
allv=FE
rk=ik.reindex(allv).rank(); rpre=ipre.reindex(allv).rank(); rpost=ipost.reindex(allv).rank()
rho_pre=spearmanr(rk,rpre).correlation; rho_post=spearmanr(rk,rpost).correlation
print(f"   SHAP 순위 Spearman: KCHS↔pre={rho_pre:.3f}, KCHS↔post={rho_post:.3f}",flush=True)
pd.DataFrame({"KCHS":ik,"pre":ipre,"post":ipost}).to_csv(f"{RES}/shap_cross.csv",encoding="utf-8-sig")

# ── ECE ──
print("\n[2] ECE (weighted, 10 bins)",flush=True)
def ece(d,p,bins=10):
    yv=d["dep"].astype(int).values; w=W(d); edges=np.quantile(p,np.linspace(0,1,bins+1)); edges[-1]+=1e-9
    idx=np.digitize(p,edges[1:-1]); e=0
    for b in range(bins):
        m=idx==b
        if w[m].sum()>0:
            conf=np.average(p[m],weights=w[m]); acc=np.average(yv[m],weights=w[m]); e+=w[m].sum()/w.sum()*abs(conf-acc)
    return e
for nm,d in [("Internal",kchs.iloc[ho]),("pre",pre),("post",post)]:
    print(f"   {nm:<10} ECE={ece(d,P(d)):.4f}",flush=True)

# ── AUPRC 부트스트랩 CI ──
print("\n[3] AUPRC 부트스트랩 CI (PSU군집)",flush=True)
def boot_ci(d,fn,B=1000):
    yy=d["dep"].astype(int).values; w=W(d); p=P(d); grp=(d["strata"].astype(str)+"|"+d["psu"].astype(str)).values
    uid=pd.unique(grp); idx={u:np.where(grp==u)[0] for u in uid}; r=np.random.RandomState(SEED); a=[]
    for _ in range(B):
        ii=np.concatenate([idx[u] for u in r.choice(uid,len(uid),replace=True)])
        try:a.append(fn(yy[ii],p[ii],w[ii]))
        except:pass
    a=np.array(a); return np.mean(a),np.percentile(a,2.5),np.percentile(a,97.5)
for nm,d in [("Internal",kchs.iloc[ho]),("pre",pre),("post",post)]:
    m,lo,hi=boot_ci(d,lambda y,p,w:average_precision_score(y,p,sample_weight=w))
    print(f"   {nm:<10} AUPRC={m:.3f} [{lo:.3f},{hi:.3f}]",flush=True)

# ── 하위군 보정 slope (KCHS) ──
print("\n[4] 하위군 보정 slope (KCHS hold-out)",flush=True)
def calslope(yv,p,w):
    p=np.clip(p,1e-6,1-1e-6); lp=np.log(p/(1-p)).reshape(-1,1)
    return LogisticRegression(penalty=None).fit(lp,yv,sample_weight=w).coef_[0,0]
dho=kchs.iloc[ho].copy(); dho["_p"]=P(dho)
for lab,mask in [("여성",dho["sex"]=="F"),("남성",dho["sex"]=="M"),("20-39",pd.to_numeric(dho["age"])<40),("60+",pd.to_numeric(dho["age"])>=60)]:
    dd=dho[mask]
    if dd["dep"].nunique()>1:
        print(f"   {lab:<8} slope={calslope(dd['dep'].astype(int).values,dd['_p'].values,dd['W'].to_numpy(float)):.3f}",flush=True)
print("\n완료. 저장: shap_cross.csv")
