# -*- coding: utf-8 -*-
"""ROUND G — 전이-강건 선택을 특징공학 공간까지 확장.
선택기준 = KCHS leave-one-year-out(9폴드) 최악연도 가중AUROC (내부만, NHANES 미사용 → leakage-free).
후보: {plain17, +상호작용, +스플라인, 조합} × {기본, 강정규화}. 선택 후 NHANES 1회 + 부트스트랩.
"""
import os, sys, warnings, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
PROC=r"C:/Users/rudtn/Desktop/claude code/지역사회건강/data/processed"; RES=r"C:/Users/rudtn/Desktop/claude code/지역사회건강/results"; SEED=42
kchs=pd.read_parquet(f"{PROC}/ext_kchs.parquet"); pre=pd.read_parquet(f"{PROC}/ext_pre.parquet"); post=pd.read_parquet(f"{PROC}/ext_post.parquet")
y=kchs["dep"].astype(int).values
NUM=["age","sleep_hrs","bmi","income_q","srh","oral_health"]
BIN=["partnered","living_alone","current_drinker","active","htn","dm","employed","food_insecure"]
CAT=["sex","edu","smoking"]
def feng(df):
    d=df.copy(); n=lambda c:pd.to_numeric(d[c],errors="coerce"); az=(n("age")-50)/15
    d["age_x_srh"]=az*(n("srh")-3); d["srh_x_oral"]=(n("srh")-3)*(n("oral_health")-3)
    d["age_x_employed"]=az*n("employed").fillna(0); d["income_x_employed"]=n("income_q").fillna(2)*n("employed").fillna(0)
    d["srh_x_food"]=(n("srh")-3)*n("food_insecure").fillna(0); d["oral_x_employed"]=(n("oral_health")-3)*n("employed").fillna(0)
    d["age_sq"]=az**2; d["bmi_sq"]=((n("bmi")-24)/5)**2; d["sleep_dev"]=(n("sleep_hrs")-7).abs()
    return d
INT=["age_x_srh","srh_x_oral","age_x_employed","income_x_employed","srh_x_food","oral_x_employed"]
SPL=["age_sq","bmi_sq","sleep_dev"]
kc,pc,sc=feng(kchs),feng(pre),feng(post)
def ct(NUMx):
    return ColumnTransformer([("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),NUMx),
        ("bin","passthrough",BIN),("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore",drop="first"))]),CAT)])
def lgbm(**kw):
    p=dict(n_estimators=600,learning_rate=0.03,num_leaves=31,subsample=0.8,colsample_bytree=0.8,random_state=SEED,n_jobs=-1,verbose=-1);p.update(kw)
    return lgb.LGBMClassifier(**p)
STRONG=dict(num_leaves=15,min_child_samples=500,reg_alpha=1.0,reg_lambda=5.0,n_estimators=900,learning_rate=0.02,subsample=0.7,colsample_bytree=0.7)
CANDS={
 "plain17":(NUM,{}), "plain17_reg":(NUM,STRONG),
 "interact":(NUM+INT,{}), "interact_reg":(NUM+INT,STRONG),
 "spline":(NUM+SPL,{}), "int+spline":(NUM+INT+SPL,{}),
 "int+spline_reg":(NUM+INT+SPL,STRONG),
}
years=sorted(kc["year"].unique())
rng=np.random.RandomState(SEED)
# LOYO 선택기준(내부만): 각 후보를 8년 학습(연도당 최대 40k)→남긴 1년 평가, 최악연도·평균 가중AUROC
print(f"{'후보':<16}{'LOYO최악':>10}{'LOYO평균':>10}",flush=True); print("-"*40,flush=True)
loyo={}
for name,(NUMx,kw) in CANDS.items():
    FE=NUMx+BIN+CAT; aucs=[]
    for yr in years:
        trm=kc["year"]!=yr; tem=kc["year"]==yr
        tri=np.where(trm.values)[0]; sub=rng.choice(tri,min(250000,len(tri)),replace=False)
        m=Pipeline([("pre",ct(NUMx)),("clf",lgbm(**kw))]).fit(kc.iloc[sub][FE],y[sub])
        te=np.where(tem.values)[0]
        aucs.append(roc_auc_score(y[te],m.predict_proba(kc.iloc[te][FE])[:,1],sample_weight=kc.iloc[te]["W"].to_numpy(float)))
    loyo[name]=(min(aucs),np.mean(aucs))
    print(f"{name:<16}{min(aucs):>10.4f}{np.mean(aucs):>10.4f}",flush=True)
best=max(loyo,key=lambda k:loyo[k][0])   # 최악연도 최대화 = 전이강건
print(f"\n★ LOYO 최악연도 기준 선택: {best}  (NHANES 미사용)",flush=True)

# 선택 모델을 전체 500k 학습 → NHANES 1회
from sklearn.model_selection import train_test_split
tr,ho=train_test_split(np.arange(len(kc)),train_size=500000,test_size=100000,stratify=y,random_state=SEED)
def evalfinal(name):
    NUMx,kw=CANDS[name]; FE=NUMx+BIN+CAT
    m=Pipeline([("pre",ct(NUMx)),("clf",lgbm(**kw))]).fit(kc.iloc[tr][FE],y[tr])
    def wauc(draw,dext): return roc_auc_score(dext["dep"].astype(int).values,m.predict_proba(draw[FE])[:,1],sample_weight=dext["W"].to_numpy(float))
    i=roc_auc_score(y[ho],m.predict_proba(kc.iloc[ho][FE])[:,1],sample_weight=kc.iloc[ho]["W"].to_numpy(float))
    return i,wauc(pc,pre),wauc(sc,post),m,FE
print("\n[최종 평가 — LOYO 선택 모델을 NHANES에 1회]",flush=True)
I,Pp,Ss,mfin,FE=evalfinal(best)
print(f"   {best}:  내부 {I:.4f}  외부 pre {Pp:.4f}  post {Ss:.4f}",flush=True)
# 참고: 모든 후보의 외부(진단용)
print("\n[참고 — 전 후보 외부 성능(진단용, 선택엔 미사용)]",flush=True)
for name in CANDS:
    i,p,s,_,_=evalfinal(name); print(f"   {name:<16} 내부 {i:.4f} pre {p:.4f} post {s:.4f}",flush=True)
# 부트스트랩
def cb(d,p,B=2000):
    yy=d["dep"].astype(int).values;W=d["W"].to_numpy(float);grp=(d["strata"].astype(str)+"|"+d["psu"].astype(str)).values
    uid=pd.unique(grp);idx={u:np.where(grp==u)[0] for u in uid};r=np.random.RandomState(SEED);a=[]
    for _ in range(B):
        ii=np.concatenate([idx[u] for u in r.choice(uid,len(uid),replace=True)])
        try:a.append(roc_auc_score(yy[ii],p[ii],sample_weight=W[ii]))
        except:pass
    a=np.array(a);return a.mean(),np.percentile(a,2.5),np.percentile(a,97.5),(a>=0.78).mean()
print(f"\n[선택모델 부트스트랩 — P(AUROC≥0.78)]",flush=True)
for nm,d in [("pre",pre),("post",post)]:
    p=mfin.predict_proba((pc if nm=='pre' else sc)[FE])[:,1]
    m,lo,hi,p78=cb(d,p); print(f"   {nm:5}: {m:.4f} [{lo:.4f},{hi:.4f}] P(≥0.78)={p78:.1%}",flush=True)
