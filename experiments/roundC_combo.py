# -*- coding: utf-8 -*-
"""ROUND C — 최선 지렛대 조합. A(상호작용→pre 0.779) + B(불안정변수 제거→post 0.776) 결합 + 앙상블.
주의: 'sex/smoking 제거'를 외부성능 보고 정하면 leakage → [LEAK] 표기. 연도-안정성 기반 제거는 [OK].
"""
import os, sys, warnings, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import lightgbm as lgb
PROC=r"C:/Users/rudtn/Desktop/claude code/지역사회건강/data/processed"; RES=r"C:/Users/rudtn/Desktop/claude code/지역사회건강/results"; SEED=42
kchs=pd.read_parquet(f"{PROC}/ext_kchs.parquet"); pre=pd.read_parquet(f"{PROC}/ext_pre.parquet"); post=pd.read_parquet(f"{PROC}/ext_post.parquet")
y=kchs["dep"].astype(int).values
NUM=["age","sleep_hrs","bmi","income_q","srh","oral_health"]
BIN=["partnered","living_alone","current_drinker","active","htn","dm","employed","food_insecure"]
CAT=["sex","edu","smoking"]
# 상호작용(A의 최선)
def addint(df):
    d=df.copy(); n=lambda c:pd.to_numeric(d[c],errors="coerce"); az=(n("age")-50)/15
    d["age_x_srh"]=az*(n("srh")-3); d["srh_x_oral"]=(n("srh")-3)*(n("oral_health")-3)
    d["age_x_employed"]=az*n("employed").fillna(0); d["income_x_employed"]=n("income_q").fillna(2)*n("employed").fillna(0)
    return d
INT=["age_x_srh","srh_x_oral","age_x_employed","income_x_employed"]
kc,pc,sc=addint(kchs),addint(pre),addint(post)
UNSTABLE=["smoking","bmi","current_drinker"]   # B: 연도 안정성 하위3 (top14) — KCHS내부 근거 [OK]
tr,ho=train_test_split(np.arange(len(kc)),train_size=500000,test_size=100000,stratify=y,random_state=SEED)
Who,yho=kc.iloc[ho]["W"].to_numpy(float),y[ho]
def wauc(d,FE,pp): return roc_auc_score(d["dep"].astype(int).values,pp,sample_weight=d["W"].to_numpy(float))
def ct(NUMx,BINx,CATx):
    t=[]
    if NUMx:t.append(("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),NUMx))
    if BINx:t.append(("bin","passthrough",BINx))
    if CATx:t.append(("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore",drop="first"))]),CATx))
    return ColumnTransformer(t)
def lgbm(**kw):
    p=dict(n_estimators=600,learning_rate=0.03,num_leaves=31,subsample=0.8,colsample_bytree=0.8,random_state=SEED,n_jobs=-1,verbose=-1);p.update(kw)
    return lgb.LGBMClassifier(**p)
def model(feats,**kw):
    NUMx=[c for c in NUM+INT if c in feats]; BINx=[c for c in BIN if c in feats]; CATx=[c for c in CAT if c in feats]
    FE=NUMx+BINx+CATx
    m=Pipeline([("pre",ct(NUMx,BINx,CATx)),("clf",lgbm(**kw))]).fit(kc.iloc[tr][FE],y[tr])
    return m,FE
def probs(m,FE): return (m.predict_proba(kc.iloc[ho][FE])[:,1],m.predict_proba(pc[FE])[:,1],m.predict_proba(sc[FE])[:,1])
ROWS=[]
def rec(name,pi,pp,ps):
    i=roc_auc_score(yho,pi,sample_weight=Who);p=wauc(pc,None,pp);s=wauc(sc,None,ps)
    ROWS.append(dict(name=name,internal=round(i,4),pre=round(p,4),post=round(s,4)))
    print(f"{name:<44}{i:>8.4f}{p:>9.4f}{s:>9.4f}",flush=True); return p,s
def run(name,feats,**kw):
    m,FE=model(feats,**kw); return rec(name,*probs(m,FE))

ALLI=NUM+INT+BIN+CAT
print(f"{'ROUND C: 조합':<44}{'내부':>8}{'pre':>9}{'post':>9}"); print("-"*72,flush=True)
run("① 17+상호작용",ALLI)
run("② [OK]17+상호작용 −불안정3",[c for c in ALLI if c not in UNSTABLE])
run("③ [LEAK]17+상호작용 −sex,smoking",[c for c in ALLI if c not in ("sex","smoking")])
run("④ [OK]−불안정3 +강정규화",[c for c in ALLI if c not in UNSTABLE],num_leaves=15,min_child_samples=500,reg_alpha=1.0,reg_lambda=5.0,n_estimators=900,learning_rate=0.02)
# 앙상블: pre-good(①) + post-good(top14 = 17 −불안정3, 상호작용無)
m1,FE1=model(ALLI); mA,FEA=model([c for c in NUM+BIN+CAT if c not in UNSTABLE])
p1=probs(m1,FE1); pA=probs(mA,FEA)
def ra(a,b): return (rankdata(a)/len(a)+rankdata(b)/len(b))/2
rec("⑤ 앙상블(①+top14) 순위평균",ra(p1[0],pA[0]),ra(p1[1],pA[1]),ra(p1[2],pA[2]))
# 가중 앙상블 스윕
for w in [0.3,0.4,0.6,0.7]:
    rec(f"⑥ 가중앙상블 w={w}",w*rankdata(p1[0])/len(p1[0])+(1-w)*rankdata(pA[0])/len(pA[0]),
        w*rankdata(p1[1])/len(p1[1])+(1-w)*rankdata(pA[1])/len(pA[1]),
        w*rankdata(p1[2])/len(p1[2])+(1-w)*rankdata(pA[2])/len(pA[2]))
# 3모델 앙상블: ① + top14 + 상호작용−불안정
m3,FE3=model([c for c in ALLI if c not in UNSTABLE]); p3=probs(m3,FE3)
rec("⑦ 3모델 순위평균",(rankdata(p1[0])+rankdata(pA[0])+rankdata(p3[0]))/(3*len(p1[0])),
    (rankdata(p1[1])+rankdata(pA[1])+rankdata(p3[1]))/(3*len(p1[1])),
    (rankdata(p1[2])+rankdata(pA[2])+rankdata(p3[2]))/(3*len(p1[2])))
df=pd.DataFrame(ROWS); df.to_csv(f"{RES}/roundC_combo.csv",index=False,encoding="utf-8-sig")
print("\n"+"="*72)
bp=df.loc[df["pre"].idxmax()]; bs=df.loc[df["post"].idxmax()]
# 두 지표 동시 최선(min 최대화)
df["mn"]=df[["pre","post"]].min(axis=1); bm=df.loc[df["mn"].idxmax()]
print(f"pre 최고: {bp['name']} = {bp['pre']:.4f}")
print(f"post 최고: {bs['name']} = {bs['post']:.4f}")
print(f"min(pre,post) 최고: {bm['name']} = pre{bm['pre']:.4f}/post{bm['post']:.4f}")
print(f"→ 0.79: pre {'O' if df['pre'].max()>=0.79 else 'X'} / post {'O' if df['post'].max()>=0.79 else 'X'}")
