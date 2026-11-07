# -*- coding: utf-8 -*-
"""새 예측 도메인이 NHANES 오라클 천장을 올리는지 진단.
새 다운로드 없이 로컬 원자료(MCQ/HUQ/HIQ)에서 후보 도메인 추출 →
기존 17변수에 추가 → NHANES 자체 5-fold CV 오라클 비교.
천장(현 pooled 0.765)이 오르면 KCHS 조화 가치 있음."""
import os, sys, warnings, numpy as np, pandas as pd, pyreadstat
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
SEED=42
ROOT="C:/Users/rudtn/Desktop/claude code/지역사회건강"
NB=f"{ROOT}/data/raw/nhanes"; D=f"{ROOT}/17var_model"
FOLD={"pre":("pre_2017_2020",lambda m:f"P_{m}.xpt"),"post":("post_2021_2023",lambda m:f"{m}_L.xpt")}
def clean(s,miss):
    s=pd.to_numeric(s,errors="coerce"); return s.mask(s.isin(miss))
def new_feats(period):
    """SEQN 키로 새 변수 추출 (DEMO 순서 유지, index=DEMO 위치)."""
    folder,namer=FOLD[period]
    def rd(mod,cols):
        df=pyreadstat.read_xport(f"{NB}/{folder}/{namer(mod)}",usecols=["SEQN"]+cols,encoding="latin1")[0]
        for c in cols:
            if c not in df.columns: df[c]=np.nan
        return df
    demo=rd("DEMO",["RIDAGEYR"]); m=demo.copy()   # index 0..N-1 = DEMO 위치
    mcq=["MCQ010","MCQ160A","MCQ160M","MCQ160P","MCQ160L","MCQ220"]
    m=m.merge(rd("MCQ",mcq),on="SEQN",how="left")
    m=m.merge(rd("HUQ",["HUQ051","HUQ071"]),on="SEQN",how="left")
    m=m.merge(rd("HIQ",["HIQ011"]),on="SEQN",how="left")
    o=pd.DataFrame(index=m.index); o["_age"]=m["RIDAGEYR"].to_numpy()
    cc=np.zeros(len(m)); anyv=np.zeros(len(m),bool)
    for c in mcq:
        v=clean(m[c],(7,9)); cc+=(v==1).astype(float); anyv|=v.notna().to_numpy()
    o["comorb"]=np.where(anyv,cc,np.nan)
    o["hc_visits"]=clean(m["HUQ051"],(77,99)).to_numpy()
    h71=clean(m["HUQ071"],(7,9))
    o["hospitalized"]=np.where(h71==1,1.0,np.where(h71==2,0.0,np.nan))
    h11=clean(m["HIQ011"],(7,9))
    o["insured"]=np.where(h11==1,1.0,np.where(h11==2,0.0,np.nan))
    return o

NUM=["age","sleep_hrs","bmi","income_q","srh","oral_health"]; BIN=["partnered","living_alone","current_drinker","active","htn","dm","employed","food_insecure"]; CAT=["sex","edu","smoking"]
def attach(period):
    ext=pd.read_parquet(f"{D}/data/ext_{period}.parquet")   # index=DEMO 위치 (보존됨)
    nf=new_feats(period).reindex(ext.index)                 # DEMO 위치로 정렬
    a1=nf["_age"].to_numpy(float); a2=ext["age"].to_numpy(float)
    chk=np.mean((a1==a2)|(np.isnan(a1)&np.isnan(a2)))
    print(f"  {period} 정렬검증 age 일치율={chk:.5f}  (comorb 결측 {nf['comorb'].isna().mean()*100:.1f}%, hc_visits 결측 {nf['hc_visits'].isna().mean()*100:.1f}%)")
    assert chk>0.999, f"{period} 정렬 실패"
    ext=ext.reset_index(drop=True)
    for c in ["comorb","hc_visits","hospitalized","insured"]: ext[c]=nf[c].to_numpy()
    return ext,chk

print("새 도메인 추출·정렬:")
pre,c1=attach("pre"); post,c2=attach("post")
both=pd.concat([pre,post],ignore_index=True)
# 단변량 신호
print("\nNHANES 우울과 새 변수 상관(pooled):")
for c in ["comorb","hc_visits","hospitalized","insured"]:
    x=both[c]; msk=x.notna()
    print(f"  {c:<14} r={np.corrcoef(x[msk],both['dep'].astype(float)[msk])[0,1]:+.3f}  (mean {x.mean():.2f})")

def ct(feN,feB,feC):
    return ColumnTransformer([("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),feN),
        ("bin",Pipeline([("i",SimpleImputer(strategy="most_frequent"))]),feB),
        ("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore",drop="first"))]),feC)])
def oracle(dd,feN,feB,feC,label):
    FE=feN+feB+feC; yy=dd["dep"].astype(int).values; w=dd["W"].to_numpy(float)
    best=0
    for nm,mk in [("Logit",lambda:LogisticRegression(max_iter=4000,C=0.3)),
                  ("LGBM",lambda:lgb.LGBMClassifier(num_leaves=15,min_child_samples=50,learning_rate=0.03,n_estimators=400,reg_lambda=5.0,random_state=SEED,n_jobs=-1,verbose=-1))]:
        skf=StratifiedKFold(5,shuffle=True,random_state=SEED); oof=np.zeros(len(dd))
        for a,b in skf.split(dd,yy):
            c=ct(feN,feB,feC); Xa=c.fit_transform(dd.iloc[a][FE]); Xb=c.transform(dd.iloc[b][FE])
            g=mk().fit(Xa,yy[a],sample_weight=w[a]); oof[b]=g.predict_proba(Xb)[:,1]
        best=max(best,roc_auc_score(yy,oof,sample_weight=w))
    print(f"  {label:<32} pooled 오라클={best:.4f}")
    return best

print("\n"+"="*64)
print("오라클 비교 (pooled NHANES, 5-fold CV, Logit/LGBM 최댓값)")
print("="*64)
b0=oracle(both,NUM,BIN,CAT,"17변수 (기준)")
oracle(both,NUM+["comorb"],BIN,CAT,"+다중이환(comorb)")
oracle(both,NUM+["hc_visits"],BIN+["hospitalized"],CAT,"+의료이용(visits+입원)")
oracle(both,NUM,BIN+["insured"],CAT,"+보험(insured)")
bAll=oracle(both,NUM+["comorb","hc_visits"],BIN+["hospitalized","insured"],CAT,"+새 도메인 전부(4)")
print(f"\n기준 오라클 {b0:.4f} → 새 도메인 전부 {bAll:.4f}  (Δ={bAll-b0:+.4f})")
print("Δ가 +0.01 이상이면 KCHS 조화 가치 있음. 미미하면 새 변수도 천장 못 올림 확정.")
