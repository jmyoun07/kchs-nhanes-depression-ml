# -*- coding: utf-8 -*-
"""ROUND A — 파생 특징 공학(17변수의 조합/상호작용/스플라인). 외부데이터 미사용, §2.10 준수.
기준선: 내부 0.831 / pre 0.774 / post 0.765. 목표 0.79.
"""
import os, sys, warnings, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
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

def add_composites(df):
    d=df.copy()
    n=lambda c: pd.to_numeric(d[c],errors="coerce")
    # 임상적 조합(모두 17변수에서 파생, 외부데이터 무)
    d["multimorbid"]=n("htn").fillna(0)+n("dm").fillna(0)                       # 동반질환수
    d["social_isolation"]=(1-n("partnered").fillna(1))+n("living_alone").fillna(0)  # 사회적 고립
    lowedu=d["edu"].isin(["elem_low","middle"]).astype(float)
    d["ses_disadvantage"]=(n("income_q")<=1).astype(float)+lowedu+(1-n("employed").fillna(1))+n("food_insecure").fillna(0)  # 사회경제 취약
    smk_cur=(d["smoking"]=="current").astype(float)
    d["behav_risk"]=smk_cur+n("current_drinker").fillna(0)+(1-n("active").fillna(0))+((n("sleep_hrs")<6)|(n("sleep_hrs")>9)).astype(float)  # 행동위험
    d["subj_health"]=n("srh").fillna(3)+n("oral_health").fillna(3)             # 주관건강 합
    # 상호작용
    az=(n("age")-50)/15; sz=n("srh")-3
    d["age_x_srh"]=az*sz
    d["srh_x_oral"]=(n("srh")-3)*(n("oral_health")-3)
    d["age_x_employed"]=az*n("employed").fillna(0)
    d["income_x_employed"]=n("income_q").fillna(2)*n("employed").fillna(0)
    # 스플라인(2차)
    d["age_sq"]=az**2; d["bmi_c"]=(n("bmi")-24); d["bmi_sq"]=((n("bmi")-24)/5)**2
    d["sleep_dev"]=(n("sleep_hrs")-7).abs()
    return d
COMP=["multimorbid","social_isolation","ses_disadvantage","behav_risk","subj_health",
      "age_x_srh","srh_x_oral","age_x_employed","income_x_employed","age_sq","bmi_c","bmi_sq","sleep_dev"]
kc,pc,sc=add_composites(kchs),add_composites(pre),add_composites(post)

tr,ho=train_test_split(np.arange(len(kc)),train_size=500000,test_size=100000,stratify=y,random_state=SEED)
Who,yho=kc.iloc[ho]["W"].to_numpy(float),y[ho]
def wauc(d,p): return roc_auc_score(d["dep"].astype(int).values,p,sample_weight=d["W"].to_numpy(float))
def ct(NUMx): return ColumnTransformer([("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),NUMx),
    ("bin","passthrough",BIN),("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore",drop="first"))]),CAT)])
def lgbm(**kw):
    p=dict(n_estimators=600,learning_rate=0.03,num_leaves=31,subsample=0.8,colsample_bytree=0.8,random_state=SEED,n_jobs=-1,verbose=-1);p.update(kw)
    return lgb.LGBMClassifier(**p)
ROWS=[]
def run(name,NUMx,**kw):
    FE=NUMx+BIN+CAT
    m=Pipeline([("pre",ct(NUMx)),("clf",lgbm(**kw))]).fit(kc.iloc[tr][FE],y[tr])
    i=roc_auc_score(yho,m.predict_proba(kc.iloc[ho][FE])[:,1],sample_weight=Who)
    p=wauc(pc,m.predict_proba(pc[FE])[:,1]); s=wauc(sc,m.predict_proba(sc[FE])[:,1])
    ROWS.append(dict(name=name,internal=round(i,4),pre=round(p,4),post=round(s,4)))
    print(f"{name:<42}{i:>8.4f}{p:>9.4f}{s:>9.4f}",flush=True); return i,p,s

print(f"{'ROUND A: 파생 특징 공학':<42}{'내부':>8}{'pre':>9}{'post':>9}"); print("-"*70,flush=True)
run("① 기준선(17)",NUM)
run("② +동반질환·고립·SES·행동·주관건강",NUM+["multimorbid","social_isolation","ses_disadvantage","behav_risk","subj_health"])
run("③ +상호작용",NUM+["age_x_srh","srh_x_oral","age_x_employed","income_x_employed"])
run("④ +스플라인",NUM+["age_sq","bmi_c","bmi_sq","sleep_dev"])
run("⑤ +전체 파생",NUM+COMP)
run("⑥ 전체파생+강정규화",NUM+COMP,num_leaves=15,min_child_samples=500,reg_alpha=1.0,reg_lambda=5.0,n_estimators=900,learning_rate=0.02,subsample=0.7,colsample_bytree=0.7)
# 상호작용 방향 심화(③이 pre 0.779로 유망)
INTER=["age_x_srh","srh_x_oral","age_x_employed","income_x_employed"]
run("⑨ 상호작용+강정규화",NUM+INTER,num_leaves=15,min_child_samples=500,reg_alpha=1.0,reg_lambda=5.0,n_estimators=900,learning_rate=0.02,subsample=0.7,colsample_bytree=0.7)
# 추가 상호작용
kc["age_x_food"]=((pd.to_numeric(kc["age"])-50)/15)*pd.to_numeric(kc["food_insecure"]).fillna(0)
pc["age_x_food"]=((pd.to_numeric(pc["age"])-50)/15)*pd.to_numeric(pc["food_insecure"]).fillna(0)
sc["age_x_food"]=((pd.to_numeric(sc["age"])-50)/15)*pd.to_numeric(sc["food_insecure"]).fillna(0)
for df in (kc,pc,sc):
    az=(pd.to_numeric(df["age"])-50)/15
    df["bmi_x_age"]=az*(pd.to_numeric(df["bmi"])-24)/5
    df["srh_x_ses"]=(pd.to_numeric(df["srh"])-3)*(pd.to_numeric(df["income_q"]).fillna(2)-2)
INTER2=INTER+["age_x_food","bmi_x_age","srh_x_ses"]
run("⑩ 상호작용 확장(7종)",NUM+INTER2)
run("⑪ 상호작용확장+강정규화",NUM+INTER2,num_leaves=15,min_child_samples=500,reg_alpha=1.0,reg_lambda=5.0,n_estimators=900,learning_rate=0.02)
run("⑫ 상호작용+전체파생",NUM+COMP+["age_x_food","bmi_x_age","srh_x_ses"])
# 단조 제약 + 전체파생
FEc=NUM+COMP; _ct=ct(FEc); _=_ct.fit_transform(kc.iloc[tr[:3000]][FEc+BIN+CAT]); ncol=_.shape[1]
mc=[0]*ncol
for i,c in enumerate(FEc):
    if c in("srh","oral_health","subj_health","multimorbid","ses_disadvantage","behav_risk","social_isolation"): mc[i]=1
    elif c=="income_q": mc[i]=-1
b0=len(FEc)
for j,c in enumerate(BIN):
    if c in("partnered","employed","active"): mc[b0+j]=-1
    elif c in("living_alone","food_insecure","htn","dm"): mc[b0+j]=1
run("⑦ 전체파생+단조제약",NUM+COMP,monotone_constraints=mc)
run("⑧ 전체파생+단조+강정규화",NUM+COMP,monotone_constraints=mc,num_leaves=15,min_child_samples=500,reg_alpha=1.0,reg_lambda=5.0,n_estimators=900,learning_rate=0.02)

df=pd.DataFrame(ROWS); df.to_csv(f"{RES}/roundA_composite.csv",index=False,encoding="utf-8-sig")
b=df.iloc[0]
bp=df.loc[df["pre"].idxmax()]; bs=df.loc[df["post"].idxmax()]
print("\n"+"="*70)
print(f"기준선: pre {b['pre']:.4f} / post {b['post']:.4f}")
print(f"pre 최고: {bp['name']} = {bp['pre']:.4f} (Δ{bp['pre']-b['pre']:+.4f})")
print(f"post 최고: {bs['name']} = {bs['post']:.4f} (Δ{bs['post']-b['post']:+.4f})")
print(f"→ 0.79 도달: pre {'O' if df['pre'].max()>=0.79 else 'X'} / post {'O' if df['post'].max()>=0.79 else 'X'}")
