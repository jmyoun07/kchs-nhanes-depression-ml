# -*- coding: utf-8 -*-
"""ROUND B — 전이-강건 특징 부분집합 최적화. 연도 안정성으로 변수 가지치기(KCHS 내부만 사용, §2.10 준수).
가설: 국가 간 전이가 나쁜 변수(연도 간 계수 불안정)를 빼면 미국에서 더 잘 작동할 수 있다.
"""
import os, sys, warnings, itertools, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, StratifiedKFold
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
CAT=["sex","edu","smoking"]; ALL=NUM+BIN+CAT
tr,ho=train_test_split(np.arange(len(kchs)),train_size=500000,test_size=100000,stratify=y,random_state=SEED)
Who,yho=kchs.iloc[ho]["W"].to_numpy(float),y[ho]
def wauc(d,FE,m): return roc_auc_score(d["dep"].astype(int).values,m.predict_proba(d[FE])[:,1],sample_weight=d["W"].to_numpy(float))
def ct(NUMx,BINx,CATx):
    t=[]
    if NUMx: t.append(("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),NUMx))
    if BINx: t.append(("bin","passthrough",BINx))
    if CATx: t.append(("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore",drop="first"))]),CATx))
    return ColumnTransformer(t)
def lgbm(): return lgb.LGBMClassifier(n_estimators=600,learning_rate=0.03,num_leaves=31,subsample=0.8,colsample_bytree=0.8,random_state=SEED,n_jobs=-1,verbose=-1)
def fit_eval(feats):
    NUMx=[c for c in NUM if c in feats]; BINx=[c for c in BIN if c in feats]; CATx=[c for c in CAT if c in feats]
    FE=NUMx+BINx+CATx
    m=Pipeline([("pre",ct(NUMx,BINx,CATx)),("clf",lgbm())]).fit(kchs.iloc[tr][FE],y[tr])
    return (roc_auc_score(yho,m.predict_proba(kchs.iloc[ho][FE])[:,1],sample_weight=Who),
            wauc(pre,FE,m), wauc(post,FE,m))
ROWS=[]
def run(name,feats):
    i,p,s=fit_eval(feats); ROWS.append(dict(name=name,k=len(feats),internal=round(i,4),pre=round(p,4),post=round(s,4)))
    print(f"{name:<40}{len(feats):>4}{i:>9.4f}{p:>9.4f}{s:>9.4f}",flush=True); return i,p,s

print(f"{'ROUND B: 전이-강건 부분집합':<40}{'k':>4}{'내부':>9}{'pre':>9}{'post':>9}"); print("-"*74,flush=True)
run("전체 17",ALL)

# (1) 연도별 단일변수 AUROC 안정성 → 불안정 변수 식별 (KCHS 내부만)
print("\n[연도 간 단일변수 판별력 안정성 — 낮을수록 전이 위험]",flush=True)
years=sorted(kchs["year"].unique())
stab={}
for c in ALL:
    aucs=[]
    for yr in years:
        m=kchs["year"]==yr; d=kchs[m]
        if c in CAT:
            xx=pd.get_dummies(d[c].astype(str),dummy_na=True).values
        else:
            xx=pd.to_numeric(d[c],errors="coerce").fillna(pd.to_numeric(kchs[c],errors="coerce").median() if c not in CAT else 0).values.reshape(-1,1)
        yy=d["dep"].astype(int).values
        if len(np.unique(yy))<2: continue
        try:
            if xx.shape[1]==1: a=roc_auc_score(yy,xx.ravel()); a=max(a,1-a)
            else:
                from sklearn.linear_model import LogisticRegression
                a=roc_auc_score(yy,LogisticRegression(max_iter=200).fit(xx,yy).predict_proba(xx)[:,1])
            aucs.append(a)
        except: pass
    stab[c]=(np.mean(aucs),np.std(aucs))
for c,(mu,sd) in sorted(stab.items(),key=lambda x:-x[1][0]):
    print(f"   {c:<18} 평균AUC {mu:.4f}  연도SD {sd:.4f}",flush=True)

# (2) 판별력 강하고 안정적인 변수 순으로 top-k 부분집합
ranked=[c for c,_ in sorted(stab.items(),key=lambda x:-(x[1][0]-x[1][1]))]  # 평균-SD(안정성 반영)
print("\n[안정성 가중 순위 상위 k개 부분집합]",flush=True)
for k in [5,8,10,12,14,15,16,17]:
    run(f"top{k}(안정성순)",ranked[:k])

# (3) 개별 변수 제거(leave-one-out) — 빼면 외부 오르는 변수?
print("\n[변수 하나 제거 시 외부 변화(음수면 그 변수가 외부에 해로움)]",flush=True)
base_i,base_p,base_s=fit_eval(ALL)
for c in ALL:
    i,p,s=fit_eval([x for x in ALL if x!=c])
    print(f"   -{c:<18} pre {p:.4f}(Δ{p-base_p:+.4f}) post {s:.4f}(Δ{s-base_s:+.4f})",flush=True)
    ROWS.append(dict(name=f"drop_{c}",k=16,internal=round(i,4),pre=round(p,4),post=round(s,4)))

df=pd.DataFrame(ROWS); df.to_csv(f"{RES}/roundB_subset.csv",index=False,encoding="utf-8-sig")
bp=df.loc[df["pre"].idxmax()]; bs=df.loc[df["post"].idxmax()]
print("\n"+"="*74)
print(f"pre 최고: {bp['name']} (k={bp['k']}) = {bp['pre']:.4f}")
print(f"post 최고: {bs['name']} (k={bs['k']}) = {bs['post']:.4f}")
print(f"→ 0.79 도달: pre {'O' if df['pre'].max()>=0.79 else 'X'} / post {'O' if df['post'].max()>=0.79 else 'X'}")
