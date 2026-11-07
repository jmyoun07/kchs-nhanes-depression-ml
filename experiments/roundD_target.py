# -*- coding: utf-8 -*-
"""ROUND D — 학습 타깃·표본 설계 변형(§2.12 민감도 범위). 평가는 항상 dep(≥10) 가중AUROC.
가설: 더 넓은 타깃(≥5)·연속점수 학습, 팬데믹기 연도 학습, 연령하한 변경이 전이에 유리할 수 있다.
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
NUM=["age","sleep_hrs","bmi","income_q","srh","oral_health"]
BIN=["partnered","living_alone","current_drinker","active","htn","dm","employed","food_insecure"]
CAT=["sex","edu","smoking"]; FE=NUM+BIN+CAT
def ct(): return ColumnTransformer([("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),NUM),
    ("bin","passthrough",BIN),("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore",drop="first"))]),CAT)])
def lgbm(**kw):
    p=dict(n_estimators=600,learning_rate=0.03,num_leaves=31,subsample=0.8,colsample_bytree=0.8,random_state=SEED,n_jobs=-1,verbose=-1);p.update(kw)
    return lgb.LGBMClassifier(**p)
def wauc(d,p): return roc_auc_score(d["dep"].astype(int).values,p,sample_weight=d["W"].to_numpy(float))
# 평가용 홀드아웃(항상 dep≥10)
ydep=kchs["dep"].astype(int).values
tr,ho=train_test_split(np.arange(len(kchs)),train_size=500000,test_size=100000,stratify=ydep,random_state=SEED)
Who=kchs.iloc[ho]["W"].to_numpy(float); yho=ydep[ho]
ROWS=[]
def evalm(name,m):
    i=roc_auc_score(yho,m.predict_proba(kchs.iloc[ho][FE])[:,1],sample_weight=Who)
    p=wauc(pre,m.predict_proba(pre[FE])[:,1]); s=wauc(post,m.predict_proba(post[FE])[:,1])
    ROWS.append(dict(name=name,internal=round(i,4),pre=round(p,4),post=round(s,4)))
    print(f"{name:<40}{i:>8.4f}{p:>9.4f}{s:>9.4f}",flush=True); return p,s

print(f"{'ROUND D: 타깃·표본 설계':<40}{'내부':>8}{'pre':>9}{'post':>9}"); print("-"*68,flush=True)
# 기준선
evalm("① 기준선(≥10, 전체)",Pipeline([("pre",ct()),("clf",lgbm())]).fit(kchs.iloc[tr][FE],ydep[tr]))
# ② 타깃 ≥5로 학습 → ≥10 평가
y5=(kchs["phq_sum"]>=5).astype(int).values
evalm("② 타깃≥5 학습",Pipeline([("pre",ct()),("clf",lgbm())]).fit(kchs.iloc[tr][FE],y5[tr]))
# ③ 연속 phq_sum 회귀 → 순위
from lightgbm import LGBMRegressor
reg=Pipeline([("pre",ct()),("clf",LGBMRegressor(n_estimators=600,learning_rate=0.03,num_leaves=31,subsample=0.8,colsample_bytree=0.8,random_state=SEED,n_jobs=-1,verbose=-1))])
reg.fit(kchs.iloc[tr][FE],pd.to_numeric(kchs.iloc[tr]["phq_sum"]).values)
def wauc_r(d,p): return roc_auc_score(d["dep"].astype(int).values,p,sample_weight=d["W"].to_numpy(float))
i=roc_auc_score(yho,reg.predict(kchs.iloc[ho][FE]),sample_weight=Who); p=wauc_r(pre,reg.predict(pre[FE])); s=wauc_r(post,reg.predict(post[FE]))
ROWS.append(dict(name="③ phq_sum 회귀",internal=round(i,4),pre=round(p,4),post=round(s,4)))
print(f"{'③ phq_sum 회귀':<40}{i:>8.4f}{p:>9.4f}{s:>9.4f}",flush=True)
# ④ 팬데믹기 연도만 학습(2021-2025)
mrec=kchs["year"]>=2021; itr_rec=np.intersect1d(tr,np.where(mrec.values)[0])
evalm("④ 2021-2025 학습",Pipeline([("pre",ct()),("clf",lgbm())]).fit(kchs.iloc[itr_rec][FE],ydep[itr_rec]))
# ⑤ 최근 3년(2023-2025)
m3=kchs["year"]>=2023; itr3=np.intersect1d(tr,np.where(m3.values)[0])
evalm("⑤ 2023-2025 학습",Pipeline([("pre",ct()),("clf",lgbm())]).fit(kchs.iloc[itr3][FE],ydep[itr3]))
# ⑥ 연령하한 30+
m30=pd.to_numeric(kchs["age"])>=30; itr30=np.intersect1d(tr,np.where(m30.values)[0])
evalm("⑥ 연령 30+ 학습",Pipeline([("pre",ct()),("clf",lgbm())]).fit(kchs.iloc[itr30][FE],ydep[itr30]))
# ⑦ 클래스가중 scale_pos_weight
evalm("⑦ scale_pos_weight=5",Pipeline([("pre",ct()),("clf",lgbm(scale_pos_weight=5))]).fit(kchs.iloc[tr][FE],ydep[tr]))
# ⑧ 전체 205만 학습(홀드아웃 제외 전량)
evalm("⑧ 전체 POOL 학습",Pipeline([("pre",ct()),("clf",lgbm())]).fit(kchs.drop(index=ho)[FE],np.delete(ydep,ho)))
df=pd.DataFrame(ROWS); df.to_csv(f"{RES}/roundD_target.csv",index=False,encoding="utf-8-sig")
print("\n"+"="*68)
bp=df.loc[df["pre"].idxmax()]; bs=df.loc[df["post"].idxmax()]
print(f"pre 최고: {bp['name']} = {bp['pre']:.4f} / post 최고: {bs['name']} = {bs['post']:.4f}")
print(f"→ 0.79: pre {'O' if df['pre'].max()>=0.79 else 'X'} / post {'O' if df['post'].max()>=0.79 else 'X'}")
