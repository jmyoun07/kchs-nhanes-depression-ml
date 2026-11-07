# -*- coding: utf-8 -*-
"""Supplement — operating characteristics(Table S1) + 확장 민감도(§2.12)."""
import os, sys, warnings, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
D=r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model"; RES=f"{D}/results"; SEED=42
kchs=pd.read_parquet(f"{D}/data/ext_kchs.parquet"); pre=pd.read_parquet(f"{D}/data/ext_pre.parquet"); post=pd.read_parquet(f"{D}/data/ext_post.parquet")
NUM=["age","sleep_hrs","bmi","income_q","srh","oral_health"]
BIN=["partnered","living_alone","current_drinker","active","htn","dm","employed","food_insecure"]
CAT=["sex","edu","smoking"]; FE=NUM+BIN+CAT
y=kchs["dep"].astype(int).values
LGB=dict(num_leaves=31,min_child_samples=1000,learning_rate=0.01,n_estimators=900,reg_alpha=1.0,reg_lambda=20.0,subsample=0.6,colsample_bytree=0.6,random_state=SEED,n_jobs=-1,verbose=-1)
def ct(native=True,add_ind=False):
    binp="passthrough" if native else SimpleImputer(strategy="most_frequent",add_indicator=add_ind)
    numimp=SimpleImputer(strategy="median",add_indicator=add_ind)
    return ColumnTransformer([("num",Pipeline([("i",numimp),("s",StandardScaler())]),NUM),
        ("bin",binp,BIN),("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore",drop="first"))]),CAT)])
def wauc(d,p): return roc_auc_score(d["dep"].astype(int).values,p,sample_weight=d["W"].to_numpy(float))
tr,ho=train_test_split(np.arange(len(kchs)),train_size=500000,test_size=100000,stratify=y,random_state=SEED)
Who,yho=kchs.iloc[ho]["W"].to_numpy(float),y[ho]
model=Pipeline([("pre",ct()),("clf",lgb.LGBMClassifier(**LGB))]).fit(kchs.iloc[tr][FE],y[tr])
def P(d): return model.predict_proba(d[FE])[:,1]

# ── Table S1: operating characteristics ──
print("[Table S1] operating characteristics",flush=True)
def opchar(d,p,thr):
    yv=d["dep"].astype(int).values; W=d["W"].to_numpy(float); pred=p>=thr
    tp=W[(pred)&(yv==1)].sum(); fp=W[(pred)&(yv==0)].sum(); tn=W[(~pred)&(yv==0)].sum(); fn=W[(~pred)&(yv==1)].sum()
    sens=tp/(tp+fn); spec=tn/(tn+fp); ppv=tp/(tp+fp) if (tp+fp)>0 else np.nan; npv=tn/(tn+fn) if (tn+fn)>0 else np.nan
    return sens,spec,ppv,npv
def youden(d,p):
    yv=d["dep"].astype(int).values;W=d["W"].to_numpy(float);best=(-1,0.5)
    for t in np.linspace(0.02,0.6,59):
        s,sp,_,_=opchar(d,p,t)
        if s+sp-1>best[0]: best=(s+sp-1,t)
    return best[1]
rows=[]
for nm,d in [("Internal",kchs.iloc[ho]),("NHANES pre",pre),("NHANES post",post)]:
    p=P(d); prev=np.average(d["dep"].astype(int),weights=d["W"].to_numpy(float)); jt=youden(d,p)
    for lab,thr in [(f"prevalence({prev:.3f})",prev),(f"Youden({jt:.2f})",jt),("fixed 0.10",0.10)]:
        s,sp,ppv,npv=opchar(d,p,thr)
        rows.append(dict(set=nm,threshold=lab,sensitivity=round(s,3),specificity=round(sp,3),PPV=round(ppv,3),NPV=round(npv,3)))
        print(f"   {nm:<12}{lab:<18} sens={s:.3f} spec={sp:.3f} PPV={ppv:.3f} NPV={npv:.3f}",flush=True)
pd.DataFrame(rows).to_csv(f"{RES}/table_s1_operating.csv",index=False,encoding="utf-8-sig")

# ── 확장 민감도 ──
print("\n[확장 민감도]",flush=True)
srows=[]
def evalm(m,label,dtr=None,ytr=None):
    dtr=kchs.iloc[tr][FE] if dtr is None else dtr; ytr=y[tr] if ytr is None else ytr
    i=roc_auc_score(yho,m.predict_proba(kchs.iloc[ho][FE])[:,1],sample_weight=Who)
    p=wauc(pre,m.predict_proba(pre[FE])[:,1]); s=wauc(post,m.predict_proba(post[FE])[:,1])
    srows.append(dict(cond=label,internal=round(i,4),ext_pre=round(p,4),ext_post=round(s,4)))
    print(f"   {label:<28} 내부 {i:.4f} pre {p:.4f} post {s:.4f}",flush=True)
# 기준
evalm(model,"기준 (tree-native, 비가중학습)")
# 가중 학습
mw=Pipeline([("pre",ct()),("clf",lgb.LGBMClassifier(**LGB))]).fit(kchs.iloc[tr][FE],y[tr],clf__sample_weight=kchs.iloc[tr]["W"].to_numpy(float))
evalm(mw,"가중 학습")
# 결측지표
mi=Pipeline([("pre",ct(native=False,add_ind=True)),("clf",lgb.LGBMClassifier(**LGB))]).fit(kchs.iloc[tr][FE],y[tr])
evalm(mi,"결측지표 추가")
# MICE
mm=Pipeline([("pre",ct(native=False)),("clf",lgb.LGBMClassifier(**LGB))]);
mm.named_steps["pre"].transformers[0]=("num",Pipeline([("i",IterativeImputer(max_iter=10,random_state=SEED)),("s",StandardScaler())]),NUM)
mm.fit(kchs.iloc[tr][FE],y[tr]); evalm(mm,"MICE 대체")
# 연령 30+
m30i=np.intersect1d(tr,np.where(pd.to_numeric(kchs["age"])>=30)[0])
m30=Pipeline([("pre",ct()),("clf",lgb.LGBMClassifier(**LGB))]).fit(kchs.iloc[m30i][FE],y[m30i]); evalm(m30,"연령 30+ 학습")
pd.DataFrame(srows).to_csv(f"{RES}/sensitivity_full.csv",index=False,encoding="utf-8-sig")
print("\n저장: results/table_s1_operating.csv, sensitivity_full.csv\n완료.")
