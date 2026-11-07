# -*- coding: utf-8 -*-
"""ROUND F — leakage-clean 최종 모델 선택. 모든 선택(특징셋·하이퍼파라미터)을 KCHS 내부 CV로만.
NHANES는 최종 1회 평가만. §2.8(정식 튜닝, 미이행) + §2.10(무재적합) 준수.
"""
import os, sys, warnings, numpy as np, pandas as pd
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
CAT=["sex","edu","smoking"]
def addint(df):
    d=df.copy(); n=lambda c:pd.to_numeric(d[c],errors="coerce"); az=(n("age")-50)/15
    d["age_x_srh"]=az*(n("srh")-3); d["srh_x_oral"]=(n("srh")-3)*(n("oral_health")-3)
    d["age_x_employed"]=az*n("employed").fillna(0); d["income_x_employed"]=n("income_q").fillna(2)*n("employed").fillna(0)
    return d
INT=["age_x_srh","srh_x_oral","age_x_employed","income_x_employed"]
kc,pc,sc=addint(kchs),addint(pre),addint(post)
def ct(NUMx,BINx,CATx):
    t=[]
    if NUMx:t.append(("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),NUMx))
    if BINx:t.append(("bin","passthrough",BINx))
    if CATx:t.append(("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore",drop="first"))]),CATx))
    return ColumnTransformer(t)
def wauc(d,p): return roc_auc_score(d["dep"].astype(int).values,p,sample_weight=d["W"].to_numpy(float))

FEATSETS={"F0_plain17":NUM+BIN+CAT, "F1_interact":NUM+INT+BIN+CAT}
# 하이퍼파라미터 랜덤서치 그리드
rng=np.random.RandomState(SEED)
GRID=[]
for _ in range(24):
    GRID.append(dict(
        num_leaves=int(rng.choice([7,15,31,63])),
        min_child_samples=int(rng.choice([50,200,500,1000,3000])),
        learning_rate=float(rng.choice([0.01,0.02,0.03,0.05])),
        n_estimators=int(rng.choice([400,600,900,1500])),
        reg_alpha=float(rng.choice([0,0.5,1.0,5.0])),
        reg_lambda=float(rng.choice([0,1.0,5.0,20.0])),
        subsample=float(rng.choice([0.6,0.8,1.0])),
        colsample_bytree=float(rng.choice([0.6,0.8,1.0])),
    ))

# 분할: 최종 refit용 500k/100k + 선택용 CV는 train내 300k 서브샘플
tr,ho=train_test_split(np.arange(len(kc)),train_size=500000,test_size=100000,stratify=y,random_state=SEED)
Who,yho=kc.iloc[ho]["W"].to_numpy(float),y[ho]
rngc=np.random.RandomState(1); cvidx=rngc.choice(tr,300000,replace=False)
Xcv=kc.iloc[cvidx]; ycv=y[cvidx]; Wcv=Xcv["W"].to_numpy(float)
skf=StratifiedKFold(4,shuffle=True,random_state=SEED)

print("[선택 단계] KCHS 내부 4-fold 가중 CV로 특징셋×하이퍼파라미터 평가 (NHANES 미사용)",flush=True)
results=[]
for fname,feats in FEATSETS.items():
    NUMx=[c for c in NUM+INT if c in feats]; BINx=[c for c in BIN if c in feats]; CATx=[c for c in CAT if c in feats]
    FE=NUMx+BINx+CATx
    for gi,g in enumerate(GRID):
        oof=np.zeros(len(Xcv))
        for tri,vai in skf.split(Xcv[FE],ycv):
            m=Pipeline([("pre",ct(NUMx,BINx,CATx)),("clf",lgb.LGBMClassifier(random_state=SEED,n_jobs=-1,verbose=-1,**g))]).fit(Xcv.iloc[tri][FE],ycv[tri])
            oof[vai]=m.predict_proba(Xcv.iloc[vai][FE])[:,1]
        cvauc=roc_auc_score(ycv,oof,sample_weight=Wcv)
        results.append((fname,gi,cvauc,g,FE))
    print(f"   {fname}: 완료",flush=True)
results.sort(key=lambda r:-r[2])
best=results[0]
print(f"\n[선택 결과] 내부 CV 최고: {best[0]} #{best[1]}  CV-AUROC={best[2]:.4f}")
print(f"   하이퍼파라미터: {best[3]}")

print("\n[최종 평가] 선택된 설정을 500k로 refit → NHANES 1회 (leakage 없음)",flush=True)
fname,gi,cvauc,g,FE=best
NUMx=[c for c in NUM+INT if c in FE]; BINx=[c for c in BIN if c in FE]; CATx=[c for c in CAT if c in FE]
mfin=Pipeline([("pre",ct(NUMx,BINx,CATx)),("clf",lgb.LGBMClassifier(random_state=SEED,n_jobs=-1,verbose=-1,**g))]).fit(kc.iloc[tr][FE],y[tr])
ip=mfin.predict_proba(kc.iloc[ho][FE])[:,1]; pp=mfin.predict_proba(pc[FE])[:,1]; sp=mfin.predict_proba(sc[FE])[:,1]
I,P,S=roc_auc_score(yho,ip,sample_weight=Who),wauc(pre,pp),wauc(post,sp)
print(f"   ★ 최종모델: 내부 {I:.4f}  외부 pre {P:.4f}  post {S:.4f}")
# 부트스트랩
def cb(d,p,B=2000):
    yy=d["dep"].astype(int).values;W=d["W"].to_numpy(float);grp=(d["strata"].astype(str)+"|"+d["psu"].astype(str)).values
    uid=pd.unique(grp);idx={u:np.where(grp==u)[0] for u in uid};r=np.random.RandomState(SEED);a=[]
    for _ in range(B):
        ii=np.concatenate([idx[u] for u in r.choice(uid,len(uid),replace=True)])
        try:a.append(roc_auc_score(yy[ii],p[ii],sample_weight=W[ii]))
        except:pass
    a=np.array(a);return a.mean(),np.percentile(a,2.5),np.percentile(a,97.5),(a>=0.79).mean()
for nm,d,p in [("pre",pre,pp),("post",post,sp)]:
    m,lo,hi,p79=cb(d,p)
    print(f"   {nm:5}: {m:.4f} 95%CI[{lo:.4f},{hi:.4f}]  P(≥0.79)={p79:.1%}",flush=True)
pd.DataFrame([(r[0],r[1],r[2]) for r in results],columns=["featset","cfg","cv_auroc"]).to_csv(f"{RES}/roundF_nested.csv",index=False,encoding="utf-8-sig")
print(f"\n→ 0.79 도달: {'O' if max(P,S)>=0.79 else 'X'}  (내부CV로 정직 선택한 최종모델 기준)")
