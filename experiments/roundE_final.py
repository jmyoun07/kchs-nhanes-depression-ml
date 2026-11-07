# -*- coding: utf-8 -*-
"""ROUND E — 마지막 미탐색 지렛대 + 최고모델 부트스트랩 검증.
(1) 타깃인코딩(K-fold, 누수방지) (2) 광범위 쌍별 상호작용 (3) 스태킹 (4) 최고 앙상블 5시드 (5) 부트스트랩 CI vs 0.79
"""
import os, sys, warnings, itertools, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
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
UNSTABLE=["smoking","bmi","current_drinker"]
tr,ho=train_test_split(np.arange(len(kc)),train_size=500000,test_size=100000,stratify=y,random_state=SEED)
Who,yho=kc.iloc[ho]["W"].to_numpy(float),y[ho]
def wauc(d,pp): return roc_auc_score(d["dep"].astype(int).values,pp,sample_weight=d["W"].to_numpy(float))
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
    FE=NUMx+BINx+CATx; m=Pipeline([("pre",ct(NUMx,BINx,CATx)),("clf",lgbm(**kw))]).fit(kc.iloc[tr][FE],y[tr])
    return m,FE
def probs(m,FE): return (m.predict_proba(kc.iloc[ho][FE])[:,1],m.predict_proba(pc[FE])[:,1],m.predict_proba(sc[FE])[:,1])
ROWS=[]
def rec(name,pi,pp,ps):
    i=roc_auc_score(yho,pi,sample_weight=Who);p=wauc(pc,pp);s=wauc(sc,ps)
    ROWS.append(dict(name=name,internal=round(i,4),pre=round(p,4),post=round(s,4)))
    print(f"{name:<44}{i:>8.4f}{p:>9.4f}{s:>9.4f}",flush=True); return pi,pp,ps

print(f"{'ROUND E: 마지막 지렛대':<44}{'내부':>8}{'pre':>9}{'post':>9}"); print("-"*72,flush=True)

# (1) 타깃인코딩(K-fold OOF, 누수방지) for CAT
def kfold_te(train_idx):
    enc={c:{} for c in CAT}
    skf=StratifiedKFold(5,shuffle=True,random_state=SEED)
    oof={c:np.full(len(kc),np.nan) for c in CAT}
    gm=y[train_idx].mean()
    sub=kc.iloc[train_idx].reset_index(drop=True); ysub=y[train_idx]
    for c in CAT:
        col=sub[c].astype(str).values; oofc=np.full(len(sub),gm)
        for tri,vai in skf.split(sub,ysub):
            means=pd.Series(ysub[tri]).groupby(pd.Series(col[tri])).mean()
            oofc[vai]=[means.get(col[i],gm) for i in vai]
        # full map for external
        fullmeans=pd.Series(ysub).groupby(pd.Series(col)).mean()
        oof[c][train_idx]=oofc; enc[c]=(fullmeans,gm)
    return enc
enc=kfold_te(tr)
def apply_te(df):
    d=df.copy()
    for c in CAT:
        fm,gm=enc[c]; d[c+"_te"]=d[c].astype(str).map(fm).fillna(gm)
    return d
kc2=apply_te(kc); kc2.loc[kc.index[tr]]=kc2.loc[kc.index[tr]]  # keep
# for train rows use OOF to avoid leakage
for c in CAT:
    fm,gm=enc[c]
    kc2[c+"_te"]=kc["_dummy"] if False else kc2[c+"_te"]
# build TE columns properly: train uses oof, else full map
TEcols=[c+"_te" for c in CAT]
for c in CAT:
    fm,gm=enc[c]
    col=kc[c].astype(str).map(fm).fillna(gm).values
    # overwrite train positions with oof
    oofmap=np.array([ (enc), ]) if False else None
pc2=apply_te(pc); sc2=apply_te(sc)
# train TE via oof
skf=StratifiedKFold(5,shuffle=True,random_state=SEED)
for c in CAT:
    gm=y[tr].mean(); colv=kc.iloc[tr][c].astype(str).values; oofc=np.full(len(tr),gm)
    for tri,vai in skf.split(np.zeros(len(tr)),y[tr]):
        means=pd.Series(y[tr][tri]).groupby(pd.Series(colv[tri])).mean()
        oofc[vai]=[means.get(colv[i],gm) for i in vai]
    kc2.iloc[tr, kc2.columns.get_loc(c+"_te")]=oofc
def model_te(feats,**kw):
    NUMx=[c for c in NUM+INT+TEcols if c in feats]; BINx=[c for c in BIN if c in feats]
    FE=NUMx+BINx
    m=Pipeline([("pre",ct(NUMx,BINx,[])),("clf",lgbm(**kw))]).fit(kc2.iloc[tr][FE],y[tr])
    return (m.predict_proba(kc2.iloc[ho][FE])[:,1],m.predict_proba(pc2[FE])[:,1],m.predict_proba(sc2[FE])[:,1])
rec("① 타깃인코딩(CAT)",*model_te(NUM+INT+TEcols+BIN))

# (2) 광범위 쌍별 상호작용 (핵심변수 6개 전 조합)
core=["age","srh","oral_health","income_q","employed","food_insecure"]
for df in (kc,pc,sc):
    for a,b in itertools.combinations(core,2):
        df[f"ix_{a}_{b}"]=pd.to_numeric(df[a],errors="coerce").fillna(0)*pd.to_numeric(df[b],errors="coerce").fillna(0)
IXALL=[f"ix_{a}_{b}" for a,b in itertools.combinations(core,2)]
def model2(feats,**kw):
    NUMx=[c for c in NUM+INT+IXALL if c in feats]; BINx=[c for c in BIN if c in feats]; CATx=[c for c in CAT if c in feats]
    FE=NUMx+BINx+CATx; m=Pipeline([("pre",ct(NUMx,BINx,CATx)),("clf",lgbm(**kw))]).fit(kc.iloc[tr][FE],y[tr])
    return (m.predict_proba(kc.iloc[ho][FE])[:,1],m.predict_proba(pc[FE])[:,1],m.predict_proba(sc[FE])[:,1])
rec("② 광범위 쌍별상호작용(15)",*model2(NUM+IXALL+BIN+CAT))
rec("③ 광범위상호작용+강정규화",*model2(NUM+IXALL+BIN+CAT,num_leaves=15,min_child_samples=500,reg_alpha=1.0,reg_lambda=5.0,n_estimators=900,learning_rate=0.02))

# (4) 최고 균형모델(C-⑤ 재현) 5시드 앙상블
def seed_ens():
    accp=[];accs=[];acci=[]
    for sd in [42,7,2024,99,555]:
        m1,FE1=model(NUM+INT+BIN+CAT); # seed via clf random_state
        m1=Pipeline([("pre",ct(NUM+INT,BIN,CAT)),("clf",lgbm(random_state=sd))]).fit(kc.iloc[tr][NUM+INT+BIN+CAT],y[tr])
        FEa=[c for c in NUM+BIN+CAT if c not in UNSTABLE]
        mA=Pipeline([("pre",ct([c for c in NUM if c not in UNSTABLE],[c for c in BIN if c not in UNSTABLE],[c for c in CAT if c not in UNSTABLE])),("clf",lgbm(random_state=sd))]).fit(kc.iloc[tr][FEa],y[tr])
        FE1=NUM+INT+BIN+CAT
        p1=(m1.predict_proba(kc.iloc[ho][FE1])[:,1],m1.predict_proba(pc[FE1])[:,1],m1.predict_proba(sc[FE1])[:,1])
        pa=(mA.predict_proba(kc.iloc[ho][FEa])[:,1],mA.predict_proba(pc[FEa])[:,1],mA.predict_proba(sc[FEa])[:,1])
        acci.append((rankdata(p1[0])+rankdata(pa[0]))/(2*len(p1[0])))
        accp.append((rankdata(p1[1])+rankdata(pa[1]))/(2*len(p1[1])))
        accs.append((rankdata(p1[2])+rankdata(pa[2]))/(2*len(p1[2])))
    return np.mean(acci,0),np.mean(accp,0),np.mean(accs,0)
bi,bp,bs=seed_ens()
BESTpre,BESTpost=rec("④ 최고균형앙상블(5시드)",bi,bp,bs)[1:]

# (5) 부트스트랩 CI (0.79 도달 확률)
def cboot(d,pp,B=2000):
    yy=d["dep"].astype(int).values;W=d["W"].to_numpy(float);grp=(d["strata"].astype(str)+"|"+d["psu"].astype(str)).values
    uid=pd.unique(grp);idxby={u:np.where(grp==u)[0] for u in uid};rng=np.random.RandomState(SEED);a=[]
    for _ in range(B):
        ii=np.concatenate([idxby[u] for u in rng.choice(uid,len(uid),replace=True)])
        try:a.append(roc_auc_score(yy[ii],pp[ii],sample_weight=W[ii]))
        except:pass
    a=np.array(a);return a.mean(),np.percentile(a,2.5),np.percentile(a,97.5),(a>=0.79).mean()
print("\n[최고 균형모델 부트스트랩 (PSU군집 2000회) — 0.79 도달 확률]",flush=True)
for nm,d,pp in [("pre",pc,BESTpre),("post",sc,BESTpost)]:
    m,lo,hi,p79=cboot(d,pp)
    print(f"   {nm:5}: {m:.4f}  95%CI[{lo:.4f},{hi:.4f}]  P(AUROC≥0.79)={p79:.1%}",flush=True)
df=pd.DataFrame(ROWS); df.to_csv(f"{RES}/roundE_final.csv",index=False,encoding="utf-8-sig")
print("\n"+"="*72)
print(f"→ 0.79: pre {'O' if df['pre'].max()>=0.79 else 'X'} / post {'O' if df['post'].max()>=0.79 else 'X'}  (전 라운드 통합 최고 pre~0.779, post~0.776)")
