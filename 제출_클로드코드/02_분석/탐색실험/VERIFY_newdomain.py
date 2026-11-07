# -*- coding: utf-8 -*-
import sys; sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pyreadstat
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
CFG={"pre":("pre_2017_2020",lambda m:f"P_{m}.xpt","P_DEMO.xpt"),
     "post":("post_2021_2023",lambda m:f"{m}_L.xpt","DEMO_L.xpt")}

def clean(s,miss):
    s=pd.to_numeric(s,errors="coerce"); return s.mask(s.isin(miss))

def build_newfeats(period):
    folder,namer,demofile=CFG[period]
    demo,_=pyreadstat.read_xport(f"{NB}/{folder}/{demofile}",usecols=["SEQN","RIDAGEYR"],encoding="latin1")
    demo=demo.reset_index(drop=True)  # position index = DEMO file order
    # MCQ multimorbidity (both periods)
    mcq_cols=["MCQ010","MCQ160A","MCQ160M","MCQ160P","MCQ160L","MCQ220"]
    mcq,_=pyreadstat.read_xport(f"{NB}/{folder}/{namer('MCQ')}",usecols=["SEQN"]+mcq_cols,encoding="latin1")
    m=demo.merge(mcq,on="SEQN",how="left")
    cc=np.zeros(len(m)); anyv=np.zeros(len(m),bool)
    for c in mcq_cols:
        v=clean(m[c],(7,9)); cc+=(v==1).astype(float); anyv|=v.notna().to_numpy()
    out=pd.DataFrame(index=demo.index)
    out["_age"]=demo["RIDAGEYR"].to_numpy()
    out["comorb"]=np.where(anyv,cc,np.nan)
    # HUQ visits/hospitalization: only exist in pre
    if period=="pre":
        huq,_=pyreadstat.read_xport(f"{NB}/{folder}/{namer('HUQ')}",usecols=["SEQN","HUQ051","HUQ071"],encoding="latin1")
        h=demo.merge(huq,on="SEQN",how="left")
        out["hc_visits"]=clean(h["HUQ051"],(77,99)).to_numpy()
        h71=clean(h["HUQ071"],(7,9))
        out["hospitalized"]=np.where(h71==1,1.0,np.where(h71==2,0.0,np.nan))
    else:
        out["hc_visits"]=np.nan
        out["hospitalized"]=np.nan
    return out

def attach(period):
    ext=pd.read_parquet(f"{D}/data/ext_{period}.parquet")
    nf=build_newfeats(period).reindex(ext.index)
    a1=nf["_age"].to_numpy(float); a2=ext["age"].to_numpy(float)
    chk=np.mean((a1==a2)|(np.isnan(a1)&np.isnan(a2)))
    print(f"  {period}: age match={chk:.5f}  comorb NaN={nf['comorb'].isna().mean()*100:.1f}%  "
          f"hc_visits NaN={nf['hc_visits'].isna().mean()*100:.1f}%  hosp NaN={nf['hospitalized'].isna().mean()*100:.1f}%")
    assert chk>0.999, f"{period} align fail"
    ext=ext.reset_index(drop=True)
    for c in ["comorb","hc_visits","hospitalized"]: ext[c]=nf[c].to_numpy()
    return ext

NUM=["age","sleep_hrs","bmi","income_q","srh","oral_health"]
BIN=["partnered","living_alone","current_drinker","active","htn","dm","employed","food_insecure"]
CAT=["sex","edu","smoking"]

def make_ct(feN,feB,feC):
    return ColumnTransformer([
        ("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),feN),
        ("bin",Pipeline([("i",SimpleImputer(strategy="most_frequent"))]),feB),
        ("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore",drop="first"))]),feC)])

def oof_oracle(dd,feN,feB,feC):
    """Return per-row OOF proba for best of Logit/LGBM (by weighted AUROC)."""
    FE=feN+feB+feC; yy=dd["dep"].astype(int).values; w=dd["W"].to_numpy(float)
    results={}
    for nm,mk in [("Logit",lambda:LogisticRegression(max_iter=4000,C=0.3)),
                  ("LGBM",lambda:lgb.LGBMClassifier(num_leaves=15,min_child_samples=50,learning_rate=0.03,
                        n_estimators=400,reg_lambda=5.0,random_state=SEED,n_jobs=-1,verbose=-1))]:
        skf=StratifiedKFold(5,shuffle=True,random_state=SEED); oof=np.zeros(len(dd))
        for a,b in skf.split(dd,yy):
            c=make_ct(feN,feB,feC)
            Xa=c.fit_transform(dd.iloc[a][FE]); Xb=c.transform(dd.iloc[b][FE])
            # LGBM needs dense for passthrough NaN; ensure array
            g=mk().fit(np.asarray(Xa),yy[a],sample_weight=w[a])
            oof[b]=g.predict_proba(np.asarray(Xb))[:,1]
        results[nm]=(oof, roc_auc_score(yy,oof,sample_weight=w))
    best_nm=max(results,key=lambda k:results[k][1])
    return results[best_nm][0], results[best_nm][1], best_nm, {k:v[1] for k,v in results.items()}

print("Extract + align new domains:")
pre=attach("pre"); post=attach("post")
both=pd.concat([pre,post],ignore_index=True)
y=both["dep"].astype(int).values; w=both["W"].to_numpy(float)
print(f"  pooled N={len(both)}, dep rate={y.mean():.4f}")

print("\nUnivariate signal (pooled, complete cases):")
for c in ["comorb","hc_visits","hospitalized"]:
    x=both[c]; msk=x.notna()
    r=np.corrcoef(x[msk].astype(float),y[msk].astype(float))[0,1]
    print(f"  {c:<12} r={r:+.3f}  mean={x.mean():.3f}  NaN%={x.isna().mean()*100:.1f}")

print("\n"+"="*64)
print("Pooled 5-fold oracle (best of Logit/LGBM, weighted AUROC)")
print("="*64)
oof0,auc0,nm0,d0=oof_oracle(both,NUM,BIN,CAT)
print(f"  17 baseline           oracle={auc0:.4f}  (best={nm0}, {d0})")
oofC,aucC,nmC,dC=oof_oracle(both,NUM+["comorb"],BIN,CAT)
print(f"  +multimorbidity       oracle={aucC:.4f}  (best={nmC}, {dC})")
oofH,aucH,nmH,dH=oof_oracle(both,NUM+["hc_visits"],BIN+["hospitalized"],CAT)
print(f"  +healthcare util      oracle={aucH:.4f}  (best={nmH}, {dH})")
oofA,aucA,nmA,dA=oof_oracle(both,NUM+["comorb","hc_visits"],BIN+["hospitalized"],CAT)
print(f"  +both new domains     oracle={aucA:.4f}  (best={nmA}, {dA})")
print(f"\n  Delta (17 -> +both) = {aucA-auc0:+.4f}")

# Bootstrap CI on the delta between baseline OOF and full-domain OOF
print("\nBootstrap 95% CI on delta (paired, 1000 resamples, weighted AUROC):")
rng=np.random.default_rng(SEED)
n=len(both); deltas=[]
for _ in range(1000):
    idx=rng.integers(0,n,n)
    try:
        a0=roc_auc_score(y[idx],oof0[idx],sample_weight=w[idx])
        aA=roc_auc_score(y[idx],oofA[idx],sample_weight=w[idx])
        deltas.append(aA-a0)
    except ValueError:
        pass
deltas=np.array(deltas)
lo,hi=np.percentile(deltas,[2.5,97.5])
print(f"  delta point={aucA-auc0:+.4f}  mean={deltas.mean():+.4f}  95% CI=[{lo:+.4f}, {hi:+.4f}]  P(delta>0)={np.mean(deltas>0):.3f}")
