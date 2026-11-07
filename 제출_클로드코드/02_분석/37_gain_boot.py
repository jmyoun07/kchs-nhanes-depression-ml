# -*- coding: utf-8 -*-
"""핵심 주장 뒷받침 — 14→17 외부 AUROC 이득의 PSU군집 짝지음 부트스트랩 CI.
같은 리샘플에서 두 모델의 가중 AUROC 차이를 계산 → 차이의 95% CI가 0을 배제하는지."""
import sys, warnings, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
SEED=42; D="C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model"
NUM=["age","sleep_hrs","bmi","income_q","srh","oral_health"]; BIN=["partnered","living_alone","current_drinker","active","htn","dm","employed","food_insecure"]; CAT=["sex","edu","smoking"]
# 14변수 = 17 - {oral_health, employed, food_insecure}
NUM14=["age","sleep_hrs","bmi","income_q","srh"]; BIN14=["partnered","living_alone","current_drinker","active","htn","dm"]; CAT14=["sex","edu","smoking"]
LGB=dict(num_leaves=31,min_child_samples=1000,learning_rate=0.01,n_estimators=900,reg_alpha=1.0,reg_lambda=20.0,subsample=0.6,colsample_bytree=0.6,random_state=SEED,n_jobs=-1,verbose=-1)
cols=list(dict.fromkeys(NUM+BIN+CAT+["dep","W","strata","psu"]))
kchs=pd.read_parquet(f"{D}/data/ext_kchs.parquet",columns=[c for c in cols if c!="strata" and c!="psu"])
pre =pd.read_parquet(f"{D}/data/ext_pre.parquet", columns=cols)
post=pd.read_parquet(f"{D}/data/ext_post.parquet",columns=cols)
y=kchs["dep"].astype(int).values
tr,ho=train_test_split(np.arange(len(kchs)),train_size=500000,test_size=100000,stratify=y,random_state=SEED)
def ct(NU,BI,CA): return ColumnTransformer([("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),NU),("bin","passthrough",BI),("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore",drop="first"))]),CA)])
def fit(NU,BI,CA):
    fe=NU+BI+CA; c=ct(NU,BI,CA); m=lgb.LGBMClassifier(**LGB).fit(c.fit_transform(kchs.iloc[tr][fe]),y[tr]); return c,m,fe
c17,m17,fe17=fit(NUM,BIN,CAT); c14,m14,fe14=fit(NUM14,BIN14,CAT14)
def P(c,m,fe,d): return m.predict_proba(c.transform(d[fe]))[:,1]

def boot_diff(d,B=2000):
    yv=d["dep"].astype(int).values; w=d["W"].to_numpy(float)
    p17=P(c17,m17,fe17,d); p14=P(c14,m14,fe14,d)
    a17=roc_auc_score(yv,p17,sample_weight=w); a14=roc_auc_score(yv,p14,sample_weight=w)
    grp=(d["strata"].astype(str)+"|"+d["psu"].astype(str)).values; uid=pd.unique(grp)
    idx={u:np.where(grp==u)[0] for u in uid}; r=np.random.RandomState(SEED); diffs=[]
    for _ in range(B):
        ii=np.concatenate([idx[u] for u in r.choice(uid,len(uid),replace=True)])
        try:
            d17=roc_auc_score(yv[ii],p17[ii],sample_weight=w[ii]); d14=roc_auc_score(yv[ii],p14[ii],sample_weight=w[ii]); diffs.append(d17-d14)
        except: pass
    diffs=np.array(diffs); lo,hi=np.percentile(diffs,[2.5,97.5]); pval=2*min((diffs<=0).mean(),(diffs>=0).mean())
    return a14,a17,a17-a14,lo,hi,pval,(diffs>0).mean()
print("14→17 외부 AUROC 이득 (PSU군집 짝지음 부트스트랩, B=2000):\n")
rows=[]
for nm,d in [("NHANES pre",pre),("NHANES post",post)]:
    a14,a17,dd,lo,hi,pval,pgt=boot_diff(d)
    sig="유의(CI 0 배제)" if (lo>0 or hi<0) else "비유의"
    print(f"  {nm}: 14변수={a14:.4f} → 17변수={a17:.4f}  Δ={dd:+.4f}  95%CI[{lo:+.4f},{hi:+.4f}]  p≈{pval:.4f}  P(Δ>0)={pgt:.3f}  → {sig}")
    rows.append((nm,round(a14,4),round(a17,4),round(dd,4),round(lo,4),round(hi,4),round(pval,4),sig))
pd.DataFrame(rows,columns=["set","AUROC_14","AUROC_17","delta","CI_lo","CI_hi","p","verdict"]).to_csv(f"{D}/results/gain_14to17_bootstrap.csv",index=False,encoding="utf-8-sig")
print("\n저장: results/gain_14to17_bootstrap.csv")
