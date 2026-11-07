# -*- coding: utf-8 -*-
"""딥러닝 표형 모델 #1 — sklearn MLP(다층 신경망) 전이. 베이스라인(GBM) 대비."""
import sys, warnings, time, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
SEED=42; D="C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model"
NUM=["age","sleep_hrs","bmi","income_q","srh","oral_health"]; BIN=["partnered","living_alone","current_drinker","active","htn","dm","employed","food_insecure"]; CAT=["sex","edu","smoking"]; FE=NUM+BIN+CAT
kchs=pd.read_parquet(f"{D}/data/ext_kchs.parquet",columns=FE+["dep","W"])
pre =pd.read_parquet(f"{D}/data/ext_pre.parquet", columns=FE+["dep","W"])
post=pd.read_parquet(f"{D}/data/ext_post.parquet",columns=FE+["dep","W"])
y=kchs["dep"].astype(int).values
tr,ho=train_test_split(np.arange(len(kchs)),train_size=300000,test_size=100000,stratify=y,random_state=SEED)
def wauc(d,p): return roc_auc_score(d["dep"].astype(int),p,sample_weight=d["W"].astype(float))
ct=ColumnTransformer([("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),NUM),("bin",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("s",StandardScaler())]),BIN),("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore",drop="first"))]),CAT)])
Xtr=ct.fit_transform(kchs.iloc[tr][FE]);
Xtr=Xtr.toarray() if hasattr(Xtr,"toarray") else Xtr
Xho=ct.transform(kchs.iloc[ho][FE]); Xho=Xho.toarray() if hasattr(Xho,"toarray") else Xho
Xpre=ct.transform(pre[FE]); Xpre=Xpre.toarray() if hasattr(Xpre,"toarray") else Xpre
Xpost=ct.transform(post[FE]); Xpost=Xpost.toarray() if hasattr(Xpost,"toarray") else Xpost

# GBM 베이스라인(동일 300k)
LGB=dict(num_leaves=31,min_child_samples=1000,learning_rate=0.01,n_estimators=900,reg_alpha=1.0,reg_lambda=20.0,subsample=0.6,colsample_bytree=0.6,random_state=SEED,n_jobs=-1,verbose=-1)
c2=ColumnTransformer([("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),NUM),("bin","passthrough",BIN),("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore",drop="first"))]),CAT)])
mb=lgb.LGBMClassifier(**LGB).fit(c2.fit_transform(kchs.iloc[tr][FE]),y[tr])
bP=wauc(pre,mb.predict_proba(c2.transform(pre[FE]))[:,1]); bQ=wauc(post,mb.predict_proba(c2.transform(post[FE]))[:,1])
print(f"[BASE-GBM] pre={bP:.4f} post={bQ:.4f}",flush=True)

archs=[(64,),(128,64),(256,128,64),(128,128,64,32)]
best=None
for arch in archs:
    t=time.time()
    m=MLPClassifier(hidden_layer_sizes=arch,alpha=1e-3,batch_size=2048,learning_rate_init=1e-3,max_iter=60,early_stopping=True,n_iter_no_change=6,validation_fraction=0.1,random_state=SEED).fit(Xtr,y[tr])
    I=roc_auc_score(y[ho],m.predict_proba(Xho)[:,1]); P=wauc(pre,m.predict_proba(Xpre)[:,1]); Q=wauc(post,m.predict_proba(Xpost)[:,1])
    print(f"[MLP {str(arch):<20}] internal={I:.4f} pre={P:.4f} post={Q:.4f}  ({time.time()-t:.0f}s, iters={m.n_iter_})",flush=True)
    if best is None or I>best[0]: best=(I,P,Q,arch)
v="★UP" if (best[1]>bP+0.001 and best[2]>bQ+0.001) else ("~" if abs(best[1]-bP)<=0.003 and abs(best[2]-bQ)<=0.003 else "down")
print(f"\n[MLP 내부선택] arch={best[3]} internal={best[0]:.4f} pre={best[1]:.4f} post={best[2]:.4f}  판정={v}")
print(f"베이스라인 GBM pre={bP:.4f} post={bQ:.4f}")
