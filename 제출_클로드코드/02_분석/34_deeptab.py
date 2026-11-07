# -*- coding: utf-8 -*-
"""딥러닝 표형 모델 #2,#3 — TabNet + torch 심층 MLP 전이. 베이스라인(GBM) 대비.
누수규칙 준수: KCHS-train만 학습/조기종료, NHANES는 평가만."""
import sys, warnings, time, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb, torch
torch.manual_seed(42); np.random.seed(42)
SEED=42; D="C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model"
NUM=["age","sleep_hrs","bmi","income_q","srh","oral_health"]; BIN=["partnered","living_alone","current_drinker","active","htn","dm","employed","food_insecure"]; CAT=["sex","edu","smoking"]; FE=NUM+BIN+CAT
kchs=pd.read_parquet(f"{D}/data/ext_kchs.parquet",columns=FE+["dep","W"])
pre =pd.read_parquet(f"{D}/data/ext_pre.parquet", columns=FE+["dep","W"])
post=pd.read_parquet(f"{D}/data/ext_post.parquet",columns=FE+["dep","W"])
y=kchs["dep"].astype(int).values
tr,ho=train_test_split(np.arange(len(kchs)),train_size=300000,test_size=100000,stratify=y,random_state=SEED)
def wauc(d,p): return roc_auc_score(d["dep"].astype(int),p,sample_weight=d["W"].astype(float))
ct=ColumnTransformer([("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),NUM),("bin",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("s",StandardScaler())]),BIN),("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore",drop="first"))]),CAT)])
def dense(X): return np.asarray(X.toarray() if hasattr(X,"toarray") else X,dtype=np.float32)
Xtr=dense(ct.fit_transform(kchs.iloc[tr][FE])); Xho=dense(ct.transform(kchs.iloc[ho][FE]))
Xpre=dense(ct.transform(pre[FE])); Xpost=dense(ct.transform(post[FE]))
ytr=y[tr].astype(np.int64); yho=y[ho].astype(np.int64)
# 내부 조기종료용 valid (KCHS train에서 분리)
vi=np.random.RandomState(SEED).choice(len(Xtr),40000,replace=False); mask=np.ones(len(Xtr),bool); mask[vi]=False
Xt,yt=Xtr[mask],ytr[mask]; Xv,yv=Xtr[vi],ytr[vi]

# GBM 베이스라인
LGB=dict(num_leaves=31,min_child_samples=1000,learning_rate=0.01,n_estimators=900,reg_alpha=1.0,reg_lambda=20.0,subsample=0.6,colsample_bytree=0.6,random_state=SEED,n_jobs=-1,verbose=-1)
c2=ColumnTransformer([("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),NUM),("bin","passthrough",BIN),("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore",drop="first"))]),CAT)])
mb=lgb.LGBMClassifier(**LGB).fit(c2.fit_transform(kchs.iloc[tr][FE]),y[tr])
bP=wauc(pre,mb.predict_proba(c2.transform(pre[FE]))[:,1]); bQ=wauc(post,mb.predict_proba(c2.transform(post[FE]))[:,1])
print(f"[BASE-GBM] pre={bP:.4f} post={bQ:.4f}",flush=True)
def vd(P,Q): return "★UP" if (P>bP+0.001 and Q>bQ+0.001) else ("~" if abs(P-bP)<=0.003 and abs(Q-bQ)<=0.003 else "down")

# ── TabNet ──
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    t=time.time()
    clf=TabNetClassifier(n_d=16,n_a=16,n_steps=4,gamma=1.5,seed=SEED,verbose=0,optimizer_params=dict(lr=2e-2))
    clf.fit(Xt,yt,eval_set=[(Xv,yv)],eval_metric=["auc"],max_epochs=50,patience=8,batch_size=8192,virtual_batch_size=1024)
    I=roc_auc_score(yho,clf.predict_proba(Xho)[:,1]); P=wauc(pre,clf.predict_proba(Xpre)[:,1]); Q=wauc(post,clf.predict_proba(Xpost)[:,1])
    print(f"[TabNet] internal={I:.4f} pre={P:.4f} post={Q:.4f}  판정={vd(P,Q)}  ({time.time()-t:.0f}s)",flush=True)
except Exception as e:
    import traceback; print(f"[TabNet] ERROR {e}\n{traceback.format_exc()[:400]}",flush=True)

# ── torch 심층 MLP (BatchNorm+Dropout) ──
try:
    import torch.nn as nn
    t=time.time(); dev="cpu"
    class Net(nn.Module):
        def __init__(s,d):
            super().__init__()
            s.net=nn.Sequential(nn.Linear(d,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(0.3),
                                nn.Linear(256,128),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(0.3),
                                nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
        def forward(s,x): return s.net(x).squeeze(-1)
    net=Net(Xt.shape[1]).to(dev)
    opt=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-4)
    pos=ytr.mean(); w=torch.tensor([(1-pos)/pos],dtype=torch.float32)
    lossf=nn.BCEWithLogitsLoss(pos_weight=w)
    Xt_t=torch.tensor(Xt); yt_t=torch.tensor(yt,dtype=torch.float32); Xv_t=torch.tensor(Xv)
    bs=4096; n=len(Xt_t); best=(0,None); bad=0
    for ep in range(60):
        net.train(); perm=torch.randperm(n)
        for i in range(0,n,bs):
            idx=perm[i:i+bs]; opt.zero_grad(); out=net(Xt_t[idx]); loss=lossf(out,yt_t[idx]); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): av=roc_auc_score(yv,torch.sigmoid(net(Xv_t)).numpy())
        if av>best[0]+1e-4: best=(av,{k:v.clone() for k,v in net.state_dict().items()}); bad=0
        else:
            bad+=1
            if bad>=8: break
    net.load_state_dict(best[1]); net.eval()
    with torch.no_grad():
        pI=torch.sigmoid(net(torch.tensor(Xho))).numpy(); pP=torch.sigmoid(net(torch.tensor(Xpre))).numpy(); pQ=torch.sigmoid(net(torch.tensor(Xpost))).numpy()
    I=roc_auc_score(yho,pI); P=wauc(pre,pP); Q=wauc(post,pQ)
    print(f"[torch-MLP] best_val={best[0]:.4f} internal={I:.4f} pre={P:.4f} post={Q:.4f}  판정={vd(P,Q)}  ({time.time()-t:.0f}s)",flush=True)
except Exception as e:
    import traceback; print(f"[torch-MLP] ERROR {e}\n{traceback.format_exc()[:400]}",flush=True)
print(f"\n베이스라인 GBM pre={bP:.4f} post={bQ:.4f}. 완료.")
