# -*- coding: utf-8 -*-
"""딥러닝 표형 모델 #4 — FT-Transformer(직접 구현) 전이. 베이스라인(GBM) 대비.
누수규칙: KCHS-train만 학습/조기종료, NHANES 평가만."""
import sys, warnings, time, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb, torch, torch.nn as nn
torch.manual_seed(42); np.random.seed(42)
SEED=42; D="C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model"
NUM=["age","sleep_hrs","bmi","income_q","srh","oral_health"]; BIN=["partnered","living_alone","current_drinker","active","htn","dm","employed","food_insecure"]; CAT=["sex","edu","smoking"]; FE=NUM+BIN+CAT
CONT=NUM+BIN
kchs=pd.read_parquet(f"{D}/data/ext_kchs.parquet",columns=FE+["dep","W"])
pre =pd.read_parquet(f"{D}/data/ext_pre.parquet", columns=FE+["dep","W"])
post=pd.read_parquet(f"{D}/data/ext_post.parquet",columns=FE+["dep","W"])
y=kchs["dep"].astype(int).values
tr,ho=train_test_split(np.arange(len(kchs)),train_size=300000,test_size=100000,stratify=y,random_state=SEED)
def wauc(d,p): return roc_auc_score(d["dep"].astype(int),p,sample_weight=d["W"].astype(float))
# 연속: impute+scale ; 범주: ordinal(+미지/결측=0)
numimp=Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]).fit(kchs.iloc[tr][CONT])
orde=OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1,encoded_missing_value=-1).fit(kchs.iloc[tr][CAT].astype(str))
cat_dims=[int(len(c))+2 for c in orde.categories_]   # +미지/결측
def make(df):
    Xc=numimp.transform(df[CONT]).astype(np.float32)
    Xk=(orde.transform(df[CAT].astype(str))+1).astype(np.int64)  # shift so -1→0
    return Xc,Xk
Xc_tr,Xk_tr=make(kchs.iloc[tr]); Xc_ho,Xk_ho=make(kchs.iloc[ho]); Xc_pre,Xk_pre=make(pre); Xc_post,Xk_post=make(post)
ytr=y[tr].astype(np.float32)
vi=np.random.RandomState(SEED).choice(len(tr),40000,replace=False); mask=np.ones(len(tr),bool); mask[vi]=False

# GBM 베이스라인
LGB=dict(num_leaves=31,min_child_samples=1000,learning_rate=0.01,n_estimators=900,reg_alpha=1.0,reg_lambda=20.0,subsample=0.6,colsample_bytree=0.6,random_state=SEED,n_jobs=-1,verbose=-1)
c2=ColumnTransformer([("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),NUM),("bin","passthrough",BIN),("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore",drop="first"))]),CAT)])
mb=lgb.LGBMClassifier(**LGB).fit(c2.fit_transform(kchs.iloc[tr][FE]),y[tr])
bP=wauc(pre,mb.predict_proba(c2.transform(pre[FE]))[:,1]); bQ=wauc(post,mb.predict_proba(c2.transform(post[FE]))[:,1])
print(f"[BASE-GBM] pre={bP:.4f} post={bQ:.4f}",flush=True)

class FTT(nn.Module):
    def __init__(s,n_cont,cat_dims,d=32,heads=4,blocks=3,drop=0.1):
        super().__init__()
        s.Wnum=nn.Parameter(torch.randn(n_cont,d)*0.02); s.bnum=nn.Parameter(torch.zeros(n_cont,d))
        s.embs=nn.ModuleList([nn.Embedding(cd,d) for cd in cat_dims])
        s.cls=nn.Parameter(torch.randn(1,1,d)*0.02)
        enc=nn.TransformerEncoderLayer(d,heads,dim_feedforward=2*d,dropout=drop,batch_first=True,activation="gelu")
        s.tr=nn.TransformerEncoder(enc,blocks); s.norm=nn.LayerNorm(d); s.head=nn.Linear(d,1)
    def forward(s,xc,xk):
        B=xc.shape[0]
        tnum=xc.unsqueeze(-1)*s.Wnum+s.bnum                     # [B,n_cont,d]
        tcat=torch.stack([e(xk[:,i]) for i,e in enumerate(s.embs)],1)  # [B,n_cat,d]
        cls=s.cls.expand(B,-1,-1)
        z=torch.cat([cls,tnum,tcat],1); z=s.tr(z)
        return s.head(s.norm(z[:,0])).squeeze(-1)
try:
    t=time.time(); dev="cpu"
    net=FTT(len(CONT),cat_dims).to(dev)
    opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-5)
    pos=ytr.mean(); lossf=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(1-pos)/pos]))
    Xc_t=torch.tensor(Xc_tr); Xk_t=torch.tensor(Xk_tr); yt=torch.tensor(ytr)
    idx_t=np.where(mask)[0]; idx_v=vi
    bs=4096; best=(0,None); bad=0
    for ep in range(50):
        net.train(); perm=np.random.permutation(idx_t)
        for i in range(0,len(perm),bs):
            b=perm[i:i+bs]; opt.zero_grad(); out=net(Xc_t[b],Xk_t[b]); loss=lossf(out,yt[b]); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): av=roc_auc_score(ytr[idx_v],torch.sigmoid(net(Xc_t[idx_v],Xk_t[idx_v])).numpy())
        if av>best[0]+1e-4: best=(av,{k:v.clone() for k,v in net.state_dict().items()}); bad=0
        else:
            bad+=1
            if bad>=8: break
    net.load_state_dict(best[1]); net.eval()
    def pr(Xc,Xk):
        with torch.no_grad(): return torch.sigmoid(net(torch.tensor(Xc),torch.tensor(Xk))).numpy()
    I=roc_auc_score(y[ho],pr(Xc_ho,Xk_ho)); P=wauc(pre,pr(Xc_pre,Xk_pre)); Q=wauc(post,pr(Xc_post,Xk_post))
    v="★UP" if (P>bP+0.001 and Q>bQ+0.001) else ("~" if abs(P-bP)<=0.003 and abs(Q-bQ)<=0.003 else "down")
    print(f"[FT-Transformer] best_val={best[0]:.4f} internal={I:.4f} pre={P:.4f} post={Q:.4f}  판정={v}  ({time.time()-t:.0f}s, ep={ep+1})",flush=True)
except Exception as e:
    import traceback; print(f"[FT] ERROR {e}\n{traceback.format_exc()[:500]}",flush=True)
print(f"\n베이스라인 GBM pre={bP:.4f} post={bQ:.4f}. 완료.")
