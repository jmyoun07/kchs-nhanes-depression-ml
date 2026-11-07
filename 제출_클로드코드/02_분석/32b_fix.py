# -*- coding: utf-8 -*-
"""M5(WoE 타깃인코딩)·M7(내부최적 튜닝 전이) 버그수정 재실행."""
import sys, warnings, time, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
SEED=42; D="C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model"
NUM=["age","sleep_hrs","bmi","income_q","srh","oral_health"]; BIN=["partnered","living_alone","current_drinker","active","htn","dm","employed","food_insecure"]; CAT=["sex","edu","smoking"]; FE=NUM+BIN+CAT
LGB=dict(num_leaves=31,min_child_samples=1000,learning_rate=0.01,n_estimators=900,reg_alpha=1.0,reg_lambda=20.0,subsample=0.6,colsample_bytree=0.6,random_state=SEED,n_jobs=-1,verbose=-1)
kchs=pd.read_parquet(f"{D}/data/ext_kchs.parquet",columns=FE+["dep","W","year"]).reset_index(drop=True)
pre =pd.read_parquet(f"{D}/data/ext_pre.parquet", columns=FE+["dep","W"])
post=pd.read_parquet(f"{D}/data/ext_post.parquet",columns=FE+["dep","W"])
y=kchs["dep"].astype(int).values
tr,ho=train_test_split(np.arange(len(kchs)),train_size=300000,test_size=100000,stratify=y,random_state=SEED)
def wauc(d,p): return roc_auc_score(d["dep"].astype(int),p,sample_weight=d["W"].astype(float))
bP,bQ=0.7694,0.7685

# ── M5 WoE (groupby 기반, OOF) ──
try:
    t=time.time()
    numimp=Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]).fit(kchs.iloc[tr][NUM])
    tri=kchs.iloc[tr].reset_index(drop=True); ytr=y[tr]
    def woe_map(vser,yv,alpha=0.5):
        d=pd.DataFrame({"v":vser.astype(str).values,"y":yv}); pos=yv.sum(); neg=len(yv)-pos; K=max(d["v"].nunique(),1)
        g=d.groupby("v")["y"].agg(["sum","count"]);
        return (np.log(((g["sum"]+alpha)/(pos+alpha*K))/(((g["count"]-g["sum"])+alpha)/(neg+alpha*K)))).to_dict()
    # OOF train encoding
    Xw=np.zeros((len(tri),len(CAT)))
    skf=StratifiedKFold(5,shuffle=True,random_state=SEED)
    for a,b in skf.split(tri,ytr):
        for j,c in enumerate(CAT):
            m=woe_map(tri.iloc[a][c],ytr[a]); Xw[b,j]=tri.iloc[b][c].astype(str).map(m).fillna(0.0).values
    gmap={c:woe_map(tri[c],ytr) for c in CAT}
    def feats(df):
        w=np.column_stack([df[c].astype(str).map(gmap[c]).fillna(0.0).values for c in CAT])
        return np.hstack([numimp.transform(df[NUM]),df[BIN].to_numpy(float),w])
    Xtr=np.hstack([numimp.transform(tri[NUM]),tri[BIN].to_numpy(float),Xw])
    m=lgb.LGBMClassifier(**LGB).fit(Xtr,ytr)
    I=roc_auc_score(y[ho],m.predict_proba(feats(kchs.iloc[ho]))[:,1]); P=wauc(pre,m.predict_proba(feats(pre))[:,1]); Q=wauc(post,m.predict_proba(feats(post))[:,1])
    v="UP" if (P>bP+0.001 and Q>bQ+0.001) else ("~" if abs(P-bP)<=0.002 and abs(Q-bQ)<=0.002 else "down")
    print(f"[M5 woe] internal={I:.4f} pre={P:.4f} post={Q:.4f}  판정={v}  ({time.time()-t:.0f}s)",flush=True)
except Exception as e:
    import traceback; print(f"[M5] ERROR {e}\n{traceback.format_exc()[:500]}",flush=True)

# ── M7 내부최적 튜닝의 외부전이 (iloc 사용) ──
try:
    t=time.time(); rng=np.random.RandomState(SEED)
    sub=rng.choice(tr,200000,replace=False); vi=rng.choice(ho,50000,replace=False)
    def ct(): return ColumnTransformer([("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),NUM),("bin","passthrough",BIN),("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore",drop="first"))]),CAT)])
    c=ct(); Xs=c.fit_transform(kchs.iloc[sub][FE]); Xv=c.transform(kchs.iloc[vi][FE]); ys=y[sub]
    best=None
    for i in range(16):
        cfg=dict(num_leaves=int(rng.choice([15,31,63,127])),min_child_samples=int(rng.choice([200,500,1000,4000])),learning_rate=float(rng.choice([0.01,0.02,0.05])),n_estimators=int(rng.choice([400,700,1000])),reg_alpha=float(rng.choice([0,1,5])),reg_lambda=float(rng.choice([5,20,50])),subsample=float(rng.choice([0.6,0.8,1.0])),colsample_bytree=float(rng.choice([0.6,0.8,1.0])),random_state=SEED,n_jobs=-1,verbose=-1)
        m=lgb.LGBMClassifier(**cfg).fit(Xs,ys); a=roc_auc_score(y[vi],m.predict_proba(Xv)[:,1])
        if best is None or a>best[0]: best=(a,cfg)
    c=ct(); Xtr=c.fit_transform(kchs.iloc[tr][FE]); m=lgb.LGBMClassifier(**best[1]).fit(Xtr,y[tr])
    I=roc_auc_score(y[ho],m.predict_proba(c.transform(kchs.iloc[ho][FE]))[:,1]); P=wauc(pre,m.predict_proba(c.transform(pre[FE]))[:,1]); Q=wauc(post,m.predict_proba(c.transform(post[FE]))[:,1])
    v="UP" if (P>bP+0.001 and Q>bQ+0.001) else ("~" if abs(P-bP)<=0.002 and abs(Q-bQ)<=0.002 else "down")
    print(f"[M7 internal-tuned] 내부val={best[0]:.4f} → internal={I:.4f} pre={P:.4f} post={Q:.4f}  판정={v}  ({time.time()-t:.0f}s)",flush=True)
    print(f"     선택 cfg={best[1]}",flush=True)
except Exception as e:
    import traceback; print(f"[M7] ERROR {e}\n{traceback.format_exc()[:500]}",flush=True)
print("완료.")
