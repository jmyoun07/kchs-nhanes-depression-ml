# -*- coding: utf-8 -*-
"""한도로 못 돌린 신규방법 7종을 로컬에서 직접 실행. 각 방법은 동일 표본 in-run 베이스라인과 비교.
누수 규칙: NHANES 라벨은 훈련/선택/튜닝/변환적합에 절대 미사용(오라클 진단 제외)."""
import sys, warnings, time, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import lightgbm as lgb
SEED=42; D="C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model"
NUM=["age","sleep_hrs","bmi","income_q","srh","oral_health"]
BIN=["partnered","living_alone","current_drinker","active","htn","dm","employed","food_insecure"]
CAT=["sex","edu","smoking"]; FE=NUM+BIN+CAT
LGB=dict(num_leaves=31,min_child_samples=1000,learning_rate=0.01,n_estimators=900,reg_alpha=1.0,reg_lambda=20.0,subsample=0.6,colsample_bytree=0.6,random_state=SEED,n_jobs=-1,verbose=-1)
print("데이터 로드...",flush=True)
kchs=pd.read_parquet(f"{D}/data/ext_kchs.parquet",columns=FE+["dep","W","year"])
pre =pd.read_parquet(f"{D}/data/ext_pre.parquet", columns=FE+["dep","W"])
post=pd.read_parquet(f"{D}/data/ext_post.parquet",columns=FE+["dep","W"])
y=kchs["dep"].astype(int).values
NTR=300000
tr,ho=train_test_split(np.arange(len(kchs)),train_size=NTR,test_size=100000,stratify=y,random_state=SEED)
def wauc(d,p): return roc_auc_score(d["dep"].astype(int),p,sample_weight=d["W"].astype(float))
def ct(): return ColumnTransformer([("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),NUM),("bin","passthrough",BIN),("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore",drop="first"))]),CAT)])
def fit_eval(Xtr,ytr,Xho,Xpre,Xpost,cfg=LGB,**fitkw):
    m=lgb.LGBMClassifier(**cfg).fit(Xtr,ytr,**fitkw)
    return (roc_auc_score(y[ho],m.predict_proba(Xho)[:,1]),
            wauc(pre,m.predict_proba(Xpre)[:,1]), wauc(post,m.predict_proba(Xpost)[:,1]))

def baseline():
    c=ct(); Xtr=c.fit_transform(kchs.iloc[tr][FE])
    return fit_eval(Xtr,y[tr],c.transform(kchs.iloc[ho][FE]),c.transform(pre[FE]),c.transform(post[FE]))
print("\n[BASE] 300k 베이스라인...",flush=True); t=time.time()
bI,bP,bQ=baseline(); print(f"[BASE] internal={bI:.4f} pre={bP:.4f} post={bQ:.4f}  ({time.time()-t:.0f}s)",flush=True)
def verdict(P,Q): return "★UP" if (P>bP+0.001 and Q>bQ+0.001) else ("~" if abs(P-bP)<=0.002 and abs(Q-bQ)<=0.002 else "down")
res=[("baseline",bI,bP,bQ,"ref")]

# ── M1 단조제약 ──
try:
    t=time.time()
    sign=[]
    for c in NUM:
        x=pd.to_numeric(kchs.iloc[tr][c],errors="coerce").values; m=~np.isnan(x)
        r=spearmanr(x[m],y[tr][m]).correlation; sign.append(1 if r>0.02 else (-1 if r<-0.02 else 0))
    numimp=Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())])
    orde=OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1,encoded_missing_value=-1)
    Xn_tr=numimp.fit_transform(kchs.iloc[tr][NUM]); Xc_tr=orde.fit_transform(kchs.iloc[tr][CAT].astype(str))
    Xb_tr=kchs.iloc[tr][BIN].to_numpy(float)
    def build(df):
        return np.hstack([numimp.transform(df[NUM]),df[BIN].to_numpy(float),orde.transform(df[CAT].astype(str))])
    Xtr=np.hstack([Xn_tr,Xb_tr,Xc_tr]); catidx=[14,15,16]
    mc=sign+[0]*8+[0]*3
    for tag,cons in [("mono",mc),("nomono",None)]:
        cfg=dict(LGB);
        if cons is not None: cfg["monotone_constraints"]=cons
        m=lgb.LGBMClassifier(**cfg).fit(Xtr,y[tr],categorical_feature=catidx)
        I=roc_auc_score(y[ho],m.predict_proba(build(kchs.iloc[ho]))[:,1]); P=wauc(pre,m.predict_proba(build(pre))[:,1]); Q=wauc(post,m.predict_proba(build(post))[:,1])
        if tag=="mono": res.append(("M1 monotonic",I,P,Q,verdict(P,Q)))
        print(f"[M1] {tag}: internal={I:.4f} pre={P:.4f} post={Q:.4f}",flush=True)
    print(f"[M1] done ({time.time()-t:.0f}s) sign={dict(zip(NUM,sign))}",flush=True)
except Exception as e: print(f"[M1] ERROR {e}",flush=True); res.append(("M1 monotonic",None,None,None,"err"))

# ── M5 WoE 타깃인코딩 ──
try:
    t=time.time()
    def woe_maps(dfx,yy,alpha=0.5):
        maps={};
        for c in CAT:
            s=dfx[c].astype(str).values; mp={}
            pos=yy.sum(); neg=len(yy)-pos; K=len(np.unique(s))
            for v in np.unique(s):
                mk=s==v; np_=yy[mk].sum(); nn=mk.sum()-np_
                mp[v]=np.log(((np_+alpha)/(pos+alpha*K))/((nn+alpha)/(neg+alpha*K)))
            maps[c]=(mp, np.log(((pos)+1e-9)/((neg)+1e-9))*0)  # default 0
        return maps
    def woe_apply(df,maps):
        out=np.zeros((len(df),len(CAT)))
        for j,c in enumerate(CAT):
            mp=maps[c][0]; out[:,j]=[mp.get(str(v),0.0) for v in df[c].values]
        return out
    # OOF for train
    skf=StratifiedKFold(5,shuffle=True,random_state=SEED)
    numimp=Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]).fit(kchs.iloc[tr][NUM])
    Xtr_woe=np.zeros((len(tr),len(CAT)))
    tri=kchs.iloc[tr].reset_index(drop=True)
    for a,b in skf.split(tri,y[tr]):
        mp=woe_maps(tri.iloc[a],y[tr][a]); Xtr_woe[b]=woe_apply(tri.iloc[b],mp)
    gmap=woe_maps(tri,y[tr])
    def feats(df,woe): return np.hstack([numimp.transform(df[NUM]),df[BIN].to_numpy(float),woe])
    Xtr=feats(tri,Xtr_woe)
    m=lgb.LGBMClassifier(**LGB).fit(Xtr,y[tr])
    I=roc_auc_score(y[ho],m.predict_proba(feats(kchs.iloc[ho],woe_apply(kchs.iloc[ho],gmap)))[:,1])
    P=wauc(pre,m.predict_proba(feats(pre,woe_apply(pre,gmap)))[:,1]); Q=wauc(post,m.predict_proba(feats(post,woe_apply(post,gmap)))[:,1])
    res.append(("M5 woe-encode",I,P,Q,verdict(P,Q))); print(f"[M5] internal={I:.4f} pre={P:.4f} post={Q:.4f} ({time.time()-t:.0f}s)",flush=True)
except Exception as e: print(f"[M5] ERROR {e}",flush=True); res.append(("M5 woe-encode",None,None,None,"err"))

# ── M6 결측처리 변형 ──
try:
    t=time.time()
    miss_feats=[c for c in FE if kchs.iloc[tr][c].isna().mean()>0.01 or pre[c].isna().mean()>0.01 or post[c].isna().mean()>0.01]
    # (a) 결측지시자 추가
    c0=ct(); Xtr0=c0.fit_transform(kchs.iloc[tr][FE])
    def add_ind(df,X):
        ind=np.column_stack([df[c].isna().astype(float).values for c in miss_feats])
        return np.hstack([X,ind])
    I,P,Q=fit_eval(add_ind(kchs.iloc[tr],Xtr0),y[tr],add_ind(kchs.iloc[ho],c0.transform(kchs.iloc[ho][FE])),add_ind(pre,c0.transform(pre[FE])),add_ind(post,c0.transform(post[FE])))
    res.append(("M6a miss-indicator",I,P,Q,verdict(P,Q))); print(f"[M6a] internal={I:.4f} pre={P:.4f} post={Q:.4f}",flush=True)
    # (b) 명시적 채움(트리네이티브 대신 BIN도 임퓨트)
    ctb=ColumnTransformer([("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),NUM),("bin",Pipeline([("i",SimpleImputer(strategy="most_frequent"))]),BIN),("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore",drop="first"))]),CAT)])
    Xtr=ctb.fit_transform(kchs.iloc[tr][FE])
    I,P,Q=fit_eval(Xtr,y[tr],ctb.transform(kchs.iloc[ho][FE]),ctb.transform(pre[FE]),ctb.transform(post[FE]))
    res.append(("M6b explicit-fill",I,P,Q,verdict(P,Q))); print(f"[M6b] internal={I:.4f} pre={P:.4f} post={Q:.4f} ({time.time()-t:.0f}s) missfeats={miss_feats}",flush=True)
except Exception as e: print(f"[M6] ERROR {e}",flush=True); res.append(("M6 missing",None,None,None,"err"))

# ── M2 전이안정 변수선택 (PSI, 라벨 미사용) ──
try:
    t=time.time()
    extboth=pd.concat([pre[FE],post[FE]],ignore_index=True)
    def psi(a,b,bins=10):
        a=pd.to_numeric(a,errors="coerce").dropna(); b=pd.to_numeric(b,errors="coerce").dropna()
        if a.nunique()<=1: return 0.0
        q=np.unique(np.quantile(a,np.linspace(0,1,bins+1)))
        if len(q)<3:
            cats=np.union1d(a.unique(),b.unique()); ea=np.array([(a==c).mean() for c in cats]); eb=np.array([(b==c).mean() for c in cats])
        else:
            ea=np.histogram(a,q)[0]/len(a); eb=np.histogram(b,q)[0]/len(b)
        ea=np.clip(ea,1e-4,None); eb=np.clip(eb,1e-4,None); return float(np.sum((eb-ea)*np.log(eb/ea)))
    psis={c:psi(kchs.iloc[tr][c],extboth[c]) for c in FE}
    order=sorted(FE,key=lambda c:-psis[c])
    # 내부 hold-out으로 k 선택
    best=None
    for k in [0,1,2,3]:
        drop=set(order[:k]); useN=[c for c in NUM if c not in drop]; useB=[c for c in BIN if c not in drop]; useC=[c for c in CAT if c not in drop]
        cc=ColumnTransformer([("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),useN),("bin","passthrough",useB),("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore",drop="first"))]),useC)])
        fe2=useN+useB+useC; Xtr=cc.fit_transform(kchs.iloc[tr][fe2])
        I=roc_auc_score(y[ho],lgb.LGBMClassifier(**LGB).fit(Xtr,y[tr]).predict_proba(cc.transform(kchs.iloc[ho][fe2]))[:,1])
        m=lgb.LGBMClassifier(**LGB).fit(Xtr,y[tr]); P=wauc(pre,m.predict_proba(cc.transform(pre[fe2]))[:,1]); Q=wauc(post,m.predict_proba(cc.transform(post[fe2]))[:,1])
        print(f"[M2] k={k} drop={sorted(drop)} internal={I:.4f} pre={P:.4f} post={Q:.4f}",flush=True)
        if best is None or I>best[0]: best=(I,P,Q,k)
    res.append((f"M2 stable-select(k*={best[3]})",best[0],best[1],best[2],verdict(best[1],best[2])))
    print(f"[M2] internal-선택 k*={best[3]}  ({time.time()-t:.0f}s)  PSI={ {c:round(psis[c],3) for c in order[:4]} }",flush=True)
except Exception as e: print(f"[M2] ERROR {e}",flush=True); res.append(("M2 stable-select",None,None,None,"err"))

# ── M4 연도별 앙상블 ──
try:
    t=time.time()
    years=sorted(kchs["year"].dropna().unique()); per=70000
    pp=np.zeros(len(pre)); qq=np.zeros(len(post)); hh=np.zeros(len(ho)); tot=0; parts=[]
    for yr in years:
        idx=kchs.index[(kchs["year"]==yr)]
        idx=np.array(idx);
        if len(idx)>per: idx=np.random.RandomState(SEED).choice(idx,per,replace=False)
        parts.append(idx); tot+=len(idx)
        c=ct(); Xtr=c.fit_transform(kchs.loc[idx][FE]); yy=kchs.loc[idx,"dep"].astype(int).values
        m=lgb.LGBMClassifier(**LGB).fit(Xtr,yy)
        pp+=m.predict_proba(c.transform(pre[FE]))[:,1]; qq+=m.predict_proba(c.transform(post[FE]))[:,1]; hh+=m.predict_proba(c.transform(kchs.iloc[ho][FE]))[:,1]
    pp/=len(years); qq/=len(years); hh/=len(years)
    I=roc_auc_score(y[ho],hh); P=wauc(pre,pp); Q=wauc(post,qq)
    res.append(("M4 year-ensemble",I,P,Q,verdict(P,Q))); print(f"[M4] {len(years)}년 앙상블 (총{tot}행) internal={I:.4f} pre={P:.4f} post={Q:.4f} ({time.time()-t:.0f}s)",flush=True)
    # 공정 비교: 동일 총행 pooled
    allidx=np.concatenate(parts); c=ct(); Xtr=c.fit_transform(kchs.loc[allidx][FE]); yy=kchs.loc[allidx,"dep"].astype(int).values
    m=lgb.LGBMClassifier(**LGB).fit(Xtr,yy); print(f"[M4] 동일행 pooled 비교 pre={wauc(pre,m.predict_proba(c.transform(pre[FE]))[:,1]):.4f} post={wauc(post,m.predict_proba(c.transform(post[FE]))[:,1]):.4f}",flush=True)
except Exception as e: print(f"[M4] ERROR {e}",flush=True); res.append(("M4 year-ensemble",None,None,None,"err"))

# ── M3 LOYO 강건 하이퍼파라미터 ──
try:
    t=time.time(); years=sorted(kchs["year"].dropna().unique())
    grid=[dict(num_leaves=15,min_child_samples=4000,reg_lambda=50),dict(num_leaves=31,min_child_samples=1000,reg_lambda=20),dict(num_leaves=63,min_child_samples=500,reg_lambda=5),dict(num_leaves=15,min_child_samples=2000,reg_lambda=100)]
    def loyo_worst(extra):
        cfg=dict(LGB); cfg.update(extra); cfg["n_estimators"]=400; worst=1.0
        for yr in years:
            tri=kchs.index[(kchs["year"]!=yr)]; tei=kchs.index[(kchs["year"]==yr)]
            tri=np.array(tri);
            if len(tri)>60000: tri=np.random.RandomState(SEED).choice(tri,60000,replace=False)
            c=ct(); Xt=c.fit_transform(kchs.loc[tri][FE]); yt=kchs.loc[tri,"dep"].astype(int).values
            m=lgb.LGBMClassifier(**cfg).fit(Xt,yt)
            dte=kchs.loc[tei]; a=roc_auc_score(dte["dep"].astype(int),m.predict_proba(c.transform(dte[FE]))[:,1])
            worst=min(worst,a)
        return worst
    scored=sorted([(loyo_worst(g),g) for g in grid],key=lambda z:-z[0])
    bestcfg=dict(LGB); bestcfg.update(scored[0][1])
    c=ct(); Xtr=c.fit_transform(kchs.iloc[tr][FE]); m=lgb.LGBMClassifier(**bestcfg).fit(Xtr,y[tr])
    I=roc_auc_score(y[ho],m.predict_proba(c.transform(kchs.iloc[ho][FE]))[:,1]); P=wauc(pre,m.predict_proba(c.transform(pre[FE]))[:,1]); Q=wauc(post,m.predict_proba(c.transform(post[FE]))[:,1])
    res.append(("M3 loyo-hparams",I,P,Q,verdict(P,Q))); print(f"[M3] best worst-year={scored[0][0]:.4f} cfg={scored[0][1]} → internal={I:.4f} pre={P:.4f} post={Q:.4f} ({time.time()-t:.0f}s)",flush=True)
except Exception as e: print(f"[M3] ERROR {e}",flush=True); res.append(("M3 loyo-hparams",None,None,None,"err"))

# ── M7 내부최적 튜닝의 외부전이 ──
try:
    t=time.time(); rng=np.random.RandomState(SEED)
    sub=rng.choice(tr,200000,replace=False); vi=rng.choice(ho,50000,replace=False)
    c=ct(); Xs=c.fit_transform(kchs.loc[sub][FE]); Xv=c.transform(kchs.loc[vi][FE]); ys=kchs.loc[sub,"dep"].astype(int).values
    best=None
    for i in range(16):
        cfg=dict(num_leaves=int(rng.choice([15,31,63,127])),min_child_samples=int(rng.choice([200,500,1000,4000])),learning_rate=float(rng.choice([0.01,0.02,0.05])),n_estimators=int(rng.choice([400,700,1000])),reg_alpha=float(rng.choice([0,1,5])),reg_lambda=float(rng.choice([5,20,50])),subsample=float(rng.choice([0.6,0.8,1.0])),colsample_bytree=float(rng.choice([0.6,0.8,1.0])),random_state=SEED,n_jobs=-1,verbose=-1)
        m=lgb.LGBMClassifier(**cfg).fit(Xs,ys); a=roc_auc_score(y[vi],m.predict_proba(Xv)[:,1])
        if best is None or a>best[0]: best=(a,cfg)
    c=ct(); Xtr=c.fit_transform(kchs.iloc[tr][FE]); m=lgb.LGBMClassifier(**best[1]).fit(Xtr,y[tr])
    I=roc_auc_score(y[ho],m.predict_proba(c.transform(kchs.iloc[ho][FE]))[:,1]); P=wauc(pre,m.predict_proba(c.transform(pre[FE]))[:,1]); Q=wauc(post,m.predict_proba(c.transform(post[FE]))[:,1])
    res.append(("M7 internal-tuned",I,P,Q,verdict(P,Q))); print(f"[M7] internal-best val={best[0]:.4f} → internal={I:.4f} pre={P:.4f} post={Q:.4f} ({time.time()-t:.0f}s)\n     cfg={best[1]}",flush=True)
except Exception as e: print(f"[M7] ERROR {e}",flush=True); res.append(("M7 internal-tuned",None,None,None,"err"))

print("\n"+"="*68); print(f"{'method':<26}{'internal':>10}{'pre':>9}{'post':>9}   판정"); print("="*68)
for nm,I,P,Q,v in res:
    si=f"{I:.4f}" if I is not None else "  -  "; sp=f"{P:.4f}" if P is not None else "  -  "; sq=f"{Q:.4f}" if Q is not None else "  -  "
    print(f"{nm:<26}{si:>10}{sp:>9}{sq:>9}   {v}")
print(f"\n베이스라인 pre={bP:.4f} post={bQ:.4f}. ★UP=양쪽 모두 +0.001 초과.")
pd.DataFrame(res,columns=["method","internal","pre","post","verdict"]).to_csv(f"{D}/results/methods_explore.csv",index=False,encoding="utf-8-sig")
print("저장: results/methods_explore.csv")
