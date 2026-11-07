# -*- coding: utf-8 -*-
"""외부 AUROC 상향 실험 — 누수 통제.
 (1) 베이스라인 17변수 전이 재현
 (2) welfare(+sleep_weekend) 추가 전이
 (3) 오라클 상한 재추정: NHANES 자체 5-fold CV (진짜 천장)
 (4) 공변량이동 보정: NHANES 특징분포로 KCHS 훈련 재가중(라벨 미사용)
 (5) 분위수 정규화 전이
모든 모델 선택은 KCHS-내부만 사용. 외부는 평가만."""
import sys, warnings, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
SEED=42
D="C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model"
kchs=pd.read_parquet(f"{D}/data/ext_kchs.parquet")
pre =pd.read_parquet(f"{D}/data/ext_pre.parquet")
post=pd.read_parquet(f"{D}/data/ext_post.parquet")
y=kchs["dep"].astype(int).values

NUM=["age","sleep_hrs","bmi","income_q","srh","oral_health"]
BIN=["partnered","living_alone","current_drinker","active","htn","dm","employed","food_insecure"]
CAT=["sex","edu","smoking"]
LGB=dict(num_leaves=31,min_child_samples=1000,learning_rate=0.01,n_estimators=900,
         reg_alpha=1.0,reg_lambda=20.0,subsample=0.6,colsample_bytree=0.6,
         random_state=SEED,n_jobs=-1,verbose=-1)
LGB_SMALL=dict(num_leaves=15,min_child_samples=50,learning_rate=0.03,n_estimators=400,
         reg_alpha=1.0,reg_lambda=5.0,subsample=0.8,colsample_bytree=0.8,
         random_state=SEED,n_jobs=-1,verbose=-1)

def make_ct(num,binf,cat,impute_bin=False):
    binstep=(Pipeline([("i",SimpleImputer(strategy="most_frequent"))]) if impute_bin else "passthrough")
    return ColumnTransformer([
        ("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),num),
        ("bin",binstep,binf),
        ("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),
                         ("o",OneHotEncoder(handle_unknown="ignore",drop="first"))]),cat)])

def wauc(d,p): return roc_auc_score(d["dep"].astype(int).values,p,sample_weight=d["W"].to_numpy(float))

tr,ho=train_test_split(np.arange(len(kchs)),train_size=500000,test_size=100000,stratify=y,random_state=SEED)

def transfer(num,binf,cat,label):
    FE=num+binf+cat
    ct=make_ct(num,binf,cat)
    m=Pipeline([("pre",ct),("clf",lgb.LGBMClassifier(**LGB))]).fit(kchs.iloc[tr][FE],y[tr])
    ah=roc_auc_score(y[ho],m.predict_proba(kchs.iloc[ho][FE])[:,1])
    ap=wauc(pre ,m.predict_proba(pre [FE])[:,1])
    aq=wauc(post,m.predict_proba(post[FE])[:,1])
    print(f"  {label:<34} internal={ah:.4f}  pre={ap:.4f}  post={aq:.4f}",flush=True)
    return ah,ap,aq

print("="*74)
print("(1)(2) 전이 실험 — 변수집합 비교 (LightGBM, KCHS 500k 훈련)")
print("="*74)
base=transfer(NUM,BIN,CAT,"17변수 (베이스라인)")
transfer(NUM,BIN+["welfare"],CAT,"+welfare (18)")
transfer(NUM,BIN+["welfare","welfare_12mo"],CAT,"+welfare+welfare_12mo (19)")
transfer(NUM+["sleep_weekend"],BIN+["welfare"],CAT,"+welfare+sleep_weekend (19)")
transfer(NUM,BIN+["welfare","binge"],CAT,"+welfare+binge (19)")

print("\n"+"="*74)
print("(3) 오라클 상한 재추정 — NHANES 자체 5-fold CV (진짜 천장)")
print("    ※ 전이모델이 이 값에 근접하면 '개념이동 천장'=모델 한계 아님")
print("="*74)
def oracle(dd,num,binf,cat,label):
    FE=num+binf+cat
    yy=dd["dep"].astype(int).values; w=dd["W"].to_numpy(float)
    skf=StratifiedKFold(5,shuffle=True,random_state=SEED)
    oof=np.zeros(len(dd)); oof_lr=np.zeros(len(dd))
    for a,b in skf.split(dd,yy):
        ct=make_ct(num,binf,cat)
        g=Pipeline([("pre",ct),("clf",lgb.LGBMClassifier(**LGB_SMALL))]).fit(dd.iloc[a][FE],yy[a],clf__sample_weight=w[a])
        oof[b]=g.predict_proba(dd.iloc[b][FE])[:,1]
        ct2=make_ct(num,binf,cat,impute_bin=True)
        l=Pipeline([("pre",ct2),("clf",LogisticRegression(max_iter=2000,C=1.0))]).fit(dd.iloc[a][FE],yy[a],clf__sample_weight=w[a])
        oof_lr[b]=l.predict_proba(dd.iloc[b][FE])[:,1]
    ag=roc_auc_score(yy,oof,sample_weight=w); al=roc_auc_score(yy,oof_lr,sample_weight=w)
    print(f"  {label:<22} GBM={ag:.4f}  Logit={al:.4f}  (max={max(ag,al):.4f})",flush=True)
    return max(ag,al)
print(" pre:")
op17=oracle(pre,NUM,BIN,CAT,"17변수")
op18=oracle(pre,NUM,BIN+["welfare"],CAT,"+welfare(18)")
print(" post:")
oq17=oracle(post,NUM,BIN,CAT,"17변수")
oq18=oracle(post,NUM,BIN+["welfare"],CAT,"+welfare(18)")
print(f"\n  → 헤드룸(전이 대비 오라클): pre {base[1]:.3f}→{op17:.3f}  post {base[2]:.3f}→{oq17:.3f}")

print("\n"+"="*74)
print("(4) 공변량이동 보정 — 도메인분류기 밀도비로 KCHS 재가중 (NHANES 라벨 미사용)")
print("="*74)
FE=NUM+BIN+CAT
ct_d=make_ct(NUM,BIN,CAT)
ns=200000
ksub=kchs.iloc[tr].sample(ns,random_state=SEED)
Xk=ct_d.fit_transform(pd.concat([ksub[FE],pre[FE],post[FE]]))  # fit on all for common space
Xk_tr=ct_d.transform(ksub[FE]); Xk_ext=ct_d.transform(pd.concat([pre[FE],post[FE]]))
import scipy.sparse as sp
Xdom=sp.vstack([Xk_tr,Xk_ext]) if sp.issparse(Xk_tr) else np.vstack([Xk_tr,Xk_ext])
ydom=np.r_[np.zeros(Xk_tr.shape[0]),np.ones(Xk_ext.shape[0])]
dom=LogisticRegression(max_iter=1000,C=1.0).fit(Xdom,ydom)
pk=dom.predict_proba(Xk_tr)[:,1]; ratio=np.clip(pk/(1-pk+1e-9),0.1,10.0)
ct=make_ct(NUM,BIN,CAT)
m=Pipeline([("pre",ct),("clf",lgb.LGBMClassifier(**LGB))]).fit(ksub[FE],ksub["dep"].astype(int).values,clf__sample_weight=ratio)
print(f"  공변량이동 보정 전이:            pre={wauc(pre,m.predict_proba(pre[FE])[:,1]):.4f}  post={wauc(post,m.predict_proba(post[FE])[:,1]):.4f}",flush=True)

print("\n"+"="*74)
print("(5) 분위수 정규화 — 연속변수를 국가별 분위로 매핑 후 전이")
print("="*74)
def qmap(train_series, apply_series):
    r=train_series.rank(pct=True)  # not used cross; we map each survey to its own pct
    return apply_series.rank(pct=True)
kq=kchs.copy(); pq=pre.copy(); qq=post.copy()
for c in NUM:
    kq[c]=pd.to_numeric(kchs[c],errors="coerce").rank(pct=True)
    pq[c]=pd.to_numeric(pre[c] ,errors="coerce").rank(pct=True)
    qq[c]=pd.to_numeric(post[c],errors="coerce").rank(pct=True)
ct=make_ct(NUM,BIN,CAT)
m=Pipeline([("pre",ct),("clf",lgb.LGBMClassifier(**LGB))]).fit(kq.iloc[tr][FE],y[tr])
print(f"  분위수 정규화 전이:              internal={roc_auc_score(y[ho],m.predict_proba(kq.iloc[ho][FE])[:,1]):.4f}  pre={wauc(pre,m.predict_proba(pq[FE])[:,1]):.4f}  post={wauc(post,m.predict_proba(qq[FE])[:,1]):.4f}",flush=True)
print("\n완료.")
