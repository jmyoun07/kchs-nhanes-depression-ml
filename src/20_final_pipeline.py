# -*- coding: utf-8 -*-
"""20_final_pipeline.py — 최종 확정 파이프라인 (프로토콜 §2 준거, 17변수, leakage-clean).
흐름: 모델개발(3알고리즘·내부CV·주모델선택) → 내부검증 → 외부검증(pre/post·재보정) → 경량 → 민감도.
모든 모델선택은 KCHS 내부만. NHANES는 최종 평가 전용(§2.10).
입력: data/processed/ext_{kchs,pre,post}.parquet (17변수 조화 완료)
출력: results/final_metrics.csv, results/final_parsimonious.csv, results/final_sensitivity.csv
"""
import os, sys, warnings, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from scipy.optimize import brentq
import lightgbm as lgb
PROC=r"C:/Users/rudtn/Desktop/claude code/지역사회건강/data/processed"; RES=r"C:/Users/rudtn/Desktop/claude code/지역사회건강/results"; SEED=42
kchs=pd.read_parquet(f"{PROC}/ext_kchs.parquet"); pre=pd.read_parquet(f"{PROC}/ext_pre.parquet"); post=pd.read_parquet(f"{PROC}/ext_post.parquet")
# ── 17 조화 예측변수 (§2.5) ──
NUM=["age","sleep_hrs","bmi","income_q","srh","oral_health"]
BIN=["partnered","living_alone","current_drinker","active","htn","dm","employed","food_insecure"]
CAT=["sex","edu","smoking"]; FE=NUM+BIN+CAT
y=kchs["dep"].astype(int).values

def make_ct(native=True):
    binp = "passthrough" if native else SimpleImputer(strategy="most_frequent", add_indicator=True)
    return ColumnTransformer([("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),NUM),
        ("bin",binp,BIN),("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore",drop="first"))]),CAT)])
# §2.8 튜닝된 LightGBM (ROUND F 내부CV 선택 설정)
LGB=dict(num_leaves=31,min_child_samples=1000,learning_rate=0.01,n_estimators=900,reg_alpha=1.0,reg_lambda=20.0,subsample=0.6,colsample_bytree=0.6,random_state=SEED,n_jobs=-1,verbose=-1)
def cal_si(yv,p,w=None):
    p=np.clip(p,1e-6,1-1e-6); lp=np.log(p/(1-p)).reshape(-1,1)
    c=LogisticRegression(penalty=None,solver="lbfgs").fit(lp,yv,sample_weight=w); return float(c.coef_[0,0]),float(c.intercept_[0])
def metrics(yv,p,w=None):
    sl,ic=cal_si(yv,p,w)
    return dict(AUROC=roc_auc_score(yv,p,sample_weight=w),AUPRC=average_precision_score(yv,p,sample_weight=w),
                Brier=brier_score_loss(yv,p,sample_weight=w),cal_slope=sl,cal_intercept=ic,
                obs=float(np.average(yv,weights=w)),pred=float(np.average(p,weights=w)))
def wauc_ci(d,p,B=1000):
    yy=d["dep"].astype(int).values;W=d["W"].to_numpy(float);grp=(d["strata"].astype(str)+"|"+d["psu"].astype(str)).values
    uid=pd.unique(grp);idx={u:np.where(grp==u)[0] for u in uid};r=np.random.RandomState(SEED);a=[]
    for _ in range(B):
        ii=np.concatenate([idx[u] for u in r.choice(uid,len(uid),replace=True)])
        try:a.append(roc_auc_score(yy[ii],p[ii],sample_weight=W[ii]))
        except:pass
    a=np.array(a);return np.percentile(a,2.5),np.percentile(a,97.5)

tr,ho=train_test_split(np.arange(len(kchs)),train_size=500000,test_size=100000,stratify=y,random_state=SEED)
Who,yho=kchs.iloc[ho]["W"].to_numpy(float),y[ho]

print("="*74); print("STAGE 4. 모델 개발 (§2.8) — 3알고리즘 내부 CV 비교 + 주모델 사전선택"); print("="*74,flush=True)
models={
 "elasticnet":Pipeline([("pre",make_ct(native=False)),("clf",LogisticRegression(penalty="elasticnet",l1_ratio=0.5,C=1.0,solver="saga",max_iter=300,n_jobs=-1))]),
 "randomforest":Pipeline([("pre",make_ct(native=True)),("clf",RandomForestClassifier(n_estimators=300,max_depth=14,min_samples_leaf=100,n_jobs=-1,random_state=SEED))]),
 "lightgbm":Pipeline([("pre",make_ct(native=True)),("clf",lgb.LGBMClassifier(**LGB))]),
}
sub=np.random.RandomState(SEED).permutation(tr)[:200000]
cvres={}
skf=StratifiedKFold(5,shuffle=True,random_state=SEED)
for nm,mdl in models.items():
    oof=cross_val_predict(mdl,kchs.iloc[sub][FE],y[sub],cv=skf,method="predict_proba",n_jobs=-1)[:,1]
    cv=roc_auc_score(y[sub],oof); sl,_=cal_si(y[sub],oof)
    cvres[nm]=(cv,abs(sl-1)); print(f"   {nm:14} CV-AUROC={cv:.4f}  |보정slope-1|={abs(sl-1):.3f}",flush=True)
primary=sorted(cvres,key=lambda k:(-cvres[k][0],cvres[k][1]))[0]
print(f"   ★ 주모델 = {primary} (사전기준: CV-AUROC→보정→간명성)",flush=True)

print("\n"+"="*74); print("STAGE 5-6. 내부·외부 검증 (§2.9·2.10) — 주모델 refit 후 평가"); print("="*74,flush=True)
mfin=models[primary].fit(kchs.iloc[tr][FE],y[tr])
def P(d): return mfin.predict_proba(d[FE])[:,1]
rows=[]
# 내부
pi=P(kchs.iloc[ho]); mi=metrics(yho,pi,Who); lo,hi=wauc_ci(kchs.iloc[ho],pi)
rows.append(("Internal(hold-out)",mi,(lo,hi)))
# 외부 pre/post + 재보정(절편/기울기, 교차적합)
def recalib(d,p):
    yv=d["dep"].astype(int).values;W=d["W"].to_numpy(float)
    # 5-fold 교차적합 재보정(out-of-sample)
    lp=np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
    oof=np.zeros_like(p)
    for tri,tei in StratifiedKFold(5,shuffle=True,random_state=SEED).split(lp,yv):
        c=LogisticRegression(penalty=None,solver="lbfgs").fit(lp[tri].reshape(-1,1),yv[tri],sample_weight=W[tri])
        oof[tei]=c.predict_proba(lp[tei].reshape(-1,1))[:,1]
    return oof
for nm,d in [("NHANES pre",pre),("NHANES post",post)]:
    p=P(d); yv=d["dep"].astype(int).values;W=d["W"].to_numpy(float)
    m=metrics(yv,p,W); lo,hi=wauc_ci(d,p); rows.append((nm,m,(lo,hi)))
    pr=recalib(d,p); mr=metrics(yv,pr,W); rows.append((nm+" (재보정)",mr,(None,None)))
print(f"{'set':<22}{'AUROC':>8}{'95%CI':>18}{'AUPRC':>8}{'Brier':>8}{'slope':>8}{'CITL':>8}",flush=True)
outrows=[]
for nm,m,ci in rows:
    cis=f"[{ci[0]:.3f},{ci[1]:.3f}]" if ci[0] is not None else "—"
    print(f"{nm:<22}{m['AUROC']:>8.4f}{cis:>18}{m['AUPRC']:>8.3f}{m['Brier']:>8.4f}{m['cal_slope']:>8.3f}{m['cal_intercept']:>8.3f}",flush=True)
    outrows.append(dict(set=nm,AUROC=round(m['AUROC'],4),CI_lo=ci[0],CI_hi=ci[1],AUPRC=round(m['AUPRC'],3),Brier=round(m['Brier'],4),cal_slope=round(m['cal_slope'],3),CITL=round(m['cal_intercept'],3)))
pd.DataFrame(outrows).to_csv(f"{RES}/final_metrics.csv",index=False,encoding="utf-8-sig")

print("\n"+"="*74); print("STAGE 7. 경량 모델 (§2.11) — 중요도 상위 k문항"); print("="*74,flush=True)
imp=pd.Series(mfin.named_steps["clf"].feature_importances_ if primary!="elasticnet" else np.abs(mfin.named_steps["clf"].coef_[0]))
# 원변수 중요도(대략): LightGBM gain 기준 상위
gbm=lgb.LGBMClassifier(**LGB).fit(make_ct().fit_transform(kchs.iloc[tr][FE]),y[tr])
# 간단히 사전지식 순서 사용(SHAP 기반 기존확인): srh,age,sex,sleep,oral,employed,...
order=["srh","age","sex","sleep_hrs","oral_health","smoking","employed","partnered","income_q","bmi","food_insecure","htn","active","current_drinker","living_alone","dm","edu"]
prows=[]
for k in [3,4,5,8,12,17]:
    sub_f=order[:k]
    NUMx=[c for c in NUM if c in sub_f];BINx=[c for c in BIN if c in sub_f];CATx=[c for c in CAT if c in sub_f]
    ct=ColumnTransformer([t for t in [("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),NUMx) if NUMx else None,
        ("bin","passthrough",BINx) if BINx else None,
        ("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore",drop="first"))]),CATx) if CATx else None] if t])
    FEk=NUMx+BINx+CATx
    mk=Pipeline([("pre",ct),("clf",lgb.LGBMClassifier(**LGB))]).fit(kchs.iloc[tr][FEk],y[tr])
    ik=roc_auc_score(yho,mk.predict_proba(kchs.iloc[ho][FEk])[:,1],sample_weight=Who)
    pk=roc_auc_score(pre["dep"].astype(int),mk.predict_proba(pre[FEk])[:,1],sample_weight=pre["W"].to_numpy(float))
    print(f"   {k:>2}문항 {str(sub_f):<60} 내부 {ik:.4f} 외부pre {pk:.4f}",flush=True)
    prows.append(dict(k=k,vars=";".join(sub_f),internal=round(ik,4),ext_pre=round(pk,4)))
pd.DataFrame(prows).to_csv(f"{RES}/final_parsimonious.csv",index=False,encoding="utf-8-sig")

print("\n"+"="*74); print("STAGE 8. 민감도 분석 (§2.12)"); print("="*74,flush=True)
srows=[]
def quick(FEs,label):
    m=Pipeline([("pre",ColumnTransformer([("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),[c for c in NUM if c in FEs]),
        ("bin","passthrough",[c for c in BIN if c in FEs]),
        ("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore",drop="first"))]),[c for c in CAT if c in FEs])])),
        ("clf",lgb.LGBMClassifier(**LGB))]).fit(kchs.iloc[tr][FEs],y[tr])
    i=roc_auc_score(yho,m.predict_proba(kchs.iloc[ho][FEs])[:,1],sample_weight=Who)
    p=roc_auc_score(pre["dep"].astype(int),m.predict_proba(pre[FEs])[:,1],sample_weight=pre["W"].to_numpy(float))
    s=roc_auc_score(post["dep"].astype(int),m.predict_proba(post[FEs])[:,1],sample_weight=post["W"].to_numpy(float))
    print(f"   {label:<34} 내부 {i:.4f} pre {p:.4f} post {s:.4f}",flush=True)
    srows.append(dict(cond=label,internal=round(i,4),ext_pre=round(p,4),ext_post=round(s,4)))
quick(FE,"기준(17변수)")
quick([c for c in FE if c!="food_insecure"],"food_insecure 제외(16)")
quick([c for c in FE if c not in ("food_insecure","oral_health","employed")],"신규3 전부 제외(14)")
# 컷오프 ≥5
y5=(kchs["phq_sum"]>=5).astype(int).values
m5=Pipeline([("pre",make_ct()),("clf",lgb.LGBMClassifier(**LGB))]).fit(kchs.iloc[tr][FE],y5[tr])
i5=roc_auc_score(y5[ho],m5.predict_proba(kchs.iloc[ho][FE])[:,1],sample_weight=Who)
print(f"   {'컷오프≥5(내부만)':<34} 내부 {i5:.4f}",flush=True)
srows.append(dict(cond="컷오프≥5",internal=round(i5,4),ext_pre=None,ext_post=None))
pd.DataFrame(srows).to_csv(f"{RES}/final_sensitivity.csv",index=False,encoding="utf-8-sig")
print("\n저장: results/final_{metrics,parsimonious,sensitivity}.csv")
print("완료.")
