# -*- coding: utf-8 -*-
"""54_review3_ci.py — 3차 심사 통계 지적 재계산.

(A) KCHS 층 정의: 가중치가 연도별로 정규화되고 층 코드가 연도 간 재사용되므로 층 = year|strata 로 구성
(B) 하위군(domain) CI: 부분집합으로 자른 뒤 설계를 재구성하면 안 됨 →
    전체 코호트 설계에서 복제가중치를 만들고, 그 가중치를 도메인에 적용
(C) 오라클 CI: 46_review_fixes.py의 구버전(미재척도) 군집 부트스트랩으로 산출돼 있었음 → Rao-Wu 재척도로 재산출

출력: results/review3_metrics.csv, review3_fairness_domain.csv, review3_oracle_ci.csv
"""
import sys, warnings, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb

D = r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model"; RES = f"{D}/results"; SEED = 42
kchs = pd.read_parquet(f"{D}/data/ext_kchs.parquet"); pre = pd.read_parquet(f"{D}/data/ext_pre.parquet"); post = pd.read_parquet(f"{D}/data/ext_post.parquet")
NUM=["age","sleep_hrs","bmi","income_q","srh","oral_health"]
BIN=["partnered","living_alone","current_drinker","active","htn","dm","employed","food_insecure"]
CAT=["sex","edu","smoking"]; FE=NUM+BIN+CAT
y=kchs["dep"].astype(int).values
LGB=dict(num_leaves=31,min_child_samples=1000,learning_rate=0.01,n_estimators=900,reg_alpha=1.0,
         reg_lambda=20.0,subsample=0.6,colsample_bytree=0.6,random_state=SEED,n_jobs=-1,verbose=-1)
def make_ct(native=True):
    return ColumnTransformer([("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),NUM),
                              ("bin","passthrough" if native else SimpleImputer(strategy="most_frequent"),BIN),
                              ("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore",drop="first"))]),CAT)])

class RaoWuBoot:
    """층 h에서 m_h=n_h-1개 PSU 복원추출 + w*(n_h/m_h)*r 재척도. 단일 PSU 층은 유사층으로 통합."""
    def __init__(self, strata, psu, w, pool_singletons=True):
        st=pd.Series(strata).astype(str).reset_index(drop=True); ps=pd.Series(psu).astype(str).reset_index(drop=True)
        # 군집 정체성은 '원래' 층|PSU 로 고정한다. 단일층 통합은 '층 라벨'만 바꿀 뿐이며,
        # 통합 후 라벨로 군집을 다시 만들면 서로 다른 층의 동일 PSU 라벨이 한 군집으로 합쳐진다(버그).
        cl=pd.factorize(st+"|"+ps)[0]
        st_group=st
        if pool_singletons:
            npsu=pd.DataFrame({"s":st,"p":ps}).drop_duplicates().groupby("s").size()
            single=set(npsu[npsu==1].index)
            st_group=st.where(~st.isin(single),"__SINGLETON_POOL__")
        sid=pd.factorize(st_group)[0]
        self.cl=cl; self.w=np.asarray(w,float); C=cl.max()+1; self.C=C
        cs=np.zeros(C,dtype=int); cs[cl]=sid; self.cl_stratum=cs
        self.n_h=np.bincount(cs); self.m_h=np.maximum(self.n_h-1,1)
        o=np.argsort(cs,kind="stable"); self.cl_sorted=o
        ss=cs[o]; sc=np.bincount(ss); start=np.concatenate([[0],np.cumsum(sc)])[:-1]
        self.slot_start=start[ss]; self.slot_n=sc[ss]; self.singleton=self.n_h[cs]==1
    def weights(self, rng):
        r=np.zeros(self.C); u=rng.random_sample(self.C)
        idx=np.arange(self.C)-self.slot_start
        sel=np.where(idx < self.m_h[self.cl_stratum[self.cl_sorted]])[0]
        pick=self.slot_start[sel]+(u[sel]*self.slot_n[sel]).astype(int)
        np.add.at(r,self.cl_sorted[pick],1.0)
        sc=np.where(self.singleton,1.0,(self.n_h[self.cl_stratum]/self.m_h[self.cl_stratum])*r)
        return self.w*sc[self.cl]

def rep_weights(d, strata, B=1000, seed=SEED):
    rw=RaoWuBoot(strata, d["psu"].values, d["W"].to_numpy(float))
    rng=np.random.RandomState(seed)
    return [rw.weights(rng) for _ in range(B)]

def ci_from(yv, p, reps, fn=roc_auc_score):
    a=[]
    for w in reps:
        m=w>0
        if m.sum()<10 or yv[m].min()==yv[m].max(): continue
        try: a.append(fn(yv,p,sample_weight=w))
        except Exception: pass
    a=np.array(a); return float(np.percentile(a,2.5)), float(np.percentile(a,97.5)), len(a)

tr, ho = train_test_split(np.arange(len(kchs)), train_size=500000, test_size=100000, stratify=y, random_state=SEED)
model=Pipeline([("pre",make_ct()),("clf",lgb.LGBMClassifier(**LGB))]).fit(kchs.iloc[tr][FE],y[tr])
dho=kchs.iloc[ho].copy(); dho["_p"]=model.predict_proba(dho[FE])[:,1]
prd=pre.copy();  prd["_p"]=model.predict_proba(pre[FE])[:,1]
pod=post.copy(); pod["_p"]=model.predict_proba(post[FE])[:,1]

# ══ (A) 내부 CI: 층 = year|strata ══
print("="*92); print("[A] KCHS 층 정의 — year|strata 로 재구성한 내부 CI"); print("="*92, flush=True)
st_old=dho["strata"].astype(str).values
st_new=(dho["year"].astype(str)+"|"+dho["strata"].astype(str)).values
yv=dho["dep"].astype(int).values; p=dho["_p"].values
rows=[]
for tag,st in [("strata only (superseded)",st_old),("year|strata (correct)",st_new)]:
    reps=rep_weights(dho,st,B=1000)
    lo,hi,n=ci_from(yv,p,reps); alo,ahi,_=ci_from(yv,p,reps,average_precision_score)
    print(f"   {tag:<28} AUROC CI [{lo:.4f},{hi:.4f}]  AUPRC CI [{alo:.3f},{ahi:.3f}]  ({n} reps)", flush=True)
    rows.append(dict(setting=tag, AUROC_lo=round(lo,4), AUROC_hi=round(hi,4), AUPRC_lo=round(alo,3), AUPRC_hi=round(ahi,3)))
pd.DataFrame(rows).to_csv(f"{RES}/review3_metrics.csv", index=False, encoding="utf-8-sig")

# ══ (B) 하위군: 전체 설계 복제가중치를 도메인에 적용 ══
print("\n"+"="*92); print("[B] 하위군 CI — 전체 코호트 설계에서 만든 복제가중치를 도메인에 적용"); print("="*92, flush=True)
def logit(x): x=np.clip(x,1e-6,1-1e-6); return np.log(x/(1-x))
def cal_slope(a,b,w): return float(LogisticRegression(penalty=None,solver="lbfgs").fit(logit(b).reshape(-1,1),a,sample_weight=w).coef_[0,0])
def citl(a,b,w):
    lp=logit(b); z=0.0
    for _ in range(200):
        mu=1/(1+np.exp(-(z+lp))); g=np.sum(w*(a-mu)); h=np.sum(w*mu*(1-mu))
        if h<=0: break
        s=g/h; z+=s
        if abs(s)<1e-10: break
    return float(z)
COH=[("KCHS hold-out",dho,(dho["year"].astype(str)+"|"+dho["strata"].astype(str)).values),
     ("NHANES pre",prd,prd["strata"].astype(str).values),
     ("NHANES post",pod,pod["strata"].astype(str).values)]
REPS={nm:rep_weights(d,st,B=1000) for nm,d,st in COH}
age=pd.to_numeric(dho["age"])
RACE={1:"Mexican American",2:"Other Hispanic",3:"Non-Hispanic White",4:"Non-Hispanic Black",6:"Non-Hispanic Asian",7:"Other/multiracial"}
groups=[("KCHS hold-out","Sex: female",dho["sex"]=="F"),("KCHS hold-out","Sex: male",dho["sex"]=="M"),
        ("KCHS hold-out","Age 20-39",age<40),("KCHS hold-out","Age 40-59",(age>=40)&(age<60)),("KCHS hold-out","Age 60+",age>=60),
        ("KCHS hold-out","Education: college",dho["edu"]=="college"),("KCHS hold-out","Education: high school",dho["edu"]=="high"),
        ("KCHS hold-out","Education: middle",dho["edu"]=="middle"),("KCHS hold-out","Education: <=elementary",dho["edu"]=="elem_low")]
for c,l in RACE.items():
    groups.append(("NHANES pre",f"Race: {l}",prd["race"]==c)); groups.append(("NHANES post",f"Race: {l}",pod["race"]==c))
SRC={nm:d for nm,d,_ in COH}
frows=[]
for coh,lab,mask in groups:
    d=SRC[coh]; m=np.asarray(mask); dd=d[m]
    if dd["dep"].nunique()<2 or len(dd)<100: continue
    yv=dd["dep"].astype(int).values; p=dd["_p"].values; w=dd["W"].to_numpy(float)
    a=roc_auc_score(yv,p,sample_weight=w)
    reps=[rw[m] for rw in REPS[coh]]
    lo,hi,n=ci_from(yv,p,reps)
    print(f"   {coh:<12} {lab:<28} n={len(dd):>6,}  AUROC {a:.3f}  CI [{lo:.3f},{hi:.3f}]  ({n} reps)", flush=True)
    frows.append(dict(cohort=coh,subgroup=lab,n=len(dd),AUROC=round(a,3),CI_lo=round(lo,3),CI_hi=round(hi,3),
                      cal_slope=round(cal_slope(yv,p,w),3),CITL=round(citl(yv,p,w),3)))
pd.DataFrame(frows).to_csv(f"{RES}/review3_fairness_domain.csv",index=False,encoding="utf-8-sig")

# ══ (C) 오라클 CI: Rao-Wu 재척도로 재산출 ══
print("\n"+"="*92); print("[C] NHANES 자기적합 오라클 CI — Rao-Wu 재척도로 재산출"); print("="*92, flush=True)
pooled=pd.concat([pre.assign(_period="pre"),post.assign(_period="post")],ignore_index=True)
st=(pooled["_period"].astype(str)+"|"+pooled["strata"].astype(str)).values
yo=pooled["dep"].astype(int).values; Wo=pooled["W"].to_numpy(float)
reps=rep_weights(pooled.assign(W=Wo),st,B=1000)
LEARN={"Logistic regression":lambda:Pipeline([("pre",make_ct(False)),("clf",LogisticRegression(max_iter=2000))]),
       "Penalized logistic (L2, C=0.1)":lambda:Pipeline([("pre",make_ct(False)),("clf",LogisticRegression(C=0.1,max_iter=2000))]),
       "Histogram gradient boosting":lambda:Pipeline([("pre",make_ct()),("clf",HistGradientBoostingClassifier(random_state=SEED,max_iter=300,learning_rate=0.05))]),
       "LightGBM (small)":lambda:Pipeline([("pre",make_ct()),("clf",lgb.LGBMClassifier(random_state=SEED,n_jobs=-1,verbose=-1,num_leaves=7,min_child_samples=200,learning_rate=0.05,n_estimators=300))])}
skf=StratifiedKFold(5,shuffle=True,random_state=SEED); orows=[]
for nm,mk in LEARN.items():
    oof=np.zeros(len(pooled))
    for tri,tei in skf.split(pooled[FE],yo):
        oof[tei]=mk().fit(pooled.iloc[tri][FE],yo[tri]).predict_proba(pooled.iloc[tei][FE])[:,1]
    a=roc_auc_score(yo,oof,sample_weight=Wo); lo,hi,n=ci_from(yo,oof,reps)
    print(f"   {nm:<34} {a:.4f}  95% CI [{lo:.3f},{hi:.3f}]", flush=True)
    orows.append(dict(learner=nm,oracle_AUROC=round(a,4),CI_lo=round(lo,3),CI_hi=round(hi,3)))
pd.DataFrame(orows).to_csv(f"{RES}/review3_oracle_ci.csv",index=False,encoding="utf-8-sig")
print("\n저장: results/review3_{metrics,fairness_domain,oracle_ci}.csv\n완료.", flush=True)
