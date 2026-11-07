# -*- coding: utf-8 -*-
"""52_subgroup_ci_fix.py — 하위군 CI의 단일 PSU 층(singleton stratum) 처리.

문제: 전체 코호트에서는 층당 PSU가 2개 이상이지만, 하위군으로 자르면 층당 PSU가 1개인 층이 생긴다
      (예: NHANES pre 'Mexican American'은 23개 층 중 5개가 단일). 재척도 부트스트랩에서 단일 층을
      그대로 두면 그 층의 분산 기여가 0이 되어 하위군 CI가 좁아진다.
해결: 단일 PSU 층들을 하나의 유사층(pseudo-stratum)으로 통합해 재표집한다(설문 분석의 표준 처리).

출력: results/review3_fairness.csv  (하위군 AUROC·CI·보정slope·CITL, singleton 통합 적용)
"""
import sys, warnings, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

D = r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model"; RES = f"{D}/results"; SEED = 42
kchs = pd.read_parquet(f"{D}/data/ext_kchs.parquet"); pre = pd.read_parquet(f"{D}/data/ext_pre.parquet"); post = pd.read_parquet(f"{D}/data/ext_post.parquet")
NUM=["age","sleep_hrs","bmi","income_q","srh","oral_health"]
BIN=["partnered","living_alone","current_drinker","active","htn","dm","employed","food_insecure"]
CAT=["sex","edu","smoking"]; FE=NUM+BIN+CAT
y=kchs["dep"].astype(int).values
LGB=dict(num_leaves=31,min_child_samples=1000,learning_rate=0.01,n_estimators=900,reg_alpha=1.0,
         reg_lambda=20.0,subsample=0.6,colsample_bytree=0.6,random_state=SEED,n_jobs=-1,verbose=-1)
def make_ct():
    return ColumnTransformer([("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),NUM),
                              ("bin","passthrough",BIN),
                              ("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore",drop="first"))]),CAT)])

class RaoWuBoot:
    """층 h에서 m_h=n_h-1개 PSU 복원추출 + 가중치 재척도.
       단일 PSU 층은 하나의 유사층으로 통합해 분산 기여가 사라지지 않게 한다."""
    def __init__(self, strata, psu, w, pool_singletons=True):
        st = pd.Series(strata).astype(str).reset_index(drop=True)
        ps = pd.Series(psu).astype(str).reset_index(drop=True)
        # 군집 정체성은 원래 층|PSU 로 고정(통합 라벨로 군집을 다시 만들면 다른 층의 동일 PSU가 합쳐진다)
        cl = pd.factorize(st + "|" + ps)[0]
        st_group = st
        if pool_singletons:
            npsu = pd.DataFrame({"s":st,"p":ps}).drop_duplicates().groupby("s").size()
            single = set(npsu[npsu==1].index)
            st_group = st.where(~st.isin(single), "__SINGLETON_POOL__")
        sid = pd.factorize(st_group)[0]
        self.cl = cl; self.w = np.asarray(w, float)
        C = cl.max()+1; self.C = C
        cl_stratum = np.zeros(C, dtype=int); cl_stratum[cl] = sid; self.cl_stratum = cl_stratum
        self.n_h = np.bincount(cl_stratum); self.m_h = np.maximum(self.n_h-1, 1)
        o = np.argsort(cl_stratum, kind="stable"); self.cl_sorted = o
        s_sorted = cl_stratum[o]; sc = np.bincount(s_sorted)
        st_start = np.concatenate([[0], np.cumsum(sc)])[:-1]
        self.slot_start = st_start[s_sorted]; self.slot_n = sc[s_sorted]
        self.singleton = self.n_h[cl_stratum] == 1
        self.n_singleton_strata = int((self.n_h==1).sum())
    def weights(self, rng):
        r = np.zeros(self.C)
        draws = rng.random_sample(self.C)
        idx = np.arange(self.C) - self.slot_start
        keep = idx < self.m_h[self.cl_stratum[self.cl_sorted]]
        sel = np.where(keep)[0]
        pick = self.slot_start[sel] + (draws[sel]*self.slot_n[sel]).astype(int)
        np.add.at(r, self.cl_sorted[pick], 1.0)
        scale = np.where(self.singleton, 1.0, (self.n_h[self.cl_stratum]/self.m_h[self.cl_stratum])*r)
        return self.w * scale[self.cl]

def logit(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
def cal_slope(yv,p,w): return float(LogisticRegression(penalty=None,solver="lbfgs").fit(logit(p).reshape(-1,1),yv,sample_weight=w).coef_[0,0])
def citl(yv,p,w):
    lp=logit(p); a=0.0
    for _ in range(200):
        mu=1/(1+np.exp(-(a+lp))); g=np.sum(w*(yv-mu)); h=np.sum(w*mu*(1-mu))
        if h<=0: break
        s=g/h; a+=s
        if abs(s)<1e-10: break
    return float(a)

tr, ho = train_test_split(np.arange(len(kchs)), train_size=500000, test_size=100000, stratify=y, random_state=SEED)
model = Pipeline([("pre",make_ct()),("clf",lgb.LGBMClassifier(**LGB))]).fit(kchs.iloc[tr][FE], y[tr])
dho=kchs.iloc[ho].copy(); dho["_p"]=model.predict_proba(dho[FE])[:,1]
prd=pre.copy();  prd["_p"]=model.predict_proba(pre[FE])[:,1]
pod=post.copy(); pod["_p"]=model.predict_proba(post[FE])[:,1]
age=pd.to_numeric(dho["age"])
RACE={1:"Mexican American",2:"Other Hispanic",3:"Non-Hispanic White",4:"Non-Hispanic Black",6:"Non-Hispanic Asian",7:"Other/multiracial"}
groups=[("KCHS hold-out","Sex: female",dho,dho["sex"]=="F"),("KCHS hold-out","Sex: male",dho,dho["sex"]=="M"),
        ("KCHS hold-out","Age 20-39",dho,age<40),("KCHS hold-out","Age 40-59",dho,(age>=40)&(age<60)),("KCHS hold-out","Age 60+",dho,age>=60),
        ("KCHS hold-out","Education: college",dho,dho["edu"]=="college"),("KCHS hold-out","Education: high school",dho,dho["edu"]=="high"),
        ("KCHS hold-out","Education: middle",dho,dho["edu"]=="middle"),("KCHS hold-out","Education: <=elementary",dho,dho["edu"]=="elem_low")]
for c,l in RACE.items():
    groups.append(("NHANES pre",f"Race: {l}",prd,prd["race"]==c))
    groups.append(("NHANES post",f"Race: {l}",pod,pod["race"]==c))

print("="*92); print("[하위군] 단일 PSU 층 통합 적용 · 재척도 부트스트랩 CI"); print("="*92, flush=True)
rows=[]
for coh,lab,src,mask in groups:
    d=src[mask]
    if d["dep"].nunique()<2 or len(d)<100: continue
    yv=d["dep"].astype(int).values; w=d["W"].to_numpy(float); p=d["_p"].values
    a=roc_auc_score(yv,p,sample_weight=w)
    out={}
    for tag,pool in [("pooled",True),("naive",False)]:
        rw=RaoWuBoot(d["strata"].values,d["psu"].values,w,pool_singletons=pool)
        rng=np.random.RandomState(SEED); est=[]
        for _ in range(1000):
            ww=rw.weights(rng); m=ww>0
            if yv[m].min()==yv[m].max(): continue
            try: est.append(roc_auc_score(yv,p,sample_weight=ww))
            except Exception: pass
        est=np.array(est); out[tag]=(float(np.percentile(est,2.5)),float(np.percentile(est,97.5)),rw.n_singleton_strata)
    lo,hi,ns=out["pooled"]; nlo,nhi,_=out["naive"]
    print(f"  {coh:<12} {lab:<28} n={len(d):>6,}  AUROC {a:.3f}  CI [{lo:.3f},{hi:.3f}]  (단일층 {ns}개, 미통합시 [{nlo:.3f},{nhi:.3f}])", flush=True)
    rows.append(dict(cohort=coh,subgroup=lab,n=len(d),AUROC=round(a,3),CI_lo=round(lo,3),CI_hi=round(hi,3),
                     CI_lo_naive=round(nlo,3),CI_hi_naive=round(nhi,3),singleton_strata=ns,
                     cal_slope=round(cal_slope(yv,p,w),3),CITL=round(citl(yv,p,w),3)))
pd.DataFrame(rows).to_csv(f"{RES}/review3_fairness.csv",index=False,encoding="utf-8-sig")
print("\n저장: results/review3_fairness.csv\n완료.", flush=True)
