# -*- coding: utf-8 -*-
"""55_oracle_grouped.py — 오라클 교차검증을 PSU 단위로 분리 + 실험 레지스트리 생성.

지적 1: 오라클이 개인 단위 StratifiedKFold를 써서 같은 PSU가 학습·검증 폴드에 동시에 들어갔다.
        → 주기 내 PSU를 그룹으로 묶는 GroupKFold로 재산출.
지적 2: CI가 고정된 out-of-fold 예측을 재표집해 '모델 재적합 변동'을 포함하지 않는다.
        → 각 부트스트랩 복제에서 오라클을 재적합하는 중첩 부트스트랩을 200회 수행해 비교.
지적 3: ">450 experiments"의 근거 파일이 지정 정본 목록에 없다.
        → 탐색 CSV들을 기계적으로 세어 중복 제거한 레지스트리를 만든다.

출력: results/review4_oracle_grouped.csv, results/experiment_registry.csv
"""
import sys, glob, os, warnings, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

D = os.path.dirname(os.path.abspath(__file__)); RES = f"{D}/results"; SEED = 42
pre = pd.read_parquet(f"{D}/data/ext_pre.parquet"); post = pd.read_parquet(f"{D}/data/ext_post.parquet")
NUM=["age","sleep_hrs","bmi","income_q","srh","oral_health"]
BIN=["partnered","living_alone","current_drinker","active","htn","dm","employed","food_insecure"]
CAT=["sex","edu","smoking"]; FE=NUM+BIN+CAT
def make_ct(native=True):
    return ColumnTransformer([("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),NUM),
                              ("bin","passthrough" if native else SimpleImputer(strategy="most_frequent"),BIN),
                              ("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore",drop="first"))]),CAT)])

pooled = pd.concat([pre.assign(_p="pre"), post.assign(_p="post")], ignore_index=True)
strata = (pooled["_p"].astype(str)+"|"+pooled["strata"].astype(str)).values
groups = (pooled["_p"].astype(str)+"|"+pooled["strata"].astype(str)+"|"+pooled["psu"].astype(str)).values
yo = pooled["dep"].astype(int).values; Wo = pooled["W"].to_numpy(float)
print(f"[설계] pooled n={len(pooled):,} · 층 {len(set(strata))} · PSU {len(set(groups))}", flush=True)

class RaoWuBoot:
    def __init__(self, strata, psu, w, pool_singletons=True):
        st=pd.Series(strata).astype(str).reset_index(drop=True); ps=pd.Series(psu).astype(str).reset_index(drop=True)
        cl=pd.factorize(st+"|"+ps)[0]; st_group=st
        if pool_singletons:
            npsu=pd.DataFrame({"s":st,"p":ps}).drop_duplicates().groupby("s").size()
            st_group=st.where(~st.isin(set(npsu[npsu==1].index)),"__SINGLETON_POOL__")
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

LEARN = {
 "Logistic regression": lambda: Pipeline([("pre",make_ct(False)),("clf",LogisticRegression(max_iter=2000))]),
 "Penalized logistic (L2, C=0.1)": lambda: Pipeline([("pre",make_ct(False)),("clf",LogisticRegression(C=0.1,max_iter=2000))]),
 "Histogram gradient boosting": lambda: Pipeline([("pre",make_ct()),("clf",HistGradientBoostingClassifier(random_state=SEED,max_iter=300,learning_rate=0.05))]),
 "LightGBM (small)": lambda: Pipeline([("pre",make_ct()),("clf",lgb.LGBMClassifier(random_state=SEED,n_jobs=-1,verbose=-1,num_leaves=7,min_child_samples=200,learning_rate=0.05,n_estimators=300))]),
}
def oof_grouped(mk, idx=None):
    """주기 내 PSU를 그룹으로 분리한 5-fold 층화 그룹 CV."""
    sub = np.arange(len(pooled)) if idx is None else idx
    yy = yo[sub]; gg = groups[sub]
    oof = np.zeros(len(sub))
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    X = pooled.iloc[sub]
    for tri, tei in cv.split(X[FE], yy, groups=gg):
        oof[tei] = mk().fit(X.iloc[tri][FE], yy[tri]).predict_proba(X.iloc[tei][FE])[:,1]
    return oof

print("\n[오라클] PSU 분리 그룹 CV", flush=True)
rw = RaoWuBoot(strata, pooled["psu"].values, Wo)
reps = [rw.weights(np.random.RandomState(SEED+i)) for i in range(1000)]
rows=[]
best_name, best_auc = None, -1
for nm, mk in LEARN.items():
    oof = oof_grouped(mk)
    a = roc_auc_score(yo, oof, sample_weight=Wo)
    est=[]
    for w in reps:
        m=w>0
        if m.sum()<10 or yo[m].min()==yo[m].max(): continue
        try: est.append(roc_auc_score(yo, oof, sample_weight=w))
        except Exception: pass
    est=np.array(est); lo,hi=np.percentile(est,[2.5,97.5])
    print(f"   {nm:<34} {a:.4f}  95% CI [{lo:.3f},{hi:.3f}]   (예측 고정 재표집)", flush=True)
    rows.append(dict(learner=nm, oracle_AUROC=round(a,4), CI_lo=round(lo,3), CI_hi=round(hi,3), method="grouped CV, fixed-OOF bootstrap"))
    if a>best_auc: best_auc, best_name, best_mk = a, nm, mk

# 중첩 부트스트랩: 복제마다 오라클을 재적합해 학습변동 포함
print(f"\n[중첩 부트스트랩] {best_name} — 복제마다 재적합(200회)", flush=True)
nest=[]
for i in range(200):
    w = reps[i]
    keep = np.where(w>0)[0]
    if yo[keep].min()==yo[keep].max(): continue
    try:
        o2 = oof_grouped(best_mk, idx=keep)
        nest.append(roc_auc_score(yo[keep], o2, sample_weight=w[keep]))
    except Exception: pass
nest=np.array(nest); nlo,nhi=np.percentile(nest,[2.5,97.5])
print(f"   {best_name}: 재적합 포함 95% CI [{nlo:.3f},{nhi:.3f}]  (n={len(nest)} 복제)", flush=True)
rows.append(dict(learner=f"{best_name} (refit within each replicate)", oracle_AUROC=round(best_auc,4),
                 CI_lo=round(nlo,3), CI_hi=round(nhi,3), method="grouped CV, nested bootstrap with refit"))
pd.DataFrame(rows).to_csv(f"{RES}/review4_oracle_grouped.csv", index=False, encoding="utf-8-sig")

# ── 실험 레지스트리 ──
print("\n[레지스트리] 탐색 실험 수 집계", flush=True)
pats = [f"{RES}/path*.csv", f"{RES}/p1_*.csv", f"{RES}/round*.csv", f"{RES}/methods_explore.csv",
        f"{RES}/deeplearning_explore.csv", f"{RES}/ceiling_experiments.csv",
        f"{D}/../results/path*.csv", f"{D}/../results/p1_*.csv", f"{D}/../results/round*.csv"]
seen=set(); reg=[]
for pat in pats:
    for f in sorted(glob.glob(pat)):
        try: d=pd.read_csv(f)
        except Exception: continue
        key=os.path.basename(f)
        if key in seen: continue
        seen.add(key)
        reg.append(dict(file=key, rows=len(d), path=os.path.relpath(f, D).replace("\\","/")))
rg=pd.DataFrame(reg).sort_values("file")
rg.to_csv(f"{RES}/experiment_registry.csv", index=False, encoding="utf-8-sig")
print(f"   파일 {len(rg)}개 · 실험행 합계 {int(rg['rows'].sum()):,}", flush=True)
print("\n저장: results/review4_oracle_grouped.csv, experiment_registry.csv\n완료.", flush=True)
