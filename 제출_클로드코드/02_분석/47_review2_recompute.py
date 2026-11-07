# -*- coding: utf-8 -*-
"""47_review2_recompute.py — 2차 동료심사 지적사항 재계산 (통계·수치 부분).

(1) Rao-Wu-Yue 재척도 군집 부트스트랩으로 모든 CI 재산출 (층당 PSU=2에서 단순 군집 부트스트랩은 하향 편향)
(2) 짝지음 부트스트랩 P를 중심화(centered) + Monte-Carlo 보정으로 재계산
(3) Youden 지점을 격자탐색이 아니라 관측 점수 전체에서 정확히 산출
(4) 식품불안정 유병률을 available-case로 재산출(결측을 '안정'으로 취급하던 오류)
(5) ECE를 파일로 영속화
(6) NHANES post 하위군(인종) 공정성 추가 + 모든 하위군 CITL 산출
(7) hold-out과 학습셋의 PSU 중첩 정량화(내부 추정치의 성격 규정)

출력: results/review2_{metrics,operating,fairness,prevalence,overlap}.csv
"""
import sys, warnings, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
import lightgbm as lgb

D = r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model"; RES = f"{D}/results"; SEED = 42
kchs = pd.read_parquet(f"{D}/data/ext_kchs.parquet"); pre = pd.read_parquet(f"{D}/data/ext_pre.parquet"); post = pd.read_parquet(f"{D}/data/ext_post.parquet")
NUM = ["age","sleep_hrs","bmi","income_q","srh","oral_health"]
BIN = ["partnered","living_alone","current_drinker","active","htn","dm","employed","food_insecure"]
CAT = ["sex","edu","smoking"]; FE = NUM+BIN+CAT
y = kchs["dep"].astype(int).values
LGB = dict(num_leaves=31, min_child_samples=1000, learning_rate=0.01, n_estimators=900, reg_alpha=1.0,
           reg_lambda=20.0, subsample=0.6, colsample_bytree=0.6, random_state=SEED, n_jobs=-1, verbose=-1)
LGB14 = dict(num_leaves=15, min_child_samples=500, learning_rate=0.01, n_estimators=900, reg_alpha=1.0,
             reg_lambda=1.0, subsample=0.8, colsample_bytree=0.6, random_state=SEED, n_jobs=-1, verbose=-1)
NUM14 = ["age","sleep_hrs","bmi","income_q","srh"]; BIN14 = ["partnered","living_alone","current_drinker","active","htn","dm"]
CAT14 = ["sex","edu","smoking"]; FE14 = NUM14+BIN14+CAT14
def make_ct(NU=NUM, BI=BIN, CA=CAT):
    return ColumnTransformer([("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), NU),
                              ("bin", "passthrough", BI),
                              ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")), ("o", OneHotEncoder(handle_unknown="ignore", drop="first"))]), CA)])

# ══════════════════ Rao-Wu-Yue 재척도 군집 부트스트랩 ══════════════════
class RaoWuBoot:
    """층 h에서 m_h = n_h - 1 개의 PSU를 복원추출하고 가중치를 재척도한다.
       w* = w · (n_h/m_h) · r_hi   (m_h = n_h-1 일 때 표준 재척도 부트스트랩; n_h=2면 half-sampling)
       n_h = 1 인 층은 재표집하지 않고 가중치를 그대로 둔다(분산 기여 0)."""
    def __init__(self, strata, psu, w):
        st = pd.Series(strata).astype(str).values
        cl = pd.factorize(pd.Series(st) + "|" + pd.Series(psu).astype(str))[0]
        sid = pd.factorize(pd.Series(st))[0]
        self.cl = cl; self.w = np.asarray(w, float)
        C = cl.max()+1
        cl_stratum = np.zeros(C, dtype=int); cl_stratum[cl] = sid
        self.cl_stratum = cl_stratum
        self.n_h = np.bincount(cl_stratum)                    # 층별 PSU 수
        self.m_h = np.maximum(self.n_h - 1, 1)
        self.C = C
        # 층별 군집 인덱스 범위(층순 정렬)
        o = np.argsort(cl_stratum, kind="stable"); self.cl_sorted = o
        s_sorted = cl_stratum[o]; sc = np.bincount(s_sorted)
        self.st_start = np.concatenate([[0], np.cumsum(sc)])[:-1]
        self.slot_start = self.st_start[s_sorted]; self.slot_n = sc[s_sorted]
        self.singleton = self.n_h[cl_stratum] == 1
    def weights(self, rng):
        r = np.zeros(self.C)
        # 각 층에서 m_h개를 복원추출 → 군집별 선택 횟수 r_hi
        draws = rng.random_sample(self.C)                      # 슬롯당 1회 = 총 C회는 과다 → 층별 m_h만 사용
        # 층별로 m_h개만 뽑기 위해 슬롯 인덱스를 층 내 순번으로 변환
        idx_in_stratum = np.arange(self.C) - self.slot_start
        keep = idx_in_stratum < self.m_h[self.cl_stratum[self.cl_sorted]]
        sel_slots = np.where(keep)[0]
        pick = self.slot_start[sel_slots] + (draws[sel_slots] * self.slot_n[sel_slots]).astype(int)
        np.add.at(r, self.cl_sorted[pick], 1.0)
        scale = np.where(self.singleton, 1.0, (self.n_h[self.cl_stratum] / self.m_h[self.cl_stratum]) * r)
        return self.w * scale[self.cl]

def boot_ci(d, p, fn, B=1000, seed=SEED):
    yv = d["dep"].astype(int).values
    rw = RaoWuBoot(d["strata"].values, d["psu"].values, d["W"].to_numpy(float))
    rng = np.random.RandomState(seed); a = []
    for _ in range(B):
        w = rw.weights(rng)
        m = w > 0
        if yv[m].min() == yv[m].max(): continue
        try: a.append(fn(yv, p, w))
        except Exception: pass
    a = np.array(a); return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)), a

AUC = lambda yv,p,w: roc_auc_score(yv,p,sample_weight=w)
APR = lambda yv,p,w: average_precision_score(yv,p,sample_weight=w)

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
def ece(yv,p,w,bins=10):
    q=np.quantile(p,np.linspace(0,1,bins+1)); q[0]-=1e-9; e=0.0; W=w.sum()
    for i in range(bins):
        m=(p>q[i])&(p<=q[i+1])
        if m.sum()==0: continue
        e+=w[m].sum()/W*abs(np.average(yv[m],weights=w[m])-np.average(p[m],weights=w[m]))
    return float(e)

tr, ho = train_test_split(np.arange(len(kchs)), train_size=500000, test_size=100000, stratify=y, random_state=SEED)
model = Pipeline([("pre", make_ct()), ("clf", lgb.LGBMClassifier(**LGB))]).fit(kchs.iloc[tr][FE], y[tr])
dho = kchs.iloc[ho].copy(); dho["_p"] = model.predict_proba(dho[FE])[:,1]
prd = pre.copy();  prd["_p"]  = model.predict_proba(pre[FE])[:,1]
pod = post.copy(); pod["_p"]  = model.predict_proba(post[FE])[:,1]
print(f"[재현] 내부 {AUC(dho.dep.astype(int).values,dho._p.values,dho.W.to_numpy(float)):.4f} / "
      f"pre {AUC(prd.dep.astype(int).values,prd._p.values,prd.W.to_numpy(float)):.4f} / "
      f"post {AUC(pod.dep.astype(int).values,pod._p.values,pod.W.to_numpy(float)):.4f}", flush=True)

# ══════ (7) hold-out과 학습셋의 PSU 중첩 ══════
key = (kchs["year"].astype(str)+"|"+kchs["strata"].astype(str)+"|"+kchs["psu"].astype(str)).values  # 층은 연도별로 정의됨
tr_cl = set(np.unique(key[tr])); ho_cl = np.unique(key[ho])
shared = sum(1 for c in ho_cl if c in tr_cl)
rec_shared = int(np.isin(key[ho], list(tr_cl)).sum())
print(f"\n[중첩] hold-out 군집 {len(ho_cl):,}개 중 학습셋과 공유 {shared:,}개 ({shared/len(ho_cl)*100:.1f}%) · "
      f"레코드 {rec_shared:,}/{len(ho):,} ({rec_shared/len(ho)*100:.1f}%)", flush=True)
pd.DataFrame([dict(holdout_clusters=len(ho_cl), shared_clusters=shared, pct_clusters=round(shared/len(ho_cl)*100,1),
                   holdout_records=len(ho), shared_records=rec_shared, pct_records=round(rec_shared/len(ho)*100,1))]
             ).to_csv(f"{RES}/review2_overlap.csv", index=False, encoding="utf-8-sig")

# ══════ (1) 식품불안정 등 available-case 유병률 ══════
print("\n[유병률] food_insecure — 결측 제외 available-case", flush=True)
prows=[]
for nm,d in [("KCHS",kchs),("NHANES pre",pre),("NHANES post",post)]:
    v=pd.to_numeric(d["food_insecure"],errors="coerce"); w=d["W"].to_numpy(float); m=v.notna().values
    naive=float(np.average((v.fillna(0)==1).values,weights=w))
    avail=float(np.average((v[m]==1).values,weights=w[m]))
    print(f"   {nm:<12} 결측 {100*(1-m.mean()):.2f}%  |  naive {naive*100:.2f}%  available-case {avail*100:.2f}%", flush=True)
    prows.append(dict(cohort=nm, missing_pct=round(100*(1-m.mean()),2), naive_pct=round(naive*100,2), available_case_pct=round(avail*100,2)))
pd.DataFrame(prows).to_csv(f"{RES}/review2_prevalence.csv", index=False, encoding="utf-8-sig")

# ══════ (1b,3) 정확 Youden + 참조 임계값 ══════
print("\n[운영지점] 관측 점수 전체에서 정확 Youden 재산출", flush=True)
def op(yv,p,w,t):
    pr=p>=t; tp=w[(yv==1)&pr].sum(); fp=w[(yv==0)&pr].sum(); fn=w[(yv==1)&~pr].sum(); tn=w[(yv==0)&~pr].sum()
    return tp/(tp+fn), tn/(tn+fp), (tp/(tp+fp) if tp+fp>0 else np.nan), (tn/(tn+fn) if tn+fn>0 else np.nan)
orows=[]
for nm,d in [("Internal",dho),("NHANES pre",prd),("NHANES post",pod)]:
    yv=d.dep.astype(int).values; p=d._p.values; w=d.W.to_numpy(float)
    fpr,tpr,th=roc_curve(yv,p,sample_weight=w); j=np.argmax(tpr-fpr); t=float(th[j])
    s,sp,ppv,npv=op(yv,p,w,t)
    prev=float(np.average(yv,weights=w))
    s2,sp2,ppv2,npv2=op(yv,p,w,prev)
    print(f"   {nm:<12} Youden t={t:.5f}  sens {s:.3f} spec {sp:.3f} PPV {ppv:.3f} NPV {npv:.3f}", flush=True)
    orows.append(dict(set=nm,threshold="Youden (exact)",value=round(t,5),sens=round(s,3),spec=round(sp,3),PPV=round(ppv,3),NPV=round(npv,3)))
    orows.append(dict(set=nm,threshold="prevalence",value=round(prev,5),sens=round(s2,3),spec=round(sp2,3),PPV=round(ppv2,3),NPV=round(npv2,3)))
pd.DataFrame(orows).to_csv(f"{RES}/review2_operating.csv", index=False, encoding="utf-8-sig")

# ══════ (1a,5) 주요 지표 + 재척도 부트스트랩 CI + ECE 영속화 ══════
print("\n[주요지표] Rao-Wu-Yue 재척도 부트스트랩 CI + ECE", flush=True)
def recalib_cv(yv,p,w):
    lp=logit(p); oof=np.zeros_like(p)
    for tri,tei in StratifiedKFold(5,shuffle=True,random_state=SEED).split(lp,yv):
        c=LogisticRegression(penalty=None,solver="lbfgs").fit(lp[tri].reshape(-1,1),yv[tri],sample_weight=w[tri])
        oof[tei]=c.predict_proba(lp[tei].reshape(-1,1))[:,1]
    return oof
mrows=[]
for nm,d in [("Internal(hold-out)",dho),("NHANES pre",prd),("NHANES post",pod)]:
    yv=d.dep.astype(int).values; p=d._p.values; w=d.W.to_numpy(float)
    lo,hi,_=boot_ci(d,p,AUC); alo,ahi,_=boot_ci(d,p,APR)
    pr=recalib_cv(yv,p,w)
    print(f"   {nm:<20} AUROC {AUC(yv,p,w):.4f} [{lo:.4f},{hi:.4f}]  AUPRC {APR(yv,p,w):.3f} [{alo:.3f},{ahi:.3f}]  ECE {ece(yv,p,w):.4f}", flush=True)
    mrows.append(dict(set=nm,AUROC=round(AUC(yv,p,w),4),CI_lo=round(lo,4),CI_hi=round(hi,4),
                      AUPRC=round(APR(yv,p,w),3),AUPRC_lo=round(alo,3),AUPRC_hi=round(ahi,3),
                      Brier=round(brier_score_loss(yv,p,sample_weight=w),4),ECE=round(ece(yv,p,w),4),
                      cal_slope=round(cal_slope(yv,p,w),3),CITL=round(citl(yv,p,w),3),
                      recal_AUROC=round(AUC(yv,pr,w),4),recal_AUPRC=round(APR(yv,pr,w),3),
                      recal_Brier=round(brier_score_loss(yv,pr,sample_weight=w),4),
                      recal_slope=round(cal_slope(yv,pr,w),3),recal_CITL=round(citl(yv,pr,w),3),
                      recal_ECE=round(ece(yv,pr,w),4)))
pd.DataFrame(mrows).to_csv(f"{RES}/review2_metrics.csv", index=False, encoding="utf-8-sig")

# ══════ (6) 하위군: pre+post 인종 + 모든 하위군 CITL ══════
print("\n[하위군] AUROC(재척도 CI) · 보정 slope · CITL — NHANES post 포함", flush=True)
age=pd.to_numeric(dho["age"])
RACE={1:"Mexican American",2:"Other Hispanic",3:"Non-Hispanic White",4:"Non-Hispanic Black",6:"Non-Hispanic Asian",7:"Other/multiracial"}
groups=[("KCHS hold-out","Sex: female",dho,dho["sex"]=="F"),("KCHS hold-out","Sex: male",dho,dho["sex"]=="M"),
        ("KCHS hold-out","Age 20-39",dho,age<40),("KCHS hold-out","Age 40-59",dho,(age>=40)&(age<60)),("KCHS hold-out","Age 60+",dho,age>=60),
        ("KCHS hold-out","Education: college",dho,dho["edu"]=="college"),("KCHS hold-out","Education: high school",dho,dho["edu"]=="high"),
        ("KCHS hold-out","Education: middle",dho,dho["edu"]=="middle"),("KCHS hold-out","Education: <=elementary",dho,dho["edu"]=="elem_low")]
for code,lab in RACE.items():
    groups.append(("NHANES pre",f"Race: {lab}",prd,prd["race"]==code))
    groups.append(("NHANES post",f"Race: {lab}",pod,pod["race"]==code))
frows=[]
for coh,lab,src,mask in groups:
    dd=src[mask]
    if dd["dep"].nunique()<2 or len(dd)<100:
        print(f"   {coh:<12} {lab:<28} n={len(dd):>6,}  (표본 부족, 생략)", flush=True); continue
    yv=dd.dep.astype(int).values; p=dd._p.values; w=dd.W.to_numpy(float)
    a=AUC(yv,p,w); lo,hi,_=boot_ci(dd,p,AUC,B=600)
    frows.append(dict(cohort=coh,subgroup=lab,n=len(dd),AUROC=round(a,3),CI_lo=round(lo,3),CI_hi=round(hi,3),
                      cal_slope=round(cal_slope(yv,p,w),3),CITL=round(citl(yv,p,w),3)))
    print(f"   {coh:<12} {lab:<28} n={len(dd):>6,}  AUROC {a:.3f} [{lo:.3f},{hi:.3f}]  slope {frows[-1]['cal_slope']:.3f}  CITL {frows[-1]['CITL']:+.3f}", flush=True)
pd.DataFrame(frows).to_csv(f"{RES}/review2_fairness.csv", index=False, encoding="utf-8-sig")

# ══════ (2) 14→17 짝지음: 재척도 부트스트랩 + 중심화 P ══════
print("\n[14→17] 재척도 부트스트랩 + 중심화 P", flush=True)
m14 = Pipeline([("pre", make_ct(NUM14,BIN14,CAT14)), ("clf", lgb.LGBMClassifier(**LGB14))]).fit(kchs.iloc[tr][FE14], y[tr])
grows=[]
for nm,d,src in [("NHANES pre",prd,pre),("NHANES post",pod,post)]:
    yv=d.dep.astype(int).values; w=d.W.to_numpy(float)
    p17=d._p.values; p14=m14.predict_proba(src[FE14])[:,1]
    obs=AUC(yv,p17,w)-AUC(yv,p14,w)
    rw=RaoWuBoot(d["strata"].values,d["psu"].values,w); rng=np.random.RandomState(SEED); ds=[]
    for _ in range(2000):
        ww=rw.weights(rng); m=ww>0
        if yv[m].min()==yv[m].max(): continue
        try: ds.append(AUC(yv,p17,ww)-AUC(yv,p14,ww))
        except Exception: pass
    ds=np.array(ds); lo,hi=np.percentile(ds,[2.5,97.5])
    cen=ds-ds.mean(); pval=(1+np.sum(np.abs(cen)>=abs(obs)))/(len(ds)+1)
    print(f"   {nm}: 14 {AUC(yv,p14,w):.4f} → 17 {AUC(yv,p17,w):.4f}  Δ={obs:+.4f} 95%CI[{lo:+.4f},{hi:+.4f}]  P={pval:.4f}", flush=True)
    grows.append(dict(set=nm,AUROC_14=round(AUC(yv,p14,w),4),AUROC_17=round(AUC(yv,p17,w),4),delta=round(obs,4),
                      CI_lo=round(lo,4),CI_hi=round(hi,4),P=round(pval,4),B=len(ds)))
pd.DataFrame(grows).to_csv(f"{RES}/review2_gain.csv", index=False, encoding="utf-8-sig")
print("\n저장: results/review2_*.csv\n완료.", flush=True)
