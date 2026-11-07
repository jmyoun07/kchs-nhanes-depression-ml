# -*- coding: utf-8 -*-
"""70_primary_corrected.py — 교정된 주분석(primary) 일괄 산출.

지금까지 '한계로 공개'했던 것들을 주분석에서 실제로 제거한다.

(A) 데이터 교정
    · 소득분위: 코호트별 qcut(duplicates="drop") → 순위기반 고정 5단계 (post의 4단계 붕괴 제거)
    · 2017 KCHS 소득 거부코드(77/99) 결측 처리
    · 수면시간: KCHS 2020+ 의 7일 가중평균을 주중 기준으로 환산 → 전 연도·NHANES 동일 정의
    · 연령: 전 코호트 80세 상한(NHANES top-code와 정렬)
(B) 군집 단위 분할: hold-out 과 학습셋이 PSU 를 공유하지 않게 (year|strata|psu) 단위로 분할
(C) 운영 임계값: 개발셋에서 민감도 0.85 기준으로 확정 → 외부에 그대로 적용(사후선택 제거)
(D) 재보정 3단계: 절편만 → 절편+기울기 → Platt(온도) 비교, 모두 5-fold 교차적합
(E) 경량모델: 사전지정 0.02 여유 규칙만(외부정보 개입한 0.01 변형 제거)
(F) 공정성: 소득 하위군과 하위군 오류율 추가
(G) 무응답: DPQ 응답성향 역확률 가중 민감도

출력: results/primary2_*.csv
"""
import sys, os, warnings, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
import lightgbm as lgb

D = os.path.dirname(os.path.abspath(__file__)); RES = f"{D}/results"; SEED = 42
NUM=["age","sleep_hrs","bmi","income_q","srh","oral_health"]
BIN=["partnered","living_alone","current_drinker","active","htn","dm","employed","food_insecure"]
CAT=["sex","edu","smoking"]; FE=NUM+BIN+CAT
LGB=dict(num_leaves=31,min_child_samples=1000,learning_rate=0.01,n_estimators=900,reg_alpha=1.0,
         reg_lambda=20.0,subsample=0.6,colsample_bytree=0.6,random_state=SEED,n_jobs=-1,verbose=-1)
def make_ct():
    return ColumnTransformer([("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),NUM),
                              ("bin","passthrough",BIN),
                              ("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore",drop="first"))]),CAT)])

# ══════════════ (A) 데이터 교정 ══════════════
def rank_quintile(raw):
    r = pd.to_numeric(raw, errors="coerce")
    pct = r.rank(pct=True, method="average")
    return pd.cut(pct, [0,.2,.4,.6,.8,1.0], labels=False, include_lowest=True).where(r.notna())

def correct(d, is_kchs):
    o = d.copy()
    if is_kchs:
        raw = pd.to_numeric(o["income_raw"], errors="coerce")
        # 2017 소득은 서열 1-8 필드(후기 연도는 금액) → 거부코드 마스킹은 2017에만, 순위는 연도별로
        o["income_raw"] = raw.mask((o["year"] == 2017) & raw.isin([77, 99]))
        swe = pd.to_numeric(o.get("sleep_weekend"), errors="coerce")
        avg = pd.to_numeric(o["sleep_hrs"], errors="coerce")
        o["sleep_hrs"] = ((7*avg - 2*swe)/5).where(swe.notna() & avg.notna(), avg).clip(lower=0, upper=24)
        # 소득분위: 연도별 순위(척도가 연도 간 다르므로 통합 순위는 오류)
        o["income_q"] = o.groupby("year")["income_raw"].transform(lambda r: rank_quintile(r)).values
        # 음주 오코딩: 2020+ 는 code 1 = 연간 비음주인데 현재음주로 코딩됨 → 잔량 0인 건을 비음주로 정정
        cd = pd.to_numeric(o["current_drinker"], errors="coerce")
        qty = pd.to_numeric(o.get("alcohol_qty_drinks"), errors="coerce")
        bad = (o["year"] >= 2020) & (cd == 1) & (qty == 0)
        o["current_drinker"] = cd.mask(bad, 0.0)
        # 2018 신체활동은 다른 구성개념(가중 58.2% vs 인접연도 34-36%) → 결측 처리
        o["active"] = pd.to_numeric(o["active"], errors="coerce").mask(o["year"] == 2018)
    else:
        o["income_q"] = rank_quintile(o["income_raw"]).values
    o["age"] = pd.to_numeric(o["age"], errors="coerce").clip(upper=80)
    return o

K = correct(pd.read_parquet(f"{D}/data/ext_kchs.parquet"), True)
P = correct(pd.read_parquet(f"{D}/data/ext_pre.parquet"),  False)
S = correct(pd.read_parquet(f"{D}/data/ext_post.parquet"), False)
y = K["dep"].astype(int).values
print("[A] 교정 완료 — income_q 단계:", {n: int(pd.to_numeric(d['income_q']).nunique()) for n,d in [("KCHS",K),("pre",P),("post",S)]}, flush=True)
print("    수면 평균(주중 기준):", {n: round(float(np.average(pd.to_numeric(d['sleep_hrs']).dropna(), weights=d['W'].to_numpy(float)[pd.to_numeric(d['sleep_hrs']).notna().values])),2) for n,d in [("KCHS",K),("pre",P),("post",S)]}, flush=True)

# ══════════════ (B) 군집 단위 분할 ══════════════
key = (K["year"].astype(str)+"|"+K["strata"].astype(str)+"|"+K["psu"].astype(str)).values
cl, uniq = pd.factorize(key)
rng = np.random.RandomState(SEED)
order = rng.permutation(len(uniq))
sizes = np.bincount(cl, minlength=len(uniq))
cum = np.cumsum(sizes[order])
n_ho = int(np.searchsorted(cum, 100000) + 1)                 # hold-out 군집 수
ho_cl = set(order[:n_ho].tolist())
is_ho = np.isin(cl, list(ho_cl))
ho = np.where(is_ho)[0]
tr_pool = np.where(~is_ho)[0]
tr = rng.permutation(tr_pool)[:500000]
shared = len(set(cl[ho]) & set(cl[tr]))
print(f"[B] 군집 분할 — 학습 {len(tr):,} / hold-out {len(ho):,} · 공유 군집 {shared}개 (0이어야 정상)", flush=True)
Who, yho = K.iloc[ho]["W"].to_numpy(float), y[ho]
Wp, Ws = P["W"].to_numpy(float), S["W"].to_numpy(float)
yp, ys = P["dep"].astype(int).values, S["dep"].astype(int).values

model = Pipeline([("pre", make_ct()), ("clf", lgb.LGBMClassifier(**LGB))]).fit(K.iloc[tr][FE], y[tr])
p_ho = model.predict_proba(K.iloc[ho][FE])[:,1]
p_pre = model.predict_proba(P[FE])[:,1]; p_post = model.predict_proba(S[FE])[:,1]
p_tr = model.predict_proba(K.iloc[tr][FE])[:,1]

# ══════════════ 부트스트랩(재척도, 단일층 통합) ══════════════
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
def reps_for(d, strata, B=1000):
    rw=RaoWuBoot(strata, d["psu"].values, d["W"].to_numpy(float)); r=np.random.RandomState(SEED)
    return [rw.weights(r) for _ in range(B)]
def ci(yv, p, reps, fn=roc_auc_score):
    a=[]
    for w in reps:
        m=w>0
        if m.sum()<10 or yv[m].min()==yv[m].max(): continue
        try: a.append(fn(yv,p,sample_weight=w))
        except Exception: pass
    a=np.array(a); return float(np.percentile(a,2.5)), float(np.percentile(a,97.5))

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

# ══════════════ (D) 재보정 3단계 (교차적합) ══════════════
def recal(yv,p,w,mode,groups=None):
    """재보정은 PSU 단위로 교차적합한다. 같은 군집의 응답자가 보정학습과 평가에 함께 들어가면
    재보정 성능이 낙관적으로 나온다(코덱스 지적 13)."""
    lp=logit(p); oof=np.zeros_like(p)
    if groups is None:
        splitter=StratifiedKFold(5,shuffle=True,random_state=SEED).split(lp,yv)
    else:
        splitter=StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(lp,yv,groups=groups)
    for tri,tei in splitter:
        if mode=="intercept":
            a=0.0; l=lp[tri]; yy=yv[tri]; ww=w[tri]
            for _ in range(200):
                mu=1/(1+np.exp(-(a+l))); g=np.sum(ww*(yy-mu)); h=np.sum(ww*mu*(1-mu))
                if h<=0: break
                st=g/h; a+=st
                if abs(st)<1e-10: break
            oof[tei]=1/(1+np.exp(-(a+lp[tei])))
        elif mode=="slope":
            c=LogisticRegression(penalty=None,solver="lbfgs").fit(lp[tri].reshape(-1,1),yv[tri],sample_weight=w[tri])
            oof[tei]=c.predict_proba(lp[tei].reshape(-1,1))[:,1]
        else:  # platt/temperature: 기울기만(절편 고정 0) = 온도 조정
            best=None
            for T in np.linspace(0.4,3.0,53):
                q=1/(1+np.exp(-lp[tri]/T)); b=brier_score_loss(yv[tri],q,sample_weight=w[tri])
                if best is None or b<best[0]: best=(b,T)
            oof[tei]=1/(1+np.exp(-lp[tei]/best[1]))
    return oof

print("\n[C-D] 주요 지표 · CI · 재보정 3단계", flush=True)
REPS = {"Internal(hold-out)": reps_for(K.iloc[ho], (K.iloc[ho]["year"].astype(str)+"|"+K.iloc[ho]["strata"].astype(str)).values),
        "NHANES pre": reps_for(P, P["strata"].astype(str).values),
        "NHANES post": reps_for(S, S["strata"].astype(str).values)}
mrows=[]
for nm, d, yv, p, w in [("Internal(hold-out)",K.iloc[ho],yho,p_ho,Who),("NHANES pre",P,yp,p_pre,Wp),("NHANES post",S,ys,p_post,Ws)]:
    lo,hi = ci(yv,p,REPS[nm]); alo,ahi = ci(yv,p,REPS[nm],average_precision_score)
    row=dict(set=nm, AUROC=round(roc_auc_score(yv,p,sample_weight=w),6), CI_lo=round(lo,6), CI_hi=round(hi,6),
             AUPRC=round(average_precision_score(yv,p,sample_weight=w),6), AUPRC_lo=round(alo,6), AUPRC_hi=round(ahi,6),
             Brier=round(brier_score_loss(yv,p,sample_weight=w),6), ECE=round(ece(yv,p,w),6),
             cal_slope=round(cal_slope(yv,p,w),6), CITL=round(citl(yv,p,w),6))
    grp=(d["strata"].astype(str)+"|"+d["psu"].astype(str)).values
    for mode,lab in [("intercept","int"),("slope","is"),("platt","platt")]:
        pr=recal(yv,p,w,mode,groups=grp)
        row[f"recal_{lab}_slope"]=round(cal_slope(yv,pr,w),6); row[f"recal_{lab}_CITL"]=round(citl(yv,pr,w),6)
        row[f"recal_{lab}_AUROC"]=round(roc_auc_score(yv,pr,sample_weight=w),6); row[f"recal_{lab}_Brier"]=round(brier_score_loss(yv,pr,sample_weight=w),6)
        if mode=="slope": row["recal_AUPRC"]=round(average_precision_score(yv,pr,sample_weight=w),6); row["recal_ECE"]=round(ece(yv,pr,w),6)
    print(f"   {nm:<20} AUROC {row['AUROC']:.4f} [{lo:.4f},{hi:.4f}]  slope {row['cal_slope']:.3f} CITL {row['CITL']:+.3f}  "
          f"→ 절편만 {row['recal_int_slope']:.3f}/{row['recal_int_CITL']:+.3f} · 절편+기울기 {row['recal_is_slope']:.3f}/{row['recal_is_CITL']:+.3f} · 온도 {row['recal_platt_slope']:.3f}/{row['recal_platt_CITL']:+.3f}", flush=True)
    mrows.append(row)
pd.DataFrame(mrows).to_csv(f"{RES}/primary2_metrics.csv", index=False, encoding="utf-8-sig")

# ══════════════ (C) 개발셋 확정 임계값 → 외부 적용 ══════════════
print("\n[C] 개발셋에서 민감도 0.85 임계값 확정 → 외부에 그대로 적용", flush=True)
Wtr = K.iloc[tr]["W"].to_numpy(float); ytr = y[tr]
fpr,tpr,th = roc_curve(ytr, p_tr, sample_weight=Wtr)
idx = np.argmin(np.abs(tpr-0.85)); THR = float(th[idx])
def op(yv,p,w,t):
    pr=p>=t; tp=w[(yv==1)&pr].sum(); fp=w[(yv==0)&pr].sum(); fn=w[(yv==1)&~pr].sum(); tn=w[(yv==0)&~pr].sum()
    sens=tp/(tp+fn); spec=tn/(tn+fp); ppv=tp/(tp+fp) if tp+fp>0 else np.nan; npv=tn/(tn+fn) if tn+fn>0 else np.nan
    return dict(sens=round(sens,6),spec=round(spec,6),PPV=round(ppv,6),NPV=round(npv,6),
                flag=round(float(w[pr].sum()/w.sum()),6),NNS=round(1/ppv,1) if ppv>0 else np.nan,
                FPR=round(1-spec,6),FNR=round(1-sens,6))
orows=[dict(set="Development (threshold source)", threshold=round(THR,5), **op(ytr,p_tr,Wtr,THR))]
for nm,yv,p,w in [("Internal(hold-out)",yho,p_ho,Who),("NHANES pre",yp,p_pre,Wp),("NHANES post",ys,p_post,Ws)]:
    r=op(yv,p,w,THR); orows.append(dict(set=nm, threshold=round(THR,5), **r))
    print(f"   {nm:<20} sens {r['sens']:.3f} spec {r['spec']:.3f} PPV {r['PPV']:.3f} NPV {r['NPV']:.3f} NNS {r['NNS']}", flush=True)
pd.DataFrame(orows).to_csv(f"{RES}/primary2_operating.csv", index=False, encoding="utf-8-sig")

# ══════════════ 순편익 (개발셋 확정 임계·교차적합 재보정) ══════════════
nbrows=[]
for nm,yv,p,w,_d in [("Internal(hold-out)",yho,p_ho,Who,K.iloc[ho]),("NHANES pre",yp,p_pre,Wp,P),("NHANES post",ys,p_post,Ws,S)]:
    # 내부는 재보정하지 않는다(hold-out 라벨로 재보정하면 낙관적). 외부만 지역 재보정.
    # 재보정 교차적합은 PSU 단위로 한다(표 2·그림과 같은 기준, 코덱스 지적 11).
    _g = (_d["strata"].astype(str)+"|"+_d["psu"].astype(str)).values
    pc = p if nm.startswith("Internal") else recal(yv,p,w,"slope",groups=_g)
    prev=float(np.average(yv,weights=w)); W=w.sum()
    for pt in (0.10,0.15):
        nb=(w[(yv==1)&(pc>=pt)].sum()/W)-(w[(yv==0)&(pc>=pt)].sum()/W)*(pt/(1-pt))
        nball=prev-(1-prev)*(pt/(1-pt))
        nbrows.append(dict(set=nm, threshold=f"pt={pt:.2f}", net_benefit=round(nb,6), screen_all_NB=round(nball,6),
                           vs_screen_all=round(nb-nball,6), vs_best_default=round(nb-max(nball,0.0),6),
                           flag_rate=round(float(w[pc>=pt].sum()/W),6)))
pd.DataFrame(nbrows).to_csv(f"{RES}/primary2_netbenefit.csv", index=False, encoding="utf-8-sig")

# ══════════════ (F) 공정성: 소득·오류율 포함 ══════════════
print("\n[F] 하위군 — 소득·오류율 추가", flush=True)
dho=K.iloc[ho].copy(); dho["_p"]=p_ho; prd=P.copy(); prd["_p"]=p_pre; pod=S.copy(); pod["_p"]=p_post
age=pd.to_numeric(dho["age"]); iq=pd.to_numeric(dho["income_q"])
RACE={1:"Mexican American",2:"Other Hispanic",3:"Non-Hispanic White",4:"Non-Hispanic Black",6:"Non-Hispanic Asian",7:"Other/multiracial"}
groups=[("KCHS hold-out","Sex: female",dho,dho["sex"]=="F"),("KCHS hold-out","Sex: male",dho,dho["sex"]=="M"),
        ("KCHS hold-out","Age 20-39",dho,age<40),("KCHS hold-out","Age 40-59",dho,(age>=40)&(age<60)),("KCHS hold-out","Age 60+",dho,age>=60),
        ("KCHS hold-out","Education: college",dho,dho["edu"]=="college"),("KCHS hold-out","Education: high school",dho,dho["edu"]=="high"),
        ("KCHS hold-out","Education: middle",dho,dho["edu"]=="middle"),("KCHS hold-out","Education: <=elementary",dho,dho["edu"]=="elem_low")]
for q in range(5):
    groups.append(("KCHS hold-out",f"Income quintile {q+1}",dho,iq==q))
for c,l in RACE.items():
    groups.append(("NHANES pre",f"Race: {l}",prd,prd["race"]==c)); groups.append(("NHANES post",f"Race: {l}",pod,pod["race"]==c))
COHREP={"KCHS hold-out":REPS["Internal(hold-out)"],"NHANES pre":REPS["NHANES pre"],"NHANES post":REPS["NHANES post"]}
SRC={"KCHS hold-out":dho,"NHANES pre":prd,"NHANES post":pod}
frows=[]
for coh,lab,src,mask in groups:
    m=np.asarray(mask); dd=src[m]
    if dd["dep"].nunique()<2 or len(dd)<100: continue
    yv=dd["dep"].astype(int).values; p=dd["_p"].values; w=dd["W"].to_numpy(float)
    lo,hi=ci(yv,p,[r[m] for r in COHREP[coh]])
    o=op(yv,p,w,THR)
    frows.append(dict(cohort=coh,subgroup=lab,n=len(dd),AUROC=round(roc_auc_score(yv,p,sample_weight=w),6),
                      CI_lo=round(lo,6),CI_hi=round(hi,6),cal_slope=round(cal_slope(yv,p,w),6),CITL=round(citl(yv,p,w),6),
                      FPR=o["FPR"],FNR=o["FNR"]))
    print(f"   {coh:<14} {lab:<26} n={len(dd):>6,} AUROC {frows[-1]['AUROC']:.3f} [{lo:.3f},{hi:.3f}] FPR {o['FPR']:.3f} FNR {o['FNR']:.3f}", flush=True)
pd.DataFrame(frows).to_csv(f"{RES}/primary2_fairness.csv", index=False, encoding="utf-8-sig")

# ══════════════ (E) 경량모델 — 사전지정 0.02 규칙만 ══════════════
print("\n[E] 경량모델 — 학습셋을 군집 단위로 적합/선택 분할, 사전지정 0.02 여유", flush=True)
import shap
# 적합셋과 선택셋도 군집 단위로 나눈다(같은 PSU 가 양쪽에 걸치면 선택이 낙관적이 된다, 코덱스 지적 15)
_tr_cl = cl[tr]
_uc = np.unique(_tr_cl); _rs = np.random.RandomState(SEED)
_perm = _rs.permutation(len(_uc))
_sizes = np.array([(_tr_cl == c).sum() for c in _uc])[_perm]
_ncl = int(np.searchsorted(np.cumsum(_sizes), 50000) + 1)
_sel_cl = set(_uc[_perm[:_ncl]].tolist())
_is_sel = np.isin(_tr_cl, list(_sel_cl))
sel_idx, fit_idx = tr[_is_sel], tr[~_is_sel]
print(f"   적합 {len(fit_idx):,} / 선택 {len(sel_idx):,} · 공유 군집 "
      f"{len(set(cl[fit_idx]) & set(cl[sel_idx]))}개 (0이어야 정상)", flush=True)
full = Pipeline([("pre",make_ct()),("clf",lgb.LGBMClassifier(**LGB))]).fit(K.iloc[fit_idx][FE], y[fit_idx])
samp = K.iloc[np.random.RandomState(SEED).choice(fit_idx,20000,replace=False)]
Xs = full.named_steps["pre"].transform(samp[FE]); names=[n.split("__")[-1] for n in full.named_steps["pre"].get_feature_names_out()]
sv = shap.TreeExplainer(full.named_steps["clf"]).shap_values(Xs)
if isinstance(sv,list): sv=sv[1]
imp=pd.Series(np.abs(sv).mean(0),index=names)
order=pd.Series({v: imp[[i for i in imp.index if i==v or i.startswith(v+"_")]].sum() for v in FE}).sort_values(ascending=False)
order.to_csv(f"{RES}/primary2_shap.csv", encoding="utf-8-sig")
Wsel=K.iloc[sel_idx]["W"].to_numpy(float); ysel=y[sel_idx]
prows=[]
for k in range(1,18):
    f=list(order.index[:k]); NU=[c for c in NUM if c in f]; BI=[c for c in BIN if c in f]; CA=[c for c in CAT if c in f]
    ctk=ColumnTransformer([t for t in [("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),NU) if NU else None,
        ("bin","passthrough",BI) if BI else None,
        ("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore",drop="first"))]),CA) if CA else None] if t])
    FEk=NU+BI+CA
    m=Pipeline([("pre",ctk),("clf",lgb.LGBMClassifier(**LGB))]).fit(K.iloc[fit_idx][FEk], y[fit_idx])
    sel=roc_auc_score(ysel,m.predict_proba(K.iloc[sel_idx][FEk])[:,1],sample_weight=Wsel)
    hoa=roc_auc_score(yho,m.predict_proba(K.iloc[ho][FEk])[:,1],sample_weight=Who)
    pa=roc_auc_score(yp,m.predict_proba(P[FEk])[:,1],sample_weight=Wp); sa=roc_auc_score(ys,m.predict_proba(S[FEk])[:,1],sample_weight=Ws)
    prows.append(dict(k=k,added=f[-1],vars=";".join(f),selection_AUROC=round(sel,6),holdout_AUROC=round(hoa,6),ext_pre=round(pa,6),ext_post=round(sa,6)))
    print(f"   k={k:>2} 선택 {sel:.4f} | hold-out {hoa:.4f} pre {pa:.4f} post {sa:.4f}  +{f[-1]}", flush=True)
pp=pd.DataFrame(prows); pp.to_csv(f"{RES}/primary2_parsimonious.csv", index=False, encoding="utf-8-sig")
fullsel=float(pp.loc[pp.k==17,"selection_AUROC"].iloc[0]); cand=pp[(pp.k<17)&(pp.selection_AUROC>=fullsel-0.02)]
kstar=int(cand.k.min()); print(f"   → 사전지정 0.02 규칙: k*={kstar}", flush=True)

# ══════════════ 14 vs 17 (짝지음, 중심화 P) ══════════════
NUM14=["age","sleep_hrs","bmi","income_q","srh"]; BIN14=["partnered","living_alone","current_drinker","active","htn","dm"]; CAT14=["sex","edu","smoking"]
FE14=NUM14+BIN14+CAT14
ct14=ColumnTransformer([("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),NUM14),
    ("bin","passthrough",BIN14),("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore",drop="first"))]),CAT14)])
LGB14=dict(LGB); LGB14.update(num_leaves=15,min_child_samples=500,reg_lambda=1.0,subsample=0.8)
m14=Pipeline([("pre",ct14),("clf",lgb.LGBMClassifier(**LGB14))]).fit(K.iloc[tr][FE14], y[tr])
grows=[]
print("\n[14→17] 교정 데이터에서 재추정", flush=True)
for nm,d,yv,w,p17 in [("NHANES pre",P,yp,Wp,p_pre),("NHANES post",S,ys,Ws,p_post)]:
    p14=m14.predict_proba(d[FE14])[:,1]
    obs=roc_auc_score(yv,p17,sample_weight=w)-roc_auc_score(yv,p14,sample_weight=w)
    ds=[]
    for wr in REPS[nm]:
        mm=wr>0
        if yv[mm].min()==yv[mm].max(): continue
        try: ds.append(roc_auc_score(yv,p17,sample_weight=wr)-roc_auc_score(yv,p14,sample_weight=wr))
        except Exception: pass
    ds=np.array(ds); lo,hi=np.percentile(ds,[2.5,97.5]); cen=ds-ds.mean()
    pv=(1+np.sum(np.abs(cen)>=abs(obs)))/(len(ds)+1)
    print(f"   {nm}: 14 {roc_auc_score(yv,p14,sample_weight=w):.4f} → 17 {roc_auc_score(yv,p17,sample_weight=w):.4f}  Δ={obs:+.4f} [{lo:+.4f},{hi:+.4f}] P={pv:.4f}", flush=True)
    grows.append(dict(set=nm,AUROC_14=round(roc_auc_score(yv,p14,sample_weight=w),6),AUROC_17=round(roc_auc_score(yv,p17,sample_weight=w),6),
                      delta=round(obs,6),CI_lo=round(lo,6),CI_hi=round(hi,6),P=round(pv,4)))
pd.DataFrame(grows).to_csv(f"{RES}/primary2_gain.csv", index=False, encoding="utf-8-sig")

# ══════════════ 민감도 + (G) 무응답 응답성향 가중 ══════════════
print("\n[민감도]", flush=True)
def quick(FEs,label,Kx=None,Px=None,Sx=None,wtr=None):
    Kx=K if Kx is None else Kx; Px=P if Px is None else Px; Sx=S if Sx is None else Sx
    NU=[c for c in NUM if c in FEs]; BI=[c for c in BIN if c in FEs]; CA=[c for c in CAT if c in FEs]
    c=ColumnTransformer([t for t in [("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),NU) if NU else None,
        ("bin","passthrough",BI) if BI else None,
        ("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore",drop="first"))]),CA) if CA else None] if t])
    # 표본가중치는 평균 1로 정규화한다: KCHS 가중치는 ~1e-5 규모라 정규화하지 않으면
    # LightGBM의 min_sum_hessian_in_leaf(기본 1e-3)에 걸려 분할이 일어나지 않고 상수예측이 된다.
    kw={} if wtr is None else {"clf__sample_weight": np.asarray(wtr,float)/np.mean(wtr)}
    m=Pipeline([("pre",c),("clf",lgb.LGBMClassifier(**LGB))]).fit(Kx.iloc[tr][FEs], y[tr], **kw)
    i=roc_auc_score(yho,m.predict_proba(Kx.iloc[ho][FEs])[:,1],sample_weight=Who)
    p=roc_auc_score(yp,m.predict_proba(Px[FEs])[:,1],sample_weight=Wp)
    s=roc_auc_score(ys,m.predict_proba(Sx[FEs])[:,1],sample_weight=Ws)
    print(f"   {label:<40} 내부 {i:.4f} pre {p:.4f} post {s:.4f}", flush=True)
    return dict(cond=label,internal=round(i,6),ext_pre=round(p,6),ext_post=round(s,6))
srows=[quick(FE,"Primary (17 predictors, corrected)")]
srows.append(quick([c for c in FE if c!="food_insecure"],"Food insecurity excluded (16)"))
srows.append(quick(FE14,"Three added predictors excluded (14)"))
srows.append(quick(FE,"Weighted training",wtr=K.iloc[tr]["W"].to_numpy(float)))
# 응답성향 역확률 가중(외부 평가 가중치 조정)
for nm,d,yv,p,w in [("NHANES pre",P,yp,p_pre,Wp),("NHANES post",S,ys,p_post,Ws)]:
    pass
np_rows=[]
for nm,d,yv,p,w in [("NHANES pre",P,yp,p_pre,Wp),("NHANES post",S,ys,p_post,Ws)]:
    # 응답성향: 학력별 무응답률(표 S8 정의)로 역확률 보정. 인종은 이 보정에 쓰지 않는다.
    # 응답률은 표 S8 정의(≥3문항 결측)로 계산한 정본에서 읽는다 — 예전 하드코딩 값은 any-missing 기준이었다
    _nr = pd.read_csv(f"{RES}/primary2_dpq_nonresponse.csv")
    _pd_ = "pre" if "pre" in nm else "post"
    resp = {g: 1 - float(_nr[(_nr.period==_pd_)&(_nr.group==g)].nonresponse_ge3_pct.iloc[0])/100
            for g in ("<HS","HS","College")}
    edu = d["edu"].map({"elem_low":"<HS","middle":"<HS","high":"HS","college":"College"}).fillna("College")
    w2 = w / edu.map(resp).to_numpy(float)
    a=roc_auc_score(yv,p,sample_weight=w2)
    np_rows.append(dict(set=nm, AUROC_ipw=round(a,6), AUROC_base=round(roc_auc_score(yv,p,sample_weight=w),6)))
    print(f"   응답성향 역확률 가중 {nm:<14} {a:.4f} (기준 {roc_auc_score(yv,p,sample_weight=w):.4f})", flush=True)
pd.DataFrame(np_rows).to_csv(f"{RES}/primary2_nonresponse_ipw.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(srows).to_csv(f"{RES}/primary2_sensitivity.csv", index=False, encoding="utf-8-sig")

pd.DataFrame([dict(train_n=len(tr), holdout_n=len(ho), shared_clusters=shared,
                   holdout_clusters=int(pd.Series(cl[ho]).nunique()), threshold=round(THR,5), kstar=kstar)]
            ).to_csv(f"{RES}/primary2_design.csv", index=False, encoding="utf-8-sig")
print("\n저장: results/primary2_*.csv\n완료.", flush=True)
