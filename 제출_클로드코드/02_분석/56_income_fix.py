# -*- coding: utf-8 -*-
"""56_income_fix.py — 소득분위(income_q) 조화 결함 정량화 및 교정판 비교.

결함: src/01_harmonize.py 는 코호트별로 pd.qcut(income_raw, 5, duplicates="drop") 를 쓴다.
      NHANES post 는 INDFMPIR 상단이 5.00 에 top-code 되어 23.8% 가 동일값이므로 상위 두 분위가
      조용히 합쳐져 **4단계**만 생성된다(0..3). income_q 는 단일 수치 서열변수로 투입되므로,
      post 코호트의 상위 절반이 학습셋의 '4분위' 수준으로 사상된다.

교정: income_raw 의 코호트 내 백분위 순위를 고정 절단점(0.2/0.4/0.6/0.8)에 사상해 항상 5단계를 만든다
      (top-code 동일값은 평균 순위로 처리 → 동일값 집단이 여러 분위에 걸치면 중앙 분위로 배정).

출력: results/income_fix_diagnostic.csv, results/income_fix_sensitivity.csv
"""
import sys, os, warnings, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

D = os.path.dirname(os.path.abspath(__file__)); RES = f"{D}/results"; SEED = 42
kchs = pd.read_parquet(f"{D}/data/ext_kchs.parquet"); pre = pd.read_parquet(f"{D}/data/ext_pre.parquet"); post = pd.read_parquet(f"{D}/data/ext_post.parquet")
NUM=["age","sleep_hrs","bmi","income_q","srh","oral_health"]
BIN=["partnered","living_alone","current_drinker","active","htn","dm","employed","food_insecure"]
CAT=["sex","edu","smoking"]; FE=NUM+BIN+CAT
LGB=dict(num_leaves=31,min_child_samples=1000,learning_rate=0.01,n_estimators=900,reg_alpha=1.0,
         reg_lambda=20.0,subsample=0.6,colsample_bytree=0.6,random_state=SEED,n_jobs=-1,verbose=-1)
def make_ct():
    return ColumnTransformer([("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),NUM),
                              ("bin","passthrough",BIN),
                              ("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore",drop="first"))]),CAT)])

# ── 진단 ──
print("="*92); print("[진단] 코호트별 income_q 실제 단계 수와 분포"); print("="*92, flush=True)
diag=[]
for nm,d in [("KCHS",kchs),("NHANES pre",pre),("NHANES post",post)]:
    v=pd.to_numeric(d["income_q"],errors="coerce"); w=d["W"].to_numpy(float)
    lv=sorted(v.dropna().unique().tolist())
    shares={int(k): round(float(w[(v==k).values].sum()/w[v.notna().values].sum())*100,1) for k in lv}
    raw=pd.to_numeric(d["income_raw"],errors="coerce")
    topmass=float((raw==raw.max()).mean()*100) if raw.notna().any() else np.nan
    print(f"   {nm:<12} 단계 {len(lv)}개 {lv}  가중분포 {shares}  · income_raw 최대값 동일비중 {topmass:.1f}%", flush=True)
    diag.append(dict(cohort=nm, n_levels=len(lv), levels=str(lv), weighted_shares=str(shares), top_code_share_pct=round(topmass,1)))
pd.DataFrame(diag).to_csv(f"{RES}/income_fix_diagnostic.csv", index=False, encoding="utf-8-sig")

# ── 교정판: 백분위 순위 → 고정 절단점 ──
def fixed_quintile(raw, w=None):
    r = pd.to_numeric(raw, errors="coerce")
    pct = r.rank(pct=True, method="average")
    q = pd.cut(pct, [0, .2, .4, .6, .8, 1.0], labels=False, include_lowest=True)
    return pd.Series(q, index=r.index).where(r.notna())

print("\n[교정] 백분위 순위 기반 5단계 재생성", flush=True)
fx={}
for nm,d in [("KCHS",kchs),("NHANES pre",pre),("NHANES post",post)]:
    q=fixed_quintile(d["income_raw"])
    lv=sorted(q.dropna().unique().tolist())
    w=d["W"].to_numpy(float)
    shares={int(k): round(float(w[(q==k).values].sum()/w[q.notna().values].sum())*100,1) for k in lv}
    print(f"   {nm:<12} 단계 {len(lv)}개 {[int(x) for x in lv]}  가중분포 {shares}", flush=True)
    fx[nm]=q

# ── 민감도: 교정 소득분위로 재적합·재평가 ──
print("\n[민감도] 교정 income_q 로 모델 재적합 후 비교", flush=True)
y = kchs["dep"].astype(int).values
tr, ho = train_test_split(np.arange(len(kchs)), train_size=500000, test_size=100000, stratify=y, random_state=SEED)
Who = kchs.iloc[ho]["W"].to_numpy(float); yho = y[ho]
Wp, Ws = pre["W"].to_numpy(float), post["W"].to_numpy(float)
yp, ys = pre["dep"].astype(int).values, post["dep"].astype(int).values

rows=[]
for tag, use_fixed in [("기준(현행 income_q)", False), ("교정 income_q(항상 5단계)", True)]:
    K, P_, S_ = kchs.copy(), pre.copy(), post.copy()
    if use_fixed:
        K["income_q"]=fx["KCHS"].values; P_["income_q"]=fx["NHANES pre"].values; S_["income_q"]=fx["NHANES post"].values
    m=Pipeline([("pre",make_ct()),("clf",lgb.LGBMClassifier(**LGB))]).fit(K.iloc[tr][FE], y[tr])
    i=roc_auc_score(yho, m.predict_proba(K.iloc[ho][FE])[:,1], sample_weight=Who)
    p=roc_auc_score(yp,  m.predict_proba(P_[FE])[:,1], sample_weight=Wp)
    s=roc_auc_score(ys,  m.predict_proba(S_[FE])[:,1], sample_weight=Ws)
    def slope(yv,pp,w):
        pp=np.clip(pp,1e-6,1-1e-6); lp=np.log(pp/(1-pp)).reshape(-1,1)
        return float(LogisticRegression(penalty=None,solver="lbfgs").fit(lp,yv,sample_weight=w).coef_[0,0])
    sl_p=slope(yp, m.predict_proba(P_[FE])[:,1], Wp); sl_s=slope(ys, m.predict_proba(S_[FE])[:,1], Ws)
    print(f"   {tag:<26} 내부 {i:.4f}  pre {p:.4f}  post {s:.4f}   (보정기울기 pre {sl_p:.3f} / post {sl_s:.3f})", flush=True)
    rows.append(dict(spec=tag, internal=round(i,4), ext_pre=round(p,4), ext_post=round(s,4),
                     slope_pre=round(sl_p,3), slope_post=round(sl_s,3)))
df=pd.DataFrame(rows); df.to_csv(f"{RES}/income_fix_sensitivity.csv", index=False, encoding="utf-8-sig")
d0,d1=df.iloc[0],df.iloc[1]
print(f"\n   Δ(교정 − 기준): 내부 {d1.internal-d0.internal:+.4f}  pre {d1.ext_pre-d0.ext_pre:+.4f}  post {d1.ext_post-d0.ext_post:+.4f}", flush=True)
print("\n저장: results/income_fix_diagnostic.csv, income_fix_sensitivity.csv\n완료.", flush=True)
