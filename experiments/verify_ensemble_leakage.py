# -*- coding: utf-8 -*-
"""앙상블 leakage 검증 — 앙상블 구성/가중치를 KCHS 내부 정보(연도 CV)로만 정했을 때도
외부 성능이 유지되는가? NHANES를 전혀 보지 않고 사전확정한 앙상블을 최종 1회 평가.

검증 논리:
 (A) 두 하위모델(상호작용 / 연도-안정부분집합)의 '선택 근거'가 KCHS 내부인지 재확인
 (B) 앙상블 가중치를 KCHS 내부(leave-one-year-out) AUROC로만 최적화 → NHANES 미사용
 (C) 그 사전확정 앙상블을 NHANES에 최종 1회 적용 → 외부 성능이 유지되는지
 (D) 대조: 만약 NHANES post로 가중치를 골랐다면(=leakage) 얼마나 부풀려지는지 (상한진단)
"""
import os, sys, warnings, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import lightgbm as lgb
PROC=r"C:/Users/rudtn/Desktop/claude code/지역사회건강/data/processed"; SEED=42
kchs=pd.read_parquet(f"{PROC}/ext_kchs.parquet"); pre=pd.read_parquet(f"{PROC}/ext_pre.parquet"); post=pd.read_parquet(f"{PROC}/ext_post.parquet")
y=kchs["dep"].astype(int).values
NUM=["age","sleep_hrs","bmi","income_q","srh","oral_health"]
BIN=["partnered","living_alone","current_drinker","active","htn","dm","employed","food_insecure"]
CAT=["sex","edu","smoking"]
def addint(df):
    d=df.copy(); n=lambda c:pd.to_numeric(d[c],errors="coerce"); az=(n("age")-50)/15
    d["age_x_srh"]=az*(n("srh")-3); d["srh_x_oral"]=(n("srh")-3)*(n("oral_health")-3)
    d["age_x_employed"]=az*n("employed").fillna(0); d["income_x_employed"]=n("income_q").fillna(2)*n("employed").fillna(0)
    return d
INT=["age_x_srh","srh_x_oral","age_x_employed","income_x_employed"]
kc,pc,sc=addint(kchs),addint(pre),addint(post)
UNSTABLE=["smoking","bmi","current_drinker"]   # B에서 KCHS 연도안정성 하위3 (NHANES 미사용)
def ct(NUMx,BINx,CATx):
    t=[]
    if NUMx:t.append(("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),NUMx))
    if BINx:t.append(("bin","passthrough",BINx))
    if CATx:t.append(("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore",drop="first"))]),CATx))
    return ColumnTransformer(t)
def lgbm(sd=SEED): return lgb.LGBMClassifier(n_estimators=600,learning_rate=0.03,num_leaves=31,subsample=0.8,colsample_bytree=0.8,random_state=sd,n_jobs=-1,verbose=-1)
def wauc(d,p): return roc_auc_score(d["dep"].astype(int).values,p,sample_weight=d["W"].to_numpy(float))

# 두 하위모델 정의 (근거는 모두 KCHS 내부)
def fit_M1(idx):  # 상호작용 모델 (도메인지식 파생 — KCHS 미사용 근거 불필요, 사전지정 특징)
    FE=NUM+INT+BIN+CAT; return Pipeline([("pre",ct(NUM+INT,BIN,CAT)),("clf",lgbm())]).fit(kc.iloc[idx][FE],y[idx]),FE
def fit_M2(idx):  # 연도-안정 부분집합 (B: KCHS 연도 CV로 선정)
    NUMx=[c for c in NUM if c not in UNSTABLE]; BINx=[c for c in BIN if c not in UNSTABLE]; CATx=[c for c in CAT if c not in UNSTABLE]
    FE=NUMx+BINx+CATx; return Pipeline([("pre",ct(NUMx,BINx,CATx)),("clf",lgbm())]).fit(kc.iloc[idx][FE],y[idx]),FE

tr,ho=train_test_split(np.arange(len(kc)),train_size=500000,test_size=100000,stratify=y,random_state=SEED)
Who,yho=kc.iloc[ho]["W"].to_numpy(float),y[ho]

print("="*72)
print("[B] 앙상블 가중치를 KCHS 내부(연도 CV)로만 최적화 — NHANES 미사용")
# 연도별 out-of-year 예측으로 두 모델의 OOF 확률 생성 → 가중치 w를 KCHS 내부 AUROC로 최적화
years=sorted(kc["year"].unique())
tr_set=set(tr.tolist())
oof1=np.full(len(kc),np.nan); oof2=np.full(len(kc),np.nan)
for yr in years:
    trainy=np.array([i for i in tr if kc["year"].iat[i]!=yr])
    testy =np.array([i for i in range(len(kc)) if kc["year"].iat[i]==yr])
    if len(trainy)<50000: continue
    # 속도 위해 학습 20만 서브샘플
    rng=np.random.RandomState(SEED); tsub=rng.choice(trainy,min(200000,len(trainy)),replace=False)
    m1,FE1=fit_M1(tsub); m2,FE2=fit_M2(tsub)
    oof1[testy]=m1.predict_proba(kc.iloc[testy][FE1])[:,1]; oof2[testy]=m2.predict_proba(kc.iloc[testy][FE2])[:,1]
msk=~np.isnan(oof1)
Wall=kc["W"].to_numpy(float)
def internal_auc_w(w):
    p=w*rankdata(oof1[msk])/msk.sum()+(1-w)*rankdata(oof2[msk])/msk.sum()
    return roc_auc_score(y[msk],p,sample_weight=Wall[msk])
ws=np.linspace(0,1,21); aucs=[internal_auc_w(w) for w in ws]
w_star=ws[int(np.argmax(aucs))]
print(f"   KCHS 내부(연도OOF) 가중치 스윕 → 최적 w*={w_star:.2f} (내부AUROC {max(aucs):.4f})")
print(f"   (w=상호작용모델 비중; w*가 0.5 근처면 두 모델 균형이 내부적으로도 최적)")

print("\n[C] 사전확정(w*) 앙상블을 NHANES에 최종 1회 적용 — 이게 정당한 성능")
m1,FE1=fit_M1(tr); m2,FE2=fit_M2(tr)
# 파생변수 포함 pc/sc로 예측 (pre/post 원본엔 상호작용 컬럼 없음)
p1p=m1.predict_proba(pc[FE1])[:,1]; p2p=m2.predict_proba(pc[FE2])[:,1]
p1s=m1.predict_proba(sc[FE1])[:,1]; p2s=m2.predict_proba(sc[FE2])[:,1]
def blend(a,b,w): return w*rankdata(a)/len(a)+(1-w)*rankdata(b)/len(b)
pe=wauc(pre,blend(p1p,p2p,w_star)); se=wauc(post,blend(p1s,p2s,w_star))
print(f"   M1(상호작용) 단독:     pre {wauc(pre,p1p):.4f}  post {wauc(post,p1s):.4f}")
print(f"   M2(안정부분집합) 단독:  pre {wauc(pre,p2p):.4f}  post {wauc(post,p2s):.4f}")
print(f"   ★ w*={w_star:.2f} 앙상블(KCHS내부로 확정):  pre {pe:.4f}  post {se:.4f}")
pe5=wauc(pre,blend(p1p,p2p,0.5)); se5=wauc(post,blend(p1s,p2s,0.5))
print(f"   (참고) w=0.50 균형 앙상블:              pre {pe5:.4f}  post {se5:.4f}")

print("\n[D] 대조: 만약 NHANES post로 가중치를 골랐다면 (=leakage, 상한진단)")
bp=0;bw=0
for w in ws:
    a=wauc(post,blend(p1s,p2s,w))
    if a>bp: bp=a;bw=w
print(f"   NHANES post 보고 고른 최적 w={bw:.2f} → post {bp:.4f}  (이건 leakage, 사용 금지)")
print(f"   → 내부확정(w*={w_star:.2f}) post {se:.4f}  vs  leakage최적 post {bp:.4f}  차이={bp-se:+.4f}")

print("\n"+"="*72)
print("판정:")
print(f"  · 앙상블 가중치를 KCHS 내부만으로 정했을 때 w*={w_star:.2f}, 외부 pre {pe:.4f}/post {se:.4f}")
print(f"  · M2(부분집합) 선정근거 = KCHS 연도 안정성 (NHANES 미사용) → §2.10 준수")
print(f"  · 내부확정 앙상블이 개별모델보다 균형 우수하면 leakage 없이 정당한 개선")
