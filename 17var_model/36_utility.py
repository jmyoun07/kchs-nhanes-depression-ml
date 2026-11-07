# -*- coding: utf-8 -*-
"""임상적 유용성 심화 — 고민감도 트리아지 운영지점 + 순편익. 17변수 & 7문항 배포모델.
가중(설계) 기반. 재보정(Platt)은 배포 시나리오로 순편익 계산에만 사용(외부라벨=지역재보정, 공개됨)."""
import sys, warnings, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
SEED=42; D="C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model"
NUM=["age","sleep_hrs","bmi","income_q","srh","oral_health"]; BIN=["partnered","living_alone","current_drinker","active","htn","dm","employed","food_insecure"]; CAT=["sex","edu","smoking"]
FE17=NUM+BIN+CAT
# 배포모델 = 5문항 (사전지정 0.02 여유, 48_parsimonious_clean.py) / 7문항은 사후 변형
NUM5=["srh","age","sleep_hrs","oral_health"]; BIN5=[]; CAT5=["sex"]; FE5=NUM5+BIN5+CAT5
NUM8=["srh","age","sleep_hrs","oral_health"]; BIN8=["employed","partnered"]; CAT8=["sex"]; FE8=NUM8+BIN8+CAT8
LGB=dict(num_leaves=31,min_child_samples=1000,learning_rate=0.01,n_estimators=900,reg_alpha=1.0,reg_lambda=20.0,subsample=0.6,colsample_bytree=0.6,random_state=SEED,n_jobs=-1,verbose=-1)
kchs=pd.read_parquet(f"{D}/data/ext_kchs.parquet",columns=FE17+["dep","W"])
pre =pd.read_parquet(f"{D}/data/ext_pre.parquet", columns=FE17+["dep","W"])
post=pd.read_parquet(f"{D}/data/ext_post.parquet",columns=FE17+["dep","W"])
y=kchs["dep"].astype(int).values
tr,ho=train_test_split(np.arange(len(kchs)),train_size=500000,test_size=100000,stratify=y,random_state=SEED)
def ctf(NU,BI,CA): return ColumnTransformer([("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),NU),("bin","passthrough",BI),("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore",drop="first"))]),CA)])
def train(NU,BI,CA):
    fe=NU+BI+CA; c=ctf(NU,BI,CA); m=lgb.LGBMClassifier(**LGB).fit(c.fit_transform(kchs.iloc[tr][fe]),y[tr])
    return c,m,fe

def wq(yv,p,w):  # 가중 지표 유틸
    return yv.astype(float),p,w.astype(float)
def sens_at(yv,p,w,target):
    # 양성자 점수의 가중 분위로 목표 민감도 임계 찾기
    pos=yv==1; order=np.argsort(-p[pos]); wp=w[pos][order]; cum=np.cumsum(wp)/wp.sum()
    k=int(np.searchsorted(cum,target)); k=min(k,len(order)-1)
    return p[pos][order][k]
def metrics(yv,p,w,t):
    P=w[(yv==1)&(p>=t)].sum(); FN=w[(yv==1)&(p<t)].sum(); FP=w[(yv==0)&(p>=t)].sum(); TN=w[(yv==0)&(p<t)].sum()
    sens=P/(P+FN) if P+FN>0 else 0; spec=TN/(TN+FP) if TN+FP>0 else 0
    ppv=P/(P+FP) if P+FP>0 else 0; npv=TN/(TN+FN) if TN+FN>0 else 0
    flag=(P+FP)/w.sum(); nns=1/ppv if ppv>0 else np.nan
    return sens,spec,ppv,npv,flag,nns
def recalib(p_ho_dummy, p_ext, yv, w):
    """절편/기울기 재보정 — §2.10대로 5-fold 교차적합(cross-fit). 같은 표본에서 적합·적용하면
    순편익이 낙관적으로 부풀기 때문에 폴드 밖 예측만 사용한다."""
    from sklearn.model_selection import StratifiedKFold
    lp=np.log(np.clip(p_ext,1e-6,1-1e-6)/(1-np.clip(p_ext,1e-6,1-1e-6))).reshape(-1,1)
    oof=np.zeros(len(p_ext))
    for tri,tei in StratifiedKFold(5,shuffle=True,random_state=SEED).split(lp,yv):
        lr=LogisticRegression(penalty=None).fit(lp[tri],yv[tri],sample_weight=w[tri])
        oof[tei]=lr.predict_proba(lp[tei])[:,1]
    return oof

rows=[]; nb_rows=[]
for tag,(NU,BI,CA) in [("17-predictor",(NUM,BIN,CAT)),("5-item",(NUM5,BIN5,CAT5)),("7-item",(NUM8,BIN8,CAT8))]:
    c,m,fe=train(NU,BI,CA)
    sets={"Internal":(kchs.iloc[ho],), "NHANES pre":(pre,), "NHANES post":(post,)}
    for nm,(d,) in sets.items():
        p=m.predict_proba(c.transform(d[fe]))[:,1]; yv=d["dep"].astype(int).values; w=d["W"].astype(float).values
        for target in [0.85,0.90]:
            t=sens_at(yv,p,w,target); s,sp,ppv,npv,flag,nns=metrics(yv,p,w,t)
            rows.append((tag,nm,f"sens≈{target:.2f}",round(s,3),round(sp,3),round(ppv,3),round(npv,3),round(flag,3),round(nns,1)))
        # 순편익(재보정 후) — 외부만 재보정, 내부는 자체
        pc=recalib(None,p,yv,w)
        prev=np.average(yv,weights=w)
        for pt in [0.10,0.15]:
            TPr=w[(yv==1)&(pc>=pt)].sum()/w.sum(); FPr=w[(yv==0)&(pc>=pt)].sum()/w.sum()
            nb=TPr-FPr*(pt/(1-pt)); nb_all=prev-(1-prev)*(pt/(1-pt)); flag=(pc>=pt).astype(float)
            flagr=w[pc>=pt].sum()/w.sum()
            # 기본전략 2종: screen-all(nb_all, 음수 가능) / screen-none(0). 임상적 유용성 = 두 기본전략을 모두 상회.
            nb_rows.append((tag,nm,f"pt={pt:.2f}",round(nb,4),round(nb_all,4),round(nb-nb_all,4),round(nb-max(nb_all,0.0),4),round(flagr,3)))

op=pd.DataFrame(rows,columns=["model","set","operating point","sens","spec","PPV","NPV","flag rate","NNS"])
nb=pd.DataFrame(nb_rows,columns=["model","set","threshold","net benefit","screen-all NB","NB vs screen-all","NB vs best default","flag rate"])
op.to_csv(f"{D}/results/table_s4_operating.csv",index=False,encoding="utf-8-sig")
nb.to_csv(f"{D}/results/table_s4_netbenefit.csv",index=False,encoding="utf-8-sig")
print("=== 고민감도 트리아지 운영지점 (가중) ===")
print(op.to_string(index=False))
print("\n=== 순편익 (재보정 후, 가중) ===")
print(nb.to_string(index=False))
print("\n저장: results/table_s4_operating.csv, table_s4_netbenefit.csv")
