# -*- coding: utf-8 -*-
"""48_parsimonious_clean.py — 경량모델 선택을 학습셋 안에서만 수행하도록 재설계.

지적사항: 기존에는 (a) SHAP 순서를 hold-out 표본에서 구하고 (b) 문항수 k도 같은 hold-out에서 골랐다.
따라서 보고된 내부 성능은 선택 후(post-selection) 값이었다.

수정 설계:
  학습셋 500,000 → 적합 450,000 / 선택용 검증 50,000 으로 분할
  · SHAP 순서: 적합 450k에서 학습한 모델을 450k 표본으로 설명(외부·hold-out 미사용)
  · k 선택: 선택용 검증 50k의 가중 AUROC로만 결정
  · hold-out 100k과 NHANES는 평가에만 사용
  · k = 1..17 전 구간 산출(부분 구간만 보고하던 문제 해소)

출력: results/parsimonious_clean.csv
"""
import sys, warnings, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb, shap

D = r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model"; RES = f"{D}/results"; SEED = 42
kchs = pd.read_parquet(f"{D}/data/ext_kchs.parquet"); pre = pd.read_parquet(f"{D}/data/ext_pre.parquet"); post = pd.read_parquet(f"{D}/data/ext_post.parquet")
NUM = ["age","sleep_hrs","bmi","income_q","srh","oral_health"]
BIN = ["partnered","living_alone","current_drinker","active","htn","dm","employed","food_insecure"]
CAT = ["sex","edu","smoking"]; FE = NUM+BIN+CAT
y = kchs["dep"].astype(int).values
LGB = dict(num_leaves=31, min_child_samples=1000, learning_rate=0.01, n_estimators=900, reg_alpha=1.0,
           reg_lambda=20.0, subsample=0.6, colsample_bytree=0.6, random_state=SEED, n_jobs=-1, verbose=-1)
def make_ct(NU, BI, CA):
    t=[]
    if NU: t.append(("num", Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]), NU))
    if BI: t.append(("bin","passthrough",BI))
    if CA: t.append(("cat", Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore",drop="first"))]), CA))
    return ColumnTransformer(t)

tr, ho = train_test_split(np.arange(len(kchs)), train_size=500000, test_size=100000, stratify=y, random_state=SEED)
fit_idx, sel_idx = train_test_split(tr, train_size=450000, test_size=50000, stratify=y[tr], random_state=SEED)
print(f"[분할] 학습 500,000 → 적합 {len(fit_idx):,} / 선택검증 {len(sel_idx):,} · hold-out {len(ho):,}(평가 전용)", flush=True)

# ── SHAP 순서: 적합셋에서만 ──
full = Pipeline([("pre", make_ct(NUM,BIN,CAT)), ("clf", lgb.LGBMClassifier(**LGB))]).fit(kchs.iloc[fit_idx][FE], y[fit_idx])
samp = kchs.iloc[np.random.RandomState(SEED).choice(fit_idx, 20000, replace=False)]
Xs = full.named_steps["pre"].transform(samp[FE])
names = full.named_steps["pre"].get_feature_names_out()
sv = shap.TreeExplainer(full.named_steps["clf"]).shap_values(Xs)
if isinstance(sv, list): sv = sv[1]
imp = pd.Series(np.abs(sv).mean(0), index=[n.split("__")[-1] for n in names])
agg = {}
for v in FE:
    agg[v] = imp[[i for i in imp.index if i == v or i.startswith(v+"_")]].sum()
order = pd.Series(agg).sort_values(ascending=False)
order.to_csv(f"{RES}/shap_importance_train.csv", encoding="utf-8-sig")
print(f"[SHAP 순서, 적합셋 기준] {list(order.index)}", flush=True)

W_sel = kchs.iloc[sel_idx]["W"].to_numpy(float); y_sel = y[sel_idx]
W_ho  = kchs.iloc[ho]["W"].to_numpy(float);      y_ho  = y[ho]
Wp, Ws = pre["W"].to_numpy(float), post["W"].to_numpy(float)
yp, ys = pre["dep"].astype(int).values, post["dep"].astype(int).values

rows=[]
for k in range(1, 18):
    f = list(order.index[:k])
    NU=[c for c in NUM if c in f]; BI=[c for c in BIN if c in f]; CA=[c for c in CAT if c in f]; FEk=NU+BI+CA
    m = Pipeline([("pre", make_ct(NU,BI,CA)), ("clf", lgb.LGBMClassifier(**LGB))]).fit(kchs.iloc[fit_idx][FEk], y[fit_idx])
    sel = roc_auc_score(y_sel, m.predict_proba(kchs.iloc[sel_idx][FEk])[:,1], sample_weight=W_sel)
    hoa = roc_auc_score(y_ho,  m.predict_proba(kchs.iloc[ho][FEk])[:,1],     sample_weight=W_ho)
    pa  = roc_auc_score(yp,    m.predict_proba(pre[FEk])[:,1],               sample_weight=Wp)
    sa  = roc_auc_score(ys,    m.predict_proba(post[FEk])[:,1],              sample_weight=Ws)
    print(f"   k={k:>2}  선택검증 {sel:.4f} | hold-out {hoa:.4f}  pre {pa:.4f}  post {sa:.4f}   +{f[-1]}", flush=True)
    rows.append(dict(k=k, added=f[-1], vars=";".join(f), selection_AUROC=round(sel,4),
                     holdout_AUROC=round(hoa,4), ext_pre=round(pa,4), ext_post=round(sa,4)))
df = pd.DataFrame(rows); df.to_csv(f"{RES}/parsimonious_clean.csv", index=False, encoding="utf-8-sig")
full_sel = float(df.loc[df.k==17,"selection_AUROC"].iloc[0])
cand = df[(df.k<17) & (df.selection_AUROC >= full_sel-0.01)]
kstar = int(cand.k.min()) if len(cand) else 17
print(f"\n[선택] 전체모델 선택검증 {full_sel:.4f} · 0.01 이내 최소 문항수 k*={kstar}", flush=True)
print(df.to_string(index=False), flush=True)
print("\n저장: results/parsimonious_clean.csv, shap_importance_train.csv\n완료.", flush=True)
