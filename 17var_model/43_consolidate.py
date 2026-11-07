# -*- coding: utf-8 -*-
"""43_consolidate.py — 감사 반영 후 정본 결과표 재작성.

(1) 경량모델 k=6,7 보강 → 배포 문항수 선택 근거(내부 성능 기준)를 명시적으로 검증
(2) final_metrics.csv     ← 정식 CITL(offset) + 층별 군집 부트스트랩 CI
(3) final_parsimonious.csv← SHAP 순서 기반, 외부 post 포함, k=3..17
(4) final_sensitivity.csv ← 컷오프≥5 외부 포함, 연령≥30 정식 버전 포함
"""
import sys, warnings, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

D = r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model"; RES = f"{D}/results"; SEED = 42
kchs = pd.read_parquet(f"{D}/data/ext_kchs.parquet"); pre = pd.read_parquet(f"{D}/data/ext_pre.parquet"); post = pd.read_parquet(f"{D}/data/ext_post.parquet")
NUM = ["age","sleep_hrs","bmi","income_q","srh","oral_health"]
BIN = ["partnered","living_alone","current_drinker","active","htn","dm","employed","food_insecure"]
CAT = ["sex","edu","smoking"]; FE = NUM+BIN+CAT
y = kchs["dep"].astype(int).values
LGB = dict(num_leaves=31, min_child_samples=1000, learning_rate=0.01, n_estimators=900, reg_alpha=1.0,
           reg_lambda=20.0, subsample=0.6, colsample_bytree=0.6, random_state=SEED, n_jobs=-1, verbose=-1)
def make_ct(NU, BI, CA):
    t = []
    if NU: t.append(("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), NU))
    if BI: t.append(("bin", "passthrough", BI))
    if CA: t.append(("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")), ("o", OneHotEncoder(handle_unknown="ignore", drop="first"))]), CA))
    return ColumnTransformer(t)
tr, ho = train_test_split(np.arange(len(kchs)), train_size=500000, test_size=100000, stratify=y, random_state=SEED)
Who, yho = kchs.iloc[ho]["W"].to_numpy(float), y[ho]
Wp, Ws = pre["W"].to_numpy(float), post["W"].to_numpy(float)
yp, ys = pre["dep"].astype(int).values, post["dep"].astype(int).values

order = [v for v in pd.read_csv(f"{RES}/shap_importance.csv", index_col=0).iloc[:,0].sort_values(ascending=False).index if v in FE]
prev = pd.read_csv(f"{RES}/audit_recompute_parsimonious.csv")
rows = {int(r.k): dict(k=int(r.k), vars=r.vars, internal=r.internal, ext_pre=r.ext_pre, ext_post=r.ext_post) for r in prev.itertuples()}
print("[경량모델] k=6,7 보강", flush=True)
for k in [6, 7]:
    f = order[:k]; NU=[c for c in NUM if c in f]; BI=[c for c in BIN if c in f]; CA=[c for c in CAT if c in f]; FEk=NU+BI+CA
    mk = Pipeline([("pre", make_ct(NU,BI,CA)), ("clf", lgb.LGBMClassifier(**LGB))]).fit(kchs.iloc[tr][FEk], y[tr])
    ik = roc_auc_score(yho, mk.predict_proba(kchs.iloc[ho][FEk])[:,1], sample_weight=Who)
    pk = roc_auc_score(yp, mk.predict_proba(pre[FEk])[:,1], sample_weight=Wp)
    sk = roc_auc_score(ys, mk.predict_proba(post[FEk])[:,1], sample_weight=Ws)
    print(f"   {k}문항 내부 {ik:.4f}  pre {pk:.4f}  post {sk:.4f}   {f}", flush=True)
    rows[k] = dict(k=k, vars=";".join(f), internal=round(ik,4), ext_pre=round(pk,4), ext_post=round(sk,4))
par = pd.DataFrame([rows[k] for k in sorted(rows)])
par.to_csv(f"{RES}/final_parsimonious.csv", index=False, encoding="utf-8-sig")
full = float(par.loc[par.k==17, "internal"].iloc[0])
cand = par[(par.k < 17) & (par.internal >= full - 0.01)]
print(f"\n[배포 문항수 선택] 전체모델 내부 {full:.4f} 대비 0.01 이내인 최소 문항수 = {int(cand.k.min()) if len(cand) else '없음'}", flush=True)
print(par.to_string(index=False), flush=True)

m = pd.read_csv(f"{RES}/audit_recompute_metrics.csv")
out = []
for r in m.itertuples():
    out.append(dict(set=r.set, AUROC=r.AUROC, CI_lo=r.CI_lo_stratified, CI_hi=r.CI_hi_stratified,
                    AUPRC=r.AUPRC, AUPRC_lo=r.AUPRC_lo, AUPRC_hi=r.AUPRC_hi, Brier=r.Brier,
                    cal_slope=r.cal_slope, CITL=r.CITL_offset))
    out.append(dict(set=r.set+" (재보정)", AUROC=r.recal_AUROC, CI_lo=None, CI_hi=None,
                    AUPRC=r.recal_AUPRC, AUPRC_lo=None, AUPRC_hi=None, Brier=r.recal_Brier,
                    cal_slope=r.recal_slope, CITL=r.recal_CITL_offset))
fm = pd.DataFrame(out)
fm = fm[fm["set"] != "Internal(hold-out) (재보정)"]
fm.to_csv(f"{RES}/final_metrics.csv", index=False, encoding="utf-8-sig")
print("\n[final_metrics.csv]"); print(fm.to_string(index=False), flush=True)

sens_old = pd.read_csv(f"{RES}/final_sensitivity.csv")
sens_new = pd.read_csv(f"{RES}/audit_recompute_sensitivity.csv")
keep = sens_old[~sens_old["cond"].astype(str).str.startswith("컷오프")]
sens = pd.concat([keep, sens_new], ignore_index=True)
sens.to_csv(f"{RES}/final_sensitivity.csv", index=False, encoding="utf-8-sig")
print("\n[final_sensitivity.csv]"); print(sens.to_string(index=False), flush=True)
print("\n완료.", flush=True)
