# -*- coding: utf-8 -*-
"""51_sap_compliance.py — 사전 통계분석계획서(SAP) 규칙 대조 산출.

SAP §9.5(주모델 선택): 1순위 CV-AUROC 최대 → **ΔAUROC<0.005면 동률**로 보고 2순위 보정기울기가 1에 근접한 모델.
  실측: LightGBM 0.8396 vs RF 0.8350 → Δ=0.0046 < 0.005 ⇒ 동률 조항 발동. 두 모델의 CV 보정기울기를 산출해 규칙대로 판정한다.
SAP §14(민감도 사전목록): '음주 대안 정의' 미이행 → 현재 음주(current_drinker) 대신 연간 음주빈도/폭음을 쓴 대안 정의를 산출한다.

출력: results/sap_tiebreak.csv, results/sap_alcohol_alt.csv
"""
import sys, warnings, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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
def make_ct(NU, BI, CA, native=True):
    return ColumnTransformer([("num", Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]), NU),
                              ("bin","passthrough" if native else SimpleImputer(strategy="most_frequent",add_indicator=True), BI),
                              ("cat", Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore",drop="first"))]), CA)])
def cal_slope(yv,p):
    p=np.clip(p,1e-6,1-1e-6); lp=np.log(p/(1-p)).reshape(-1,1)
    return float(LogisticRegression(penalty=None,solver="lbfgs").fit(lp,yv).coef_[0,0])

tr, ho = train_test_split(np.arange(len(kchs)), train_size=500000, test_size=100000, stratify=y, random_state=SEED)

# ══ SAP §9.5 동률 판정 ══
print("="*80); print("[SAP §9.5] 동률(ΔAUROC<0.005) 판정 — 알고리즘 비교와 동일 프로토콜(5-fold, 20만)"); print("="*80, flush=True)
sub = np.random.RandomState(SEED).permutation(tr)[:200000]
Xs, ys_ = kchs.iloc[sub], y[sub]
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
BEST = {
 "LightGBM": Pipeline([("pre", make_ct(NUM,BIN,CAT)), ("clf", lgb.LGBMClassifier(**LGB))]),
 "Random forest": Pipeline([("pre", make_ct(NUM,BIN,CAT)), ("clf", RandomForestClassifier(n_estimators=300, max_depth=14, min_samples_leaf=100, random_state=SEED, n_jobs=-1))]),
 "Elastic net": Pipeline([("pre", make_ct(NUM,BIN,CAT,native=False)), ("clf", LogisticRegression(penalty="elasticnet", solver="saga", C=1.0, l1_ratio=0.5, max_iter=300, n_jobs=-1))]),
}
rows=[]
for nm, mdl in BEST.items():
    oof = cross_val_predict(mdl, Xs[FE], ys_, cv=skf, method="predict_proba", n_jobs=-1)[:,1]
    a = roc_auc_score(ys_, oof); sl = cal_slope(ys_, oof)
    print(f"   {nm:<15} CV-AUROC {a:.4f}   보정기울기 {sl:.4f}   |slope-1| {abs(sl-1):.4f}", flush=True)
    rows.append(dict(algorithm=nm, cv_auroc=round(a,4), cal_slope=round(sl,4), abs_slope_minus_1=round(abs(sl-1),4)))
df = pd.DataFrame(rows).sort_values("cv_auroc", ascending=False).reset_index(drop=True)
top = df.iloc[0]; tied = df[df.cv_auroc >= top.cv_auroc - 0.005]
print(f"\n   1순위 {top.algorithm} (AUROC {top.cv_auroc:.4f}) · 동률권(Δ<0.005) {list(tied.algorithm)}")
win = tied.sort_values("abs_slope_minus_1").iloc[0]
print(f"   → SAP §9.5 규칙 적용 결과: 보정기울기 기준 승자 = **{win.algorithm}** (|slope-1|={win.abs_slope_minus_1:.4f})", flush=True)
df["in_tie_band"] = df.cv_auroc >= top.cv_auroc - 0.005
df["selected_by_SAP_rule"] = df.algorithm == win.algorithm
df.to_csv(f"{RES}/sap_tiebreak.csv", index=False, encoding="utf-8-sig")

# ══ SAP §14 음주 대안 정의 ══
print("\n" + "="*80); print("[SAP §14] 음주 대안 정의 민감도 (사전목록 중 미이행분)"); print("="*80, flush=True)
Who, yho = kchs.iloc[ho]["W"].to_numpy(float), y[ho]
Wp, Ws = pre["W"].to_numpy(float), post["W"].to_numpy(float)
yp, ys2 = pre["dep"].astype(int).values, post["dep"].astype(int).values
def run(NU, BI, CA, label):
    F = NU+BI+CA
    m = Pipeline([("pre", make_ct(NU,BI,CA)), ("clf", lgb.LGBMClassifier(**LGB))]).fit(kchs.iloc[tr][F], y[tr])
    i = roc_auc_score(yho, m.predict_proba(kchs.iloc[ho][F])[:,1], sample_weight=Who)
    p = roc_auc_score(yp,  m.predict_proba(pre[F])[:,1],  sample_weight=Wp)
    s = roc_auc_score(ys2, m.predict_proba(post[F])[:,1], sample_weight=Ws)
    print(f"   {label:<44} 내부 {i:.4f}  pre {p:.4f}  post {s:.4f}", flush=True)
    return dict(cond=label, internal=round(i,4), ext_pre=round(p,4), ext_post=round(s,4))
arows = [run(NUM, BIN, CAT, "기준(현재 음주 여부)")]
arows.append(run(NUM+["alcohol_freq"], [b for b in BIN if b!="current_drinker"], CAT, "대안 A: 연간 음주빈도(연속)로 대체"))
arows.append(run(NUM, [b for b in BIN if b!="current_drinker"]+["binge"], CAT, "대안 B: 폭음 여부로 대체"))
arows.append(run(NUM+["alcohol_freq"], BIN+["binge"], CAT, "대안 C: 현재음주+빈도+폭음 모두 포함"))
pd.DataFrame(arows).to_csv(f"{RES}/sap_alcohol_alt.csv", index=False, encoding="utf-8-sig")
print("\n저장: results/sap_tiebreak.csv, sap_alcohol_alt.csv\n완료.", flush=True)
