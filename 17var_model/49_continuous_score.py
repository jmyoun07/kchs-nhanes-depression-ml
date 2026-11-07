# -*- coding: utf-8 -*-
"""49_continuous_score.py — 프로토콜이 사전지정한 '연속점수' 민감도 분석(미이행분 이행).

프로토콜 원문: "Sensitivity analyses use a ≥5 cut-off and the continuous score."
컷오프 ≥5는 이미 수행되었으나 연속점수 분석은 누락되어 있었다.

(a) 배포된 이진 분류모델이 연속 PHQ-9 총점을 얼마나 잘 순위화하는가 (가중 Spearman)
(b) 동일 예측변수로 연속 총점을 직접 학습한 회귀모델의 내부·외부 성능 (가중 Spearman, R²)

출력: results/continuous_score.csv
"""
import sys, warnings, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import lightgbm as lgb

D = r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model"; RES = f"{D}/results"; SEED = 42
kchs = pd.read_parquet(f"{D}/data/ext_kchs.parquet"); pre = pd.read_parquet(f"{D}/data/ext_pre.parquet"); post = pd.read_parquet(f"{D}/data/ext_post.parquet")
NUM = ["age","sleep_hrs","bmi","income_q","srh","oral_health"]
BIN = ["partnered","living_alone","current_drinker","active","htn","dm","employed","food_insecure"]
CAT = ["sex","edu","smoking"]; FE = NUM+BIN+CAT
y = kchs["dep"].astype(int).values
P = dict(num_leaves=31, min_child_samples=1000, learning_rate=0.01, n_estimators=900, reg_alpha=1.0,
         reg_lambda=20.0, subsample=0.6, colsample_bytree=0.6, random_state=SEED, n_jobs=-1, verbose=-1)
def make_ct():
    return ColumnTransformer([("num", Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]), NUM),
                              ("bin","passthrough",BIN),
                              ("cat", Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore",drop="first"))]), CAT)])

def wspearman(a, b, w):
    ra = pd.Series(a).rank().to_numpy(); rb = pd.Series(b).rank().to_numpy()
    ma = np.average(ra, weights=w); mb = np.average(rb, weights=w)
    ca = ra-ma; cb = rb-mb
    return float(np.sum(w*ca*cb) / np.sqrt(np.sum(w*ca*ca)*np.sum(w*cb*cb)))
def wr2(ytrue, ypred, w):
    m = np.average(ytrue, weights=w)
    return float(1 - np.sum(w*(ytrue-ypred)**2)/np.sum(w*(ytrue-m)**2))

tr, ho = train_test_split(np.arange(len(kchs)), train_size=500000, test_size=100000, stratify=y, random_state=SEED)
clf = Pipeline([("pre", make_ct()), ("clf", lgb.LGBMClassifier(**P))]).fit(kchs.iloc[tr][FE], y[tr])
sc = pd.to_numeric(kchs["phq_sum"]).to_numpy(float)
reg = Pipeline([("pre", make_ct()), ("clf", lgb.LGBMRegressor(**P))]).fit(kchs.iloc[tr][FE], sc[tr])

rows = []
for nm, d, s_col in [("Internal(hold-out)", kchs.iloc[ho], sc[ho]),
                     ("NHANES pre", pre, pd.to_numeric(pre["phq_sum"]).to_numpy(float)),
                     ("NHANES post", post, pd.to_numeric(post["phq_sum"]).to_numpy(float))]:
    w = d["W"].to_numpy(float)
    p_clf = clf.predict_proba(d[FE])[:,1]
    p_reg = reg.predict(d[FE])
    r_clf = wspearman(p_clf, s_col, w)
    r_reg = wspearman(p_reg, s_col, w)
    r2 = wr2(s_col, p_reg, w)
    print(f"  {nm:<20} 분류모델 순위상관 ρ={r_clf:.3f} | 연속회귀 ρ={r_reg:.3f}, R²={r2:.3f}  (평균 PHQ-9 {np.average(s_col,weights=w):.2f})", flush=True)
    rows.append(dict(set=nm, spearman_classifier=round(r_clf,3), spearman_regressor=round(r_reg,3),
                     R2_regressor=round(r2,3), mean_phq9=round(float(np.average(s_col,weights=w)),2)))
pd.DataFrame(rows).to_csv(f"{RES}/continuous_score.csv", index=False, encoding="utf-8-sig")
print("\n저장: results/continuous_score.csv\n완료.", flush=True)
