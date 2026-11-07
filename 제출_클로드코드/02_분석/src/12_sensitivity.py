# -*- coding: utf-8 -*-
"""
12_sensitivity.py (프로토콜 §2.12) — 내부(가중 holdout) AUROC 민감도분석.
컷오프 ≥5, PA·음주 제외, 가중학습, 결측지표 추가. (기왕력제외는 KCHS 2017+에 진단변수 부재로 불가 → 기록)
결과: results/sensitivity.csv
"""
import os, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

PROC = r"C:/Users/rudtn/Desktop/claude code/지역사회건강/data/processed"; RES = r"C:/Users/rudtn/Desktop/claude code/지역사회건강/results"
SEED = 42
dev = pd.read_parquet(os.path.join(PROC, "kchs_dev.parquet"))
NUM = ["age", "sleep_hrs", "bmi", "income_q", "srh"]; BIN = ["partnered", "living_alone", "current_drinker", "active", "htn", "dm"]; CAT = ["sex", "edu", "smoking"]

def pipe(num, binf, catf):
    ct = ColumnTransformer([("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num),
                            ("bin", SimpleImputer(strategy="most_frequent"), binf),
                            ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")), ("o", OneHotEncoder(handle_unknown="ignore", drop="first"))]), catf)])
    return Pipeline([("pre", ct), ("clf", lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1))])

def run(y, num, binf, catf, weighted_train=False):
    idxtr, idxho = train_test_split(np.arange(len(dev)), test_size=0.2, stratify=y, random_state=SEED)
    p = pipe(num, binf, catf)
    if weighted_train:
        p.fit(dev.iloc[idxtr][num + binf + catf], y[idxtr], clf__sample_weight=dev.iloc[idxtr]["W"].to_numpy(float))
    else:
        p.fit(dev.iloc[idxtr][num + binf + catf], y[idxtr])
    pr = p.predict_proba(dev.iloc[idxho][num + binf + catf])[:, 1]
    return roc_auc_score(y[idxho], pr, sample_weight=dev.iloc[idxho]["W"].to_numpy(float))

y10 = (dev["phq_sum"] >= 10).astype(int).values
y5 = (dev["phq_sum"] >= 5).astype(int).values
# 결측지표 추가본
dev["srh_miss"] = dev["srh"].isna().astype(int); dev["sleep_miss"] = dev["sleep_hrs"].isna().astype(int); dev["bmi_miss"] = dev["bmi"].isna().astype(int)

R = []
R.append(("기준(≥10, 14변수, 비가중학습)", round(run(y10, NUM, BIN, CAT), 4), f"유병 {y10.mean():.1%}"))
R.append(("컷오프 ≥5", round(run(y5, NUM, BIN, CAT), 4), f"유병 {y5.mean():.1%}"))
R.append(("PA·음주 제외(12변수)", round(run(y10, NUM, [b for b in BIN if b not in ("current_drinker", "active")], CAT), 4), "비교불가 변수 제외"))
R.append(("가중 학습", round(run(y10, NUM, BIN, CAT, weighted_train=True), 4), "§2.8 대안"))
R.append(("결측지표 추가", round(run(y10, NUM + ["srh_miss", "sleep_miss", "bmi_miss"], BIN, CAT), 4), "§2.6 대안"))
res = pd.DataFrame(R, columns=["민감도조건", "내부AUROC(가중)", "비고"])
res.to_csv(os.path.join(RES, "sensitivity.csv"), index=False, encoding="utf-8-sig")
print(res.to_string(index=False))
print("\n[기록] 우울증 기왕력 제외: KCHS 2017+ 자료에 우울증 의사진단(dpa_*) 변수 부재 → §2.12 해당 민감도 불가(문서화).")
