# -*- coding: utf-8 -*-
"""pathG_02: 한계치(ceiling) 진단.

핵심 질문 - 17변수로 NHANES에서 가중 AUROC 0.80이 원리적으로 가능한가?
→ NHANES **라벨을 직접 써서** 자체 학습(교차검증)한 in-domain 상한을 측정.
   이는 배포모델이 아니라 진단이며 protocol_compliant=False 로 표기한다.
   전이모델이 이 상한 근처에 있다면 도메인적응으로는 더 못 올린다.
"""
import os, sys, time, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathG_common import *
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import lightgbm as lgb

T0 = time.time()
k, pre, post = load()
OUT = []


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def make_ct_dense():
    """이진변수도 최빈값 대체 (LogisticRegression/RF는 NaN 미지원)."""
    return ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")),
                          ("s", StandardScaler())]), NUM),
        ("bin", SimpleImputer(strategy="most_frequent"), BIN),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", OneHotEncoder(handle_unknown="ignore",
                                              drop="first"))]), CAT),
    ])


def cv_oof(df, model_fn, n_splits=5, seed=SEED, reps=3, dense=False):
    """설계 가중 AUROC (out-of-fold). reps회 반복 후 평균/표준편차."""
    X = df[FEATURES].reset_index(drop=True)
    yv = df["dep"].astype(int).values
    wv = df["W"].to_numpy(float)
    aucs = []
    for rp in range(reps):
        oof = np.zeros(len(df))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + rp)
        for a, b in skf.split(X, yv):
            pipe = Pipeline([("pre", make_ct_dense() if dense else make_ct()),
                             ("clf", model_fn())])
            pipe.fit(X.iloc[a], yv[a])
            oof[b] = pipe.predict_proba(X.iloc[b])[:, 1]
        aucs.append(wauc(yv, oof, wv))
    return float(np.mean(aucs)), float(np.std(aucs))


MODELS = {
    "logit_C1": lambda: LogisticRegression(max_iter=3000, C=1.0),
    "logit_C0.1": lambda: LogisticRegression(max_iter=3000, C=0.1),
    "lgb_small": lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                           num_leaves=7, min_child_samples=50,
                                           subsample=0.8, colsample_bytree=0.8,
                                           random_state=SEED, n_jobs=-1, verbose=-1),
    "lgb_base": lambda: lgb.LGBMClassifier(**LGB_PARAMS),
    "rf": lambda: RandomForestClassifier(n_estimators=400, min_samples_leaf=20,
                                         n_jobs=-1, random_state=SEED),
}

print("=== in-domain ORACLE ceiling (NHANES 라벨 사용, 진단 전용) ===", flush=True)
for dtag, df in [("pre", pre), ("post", post),
                 ("pooled", pd.concat([pre, post], ignore_index=True))]:
    for mtag, fn in MODELS.items():
        m, s = cv_oof(df, fn, dense=mtag.startswith(("logit","rf")))
        OUT.append(dict(name=f"CEIL_oracle_{dtag}_{mtag}", scope=dtag, model=mtag,
                        n=len(df), wauc=round(m, 5), sd=round(s, 5),
                        protocol_compliant=False,
                        note="NHANES 라벨로 자체 5-fold CV(3회 반복) → in-domain 상한 진단"))
        print(json.dumps(OUT[-1], ensure_ascii=False), flush=True)
        pd.DataFrame(OUT).to_csv(os.path.join(RES, "pathG_02_ceiling.csv"),
                                 index=False, encoding="utf-8-sig")

print("=== KCHS 동일 표본크기 대조 ===", flush=True)
rng = np.random.RandomState(SEED)
for n in [7888, 13098, 50000]:
    idx = rng.choice(len(k), size=n, replace=False)
    sub = k.iloc[idx].reset_index(drop=True)
    for mtag in ["logit_C1", "lgb_small", "lgb_base"]:
        m, s = cv_oof(sub, MODELS[mtag], reps=3 if n <= 13098 else 1, dense=mtag.startswith(("logit","rf")))
        OUT.append(dict(name=f"CEIL_kchs_n{n}_{mtag}", scope=f"kchs_n{n}", model=mtag,
                        n=n, wauc=round(m, 5), sd=round(s, 5), protocol_compliant=True,
                        note="KCHS를 NHANES와 같은 표본크기로 줄여 5-fold CV → 표본크기 효과 분리"))
        print(json.dumps(OUT[-1], ensure_ascii=False), flush=True)
pd.DataFrame(OUT).to_csv(os.path.join(RES, "pathG_02_ceiling.csv"),
                         index=False, encoding="utf-8-sig")

print("=== 단변량 가중 AUROC (도메인별 신호 비교) ===", flush=True)
uni = []
ksub = k.iloc[rng.choice(len(k), size=200000, replace=False)].reset_index(drop=True)
for c in FEATURES:
    row = dict(feature=c)
    for dtag, df in [("kchs", ksub), ("pre", pre), ("post", post)]:
        v = df[c]
        yv = df["dep"].astype(int).values
        wv = df["W"].to_numpy(float)
        if v.dtype == object:
            vv = pd.Categorical(v).codes.astype(float)
            vv[pd.isna(v).values] = np.nan
            # 범주형은 단조성이 없으므로 카테고리별 유병률로 점수화
            tmp = pd.DataFrame(dict(v=v, y=yv, w=wv))
            mp = tmp.dropna().groupby("v").apply(
                lambda g: np.average(g["y"], weights=g["w"]))
            sc = v.map(mp).astype(float).values
        else:
            sc = v.astype(float).values
        m = ~np.isnan(sc)
        try:
            a = wauc(yv[m], sc[m], wv[m])
        except Exception:
            a = np.nan
        row[dtag] = round(max(a, 1 - a), 4) if a == a else np.nan
    uni.append(row)
ud = pd.DataFrame(uni).sort_values("kchs", ascending=False)
print(ud.to_string(), flush=True)
ud.to_csv(os.path.join(RES, "pathG_02_univariate.csv"), index=False, encoding="utf-8-sig")
print("TOTAL %.1f min" % ((time.time() - T0) / 60))
