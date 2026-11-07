# -*- coding: utf-8 -*-
"""pathG_02b: 02의 나머지(동일 표본크기 KCHS 대조 + 단변량 신호 비교)."""
import os, sys, time, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathG_common import *
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import lightgbm as lgb

T0 = time.time()
k, pre, post = load()


def make_ct_dense():
    return ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")),
                          ("s", StandardScaler())]), NUM),
        ("bin", SimpleImputer(strategy="most_frequent"), BIN),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", OneHotEncoder(handle_unknown="ignore",
                                              drop="first"))]), CAT),
    ])


def cv_oof(df, model_fn, n_splits=5, seed=SEED, reps=1, dense=False):
    X = df[FEATURES].reset_index(drop=True)
    yv = df["dep"].astype(int).values
    wv = df["W"].to_numpy(float)
    aucs = []
    for rp in range(reps):
        oof = np.zeros(len(df))
        for a, b in StratifiedKFold(n_splits, shuffle=True,
                                    random_state=seed + rp).split(X, yv):
            pipe = Pipeline([("pre", make_ct_dense() if dense else make_ct()),
                             ("clf", model_fn())])
            pipe.fit(X.iloc[a], yv[a])
            oof[b] = pipe.predict_proba(X.iloc[b])[:, 1]
        aucs.append(wauc(yv, oof, wv))
    return float(np.mean(aucs)), float(np.std(aucs))


MODELS = {
    "logit_C1": (lambda: LogisticRegression(max_iter=3000, C=1.0), True),
    "lgb_small": (lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                             num_leaves=7, min_child_samples=50,
                                             subsample=0.8, colsample_bytree=0.8,
                                             random_state=SEED, n_jobs=4, verbose=-1), False),
    "lgb_base": (lambda: lgb.LGBMClassifier(**{**LGB_PARAMS, "n_jobs": 4}), False),
}
OUT = []
rng = np.random.RandomState(SEED)
for n in [7888, 13098, 50000, 200000]:
    idx = rng.choice(len(k), size=n, replace=False)
    sub = k.iloc[idx].reset_index(drop=True)
    for mtag, (fn, dense) in MODELS.items():
        m, s = cv_oof(sub, fn, reps=2 if n <= 13098 else 1, dense=dense)
        OUT.append(dict(name=f"CEIL_kchs_n{n}_{mtag}", scope=f"kchs_n{n}", model=mtag,
                        n=n, wauc=round(m, 5), sd=round(s, 5), protocol_compliant=True,
                        note="KCHS를 NHANES와 같은 표본크기로 줄여 5-fold CV → 표본크기 효과 분리"))
        print(json.dumps(OUT[-1], ensure_ascii=False), flush=True)
        pd.DataFrame(OUT).to_csv(os.path.join(RES, "pathG_02b_kchs_samesize.csv"),
                                 index=False, encoding="utf-8-sig")

print("=== 단변량 가중 AUROC ===", flush=True)
uni = []
ksub = k.iloc[rng.choice(len(k), size=200000, replace=False)].reset_index(drop=True)
for c in FEATURES:
    row = dict(feature=c)
    for dtag, df in [("kchs", ksub), ("pre", pre), ("post", post)]:
        v = df[c]
        yv = df["dep"].astype(int).values
        wv = df["W"].to_numpy(float)
        if v.dtype == object or str(v.dtype).startswith(("string","str")) or c in CAT:
            tmp = pd.DataFrame(dict(v=v.values, y=yv, w=wv)).dropna()
            mp = tmp.groupby("v").apply(lambda g: np.average(g["y"], weights=g["w"]))
            sc = v.map(mp).astype(float).values
        else:
            sc = v.astype(float).values
        m = ~np.isnan(sc)
        try:
            a = wauc(yv[m], sc[m], wv[m])
            row[dtag] = round(max(a, 1 - a), 4)
        except Exception:
            row[dtag] = np.nan
    uni.append(row)
ud = pd.DataFrame(uni).sort_values("kchs", ascending=False)
print(ud.to_string(), flush=True)
ud.to_csv(os.path.join(RES, "pathG_02_univariate.csv"), index=False, encoding="utf-8-sig")
print("TOTAL %.1f min" % ((time.time() - T0) / 60))
