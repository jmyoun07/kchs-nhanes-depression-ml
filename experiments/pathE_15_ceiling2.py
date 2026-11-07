# -*- coding: utf-8 -*-
"""경로E-15: 17변수의 NHANES 내 정보 상한을 여러 알고리즘으로 엄밀히 확인.

이 결과는 '외부 0.80이 달성 가능한가'라는 질문의 답을 준다.
전부 §2.10 위반(NHANES 라벨 사용) — 진단/상한 확인 전용, 최종 모델에 반영하지 않음.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathE_10_common import *
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

t0 = time.time()
k, pre, post = load()
rows = []


def make_pre_dense():
    """NaN 미허용 모델용: 이진형도 최빈값 대체(전결측 열은 0)."""
    npipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    bpipe = SimpleImputer(strategy="constant", fill_value=0.0)
    cpipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore", drop="first"))])
    return ColumnTransformer([("num", npipe, NUM), ("bin", bpipe, BIN), ("cat", cpipe, CAT)])


NEEDS_DENSE = {"logit", "logit_C0.1", "rf", "hgb"}


def oof_auc(df, model, tag, folds=5, seeds=(42, 7, 2024)):
    X = df[FEATURES]; ya = df["dep"].astype(int).values; Wa = df["W"].to_numpy(float)
    mk = make_pre_dense if any(t in tag for t in NEEDS_DENSE) else make_pre
    aucs = []
    for s in seeds:
        oof = np.zeros(len(df))
        for tr, te in StratifiedKFold(folds, shuffle=True, random_state=s).split(X, ya):
            pipe = Pipeline([("pre", mk()), ("clf", model())])
            pipe.fit(X.iloc[tr], ya[tr])
            oof[te] = pipe.predict_proba(X.iloc[te])[:, 1]
        aucs.append(wauc(ya, oof, Wa))
    return float(np.mean(aucs)), float(np.std(aucs))


MODELS = {
    "logit": lambda: LogisticRegression(max_iter=2000, C=1.0),
    "logit_C0.1": lambda: LogisticRegression(max_iter=2000, C=0.1),
    "rf": lambda: RandomForestClassifier(n_estimators=500, min_samples_leaf=20,
                                         n_jobs=-1, random_state=SEED),
    "lgb_small": lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03,
                                            num_leaves=7, min_child_samples=50,
                                            subsample=0.8, colsample_bytree=0.8,
                                            random_state=SEED, n_jobs=-1, verbose=-1),
    "lgb_tiny": lambda: lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05,
                                           num_leaves=4, min_child_samples=100,
                                           reg_lambda=10, random_state=SEED,
                                           n_jobs=-1, verbose=-1),
    "hgb": lambda: HistGradientBoostingClassifier(max_iter=300, max_leaf_nodes=8,
                                                  learning_rate=0.05, random_state=SEED),
}

for dname, df in [("pre", pre), ("post", post), ("prepost", pd.concat([pre, post], ignore_index=True))]:
    for mname, m in MODELS.items():
        a, s = oof_auc(df, m, f"{dname}_{mname}")
        rows.append(dict(dataset=dname, model=mname, oof_wauc=round(a, 4), sd=round(s, 4)))
        print(rows[-1], flush=True)

out = pd.DataFrame(rows)
out.to_csv(os.path.join(RES, "pathE_15_ceiling_oracle.csv"), index=False, encoding="utf-8-sig")
print(out.groupby("dataset").oof_wauc.max())
print(f"[{time.time()-t0:.0f}s]")
