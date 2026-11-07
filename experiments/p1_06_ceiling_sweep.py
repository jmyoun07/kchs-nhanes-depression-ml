# -*- coding: utf-8 -*-
"""경로1-06: 정보 상한(ceiling)을 최대한 관대하게 추정.

NHANES 자체 라벨로 여러 모형을 5-fold CV 적합 → 최댓값을 상한으로 채택.
(모형선택을 CV 최댓값으로 하므로 위쪽으로 편향 = 상한 주장에 유리)
★ 전부 §2.10 비준수(진단 전용). 어떤 결과도 KCHS 모델 선택에 사용하지 않는다.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_common import *
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

k, pre, post = load()
pool = pd.concat([pre, post], ignore_index=True)
SETS = {"pre": pre, "post": post, "pooled": pool}

MODELS = {
    "logit_C1": lambda: LogisticRegression(max_iter=3000, C=1.0),
    "logit_C0.1": lambda: LogisticRegression(max_iter=3000, C=0.1),
    "logit_C0.01": lambda: LogisticRegression(max_iter=3000, C=0.01),
    "lgb_stump": lambda: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                                            max_depth=1, num_leaves=2,
                                            random_state=SEED, n_jobs=4, verbose=-1),
    "lgb_l7_reg": lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                             num_leaves=7, min_child_samples=100,
                                             reg_lambda=10.0, random_state=SEED,
                                             n_jobs=4, verbose=-1),
    "lgb_l15": lambda: lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03,
                                          num_leaves=15, min_child_samples=50,
                                          random_state=SEED, n_jobs=4, verbose=-1),
    "lgb_base": lambda: lgb.LGBMClassifier(**dict(LGB_BASE, n_jobs=4)),
    "rf": lambda: RandomForestClassifier(n_estimators=500, min_samples_leaf=25,
                                         random_state=SEED, n_jobs=4),
}

rows = []
for sname, d in SETS.items():
    X = d[FEATURES]
    yv = d["dep"].astype(int).values
    wv = d["W"].values
    for mname, mk in MODELS.items():
        oof = np.zeros(len(d))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, yv):
            steps = [("pre", make_pre())]
            if mname.startswith(("logit", "rf")):
                steps.append(("imp2", SimpleImputer(strategy="most_frequent")))
            steps.append(("clf", mk()))
            m = Pipeline(steps)
            m.fit(X.iloc[tr], yv[tr])
            oof[te] = m.predict_proba(X.iloc[te])[:, 1]
        a = wauc(yv, oof, wv)
        r = dict(dataset=sname, model=mname, auc=round(float(a), 6),
                 protocol_compliant=False)
        if sname == "pooled":
            n0 = len(pre)
            r["auc_pre_part"] = round(float(wauc(pre["dep"], oof[:n0], pre["W"])), 6)
            r["auc_post_part"] = round(float(wauc(post["dep"], oof[n0:], post["W"])), 6)
        rows.append(r)
        print(json.dumps(r), flush=True)

df = dump(rows, "p1_06_ceiling_sweep.csv")
print("\n=== 상한 요약 (자체적합 5-fold CV 최댓값) ===")
print(df.groupby("dataset")["auc"].max())
print("pooled 부분:", df[df.dataset == "pooled"][["model", "auc_pre_part", "auc_post_part"]]
      .sort_values("auc_pre_part", ascending=False).head(3).to_string(index=False))
