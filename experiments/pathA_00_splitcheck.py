# -*- coding: utf-8 -*-
"""기준선 재현: train_test_split 변형별 내부/외부 AUROC 대조."""
import os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathA_feature_repr import (K, PRE, POST, yK, WK, NUM, BIN, CAT, LGBM,
                                make_prep, wauc, SEED)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import lightgbm as lgb

FEATS = NUM + BIN + CAT
X = K[FEATS]


def run(itr, ite, tag):
    t0 = time.time()
    pipe = Pipeline([("pre", make_prep(NUM, BIN, CAT)),
                     ("clf", lgb.LGBMClassifier(**LGBM))])
    pipe.fit(X.iloc[itr], yK[itr])
    a = wauc(yK[ite], pipe.predict_proba(X.iloc[ite])[:, 1], WK[ite])
    p = wauc(PRE["dep"], pipe.predict_proba(PRE[FEATS])[:, 1], PRE["W"])
    o = wauc(POST["dep"], pipe.predict_proba(POST[FEATS])[:, 1], POST["W"])
    print(f"{tag:34s} int={a:.4f} pre={p:.4f} post={o:.4f} [{time.time()-t0:.0f}s]", flush=True)
    return dict(variant=tag, internal=a, ext_pre=p, ext_post=o)


rows = []
idx = np.arange(len(K))

a, b = train_test_split(idx, train_size=500_000, test_size=100_000,
                        stratify=yK, random_state=SEED)
rows.append(run(a, b, "v1_ttsplit_unsorted"))
rows.append(run(np.sort(a), np.sort(b), "v2_ttsplit_sorted"))

# v3: 먼저 test 100k 떼고 나머지에서 train 500k
rest, te = train_test_split(idx, test_size=100_000, stratify=yK, random_state=SEED)
tr, _ = train_test_split(rest, train_size=500_000, stratify=yK[rest], random_state=SEED)
rows.append(run(tr, te, "v3_test_then_train"))

# v4: 먼저 train 500k 떼고 나머지에서 test 100k
tr2, rest2 = train_test_split(idx, train_size=500_000, stratify=yK, random_state=SEED)
te2, _ = train_test_split(rest2, train_size=100_000, stratify=yK[rest2], random_state=SEED)
rows.append(run(tr2, te2, "v4_train_then_test"))

# v5: 60만 서브샘플 후 5:1
sub, _ = train_test_split(idx, train_size=600_000, stratify=yK, random_state=SEED)
tr3, te3 = train_test_split(sub, train_size=500_000, stratify=yK[sub], random_state=SEED)
rows.append(run(tr3, te3, "v5_sub600k_then_split"))

pd.DataFrame(rows).to_csv(
    r"C:/Users/rudtn/Desktop/claude code/지역사회건강/results/pathA_splitcheck.csv",
    index=False, encoding="utf-8-sig")
print("target: int=0.8336 pre=0.7681 post=0.7682")
