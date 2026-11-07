# -*- coding: utf-8 -*-
"""
경로6 부속 — ★한계치 확정 (진단 전용, §2.10 비준수: NHANES 라벨 사용)

목적: '17변수로 NHANES에서 달성 가능한 최대 가중 AUROC'의 상한을 추정한다.
      KCHS 학습 없이 NHANES 자체에 직접 적합(oracle)한 교차적합 성능이 상한 근사이다.
      전이학습(KCHS→NHANES)이 이 상한을 넘을 수는 없다.
결과는 절대 모델 선택에 사용하지 않으며 보고서의 '한계' 진술 근거로만 쓴다.
"""
import numpy as np, pandas as pd, time
from path6_common import *
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

rows = []
T0 = time.time()

PA = PRE.reset_index(drop=True); QA = POST.reset_index(drop=True)
POOL = pd.concat([PA, QA], ignore_index=True)
src = np.r_[np.zeros(len(PA)), np.ones(len(QA))].astype(int)
yP = POOL["dep"].astype(int).to_numpy(); wP = POOL["W"].to_numpy(float)
XPOOL = POOL[FEATURES]


def crossfit(make_clf, tag, X, y, w, folds=5, reps=3):
    pre_s, post_s, all_s = [], [], []
    for rep in range(reps):
        oof = np.zeros(len(X))
        for tr, va in StratifiedKFold(folds, shuffle=True, random_state=100 + rep).split(X, y):
            clf = make_clf()
            bi = not isinstance(clf, lgb.LGBMClassifier)
            from sklearn.pipeline import Pipeline
            pipe = Pipeline([("pre", make_prep(NUM, BIN, CAT, bi)), ("clf", clf)])
            pipe.fit(X.iloc[tr], y[tr])
            oof[va] = pipe.predict_proba(X.iloc[va])[:, 1]
        m0, m1 = src == 0, src == 1
        pre_s.append(wauc(y[m0], oof[m0], w[m0]))
        post_s.append(wauc(y[m1], oof[m1], w[m1]))
        all_s.append(wauc(y, oof, w))
    return dict(name=tag, internal=None,
                ext_pre=round(float(np.mean(pre_s)), 6),
                ext_post=round(float(np.mean(post_s)), 6),
                protocol_compliant=False, seconds=None,
                note=f"★ORACLE 상한진단(§2.10 위반): NHANES 자체 교차적합 5-fold×3rep, "
                     f"pooled={np.mean(all_s):.4f}, sd_pre={np.std(pre_s):.4f}")


cands = [
    (lambda: LogisticRegression(C=1.0, max_iter=5000), "CEIL_oracle_logit_C1"),
    (lambda: LogisticRegression(C=0.1, max_iter=5000), "CEIL_oracle_logit_C0.1"),
    (lambda: LogisticRegression(C=0.03, max_iter=5000), "CEIL_oracle_logit_C0.03"),
    (lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7,
                                max_depth=3, subsample=0.8, colsample_bytree=0.8,
                                min_child_samples=100, reg_lambda=10,
                                random_state=SEED, n_jobs=-1, verbose=-1),
     "CEIL_oracle_lgbm_shallow"),
    (lambda: lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=4,
                                max_depth=2, min_child_samples=200, reg_lambda=30,
                                random_state=SEED, n_jobs=-1, verbose=-1),
     "CEIL_oracle_lgbm_d2"),
]
for f, t in cands:
    r = crossfit(f, t, XPOOL, yP, wP)
    rows.append(r); print(r, flush=True)

# 학습곡선: 홀드아웃 크기를 고정(30%)하고 학습표본만 늘려 상한 외삽
lc = []
from sklearn.pipeline import Pipeline
for frac in [0.15, 0.3, 0.5, 0.7, 1.0]:
    sc = []
    for rep in range(10):
        rs = np.random.RandomState(200 + rep)
        perm = rs.permutation(len(XPOOL))
        hold = perm[:int(0.3 * len(XPOOL))]
        pool = perm[int(0.3 * len(XPOOL)):]
        idx = pool[:int(frac * len(pool))]
        pipe = Pipeline([("pre", make_prep(NUM, BIN, CAT, True)),
                         ("clf", LogisticRegression(C=0.1, max_iter=5000))])
        pipe.fit(XPOOL.iloc[idx], yP[idx])
        sc.append(wauc(yP[hold], pipe.predict_proba(XPOOL.iloc[hold])[:, 1], wP[hold]))
    lc.append((frac, len(idx), float(np.mean(sc)), float(np.std(sc))))
    print("learncurve", lc[-1], flush=True)
pd.DataFrame(lc, columns=["frac", "n_train", "wauc_mean", "wauc_sd"]).to_csv(
    os.path.join(RES, "path6_98_learncurve.csv"), index=False, encoding="utf-8-sig")

# KCHS 내부에서 동일 17변수가 낼 수 있는 상한 (참고: 대표본 in-domain)
rows.append(dict(name="REF_kchs_internal_baseline", internal=0.830969,
                 ext_pre=None, ext_post=None, protocol_compliant=True, seconds=None,
                 note="참고: 동일 17변수 KCHS 내부(대표본, 동일분포) 성능"))

save(rows, "path6_98_ceiling.csv")
print("TOTAL", round(time.time() - T0, 1), "s")
