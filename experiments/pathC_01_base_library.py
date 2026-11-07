# -*- coding: utf-8 -*-
"""
pathC_01_base_library.py — 앙상블 재료가 될 기저모델 라이브러리 구축.

§2.10 준수: 모든 fit은 KCHS train(1,638,808행)만 사용. NHANES는 predict만.
각 모델의 (holdout, pre, post) 예측확률을 개별 npz로 캐시 → 이후 앙상블 실험에서 재사용.
개별 캐시이므로 중단·재실행에 안전.
출력: experiments/cache/base/<model>.npz, results/pathC_base_models.csv
"""
import os, sys, time, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from pathC_common import load, make_pre, wauc, RES, SEED

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "base")
os.makedirs(CACHE, exist_ok=True)

LGB_BASE = dict(n_estimators=600, learning_rate=0.03, num_leaves=31,
                subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbose=-1)
SEEDS = [42, 7, 2024, 1, 13, 99, 777, 31337, 555, 2718]


def mk(name):
    """모델 팩토리. 하이퍼파라미터는 KCHS 내부 근거/기존 프로토콜에서만 유래(NHANES 미참조)."""
    m_seed = re.fullmatch(r"lgbm_s(\d+)", name)   # lgbm_shallow 오매칭 방지
    if m_seed:
        s = int(m_seed.group(1))
        return Pipeline([("pre", make_pre(True)),
                         ("clf", lgb.LGBMClassifier(**LGB_BASE, random_state=s))])
    if name == "lgbm_reg":
        return Pipeline([("pre", make_pre(True)),
                         ("clf", lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03,
                                                    num_leaves=15, min_child_samples=200,
                                                    reg_lambda=10.0, subsample=0.7,
                                                    colsample_bytree=0.7, random_state=SEED,
                                                    n_jobs=-1, verbose=-1))])
    if name == "lgbm_shallow":
        return Pipeline([("pre", make_pre(True)),
                         ("clf", lgb.LGBMClassifier(n_estimators=800, learning_rate=0.03,
                                                    max_depth=3, num_leaves=8,
                                                    subsample=0.8, colsample_bytree=0.8,
                                                    random_state=SEED, n_jobs=-1, verbose=-1))])
    if name == "lgbm_deep":
        return Pipeline([("pre", make_pre(True)),
                         ("clf", lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03,
                                                    num_leaves=63, subsample=0.8,
                                                    colsample_bytree=0.8, random_state=SEED,
                                                    n_jobs=-1, verbose=-1))])
    if name == "lgbm_dart":
        return Pipeline([("pre", make_pre(True)),
                         ("clf", lgb.LGBMClassifier(boosting_type="dart", n_estimators=400,
                                                    learning_rate=0.05, num_leaves=31,
                                                    subsample=0.8, colsample_bytree=0.8,
                                                    random_state=SEED, n_jobs=-1, verbose=-1))])
    if name == "lgbm_goss":
        return Pipeline([("pre", make_pre(True)),
                         ("clf", lgb.LGBMClassifier(boosting_type="goss", n_estimators=600,
                                                    learning_rate=0.03, num_leaves=31,
                                                    colsample_bytree=0.8, random_state=SEED,
                                                    n_jobs=-1, verbose=-1))])
    if name == "rf":
        return Pipeline([("pre", make_pre(True)),
                         # n_jobs 축소·트리수 축소: 1.64M행에서 n_jobs=-1은 MemoryError 발생
                         ("clf", RandomForestClassifier(n_estimators=200, max_depth=14,
                                                        min_samples_leaf=100, n_jobs=3,
                                                        random_state=SEED))])
    if name == "et":
        return Pipeline([("pre", make_pre(False)),
                         ("clf", ExtraTreesClassifier(n_estimators=200, max_depth=16,
                                                      min_samples_leaf=100, n_jobs=3,
                                                      random_state=SEED))])
    if name == "histgb":
        return Pipeline([("pre", make_pre(True)),
                         ("clf", HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                                                max_leaf_nodes=31,
                                                                l2_regularization=1.0,
                                                                random_state=SEED))])
    if name == "enet":
        return Pipeline([("pre", make_pre(False)),
                         ("clf", LogisticRegression(penalty="elasticnet", l1_ratio=0.5, C=1.0,
                                                    solver="saga", max_iter=200, n_jobs=-1))])
    if name == "logit_plain":
        return Pipeline([("pre", make_pre(False)),
                         ("clf", LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                                    max_iter=500, n_jobs=-1))])
    if name == "gnb":
        return Pipeline([("pre", make_pre(False)), ("clf", GaussianNB())])
    raise ValueError(name)


NAMES = [f"lgbm_s{s}" for s in SEEDS] + \
        ["lgbm_reg", "lgbm_shallow", "lgbm_deep", "lgbm_dart", "lgbm_goss",
         "rf", "et", "histgb", "enet", "logit_plain", "gnb"]

if __name__ == "__main__":
    d = load("full02")
    print(f"train={len(d.Xtr)} holdout={len(d.Xho)} pre={len(d.Xpre)} post={len(d.Xpost)}")
    sys.stdout.flush()
    # 라벨/가중치도 1회 저장(앙상블 스크립트에서 재사용)
    np.savez_compressed(os.path.join(CACHE, "_labels.npz"),
                        yho=d.yho, Who=d.Who, ypre=d.ypre, wpre=d.wpre,
                        ypost=d.ypost, wpost=d.wpost)
    rows = []
    for nm in NAMES:
        f = os.path.join(CACHE, nm + ".npz")
        t0 = time.time()
        if os.path.exists(f):
            z = np.load(f)
            pho, ppre, ppost = z["ho"], z["pre"], z["post"]
            secs = float(z["secs"])
        else:
            try:
                m = mk(nm)
                m.fit(d.Xtr, d.ytr)
                pho = m.predict_proba(d.Xho)[:, 1]
                ppre = m.predict_proba(d.Xpre)[:, 1]
                ppost = m.predict_proba(d.Xpost)[:, 1]
            except Exception as e:
                print(f"[{nm}] 실패: {type(e).__name__}: {e}")
                sys.stdout.flush()
                continue
            secs = time.time() - t0
            np.savez_compressed(f, ho=pho, pre=ppre, post=ppost, secs=secs)
        r = dict(model=nm, internal=wauc(d.yho, pho, d.Who),
                 ext_pre=wauc(d.ypre, ppre, d.wpre),
                 ext_post=wauc(d.ypost, ppost, d.wpost), secs=secs)
        rows.append(r)
        print(f"{nm:14s} 내부={r['internal']:.4f} pre={r['ext_pre']:.4f} "
              f"post={r['ext_post']:.4f} [{secs:.0f}s]")
        sys.stdout.flush()
    pd.DataFrame(rows).to_csv(os.path.join(RES, "pathC_base_models.csv"),
                              index=False, encoding="utf-8-sig")
    print("완료:", CACHE)
