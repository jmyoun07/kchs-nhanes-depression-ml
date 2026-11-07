# -*- coding: utf-8 -*-
"""
pathC_04_bagging.py — 배깅/서브샘플 앙상블.

모두 KCHS train만 사용(§2.10). NHANES는 predict만.
  (1) bootstrap 배깅: LGBM을 부트스트랩 표본 B개에 적합 → 평균/순위평균
  (2) subsample 배깅: 무복원 25% 표본 B개
  (3) 연도블록 배깅: 연도별로 따로 학습한 뒤 평균(데이터 분할 다양성)
출력: results/pathC_bagging.csv, experiments/cache/bag/*.npz
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from pathC_common import load, make_pre, wauc, RES, SEED

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache", "bag")
os.makedirs(CACHE, exist_ok=True)

LGB = dict(n_estimators=600, learning_rate=0.03, num_leaves=31,
           subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbose=-1)
B_N = 15


def rank01(p):
    return (pd.Series(p).rank(method="average").to_numpy() - 1) / max(len(p) - 1, 1)


def agg(mats, how):
    if how == "prob":
        return mats.mean(0)
    if how == "rank":
        return np.vstack([rank01(m) for m in mats]).mean(0)
    if how == "logit":
        q = np.clip(mats, 1e-7, 1 - 1e-7)
        return np.log(q / (1 - q)).mean(0)


def fit_one(Xt, yt, seed, Xs):
    m = Pipeline([("pre", make_pre(True)),
                  ("clf", lgb.LGBMClassifier(**LGB, random_state=seed))])
    m.fit(Xt, yt)
    return [m.predict_proba(x)[:, 1] for x in Xs]


if __name__ == "__main__":
    d = load("full02")
    Xs = [d.Xho, d.Xpre, d.Xpost]
    rows = []

    def run(tag, splitter):
        f = os.path.join(CACHE, tag + ".npz")
        if os.path.exists(f):
            z = np.load(f)
            P = [z["ho"], z["pre"], z["post"]]
        else:
            acc = [[], [], []]
            for b, (idx, sd) in enumerate(splitter()):
                t0 = time.time()
                pr = fit_one(d.Xtr.iloc[idx], d.ytr[idx], sd, Xs)
                for j in range(3):
                    acc[j].append(pr[j])
                print(f"  {tag} b{b+1} n={len(idx)} [{time.time()-t0:.0f}s]")
                sys.stdout.flush()
            P = [np.vstack(a) for a in acc]
            np.savez_compressed(f, ho=P[0], pre=P[1], post=P[2])
        for how in ["prob", "rank", "logit"]:
            for k in [3, 5, len(P[0])]:
                r = dict(method=f"{tag}_B{k}_{how}", n_members=k,
                         internal=wauc(d.yho, agg(P[0][:k], how), d.Who),
                         ext_pre=wauc(d.ypre, agg(P[1][:k], how), d.wpre),
                         ext_post=wauc(d.ypost, agg(P[2][:k], how), d.wpost))
                rows.append(r)
                print(f"{r['method']:32s} 내부={r['internal']:.4f} "
                      f"pre={r['ext_pre']:.4f} post={r['ext_post']:.4f}")
                sys.stdout.flush()

    n = len(d.Xtr)

    def boot():
        rs = np.random.RandomState(SEED)
        for b in range(B_N):
            yield rs.choice(n, n, replace=True), 1000 + b

    def sub25():
        rs = np.random.RandomState(SEED + 1)
        for b in range(B_N):
            yield rs.choice(n, int(n * 0.25), replace=False), 2000 + b

    def yearblock():
        yr = d.k["year"].values[d.itr]
        for b, yv in enumerate(np.unique(yr)):
            yield np.where(yr == yv)[0], 3000 + b

    run("bootstrap", boot)
    run("sub25", sub25)
    run("yearblock", yearblock)

    pd.DataFrame(rows).to_csv(os.path.join(RES, "pathC_bagging.csv"),
                              index=False, encoding="utf-8-sig")
    print("완료")
