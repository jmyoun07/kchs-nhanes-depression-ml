# -*- coding: utf-8 -*-
"""
pathC_05_oracle.py — 앙상블로 도달 가능한 '상한'(oracle) 진단.

★ 이 스크립트의 oracle 결과는 §2.10 위반(NHANES 라벨로 가중치 최적화)이며
   최종 모델로 절대 사용하지 않는다. 오직 "앙상블 경로로 0.80이 가능한가?"를
   판정하기 위한 진단용 상한선이다. 결과표에 protocol_compliant=False로 표기.

계산:
  (1) 기저모델 예측 간 상관(다양성) — holdout / pre / post
  (2) NHANES pre에서 직접 최적화한 볼록결합 가중(=상한). post에도 동일 절차.
  (3) 그 상한과 KCHS 내부선택 앙상블의 격차 → '선택 손실' 정량화
출력: results/pathC_oracle.csv, results/pathC_diversity.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.optimize import minimize
from pathC_common import wauc, RES

HERE = os.path.dirname(os.path.abspath(__file__))
CB = os.path.join(HERE, "cache", "base")
SEEDS = [42, 7, 2024, 1, 13, 99, 777, 31337, 555, 2718]
ALL = [f"lgbm_s{s}" for s in SEEDS] + \
      ["lgbm_reg", "lgbm_shallow", "lgbm_deep", "lgbm_dart", "lgbm_goss",
       "rf", "et", "histgb", "enet", "logit_plain", "gnb"]

LB = np.load(os.path.join(CB, "_labels.npz"))
yho, Who, ypre, wpre, ypost, wpost = (LB["yho"], LB["Who"], LB["ypre"],
                                      LB["wpre"], LB["ypost"], LB["wpost"])
names, ho, pre, post = [], [], [], []
for nm in ALL:
    f = os.path.join(CB, nm + ".npz")
    if not os.path.exists(f):
        continue
    z = np.load(f)
    names.append(nm); ho.append(z["ho"]); pre.append(z["pre"]); post.append(z["post"])
ho, pre, post = np.vstack(ho), np.vstack(pre), np.vstack(post)
M = len(names)
print(f"기저모델 {M}개: {names}")


def rank01(p):
    return (pd.Series(p).rank(method="average").to_numpy() - 1) / max(len(p) - 1, 1)


# ---- (1) 다양성 ----
div = []
Rpre = np.vstack([rank01(pre[i]) for i in range(M)])
Cpre = np.corrcoef(Rpre)
Rho = np.vstack([rank01(ho[i]) for i in range(M)])
Cho = np.corrcoef(Rho)
for i in range(M):
    div.append(dict(model=names[i],
                    mean_rankcorr_ho=float((Cho[i].sum() - 1) / (M - 1)),
                    mean_rankcorr_pre=float((Cpre[i].sum() - 1) / (M - 1)),
                    auc_ho=wauc(yho, ho[i], Who), auc_pre=wauc(ypre, pre[i], wpre),
                    auc_post=wauc(ypost, post[i], wpost)))
pd.DataFrame(div).to_csv(os.path.join(RES, "pathC_diversity.csv"),
                         index=False, encoding="utf-8-sig")
print("\n다양성(평균 순위상관):")
print(pd.DataFrame(div).to_string())


# ---- (2) oracle 볼록결합 ----
def opt_convex(mats, y, w, restarts=6, seed=0):
    """볼록결합 가중을 해당 데이터의 AUROC에 직접 최적화(=상한). softmax 파라미터화."""
    R = np.vstack([rank01(mats[i]) for i in range(mats.shape[0])])

    def neg(a):
        e = np.exp(a - a.max()); ww = e / e.sum()
        return -wauc(y, ww @ R, w)

    best, bw = -1, None
    rs = np.random.RandomState(seed)
    for r in range(restarts):
        a0 = np.zeros(M) if r == 0 else rs.normal(0, 1, M)
        res = minimize(neg, a0, method="Nelder-Mead",
                       options=dict(maxiter=4000, xatol=1e-3, fatol=1e-6))
        if -res.fun > best:
            e = np.exp(res.x - res.x.max()); best, bw = -res.fun, e / e.sum()
    return best, bw


rows = []
# 단일모델 oracle
i_pre = int(np.argmax([wauc(ypre, pre[i], wpre) for i in range(M)]))
i_post = int(np.argmax([wauc(ypost, post[i], wpost) for i in range(M)]))
rows.append(dict(what="oracle_best_single_on_pre", target="pre", model=names[i_pre],
                 internal=wauc(yho, ho[i_pre], Who), ext_pre=wauc(ypre, pre[i_pre], wpre),
                 ext_post=wauc(ypost, post[i_pre], wpost), protocol_compliant=False))
rows.append(dict(what="oracle_best_single_on_post", target="post", model=names[i_post],
                 internal=wauc(yho, ho[i_post], Who), ext_pre=wauc(ypre, pre[i_post], wpre),
                 ext_post=wauc(ypost, post[i_post], wpost), protocol_compliant=False))

for tgt, mats, yy, ww in [("pre", pre, ypre, wpre), ("post", post, ypost, wpost)]:
    a, w = opt_convex(mats, yy, ww, seed=1)
    Rall = {k: np.vstack([rank01(v[i]) for i in range(M)])
            for k, v in [("ho", ho), ("pre", pre), ("post", post)]}
    rows.append(dict(what=f"oracle_convex_opt_on_{tgt}", target=tgt,
                     model="+".join(f"{names[j]}:{w[j]:.2f}" for j in range(M) if w[j] > 0.02),
                     internal=wauc(yho, w @ Rall["ho"], Who),
                     ext_pre=wauc(ypre, w @ Rall["pre"], wpre),
                     ext_post=wauc(ypost, w @ Rall["post"], wpost),
                     protocol_compliant=False))
    print(f"\n[oracle {tgt}] 최적 볼록결합 AUROC={a:.4f}")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, "pathC_oracle.csv"), index=False, encoding="utf-8-sig")
print("\n", df.to_string())
print("\n★ 위 oracle 값은 NHANES 라벨로 최적화한 상한이며 §2.10 위반 — 최종모델 아님.")
