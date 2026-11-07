# -*- coding: utf-8 -*-
"""
pathH_04_ensembles.py — 캐시된 기저모델 예측만으로 앙상블 결합(재학습 없음).

§2.10 엄수:
  - 모든 가중치·조합 선택은 KCHS train(500k) 5-fold OOF 의 가중 AUROC 로만 수행.
  - 내부 holdout(100k)·NHANES pre/post 는 결과 보고에만 사용.
  - 단, '상한선 진단' 목적의 oracle 항목은 protocol_compliant=False 로 명시하고
    선택에 외부 라벨을 쓴다(연구 결론용 아님, 한계치 파악용).

결합 규칙: 확률평균(prob) / 순위평균(rank) / 로짓평균(logit)
가중 결정: 균등 / OOF-AUC 가중 / 탐욕적 힐클라이밍(Caruana) / NNLS / 로지스틱 스태킹
출력: results/pathH_04_ensembles.csv
"""
import os, sys, time, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from pathH_common import *
from scipy.optimize import nnls
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

OUT = os.path.join(RES, "pathH_04_ensembles.csv")
d = load()
names = list_cached()
print(f"cached models: {len(names)}")

# ---------------- 예측 행렬 적재 ----------------
P = {}
for n in names:
    z = load_preds(n)
    P[n] = z
NAMES = [n for n in names]
M = len(NAMES)


def mat(key, sub=None):
    sub = NAMES if sub is None else sub
    return np.vstack([P[n][key] for n in sub])


# 랭크 변환 캐시 (선택 탐색 속도용)
print("rank transform ...", flush=True)
t0 = time.time()
RK = {k: {n: rank01(P[n][k]) for n in NAMES} for k in ("oof", "ho", "pre", "post")}
LG = {k: {n: logit(P[n][k]) for n in NAMES} for k in ("oof", "ho", "pre", "post")}
print(f"  {time.time()-t0:.0f}s", flush=True)

BASE_OOF = {n: wauc(d.ytr, P[n]["oof"], d.Wtr) for n in NAMES}
ROWS = []


def combine(sub, how, w=None, key="oof"):
    w = np.ones(len(sub)) if w is None else np.asarray(w, float)
    w = w / w.sum()
    if how == "prob":
        Z = np.vstack([P[n][key] for n in sub])
    elif how == "rank":
        Z = np.vstack([RK[key][n] for n in sub])
    elif how == "logit":
        Z = np.vstack([LG[key][n] for n in sub])
    else:
        raise ValueError(how)
    return w @ Z


def evaluate(name, sub, how, w=None, compliant=True, note="", method=""):
    r = dict(name=name, method=method, how=how, n_members=len(sub),
             oof=wauc(d.ytr, combine(sub, how, w, "oof"), d.Wtr),
             internal=wauc(d.yho, combine(sub, how, w, "ho"), d.Who),
             ext_pre=wauc(d.ypre, combine(sub, how, w, "pre"), d.wpre),
             ext_post=wauc(d.ypost, combine(sub, how, w, "post"), d.wpost),
             protocol_compliant=bool(compliant),
             members=";".join(sub) if len(sub) <= 25 else f"{len(sub)} models",
             note=note)
    ROWS.append(r)
    print(f"{name:34s} OOF={r['oof']:.4f} 내부={r['internal']:.4f} "
          f"pre={r['ext_pre']:.4f} post={r['ext_post']:.4f} "
          f"{'' if compliant else '[NON-COMPLIANT]'}", flush=True)
    pd.DataFrame(ROWS).to_csv(OUT, index=False, encoding="utf-8-sig")
    return r


# ---------------- 0) 단일 기준 ----------------
if "lgbm_base" in P:
    evaluate("BASE_lgbm_base", ["lgbm_base"], "prob", method="single",
             note="명세 기준선")

# ---------------- 1) 가족(family) 정의 ----------------
def fam(pred):
    return [n for n in NAMES if pred(n)]


GBDT = fam(lambda n: n.startswith(("lgbm_", "xgb_", "cat_")) or n == "histgb")
DEPTH = [n for n in ("lgbm_stump", "lgbm_d2", "lgbm_d3", "lgbm_d4", "lgbm_base",
                     "lgbm_deep") if n in P]
ALGDIV = [n for n in ("lgbm_base", "xgb_base", "cat_base", "histgb", "rf", "et",
                      "logit_l2", "logit_spline", "mlp32") if n in P]
ALGDIV_NOWEAK = [n for n in ALGDIV if n not in ("gnb",)]
SUBSP = fam(lambda n: n.startswith(("drop_", "rsub", "blk_")))
DROPONE = fam(lambda n: n.startswith("drop_"))
RSUB = fam(lambda n: n.startswith("rsub"))
EXPERTS = fam(lambda n: n.startswith("exp_"))
ALL = NAMES

FAMS = dict(all=ALL, gbdt=GBDT, depth=DEPTH, algdiv=ALGDIV, subspace=SUBSP,
            dropone=DROPONE, rsub=RSUB, experts=EXPERTS)
for fn, sub in FAMS.items():
    if len(sub) < 2:
        continue
    for how in ("prob", "rank", "logit"):
        evaluate(f"F_{fn}_{how}", sub, how, method="uniform",
                 note=f"{fn} 균등 결합")

# ---------------- 2) OOF 상위 k 순위평균 ----------------
order = sorted(NAMES, key=lambda n: -BASE_OOF[n])
for k in (2, 3, 5, 8, 12, 20):
    if k <= M:
        for how in ("rank", "logit"):
            evaluate(f"TOP{k}_{how}", order[:k], how, method="topk_oof",
                     note=f"OOF AUC 상위 {k}개")

# ---------------- 3) OOF-AUC 가중 ----------------
for fn in ("all", "gbdt", "algdiv", "subspace"):
    sub = FAMS[fn]
    if len(sub) < 2:
        continue
    w = np.array([max(BASE_OOF[n] - 0.5, 1e-6) ** 4 for n in sub])
    evaluate(f"WAUC_{fn}_rank", sub, "rank", w, method="oofauc_weight",
             note="OOF AUC^4 가중")


# ---------------- 4) 탐욕적 힐클라이밍 (Caruana) ----------------
def greedy(sub, how, n_iter=40, init_k=1, bag=None, seed=0, verbose=False):
    """OOF 가중AUROC 를 최대화하는 중복허용 탐욕 선택. 반환: 카운트 가중치."""
    rng = np.random.RandomState(seed)
    Z = np.vstack([(RK["oof"][n] if how == "rank" else
                    LG["oof"][n] if how == "logit" else P[n]["oof"]) for n in sub])
    cnt = np.zeros(len(sub))
    # 초기화: OOF 최고 모델
    scores = np.array([wauc(d.ytr, Z[i], d.Wtr) for i in range(len(sub))])
    for i in np.argsort(-scores)[:init_k]:
        cnt[i] += 1
    cur = (cnt @ Z) / cnt.sum()
    best = wauc(d.ytr, cur, d.Wtr)
    for it in range(n_iter):
        pool = range(len(sub)) if bag is None else rng.choice(
            len(sub), size=min(bag, len(sub)), replace=False)
        cand_best, cand_i = best, -1
        tot = cnt.sum()
        for i in pool:
            newp = (cur * tot + Z[i]) / (tot + 1)
            s = wauc(d.ytr, newp, d.Wtr)
            if s > cand_best + 1e-9:
                cand_best, cand_i = s, i
        if cand_i < 0:
            break
        cnt[cand_i] += 1
        cur = (cnt @ Z) / cnt.sum()
        best = cand_best
        if verbose:
            print(f"   greedy it{it}: +{sub[cand_i]} OOF={best:.5f}", flush=True)
    return cnt, best


t0 = time.time()
for how in ("rank", "logit"):
    cnt, s = greedy(ALL, how, n_iter=30, verbose=True)
    sel = [n for n, c in zip(ALL, cnt) if c > 0]
    w = np.array([c for c in cnt if c > 0])
    evaluate(f"GREEDY_all_{how}", sel, how, w, method="greedy_oof",
             note=f"Caruana 탐욕(OOF만), 반복30, OOF={s:.5f}")
print(f"greedy {time.time()-t0:.0f}s", flush=True)

# 배깅 탐욕(모델 부분집합 무작위 풀) — 과적합 완화
cnts = np.zeros(M)
for b in range(10):
    c, s = greedy(ALL, "rank", n_iter=15, bag=max(5, M // 2), seed=100 + b)
    cnts += c
sel = [n for n, c in zip(ALL, cnts) if c > 0]
w = np.array([c for c in cnts if c > 0])
evaluate("GREEDYBAG_all_rank", sel, "rank", w, method="greedy_bagged",
         note="배깅 탐욕 10회 평균(OOF만)")

# ---------------- 5) NNLS ----------------
sub = ALL
A = np.vstack([RK["oof"][n] for n in sub]).T
sw = np.sqrt(d.Wtr / d.Wtr.mean())
yc = d.ytr.astype(float)
try:
    coef, _ = nnls(A * sw[:, None], yc * sw)
    if coef.sum() > 0:
        keep = coef > 1e-8
        evaluate("NNLS_all_rank", [n for n, k in zip(sub, keep) if k], "rank",
                 coef[keep], method="nnls_oof", note="가중 NNLS(OOF만)")
except Exception as e:
    print("NNLS fail", e)


# ---------------- 6) 로지스틱 스태킹 ----------------
def stack_eval(sub, feat="logit", weighted=True, C=1.0, tag=""):
    Xo = np.vstack([(LG["oof"][n] if feat == "logit" else RK["oof"][n])
                    for n in sub]).T
    y = d.ytr
    # 정직한 선택지표: OOF 내부 5-fold CV
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof2 = np.zeros(len(y))
    for ia, ib in skf.split(Xo, y):
        m = LogisticRegression(C=C, max_iter=3000)
        m.fit(Xo[ia], y[ia], sample_weight=d.Wtr[ia] if weighted else None)
        oof2[ib] = m.decision_function(Xo[ib])
    sel_score = wauc(y, oof2, d.Wtr)
    m = LogisticRegression(C=C, max_iter=3000)
    m.fit(Xo, y, sample_weight=d.Wtr if weighted else None)
    outs = {}
    for key, yy, ww in (("ho", d.yho, d.Who), ("pre", d.ypre, d.wpre),
                        ("post", d.ypost, d.wpost)):
        Xk = np.vstack([(LG[key][n] if feat == "logit" else RK[key][n])
                        for n in sub]).T
        outs[key] = wauc(yy, m.decision_function(Xk), ww)
    r = dict(name=f"STACK_{tag}", method="stacking", how=feat, n_members=len(sub),
             oof=sel_score, internal=outs["ho"], ext_pre=outs["pre"],
             ext_post=outs["post"], protocol_compliant=True,
             members=f"{len(sub)} models",
             note=f"메타=로지스틱 C={C} weighted={weighted}; 선택지표=OOF내 5-fold CV")
    ROWS.append(r)
    print(f"{r['name']:34s} OOF={r['oof']:.4f} 내부={r['internal']:.4f} "
          f"pre={r['ext_pre']:.4f} post={r['ext_post']:.4f}", flush=True)
    pd.DataFrame(ROWS).to_csv(OUT, index=False, encoding="utf-8-sig")
    return r


for tag, sub in (("all", ALL), ("gbdt", GBDT), ("algdiv", ALGDIV),
                 ("subspace", SUBSP)):
    if len(sub) < 2:
        continue
    for C in (1.0, 0.01):
        for wgt in (True, False):
            stack_eval(sub, "logit", wgt, C, tag=f"{tag}_C{C}_w{int(wgt)}")

# ---------------- 7) [비준수] oracle 상한 진단 ----------------
# 외부(pre) 라벨로 가중치를 고른 경우의 상한 — 연구결론 아님, 한계치 파악용.
def greedy_ext(sub, how, key="pre", n_iter=30):
    Z = np.vstack([(RK[key][n] if how == "rank" else LG[key][n]) for n in sub])
    yy, ww = (d.ypre, d.wpre) if key == "pre" else (d.ypost, d.wpost)
    cnt = np.zeros(len(sub))
    scores = np.array([wauc(yy, Z[i], ww) for i in range(len(sub))])
    cnt[int(np.argmax(scores))] += 1
    cur = (cnt @ Z) / cnt.sum()
    best = wauc(yy, cur, ww)
    for it in range(n_iter):
        cb, ci = best, -1
        tot = cnt.sum()
        for i in range(len(sub)):
            s = wauc(yy, (cur * tot + Z[i]) / (tot + 1), ww)
            if s > cb + 1e-9:
                cb, ci = s, i
        if ci < 0:
            break
        cnt[ci] += 1
        cur = (cnt @ Z) / cnt.sum()
        best = cb
    return cnt, best


for key in ("pre", "post"):
    cnt, s = greedy_ext(ALL, "rank", key=key)
    sel = [n for n, c in zip(ALL, cnt) if c > 0]
    w = np.array([c for c in cnt if c > 0])
    evaluate(f"ORACLE_greedy_{key}_rank", sel, "rank", w, compliant=False,
             method="oracle_ext", note=f"[비준수] {key} 라벨로 가중치 선택. 상한 진단용")

best_single_pre = max(NAMES, key=lambda n: wauc(d.ypre, P[n]["pre"], d.wpre))
evaluate("ORACLE_best_single_pre", [best_single_pre], "prob", compliant=False,
         method="oracle_ext", note="[비준수] pre 기준 최고 단일모델")
best_single_post = max(NAMES, key=lambda n: wauc(d.ypost, P[n]["post"], d.wpost))
evaluate("ORACLE_best_single_post", [best_single_post], "prob", compliant=False,
         method="oracle_ext", note="[비준수] post 기준 최고 단일모델")

pd.DataFrame(ROWS).to_csv(OUT, index=False, encoding="utf-8-sig")
print("SAVED", OUT)

# 기저모델 단일 성능표도 저장
bm = pd.DataFrame([dict(name=n, oof=BASE_OOF[n],
                        internal=wauc(d.yho, P[n]["ho"], d.Who),
                        ext_pre=wauc(d.ypre, P[n]["pre"], d.wpre),
                        ext_post=wauc(d.ypost, P[n]["post"], d.wpost),
                        secs=P[n]["secs"], note=P[n]["note"]) for n in NAMES])
bm.sort_values("oof", ascending=False).to_csv(
    os.path.join(RES, "pathH_base_models.csv"), index=False, encoding="utf-8-sig")
print("SAVED", os.path.join(RES, "pathH_base_models.csv"))
