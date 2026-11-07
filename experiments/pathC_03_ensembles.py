# -*- coding: utf-8 -*-
"""
pathC_03_ensembles.py — 앙상블·스태킹·보정 실험 본체(캐시된 예측만 사용, 재학습 없음).

두 계열:
  [A] base 계열 — 기저모델을 KCHS train 전체(1.64M)에서 fit. 고정 규칙 앙상블(평균/순위평균/로짓평균).
      조합 선택은 KCHS 내부 holdout 기준으로만(§2.10 준수, NHANES 미참조).
  [B] oof 계열 — 기저모델을 base_tr(80%)에서 fit, meta(20%)로 가중치·스태킹·보정 학습.
      holdout/pre/post는 순수 평가. 가장 엄정한 설계.

핵심 구분(과제 요구):
  - AUROC는 단조변환 불변 → 단일모델 확률보정(Platt/Isotonic)은 AUROC를 바꾸지 않는다(E섹션에서 실증).
  - 그러나 '보정 후 평균'은 각 모델의 척도를 바꾼 뒤 합치므로 순위가 바뀌고 AUROC가 달라진다(D섹션).
출력: results/pathC_ensembles.csv
"""
import os, sys, time, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from pathC_common import wauc, RES

HERE = os.path.dirname(os.path.abspath(__file__))
CB = os.path.join(HERE, "cache", "base")
CO = os.path.join(HERE, "cache", "oof")
SEEDS = [42, 7, 2024, 1, 13, 99, 777, 31337, 555, 2718]
ROWS = []


def add(family, method, members, internal, ext_pre, ext_post, note=""):
    ROWS.append(dict(family=family, method=method, members=members,
                     internal=internal, ext_pre=ext_pre, ext_post=ext_post, note=note))
    print(f"[{family}] {method:34s} 내부={internal:.4f} pre={ext_pre:.4f} "
          f"post={ext_post:.4f}  {members[:60]}")
    sys.stdout.flush()


# ---------- 결합 규칙 ----------
def rank01(p):
    return (pd.Series(p).rank(method="average").to_numpy() - 1) / max(len(p) - 1, 1)


def logit(p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def comb(mats, how, w=None):
    """mats: (M, n) 예측행렬. how in {prob, rank, logit}. w: 가중치(정규화 전)."""
    M = mats.shape[0]
    w = np.ones(M) if w is None else np.asarray(w, float)
    w = w / w.sum()
    if how == "prob":
        Z = mats
    elif how == "rank":
        Z = np.vstack([rank01(mats[i]) for i in range(M)])
    elif how == "logit":
        Z = np.vstack([logit(mats[i]) for i in range(M)])
    else:
        raise ValueError(how)
    return w @ Z


def load_cache(cdir, names, keys):
    out = {k: [] for k in keys}
    keep = []
    for nm in names:
        f = os.path.join(cdir, nm + ".npz")
        if not os.path.exists(f):
            print(f"  (없음, 건너뜀) {nm}")
            continue
        z = np.load(f)
        for k in keys:
            out[k].append(z[k])
        keep.append(nm)
    return keep, {k: np.vstack(v) for k, v in out.items()}


# ================= [A] base 계열 =================
LB = np.load(os.path.join(CB, "_labels.npz"))
yho, Who, ypre, wpre, ypost, wpost = (LB["yho"], LB["Who"], LB["ypre"],
                                      LB["wpre"], LB["ypost"], LB["wpost"])
ALL_A = [f"lgbm_s{s}" for s in SEEDS] + \
        ["lgbm_reg", "lgbm_shallow", "lgbm_deep", "lgbm_dart", "lgbm_goss",
         "rf", "et", "histgb", "enet", "logit_plain", "gnb"]
nmA, A = load_cache(CB, ALL_A, ["ho", "pre", "post"])
idxA = {n: i for i, n in enumerate(nmA)}
print(f"\n[A] 사용가능 기저모델 {len(nmA)}개: {nmA}\n")


def evalA(method, members, how, w=None, note=""):
    ii = [idxA[m] for m in members if m in idxA]
    if len(ii) == 0:
        return
    add("A_base", method, "+".join([nmA[i] for i in ii]),
        wauc(yho, comb(A["ho"][ii], how, w), Who),
        wauc(ypre, comb(A["pre"][ii], how, w), wpre),
        wauc(ypost, comb(A["post"][ii], how, w), wpost), note)


# A0. 단일 기준선
if "lgbm_s42" in idxA:
    i = idxA["lgbm_s42"]
    add("A_base", "single_lgbm_s42(기준선)", "lgbm_s42",
        wauc(yho, A["ho"][i], Who), wauc(ypre, A["pre"][i], wpre),
        wauc(ypost, A["post"][i], wpost), "baseline")

# A1. 시드 앙상블 (k=2,3,5,10)
for k in [2, 3, 5, 10]:
    mem = [f"lgbm_s{s}" for s in SEEDS[:k]]
    for how in ["prob", "rank", "logit"]:
        evalA(f"seed_ens_k{k}_{how}", mem, how, note="분산감소")

# A2. 이종 앙상블 — 고정 조합
COMBOS = {
    "lgbm+rf": ["lgbm_s42", "rf"],
    "lgbm+enet": ["lgbm_s42", "enet"],
    "lgbm+histgb": ["lgbm_s42", "histgb"],
    "lgbm+rf+enet+histgb(과제지정)": ["lgbm_s42", "rf", "enet", "histgb"],
    "lgbm+rf+et+histgb(트리계)": ["lgbm_s42", "rf", "et", "histgb"],
    "lgbm+enet+logit+gnb(선형계혼합)": ["lgbm_s42", "enet", "logit_plain", "gnb"],
    "lgbm+reg+shallow+deep(LGBM변형)": ["lgbm_s42", "lgbm_reg", "lgbm_shallow", "lgbm_deep"],
    "seed10+rf+enet+histgb": [f"lgbm_s{s}" for s in SEEDS] + ["rf", "enet", "histgb"],
    "전체16": ALL_A,
}
for cname, mem in COMBOS.items():
    for how in ["prob", "rank", "logit"]:
        evalA(f"het_{cname}_{how}", mem, how)

# A3. 시드앙상블 기반 2단 결합(시드평균을 1개 모델처럼 취급)
if all(f"lgbm_s{s}" in idxA for s in SEEDS):
    for how in ["prob", "rank", "logit"]:
        se = {k: comb(A[k][[idxA[f"lgbm_s{s}"] for s in SEEDS]], how)
              for k in ["ho", "pre", "post"]}
        for partner in [["rf"], ["enet"], ["rf", "enet"], ["rf", "enet", "histgb"]]:
            if not all(p in idxA for p in partner):
                continue
            mats = {k: np.vstack([se[k]] + [rank01(A[k][idxA[p]]) if how == "rank"
                                            else (logit(A[k][idxA[p]]) if how == "logit"
                                                  else A[k][idxA[p]]) for p in partner])
                    for k in ["ho", "pre", "post"]}
            w = np.ones(1 + len(partner)); w /= w.sum()
            add("A_base", f"seedens+{'+'.join(partner)}_{how}_2stage",
                "seed10+" + "+".join(partner),
                wauc(yho, w @ mats["ho"], Who), wauc(ypre, w @ mats["pre"], wpre),
                wauc(ypost, w @ mats["post"], wpost), "2단결합")

# ================= [B] oof 계열 =================
LO = np.load(os.path.join(CO, "_labels.npz"))
ym, Wm = LO["ym"], LO["Wm"]
ALL_B = ["lgbm_s42", "lgbm_reg", "lgbm_shallow", "lgbm_deep", "lgbm_goss",
         "rf", "et", "histgb", "enet", "logit_plain", "gnb"]
nmB, B = load_cache(CO, ALL_B, ["meta", "ho", "pre", "post"])
idxB = {n: i for i, n in enumerate(nmB)}
print(f"\n[B] 사용가능 기저모델 {len(nmB)}개: {nmB}\n")


def evalB(method, members, how, w=None, note=""):
    ii = [idxB[m] for m in members if m in idxB]
    if len(ii) == 0:
        return
    add("B_oof", method, "+".join([nmB[i] for i in ii]),
        wauc(yho, comb(B["ho"][ii], how, w), Who),
        wauc(ypre, comb(B["pre"][ii], how, w), wpre),
        wauc(ypost, comb(B["post"][ii], how, w), wpost), note)


if len(nmB) > 0:
    # B0. 단일(base_tr fit) 기준
    if "lgbm_s42" in idxB:
        i = idxB["lgbm_s42"]
        add("B_oof", "single_lgbm_s42(base_tr80%)", "lgbm_s42",
            wauc(yho, B["ho"][i], Who), wauc(ypre, B["pre"][i], wpre),
            wauc(ypost, B["post"][i], wpost), "참조")

    # B1. 균등 결합
    for how in ["prob", "rank", "logit"]:
        evalB(f"equal_all_{how}", nmB, how)

    # B2. 스태킹 — 메타러너는 meta(KCHS 내부)에서만 학습
    Zm = logit(B["meta"]).T
    for pen, C in [("l2", 1.0), ("l2", 0.01)]:
        lr = LogisticRegression(penalty=pen, C=C, max_iter=1000)
        lr.fit(Zm, ym, sample_weight=Wm)
        cf = lr.coef_.ravel()
        pr = {k: lr.predict_proba(logit(B[k]).T)[:, 1] for k in ["ho", "pre", "post"]}
        add("B_oof", f"stack_LR_logit_C{C}", "+".join(nmB),
            wauc(yho, pr["ho"], Who), wauc(ypre, pr["pre"], wpre),
            wauc(ypost, pr["post"], wpost),
            "메타러너=meta(KCHS)만; coef=" + ",".join(f"{c:.2f}" for c in cf))

    # B3. 비음수 최소제곱 가중(NNLS) — meta에서만
    Wsq = np.sqrt(Wm)
    Am_ = (B["meta"].T * Wsq[:, None])
    bm_ = ym * Wsq
    wnn, _ = nnls(Am_, bm_)
    if wnn.sum() > 0:
        for how in ["prob", "rank", "logit"]:
            evalB(f"nnls_w_{how}", nmB, how, w=wnn,
                  note="w=" + ",".join(f"{v:.3f}" for v in wnn / wnn.sum()))

    # B4. 힐클라이밍(Caruana) 앙상블 선택 — meta 가중AUROC 최대화
    def hillclimb(how, iters=30):
        M = len(nmB)
        Zm_ = np.vstack([rank01(B["meta"][i]) if how == "rank" else
                         (logit(B["meta"][i]) if how == "logit" else B["meta"][i])
                         for i in range(M)])
        cnt = np.zeros(M)
        cur = np.zeros(len(ym))
        best_hist = []
        for t in range(iters):
            best, bi = -1, -1
            for i in range(M):
                cand = (cur * cnt.sum() + Zm_[i]) / (cnt.sum() + 1)
                a = wauc(ym, cand, Wm)
                if a > best:
                    best, bi = a, i
            cnt[bi] += 1
            cur = (cur * (cnt.sum() - 1) + Zm_[bi]) / cnt.sum()
            best_hist.append(best)
        return cnt, best_hist[-1]

    for how in ["prob", "rank", "logit"]:
        cnt, mauc = hillclimb(how)
        sel = "+".join(f"{nmB[i]}x{int(cnt[i])}" for i in range(len(nmB)) if cnt[i] > 0)
        add("B_oof", f"hillclimb_{how}", sel,
            wauc(yho, comb(B["ho"], how, cnt), Who),
            wauc(ypre, comb(B["pre"], how, cnt), wpre),
            wauc(ypost, comb(B["post"], how, cnt), wpost),
            f"meta선택 가중AUROC={mauc:.4f}")

    # ===== [D] 보정 후 평균 (순위 변화 O) =====
    def calib_all(kind):
        out = {k: [] for k in ["ho", "pre", "post"]}
        for i in range(len(nmB)):
            if kind == "platt":
                lr = LogisticRegression(max_iter=1000)
                lr.fit(logit(B["meta"][i]).reshape(-1, 1), ym, sample_weight=Wm)
                f = lambda v, lr=lr: lr.predict_proba(logit(v).reshape(-1, 1))[:, 1]
            else:
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
                iso.fit(B["meta"][i], ym, sample_weight=Wm)
                f = lambda v, iso=iso: iso.predict(v)
            for k in out:
                out[k].append(f(B[k][i]))
        return {k: np.vstack(v) for k, v in out.items()}

    for kind in ["platt", "isotonic"]:
        Cc = calib_all(kind)
        for how in ["prob", "logit"]:
            add("D_calib_then_avg", f"{kind}_then_{how}avg", "+".join(nmB),
                wauc(yho, comb(Cc["ho"], how), Who),
                wauc(ypre, comb(Cc["pre"], how), wpre),
                wauc(ypost, comb(Cc["post"], how), wpost),
                "각 모델 meta에서 보정 후 평균 → 순위 변함")

    # ===== [E] 단일모델 보정의 AUROC 불변성 실증 =====
    i = idxB.get("lgbm_s42", 0)
    base_auc = (wauc(yho, B["ho"][i], Who), wauc(ypre, B["pre"][i], wpre),
                wauc(ypost, B["post"][i], wpost))
    add("E_calib_invariance", "single_uncalibrated", nmB[i], *base_auc, "기준")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(logit(B["meta"][i]).reshape(-1, 1), ym, sample_weight=Wm)
    pl = {k: lr.predict_proba(logit(B[k][i]).reshape(-1, 1))[:, 1] for k in ["ho", "pre", "post"]}
    add("E_calib_invariance", "single_platt", nmB[i],
        wauc(yho, pl["ho"], Who), wauc(ypre, pl["pre"], wpre), wauc(ypost, pl["post"], wpost),
        "Platt=단조 → AUROC 불변 예상")
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(B["meta"][i], ym, sample_weight=Wm)
    isop = {k: iso.predict(B[k][i]) for k in ["ho", "pre", "post"]}
    add("E_calib_invariance", "single_isotonic", nmB[i],
        wauc(yho, isop["ho"], Who), wauc(ypre, isop["pre"], wpre),
        wauc(ypost, isop["post"], wpost),
        "Isotonic=단조(계단) → 동점발생분만 미세변동")

df = pd.DataFrame(ROWS)
out = os.path.join(RES, "pathC_ensembles.csv")
df.to_csv(out, index=False, encoding="utf-8-sig")
print("\n저장:", out, df.shape)
print(df.sort_values("ext_pre", ascending=False).head(15).to_string())
