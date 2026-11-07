# -*- coding: utf-8 -*-
"""
pathH_05_final.py — 경로4 최종 정리 + 유의성(부트스트랩) 검정.

1) 전체 캐시(기저모델 라이브러리)에서 다양성 앙상블 후보 재평가.
2) '정직한 선택': KCHS 내부 OOF 가중AUROC 가 최대인 앙상블을 protocol_compliant 후보로 확정.
3) 후보 vs 기준선의 외부(pre/post) 개선이 표본잡음 이내인지 복합표본 근사 부트스트랩 CI 로 판정.
   - NHANES 재표집은 PSU(군집) 단위로 수행(복합표본 근사). W 는 가중AUROC 에 반영.
출력: results/pathH_05_final.csv, results/pathH_05_bootstrap.csv
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from pathH_common import *

d = load()
names = list_cached()
P = {n: load_preds(n) for n in names}
NAMES = list(names)
print(f"cached: {len(NAMES)}")

RK = {k: {n: rank01(P[n][k]) for n in NAMES} for k in ("oof", "ho", "pre", "post")}
LG = {k: {n: logit(P[n][k]) for n in NAMES} for k in ("oof", "ho", "pre", "post")}
BASE_OOF = {n: wauc(d.ytr, P[n]["oof"], d.Wtr) for n in NAMES}


def comb(sub, how, w, key):
    w = np.asarray(w, float); w = w / w.sum()
    if how == "prob":
        Z = np.vstack([P[n][key] for n in sub])
    elif how == "rank":
        Z = np.vstack([RK[key][n] for n in sub])
    else:
        Z = np.vstack([LG[key][n] for n in sub])
    return w @ Z


# ---- 후보 앙상블 정의(모두 KCHS만으로 결정) ----
ALGDIV = [n for n in ("lgbm_base", "xgb_base", "cat_base", "histgb", "rf", "et",
                      "logit_l2", "logit_spline", "mlp32") if n in P]
GBDT = [n for n in NAMES if n.startswith(("lgbm_", "xgb_", "cat_")) or n == "histgb"]

CANDS = {
    "baseline": (["lgbm_base"], "prob", None),
    "algdiv_logit": (ALGDIV, "logit", None),
    "algdiv_rank": (ALGDIV, "rank", None),
    "gbdt_rank": (GBDT, "rank", None),
    "all_rank": (NAMES, "rank", None),
}


def greedy(sub, how, n_iter=30):
    Z = np.vstack([(RK["oof"][n] if how == "rank" else LG["oof"][n] if how == "logit"
                    else P[n]["oof"]) for n in sub])
    cnt = np.zeros(len(sub))
    sc = np.array([wauc(d.ytr, Z[i], d.Wtr) for i in range(len(sub))])
    cnt[int(np.argmax(sc))] += 1
    cur = (cnt @ Z) / cnt.sum(); best = wauc(d.ytr, cur, d.Wtr)
    for _ in range(n_iter):
        cb, ci, tot = best, -1, cnt.sum()
        for i in range(len(sub)):
            s = wauc(d.ytr, (cur * tot + Z[i]) / (tot + 1), d.Wtr)
            if s > cb + 1e-9:
                cb, ci = s, i
        if ci < 0:
            break
        cnt[ci] += 1; cur = (cnt @ Z) / cnt.sum(); best = cb
    return cnt


for how in ("rank", "logit"):
    cnt = greedy(NAMES, how)
    sel = [n for n, c in zip(NAMES, cnt) if c > 0]
    CANDS[f"greedy_{how}"] = (sel, how, np.array([c for c in cnt if c > 0]))

rows = []
for tag, (sub, how, w) in CANDS.items():
    w = np.ones(len(sub)) if w is None else w
    rows.append(dict(
        name=tag, how=how, n=len(sub),
        oof=wauc(d.ytr, comb(sub, how, w, "oof"), d.Wtr),
        internal=wauc(d.yho, comb(sub, how, w, "ho"), d.Who),
        ext_pre=wauc(d.ypre, comb(sub, how, w, "pre"), d.wpre),
        ext_post=wauc(d.ypost, comb(sub, how, w, "post"), d.wpost),
        members=";".join(sub) if len(sub) <= 12 else f"{len(sub)} models"))
fin = pd.DataFrame(rows)
fin.to_csv(os.path.join(RES, "pathH_05_final.csv"), index=False, encoding="utf-8-sig")
print(fin[["name", "how", "n", "oof", "internal", "ext_pre", "ext_post"]].to_string(index=False))

# ---- 정직한 선택: OOF 최대 (기준선 제외) ----
cand = fin[fin.name != "baseline"].sort_values("oof", ascending=False).iloc[0]
print(f"\n[정직한 선택 = OOF 최대] {cand['name']} "
      f"OOF={cand.oof:.4f} pre={cand.ext_pre:.4f} post={cand.ext_post:.4f}")

# ---- 부트스트랩: 선택 앙상블 vs 기준선, PSU 군집 재표집 ----
def boot_delta(sub_a, how_a, w_a, key, n_boot=2000, seed=1):
    """key in {pre,post}. Δ가중AUROC(선택 - 기준선)의 부트스트랩 분포. PSU 군집 재표집."""
    y = (d.ypre if key == "pre" else d.ypost)
    ww = (d.wpre if key == "pre" else d.wpost)
    pa = comb(sub_a, how_a, w_a, key)
    pb = P["lgbm_base"][key]
    # PSU 라벨
    src = pd.read_parquet(os.path.join(PROC, f"ext_{key}.parquet"), columns=["psu", "strata"])
    psu = (src["strata"].astype(str) + "_" + src["psu"].astype(str)).values
    uq = np.unique(psu)
    idx_by = {u: np.where(psu == u)[0] for u in uq}
    rng = np.random.RandomState(seed)
    base_a = wauc(y, pa, ww); base_b = wauc(y, pb, ww)
    dl = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(uq, size=len(uq), replace=True)
        ii = np.concatenate([idx_by[u] for u in pick])
        ya, wwa = y[ii], ww[ii]
        if ya.min() == ya.max():
            dl[b] = np.nan; continue
        dl[b] = wauc(ya, pa[ii], wwa) - wauc(ya, pb[ii], wwa)
    dl = dl[~np.isnan(dl)]
    return dict(key=key, cand=cand["name"], base_pre_or_post=base_b, cand_auc=base_a,
                delta=base_a - base_b, lo=np.percentile(dl, 2.5),
                hi=np.percentile(dl, 97.5), p_gt0=float((dl > 0).mean()))


sub_a, how_a, w_a = CANDS[cand["name"]]
w_a = np.ones(len(sub_a)) if w_a is None else w_a
brows = []
for key in ("pre", "post"):
    r = boot_delta(sub_a, how_a, w_a, key, n_boot=2000, seed=7)
    brows.append(r)
    print(f"[boot {key}] Δ={r['delta']:+.4f}  95%CI=[{r['lo']:+.4f},{r['hi']:+.4f}]  "
          f"P(Δ>0)={r['p_gt0']:.3f}", flush=True)
pd.DataFrame(brows).to_csv(os.path.join(RES, "pathH_05_bootstrap.csv"),
                           index=False, encoding="utf-8-sig")
print("SAVED results/pathH_05_final.csv, results/pathH_05_bootstrap.csv")
