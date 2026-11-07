# -*- coding: utf-8 -*-
"""경로1-01: 하이퍼파라미터 그리드를 KCHS 내부 CV로만 평가.

  (a) LOYO CV      : 연도 9폴드 (한 해 제외 학습 → 그 해 평가)
  (b) 지역 블록 CV : strata 연속블록 5폴드 (행정순 = 지역 대리, 연령/주관건강 구배 확인됨)

각 폴드 학습표본 = 500,000 (최종 프로토콜과 동일 크기), 모든 설정에 동일한 부분표본 사용(대응비교).
평가는 해당 폴드 전체 행(가중 AUROC).

★ NHANES는 이 스크립트에서 전혀 건드리지 않는다. (§2.10)

사용: python p1_01_grid.py <shard> <nshard>
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_common import *
import numpy as np

SHARD = int(sys.argv[1]) if len(sys.argv) > 1 else 0
NSHARD = int(sys.argv[2]) if len(sys.argv) > 2 else 1
NJOBS = int(sys.argv[3]) if len(sys.argv) > 3 else 7
TRAIN_N = int(os.environ.get("P1_TRAIN_N", 250000))
EVAL_N = int(os.environ.get("P1_EVAL_N", 150000))
K_REGION = 5

G = []


def add(name, params=None, note="", **kw):
    s = dict(kw)
    s["params"] = dict(params or {})
    s["params"]["n_jobs"] = NJOBS
    G.append((name, s, note))


# ---------------- A. 기준선 및 1변수 변화 ----------------
add("A00_base", {}, "기준선 하이퍼파라미터")
add("A01_leaves4", dict(num_leaves=4))
add("A02_leaves7", dict(num_leaves=7))
add("A03_leaves15", dict(num_leaves=15))
add("A04_leaves63", dict(num_leaves=63))
add("A05_leaves127", dict(num_leaves=127))
add("A06_mcs100", dict(min_child_samples=100))
add("A07_mcs500", dict(min_child_samples=500))
add("A08_mcs2000", dict(min_child_samples=2000))
add("A09_mcs5000", dict(min_child_samples=5000))
add("A10_mcs20000", dict(min_child_samples=20000))
add("A11_lam1", dict(reg_lambda=1.0))
add("A12_lam10", dict(reg_lambda=10.0))
add("A13_lam100", dict(reg_lambda=100.0))
add("A14_lam1000", dict(reg_lambda=1000.0))
add("A15_lam5000", dict(reg_lambda=5000.0))
add("A16_alp1", dict(reg_alpha=1.0))
add("A17_alp10", dict(reg_alpha=10.0))
add("A18_alp100", dict(reg_alpha=100.0))
add("A19_depth2", dict(max_depth=2, num_leaves=4))
add("A20_depth3", dict(max_depth=3, num_leaves=8))
add("A21_depth4", dict(max_depth=4, num_leaves=16))
add("A22_depth6", dict(max_depth=6, num_leaves=64))
add("A23_lr001_ne1800", dict(learning_rate=0.01, n_estimators=1800))
add("A24_lr001_ne600", dict(learning_rate=0.01, n_estimators=600))
add("A25_lr006_ne300", dict(learning_rate=0.06, n_estimators=300))
add("A26_lr010_ne200", dict(learning_rate=0.10, n_estimators=200))
add("A27_ne150", dict(n_estimators=150))
add("A28_ne300", dict(n_estimators=300))
add("A29_ne1200", dict(n_estimators=1200))
add("A30_ne2000", dict(n_estimators=2000))
add("A31_col04", dict(colsample_bytree=0.4))
add("A32_col06", dict(colsample_bytree=0.6))
add("A33_col10", dict(colsample_bytree=1.0))
add("A34_sub05f1", dict(subsample=0.5, subsample_freq=1))
add("A35_sub08f1", dict(subsample=0.8, subsample_freq=1))
add("A36_msg001", dict(min_split_gain=0.01))
add("A37_msg01", dict(min_split_gain=0.1))
add("A38_maxbin32", dict(max_bin=32))
add("A39_maxbin63", dict(max_bin=63))
add("A40_extratrees", dict(extra_trees=True))
add("A41_pathsmooth100", dict(path_smooth=100.0))
add("A42_pathsmooth1000", dict(path_smooth=1000.0))
add("A43_mcw10", dict(min_child_weight=10.0))
add("A44_mcw100", dict(min_child_weight=100.0))
add("A45_goss", dict(boosting_type="goss", subsample=1.0))
add("A46_lineartree", dict(linear_tree=True, linear_lambda=1.0),
    "잎에 선형모형(외삽 강건 가설)")
add("A47_surveyw", {}, "복합표본 가중 학습", sw="survey")

# ---------------- B. 구조적 단순화 / 복합 정규화 ----------------
add("B00_stumps", dict(max_depth=1, num_leaves=2, n_estimators=1500,
                       learning_rate=0.05), "순수 가법(그루터기)")
add("B01_stumps2000", dict(max_depth=1, num_leaves=2, n_estimators=3000,
                           learning_rate=0.03))
add("B02_l7_mcs2000_lam100", dict(num_leaves=7, min_child_samples=2000,
                                  reg_lambda=100.0))
add("B03_l15_mcs1000_lam100_c06", dict(num_leaves=15, min_child_samples=1000,
                                       reg_lambda=100.0, colsample_bytree=0.6))
add("B04_d3_mcs5000_lam100", dict(max_depth=3, num_leaves=8,
                                  min_child_samples=5000, reg_lambda=100.0))
add("B05_l31_mcs5000_lam1000_c05_s06", dict(min_child_samples=5000,
                                            reg_lambda=1000.0, colsample_bytree=0.5,
                                            subsample=0.6, subsample_freq=1))
add("B06_l7_lr001_ne1800", dict(num_leaves=7, learning_rate=0.01, n_estimators=1800))
add("B07_l15_mcs2000_ps100", dict(num_leaves=15, min_child_samples=2000,
                                  path_smooth=100.0))
add("B08_d4_mcs2000_lam10_c06", dict(max_depth=4, num_leaves=16,
                                     min_child_samples=2000, reg_lambda=10.0,
                                     colsample_bytree=0.6))
add("B09_l63_mcs5000_lam1000", dict(num_leaves=63, min_child_samples=5000,
                                    reg_lambda=1000.0))
add("B10_extratrees_l15_mcs2000", dict(extra_trees=True, num_leaves=15,
                                       min_child_samples=2000))
add("B11_bin32_l15_mcs2000", dict(max_bin=32, num_leaves=15, min_child_samples=2000))
add("B12_d2_ne2000_lr002", dict(max_depth=2, num_leaves=4, n_estimators=2000,
                                learning_rate=0.02))
add("B13_d3_ne1500_lr002_c06", dict(max_depth=3, num_leaves=8, n_estimators=1500,
                                    learning_rate=0.02, colsample_bytree=0.6))
add("B14_l31_ne300_lam100_mcs2000", dict(n_estimators=300, reg_lambda=100.0,
                                         min_child_samples=2000))
add("B15_ultrareg", dict(num_leaves=7, min_child_samples=20000, reg_lambda=5000.0,
                         colsample_bytree=0.4, subsample=0.5, subsample_freq=1,
                         n_estimators=800, learning_rate=0.03), "초강정규화")
add("B16_l15_mcs2000_lam100_col06_sub08f1",
    dict(num_leaves=15, min_child_samples=2000, reg_lambda=100.0,
         colsample_bytree=0.6, subsample=0.8, subsample_freq=1))
add("B17_lineartree_d3", dict(linear_tree=True, linear_lambda=10.0,
                              max_depth=3, num_leaves=8))
add("B18_l7_mcs5000_lam1000_ne1200_lr002",
    dict(num_leaves=7, min_child_samples=5000, reg_lambda=1000.0,
         n_estimators=1200, learning_rate=0.02))
add("B19_d5_mcs1000_lam10", dict(max_depth=5, num_leaves=32,
                                 min_child_samples=1000, reg_lambda=10.0))

# ---------------- C. 랜덤 서치 (고정 시드) ----------------
_r = np.random.default_rng(20260724)
N_RANDOM = int(os.environ.get("P1_NRANDOM", 12))
for i in range(N_RANDOM):
    nl = int(_r.choice([4, 6, 8, 12, 16, 24, 31, 48, 63]))
    md = int(_r.choice([-1, -1, 2, 3, 4, 5, 6, 8]))
    if md > 0:
        nl = min(nl, 2 ** md)
    mcs = int(np.exp(_r.uniform(np.log(20), np.log(20000))))
    lam = float(np.exp(_r.uniform(np.log(0.05), np.log(3000)))) if _r.random() < .85 else 0.0
    alp = float(np.exp(_r.uniform(np.log(0.05), np.log(100)))) if _r.random() < .5 else 0.0
    lr = float(_r.choice([0.01, 0.02, 0.03, 0.05, 0.08]))
    ne = int(round(18.0 / lr * _r.uniform(0.5, 2.0)))
    col = float(_r.choice([0.4, 0.5, 0.6, 0.8, 1.0]))
    subs = float(_r.choice([0.5, 0.6, 0.8, 1.0]))
    sf = 0 if subs >= 1.0 else int(_r.choice([0, 1]))
    ps = float(_r.choice([0.0, 0.0, 10.0, 100.0, 1000.0]))
    add(f"C{i:02d}", dict(num_leaves=nl, max_depth=md, min_child_samples=mcs,
                          reg_lambda=lam, reg_alpha=alp, learning_rate=lr,
                          n_estimators=ne, colsample_bytree=col, subsample=subs,
                          subsample_freq=sf, path_smooth=ps), "random search")

VERBOSE = os.environ.get("P1_VERBOSE", "1") == "1"

if __name__ == "__main__":
    import glob
    C = Cache()
    TAG = os.environ.get("P1_TAG", "")
    OUT = os.path.join(RES, f"p1_grid{TAG}_s{SHARD}.csv")
    done = set()
    for f in glob.glob(os.path.join(RES, "p1_grid*_s*.csv")) + \
             glob.glob(os.path.join(RES, "p1_grid_s*.csv")):
        try:
            dd = pd.read_csv(f)
            done |= set(dd[dd["loyo_mean"].notna()]["name"])
        except Exception:
            pass
    mine = [(i, n, s, nt) for i, (n, s, nt) in enumerate(G)
            if i % NSHARD == SHARD and n not in done]
    print(f"shard {SHARD}/{NSHARD}: {len(mine)} configs (grid={len(G)}, "
          f"already done={len(done)})", flush=True)
    rows = []
    t0 = time.time()
    for j, (i, name, spec, note) in enumerate(mine, 1):
        t = time.time()
        try:
            cy = cv_run(C, spec, "year", TRAIN_N, eval_n=EVAL_N, verbose=VERBOSE)
            cr = cv_run(C, spec, "region", TRAIN_N, k_region=K_REGION,
                        eval_n=EVAL_N, verbose=VERBOSE)
        except Exception as e:
            print(f"[{j}/{len(mine)}] {name} FAILED {type(e).__name__}: {e}", flush=True)
            rows.append(dict(idx=i, name=name, note=f"FAILED {type(e).__name__}: {e}"))
            pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
            continue
        r = dict(idx=i, name=name, note=note,
                 params=json.dumps({k_: v for k_, v in spec["params"].items()
                                    if k_ != "n_jobs"}),
                 sw=spec.get("sw", ""),
                 loyo_mean=round(cy["mean"], 6), loyo_min=round(cy["min"], 6),
                 loyo_std=round(cy["std"], 6),
                 loyo_mean_m_std=round(cy["mean"] - cy["std"], 6),
                 reg_mean=round(cr["mean"], 6), reg_min=round(cr["min"], 6),
                 reg_std=round(cr["std"], 6),
                 worst_all=round(min(cy["min"], cr["min"]), 6),
                 mean_all=round((cy["mean"] + cr["mean"]) / 2, 6),
                 seconds=round(time.time() - t, 1),
                 loyo_folds=json.dumps(cy["per_fold"]),
                 reg_folds=json.dumps(cr["per_fold"]))
        rows.append(r)
        print(f"[{j}/{len(mine)}] {name} loyo m={r['loyo_mean']:.5f} min={r['loyo_min']:.5f} "
              f"sd={r['loyo_std']:.5f} | reg m={r['reg_mean']:.5f} min={r['reg_min']:.5f} "
              f"| worst={r['worst_all']:.5f} [{r['seconds']:.0f}s tot={time.time()-t0:.0f}s]",
              flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
    print("done", OUT, f"{time.time()-t0:.0f}s", flush=True)
