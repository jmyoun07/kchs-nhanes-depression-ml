# -*- coding: utf-8 -*-
"""경로1-00: 기준선 재현 + 캐시 설계행렬 등가성 확인 + 폴드 정의 진단 + 시간 측정."""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_common import *

rows = []
t0 = time.time()
k, pre, post = load()
y = k["dep"].astype(int).to_numpy()
itr, iho = base_split(len(k), y)

# ---------- (1) 기준선: 명세 그대로 (파이프라인 폴드내 적합) ----------
t = time.time()
pipe = Pipeline([("pre", make_pre()), ("clf", lgb.LGBMClassifier(**LGB_BASE))])
pipe.fit(k.iloc[itr][FEATURES], y[itr])
pi = pipe.predict_proba(k.iloc[iho][FEATURES])[:, 1]
pp = pipe.predict_proba(pre[FEATURES])[:, 1]
ps = pipe.predict_proba(post[FEATURES])[:, 1]
b = (wauc(y[iho], pi, k["W"].to_numpy(float)[iho]),
     wauc(pre["dep"], pp, pre["W"]), wauc(post["dep"], ps, post["W"]))
rec(rows, name="baseline_literal_500k_100k", internal=round(b[0], 6),
    ext_pre=round(b[1], 6), ext_post=round(b[2], 6), protocol_compliant=True,
    note=f"명세 그대로. 목표 0.8310/0.7740/0.7653. fit {time.time()-t:.0f}s")

# ---------- (2) 캐시(전체적합 전처리) 등가성 ----------
C = Cache()
t = time.time()
(pi2, pp2, ps2), _ = fit_predict({"_cols": C.cols}, C.X[itr], C.y[itr],
                                 [C.X[iho], C.Xpre, C.Xpost])
c = (wauc(C.y[iho], pi2, C.W[iho]), wauc(pre["dep"], pp2, pre["W"]),
     wauc(post["dep"], ps2, post["W"]))
rec(rows, name="cache_equiv_500k_100k", internal=round(c[0], 6),
    ext_pre=round(c[1], 6), ext_post=round(c[2], 6), protocol_compliant=True,
    note=f"전처리를 KCHS 전체에 1회 적합한 캐시 행렬. CV 스크리닝용 근사. "
         f"delta int={c[0]-b[0]:+.4f} pre={c[1]-b[1]:+.4f} post={c[2]-b[2]:+.4f}. fit {time.time()-t:.0f}s")

# ---------- (3) 폴드 정의 진단 ----------
yf = C.year_folds()
rf = C.region_folds(5)
diag = dict(year_folds={int(a): int(len(b_)) for a, b_ in yf},
            region_folds={a: int(len(b_)) for a, b_ in rf},
            region_dep={a: round(float(C.y[b_].mean()), 4) for a, b_ in rf},
            region_strata_range={a: [int(C.strata[b_].min()), int(C.strata[b_].max())]
                                 for a, b_ in rf})
print(json.dumps(diag, ensure_ascii=False, indent=1))
json.dump(diag, open(os.path.join(RES, "p1_folds.json"), "w"), ensure_ascii=False, indent=1)

# ---------- (4) 폴드 1개 시간 측정 (스케일 계획용) ----------
import numpy as np
rng = np.random.default_rng(0)
for n in (100000, 200000, 400000):
    te = yf[0][1]
    tr = np.setdiff1d(np.arange(len(C.y)), te)
    tr = rng.choice(tr, size=n, replace=False)
    t = time.time()
    (p,), _ = fit_predict({"_cols": C.cols}, C.X[tr], C.y[tr], [C.X[te]])
    print(f"timing train_n={n}: {time.time()-t:.1f}s wauc={wauc(C.y[te], p, C.W[te]):.5f}", flush=True)

dump(rows, "p1_00_baseline.csv")
print(f"[total {time.time()-t0:.0f}s]")
