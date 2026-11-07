# -*- coding: utf-8 -*-
"""
path3_04_combo.py — 경로3 유망 방향 결합 및 내부기준 최종 선택.

블록1/2에서 방향성이 (약하게라도) 양(+)이었던 설계요소를 결합:
  - 표본크기: 전체 POOL(내부 최고) vs 500k
  - 설계가중: survey W (pre에 소폭+)
  - 불균형: scale_pos_weight 3 (post에 소폭+)
  - 연도: postpandemic (post에 소폭+)

★모델 선택은 KCHS 내부 홀드아웃(int)로만. 외부는 기록만(§2.10).
  → '내부기준 최적 설계'가 외부에서 실제로 무엇을 주는지 정직하게 확인.
출력: results/path3_04_combo.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from path3_common import Bench, Recorder, RES, SEED

B = Bench()
R = Recorder(os.path.join(RES, "path3_04_combo.csv"))
year_pool = B.year[B.pool]

base, _, _ = B.run("BASE_spec_500k", B.tr, note="과제 기준선")
R.base = base
R.add(base, "0_BASE", True)

FULL = B.pool
rng = np.random.RandomState(SEED)
tr500 = rng.choice(B.pool, 500000, replace=False)


def survey_sw(idx):
    w = B.W[idx]
    return w / w.mean()


# --- 전체 POOL 단독 (내부 최고 후보) ---
r, _, _ = B.run("C_full", FULL, note="전체 POOL")
R.add(r, "C_combo", True)

# --- 전체 POOL + survey W ---
r, _, _ = B.run("C_full_surveyW", FULL, sw=survey_sw(FULL), note="전체 POOL + 설계가중 W")
R.add(r, "C_combo", True)

# --- 전체 POOL + spw3 ---
r, _, _ = B.run("C_full_spw3", FULL, scale_pos_weight=3.0, note="전체 POOL + spw3")
R.add(r, "C_combo", True)

# --- 전체 POOL + survey W + spw3 ---
r, _, _ = B.run("C_full_surveyW_spw3", FULL, sw=survey_sw(FULL), scale_pos_weight=3.0,
                note="전체 POOL + 설계가중 W + spw3")
R.add(r, "C_combo", True)

# --- 전체 POOL + spw3 + 모델seed 앙상블(3seed 확률평균) ---
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from path3_common import make_pre, FEATURES, wauc
import time
t0 = time.time()
pis, pps, pos = [], [], []
for s in [42, 7, 2024]:
    p = Pipeline([("pre", make_pre()),
                  ("clf", lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03,
                      num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                      random_state=s, n_jobs=-1, verbose=-1, scale_pos_weight=3.0))])
    p.fit(B.X.iloc[FULL], B.y[FULL])
    pis.append(p.predict_proba(B.Xho)[:, 1])
    pps.append(p.predict_proba(B.Xpre)[:, 1])
    pos.append(p.predict_proba(B.Xpost)[:, 1])
pi = np.mean(pis, 0); pp = np.mean(pps, 0); po = np.mean(pos, 0)
r = dict(name="C_full_spw3_seedens3", train_n=len(FULL),
         internal=wauc(B.yho, pi, B.who), ext_pre=wauc(B.ypre, pp, B.wpre),
         ext_post=wauc(B.ypost, po, B.wpost), secs=round(time.time() - t0, 1),
         note="전체 POOL + spw3 + 3seed 확률평균 앙상블")
R.add(r, "C_combo", True)

# --- postpandemic 전량 + spw3 ---
elig = B.pool[np.isin(year_pool, [2021, 2022, 2023, 2024, 2025])]
r, _, _ = B.run("C_postpandemic_all_spw3", elig, scale_pos_weight=3.0,
                note=f"팬데믹이후 전량({len(elig):,}) + spw3")
R.add(r, "C_combo", True)

# --- 내부기준 요약 ---
R.log("\n########## 내부기준 선택 시뮬레이션 ##########")
d = pd.DataFrame(R.rows)
best = d.loc[d["internal"].idxmax()]
R.log(f"  내부 최고 설계 = {best['name']} : int={best['internal']:.4f} "
      f"pre={best['ext_pre']:.4f} post={best['ext_post']:.4f}")
R.log(f"  (기준선 대비 외부 Δpre={best['ext_pre']-base['ext_pre']:+.4f} "
      f"Δpost={best['ext_post']-base['ext_post']:+.4f})")
R.log(f"\n완료: {R.out}")
