# -*- coding: utf-8 -*-
"""
path3_01_sample.py — 경로3 블록 N/S/Y/W.

N) 노이즈 바닥   : 동일 사양에서 학습표본 draw seed·모델 seed만 변경 → 해석 밴드
S) 표본 크기     : 50k~전체 POOL(1,948,510)
Y) 연도 서브셋   : 전체 vs 최근3년 vs 팬데믹이후 vs 팬데믹이전 (n=500k로 크기 통제)
W) 연도/설계 가중: 연도 균등 vs 인구비례(W) vs 최신연도 가중

내부 홀드아웃 ho(100k)는 전 실험 고정. NHANES는 predict만(§2.10).
출력: results/path3_01_sample.csv
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from path3_common import Bench, Recorder, RES, SEED

B = Bench()
R = Recorder(os.path.join(RES, "path3_01_sample.csv"))
R.log(f"tr={len(B.tr):,} ho={len(B.ho):,} pool={len(B.pool):,}")

year_pool = B.year[B.pool]
N500 = 500000


def draw(eligible, n=N500, seed=SEED):
    """eligible(전역 인덱스 배열)에서 최대 n개 무작위 추출."""
    rng = np.random.RandomState(seed)
    if len(eligible) <= n:
        return np.array(eligible)
    return rng.choice(eligible, size=n, replace=False)


# ============================================================
# 0) 기준선
# ============================================================
R.log("\n########## 0) 기준선 (과제 지정 코드) ##########")
base, _, _ = B.run("BASE_spec_500k", B.tr, note="train_test_split(500k/100k, seed42)")
R.base = base
R.add(base, "0_BASE", True)

# ============================================================
# N) 노이즈 바닥
# ============================================================
R.log("\n########## N) 노이즈 바닥 (해석 밴드 산출) ##########")
noise = []
for s in [42, 7, 99, 2024, 1234]:
    idx = draw(B.pool, N500, seed=s)
    r, _, _ = B.run(f"N_draw500k_s{s}", idx, note="POOL에서 500k 무작위 draw(seed만 변경)")
    R.add(r, "N_noise", True)
    noise.append(r)
for s in [7, 2024, 777]:
    r, _, _ = B.run(f"N_modelseed{s}", B.tr, random_state=s, note="LGBM random_state만 변경")
    R.add(r, "N_noise", True)
    noise.append(r)
nz = pd.DataFrame(noise)
R.log(f"  >>> 노이즈 밴드: int sd={nz['internal'].std():.4f} rng={nz['internal'].max()-nz['internal'].min():.4f}"
      f" | pre sd={nz['ext_pre'].std():.4f} rng={nz['ext_pre'].max()-nz['ext_pre'].min():.4f}"
      f" | post sd={nz['ext_post'].std():.4f} rng={nz['ext_post'].max()-nz['ext_post'].min():.4f}")
REF500 = nz.iloc[0].to_dict()   # seed42 POOL draw = 서브셋 실험의 크기통제 기준

# ============================================================
# S) 학습 표본 크기
# ============================================================
R.log("\n########## S) 학습 표본 크기 (ho 고정) ##########")
for n in [50000, 100000, 200000, 500000, 1000000, 1500000, len(B.pool)]:
    tag = f"S_n{n//1000}k" if n < len(B.pool) else f"S_nFULL{len(B.pool)//1000}k"
    idx = draw(B.pool, n, seed=SEED)
    r, _, _ = B.run(tag, idx, note="POOL 무작위, ho 고정")
    R.add(r, "S_size", True)

# 크기 효과의 seed 안정성 (전체 POOL은 draw 무작위성 없음 → 모델 seed 3개)
R.log("\n--- 전체 POOL 학습의 모델 seed 안정성 ---")
for s in [7, 2024]:
    r, _, _ = B.run(f"S_nFULL_mseed{s}", B.pool, random_state=s, note="전체 POOL, 모델 seed 변경")
    R.add(r, "S_size", True)

# ============================================================
# Y) 연도 서브셋 (n=500k 크기 통제)
# ============================================================
R.log("\n########## Y) 연도 서브셋 학습 (n=500k 통제) ##########")
YSETS = [
    ("Y_all",                 list(range(2017, 2026))),
    ("Y_pre_2017_2019",       [2017, 2018, 2019]),
    ("Y_prepandemic_17_20",   [2017, 2018, 2019, 2020]),
    ("Y_pandemic_20_22",      [2020, 2021, 2022]),
    ("Y_postpandemic_21_25",  [2021, 2022, 2023, 2024, 2025]),
    ("Y_recent3_23_25",       [2023, 2024, 2025]),
    ("Y_latest2_24_25",       [2024, 2025]),
    ("Y_latest1_2025",        [2025]),
    ("Y_oldest1_2017",        [2017]),
]
for tag, yrs in YSETS:
    elig = B.pool[np.isin(year_pool, yrs)]
    idx = draw(elig, N500, seed=SEED)
    r, _, _ = B.run(tag, idx, note=f"학습연도={yrs[0]}-{yrs[-1]} (n<=500k)")
    R.add(r, "Y_year", True, base=REF500, extra="델타는 REF(POOL 500k seed42) 대비")

# 연도별 개별 학습(전이 곡선) — 각 연도 전량(~226k)
R.log("\n--- 단일 연도 학습(각 ~200k) ---")
for yv in range(2017, 2026):
    elig = B.pool[year_pool == yv]
    r, _, _ = B.run(f"Y_single_{yv}", elig, note=f"{yv}년 단독 학습")
    R.add(r, "Y_single", True)

# 크기통제 대조: 단일연도와 동일 n의 전연도 무작위
R.log("\n--- 단일연도 대조(동일 n 전연도 무작위) ---")
n1 = int((year_pool == 2025).sum())
r, _, _ = B.run(f"Yctrl_random_n{n1//1000}k", draw(B.pool, n1, seed=SEED),
                note="단일연도와 동일 n, 전연도 무작위")
R.add(r, "Y_single", True)

# ============================================================
# W) 연도 균등 vs 인구비례 가중
# ============================================================
R.log("\n########## W) 연도/설계 가중 ##########")
tr = B.tr
yr_tr = B.year[tr]
w_tr = B.W[tr]

# (a) 인구비례: 복합표본 가중치 W 그대로
sw = w_tr / w_tr.mean()
r, _, _ = B.run("W_survey_W", tr, sw=sw, note="sample_weight=W(인구비례)")
R.add(r, "W_weight", True)

# (b) 연도 균등: 연도별 총가중 동일 (설계가중 무시)
sw = np.ones(len(tr))
for yv in np.unique(yr_tr):
    m = yr_tr == yv
    sw[m] = len(tr) / (m.sum() * len(np.unique(yr_tr)))
r, _, _ = B.run("W_year_equal", tr, sw=sw, note="연도별 총 가중 균등")
R.add(r, "W_weight", True)

# (c) W를 연도 내에서 정규화 후 연도 균등 (설계가중 + 연도균등)
sw = w_tr.copy()
for yv in np.unique(yr_tr):
    m = yr_tr == yv
    sw[m] = sw[m] / sw[m].mean() / len(np.unique(yr_tr)) * len(np.unique(yr_tr))
sw = sw / sw.mean()
r, _, _ = B.run("W_survey_yearnorm", tr, sw=sw, note="연도내 W 정규화(연도 총가중 균등)")
R.add(r, "W_weight", True)

# (d) 최신연도 가중 (지수 recency)
for tau, tag in [(2.0, "W_recency_tau2"), (4.0, "W_recency_tau4"), (8.0, "W_recency_tau8")]:
    sw = np.exp((yr_tr - 2025) / tau)
    sw = sw / sw.mean()
    r, _, _ = B.run(tag, tr, sw=sw, note=f"exp((year-2025)/{tau}) 최신연도 가중")
    R.add(r, "W_weight", True)

# (e) 선형 recency
sw = (yr_tr - 2016).astype(float)
sw = sw / sw.mean()
r, _, _ = B.run("W_recency_linear", tr, sw=sw, note="선형 최신연도 가중")
R.add(r, "W_weight", True)

R.log(f"\n완료. 저장: {R.out}")
