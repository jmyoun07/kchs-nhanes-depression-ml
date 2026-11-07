# -*- coding: utf-8 -*-
"""
pathH_03_experts.py — 기저모델 라이브러리 [전문가(부분표본) 다양성].

  (1) 연도별 전문가 : 2017..2025 각 연도에서 학습한 모델들의 평균 (시간적 다양성)
  (2) 시기 전문가   : 2017-19 / 2020-21 / 2022-25
  (3) 설계군집 전문가: strata 를 K개 그룹으로 분할해 각 그룹에서 학습 (군집/지역 다양성)
      - blk : strata 코드 연속 블록(코드 정렬이 행정구역 순서라는 가정 → '지역' 대리, 미검증)
      - rnd : strata 무작위 분할(순수한 disjoint 부분표본 전문가, 가정 없음)
  (4) 부트스트랩 배깅 전문가(bag) : 참고용

결합: 순위평균(rank) 기본. prob/logit 도 일부 비교.
출력: results/pathH_03_experts.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from pathH_common import *
import lightgbm as lgb

ROWS = []
OUT = os.path.join(RES, "pathH_03_experts.csv")
# 전문가는 작은 부분표본 다수 → 스레드 과청약(oversubscription) 방지 위해 n_jobs 제한
_LP = dict(LGB_BASE); _LP["n_jobs"] = 4
L = lambda **kw: (lambda: lgb.LGBMClassifier(**{**_LP, **kw}))
d = load()


def go(name, groups, combine="rank", note="", est=None, min_n=2000):
    try:
        r = run_group_experts(name, est or L(), groups, combine=combine,
                              note=note, min_n=min_n)
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}", flush=True)
        return
    ROWS.append(r)
    pd.DataFrame(ROWS).to_csv(OUT, index=False, encoding="utf-8-sig")


# (1) 연도별 전문가
yr = d.year_tr
go("exp_year_rank", yr, "rank", "연도별 전문가 9개, 순위평균")
go("exp_year_logit", yr, "logit", "연도별 전문가 9개, 로짓평균")

# (2) 시기 전문가
per = np.where(yr <= 2019, "p1", np.where(yr <= 2021, "p2", "p3"))
go("exp_period_rank", per, "rank", "시기 3구간 전문가, 순위평균")

# (3) 설계군집 전문가
s = d.strata_tr.astype(int)
for K in (10,):
    blk = (s * K // (s.max() + 1)).astype(str)
    go(f"exp_strataBlk{K}_rank", blk, "rank",
       f"strata 코드 연속블록 {K}개 전문가(‘지역’ 대리 가정, 미검증)")
rng = np.random.RandomState(SEED)
uq = np.unique(s)
for K in (10,):
    lab = dict(zip(uq, rng.randint(0, K, size=len(uq))))
    grp = np.array([lab[v] for v in s]).astype(str)
    go(f"exp_strataRnd{K}_rank", grp, "rank",
       f"strata 무작위 분할 {K}개 전문가(disjoint 부분표본)")

# (4) 연령대 전문가 (표현 다양성; 예측변수 age 는 그대로 유지)
ag = pd.cut(d.Xtr["age"], [0, 39, 54, 64, 74, 200], labels=list("abcde")).astype(str).values
go("exp_agegrp_rank", ag, "rank", "연령대 5구간 전문가, 순위평균")

pd.DataFrame(ROWS).to_csv(OUT, index=False, encoding="utf-8-sig")
print("SAVED", OUT)
