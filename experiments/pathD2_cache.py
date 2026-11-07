# -*- coding: utf-8 -*-
"""
pathD2_cache.py — CV 탐색 가속용 설계행렬 캐시.

전처리기(median대체+표준화 / bin passthrough / 최빈값+OneHot)를 **전역 train pool(itr)에서 1회 적합**한 뒤
KCHS 전체·NHANES pre/post를 numpy float32로 변환해 재사용한다.

정당성: 전처리 적합에 쓰이는 정보는 KCHS train pool뿐이며 NHANES 라벨/분포는 일절 사용되지 않는다.
폴드별 재적합 대비 median/최빈값 차이는 무시할 수준이고, 모든 후보 설정에 동일하게 적용되므로
설정 간 비교(=모델 선택)에 편향을 주지 않는다. 최종 보고 모델은 폴드 재적합 파이프라인으로 재검증한다.
"""
import os
import numpy as np

from pathD2_common import (Bench, make_pre, FEATURES, RES, SEED)

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE, exist_ok=True)


class Cached:
    def __init__(self, bench=None):
        self.b = bench if bench is not None else Bench()
        b = self.b
        self.ct = make_pre()
        self.ct.fit(b.X.iloc[b.itr])
        self.cols = list(self.ct.get_feature_names_out())
        self.Z = self.ct.transform(b.X).astype(np.float32)
        self.Zpre = self.ct.transform(b.Xpre).astype(np.float32)
        self.Zpost = self.ct.transform(b.Xpost).astype(np.float32)
        # 원본 특징명 -> 설계행렬 열 인덱스 (변수 단위 조작용)
        self.feat_cols = {}
        for f in FEATURES:
            ix = [i for i, c in enumerate(self.cols)
                  if c.split("__", 1)[1].startswith(f)]
            self.feat_cols[f] = ix
        # 접두 충돌 정리(예: 'sleep_hrs' vs 'sleep_weekend'은 미포함이라 안전,
        # 'active' 등 단독; 'income_q'는 유일) — 정확 일치 우선
        for f in FEATURES:
            exact = [i for i, c in enumerate(self.cols)
                     if c.split("__", 1)[1] == f]
            if exact:
                self.feat_cols[f] = exact

    # 편의 접근자
    @property
    def y(self):
        return self.b.y

    @property
    def W(self):
        return self.b.W


def loyo(b, pool=None):
    pool = b.itr if pool is None else pool
    yr = b.year[pool]
    for yv in b.years:
        m = yr == yv
        yield pool[~m], pool[m], int(yv)


def strata_kfold(b, n_splits=5, seed=SEED, pool=None):
    pool = b.itr if pool is None else pool
    g = b.strata[pool]
    uq, inv = np.unique(g, return_inverse=True)
    rs = np.random.RandomState(seed)
    fold_of_group = rs.randint(0, n_splits, size=len(uq))
    fold = fold_of_group[inv]
    for f in range(n_splits):
        m = fold == f
        yield pool[~m], pool[m], f
