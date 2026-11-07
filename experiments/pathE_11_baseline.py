# -*- coding: utf-8 -*-
"""경로E-11: 기준선 재현. 과제 명세(500k/100k)와 저장소 실측(80/20 + income_q train-only) 둘 다."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathE_10_common import *

t0 = time.time()
k, pre, post = load()
y = k["dep"].astype(int).values

# A: 과제 명세 그대로
itr, iho = base_split(k)
a = fit_eval(k, pre, post, itr, iho)
rec("baseline_A_500k100k", a[0], a[1], a[2], True,
    "과제 명세 그대로. 목표 0.8336/0.7681/0.7682와 불일치")

# B: 80/20 split, income_q 원본
itr2, iho2 = train_test_split(np.arange(len(k)), test_size=0.2, stratify=y, random_state=SEED)
b = fit_eval(k, pre, post, itr2, iho2)
rec("baseline_B_8020", b[0], b[1], b[2], True, "80/20 split")

# C: 80/20 + income_q를 train 데이터만으로 연도별 5분위 재계산
def income_q_trainonly(df, is_tr):
    iq = np.full(len(df), np.nan); yr = df["year"].values; inc = df["income_raw"].to_numpy(float)
    for yv in np.unique(yr):
        m = yr == yv; v = inc[m & is_tr]; v = v[~np.isnan(v)]
        if len(v) < 50: continue
        e = np.unique(np.quantile(v, [0, .2, .4, .6, .8, 1.0]))
        if len(e) < 3: continue
        ii = np.where(m)[0]
        bb = np.clip(np.digitize(inc[ii], e[1:-1]), 0, len(e) - 2).astype(float)
        bb[np.isnan(inc[ii])] = np.nan; iq[ii] = bb
    return iq

is_tr = np.zeros(len(k), bool); is_tr[itr2] = True
kc = k.copy(); kc["income_q"] = income_q_trainonly(kc, is_tr)
c = fit_eval(k, pre, post, itr2, iho2, kdf=kc)
rec("baseline_C_8020_incomeq_trainonly", c[0], c[1], c[2], True,
    "저장소 pathE_baseline_repro의 spec C = 목표값 재현본")

dump("pathE_11_baseline.csv")
print(f"[{time.time()-t0:.0f}s]")
