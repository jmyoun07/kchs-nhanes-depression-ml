# -*- coding: utf-8 -*-
"""
pathD2_03_grid.py — 경로D 주 실험 그리드.

각 설정에 대해:
  (a) LOYO CV (연도 9폴드)      : mean / min(최악연도) / std
  (b) 지역 GroupKFold CV (5폴드): mean / min(최악지역) / std
  (c) 최종 적합(전체 train pool) 후 내부 홀드아웃 + NHANES pre/post  <- **기록 전용**

선택 규칙은 사전 선언(§2.10):
  R1 = LOYO mean 최대
  R2 = LOYO min(최악 연도) 최대            <- 분포이동 강건성 대리지표
  R3 = strata min(최악 지역) 최대
  R4 = mean - 1*std (LOYO)  (분산 페널티, V-REx 취지)
외부(pre/post)는 어떤 규칙에도 들어가지 않는다.
"""
import os
import sys
import time
import json

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathD2_engine import cv_eval, final_eval, data, RES, SEED, line

OUT = os.path.join(RES, "pathD2_grid.csv")
TRAIN_N_CV = 250000          # 폴드별 학습 표본(속도). 모든 설정에 동일 적용.

# ---- 안정성 분석(pathD2_02) 결과에서 도출한 변수군 ----
ALL17 = ["age", "sleep_hrs", "bmi", "income_q", "srh", "oral_health",
         "partnered", "living_alone", "current_drinker", "active", "htn", "dm",
         "employed", "food_insecure", "sex", "edu", "smoking"]
STABLE9 = ["srh", "oral_health", "partnered", "smoking", "employed", "sex",
           "income_q", "age", "sleep_hrs"]
STABLE7 = ["srh", "oral_health", "partnered", "smoking", "employed", "sex", "age"]
DROP_UNSTABLE = [f for f in ALL17 if f not in ("bmi", "dm", "living_alone", "htn")]
DROP_BMI = [f for f in ALL17 if f != "bmi"]

MONO_CORE = dict(srh=1, oral_health=1, income_q=-1, employed=-1,
                 partnered=-1, food_insecure=1)
MONO_EXT = dict(MONO_CORE, age=-1, sleep_hrs=-1)

G = []


def add(name, spec, note=""):
    s = dict(spec)
    s.setdefault("family", "lgb")
    s["train_n"] = TRAIN_N_CV
    G.append((name, s, note))


# ---------- A. 복잡도/정규화 ----------
add("A0_base", dict(), "기준선 하이퍼파라미터")
add("A1_leaves7", dict(params=dict(num_leaves=7)))
add("A2_leaves15", dict(params=dict(num_leaves=15)))
add("A3_leaves63", dict(params=dict(num_leaves=63)))
add("A4_mcs2000", dict(params=dict(min_child_samples=2000)))
add("A5_mcs10000", dict(params=dict(min_child_samples=10000)))
add("A6_lambda100", dict(params=dict(reg_lambda=100.0)))
add("A7_lambda1000", dict(params=dict(reg_lambda=1000.0)))
add("A8_depth3", dict(params=dict(max_depth=3, num_leaves=8)))
add("A9_depth2", dict(params=dict(max_depth=2, num_leaves=4)))
add("A10_ne200", dict(params=dict(n_estimators=200)))
add("A11_bagging", dict(params=dict(subsample_freq=1, colsample_bytree=0.5)))
add("A12_heavyreg", dict(params=dict(num_leaves=15, min_child_samples=2000,
                                     reg_lambda=100.0, colsample_bytree=0.5,
                                     subsample_freq=1)),
    "복합 강정규화")

# ---------- B. 가법/단조 (구조적 단순화) ----------
add("B0_stumps", dict(params=dict(max_depth=1, num_leaves=2, n_estimators=1200)),
    "순수 가법(그루터기) 부스팅")
add("B1_mono_core", dict(mono=MONO_CORE))
add("B2_mono_ext", dict(mono=MONO_EXT))
add("B3_mono_core_depth3", dict(mono=MONO_CORE, params=dict(max_depth=3, num_leaves=8)))
add("B4_logit", dict(family="logit"))
add("B5_logit_C001", dict(family="logit", params=dict(C=0.01)))

# ---------- C. 특징 안정성 선택 ----------
add("C1_stable9", dict(keep=STABLE9), "연도간 stability_t 상위")
add("C2_stable7", dict(keep=STABLE7))
add("C3_drop_unstable4", dict(keep=DROP_UNSTABLE), "bmi/dm/living_alone/htn 제거")
add("C4_drop_bmi", dict(keep=DROP_BMI))
add("C5_stable9_heavyreg", dict(keep=STABLE9,
                                params=dict(num_leaves=15, min_child_samples=2000,
                                            reg_lambda=100.0)))

# ---------- D. DRO / 그룹 최악성능 ----------
add("D1_dro_year_e1", dict(sw="dro", dro_groups="year", dro_eta=1.0, dro_rounds=2))
add("D2_dro_year_e3", dict(sw="dro", dro_groups="year", dro_eta=3.0, dro_rounds=3))
add("D3_dro_strata_e1", dict(sw="dro", dro_groups="strata", dro_eta=1.0, dro_rounds=2))
add("D4_dro_strata_e3", dict(sw="dro", dro_groups="strata", dro_eta=3.0, dro_rounds=3))
add("D5_survey_w", dict(sw="survey"), "복합표본 가중 학습")

rows = []
t0 = time.time()
for i, (name, spec, note) in enumerate(G, 1):
    t = time.time()
    try:
        cvy = cv_eval(spec, "loyo")
        cvs = cv_eval(spec, "strata")
        ext, _ = final_eval(spec, name)
    except Exception as e:
        print(f"[{i}/{len(G)}] {name} FAILED: {type(e).__name__}: {e}")
        rows.append(dict(name=name, note=f"FAILED {type(e).__name__}: {e}"))
        pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
        continue
    r = dict(name=name, note=note,
             loyo_mean=cvy["mean"], loyo_min=cvy["min"], loyo_std=cvy["std"],
             loyo_gap=cvy["gap"],
             loyo_mean_minus_std=cvy["mean"] - cvy["std"],
             strata_mean=cvs["mean"], strata_min=cvs["min"], strata_std=cvs["std"],
             internal=ext["internal"], ext_pre=ext["ext_pre"], ext_post=ext["ext_post"],
             seconds=round(time.time() - t, 1),
             loyo_per_fold=json.dumps(cvy["per_fold"]),
             strata_per_fold=json.dumps(cvs["per_fold"]))
    rows.append(r)
    print(f"[{i}/{len(G)}] " + line(name, cvy, ext) +
          f" | strata_min={cvs['min']:.4f} [{r['seconds']:.0f}s tot={time.time()-t0:.0f}s]")
    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")

print("saved:", OUT, f"total {time.time()-t0:.0f}s")
