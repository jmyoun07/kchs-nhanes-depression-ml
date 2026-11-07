# -*- coding: utf-8 -*-
"""경로1-02: 선택규칙 사전선언 및 적용 (KCHS 내부 CV 지표만 사용).

★ 이 스크립트는 NHANES를 전혀 읽지 않는다. 그리드 결과를 보기 전에 작성됨.

사전선언 규칙:
  R1 loyo_mean          최대  — 연도 평균 성능
  R2 loyo_min           최대  — 최악 연도(시간 분포이동 강건성)
  R3 reg_min            최대  — 최악 지역(공간 분포이동 강건성)
  R4 loyo_mean - loyo_std 최대 — 분산 페널티(V-REx 취지)
  R5 min(loyo_min, reg_min) 최대  ← ★주 규칙: 연도·지역 통합 최악성능
  R6 (loyo_min + reg_min)/2 최대
  R7 (loyo_mean + reg_mean)/2 최대
  R8 최악 3개 연도 평균(CVaR_1/3) 최대 — min의 표본잡음 완화판
  R9 R5 상위 5개 설정의 확률 평균 앙상블
동률 시 idx(그리드 등록순) 작은 쪽.
"""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_common import RES
import numpy as np
import pandas as pd


def merge():
    fs = sorted(set(glob.glob(os.path.join(RES, "p1_grid_s*.csv")) +
                    glob.glob(os.path.join(RES, "p1_grid?_s*.csv"))))
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    d = d[d["loyo_mean"].notna()].copy()
    d = d.drop_duplicates(subset="name", keep="first")
    d = d.sort_values("idx").reset_index(drop=True)
    ly = d["loyo_folds"].apply(lambda s: np.array(list(json.loads(s).values()), float))
    d["cvar3"] = ly.apply(lambda v: float(np.sort(v)[:3].mean()))
    d["loyo_range"] = ly.apply(lambda v: float(v.max() - v.min()))
    return d


RULES = {
    "R1_loyo_mean":        lambda d: d["loyo_mean"],
    "R2_loyo_min":         lambda d: d["loyo_min"],
    "R3_reg_min":          lambda d: d["reg_min"],
    "R4_loyo_mean_m_std":  lambda d: d["loyo_mean_m_std"],
    "R5_worst_all":        lambda d: d["worst_all"],
    "R6_avg_mins":         lambda d: (d["loyo_min"] + d["reg_min"]) / 2,
    "R7_avg_means":        lambda d: (d["loyo_mean"] + d["reg_mean"]) / 2,
    "R8_cvar3_years":      lambda d: d["cvar3"],
}


def picks(d):
    out = {}
    for r, f in RULES.items():
        s = f(d)
        i = int(np.lexsort((d["idx"].values, -s.values))[0])
        out[r] = d.iloc[i]["name"]
    out["R9_ens_top5_worst_all"] = ",".join(
        d.sort_values(["worst_all", "idx"], ascending=[False, True])
         .head(5)["name"].tolist())
    return out


if __name__ == "__main__":
    d = merge()
    print(f"configs: {len(d)}")
    cols = ["idx", "name", "loyo_mean", "loyo_min", "loyo_std", "cvar3",
            "reg_mean", "reg_min", "worst_all"]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    print(d[cols].round(5).to_string())
    p = picks(d)
    print("\n=== 사전선언 선택 결과 (NHANES 미사용) ===")
    for k, v in p.items():
        print(f"{k:24s} -> {v}")
    d.to_csv(os.path.join(RES, "p1_grid_merged.csv"), index=False, encoding="utf-8-sig")
    json.dump(p, open(os.path.join(RES, "p1_selection.json"), "w"),
              ensure_ascii=False, indent=1)
    print("saved", os.path.join(RES, "p1_selection.json"))
