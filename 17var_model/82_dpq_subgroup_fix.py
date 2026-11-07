# -*- coding: utf-8 -*-
"""82_dpq_subgroup_fix.py — DPQ 미응답 하위군 비율을 표 정의(≥3문항 결측)대로 다시 계산한다.

기존 src/11_dpq_missingness.py 는 하위군 비율만 `nmiss > 0`(1문항이라도 결측)으로 계산해
표 S8 의 정의(≥3문항 = 미응답)와 어긋났다. 전체 비율(14.3/32.7)은 ≥3 기준이라 옳다.
"""
import os, sys, numpy as np, pandas as pd, pyreadstat
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:/Users/rudtn/Desktop/claude code/지역사회건강/data/raw/nhanes"
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DPQ = [f"DPQ0{i}0" for i in range(1, 10)]

def rd(folder, namer, mod, cols):
    return pyreadstat.read_xport(os.path.join(NB, folder, namer(mod)), usecols=["SEQN"] + cols, encoding="latin1")[0]

EDU = {1: "<HS", 2: "<HS", 3: "HS", 4: "College", 5: "College"}
RACE = {1: "Mex", 2: "OthHisp", 3: "White", 4: "Black", 6: "Asian", 7: "Other"}
rows = []
for period, folder, namer in [("pre", "pre_2017_2020", lambda m: f"P_{m}.xpt"),
                              ("post", "post_2021_2023", lambda m: f"{m}_L.xpt")]:
    demo = rd(folder, namer, "DEMO", ["RIDAGEYR", "DMDEDUC2", "RIDRETH3"])
    dpq = rd(folder, namer, "DPQ", DPQ)
    ad = demo.merge(dpq, on="SEQN", how="left")
    ad = ad[ad["RIDAGEYR"] >= 20].copy()
    nmiss = ad[DPQ].apply(pd.to_numeric, errors="coerce").mask(lambda x: x.isin([7, 9])).isna().sum(axis=1)
    ad["nr3"] = (nmiss >= 3).astype(float)          # 표 정의
    ad["nr_any"] = (nmiss > 0).astype(float)        # 종전 계산
    print(f"[{period}] 성인20+ n={len(ad)} · ≥3결측 {ad['nr3'].mean()*100:.1f}% · any결측 {ad['nr_any'].mean()*100:.1f}%")
    for fac, col, mp in [("education", "DMDEDUC2", EDU), ("race", "RIDRETH3", RACE)]:
        g = ad.assign(_g=ad[col].map(mp)).groupby("_g")[["nr3", "nr_any"]].agg(["mean", "size"])
        for k in g.index:
            rows.append(dict(period=period, factor=fac, group=k, n=int(g.loc[k, ("nr3", "size")]),
                             nonresponse_ge3_pct=round(float(g.loc[k, ("nr3", "mean")])*100, 1),
                             any_missing_pct=round(float(g.loc[k, ("nr_any", "mean")])*100, 1)))
            print(f"   {period:>4} {fac:>9} {k:>9}  ≥3 {rows[-1]['nonresponse_ge3_pct']:>5.1f}%   any {rows[-1]['any_missing_pct']:>5.1f}%  (n={rows[-1]['n']})")
pd.DataFrame(rows).to_csv(f"{RES}/primary2_dpq_nonresponse.csv", index=False, encoding="utf-8-sig")
print(f"\n저장: results/primary2_dpq_nonresponse.csv ({len(rows)}행)")
