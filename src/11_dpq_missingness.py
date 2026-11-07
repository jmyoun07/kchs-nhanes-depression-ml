# -*- coding: utf-8 -*-
"""
11_dpq_missingness.py (Codex #13/#14/#19) — NHANES DPQ(PHQ-9) 결측 특성화 + 민감도.
- 성인20+ 중 DPQ 결측을 성·연령·교육·인종별로.
- 1-2문항 결측 person-mean 대체 민감도(완전례 vs 대체후 유병률).
- 연령 18-19 포함 시 영향(#19).
결과: results/dpq_missingness.txt
"""
import os, numpy as np, pandas as pd, pyreadstat
NB = r"C:/Users/rudtn/Desktop/claude code/지역사회건강/data/raw/nhanes"
OUT = r"C:/Users/rudtn/Desktop/claude code/지역사회건강/results/dpq_missingness.txt"
DPQ = [f"DPQ0{i}0" for i in range(1, 10)]
L = []
def rd(folder, namer, mod, cols):
    return pyreadstat.read_xport(os.path.join(NB, folder, namer(mod)), usecols=["SEQN"] + cols, encoding="latin1")[0]

for period, folder, namer in [("pre", "pre_2017_2020", lambda m: f"P_{m}.xpt"), ("post", "post_2021_2023", lambda m: f"{m}_L.xpt")]:
    demo = rd(folder, namer, "DEMO", ["RIDAGEYR", "RIAGENDR", "DMDEDUC2", "RIDRETH3"])
    dpq = rd(folder, namer, "DPQ", DPQ)
    m = demo.merge(dpq, on="SEQN", how="left")
    ad = m[m["RIDAGEYR"] >= 20].copy()
    items = ad[DPQ].apply(pd.to_numeric, errors="coerce").mask(lambda x: x.isin([7, 9]))
    nmiss = items.isna().sum(axis=1)
    L.append(f"\n########## NHANES {period} ##########")
    L.append(f"성인20+ n={len(ad)} | DPQ 완전례(0결측)={np.mean(nmiss==0)*100:.1f}% | 1-2결측={np.mean(nmiss.between(1,2))*100:.1f}% | ≥3결측(사실상 미응답)={np.mean(nmiss>=3)*100:.1f}%")
    # 결측(any) 하위군별
    ad["anymiss"] = (nmiss > 0).astype(int)
    for fac, col, mp in [("성별", "RIAGENDR", {1: "M", 2: "F"}),
                          ("교육", "DMDEDUC2", {1: "<HS", 2: "<HS", 3: "HS", 4: "College", 5: "College"}),
                          ("인종", "RIDRETH3", {1: "Mex", 2: "OthHisp", 3: "White", 4: "Black", 6: "Asian", 7: "Other"})]:
        ad["_g"] = ad[col].map(mp)
        g = ad.groupby("_g")["anymiss"].agg(["mean", "size"])
        L.append(f"  {fac} 결측률: " + ", ".join(f"{k}={v['mean']*100:.1f}%(n{int(v['size'])})" for k, v in g.iterrows()))
    ab = pd.cut(ad["RIDAGEYR"], [19, 39, 59, 200], labels=["20-39", "40-59", "60+"])
    g = ad.groupby(ab)["anymiss"].mean()
    L.append("  연령 결측률: " + ", ".join(f"{k}={v*100:.1f}%" for k, v in g.items()))
    # person-mean 민감도(1-2결측)
    cc = nmiss == 0
    prev_cc = ((items[cc].sum(axis=1)) >= 10).mean()
    use = nmiss <= 2
    filled = items.copy()
    pm = items.mean(axis=1)  # 개인 평균
    for c in DPQ: filled[c] = filled[c].fillna(pm)
    sums = filled[use].sum(axis=1)
    prev_pm = (sums >= 10).mean()
    L.append(f"  유병률: 완전례만 {prev_cc*100:.2f}% (n={int(cc.sum())}) vs 1-2결측 person-mean 포함 {prev_pm*100:.2f}% (n={int(use.sum())}) → 차이 {abs(prev_pm-prev_cc)*100:.2f}%p")
    # #19 연령 18-19 영향
    a1819 = m[(m["RIDAGEYR"].isin([18, 19]))]
    it19 = a1819[DPQ].apply(pd.to_numeric, errors="coerce").mask(lambda x: x.isin([7, 9]))
    cc19 = it19.notna().all(axis=1)
    if cc19.sum() > 0:
        L.append(f"  #19 연령18-19 완전례 n={int(cc19.sum())}, 우울유병률 {((it19[cc19].sum(axis=1))>=10).mean()*100:.1f}% (현재 ≥20 기준서 제외됨)")

open(OUT, "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L))
