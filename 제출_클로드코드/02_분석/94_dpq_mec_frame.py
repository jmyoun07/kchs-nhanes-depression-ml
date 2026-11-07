# -*- coding: utf-8 -*-
"""94_dpq_mec_frame.py — DPQ 미응답을 올바른 추정 프레임에서 다시 계산한다.

지적 내용: 14.3%/32.7% 는 '면접을 받은 20세 이상 전원'을 분모로 삼았다. 그런데 그 가운데
상당수는 애초에 MEC 검진을 받지 않아 MEC 가중치가 0 이다 — 논문의 모든 가중 추정의 프레임
밖에 있는 사람들이다. DPQ 는 MEC 에서 시행되므로, 미응답률은 MEC 프레임에서 세어야 한다.

세 가지를 모두 낸다:
  · 면접 프레임(종전): 분모 = 20세 이상 면접자 전원
  · MEC 프레임(정본): 분모 = MEC 가중치 > 0 인 20세 이상
  · 가중 MEC 프레임: MEC 가중치로 가중한 비율

출력: results/primary2_dpq_nonresponse.csv (프레임 열 추가)
"""
import os, sys, numpy as np, pandas as pd, pyreadstat
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:/Users/rudtn/Desktop/claude code/지역사회건강/data/raw/nhanes"
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DPQ = [f"DPQ0{i}0" for i in range(1, 10)]
EDU = {1: "<HS", 2: "<HS", 3: "HS", 4: "College", 5: "College"}
RACE = {1: "Mex", 2: "OthHisp", 3: "White", 4: "Black", 6: "Asian", 7: "Other"}

def rd(folder, namer, mod, cols):
    return pyreadstat.read_xport(os.path.join(NB, folder, namer(mod)), usecols=["SEQN"] + cols, encoding="latin1")[0]

rows = []
for period, folder, namer, wtcol in [("pre", "pre_2017_2020", lambda m: f"P_{m}.xpt", "WTMECPRP"),
                                     ("post", "post_2021_2023", lambda m: f"{m}_L.xpt", "WTMEC2YR")]:
    demo = rd(folder, namer, "DEMO", ["RIDAGEYR", "DMDEDUC2", "RIDRETH3", wtcol])
    dpq = rd(folder, namer, "DPQ", DPQ)
    ad = demo.merge(dpq, on="SEQN", how="left")
    ad = ad[ad["RIDAGEYR"] >= 20].copy()
    nm = ad[DPQ].apply(pd.to_numeric, errors="coerce").mask(lambda x: x.isin([7, 9])).isna().sum(axis=1)
    ad["nr3"] = (nm >= 3).astype(float)
    ad["anymiss"] = (nm > 0).astype(float)
    w = pd.to_numeric(ad[wtcol], errors="coerce").fillna(0.0)
    mec = w > 0
    print(f"\n[{period}] 면접 20+ n={len(ad):,} · MEC 가중>0 n={int(mec.sum()):,} "
          f"({mec.mean()*100:.1f}%) · MEC 미검진 {int((~mec).sum()):,}", flush=True)
    print(f"    ≥3결측  면접프레임 {ad['nr3'].mean()*100:.1f}% · MEC프레임 {ad.loc[mec,'nr3'].mean()*100:.1f}% "
          f"· MEC가중 {np.average(ad.loc[mec,'nr3'], weights=w[mec])*100:.1f}%", flush=True)
    print(f"    비응답자 중 MEC 미검진 비율: {(~mec & (ad['nr3']==1)).sum()/max((ad['nr3']==1).sum(),1)*100:.1f}%", flush=True)
    rows.append(dict(period=period, factor="__overall__", group="all",
                     n_interview=int(len(ad)), n_mec=int(mec.sum()),
                     nr3_interview_pct=round(float(ad["nr3"].mean()*100), 1),
                     nr3_mec_pct=round(float(ad.loc[mec, "nr3"].mean()*100), 1),
                     nr3_mec_weighted_pct=round(float(np.average(ad.loc[mec, "nr3"], weights=w[mec])*100), 1),
                     anymiss_interview_pct=round(float(ad["anymiss"].mean()*100), 1)))
    for fac, col, mp in [("education", "DMDEDUC2", EDU), ("race", "RIDRETH3", RACE)]:
        gcol = ad[col].map(mp)
        for k in sorted(set(gcol.dropna())):
            sel = gcol == k; selm = sel & mec
            if selm.sum() == 0: continue
            rows.append(dict(period=period, factor=fac, group=k,
                             n_interview=int(sel.sum()), n_mec=int(selm.sum()),
                             nr3_interview_pct=round(float(ad.loc[sel, "nr3"].mean()*100), 1),
                             nr3_mec_pct=round(float(ad.loc[selm, "nr3"].mean()*100), 1),
                             nr3_mec_weighted_pct=round(float(np.average(ad.loc[selm, "nr3"], weights=w[selm])*100), 1),
                             anymiss_interview_pct=round(float(ad.loc[sel, "anymiss"].mean()*100), 1)))
            print(f"      {fac:<9} {k:<9} MEC프레임 ≥3결측 {rows[-1]['nr3_mec_pct']:>5.1f}% "
                  f"(면접프레임 {rows[-1]['nr3_interview_pct']:>5.1f}%, n_mec={rows[-1]['n_mec']:,})", flush=True)
pd.DataFrame(rows).to_csv(f"{RES}/primary2_dpq_nonresponse.csv", index=False, encoding="utf-8-sig")
print(f"\n저장: results/primary2_dpq_nonresponse.csv ({len(rows)}행)")
