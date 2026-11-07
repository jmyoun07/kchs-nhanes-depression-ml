# -*- coding: utf-8 -*-
"""후보 변수 코딩 의미 검증: value_labels + 교차표"""
import os, sys, pyreadstat, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
KB = r"C:/Users/rudtn/Desktop/claude code/지역사회건강/data/raw/kchs"
def wl(p):
    p = os.path.abspath(p).replace("/", "\\"); return "\\\\?\\" + p
def kp(y):
    dd = os.path.join(KB, str(y), "data")
    f = [x for x in sorted(os.listdir(dd)) if x.lower().endswith(".sas7bdat") and "ctv" not in x.lower()][0]
    return os.path.join(dd, f)
for y in [2017, 2021]:
    _, meta = pyreadstat.read_sas7bdat(wl(kp(y)), metadataonly=True)
    vvl = getattr(meta, "variable_value_labels", {}) or {}
    print(f"== {y} value_labels 개수: {len(vvl)}")
    for v in ["drb_01z2", "drb_01z3", "drb_03z1", "drb_04z1", "fma_04z1", "nue_01z1", "soa_01z1"]:
        if v in vvl: print("   ", v, vvl[v])

# 교차검증: 2021 drb_01z3==1 이 '연간 비음주'인지 (drb_03z1 비해당8 / dra_01z1 평생음주와 대조)
df, _ = pyreadstat.read_sas7bdat(wl(kp(2021)), usecols=["drb_01z3", "drb_03z1", "dra_01z1", "drb_04z1", "drb_05z1"])
print("\n2021 drb_01z3 x drb_03z1:")
print(pd.crosstab(df["drb_01z3"], df["drb_03z1"], dropna=False))
print("\n2021 drb_01z3 x dra_01z1(평생음주 1=예,2=아니오):")
print(pd.crosstab(df["drb_01z3"], df["dra_01z1"], dropna=False))
print("\n2021 drb_01z3 x drb_04z1(폭음_남):")
print(pd.crosstab(df["drb_01z3"], df["drb_04z1"], dropna=False))

d17, _ = pyreadstat.read_sas7bdat(wl(kp(2017)), usecols=["drb_01z2", "drb_03z1", "dra_01z1"])
print("\n2017 drb_01z2 x dra_01z1:")
print(pd.crosstab(d17["drb_01z2"], d17["dra_01z1"], dropna=False))
