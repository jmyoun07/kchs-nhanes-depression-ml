# -*- coding: utf-8 -*-
"""후보 변수 값 분포 확인 (KCHS)"""
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
V = ["drb_01z2", "drb_01z3", "drb_03z1", "drb_04z1", "drb_05z1", "soa_01z1",
     "ora_01z1", "fma_04z1", "nue_01z1", "mtc_18z1", "mtc_17z1"]
for y in [2017, 2019, 2021, 2024]:
    _, meta = pyreadstat.read_sas7bdat(wl(kp(y)), metadataonly=True)
    have = [v for v in V if v in meta.column_names]
    df, _ = pyreadstat.read_sas7bdat(wl(kp(y)), usecols=have)
    print(f"===== KCHS {y} (n={len(df)}) =====")
    for v in have:
        vc = df[v].value_counts(dropna=False).sort_index()
        print(f"  {v}: " + ", ".join(f"{k}:{int(n)}" for k, n in list(vc.items())[:18]))
