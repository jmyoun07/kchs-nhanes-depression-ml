# -*- coding: utf-8 -*-
"""FSQ165(ever) vs FSQ012(last 12mo) vs FSD230(now) 구분 확인"""
import os, sys, pandas as pd, pyreadstat
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
NB = r"C:/Users/rudtn/Desktop/claude code/지역사회건강/data/raw/nhanes"
for per, fol, fn in [("pre", "pre_2017_2020", "P_FSQ.xpt"), ("post", "post_2021_2023", "FSQ_L.xpt")]:
    p = os.path.join(NB, fol, fn)
    _, meta = pyreadstat.read_xport(p, metadataonly=True, encoding="latin1")
    lab = dict(zip(meta.column_names, meta.column_labels or []))
    for v in ["FSQ165", "FSQ012", "FSD230", "FSDAD", "FSD151"]:
        print(f"{per} {v}: {lab.get(v)}")
    d = pyreadstat.read_xport(p, usecols=["SEQN", "FSQ165", "FSQ012", "FSD230"], encoding="latin1")[0]
    for v in ["FSQ165", "FSQ012", "FSD230"]:
        print(f"  {per} {v} raw:", d[v].value_counts(dropna=False).sort_index().to_dict())
    print(f"  {per} FSQ165 x FSD230:\n", pd.crosstab(d["FSQ165"], d["FSD230"], dropna=False))
    print()
