# -*- coding: utf-8 -*-
"""확장 후보 변수 탐색: KCHS 연도별 실제 변수명/라벨/값분포, NHANES 컬럼 존재확인."""
import os, sys, numpy as np, pandas as pd, pyreadstat
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
ROOT = r"C:/Users/rudtn/Desktop/claude code/지역사회건강"
KB = os.path.join(ROOT, "data/raw/kchs"); NB = os.path.join(ROOT, "data/raw/nhanes")
def wl(p):
    p = os.path.abspath(p).replace("/", "\\"); return "\\\\?\\" + p if not p.startswith("\\\\?\\") else p
def kpath(year):
    dd = os.path.join(KB, str(year), "data")
    f = [x for x in sorted(os.listdir(dd)) if x.lower().endswith(".sas7bdat") and "ctv" not in x.lower()][0]
    return os.path.join(dd, f)

PREF = ("soa_01", "ora_01", "drb_", "fma_04", "mtc_17", "mtc_18", "nue_01")
for y in range(2017, 2026):
    _, meta = pyreadstat.read_sas7bdat(wl(kpath(y)), metadataonly=True)
    n2l = dict(zip(meta.column_names, meta.column_labels or [""] * len(meta.column_names)))
    hits = {n: l for n, l in n2l.items() if n.lower().startswith(PREF)}
    print(f"=== KCHS {y} (n_col={len(meta.column_names)}) ===")
    for n, l in sorted(hits.items()): print(f"   {n:12s} | {l}")

print("\n=== NHANES 컬럼 존재확인 ===")
NEED = {("pre", "P_OCQ.xpt"): ["OCD150"], ("post", "OCQ_L.xpt"): ["OCD150"],
        ("pre", "P_OHQ.xpt"): ["OHQ845"], ("post", "OHQ_L.xpt"): ["OHQ845"],
        ("pre", "P_ALQ.xpt"): ["ALQ111", "ALQ121", "ALQ130", "ALQ142"],
        ("post", "ALQ_L.xpt"): ["ALQ111", "ALQ121", "ALQ130", "ALQ142"],
        ("pre", "P_FSQ.xpt"): ["FSQ165", "FSDAD"], ("post", "FSQ_L.xpt"): ["FSQ165", "FSDAD"],
        ("pre", "P_SLQ.xpt"): ["SLD012", "SLD013"], ("post", "SLQ_L.xpt"): ["SLD012", "SLD013"]}
FOLD = {"pre": "pre_2017_2020", "post": "post_2021_2023"}
for (per, fn), cols in NEED.items():
    p = os.path.join(NB, FOLD[per], fn)
    if not os.path.exists(p): print(f"  [X] {per}/{fn} 파일없음"); continue
    _, meta = pyreadstat.read_xport(p, metadataonly=True, encoding="latin1")
    A = set(meta.column_names)
    print(f"  {per}/{fn}: 있음={[c for c in cols if c in A]} 없음={[c for c in cols if c not in A]}")
    print(f"      전체컬럼={meta.column_names}")
