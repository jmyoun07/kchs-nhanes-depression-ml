# -*- coding: utf-8 -*-
"""경로A 단계1: 기준선 재현 + within-survey rank/quantile 변환 (척도 비동등 보정)."""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathA_common import *

OUT = os.path.join(RES, "pathA_rank.csv")
ROWS = []


def rec(res, compliant, note=""):
    r = dict(res); r["protocol_compliant"] = bool(compliant); r["note"] = note
    ROWS.append(r)
    pd.DataFrame(ROWS).to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"{r['name']:36s} int={r['internal']:.4f} pre={r['ext_pre']:.4f} "
          f"post={r['ext_post']:.4f} ok={compliant} [{r['seconds']}s] {note}", flush=True)
    return r


print(f"kchs={K.shape} pre={PRE.shape} post={POST.shape} train={len(ITR)} test={len(ITE)}", flush=True)

# ---------------------------------------------------------------- 0) baseline
b, _ = evaluate(K, PRE, POST, NUM, BIN, CAT, tag="A00_baseline_literal")
rec(b, True, "프롬프트 문자그대로 spec(500k/100k)")

# ---------------------------------------------------------------- 1) rank variants
SUBJ = ["srh", "oral_health"]
SUBJ_I = ["srh", "oral_health", "income_q"]
PHYS = ["age", "sleep_hrs", "bmi"]

specs = [
    ("A01_rank_srh_oh",      SUBJ,   dict(weighted=True,  normal=False)),
    ("A02_rank_srh_oh_inc",  SUBJ_I, dict(weighted=True,  normal=False)),
    ("A03_rank_all6",        NUM,    dict(weighted=True,  normal=False)),
    ("A04_rank_phys3",       PHYS,   dict(weighted=True,  normal=False)),
    ("A05_rank_srh_only",    ["srh"], dict(weighted=True, normal=False)),
    ("A06_rank_srh_oh_unw",  SUBJ,   dict(weighted=False, normal=False)),
    ("A07_rank_all6_unw",    NUM,    dict(weighted=False, normal=False)),
    ("A08_rankN_srh_oh",     SUBJ,   dict(weighted=True,  normal=True)),
    ("A09_rankN_all6",       NUM,    dict(weighted=True,  normal=True)),
    ("A10_rank_all6_postref_pre", NUM, dict(weighted=True, normal=False, post_ref="pre")),
    ("A11_rank_srh_oh_postref_pre", SUBJ, dict(weighted=True, normal=False, post_ref="pre")),
]

for tag, cols, kw in specs:
    Kc, Pc, Qc = rank_transform(cols, **kw)
    r, _ = evaluate(Kc, Pc, Qc, NUM, BIN, CAT, tag=tag)
    note = f"cols={','.join(cols)}; " + ";".join(f"{k}={v}" for k, v in kw.items())
    rec(r, True, note + " | NHANES 공변량분포만 사용(라벨 미사용)")

print("saved:", OUT)
