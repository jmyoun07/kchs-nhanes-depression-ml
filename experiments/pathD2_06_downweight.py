# -*- coding: utf-8 -*-
"""
pathD2_06_downweight.py — 기법(3): 불안정 변수 다운웨이팅.

pathD2_02에서 얻은 연도간 안정성(stability_t = |평균계수| / 연도간 sd)을 이용해
LightGBM의 feature_contri(변수별 이득 배율)로 불안정 변수의 영향력을 억제한다.
가중치 산출에 NHANES 정보는 일절 쓰이지 않는다(KCHS 연도 간 변동만 사용).

추가로 srh 편중 완화(단일변수 gain 41%)를 feature_contri로 시험한다.
"""
import os
import sys
import time
import json

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathD2_engine import cv_eval, final_eval, data, RES, SEED, line

OUT = os.path.join(RES, "pathD2_downweight.csv")
TRAIN_N_CV = 250000

d = data()
st = pd.read_csv(os.path.join(RES, "pathD2_feature_stability.csv"))
st = st.set_index("col").loc[d.cols]
t = st["stability_t"].to_numpy(float)

# 안정성 -> feature_contri (0.1~1.0). 상대순위만 사용.
r = (t - t.min()) / (t.max() - t.min())
CONTRI_LIN = np.clip(0.1 + 0.9 * r, 0.1, 1.0)
CONTRI_SQRT = np.clip(0.1 + 0.9 * np.sqrt(r), 0.1, 1.0)
CONTRI_HARD = np.where(t >= np.median(t), 1.0, 0.2)

SRH = d.cols.index("srh")
CONTRI_SRH50 = np.ones(len(d.cols)); CONTRI_SRH50[SRH] = 0.5
CONTRI_SRH25 = np.ones(len(d.cols)); CONTRI_SRH25[SRH] = 0.25
CONTRI_LIN_SRH = CONTRI_LIN.copy(); CONTRI_LIN_SRH[SRH] = 0.5

print("feature_contri (linear):")
for c, a, b in zip(d.cols, t, CONTRI_LIN):
    print(f"  {c:16s} stability_t={a:8.2f} contri={b:.3f}")

G = [
    ("F1_contri_linear", dict(params=dict(feature_contri=list(CONTRI_LIN))),
     "안정성 선형 배율"),
    ("F2_contri_sqrt", dict(params=dict(feature_contri=list(CONTRI_SQRT))),
     "안정성 sqrt 배율"),
    ("F3_contri_hard", dict(params=dict(feature_contri=list(CONTRI_HARD))),
     "중앙값 미만 0.2배"),
    ("F4_srh_50", dict(params=dict(feature_contri=list(CONTRI_SRH50))),
     "srh 편중 완화 0.5"),
    ("F5_srh_25", dict(params=dict(feature_contri=list(CONTRI_SRH25))),
     "srh 편중 완화 0.25"),
    ("F6_contri_lin_srh50", dict(params=dict(feature_contri=list(CONTRI_LIN_SRH))),
     "안정성배율 + srh 완화"),
]

rows = []
t0 = time.time()
for i, (name, spec, note) in enumerate(G, 1):
    s = dict(spec); s.setdefault("family", "lgb"); s["train_n"] = TRAIN_N_CV
    tt = time.time()
    cvy = cv_eval(s, "loyo")
    ext, _ = final_eval(s, name)
    rows.append(dict(name=name, note=note,
                     loyo_mean=cvy["mean"], loyo_min=cvy["min"], loyo_std=cvy["std"],
                     loyo_mean_minus_std=cvy["mean"] - cvy["std"],
                     internal=ext["internal"], ext_pre=ext["ext_pre"],
                     ext_post=ext["ext_post"], seconds=round(time.time() - tt, 1),
                     loyo_per_fold=json.dumps(cvy["per_fold"])))
    print(f"[{i}/{len(G)}] " + line(name, cvy, ext) + f" [{time.time()-t0:.0f}s]")
    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")

print("saved:", OUT)
