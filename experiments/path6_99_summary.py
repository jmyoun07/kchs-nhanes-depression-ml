# -*- coding: utf-8 -*-
"""경로6 종합 — 모든 실험 결과를 하나의 마스터 표로 병합하고 기준선 대비 델타를 계산."""
import os, glob
import numpy as np, pandas as pd

RES = r"C:/Users/rudtn/Desktop/claude code/지역사회건강/results"
BASE = dict(internal=0.830969, ext_pre=0.774013, ext_post=0.765262)

files = ["path6_00_baseline.csv", "path6_01_strat.csv", "path6_02_blend.csv",
         "path6_03_twostage.csv", "path6_04_latent.csv", "path6_05_gate.csv",
         "path6_06_proxy.csv", "path6_07_structure.csv", "path6_08_partlin.csv",
         "path6_10_gate_shallow.csv"]
frames = []
for f in files:
    p = os.path.join(RES, f)
    if not os.path.exists(p):
        print("missing:", f); continue
    d = pd.read_csv(p)
    d.insert(0, "stage", f.replace("path6_", "").replace(".csv", ""))
    frames.append(d)
M = pd.concat(frames, ignore_index=True)
for c in ["internal", "ext_pre", "ext_post"]:
    M["d_" + c] = (M[c] - BASE[c]).round(6)
M["ext_mean"] = ((M.ext_pre + M.ext_post) / 2).round(6)
M["d_ext_mean"] = (M.ext_mean - (BASE["ext_pre"] + BASE["ext_post"]) / 2).round(6)
M = M[["stage", "name", "internal", "ext_pre", "ext_post", "ext_mean",
       "d_internal", "d_ext_pre", "d_ext_post", "d_ext_mean",
       "protocol_compliant", "seconds", "note"]]
M.to_csv(os.path.join(RES, "path6_MASTER.csv"), index=False, encoding="utf-8-sig")
print("rows:", len(M))

C = M[M.protocol_compliant == True].sort_values("d_ext_mean", ascending=False)
print("\n=== §2.10 준수 기법 상위 15 (외부 pre/post 평균 기준) ===")
print(C.head(15)[["stage", "name", "internal", "ext_pre", "ext_post",
                  "d_ext_pre", "d_ext_post"]].to_string(index=False))
print("\n=== §2.10 준수 기법 하위 8 ===")
print(C.tail(8)[["stage", "name", "internal", "ext_pre", "ext_post",
                 "d_ext_pre", "d_ext_post"]].to_string(index=False))
print("\n=== 비준수/진단 전용 중 외부 최고 5 ===")
N = M[M.protocol_compliant == False].sort_values("ext_mean", ascending=False)
print(N.head(5)[["stage", "name", "ext_pre", "ext_post", "note"]].to_string(index=False))

print("\n최고 ext_pre (준수):", C.ext_pre.max(), "| 최고 ext_post (준수):", C.ext_post.max())
print("0.80 도달 여부 (준수, pre):", bool((C.ext_pre >= 0.80).any()),
      "| (준수, post):", bool((C.ext_post >= 0.80).any()))
print("0.80 도달 여부 (비준수 포함):", bool((M.ext_pre >= 0.80).any() or (M.ext_post >= 0.80).any()))
C.to_csv(os.path.join(RES, "path6_COMPLIANT_RANKED.csv"), index=False, encoding="utf-8-sig")
print("saved -> path6_MASTER.csv, path6_COMPLIANT_RANKED.csv")
