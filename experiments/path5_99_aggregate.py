# -*- coding: utf-8 -*-
"""
path5_99_aggregate.py — 경로5 전체 결과 통합 -> results/path5_ALL.csv

기준선: 내부 0.830969 / ext_pre 0.774013 / ext_post 0.765262 (path5_00 재현)
"""
import os, glob, json
import pandas as pd, numpy as np

RES = r"C:/Users/rudtn/Desktop/claude code/지역사회건강/results"
BASE = dict(internal=0.830969, ext_pre=0.774013, ext_post=0.765262)

files = {
    "path5_01_screen.csv": "focal/labelsmooth/gce/gradclip/savage/brier",
    "path5_01b_rest.csv":  "gce/gradclip/brier/native대안(xent,huber,fair)",
    "path5_02_rank.csv":   "lambdarank/rank_xendcg/RankNet(pairwise)",
    "path5_03_dro.csv":    "groupDRO/전이정규화",
    "path5_04_extra.csv":  "class-weight/undersample/sigmoid",
    "path5_08_seedstab.csv": "RankNet혼합 시드안정성(5시드)+앙상블",
}
frames = []
for f, grp in files.items():
    p = os.path.join(RES, f)
    if not os.path.exists(p):
        continue
    df = pd.read_csv(p)
    df = df[[c for c in ["name", "internal", "ext_pre", "ext_post", "protocol_compliant", "note"] if c in df.columns]].copy()
    df.insert(0, "group", grp)
    df.insert(1, "src", f)
    frames.append(df)

alldf = pd.concat(frames, ignore_index=True)
alldf = alldf.dropna(subset=["internal", "ext_pre", "ext_post"], how="all")
for c in ["internal", "ext_pre", "ext_post"]:
    alldf[c] = pd.to_numeric(alldf[c], errors="coerce")
alldf["d_internal"] = (alldf["internal"] - BASE["internal"]).round(6)
alldf["d_pre"] = (alldf["ext_pre"] - BASE["ext_pre"]).round(6)
alldf["d_post"] = (alldf["ext_post"] - BASE["ext_post"]).round(6)
alldf["improves_all3"] = (alldf["d_internal"] > 0) & (alldf["d_pre"] > 0) & (alldf["d_post"] > 0)
alldf["improves_both_ext"] = (alldf["d_pre"] > 0) & (alldf["d_post"] > 0)

alldf.to_csv(os.path.join(RES, "path5_ALL.csv"), index=False, encoding="utf-8-sig")
print("saved path5_ALL.csv  rows:", len(alldf))

print("\n--- 외부 pre·post 동시 개선 후보 ---")
sub = alldf[alldf.improves_both_ext].sort_values("d_post", ascending=False)
print(sub[["src", "name", "internal", "ext_pre", "ext_post", "d_pre", "d_post"]].to_string(index=False))

print("\n--- 내부·pre·post 3개 모두 개선 ---")
sub3 = alldf[alldf.improves_all3].sort_values("d_post", ascending=False)
print(sub3[["src", "name", "internal", "ext_pre", "ext_post", "d_internal", "d_pre", "d_post"]].to_string(index=False))

print("\n--- 내부 성능 상위 8 (내부 CV 선택 기준의 프록시) ---")
print(alldf.sort_values("internal", ascending=False).head(8)[
    ["src", "name", "internal", "ext_pre", "ext_post"]].to_string(index=False))

print("\n--- 외부 pre 최고 8 ---")
print(alldf.sort_values("ext_pre", ascending=False).head(8)[
    ["src", "name", "internal", "ext_pre", "ext_post"]].to_string(index=False))
print("\n--- 외부 post 최고 8 ---")
print(alldf.sort_values("ext_post", ascending=False).head(8)[
    ["src", "name", "internal", "ext_pre", "ext_post"]].to_string(index=False))

print(f"\nBASELINE: internal={BASE['internal']} pre={BASE['ext_pre']} post={BASE['ext_post']}")
print("0.80 도달?  pre max=%.4f  post max=%.4f" % (alldf.ext_pre.max(), alldf.ext_post.max()))
