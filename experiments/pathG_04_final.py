# -*- coding: utf-8 -*-
"""pathG_04: 마무리 - 대각 CORAL, LAmask, combo, 그리고 기준선 외부 AUROC 부트스트랩 CI.

부트스트랩 CI는 '0.80 도달 가능성'을 통계적으로 판정한다:
기준선 외부 AUROC의 복합표본(PSU/strata) 95% 상한이 0.80에 못 미치면,
17변수 전이로 0.80은 표본 변동으로도 도달 불가.
"""
import os, sys, time, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathG_common import *
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

T0 = time.time()
k, pre, post = load()
tr, ho = base_split(k)
y = k["dep"].astype(int).values
Wk = k["W"].to_numpy(float)
ypre, wpre = pre["dep"].astype(int).values, pre["W"].to_numpy(float)
ypost, wpost = post["dep"].astype(int).values, post["W"].to_numpy(float)
Ktr = k.iloc[tr][FEATURES].reset_index(drop=True)
Kho = k.iloc[ho][FEATURES].reset_index(drop=True)
Pre = pre[FEATURES].reset_index(drop=True)
Post = post[FEATURES].reset_index(drop=True)
ytr, yho, who = y[tr], y[ho], Wk[ho]

CT = make_ct()
Xtr = CT.fit_transform(Ktr).astype(np.float64)
Xho = CT.transform(Kho).astype(np.float64)
Xpre = CT.transform(Pre).astype(np.float64)
Xpost = CT.transform(Post).astype(np.float64)
FN = list(CT.get_feature_names_out())
LA = FN.index("bin__living_alone")
BASE = None


def run(name, xtr, xho, xpre, xpost, sw=None, compliant=False, note="", ytr_=None):
    t = time.time()
    m = lgb.LGBMClassifier(**LGB_PARAMS)
    m.fit(xtr, ytr if ytr_ is None else ytr_, sample_weight=sw)
    return rec(name,
               wauc(yho, m.predict_proba(xho)[:, 1], who),
               wauc(ypre, m.predict_proba(xpre)[:, 1], wpre),
               wauc(ypost, m.predict_proba(xpost)[:, 1], wpost),
               compliant, note, base=BASE, secs=time.time() - t), m


r0, m0 = run("G04_base", Xtr, Xho, Xpre, Xpost, compliant=True, note="기준선 재현")
BASE = (r0["internal"], r0["ext_pre"], r0["ext_post"])

# 대각 CORAL (도메인별 평균·표준편차 정렬, target->source)
Zs = np.nan_to_num(Xtr, nan=0.0)
ms, ss = Zs.mean(0), Zs.std(0) + 1e-9
for ttag, Xt in [("pre", Xpre), ("post", Xpost), ("both", np.vstack([Xpre, Xpost]))]:
    Zt = np.nan_to_num(Xt, nan=0.0); mt, st = Zt.mean(0), Zt.std(0) + 1e-9
    xp = (np.nan_to_num(Xpre, nan=0.0) - mt) / st * ss + ms
    xq = (np.nan_to_num(Xpost, nan=0.0) - mt) / st * ss + ms
    run(f"G04_DIAGCORAL_t2s_{ttag}", Zs, np.nan_to_num(Xho, nan=0.0), xp, xq,
        compliant=False, note=f"대각 CORAL(도메인별 표준화, 기준 {ttag}). NHANES 공변량 → 회색지대")
    dump("pathG_04_final.csv")

# living_alone 학습 마스킹
rng = np.random.RandomState(SEED)
for pm in [0.5, 1.0]:
    Xa = Xtr.copy(); mk = rng.rand(len(Xa)) < pm; Xa[mk, LA] = np.nan
    run(f"G04_LAMASK_p{pm}", Xa, Xho, Xpre, Xpost, compliant=False,
        note=f"living_alone {pm:.0%} 학습결측(NHANES pre 100%결측 구조 이용). 회색지대")
    dump("pathG_04_final.csv")

# combo: age+sex raking + 대각 CORAL(both)
AGE_BINS = [20, 30, 40, 50, 60, 70, 80, 200]
cs = pd.cut(Ktr["age"], AGE_BINS, right=False).astype(str).str.cat(Ktr["sex"].astype(str), sep="|")
tgt = pd.concat([Pre, Post], ignore_index=True)
tw = np.r_[wpre, wpost]
ct_ = pd.cut(tgt["age"], AGE_BINS, right=False).astype(str).str.cat(tgt["sex"].astype(str), sep="|")
ps = cs.value_counts(normalize=True)
twg = pd.Series(tw, index=ct_.index).groupby(ct_).sum(); pt = twg / twg.sum()
ratio = (pt.reindex(ps.index).fillna(0.0) / ps).clip(0.1, 10.0)
w = cs.map(ratio).fillna(1.0).to_numpy(float); w = w / w.mean()
Zt = np.nan_to_num(np.vstack([Xpre, Xpost]), nan=0.0); mt, st = Zt.mean(0), Zt.std(0) + 1e-9
xp = (np.nan_to_num(Xpre, nan=0.0) - mt) / st * ss + ms
xq = (np.nan_to_num(Xpost, nan=0.0) - mt) / st * ss + ms
run("G04_COMBO_rakeAgeSex+diag_both", Zs, np.nan_to_num(Xho, nan=0.0), xp, xq,
    sw=w, compliant=False, note="age×sex raking + 대각 CORAL(both) 조합. 회색지대")
dump("pathG_04_final.csv")


# ---------------- 부트스트랩 CI (PSU/strata 복원추출) ----------------
def boot_ci(df, p, n=3000, seed=SEED):
    yv = df["dep"].astype(int).values
    wv = df["W"].to_numpy(float)
    st = df["strata"].values; ps = df["psu"].values
    groups = {}
    for s in np.unique(st):
        sm = st == s
        groups[s] = [np.where(sm & (ps == u))[0] for u in np.unique(ps[sm])]
    rs = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        idx = []
        for s, us in groups.items():
            if len(us) < 2:
                idx.append(np.concatenate(us) if us else np.array([], int)); continue
            pick = rs.randint(0, len(us), len(us))
            idx.append(np.concatenate([us[j] for j in pick]))
        idx = np.concatenate(idx)
        if len(np.unique(yv[idx])) < 2:
            continue
        try:
            out.append(roc_auc_score(yv[idx], p[idx], sample_weight=wv[idx]))
        except Exception:
            pass
    out = np.array(out)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)),
            float(out.mean()), len(out), float((out >= 0.80).mean()))


pp = m0.predict_proba(Xpre)[:, 1]
po = m0.predict_proba(Xpost)[:, 1]
cis = []
for tag, df, p in [("pre", pre, pp), ("post", post, po)]:
    lo, hi, mn, nb, p80 = boot_ci(df, p)
    row = {"set": tag, "wauc": round(wauc(df["dep"], p, df["W"]), 5),
               "boot_mean": round(mn, 5), "ci95_lo": round(lo, 5), "ci95_hi": round(hi, 5),
               "p_ge_0.80": round(p80, 4), "n_boot": nb,
               "note": "기준선 17변수 전이모델, PSU/strata 복원추출 3000회. p_ge_0.80=부트스트랩표본이 0.80 넘는 비율"}
    cis.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)
pd.DataFrame(cis).to_csv(os.path.join(RES, "pathG_04_bootci.csv"),
                         index=False, encoding="utf-8-sig")
dump("pathG_04_final.csv")
print("TOTAL %.1f min" % ((time.time() - T0) / 60))
