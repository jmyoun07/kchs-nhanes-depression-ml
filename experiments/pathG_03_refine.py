# -*- coding: utf-8 -*-
"""pathG_03: 경로2 심화 - 안정적 재가중(raking), 대각 CORAL, 조합, 부트스트랩 한계.

중요도가중이 도메인분류기 AUC~0.98(거의 완전 분리)로 붕괴하는 것을 확인했으므로,
저차원 주변분포(marginal)만 맞추는 안정적 재가중을 시도한다.
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


def run(name, xtr, xho, xpre, xpost, sw=None, params=None, compliant=False,
        note="", ytr_=None, extra=None):
    t = time.time()
    p = dict(LGB_PARAMS)
    if params:
        p.update(params)
    m = lgb.LGBMClassifier(**p)
    m.fit(xtr, ytr if ytr_ is None else ytr_, sample_weight=sw)
    r = rec(name,
            wauc(yho, m.predict_proba(xho)[:, 1], who),
            wauc(ypre, m.predict_proba(xpre)[:, 1], wpre),
            wauc(ypost, m.predict_proba(xpost)[:, 1], wpost),
            compliant, note, base=BASE, extra=extra, secs=time.time() - t)
    dump("pathG_03_refine.csv")
    return r, m


r0, m0 = run("G03_base", Xtr, Xho, Xpre, Xpost, compliant=True, note="기준선 재현")
BASE = (r0["internal"], r0["ext_pre"], r0["ext_post"])

# ---------------- (A) 주변분포 raking 재가중 ----------------
AGE_BINS = [20, 30, 40, 50, 60, 70, 80, 200]
BMI_BINS = [0, 20, 23, 25, 30, 100]


def cells(df, keys):
    parts = []
    for kk in keys:
        if kk == "age":
            parts.append(pd.cut(df["age"], AGE_BINS, right=False).astype(str))
        elif kk == "bmi":
            parts.append(pd.cut(df["bmi"], BMI_BINS, right=False).astype(str))
        elif kk == "sleep":
            parts.append(pd.cut(df["sleep_hrs"], [0, 6, 7, 8, 30], right=False).astype(str))
        else:
            parts.append(df[kk].astype(str))
    return parts[0].str.cat(parts[1:], sep="|") if len(parts) > 1 else parts[0]


def rake_w(keys, tgt_df, tgt_w, lo=0.1, hi=10.0):
    cs = cells(Ktr, keys)
    ct_ = cells(tgt_df, keys)
    ps = cs.value_counts(normalize=True)
    tw = pd.Series(tgt_w, index=ct_.index).groupby(ct_).sum()
    pt = tw / tw.sum()
    ratio = (pt.reindex(ps.index).fillna(0.0) / ps).clip(lo, hi)
    w = cs.map(ratio).fillna(1.0).to_numpy(float)
    return w / w.mean()


TGT = {"pre": (Pre.assign(**{}), wpre),
       "post": (Post, wpost),
       "both": (pd.concat([Pre, Post], ignore_index=True), np.r_[wpre, wpost])}
# Pre/Post 는 FEATURES만 있으므로 그대로 사용
TGT["pre"] = (Pre, wpre)

KEYSETS = [("age", "sex"), ("age", "sex", "edu"),
           ("age", "sex", "edu", "bmi"),
           ("age", "sex", "edu", "smoking", "bmi"),
           ("bmi",), ("age", "sex", "edu", "sleep", "bmi")]

for ttag, (tdf, tw) in TGT.items():
    for ks in KEYSETS:
        for lo, hi in [(0.1, 10.0), (0.2, 5.0), (1.0, 10.0)]:
            w = rake_w(ks, tdf, tw, lo, hi)
            e = float(w.sum() ** 2 / np.sum(w ** 2))
            run(f"G03_RAKE_{ttag}_{'+'.join(ks)}_c{lo}-{hi}", Xtr, Xho, Xpre, Xpost,
                sw=w, compliant=False,
                note=(f"주변분포 raking: {'+'.join(ks)} 셀에서 NHANES {ttag}(설계가중) "
                      f"비율에 맞춰 KCHS 재가중. ESS={e:,.0f}. "
                      f"NHANES 공변량 주변분포 사용 → §2.10 회색지대(단, 인구 마진은 "
                      f"공개통계로도 얻을 수 있어 배포현실성은 상대적으로 높음)"),
                extra=dict(ess=round(e)))
    dump("pathG_03_refine.csv")

# ---------------- (B) 대각 CORAL (도메인별 z-표준화) ----------------
def diag_align(Xs, Xt, mode="t2s"):
    Zs, Zt = np.nan_to_num(Xs, nan=0.0), np.nan_to_num(Xt, nan=0.0)
    ms, ss = Zs.mean(0), Zs.std(0) + 1e-9
    mt, st = Zt.mean(0), Zt.std(0) + 1e-9
    if mode == "t2s":
        return None, (Zt - mt) / st * ss + ms
    return (Zs - ms) / ss * st + mt, None


for ttag, Xt in [("pre", Xpre), ("post", Xpost), ("both", np.vstack([Xpre, Xpost]))]:
    _, _ = None, None
    Zs = np.nan_to_num(Xtr, nan=0.0)
    ms, ss = Zs.mean(0), Zs.std(0) + 1e-9
    Ztall = np.nan_to_num(Xt, nan=0.0)
    mt, st = Ztall.mean(0), Ztall.std(0) + 1e-9
    xp = (np.nan_to_num(Xpre, nan=0.0) - mt) / st * ss + ms
    xq = (np.nan_to_num(Xpost, nan=0.0) - mt) / st * ss + ms
    run(f"G03_DIAGCORAL_t2s_{ttag}", Zs, np.nan_to_num(Xho, nan=0.0), xp, xq,
        compliant=False,
        note=f"대각 CORAL(도메인별 평균·표준편차 정렬, 기준 {ttag}). 회색지대")
dump("pathG_03_refine.csv")

# ---------------- (C) living_alone 구조적 결측 대응 ----------------
rng = np.random.RandomState(SEED)
for pm in [0.25, 0.5, 1.0]:
    Xa = Xtr.copy()
    m = rng.rand(len(Xa)) < pm
    Xa[m, LA] = np.nan
    run(f"G03_LAMASK_p{pm}", Xa, Xho, Xpre, Xpost, compliant=False,
        note=(f"학습 시 living_alone을 {pm:.0%} 행에서 결측 처리. "
              f"NHANES pre의 100% 결측 구조를 이용 → 회색지대(라벨 미사용)"))
dump("pathG_03_refine.csv")

# ---------------- (D) 조합: raking + 대각정렬 ----------------
best_keys = ("age", "sex", "edu")
for ttag, (tdf, tw) in TGT.items():
    w = rake_w(best_keys, tdf, tw, 0.1, 10.0)
    Zs = np.nan_to_num(Xtr, nan=0.0)
    ms, ss = Zs.mean(0), Zs.std(0) + 1e-9
    Xt = {"pre": Xpre, "post": Xpost, "both": np.vstack([Xpre, Xpost])}[ttag]
    Ztall = np.nan_to_num(Xt, nan=0.0)
    mt, st = Ztall.mean(0), Ztall.std(0) + 1e-9
    xp = (np.nan_to_num(Xpre, nan=0.0) - mt) / st * ss + ms
    xq = (np.nan_to_num(Xpost, nan=0.0) - mt) / st * ss + ms
    run(f"G03_COMBO_rake+diag_{ttag}", Zs, np.nan_to_num(Xho, nan=0.0), xp, xq,
        sw=w, compliant=False, note=f"raking(age+sex+edu, {ttag}) + 대각 CORAL 조합. 회색지대")
dump("pathG_03_refine.csv")

# ---------------- (E) 기준선 외부 AUROC 부트스트랩 (한계 판단용) ----------------
def boot_ci(df, p, n=2000, seed=SEED):
    yv = df["dep"].astype(int).values
    wv = df["W"].to_numpy(float)
    st = df["strata"].values
    ps = df["psu"].values
    key = pd.Series([f"{a}_{b}" for a, b in zip(st, ps)])
    groups = {s: [np.where((st == s) & (ps == u))[0] for u in np.unique(ps[st == s])]
              for s in np.unique(st)}
    rs = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        idx = []
        for s, us in groups.items():
            if len(us) < 2:
                idx.extend(np.concatenate(us)); continue
            pick = rs.randint(0, len(us), len(us))
            idx.extend(np.concatenate([us[j] for j in pick]))
        idx = np.array(idx)
        if len(np.unique(yv[idx])) < 2:
            continue
        try:
            out.append(roc_auc_score(yv[idx], p[idx], sample_weight=wv[idx]))
        except Exception:
            pass
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), len(out)


pp = m0.predict_proba(Xpre)[:, 1]
po = m0.predict_proba(Xpost)[:, 1]
cis = []
for tag, df, p in [("pre", pre, pp), ("post", post, po)]:
    lo, hi, nb = boot_ci(df, p)
    cis.append(dict(set=tag, wauc=round(wauc(df["dep"], p, df["W"]), 5),
                    ci_lo=round(lo, 5), ci_hi=round(hi, 5), n_boot=nb,
                    note="기준선 17변수 전이모델, PSU/strata 복원추출 2000회"))
    print(json.dumps(cis[-1], ensure_ascii=False), flush=True)
pd.DataFrame(cis).to_csv(os.path.join(RES, "pathG_03_bootci.csv"),
                         index=False, encoding="utf-8-sig")
dump("pathG_03_refine.csv")
print("TOTAL %.1f min" % ((time.time() - T0) / 60))
