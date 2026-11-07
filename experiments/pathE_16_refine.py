# -*- coding: utf-8 -*-
"""경로E-16: (a) E12에서 메모리오류로 누락된 실험, (b) 중첩(positivity) 진단,
(c) living_alone 제외 도메인 판별기 기반 중요도가중 재시도, (d) 유망 조합.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathE_10_common import *

t0 = time.time()
k, pre, post = load()
y = k["dep"].astype(int).values
W = k["W"].to_numpy(float)
itr, iho = train_test_split(np.arange(len(k)), test_size=0.2, stratify=y, random_state=SEED)
is_tr = np.zeros(len(k), bool); is_tr[itr] = True


def income_q_trainonly(df, is_tr):
    iq = np.full(len(df), np.nan); yr = df["year"].values; inc = df["income_raw"].to_numpy(float)
    for yv in np.unique(yr):
        m = yr == yv; v = inc[m & is_tr]; v = v[~np.isnan(v)]
        if len(v) < 50: continue
        e = np.unique(np.quantile(v, [0, .2, .4, .6, .8, 1.0]))
        if len(e) < 3: continue
        ii = np.where(m)[0]
        bb = np.clip(np.digitize(inc[ii], e[1:-1]), 0, len(e) - 2).astype(float)
        bb[np.isnan(inc[ii])] = np.nan; iq[ii] = bb
    return iq


k = k.copy(); k["income_q"] = income_q_trainonly(k, is_tr)
base = fit_eval(k, pre, post, itr, iho)
BASE = (base[0], base[1], base[2])
rec("E16_base(=C)", *BASE, True, "참조 기준선")

# ---------- (a) E12 누락분 ------------------------------------------------
r = fit_eval(k, pre, post, itr, iho, params=dict(max_depth=2, num_leaves=4, n_estimators=1200))
rec("E16_a_reg_depth2", r[0], r[1], r[2], True, "max_depth=2, 1200 trees", BASE)

rng = np.random.default_rng(SEED)
sub = rng.choice(itr, size=600000, replace=False)  # 메모리 여유 위해 축소
for prob, tag in [(0.15, "p15"), (0.30, "p30")]:
    Xa = k.iloc[sub][FEATURES].copy()
    reps = [Xa]
    for _ in range(2):
        Z = Xa.copy()
        M = rng.random(Z.shape) < prob
        Z = Z.mask(pd.DataFrame(M, index=Z.index, columns=Z.columns))
        reps.append(Z)
    Xaug = pd.concat(reps, axis=0); yaug = np.tile(y[sub], 3)
    pipe = Pipeline([("pre", make_pre()), ("clf", lgb.LGBMClassifier(**LGB_PARAMS))])
    pipe.fit(Xaug, yaug)
    pi = pipe.predict_proba(k.iloc[iho][FEATURES])[:, 1]
    pp = pipe.predict_proba(pre[FEATURES])[:, 1]
    ps = pipe.predict_proba(post[FEATURES])[:, 1]
    rec(f"E16_a_feature_mask_aug_{tag}", wauc(y[iho], pi, W[iho]), wauc(pre.dep, pp, pre.W),
        wauc(post.dep, ps, post.W), True, f"랜덤 마스킹 증강 p={prob} x3 (train 600k)", BASE)
    del Xaug, reps, Xa

# 600k 서브샘플 자체의 효과 확인용 대조군
r = fit_eval(k, pre, post, sub, iho)
rec("E16_a_sub600k_control", r[0], r[1], r[2], True, "마스킹 증강의 대조군(동일 600k, 증강없음)", BASE)

# ---------- (b) 중첩(positivity) 진단 -------------------------------------
def domain_auc(cols, tgt, tag, n_k=150000):
    rr = np.random.default_rng(SEED)
    s = rr.choice(itr, size=n_k, replace=False)
    X = pd.concat([k.iloc[s][cols], tgt[cols]], axis=0)
    d = np.r_[np.zeros(n_k), np.ones(len(tgt))]
    num = [c for c in cols if c in NUM]; bn = [c for c in cols if c in BIN]; ct = [c for c in cols if c in CAT]
    dm = Pipeline([("pre", make_pre(num, bn, ct)),
                   ("clf", lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                              num_leaves=31, random_state=SEED, n_jobs=-1, verbose=-1))])
    dm.fit(X, d)
    p = dm.predict_proba(X)[:, 1]
    a = roc_auc_score(d, p)
    print(f"  domain AUC [{tag}] = {a:.4f}", flush=True)
    return a, dm, s


diag = []
F_noLA = [c for c in FEATURES if c != "living_alone"]
for tname, tgt in [("pre", pre), ("post", post)]:
    for cols, ctag in [(FEATURES, "all17"), (F_noLA, "no_living_alone")]:
        a, _, _ = domain_auc(cols, tgt, f"{tname}_{ctag}")
        diag.append(dict(target=tname, featureset=ctag, domain_auc=round(a, 4)))
    for c in FEATURES:
        a, _, _ = domain_auc([c], tgt, f"{tname}_only_{c}")
        diag.append(dict(target=tname, featureset=f"only_{c}", domain_auc=round(a, 4)))
pd.DataFrame(diag).to_csv(os.path.join(RES, "pathE_16_overlap_diag.csv"),
                          index=False, encoding="utf-8-sig")

# ---------- (c) living_alone 제외 도메인판별기 기반 중요도가중 -------------
def dratio(tgt, cols):
    rr = np.random.default_rng(SEED)
    s = rr.choice(itr, size=200000, replace=False)
    Xk = k.iloc[s][cols]; Xt = tgt[cols]
    X = pd.concat([Xk, Xt], axis=0)
    d = np.r_[np.zeros(len(Xk)), np.ones(len(Xt))]
    swd = np.r_[np.ones(len(Xk)),
                tgt["W"].to_numpy(float) / tgt["W"].mean() * (len(Xk) / len(Xt))]
    num = [c for c in cols if c in NUM]; bn = [c for c in cols if c in BIN]; ct = [c for c in cols if c in CAT]
    dm = Pipeline([("pre", make_pre(num, bn, ct)),
                   ("clf", lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                              num_leaves=31, random_state=SEED, n_jobs=-1, verbose=-1))])
    dm.fit(X, d, clf__sample_weight=swd)
    p = np.clip(dm.predict_proba(k.iloc[itr][cols])[:, 1], 1e-4, 1 - 1e-4)
    return p / (1 - p)


w = dratio(pre, F_noLA)
print(f"  w(noLA,pre): mean={w.mean():.4f} p50={np.median(w):.4f} p99={np.percentile(w,99):.3f} max={w.max():.1f}", flush=True)
for q, qt in [(99, "clip99"), (95, "clip95")]:
    ww = np.minimum(w, np.percentile(w, q)); ww = ww / ww.mean()
    r = fit_eval(k, pre, post, itr, iho, sw=ww)
    rec(f"E16_c_iw_noLA_pre_{qt}", r[0], r[1], r[2], False,
        "living_alone 제외 도메인판별기 밀도비 가중. NHANES pre 공변량 사용 → §2.10 위반", BASE)
ww = np.sqrt(np.minimum(w, np.percentile(w, 99))); ww = ww / ww.mean()
r = fit_eval(k, pre, post, itr, iho, sw=ww)
rec("E16_c_iw_noLA_pre_sqrt", r[0], r[1], r[2], False, "sqrt 완화", BASE)

# 트리밍
for lo, tag in [(25, "top75"), (50, "top50"), (75, "top25")]:
    sel = itr[w >= np.percentile(w, lo)]
    r = fit_eval(k, pre, post, sel, iho)
    rec(f"E16_c_trim_noLA_{tag}", r[0], r[1], r[2], False,
        f"밀도비 상위구간 학습 n={len(sel)} → §2.10 위반", BASE)

dump("pathE_16_refine.csv")
print(f"[{time.time()-t0:.0f}s]")
