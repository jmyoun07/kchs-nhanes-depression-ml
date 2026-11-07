# -*- coding: utf-8 -*-
"""경로E-13: 회색지대 — NHANES **공변량 분포만**(라벨 미사용) 사용하는 보정.

  (1) 중요도 가중 (density-ratio importance weighting)
  (2) 표본 정합/트리밍 (overlap region)
  (3) 대상집단 내 분위수 정렬 (within-population quantile alignment)
  (4) 대상집단 내 표준화

§2.10 판정: 외부검증 데이터(공변량)를 모델 적합에 사용하므로 '재적합 없이 적용' 원칙을
엄밀히는 위반한다. 라벨은 쓰지 않았음을 note에 명시. 모두 protocol_compliant=False로 기록.
단 (1)(2)의 'pre만 사용 → post로 평가' 변형은 post 기준으로는 진짜 외부이므로 별도 표기.
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
rec("E13_base(=C)", *BASE, True, "참조 기준선")

# ================= (1) 중요도 가중 =========================================
def density_ratio(target_df, tag, n_k=200000, clip=None):
    """KCHS train 표본 vs target 공변량 판별기 → w = p/(1-p) (라벨 미사용)."""
    rng = np.random.default_rng(SEED)
    sub = rng.choice(itr, size=min(n_k, len(itr)), replace=False)
    Xk = k.iloc[sub][FEATURES]
    Xt = target_df[FEATURES]
    X = pd.concat([Xk, Xt], axis=0)
    d = np.r_[np.zeros(len(Xk)), np.ones(len(Xt))]
    # target은 복합표본 가중치로, KCHS는 균등가중으로 (모집단 분포 대조)
    swd = np.r_[np.ones(len(Xk)), target_df["W"].to_numpy(float) / target_df["W"].mean() * (len(Xk) / len(Xt))]
    dm = Pipeline([("pre", make_pre()),
                   ("clf", lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                              num_leaves=31, random_state=SEED,
                                              n_jobs=-1, verbose=-1))])
    dm.fit(X, d, clf__sample_weight=swd)
    p = dm.predict_proba(k.iloc[itr][FEATURES])[:, 1]
    p = np.clip(p, 1e-4, 1 - 1e-4)
    w = p / (1 - p)
    auc_d = roc_auc_score(d, dm.predict_proba(X)[:, 1], sample_weight=swd)
    print(f"  [{tag}] domain-classifier AUC={auc_d:.4f} w: mean={w.mean():.3f} p99={np.percentile(w,99):.2f} max={w.max():.1f}")
    return w, auc_d


for tgt, tname in [(pre, "pre"), (pd.concat([pre, post]), "prepost")]:
    w, aucd = density_ratio(tgt, tname)
    for q, qt in [(99, "clip99"), (95, "clip95"), (None, "noclip")]:
        ww = w.copy()
        if q is not None:
            ww = np.minimum(ww, np.percentile(w, q))
        ww = ww / ww.mean()
        r = fit_eval(k, pre, post, itr, iho, sw=ww)
        rec(f"E13_1_iw_{tname}_{qt}", r[0], r[1], r[2], False,
            f"중요도가중(라벨 미사용). 도메인판별AUC={aucd:.3f}. NHANES {tname} 공변량 사용 → §2.10 위반",
            BASE)
    # 정규화된 IW(제곱근 완화)
    ww = np.sqrt(np.minimum(w, np.percentile(w, 99))); ww = ww / ww.mean()
    r = fit_eval(k, pre, post, itr, iho, sw=ww)
    rec(f"E13_1_iw_{tname}_sqrt", r[0], r[1], r[2], False, "중요도가중 sqrt 완화", BASE)

# ================= (2) 트리밍 / 정합 =======================================
w, _ = density_ratio(pre, "pre_trim")
for lo, hi, tag in [(10, 100, "top90"), (25, 100, "top75"), (50, 100, "top50")]:
    thr = np.percentile(w, lo)
    sel = itr[w >= thr]
    r = fit_eval(k, pre, post, sel, iho)
    rec(f"E13_2_trim_{tag}", r[0], r[1], r[2], False,
        f"밀도비 상위 구간만 학습 n={len(sel)}. NHANES pre 공변량 사용 → §2.10 위반", BASE)

# ================= (3) 대상집단 내 분위수 정렬 =============================
def wecdf_transform(vals, ref_vals, ref_w):
    """ref 분포(가중) 내에서의 백분위로 변환."""
    o = np.argsort(ref_vals, kind="mergesort")
    rv = ref_vals[o]; rw = ref_w[o]
    cw = np.cumsum(rw) - 0.5 * rw
    cw = cw / rw.sum()
    out = np.interp(vals, rv, cw, left=0.0, right=1.0)
    return out


def rank_align(df, wcol="W", cols=NUM):
    out = df.copy()
    for c in cols:
        v = df[c].to_numpy(float)
        m = ~np.isnan(v)
        ww = df[wcol].to_numpy(float)[m]
        r = np.full(len(v), np.nan)
        r[m] = wecdf_transform(v[m], v[m], ww)
        out[c] = r
    return out


ka = rank_align(k); prea = rank_align(pre); posta = rank_align(post)
r = fit_eval(k, pre, post, itr, iho, kdf=ka, predf=prea, postf=posta)
rec("E13_3_quantile_align_all_num", r[0], r[1], r[2], False,
    "수치형 6개를 각 집단 내 가중 백분위로 변환. NHANES 공변량 분포 사용 → §2.10 위반", BASE)

# bmi/sleep만 (가장 이동이 큰 변수)
for cols, tag in [(["bmi"], "bmi"), (["bmi", "sleep_hrs"], "bmi_sleep"),
                  (["bmi", "sleep_hrs", "age"], "bmi_sleep_age"),
                  (["srh", "oral_health", "income_q"], "ordinals")]:
    ka = rank_align(k, cols=cols); prea = rank_align(pre, cols=cols); posta = rank_align(post, cols=cols)
    r = fit_eval(k, pre, post, itr, iho, kdf=ka, predf=prea, postf=posta)
    rec(f"E13_3_quantile_align_{tag}", r[0], r[1], r[2], False,
        f"{cols} 집단내 백분위 변환 → §2.10 위반", BASE)

# ================= (4) 대상집단 내 표준화 ==================================
def zself(df, cols=NUM):
    out = df.copy()
    ww = df["W"].to_numpy(float)
    for c in cols:
        v = df[c].to_numpy(float); m = ~np.isnan(v)
        mu = np.average(v[m], weights=ww[m])
        sd = np.sqrt(np.average((v[m] - mu) ** 2, weights=ww[m]))
        out[c] = (v - mu) / (sd if sd > 0 else 1)
    return out


ka = zself(k); prea = zself(pre); posta = zself(post)
r = fit_eval(k, pre, post, itr, iho, kdf=ka, predf=prea, postf=posta)
rec("E13_4_within_pop_zscore", r[0], r[1], r[2], False,
    "수치형을 각 집단 자체 평균/표준편차로 z화 → §2.10 위반", BASE)

dump("pathE_13_gray.csv")
print(f"[{time.time()-t0:.0f}s]")
