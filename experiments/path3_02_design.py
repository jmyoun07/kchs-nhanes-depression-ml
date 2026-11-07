# -*- coding: utf-8 -*-
"""
path3_02_design.py — 경로3 블록 A/I/O.

A) 연령 하한·상한·topcode·재가중·층화학습
I) 클래스 불균형 (scale_pos_weight / is_unbalance / 언더·오버샘플 / focal loss)
O) 이상치 처리 (winsorize / 타당범위 clip / 결측화)

내부 홀드아웃 ho(100k) 고정. NHANES는 predict만(§2.10).
회색지대(외부 공변량 분포 사용)는 protocol_compliant=False.
출력: results/path3_02_design.csv
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from path3_common import (Bench, Recorder, RES, SEED, FEATURES, NUM, make_pre, wauc)

B = Bench()
R = Recorder(os.path.join(RES, "path3_02_design.csv"))
N500 = 500000
age_pool = B.age[B.pool]


def draw(elig, n=N500, seed=SEED):
    rng = np.random.RandomState(seed)
    if len(elig) <= n:
        return np.array(elig)
    return rng.choice(elig, size=n, replace=False)


# 기준선 + 크기통제 기준(POOL 무작위 500k)
base, _, _ = B.run("BASE_spec_500k", B.tr, note="과제지정 기준선")
R.base = base
R.add(base, "0_BASE", True)
ref, _, _ = B.run("REF_pool500k_s42", draw(B.pool, N500, SEED), note="크기통제 기준")
R.add(ref, "0_REF", True)
REF = ref

# ============================================================
# A) 연령
# ============================================================
R.log("\n########## A) 연령 하한/상한 (n=500k 통제, 평가는 20+ 고정) ##########")
R.log(f"  주의: KCHS 최소연령={B.age.min():.0f} → 19+ 학습표본은 존재하지 않음(민감도 불가)")
AGE_SETS = [("A_20plus", 20, 200), ("A_25plus", 25, 200), ("A_30plus", 30, 200),
            ("A_40plus", 40, 200), ("A_50plus", 50, 200),
            ("A_20to64", 20, 65), ("A_20to79", 20, 80), ("A_20to84", 20, 85)]
for tag, lo, hi in AGE_SETS:
    elig = B.pool[(age_pool >= lo) & (age_pool < hi)]
    idx = draw(elig, N500, SEED)
    r, _, _ = B.run(tag, idx, note=f"학습 {lo}<=age<{hi}, n<=500k")
    R.add(r, "A_age", True, base=REF, extra="델타는 REF 대비")

# --- age topcode 80 (NHANES 공개 코드북의 상한코딩과 스케일 일치) ---
R.log("\n--- 연령 상한코딩 정렬 (KCHS age=min(age,80)) ---")
Xk = B.k[FEATURES].copy()
Xk["age"] = np.minimum(Xk["age"].to_numpy(float), 80.0)
r, _, _ = B.run("A_topcode80", draw(B.pool, N500, SEED), X=Xk, Xho=Xk.iloc[B.ho],
                note="KCHS age를 80에서 상한코딩(NHANES 공개코드북 상한과 스케일 일치). NHANES 데이터/라벨 미사용")
R.add(r, "A_age", True, base=REF, extra="조화(harmonization) 성격; 외부 데이터값 미사용")

# ============================================================
# A2) 연령 재가중 / 층화학습
# ============================================================
R.log("\n########## A2) 연령 재가중 / 층화학습 ##########")
tr500 = draw(B.pool, N500, SEED)
BANDS = np.arange(20, 105, 5)


def band_of(a):
    return np.clip(np.digitize(a, BANDS) - 1, 0, len(BANDS) - 2)


b_tr = band_of(B.age[tr500])
nb = len(BANDS) - 1

# (a) 내부 준거: KCHS 설계가중(W) 연령분포에 맞춤 (compliant)
tgt = np.bincount(band_of(B.age), weights=B.W, minlength=nb)[:nb]
tgt = tgt / tgt.sum()
cur = np.bincount(b_tr, minlength=nb)[:nb].astype(float)
cur = cur / cur.sum()
w = np.where(cur[b_tr] > 0, tgt[b_tr] / np.maximum(cur[b_tr], 1e-12), 0.0)
r, _, _ = B.run("A2_rw_kchs_surveyW_age", tr500, sw=w / w.mean(),
                note="KCHS 설계가중 연령분포로 재가중(내부정보만)")
R.add(r, "A2_reweight", True, base=REF)

# (b) 연령밴드 균등 (compliant)
w = np.where(cur[b_tr] > 0, 1.0 / np.maximum(cur[b_tr], 1e-12), 0.0)
r, _, _ = B.run("A2_rw_ageband_uniform", tr500, sw=w / w.mean(),
                note="5세 연령밴드 균등 가중(내부정보만)")
R.add(r, "A2_reweight", True, base=REF)

# (c/d) NHANES 연령분포에 맞춤 — GRAY (외부 공변량 분포 사용, 라벨 미사용)
for tag, ext, note in [("A2_rw_match_NHANESpre_age", B.pre, "NHANES pre 연령분포(가중)로 재가중"),
                       ("A2_rw_match_NHANESpost_age", B.post, "NHANES post 연령분포(가중)로 재가중")]:
    ea = ext["age"].to_numpy(float)
    ew = ext["W"].to_numpy(float)
    t = np.bincount(band_of(ea), weights=ew, minlength=nb)[:nb]
    t = t / t.sum()
    w = np.where(cur[b_tr] > 0, t[b_tr] / np.maximum(cur[b_tr], 1e-12), 0.0)
    r, _, _ = B.run(tag, tr500, sw=w / max(w.mean(), 1e-12), note=note)
    R.add(r, "A2_reweight", False, base=REF,
          extra="GRAY: NHANES 공변량 분포 사용(라벨 미사용). 사전에 외부 분포를 알아야 하므로 §2.10 정신에 저촉")

# (e) 연령×성 결합 매칭 — GRAY
sx = B.k["sex"].astype(str).to_numpy()
sx_tr = sx[tr500]
key_tr = b_tr * 10 + (sx_tr == np.unique(sx)[0]).astype(int)
ea = B.pre["age"].to_numpy(float)
esx = B.pre["sex"].astype(str).to_numpy()
key_e = band_of(ea) * 10 + (esx == np.unique(sx)[0]).astype(int)
ew = B.pre["W"].to_numpy(float)
tt = pd.Series(ew).groupby(key_e).sum()
tt = tt / tt.sum()
cc = pd.Series(np.ones(len(key_tr))).groupby(key_tr).sum()
cc = cc / cc.sum()
w = np.array([tt.get(k_, 0.0) / cc.get(k_, 1e-12) for k_ in key_tr])
r, _, _ = B.run("A2_rw_match_NHANESpre_agesex", tr500, sw=w / max(w.mean(), 1e-12),
                note="NHANES pre 연령x성 결합분포로 재가중")
R.add(r, "A2_reweight", False, base=REF, extra="GRAY: 외부 공변량 분포 사용")

# (f) 도메인 분류기 밀도비 가중 (17변수 전체) — GRAY
R.log("\n--- 도메인 분류기 밀도비 가중 (GRAY) ---")
Xd = pd.concat([B.k[FEATURES].iloc[tr500], B.pre[FEATURES]], axis=0)
yd = np.r_[np.zeros(len(tr500)), np.ones(len(B.pre))]
dpipe = Pipeline([("pre", make_pre()),
                  ("clf", lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                             num_leaves=31, random_state=SEED,
                                             n_jobs=-1, verbose=-1))])
dpipe.fit(Xd, yd)
pd_ = dpipe.predict_proba(B.k[FEATURES].iloc[tr500])[:, 1]
R.log(f"  domain AUC(train fit) = {wauc(yd, dpipe.predict_proba(Xd)[:,1], np.ones(len(yd))):.4f}")
for clipq, tag in [(0.99, "A2_rw_densratio_clip99"), (0.95, "A2_rw_densratio_clip95")]:
    w = pd_ / np.maximum(1 - pd_, 1e-9)
    w = np.minimum(w, np.quantile(w, clipq))
    r, _, _ = B.run(tag, tr500, sw=w / w.mean(),
                    note=f"KCHS vs NHANESpre 도메인분류기 밀도비, {clipq} clip")
    R.add(r, "A2_reweight", False, base=REF, extra="GRAY: 외부 공변량 전체 사용")

# (g) 연령 층화학습 (밴드별 개별 모델) — compliant
R.log("\n--- 연령 층화학습 (밴드별 개별 모델) ---")
CUTS = [(20, 40), (40, 60), (60, 200)]
t0 = time.time()
pi = np.zeros(len(B.ho)); pp = np.zeros(len(B.pre)); po = np.zeros(len(B.post))
aho = B.age[B.ho]; apre = B.pre["age"].to_numpy(float); apost = B.post["age"].to_numpy(float)
for lo, hi in CUTS:
    m = (B.age[tr500] >= lo) & (B.age[tr500] < hi)
    idx = tr500[m]
    p = Pipeline([("pre", make_pre()),
                  ("clf", lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03,
                                             num_leaves=31, subsample=0.8,
                                             colsample_bytree=0.8, random_state=SEED,
                                             n_jobs=-1, verbose=-1))])
    p.fit(B.k[FEATURES].iloc[idx], B.y[idx])
    mh = (aho >= lo) & (aho < hi); pi[mh] = p.predict_proba(B.Xho[mh])[:, 1]
    m1 = (apre >= lo) & (apre < hi); pp[m1] = p.predict_proba(B.Xpre[m1])[:, 1]
    m2 = (apost >= lo) & (apost < hi); po[m2] = p.predict_proba(B.Xpost[m2])[:, 1]
r = dict(name="A2_stratified_3band", train_n=len(tr500),
         internal=wauc(B.yho, pi, B.who), ext_pre=wauc(B.ypre, pp, B.wpre),
         ext_post=wauc(B.ypost, po, B.wpost), secs=round(time.time() - t0, 1),
         note="연령 3밴드 개별 모델, 밴드별 확률 그대로 결합")
R.add(r, "A2_reweight", True, base=REF)

# ============================================================
# I) 클래스 불균형
# ============================================================
R.log("\n########## I) 클래스 불균형 (AUROC=순위지표에 대한 실제 영향) ##########")
ytr = B.y[tr500]
npos, nneg = int(ytr.sum()), int((1 - ytr).sum())
R.log(f"  pos={npos:,} neg={nneg:,} ratio={nneg/npos:.1f}")
for spw in [1, 2, 3, 5, 10, round(nneg / npos, 1)]:
    r, _, _ = B.run(f"I_spw{spw}", tr500, scale_pos_weight=float(spw),
                    note=f"scale_pos_weight={spw}")
    R.add(r, "I_imbalance", True, base=REF)
r, _, _ = B.run("I_is_unbalance", tr500, is_unbalance=True, note="is_unbalance=True")
R.add(r, "I_imbalance", True, base=REF)

R.log("\n--- 언더샘플링(양성 전량 유지, 음성 축소) ---")
rng = np.random.RandomState(SEED)
pos = tr500[ytr == 1]; neg = tr500[ytr == 0]
for ratio in [1, 4, 9]:
    nn = min(len(neg), ratio * len(pos))
    idx = np.r_[pos, rng.choice(neg, nn, replace=False)]
    r, _, _ = B.run(f"I_under_1to{ratio}", idx, note=f"음성:양성={ratio}:1 언더샘플")
    R.add(r, "I_imbalance", True, base=REF)

R.log("\n--- 오버샘플링(양성 복제) ---")
for mult in [3, 5, 10]:
    idx = np.r_[tr500, np.repeat(pos, mult - 1)]
    r, _, _ = B.run(f"I_over_pos_x{mult}", idx, note=f"양성 {mult}배 복제")
    R.add(r, "I_imbalance", True, base=REF)


class FocalLGBM(lgb.LGBMClassifier):
    """focal loss 커스텀 목적함수. predict_proba는 raw score의 sigmoid(순위 동일)."""
    def predict_proba(self, X, **kw):
        raw = self.predict(X, raw_score=True, **kw)
        p = 1.0 / (1.0 + np.exp(-raw))
        return np.vstack([1 - p, p]).T


def focal_obj(alpha, gamma):
    def fl(x, t):
        p = 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        pt = t * p + (1 - t) * (1 - p)
        at = alpha * t + (1 - alpha) * (1 - t)
        return -at * (1 - pt) ** gamma * np.log(pt)

    def obj(y_true, y_pred):
        h = 1e-4
        f1 = fl(y_pred + h, y_true); f0 = fl(y_pred - h, y_true); fc = fl(y_pred, y_true)
        grad = (f1 - f0) / (2 * h)
        hess = (f1 - 2 * fc + f0) / (h * h)
        hess = np.maximum(hess, 1e-6)
        return grad, hess
    return obj


R.log("\n--- focal loss ---")
for a, g in [(0.25, 1.0), (0.25, 2.0), (0.5, 2.0), (0.75, 2.0)]:
    est = FocalLGBM(n_estimators=600, learning_rate=0.03, num_leaves=31,
                    subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                    n_jobs=-1, verbose=-1, objective=focal_obj(a, g))
    r, _, _ = B.run(f"I_focal_a{a}_g{g}", tr500, est=est, note=f"focal alpha={a} gamma={g}")
    R.add(r, "I_imbalance", True, base=REF)

# ============================================================
# O) 이상치 처리
# ============================================================
R.log("\n########## O) 이상치/극단값 처리 (cutpoint는 KCHS train에서만 산출) ##########")


def wins_frames(cols, lo_q, hi_q):
    lo = {c: np.nanquantile(B.k[c].to_numpy(float)[tr500], lo_q) for c in cols}
    hi = {c: np.nanquantile(B.k[c].to_numpy(float)[tr500], hi_q) for c in cols}
    Xk = B.k[FEATURES].copy(); Xp = B.pre[FEATURES].copy(); Xq = B.post[FEATURES].copy()
    for c in cols:
        for D in (Xk, Xp, Xq):
            D[c] = np.clip(D[c].to_numpy(float), lo[c], hi[c])
    return Xk, Xp, Xq, lo, hi


for cols, lq, hq, tag in [(["bmi", "sleep_hrs"], .01, .99, "O_wins_bmisleep_1_99"),
                          (["bmi", "sleep_hrs"], .005, .995, "O_wins_bmisleep_05_995"),
                          (["bmi", "sleep_hrs"], .025, .975, "O_wins_bmisleep_25_975"),
                          (NUM, .01, .99, "O_wins_allnum_1_99")]:
    Xk, Xp, Xq, lo, hi = wins_frames(cols, lq, hq)
    r, _, _ = B.run(tag, tr500, X=Xk, Xho=Xk.iloc[B.ho], Xpre=Xp, Xpost=Xq,
                    note=f"winsorize {cols} [{lq},{hq}] cut={ {c:(round(lo[c],1),round(hi[c],1)) for c in cols} }")
    R.add(r, "O_outlier", True, base=REF)

# 타당범위 clip / 결측화
RANGES = {"bmi": (13.0, 50.0), "sleep_hrs": (3.0, 13.0)}
Xk = B.k[FEATURES].copy(); Xp = B.pre[FEATURES].copy(); Xq = B.post[FEATURES].copy()
for c, (lo, hi) in RANGES.items():
    for D in (Xk, Xp, Xq):
        D[c] = np.clip(D[c].to_numpy(float), lo, hi)
r, _, _ = B.run("O_clip_plausible", tr500, X=Xk, Xho=Xk.iloc[B.ho], Xpre=Xp, Xpost=Xq,
                note="bmi[13,50] sleep[3,13] 타당범위 clip")
R.add(r, "O_outlier", True, base=REF)

Xk = B.k[FEATURES].copy(); Xp = B.pre[FEATURES].copy(); Xq = B.post[FEATURES].copy()
for c, (lo, hi) in RANGES.items():
    for D in (Xk, Xp, Xq):
        v = D[c].to_numpy(float)
        v[(v < lo) | (v > hi)] = np.nan
        D[c] = v
r, _, _ = B.run("O_implausible_to_nan", tr500, X=Xk, Xho=Xk.iloc[B.ho], Xpre=Xp, Xpost=Xq,
                note="타당범위 밖은 결측화 후 median 대체")
R.add(r, "O_outlier", True, base=REF)

R.log(f"\n완료. 저장: {R.out}")
