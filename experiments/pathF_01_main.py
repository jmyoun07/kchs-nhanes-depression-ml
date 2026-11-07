# -*- coding: utf-8 -*-
"""
pathF_01_main.py — 경로F: 타깃·표본 정의 재검토 실험 배터리.

고정: 내부 홀드아웃 iho(20+, PHQ9>=10, 전 연도, n=409,702) — 모든 실험 공통.
가변: 학습표본 정의 / 결과 컷오프 / 불균형 처리 / 표본크기 / 이상치 처리.
NHANES는 predict만(§2.10). 외부 수치는 기록만 하며 모델 선택 근거로 쓰지 않음.

블록:
 N) 노이즈 바닥 — 동일 사양·서브샘플 seed만 바꿈 (해석 기준선)
 S) 학습표본 크기
 A) 연령 하한/상한
 Y) 연도 서브셋
 T) 결과 컷오프 (>=5 / >=10 / >=15 / 연속점수 회귀)
 I) 클래스 불균형
 O) 이상치/극단값 처리
출력: results/pathF_01_main.csv
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from pathF_common import (RES, SEED, FEATURES, NUM, Bench, make_pipe, make_pre, wauc)
from sklearn.pipeline import Pipeline

QC = []
def qc(s):
    QC.append(str(s)); print(s); sys.stdout.flush()

B = Bench()
k, pre, post = B.k, B.pre, B.post
qc(f"로드 완료 kchs={k.shape} pre={pre.shape} post={post.shape} | itr={len(B.itr):,} iho={len(B.iho):,}")

ROWS = []
OUT = os.path.join(RES, "pathF_01_main.csv")

# 연령 밴드별 외부 평가(부차 지표)
def band_auc(mask_k, mask_pre, mask_post, pi, pp, po):
    out = {}
    try:
        out["internal_band"] = wauc(B.y[B.iho][mask_k], pi[mask_k], B.W[B.iho][mask_k])
    except Exception:
        out["internal_band"] = np.nan
    try:
        out["pre_band"] = wauc(B.ypre[mask_pre], pp[mask_pre], B.wpre[mask_pre])
    except Exception:
        out["pre_band"] = np.nan
    try:
        out["post_band"] = wauc(B.ypost[mask_post], po[mask_post], B.wpost[mask_post])
    except Exception:
        out["post_band"] = np.nan
    return out


def record(r, base=None, block="", compliant=True, extra_note=""):
    r = dict(r)
    r["block"] = block
    r["protocol_compliant"] = compliant
    if base is not None:
        r["d_internal"] = r["internal"] - base["internal"]
        r["d_pre"] = r["ext_pre"] - base["ext_pre"]
        r["d_post"] = r["ext_post"] - base["ext_post"]
    else:
        r["d_internal"] = r["d_pre"] = r["d_post"] = 0.0
    if extra_note:
        r["note"] = (r.get("note", "") + " | " + extra_note).strip(" |")
    ROWS.append(r)
    pd.DataFrame(ROWS).to_csv(OUT, index=False, encoding="utf-8-sig")
    s = (f"[{block}] {r['name']:38s} n={r['train_n']:>9,d} "
         f"내부={r['internal']:.4f} pre={r['ext_pre']:.4f} post={r['ext_post']:.4f}")
    if base is not None:
        s += f"  Δ={r['d_internal']:+.4f}/{r['d_pre']:+.4f}/{r['d_post']:+.4f}"
    qc(s)
    return r


def fit_eval(tag, tr_idx, label=None, sw=None, est=None,
             is_reg=False, note="", **lgbkw):
    """tr_idx로 학습, 고정 iho + NHANES 평가. (전역 k/pre/post를 사용 — 변환 실험 지원)"""
    t0 = time.time()
    X = k[FEATURES]
    if est is None:
        params = dict(n_estimators=600, learning_rate=0.03, num_leaves=31,
                      subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                      n_jobs=-1, verbose=-1)
        params.update(lgbkw)
        est = lgb.LGBMRegressor(**params) if is_reg else lgb.LGBMClassifier(**params)
    pipe = Pipeline([("pre", make_pre()), ("clf", est)])
    ytr = (B.y if label is None else label)[tr_idx]
    if sw is not None:
        pipe.fit(X.iloc[tr_idx], ytr, clf__sample_weight=sw)
    else:
        pipe.fit(X.iloc[tr_idx], ytr)
    if is_reg:
        pi = pipe.predict(X.iloc[B.iho]); pp = pipe.predict(pre[FEATURES]); po = pipe.predict(post[FEATURES])
    else:
        pi = pipe.predict_proba(X.iloc[B.iho])[:, 1]
        pp = pipe.predict_proba(pre[FEATURES])[:, 1]
        po = pipe.predict_proba(post[FEATURES])[:, 1]
    r = dict(name=tag, train_n=len(tr_idx),
             internal=wauc(B.y[B.iho], pi, B.W[B.iho]),
             ext_pre=wauc(B.ypre, pp, B.wpre),
             ext_post=wauc(B.ypost, po, B.wpost),
             secs=round(time.time() - t0, 1), note=note)
    return r, (pi, pp, po)


# ============================================================
# 0) 기준선 (주분석 정의: 20+, PHQ9>=10, 전 연도, 전체 train)
# ============================================================
qc("\n########## 0) 기준선 ##########")
base, pbase = fit_eval("BASE_20plus_dep10_allyears", B.itr, note="주분석 정의")
record(base, None, "0_BASE", True, "protocol primary definition")
BASE = base

# ============================================================
# N) 노이즈 바닥 — 동일 사양, 학습 서브샘플 seed만 변경 (n=500k)
# ============================================================
qc("\n########## N) 노이즈 바닥 (n=500k, seed 5개) ##########")
noise = []
for s in [42, 7, 2024, 1234, 99]:
    idx = np.random.RandomState(s).choice(B.itr, size=500000, replace=False)
    r, _ = fit_eval(f"NOISE_500k_seed{s}", idx, note="동일사양 서브샘플 seed만 변경")
    record(r, BASE, "N_noise", True)
    noise.append(r)
nz = pd.DataFrame(noise)
qc(f"  → 노이즈 폭: 내부 sd={nz['internal'].std():.4f} range={nz['internal'].max()-nz['internal'].min():.4f} | "
   f"pre sd={nz['ext_pre'].std():.4f} range={nz['ext_pre'].max()-nz['ext_pre'].min():.4f} | "
   f"post sd={nz['ext_post'].std():.4f} range={nz['ext_post'].max()-nz['ext_post'].min():.4f}")

# LightGBM seed(모델 seed)만 변경
qc("\n--- 모델 seed만 변경 (전체 train) ---")
for s in [7, 2024]:
    r, _ = fit_eval(f"NOISE_modelseed{s}", B.itr, random_state=s, note="LGBM random_state만 변경")
    record(r, BASE, "N_noise", True)

# ============================================================
# S) 학습표본 크기
# ============================================================
qc("\n########## S) 학습표본 크기 ##########")
for n in [50000, 100000, 200000, 500000, 1000000]:
    idx = np.random.RandomState(SEED).choice(B.itr, size=n, replace=False)
    r, _ = fit_eval(f"SIZE_n{n//1000}k", idx)
    record(r, BASE, "S_size", True)
qc(f"  (전체 {len(B.itr):,} = 기준선)")

# ============================================================
# A) 연령 하한/상한
# ============================================================
qc("\n########## A) 연령 하한/상한 (학습표본 한정; 평가는 20+ 고정) ##########")
age_k = k["age"].to_numpy(float)
age_iho = age_k[B.iho]
age_pre = pre["age"].to_numpy(float); age_post = post["age"].to_numpy(float)
AGE_SETS = [("A_train_30plus", 30, 200), ("A_train_40plus", 40, 200),
            ("A_train_20to64", 20, 65), ("A_train_20to79", 20, 80),
            ("A_train_20to39", 20, 40), ("A_train_65plus", 65, 200)]
for tag, lo, hi in AGE_SETS:
    m = (age_k[B.itr] >= lo) & (age_k[B.itr] < hi)
    idx = B.itr[m]
    r, (pi, pp, po) = fit_eval(tag, idx, note=f"학습 {lo}<=age<{hi}")
    bd = band_auc((age_iho >= lo) & (age_iho < hi),
                  (age_pre >= lo) & (age_pre < hi),
                  (age_post >= lo) & (age_post < hi), pi, pp, po)
    r.update(bd)
    record(r, BASE, "A_age", True, "밴드값=동일 연령대만 평가(부차)")

# 연령가중 학습(20-39 비중 축소/확대)는 표본정의 민감도로 별도
qc("\n--- 연령 재가중 학습 (20-39 가중치 조정) ---")
for wfac, tag in [(0.5, "A_reweight_2039_x0.5"), (2.0, "A_reweight_2039_x2.0"), (0.0, "A_drop_2039")]:
    if wfac == 0.0:
        idx = B.itr[age_k[B.itr] >= 40]
        r, _ = fit_eval(tag, idx, note="20-39 제외")
    else:
        idx = B.itr
        swv = np.where(age_k[idx] < 40, wfac, 1.0)
        r, _ = fit_eval(tag, idx, sw=swv, note=f"20-39 sample_weight x{wfac}")
    record(r, BASE, "A_age", True)

# ============================================================
# Y) 연도 서브셋
# ============================================================
qc("\n########## Y) 연도 서브셋 학습 ##########")
yr_k = k["year"].to_numpy(int)
YEAR_SETS = [("Y_2017_2019", [2017, 2018, 2019]),
             ("Y_2020_2022", [2020, 2021, 2022]),
             ("Y_2021_2025_postpandemic", [2021, 2022, 2023, 2024, 2025]),
             ("Y_2023_2025_recent", [2023, 2024, 2025]),
             ("Y_2024_2025_latest", [2024, 2025]),
             ("Y_2017_2020_prepandemic", [2017, 2018, 2019, 2020])]
for tag, yrs in YEAR_SETS:
    m = np.isin(yr_k[B.itr], yrs)
    r, _ = fit_eval(tag, B.itr[m], note=f"학습연도={yrs}")
    record(r, BASE, "Y_year", True)

# 연도 크기 통제: 최근연도 서브셋과 동일 n의 무작위 표본
qc("\n--- 연도효과 vs 표본크기효과 분리(동일 n 무작위 대조) ---")
for tag, yrs in [("Y_2023_2025_recent", [2023, 2024, 2025]), ("Y_2017_2019", [2017, 2018, 2019])]:
    n = int(np.isin(yr_k[B.itr], yrs).sum())
    idx = np.random.RandomState(SEED).choice(B.itr, size=n, replace=False)
    r, _ = fit_eval(f"Yctrl_random_n{n//1000}k", idx, note=f"{tag} 대조(무작위 동일 n)")
    record(r, BASE, "Y_year", True)

# ============================================================
# T) 결과 컷오프 민감도 (학습 라벨 변경, 평가는 항상 >=10)
# ============================================================
qc("\n########## T) 결과 컷오프 (학습만 변경, 평가는 항상 PHQ9>=10) ##########")
phq = k["phq_sum"].to_numpy(float)
for cut in [3, 5, 7, 15]:
    lab = (phq >= cut).astype(int)
    r, _ = fit_eval(f"T_trainlabel_ge{cut}", B.itr, label=lab,
                    note=f"학습라벨 PHQ9>={cut}, 평가 >=10")
    record(r, BASE, "T_cutoff", True)

qc("\n--- 연속점수 회귀 학습 → 순위로 평가 ---")
r, _ = fit_eval("T_regress_phqsum", B.itr, label=phq, is_reg=True,
                note="LGBMRegressor(L2) on PHQ9 총점, 평가 >=10")
record(r, BASE, "T_cutoff", True)
r, _ = fit_eval("T_regress_phqsum_poisson", B.itr, label=phq, is_reg=True,
                objective="poisson", note="Poisson 회귀 on PHQ9 총점")
record(r, BASE, "T_cutoff", True)
r, _ = fit_eval("T_regress_log1p_phqsum", B.itr, label=np.log1p(phq), is_reg=True,
                note="log1p(PHQ9) 회귀")
record(r, BASE, "T_cutoff", True)

qc("\n--- 다중컷오프 앙상블(평균순위) ---")
from scipy.stats import rankdata
preds_multi = {}
for cut in [5, 10, 15]:
    lab = (phq >= cut).astype(int)
    _, pr = fit_eval(f"_tmp_ge{cut}", B.itr, label=lab)
    preds_multi[cut] = pr
def rank_avg(triple_list):
    out = []
    for j in range(3):
        rs = [rankdata(t[j]) / len(t[j]) for t in triple_list]
        out.append(np.mean(rs, axis=0))
    return out
pi, pp, po = rank_avg(list(preds_multi.values()))
r = dict(name="T_ens_rankavg_5_10_15", train_n=len(B.itr),
         internal=wauc(B.y[B.iho], pi, B.W[B.iho]),
         ext_pre=wauc(B.ypre, pp, B.wpre), ext_post=wauc(B.ypost, po, B.wpost),
         secs=np.nan, note="컷오프 5/10/15 모델 평균순위 앙상블")
record(r, BASE, "T_cutoff", True)

# ============================================================
# I) 클래스 불균형
# ============================================================
qc("\n########## I) 클래스 불균형 처리 ##########")
npos = int(B.y[B.itr].sum()); nneg = len(B.itr) - npos
qc(f"  train pos={npos:,} neg={nneg:,} ratio={nneg/npos:.1f}")
for spw in [3, 5, 10, round(nneg / npos, 1)]:
    r, _ = fit_eval(f"I_scale_pos_weight_{spw}", B.itr, scale_pos_weight=float(spw))
    record(r, BASE, "I_imbalance", True)
r, _ = fit_eval("I_is_unbalance", B.itr, is_unbalance=True)
record(r, BASE, "I_imbalance", True)

qc("\n--- 언더샘플링 ---")
rs = np.random.RandomState(SEED)
pos_idx = B.itr[B.y[B.itr] == 1]; neg_idx = B.itr[B.y[B.itr] == 0]
for ratio, tag in [(1, "I_undersample_1to1"), (4, "I_undersample_1to4"), (10, "I_undersample_1to10")]:
    take = min(len(neg_idx), ratio * len(pos_idx))
    idx = np.concatenate([pos_idx, rs.choice(neg_idx, size=take, replace=False)])
    r, _ = fit_eval(tag, idx, note=f"음성:양성={ratio}:1")
    record(r, BASE, "I_imbalance", True)

qc("\n--- focal loss ---")
def focal_obj(gamma=2.0, alpha=1.0):
    def obj(yt, yp):
        p = 1.0 / (1.0 + np.exp(-yp))
        pt = np.where(yt == 1, p, 1 - p)
        a = np.where(yt == 1, alpha, 1.0)
        # d/dz of  -a*(1-pt)^g*log(pt)
        g = gamma
        dpt_dz = np.where(yt == 1, p * (1 - p), -p * (1 - p))
        dL_dpt = a * ((1 - pt) ** g) * (-1.0 / np.clip(pt, 1e-9, 1)) + \
                 a * (-g) * ((1 - pt) ** (g - 1)) * np.log(np.clip(pt, 1e-9, 1))
        grad = dL_dpt * dpt_dz
        eps = 1e-4
        pt2 = np.clip(pt + eps, 1e-9, 1 - 1e-9)
        dL_dpt2 = a * ((1 - pt2) ** g) * (-1.0 / pt2) + \
                  a * (-g) * ((1 - pt2) ** (g - 1)) * np.log(pt2)
        hess = np.abs((dL_dpt2 - dL_dpt) / eps * dpt_dz ** 2) + 1e-6
        return grad, hess
    return obj

for gm in [1.0, 2.0]:
    est = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31,
                             subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                             n_jobs=-1, verbose=-1, objective=focal_obj(gamma=gm))
    r, _ = fit_eval(f"I_focal_gamma{gm}", B.itr, est=est, note="custom focal objective")
    record(r, BASE, "I_imbalance", True)

# ============================================================
# O) 이상치/극단값 처리
# ============================================================
qc("\n########## O) 이상치/극단값 처리 ##########")
CLIPVARS = ["bmi", "sleep_hrs", "age"]

def clipped_frames(qlo, qhi, vars_=CLIPVARS):
    """KCHS train 분위로 cutpoint 산출 → KCHS 전체·NHANES에 동일 적용."""
    kk = k.copy(); pp_ = pre.copy(); oo = post.copy()
    cps = {}
    for v in vars_:
        tv = k[v].to_numpy(float)[B.itr]
        lo, hi = np.nanquantile(tv, [qlo, qhi])
        cps[v] = (float(lo), float(hi))
        for d in (kk, pp_, oo):
            d[v] = np.clip(d[v].to_numpy(float), lo, hi)
    return kk, pp_, oo, cps

_k_orig, _pre_orig, _post_orig = k, pre, post
for qlo, qhi, tag in [(0.01, 0.99, "O_winsor_1_99"), (0.005, 0.995, "O_winsor_0.5_99.5"),
                      (0.025, 0.975, "O_winsor_2.5_97.5")]:
    kk, pp_, oo, cps = clipped_frames(qlo, qhi)
    k, pre, post = kk, pp_, oo
    globals()['k'], globals()['pre'], globals()['post'] = kk, pp_, oo
    r, _ = fit_eval(tag, B.itr, note=f"bmi/sleep/age winsor {cps}")
    record(r, BASE, "O_outlier", True)
k, pre, post = _k_orig, _pre_orig, _post_orig
globals()['k'], globals()['pre'], globals()['post'] = _k_orig, _pre_orig, _post_orig

qc("\n--- 극단 BMI/수면 행 제거(학습만) ---")
bmi = k["bmi"].to_numpy(float); slp = k["sleep_hrs"].to_numpy(float)
m = ~(((bmi < 13) | (bmi > 50)) | ((slp < 2) | (slp > 14)))
m = np.where(np.isnan(bmi) | np.isnan(slp), True, m)
idx = B.itr[m[B.itr]]
r, _ = fit_eval("O_drop_extreme_rows", idx, note="BMI<13/>50 또는 수면<2/>14 학습행 제거")
record(r, BASE, "O_outlier", True)

qc("\n--- 수치형 분위변환(quantile rank) ---")
kq = k.copy(); pq = pre.copy(); oq = post.copy()
for v in ["age", "sleep_hrs", "bmi"]:
    tv = k[v].to_numpy(float)[B.itr]; tv = tv[~np.isnan(tv)]
    grid = np.nanquantile(tv, np.linspace(0, 1, 101))
    grid = np.unique(grid)
    for d in (kq, pq, oq):
        x = d[v].to_numpy(float)
        z = np.searchsorted(grid, x).astype(float) / max(len(grid), 1)
        z[np.isnan(x)] = np.nan
        d[v] = z
k, pre, post = kq, pq, oq
globals()['k'], globals()['pre'], globals()['post'] = kq, pq, oq
r, _ = fit_eval("O_quantile_transform", B.itr, note="age/sleep/bmi를 KCHS train 분위순위로 변환")
record(r, BASE, "O_outlier", True)
k, pre, post = _k_orig, _pre_orig, _post_orig
globals()['k'], globals()['pre'], globals()['post'] = _k_orig, _pre_orig, _post_orig

# ============================================================
# H) 연령 topcode 조화 (NHANES는 80+를 모두 80으로 코딩; KCHS는 실연령 최대 110)
#    → NHANES '코딩 규약'만 사용(라벨 미사용) 하므로 §2.10 준수.
# ============================================================
qc("\n########## H) 연령 topcode 조화 (KCHS age를 80에서 절단) ##########")
kh = k.copy()
kh["age"] = np.clip(kh["age"].to_numpy(float), None, 80.0)
k, pre, post = kh, _pre_orig, _post_orig
globals()['k'], globals()['pre'], globals()['post'] = kh, _pre_orig, _post_orig
r, _ = fit_eval("H_age_topcode80", B.itr, note="KCHS age=min(age,80) — NHANES 코딩 규약과 정렬")
record(r, BASE, "H_topcode", True, "NHANES 라벨 미사용, 코딩규약만 반영")

# topcode + 학습 20-79 한정
age_kh = _k_orig["age"].to_numpy(float)
idx79 = B.itr[age_kh[B.itr] < 80]
r, _ = fit_eval("H_topcode80_train20to79", idx79, note="topcode + 학습 20<=age<80")
record(r, BASE, "H_topcode", True)
k, pre, post = _k_orig, _pre_orig, _post_orig
globals()['k'], globals()['pre'], globals()['post'] = _k_orig, _pre_orig, _post_orig

pd.DataFrame(ROWS).to_csv(OUT, index=False, encoding="utf-8-sig")
open(os.path.join(RES, "pathF_01_main_qc.txt"), "w", encoding="utf-8").write("\n".join(QC))
qc(f"\n완료 — {OUT}")
