# -*- coding: utf-8 -*-
"""경로E-12: KCHS 정보만 쓰는(=§2.10 완전 준수) 가중/결측/정규화 전략들.

NHANES는 각 실험당 최종 1회 평가만. 선택은 KCHS 내부 지표로만.
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


k = k.copy()
k["income_q"] = income_q_trainonly(k, is_tr)

base = fit_eval(k, pre, post, itr, iho)
BASE = (base[0], base[1], base[2])
rec("E12_base(=C)", *BASE, True, "참조 기준선")

# ---- 1) KCHS 복합표본 가중치로 학습 -------------------------------------
sw = W[itr] / W[itr].mean()
r = fit_eval(k, pre, post, itr, iho, sw=sw)
rec("E12_1_survey_weighted_train", r[0], r[1], r[2], True,
    "KCHS W로 가중학습(외부 정보 없음)", BASE)

# ---- 2) 최근 연도만 학습 -------------------------------------------------
for yrs, tag in [((2021, 2022, 2023, 2024, 2025), "2021_2025"),
                 ((2022, 2023, 2024, 2025), "2022_2025"),
                 ((2017, 2018, 2019), "2017_2019")]:
    sub = itr[np.isin(k["year"].values[itr], yrs)]
    r = fit_eval(k, pre, post, sub, iho)
    rec(f"E12_2_train_years_{tag}", r[0], r[1], r[2], True,
        f"학습 연도 제한 n={len(sub)}", BASE)

# ---- 3) 연도 리커런시 가중 ----------------------------------------------
yr_tr = k["year"].values[itr].astype(float)
for half, tag in [(3.0, "hl3"), (6.0, "hl6")]:
    swy = 0.5 ** ((2025 - yr_tr) / half)
    r = fit_eval(k, pre, post, itr, iho, sw=swy / swy.mean())
    rec(f"E12_3_year_recency_{tag}", r[0], r[1], r[2], True,
        "최근연도 지수가중", BASE)

# ---- 4) food_insecure 결측 처리 -----------------------------------------
# (a) 결측지표 추가
ka = k.copy(); ka["fi_miss"] = k["food_insecure"].isna().astype(float)
prea = pre.copy(); prea["fi_miss"] = pre["food_insecure"].isna().astype(float)
posta = post.copy(); posta["fi_miss"] = post["food_insecure"].isna().astype(float)
BIN_a = BIN + ["fi_miss"]
r = fit_eval(k, pre, post, itr, iho, feats=NUM + BIN_a + CAT, bin_=BIN_a,
             kdf=ka, predf=prea, postf=posta)
rec("E12_4a_fi_missing_indicator", r[0], r[1], r[2], True, "food_insecure 결측지표", BASE)

# (b) food_insecure 0 대체
kb = k.copy(); kb["food_insecure"] = k["food_insecure"].fillna(0.0)
preb = pre.copy(); preb["food_insecure"] = pre["food_insecure"].fillna(0.0)
postb = post.copy(); postb["food_insecure"] = post["food_insecure"].fillna(0.0)
r = fit_eval(k, pre, post, itr, iho, kdf=kb, predf=preb, postf=postb)
rec("E12_4b_fi_fill0", r[0], r[1], r[2], True, "food_insecure 결측=0", BASE)

# (c) food_insecure 관측 연도만 학습 (2020,2021 제외)
sub = itr[~np.isin(k["year"].values[itr], (2020, 2021))]
r = fit_eval(k, pre, post, sub, iho)
rec("E12_4c_fi_observed_years_only", r[0], r[1], r[2], True,
    f"2020/2021 제외 n={len(sub)}", BASE)

# ---- 5) 수치형 NaN 네이티브(중앙값대체 제거) -----------------------------
class Ident:
    pass

from sklearn.preprocessing import FunctionTransformer
ct_native = ColumnTransformer([
    ("num", "passthrough", NUM),
    ("bin", "passthrough", BIN),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore", drop="first"))]), CAT)])
pipe = Pipeline([("pre", ct_native), ("clf", lgb.LGBMClassifier(**LGB_PARAMS))])
pipe.fit(k.iloc[itr][FEATURES], y[itr])
pi = pipe.predict_proba(k.iloc[iho][FEATURES])[:, 1]
pp = pipe.predict_proba(pre[FEATURES])[:, 1]
ps = pipe.predict_proba(post[FEATURES])[:, 1]
rec("E12_5_numeric_nan_native", wauc(y[iho], pi, W[iho]), wauc(pre.dep, pp, pre.W),
    wauc(post.dep, ps, post.W), True, "수치형 median대체 제거(LGBM 네이티브 NaN)", BASE)

# ---- 6) 단조제약(도메인 사전지식, KCHS만) --------------------------------
# 전처리 후 컬럼 순서: NUM(6) + BIN(8) + onehot(cat)
mono = [0] * 6 + [0] * 8
mono[NUM.index("srh")] = 1          # 나쁠수록(값↑) 우울↑
mono[NUM.index("oral_health")] = 1  # 나쁠수록 우울↑
mono[NUM.index("income_q")] = -1    # 소득↑ 우울↓
n_oh = None
pipe = Pipeline([("pre", make_pre()), ("clf", lgb.LGBMClassifier(**LGB_PARAMS))])
Xtr = make_pre().fit(k.iloc[itr][FEATURES])
n_oh = Xtr.transform(k.iloc[itr[:10]][FEATURES]).shape[1] - 14
mono_full = mono + [0] * n_oh
r = fit_eval(k, pre, post, itr, iho, params=dict(monotone_constraints=mono_full))
rec("E12_6_monotone_srh_oral_income", r[0], r[1], r[2], True,
    "srh/oral_health/income_q 단조제약", BASE)

# ---- 7) 강정규화 (KCHS 내부 LOYO로 선택 예정, 여기선 스캔) ---------------
for tag, p in [("reg_leaves15_l2_50", dict(num_leaves=15, reg_lambda=50)),
               ("reg_leaves8_l2_100", dict(num_leaves=8, reg_lambda=100)),
               ("reg_depth3", dict(max_depth=3, num_leaves=8)),
               ("reg_depth2", dict(max_depth=2, num_leaves=4, n_estimators=1200))]:
    r = fit_eval(k, pre, post, itr, iho, params=p)
    rec(f"E12_7_{tag}", r[0], r[1], r[2], True, str(p), BASE)

# ---- 8) 특징 마스킹 증강(도메인 일반화) ----------------------------------
rng = np.random.default_rng(SEED)
for prob, tag in [(0.15, "p15"), (0.30, "p30")]:
    kk = k.copy()
    Xa = kk.iloc[itr][FEATURES].copy()
    reps = [Xa]
    for _ in range(2):
        Z = Xa.copy()
        M = rng.random(Z.shape) < prob
        Z = Z.mask(pd.DataFrame(M, index=Z.index, columns=Z.columns))
        reps.append(Z)
    Xaug = pd.concat(reps, axis=0)
    yaug = np.tile(y[itr], 3)
    pipe = Pipeline([("pre", make_pre()), ("clf", lgb.LGBMClassifier(**LGB_PARAMS))])
    pipe.fit(Xaug, yaug)
    pi = pipe.predict_proba(k.iloc[iho][FEATURES])[:, 1]
    pp = pipe.predict_proba(pre[FEATURES])[:, 1]
    ps = pipe.predict_proba(post[FEATURES])[:, 1]
    rec(f"E12_8_feature_mask_aug_{tag}", wauc(y[iho], pi, W[iho]), wauc(pre.dep, pp, pre.W),
        wauc(post.dep, ps, post.W), True, f"랜덤 마스킹 증강 p={prob}, x3", BASE)

dump("pathE_12_kchs_only.csv")
print(f"[{time.time()-t0:.0f}s]")
