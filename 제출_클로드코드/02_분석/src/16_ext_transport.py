# -*- coding: utf-8 -*-
"""
16_ext_transport.py — 확장 후보변수 전이가능성(transportability) 실측.
A) 기존14변수 보정 가중 로지스틱 OR: KCHS vs NHANES pre/post (부호·강도 비교)
B) 연령층(20-39/40-59/60+)별 보정 OR — 구조 역전(교호작용) 확인
C) 증분 예측력: 14변수 LightGBM 기준선 대비 후보 1개씩 추가 시 내부/외부 가중 AUROC Δ
   (학습은 KCHS만, NHANES는 재적합 없이 적용 — §2.10)
출력: results/ext_transport_or.csv, ext_transport_agestrat.csv, ext_transport_delta.csv, qc_transport.txt
"""
import os, sys, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

ROOT = r"C:/Users/rudtn/Desktop/claude code/지역사회건강"
PROC = os.path.join(ROOT, "data/processed"); RES = os.path.join(ROOT, "results")
SEED = 42
QC = []
def qc(s):
    QC.append(str(s)); print(s); sys.stdout.flush()

NUM = ["age", "sleep_hrs", "bmi", "income_q", "srh"]
BIN = ["partnered", "living_alone", "current_drinker", "active", "htn", "dm"]
CAT = ["sex", "edu", "smoking"]
FEATURES = NUM + BIN + CAT
CAND = ["employed", "oral_health", "alcohol_freq", "alcohol_qty", "alcohol_qty_drinks",
        "binge", "welfare", "welfare_12mo", "sleep_weekend", "food_insecure"]

k = pd.read_parquet(os.path.join(PROC, "ext_kchs.parquet"))
pre = pd.read_parquet(os.path.join(PROC, "ext_pre.parquet"))
post = pd.read_parquet(os.path.join(PROC, "ext_post.parquet"))
qc(f"로드: kchs={k.shape} pre={pre.shape} post={post.shape}")

# ============ A) 보정 가중 로지스틱 ============
# KCHS는 2M행 → GLM 반복비용 때문에 seed고정 무작위 40만 서브샘플(가중치 유지)
rng = np.random.RandomState(SEED)
ksub_idx = rng.choice(len(k), size=400000, replace=False)
ksub = k.iloc[np.sort(ksub_idx)].reset_index(drop=True)
qc(f"KCHS 회귀용 서브샘플 n={len(ksub)} (seed={SEED}) | 우울(가중)={np.average(ksub['dep'], weights=ksub['W']):.4f}")


def design(df, cand):
    """14변수 보정 설계행렬 + 후보변수. 각 데이터셋 내부에서 대체(중앙값/최빈값)."""
    d = df.copy()
    m = d[cand].notna() & d["dep"].notna() & d["W"].notna()
    d = d.loc[m]
    if len(d) < 100:
        return None
    cols = {}
    for c in NUM:
        v = d[c].to_numpy(float)
        if np.isnan(v).all():
            continue
        v = np.where(np.isnan(v), np.nanmedian(v), v)
        s = v.std()
        if s == 0:
            continue
        cols[c] = (v - v.mean()) / s
    for c in BIN:
        v = d[c].to_numpy(float)
        if np.isnan(v).all():
            continue
        mode = 1.0 if np.nanmean(v) > 0.5 else 0.0
        v = np.where(np.isnan(v), mode, v)
        if v.std() == 0:
            continue
        cols[c] = v
    for c in CAT:
        s = d[c].astype("object")
        s = s.fillna(s.mode().iloc[0] if len(s.mode()) else "NA").astype(str)
        lv = sorted(s.unique())
        for L in lv[1:]:
            v = (s == L).astype(float).to_numpy()
            if v.std() > 0:
                cols[f"{c}={L}"] = v
    xc = d[cand].to_numpy(float)
    if np.nanstd(xc) == 0:
        return None
    cols["_CAND_"] = xc
    X = pd.DataFrame(cols, index=d.index)
    X = sm.add_constant(X, has_constant="add")
    w = d["W"].to_numpy(float)
    w = w / w.mean()          # 정규화 가중(합=n) → 유사-ML 표준오차
    return X, d["dep"].to_numpy(float), w, len(d)


def fit_or(df, cand):
    out = design(df, cand)
    if out is None:
        return dict(logOR=np.nan, se=np.nan, OR=np.nan, lo=np.nan, hi=np.nan, n=0, p=np.nan)
    X, y, w, n = out
    try:
        r = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=w).fit()
        b = float(r.params["_CAND_"]); se = float(r.bse["_CAND_"])
        return dict(logOR=b, se=se, OR=np.exp(b), lo=np.exp(b - 1.96 * se),
                    hi=np.exp(b + 1.96 * se), n=n, p=float(r.pvalues["_CAND_"]))
    except Exception as e:
        qc(f"  [GLM 실패] {cand}: {e}")
        return dict(logOR=np.nan, se=np.nan, OR=np.nan, lo=np.nan, hi=np.nan, n=n, p=np.nan)


qc("\n=== A) 14변수 보정 가중 로지스틱 OR (후보 1단위 증가) ===")
rows = []
for c in CAND:
    t0 = time.time()
    rk = fit_or(ksub, c); rp = fit_or(pre, c); ro = fit_or(post, c)
    same_pre = (np.sign(rk["logOR"]) == np.sign(rp["logOR"]))
    same_post = (np.sign(rk["logOR"]) == np.sign(ro["logOR"]))
    ratio_pre = rp["logOR"] / rk["logOR"] if rk["logOR"] not in (0, np.nan) else np.nan
    ratio_post = ro["logOR"] / rk["logOR"] if rk["logOR"] not in (0, np.nan) else np.nan
    rows.append(dict(var=c,
                     OR_kchs=rk["OR"], lo_kchs=rk["lo"], hi_kchs=rk["hi"], n_kchs=rk["n"], p_kchs=rk["p"],
                     OR_pre=rp["OR"], lo_pre=rp["lo"], hi_pre=rp["hi"], n_pre=rp["n"], p_pre=rp["p"],
                     OR_post=ro["OR"], lo_post=ro["lo"], hi_post=ro["hi"], n_post=ro["n"], p_post=ro["p"],
                     logOR_kchs=rk["logOR"], logOR_pre=rp["logOR"], logOR_post=ro["logOR"],
                     same_dir_pre=same_pre, same_dir_post=same_post,
                     ratio_pre=ratio_pre, ratio_post=ratio_post))
    qc(f"{c:20s} KCHS OR={rk['OR']:.3f}({rk['lo']:.3f}-{rk['hi']:.3f}) | "
       f"pre OR={rp['OR']:.3f}({rp['lo']:.3f}-{rp['hi']:.3f}) p={rp['p']:.3g} | "
       f"post OR={ro['OR']:.3f}({ro['lo']:.3f}-{ro['hi']:.3f}) p={ro['p']:.3g} | "
       f"同부호 pre={same_pre} post={same_post} | 비율 {ratio_pre:.2f}/{ratio_post:.2f} [{time.time()-t0:.0f}s]")
orres = pd.DataFrame(rows)
orres.to_csv(os.path.join(RES, "ext_transport_or.csv"), index=False, encoding="utf-8-sig")

# ============ B) 연령층별 ============
qc("\n=== B) 연령층별 보정 OR (교호작용 위험) ===")
BANDS = [("20-39", 20, 40), ("40-59", 40, 60), ("60+", 60, 200)]
rows = []
for c in CAND:
    line = [f"{c:20s}"]
    for bn, lo, hi in BANDS:
        r = {}
        for nm, d in [("kchs", ksub), ("pre", pre), ("post", post)]:
            sub = d[(d["age"] >= lo) & (d["age"] < hi)]
            r[nm] = fit_or(sub, c)
        flip_pre = (np.sign(r["kchs"]["logOR"]) != np.sign(r["pre"]["logOR"]))
        flip_post = (np.sign(r["kchs"]["logOR"]) != np.sign(r["post"]["logOR"]))
        rows.append(dict(var=c, band=bn,
                         OR_kchs=r["kchs"]["OR"], OR_pre=r["pre"]["OR"], OR_post=r["post"]["OR"],
                         p_pre=r["pre"]["p"], p_post=r["post"]["p"],
                         n_kchs=r["kchs"]["n"], n_pre=r["pre"]["n"], n_post=r["post"]["n"],
                         flip_pre=flip_pre, flip_post=flip_post))
        line.append(f"{bn}: K={r['kchs']['OR']:.2f} P={r['pre']['OR']:.2f} O={r['post']['OR']:.2f}"
                    + ("*역전*" if (flip_pre or flip_post) else ""))
    qc(" | ".join(line))
agres = pd.DataFrame(rows)
agres.to_csv(os.path.join(RES, "ext_transport_agestrat.csv"), index=False, encoding="utf-8-sig")

# ============ C) 증분 예측력 (LightGBM) ============
qc("\n=== C) 증분 예측력: 14변수 기준선 대비 후보 추가 ===")
y = k["dep"].astype(int).values
W = k["W"].to_numpy(float)
idx = np.arange(len(k))
itr, iho = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
is_tr = np.zeros(len(k), bool); is_tr[itr] = True


def _income_q_trainonly(df, is_tr):
    iq = np.full(len(df), np.nan); yr = df["year"].values; inc = df["income_raw"].to_numpy(float)
    for yv in np.unique(yr):
        m = yr == yv; v = inc[m & is_tr]; v = v[~np.isnan(v)]
        if len(v) < 50:
            continue
        e = np.unique(np.quantile(v, [0, .2, .4, .6, .8, 1.0]))
        if len(e) < 3:
            continue
        ii = np.where(m)[0]
        b = np.clip(np.digitize(inc[ii], e[1:-1]), 0, len(e) - 2).astype(float)
        b[np.isnan(inc[ii])] = np.nan; iq[ii] = b
    return iq


kk = k.copy(); kk["income_q"] = _income_q_trainonly(kk, is_tr)


def make_pre(extra):
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore", drop="first"))])
    blocks = [("num", num, NUM), ("bin", "passthrough", BIN), ("cat", cat, CAT)]
    if extra:
        blocks.append(("cand", "passthrough", extra))   # 트리 native NaN 유지
    return ColumnTransformer(blocks)


def run(extra, tag):
    feats = FEATURES + list(extra)
    pipe = Pipeline([("pre", make_pre(list(extra))),
                     ("clf", lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31,
                                                subsample=0.8, colsample_bytree=0.8,
                                                random_state=SEED, n_jobs=-1, verbose=-1))])
    X = kk[feats]
    pipe.fit(X.iloc[itr], y[itr])
    a_int = roc_auc_score(y[iho], pipe.predict_proba(X.iloc[iho])[:, 1], sample_weight=W[iho])
    out = {"model": tag, "internal": a_int}
    for nm, d in [("pre", pre), ("post", post)]:
        p = pipe.predict_proba(d[feats])[:, 1]
        out[nm] = roc_auc_score(d["dep"].astype(int).values, p, sample_weight=d["W"].to_numpy(float))
    return out


t0 = time.time()
base = run([], "baseline_14")
qc(f"기준선(14): 내부={base['internal']:.4f} pre={base['pre']:.4f} post={base['post']:.4f} [{time.time()-t0:.0f}s]")
rows = [dict(base, d_internal=0.0, d_pre=0.0, d_post=0.0)]
for c in CAND:
    t0 = time.time()
    r = run([c], f"+{c}")
    r.update(d_internal=r["internal"] - base["internal"], d_pre=r["pre"] - base["pre"],
             d_post=r["post"] - base["post"])
    rows.append(r)
    qc(f"+{c:20s} 내부={r['internal']:.4f}({r['d_internal']:+.4f}) "
       f"pre={r['pre']:.4f}({r['d_pre']:+.4f}) post={r['post']:.4f}({r['d_post']:+.4f}) [{time.time()-t0:.0f}s]")
dl = pd.DataFrame(rows)
dl.to_csv(os.path.join(RES, "ext_transport_delta.csv"), index=False, encoding="utf-8-sig")

# 조합 모델: 부호일치 + 외부Δ 비음(≥ -0.001) 후보
keep = [r["model"][1:] for r in rows[1:]
        if r["d_pre"] >= -0.001 and r["d_post"] >= -0.001]
qc(f"\n조합 후보(외부Δ 비음): {keep}")
combos = {}
if keep:
    combos["COMBO_all_nonneg"] = keep
sig = orres[(orres.same_dir_pre) & (orres.same_dir_post)]["var"].tolist()
combos["COMBO_samedir"] = sig
for tag, cs in combos.items():
    if not cs:
        continue
    t0 = time.time()
    r = run(cs, tag)
    r.update(d_internal=r["internal"] - base["internal"], d_pre=r["pre"] - base["pre"],
             d_post=r["post"] - base["post"])
    rows.append(r)
    qc(f"{tag} {cs}\n   내부={r['internal']:.4f}({r['d_internal']:+.4f}) "
       f"pre={r['pre']:.4f}({r['d_pre']:+.4f}) post={r['post']:.4f}({r['d_post']:+.4f}) [{time.time()-t0:.0f}s]")
pd.DataFrame(rows).to_csv(os.path.join(RES, "ext_transport_delta.csv"), index=False, encoding="utf-8-sig")
open(os.path.join(RES, "qc_transport.txt"), "w", encoding="utf-8").write("\n".join(QC))
qc("\n완료")
