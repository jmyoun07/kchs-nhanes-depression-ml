# -*- coding: utf-8 -*-
"""
17_ext_transport_combo.py — 후보 조합 정련 + ΔAUROC 불확실성 실측.
- 중복변수 정리(welfare/welfare_12mo, alcohol_qty/alcohol_qty_drinks) 후 조합 비교
- 외부 ΔAUROC의 짝지은 부트스트랩 95%CI (가중, 1000회)
- LightGBM seed 3개 재현성
출력: results/ext_transport_combo.csv, ext_transport_boot.csv, qc_transport_combo.txt
"""
import os, sys, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

ROOT = r"C:/Users/rudtn/Desktop/claude code/지역사회건강"
PROC = os.path.join(ROOT, "data/processed"); RES = os.path.join(ROOT, "results")
SEED = 42; QC = []
def qc(s):
    QC.append(str(s)); print(s); sys.stdout.flush()

NUM = ["age", "sleep_hrs", "bmi", "income_q", "srh"]
BIN = ["partnered", "living_alone", "current_drinker", "active", "htn", "dm"]
CAT = ["sex", "edu", "smoking"]
FEATURES = NUM + BIN + CAT

k = pd.read_parquet(os.path.join(PROC, "ext_kchs.parquet"))
pre = pd.read_parquet(os.path.join(PROC, "ext_pre.parquet"))
post = pd.read_parquet(os.path.join(PROC, "ext_post.parquet"))

y = k["dep"].astype(int).values; W = k["W"].to_numpy(float)
itr, iho = train_test_split(np.arange(len(k)), test_size=0.2, stratify=y, random_state=SEED)
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
    b = [("num", num, NUM), ("bin", "passthrough", BIN), ("cat", cat, CAT)]
    if extra:
        b.append(("cand", "passthrough", extra))
    return ColumnTransformer(b)


def fitpred(extra, seed=SEED):
    feats = FEATURES + list(extra)
    pipe = Pipeline([("pre", make_pre(list(extra))),
                     ("clf", lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31,
                                                subsample=0.8, colsample_bytree=0.8,
                                                random_state=seed, n_jobs=-1, verbose=-1))])
    X = kk[feats]
    pipe.fit(X.iloc[itr], y[itr])
    p_int = pipe.predict_proba(X.iloc[iho])[:, 1]
    p_pre = pipe.predict_proba(pre[feats])[:, 1]
    p_post = pipe.predict_proba(post[feats])[:, 1]
    return p_int, p_pre, p_post


def auc(d_y, p, w):
    return roc_auc_score(d_y, p, sample_weight=w)


ypre = pre["dep"].astype(int).values; wpre = pre["W"].to_numpy(float)
ypost = post["dep"].astype(int).values; wpost = post["W"].to_numpy(float)

SETS = {
    "baseline_14": [],
    "A_core3": ["employed", "oral_health", "food_insecure"],
    "B_core4": ["employed", "oral_health", "food_insecure", "welfare_12mo"],
    "C_core4+binge": ["employed", "oral_health", "food_insecure", "welfare_12mo", "binge"],
    "D_core4+alcfreq": ["employed", "oral_health", "food_insecure", "welfare_12mo", "alcohol_freq"],
    "E_no_foodins": ["employed", "oral_health", "welfare_12mo"],
    "F_all9": ["employed", "oral_health", "alcohol_freq", "alcohol_qty", "alcohol_qty_drinks",
               "binge", "welfare", "welfare_12mo", "food_insecure"],
}
qc("=== 조합 비교 (가중 AUROC) ===")
rows = []; preds = {}
for tag, cs in SETS.items():
    t0 = time.time()
    pi, pp, po = fitpred(cs)
    preds[tag] = (pi, pp, po)
    r = dict(model=tag, k=len(cs), vars="|".join(cs),
             internal=auc(y[iho], pi, W[iho]), pre=auc(ypre, pp, wpre), post=auc(ypost, po, wpost))
    rows.append(r)
    qc(f"{tag:18s} (+{len(cs)}) 내부={r['internal']:.4f} pre={r['pre']:.4f} post={r['post']:.4f} [{time.time()-t0:.0f}s]")
b = rows[0]
for r in rows:
    r.update(d_internal=r["internal"] - b["internal"], d_pre=r["pre"] - b["pre"], d_post=r["post"] - b["post"])
pd.DataFrame(rows).to_csv(os.path.join(RES, "ext_transport_combo.csv"), index=False, encoding="utf-8-sig")

# ---- 짝지은 부트스트랩 ΔAUROC 95%CI ----
qc("\n=== 외부 ΔAUROC 짝지은 부트스트랩 95%CI (B=1000, 가중) ===")


def boot_delta(yv, wv, p0, p1, B=1000, seed=SEED):
    rng = np.random.RandomState(seed); n = len(yv); out = np.empty(B)
    for i in range(B):
        ii = rng.randint(0, n, n)
        if yv[ii].sum() < 5 or yv[ii].sum() == len(ii):
            out[i] = np.nan; continue
        out[i] = (roc_auc_score(yv[ii], p1[ii], sample_weight=wv[ii])
                  - roc_auc_score(yv[ii], p0[ii], sample_weight=wv[ii]))
    return np.nanpercentile(out, [2.5, 97.5]), np.nanmean(out)


brows = []
for tag in SETS:
    if tag == "baseline_14":
        continue
    for nm, yv, wv, j in [("pre", ypre, wpre, 1), ("post", ypost, wpost, 2)]:
        (lo, hi), mn = boot_delta(yv, wv, preds["baseline_14"][j], preds[tag][j])
        d = auc(yv, preds[tag][j], wv) - auc(yv, preds["baseline_14"][j], wv)
        sig = "유의" if lo > 0 else ("음의유의" if hi < 0 else "n.s.")
        brows.append(dict(model=tag, set=nm, delta=d, lo=lo, hi=hi, sig=sig))
        qc(f"{tag:18s} {nm:5s} Δ={d:+.4f} 95%CI [{lo:+.4f}, {hi:+.4f}] {sig}")

# 단일변수도 주요 3개 부트스트랩
qc("\n--- 단일변수(주요) ---")
for c in ["employed", "oral_health", "food_insecure", "welfare_12mo", "sleep_weekend"]:
    pi, pp, po = fitpred([c])
    for nm, yv, wv, pp_ in [("pre", ypre, wpre, pp), ("post", ypost, wpost, po)]:
        j = 1 if nm == "pre" else 2
        (lo, hi), _ = boot_delta(yv, wv, preds["baseline_14"][j], pp_)
        d = auc(yv, pp_, wv) - auc(yv, preds["baseline_14"][j], wv)
        sig = "유의" if lo > 0 else ("음의유의" if hi < 0 else "n.s.")
        brows.append(dict(model=f"+{c}", set=nm, delta=d, lo=lo, hi=hi, sig=sig))
        qc(f"+{c:18s} {nm:5s} Δ={d:+.4f} 95%CI [{lo:+.4f}, {hi:+.4f}] {sig}")
pd.DataFrame(brows).to_csv(os.path.join(RES, "ext_transport_boot.csv"), index=False, encoding="utf-8-sig")

# ---- seed 재현성 ----
qc("\n=== seed 재현성 (B_core4) ===")
for s in [42, 7, 2024]:
    pi, pp, po = fitpred(SETS["B_core4"], seed=s)
    qc(f" seed={s}: 내부={auc(y[iho],pi,W[iho]):.4f} pre={auc(ypre,pp,wpre):.4f} post={auc(ypost,po,wpost):.4f}")
open(os.path.join(RES, "qc_transport_combo.txt"), "w", encoding="utf-8").write("\n".join(QC))
qc("완료")
