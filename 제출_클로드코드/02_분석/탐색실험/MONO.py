import sys; sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from lightgbm import LGBMClassifier

KCHS = "C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model/data/ext_kchs.parquet"
PRE  = "C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model/data/ext_pre.parquet"
POST = "C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model/data/ext_post.parquet"

NUM = ["age","sleep_hrs","bmi","income_q","srh","oral_health"]
BIN = ["partnered","living_alone","current_drinker","active","htn","dm","employed","food_insecure"]
CAT = ["sex","edu","smoking"]
FEATURES = NUM + BIN + CAT
# manual design-matrix column order (known): NUM, then BIN, then CAT (integer-coded)
COLS = NUM + BIN + CAT

kchs = pd.read_parquet(KCHS, columns=FEATURES+["dep","W","year","strata","psu"])
pre  = pd.read_parquet(PRE,  columns=FEATURES+["dep","W","strata","psu"])
post = pd.read_parquet(POST, columns=FEATURES+["dep","W","strata","psu"])
print("shapes", kchs.shape, pre.shape, post.shape)

y = kchs["dep"].astype(int).values
tr, ho = train_test_split(np.arange(len(kchs)), train_size=500000, test_size=100000,
                          stratify=y, random_state=42)
print("train", len(tr), "holdout", len(ho))

# ---------- Fit transforms on KCHS TRAIN only ----------
Xtr_raw = kchs.iloc[tr][FEATURES]

# NUM: median impute + standard scale (fit on train)
num_imp = SimpleImputer(strategy="median").fit(Xtr_raw[NUM])
num_sc  = StandardScaler().fit(num_imp.transform(Xtr_raw[NUM]))

# CAT: most_frequent impute (fit on train) + integer code from TRAIN categories only
cat_imp = SimpleImputer(strategy="most_frequent").fit(Xtr_raw[CAT])
cat_tr_imp = pd.DataFrame(cat_imp.transform(Xtr_raw[CAT]), columns=CAT)
cat_maps = {}  # feature -> {category: int code}
for c in CAT:
    cats = pd.Index(pd.unique(cat_tr_imp[c].astype(str)))
    cat_maps[c] = {v: i for i, v in enumerate(cats)}

def build_matrix(df):
    d = df[FEATURES]
    out = {}
    # NUM imputed+scaled
    num_t = num_sc.transform(num_imp.transform(d[NUM]))
    for j, c in enumerate(NUM):
        out[c] = num_t[:, j]
    # BIN as-is (NaN kept)
    for c in BIN:
        out[c] = d[c].astype(float).values
    # CAT integer-coded via train map; impute most_frequent first; unseen -> -1 (missing)
    cat_t = pd.DataFrame(cat_imp.transform(d[CAT]), columns=CAT)
    for c in CAT:
        out[c] = cat_t[c].astype(str).map(cat_maps[c]).fillna(-1).astype("int32").values
    return pd.DataFrame(out, columns=COLS)

Xtr = build_matrix(kchs.iloc[tr])
Xho = build_matrix(kchs.iloc[ho])
Xpre = build_matrix(pre)
Xpost = build_matrix(post)

cat_idx = [COLS.index(c) for c in CAT]

# ---------- Monotone direction from KCHS-TRAIN univariate Spearman sign ONLY ----------
ytr = y[tr]
mono = []
print("\n--- NUM monotone directions (Spearman vs dep on TRAIN) ---")
for c in NUM:
    xr = Xtr_raw[c].astype(float).values
    m = ~np.isnan(xr)
    rho, _ = spearmanr(xr[m], ytr[m])
    if np.isnan(rho) or abs(rho) < 0.02:
        d = 0
    else:
        d = 1 if rho > 0 else -1
    mono.append(d)
    print(f"  {c:12s} rho={rho:+.4f} -> constraint {d:+d}")
# constraints aligned to COLS: NUM signs, BIN 0, CAT 0
mono_constraints = mono + [0]*len(BIN) + [0]*len(CAT)
print("monotone_constraints (aligned to COLS):", mono_constraints)

LGB = dict(num_leaves=31, min_child_samples=1000, learning_rate=0.01, n_estimators=900,
           reg_alpha=1.0, reg_lambda=20.0, subsample=0.6, colsample_bytree=0.6,
           random_state=42, n_jobs=-1, verbose=-1)

def evaluate(clf, tag):
    p_ho  = clf.predict_proba(Xho)[:,1]
    p_pre = clf.predict_proba(Xpre)[:,1]
    p_post= clf.predict_proba(Xpost)[:,1]
    internal = roc_auc_score(y[ho], p_ho)
    pre_a  = roc_auc_score(pre["dep"].astype(int),  p_pre,  sample_weight=pre["W"].astype(float))
    post_a = roc_auc_score(post["dep"].astype(int), p_post, sample_weight=post["W"].astype(float))
    print(f"[{tag}] internal={internal:.4f}  pre={pre_a:.4f}  post={post_a:.4f}")
    return internal, pre_a, post_a

# ---------- In-run baseline: same split, same manual matrix, NO constraints ----------
base = LGBMClassifier(**LGB)
base.fit(Xtr, ytr, categorical_feature=cat_idx)
b_int, b_pre, b_post = evaluate(base, "BASELINE (manual matrix, no constraints)")

# ---------- Monotonic-constrained model ----------
mdl = LGBMClassifier(monotone_constraints=mono_constraints, **LGB)
mdl.fit(Xtr, ytr, categorical_feature=cat_idx)
m_int, m_pre, m_post = evaluate(mdl, "MONOTONE")

beats = (m_pre - b_pre > 0.001) and (m_post - b_post > 0.001)
print(f"\nDELTA pre={m_pre-b_pre:+.4f}  post={m_post-b_post:+.4f}  internal={m_int-b_int:+.4f}")
print(f"beats_baseline (both pre&post >+0.001) = {beats}")
print("DONE")
