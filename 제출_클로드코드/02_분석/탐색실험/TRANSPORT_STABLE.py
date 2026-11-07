import sys; sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

KCHS = "C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model/data/ext_kchs.parquet"
PRE  = "C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model/data/ext_pre.parquet"
POST = "C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model/data/ext_post.parquet"

NUM = ["age","sleep_hrs","bmi","income_q","srh","oral_health"]
BIN = ["partnered","living_alone","current_drinker","active","htn","dm","employed","food_insecure"]
CAT = ["sex","edu","smoking"]
FEATURES = NUM + BIN + CAT

TRAIN_SIZE = 300000   # per recipe allowance; in-run baseline computed at SAME size
TEST_SIZE  = 100000

kchs = pd.read_parquet(KCHS, columns=FEATURES+["dep","W","year","strata","psu"])
pre  = pd.read_parquet(PRE,  columns=FEATURES+["dep","W","strata","psu"])
post = pd.read_parquet(POST, columns=FEATURES+["dep","W","strata","psu"])
print("shapes", kchs.shape, pre.shape, post.shape)

y = kchs["dep"].astype(int).values
tr, ho = train_test_split(np.arange(len(kchs)), train_size=TRAIN_SIZE, test_size=TEST_SIZE,
                          stratify=y, random_state=42)
print("train", len(tr), "holdout", len(ho), "prev tr/ho", y[tr].mean().round(4), y[ho].mean().round(4))

Xtr_all = kchs.iloc[tr][FEATURES].reset_index(drop=True)
ytr = y[tr]
Xho = kchs.iloc[ho][FEATURES].reset_index(drop=True)
yho = y[ho]
ypre = pre["dep"].astype(int).values
ypost = post["dep"].astype(int).values

def make_ct(num, binf, cat):
    parts = []
    if num:  parts.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                            ("sc", StandardScaler())]), num))
    if binf: parts.append(("bin", "passthrough", binf))
    if cat:  parts.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                            ("oh", OneHotEncoder(handle_unknown="ignore", drop="first"))]), cat))
    return ColumnTransformer(parts)

def make_clf():
    return LGBMClassifier(num_leaves=31, min_child_samples=1000, learning_rate=0.01,
                          n_estimators=900, reg_alpha=1.0, reg_lambda=20.0,
                          subsample=0.6, colsample_bytree=0.6,
                          random_state=42, n_jobs=-1, verbose=-1)

# ================================================================
# STEP 1: Transport-stability score per feature -- COVARIATES ONLY.
# Population Stability Index (PSI) between KCHS-train (expected) and
# pooled NHANES pre+post covariates (actual). NO labels used anywhere.
# Higher PSI => more distributionally unstable across domains.
# ================================================================
nh = pd.concat([pre[FEATURES], post[FEATURES]], ignore_index=True)  # covariates only
EPS = 1e-6

def psi_numeric(exp_ser, act_ser, bins=10):
    e = pd.to_numeric(exp_ser, errors="coerce").dropna().values
    a = pd.to_numeric(act_ser, errors="coerce").dropna().values
    qs = np.unique(np.quantile(e, np.linspace(0, 1, bins+1)))
    if len(qs) < 3:
        return 0.0
    qs[0], qs[-1] = -np.inf, np.inf
    e_p = np.histogram(e, bins=qs)[0] / len(e)
    a_p = np.histogram(a, bins=qs)[0] / len(a)
    e_p = np.clip(e_p, EPS, None); a_p = np.clip(a_p, EPS, None)
    return float(np.sum((a_p - e_p) * np.log(a_p / e_p)))

def psi_categorical(exp_ser, act_ser):
    cats = pd.Index(exp_ser.dropna().unique()).union(act_ser.dropna().unique())
    e = exp_ser.value_counts(normalize=True, dropna=True)
    a = act_ser.value_counts(normalize=True, dropna=True)
    e_p = np.clip(e.reindex(cats).fillna(0.0).values, EPS, None)
    a_p = np.clip(a.reindex(cats).fillna(0.0).values, EPS, None)
    return float(np.sum((a_p - e_p) * np.log(a_p / e_p)))

psi = {}
for f in NUM:
    psi[f] = psi_numeric(Xtr_all[f], nh[f])
for f in BIN:
    psi[f] = psi_categorical(Xtr_all[f].astype("object"), nh[f].astype("object"))
for f in CAT:
    psi[f] = psi_categorical(Xtr_all[f].astype("object"), nh[f].astype("object"))

ranked = sorted(FEATURES, key=lambda f: psi[f], reverse=True)  # most unstable first
print("\nPSI (KCHS-train vs pooled NHANES covariates), most-unstable first:")
for f in ranked:
    print(f"  {f:16s} PSI={psi[f]:.4f}")

# ================================================================
# STEP 2: For k in {0,1,2,3}, drop the k most-unstable features,
# evaluate internal via 5-fold KCHS CV (labels = KCHS only).
# ================================================================
def feat_split(feats):
    return ([f for f in NUM if f in feats],
            [f for f in BIN if f in feats],
            [f for f in CAT if f in feats])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(Xtr_all, ytr))

cv_results = {}
for k in [0, 1, 2, 3]:
    drop = set(ranked[:k])
    feats = [f for f in FEATURES if f not in drop]
    num, binf, cat = feat_split(feats)
    aucs = []
    for fi, (a, b) in enumerate(folds):
        ct = make_ct(num, binf, cat)
        ct.fit(Xtr_all.iloc[a][feats])
        clf = make_clf()
        clf.fit(ct.transform(Xtr_all.iloc[a][feats]), ytr[a])
        p = clf.predict_proba(ct.transform(Xtr_all.iloc[b][feats]))[:, 1]
        aucs.append(roc_auc_score(ytr[b], p))
    cv_results[k] = (np.mean(aucs), np.std(aucs), sorted(drop))
    print(f"\nk={k} dropped={sorted(drop)} CV-AUC={np.mean(aucs):.4f} (+/-{np.std(aucs):.4f})")

# ================================================================
# STEP 3: SELECT k by KCHS-internal CV (NOT external).
# ================================================================
best_k = max(cv_results, key=lambda k: cv_results[k][0])
print(f"\nInternal-CV selected k = {best_k}  (dropped {cv_results[best_k][2]})")

# ================================================================
# STEP 4: Fit full-train models and evaluate external for every k,
# so we can report in-run full-feature baseline (k=0) vs selected.
# ================================================================
def fit_eval(feats):
    num, binf, cat = feat_split(feats)
    ct = make_ct(num, binf, cat)
    ct.fit(Xtr_all[feats])               # fit transforms on KCHS TRAIN only
    clf = make_clf()
    clf.fit(ct.transform(Xtr_all[feats]), ytr)
    def pr(df): return clf.predict_proba(ct.transform(df[feats]))[:, 1]
    internal = roc_auc_score(yho, pr(Xho))
    pre_a = roc_auc_score(ypre, pr(pre), sample_weight=pre["W"].astype(float))
    post_a = roc_auc_score(ypost, pr(post), sample_weight=post["W"].astype(float))
    return internal, pre_a, post_a

ext = {}
for k in [0, 1, 2, 3]:
    drop = set(ranked[:k])
    feats = [f for f in FEATURES if f not in drop]
    ext[k] = fit_eval(feats)
    print(f"k={k}: internal={ext[k][0]:.4f} pre={ext[k][1]:.4f} post={ext[k][2]:.4f}")

base_int, base_pre, base_post = ext[0]                 # in-run full-feature baseline
sel_int, sel_pre, sel_post   = ext[best_k]

beats = (sel_pre > base_pre + 0.001) and (sel_post > base_post + 0.001)
print("\n==================== RESULT ====================")
print(f"in-run baseline (k=0): internal={base_int:.4f} pre={base_pre:.4f} post={base_post:.4f}")
print(f"selected (k={best_k}) : internal={sel_int:.4f} pre={sel_pre:.4f} post={sel_post:.4f}")
print(f"delta pre={sel_pre-base_pre:+.4f} post={sel_post-base_post:+.4f}")
print(f"beats_baseline (both >+0.001) = {beats}")
print("DONE")
