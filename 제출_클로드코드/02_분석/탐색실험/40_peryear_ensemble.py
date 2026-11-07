import sys; sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
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
CAP = 80000  # max rows per year

CFG = dict(num_leaves=31, min_child_samples=1000, learning_rate=0.01, n_estimators=900,
           reg_alpha=1.0, reg_lambda=20.0, subsample=0.6, colsample_bytree=0.6,
           random_state=42, n_jobs=-1, verbose=-1)

def make_ct():
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("oh", OneHotEncoder(handle_unknown="ignore", drop="first"))])
    return ColumnTransformer([("num", num_pipe, NUM),
                              ("bin", "passthrough", BIN),
                              ("cat", cat_pipe, CAT)])

def fit_model(df, y):
    ct = make_ct()
    ct.fit(df[FEATURES])
    clf = LGBMClassifier(**CFG)
    clf.fit(ct.transform(df[FEATURES]), y)
    return ct, clf

def pred(ct, clf, df):
    return clf.predict_proba(ct.transform(df[FEATURES]))[:, 1]

# ---- load ----
kchs = pd.read_parquet(KCHS, columns=FEATURES+["dep","W","year"])
pre  = pd.read_parquet(PRE,  columns=FEATURES+["dep","W"])
post = pd.read_parquet(POST, columns=FEATURES+["dep","W"])
print("shapes", kchs.shape, pre.shape, post.shape)

y = kchs["dep"].astype(int).values
tr, ho = train_test_split(np.arange(len(kchs)), train_size=500000, test_size=100000,
                          stratify=y, random_state=42)
train_df = kchs.iloc[tr].reset_index(drop=True); ytr = y[tr]
ho_df = kchs.iloc[ho].reset_index(drop=True);    yho = y[ho]
ypre = pre["dep"].astype(int).values
ypost = post["dep"].astype(int).values
wpre = pre["W"].astype(float).values
wpost = post["W"].astype(float).values
print("train", len(train_df), "holdout", len(ho_df),
      "prev tr/ho", round(ytr.mean(),4), round(yho.mean(),4))

# ---- per-year models (subsample <=CAP rows/year from TRAIN only) ----
years = sorted(train_df["year"].unique().tolist())
rng = np.random.default_rng(42)
P_ho, P_pre, P_post = [], [], []
yr_internal = []
n_used = 0
print("\n-- per-year models --")
for yr in years:
    idx = np.where(train_df["year"].values == yr)[0]
    if len(idx) > CAP:
        idx = rng.choice(idx, size=CAP, replace=False)
    n_used += len(idx)
    sub = train_df.iloc[idx]
    ct, clf = fit_model(sub, ytr[idx])
    p_ho = pred(ct, clf, ho_df); p_pre = pred(ct, clf, pre); p_post = pred(ct, clf, post)
    a_int = roc_auc_score(yho, p_ho)
    yr_internal.append(a_int)
    P_ho.append(p_ho); P_pre.append(p_pre); P_post.append(p_post)
    print(f"year {yr}: n={len(idx):6d}  internal={a_int:.4f}  "
          f"pre={roc_auc_score(ypre,p_pre,sample_weight=wpre):.4f}  "
          f"post={roc_auc_score(ypost,p_post,sample_weight=wpost):.4f}")

P_ho = np.array(P_ho); P_pre = np.array(P_pre); P_post = np.array(P_post)
yr_internal = np.array(yr_internal)
print("total rows used by ensemble:", n_used)

def evalset(p_ho, p_pre, p_post, tag):
    a = roc_auc_score(yho, p_ho)
    b = roc_auc_score(ypre, p_pre, sample_weight=wpre)
    c = roc_auc_score(ypost, p_post, sample_weight=wpost)
    print(f"{tag:28s} internal={a:.4f}  pre={b:.4f}  post={c:.4f}")
    return a, b, c

# ---- simple average ensemble ----
print("\n-- ensembles --")
simp = evalset(P_ho.mean(0), P_pre.mean(0), P_post.mean(0), "SIMPLE avg")

# ---- weighted ensemble: weights = per-year KCHS-internal holdout AUROC (label-safe) ----
w = yr_internal / yr_internal.sum()
wtd = evalset((w[:,None]*P_ho).sum(0), (w[:,None]*P_pre).sum(0), (w[:,None]*P_post).sum(0),
              "WEIGHTED (holdout-AUROC)")
print("per-year weights:", np.round(w,4).tolist())

# ---- in-run POOLED baseline on the SAME total number of rows ----
print("\n-- pooled baseline (same total rows) --")
pool_idx = np.arange(len(train_df))
if n_used < len(train_df):
    pool_idx = rng.choice(len(train_df), size=n_used, replace=False)
pool_df = train_df.iloc[pool_idx]
ct_p, clf_p = fit_model(pool_df, ytr[pool_idx])
pool = evalset(pred(ct_p, clf_p, ho_df), pred(ct_p, clf_p, pre), pred(ct_p, clf_p, post),
               f"POOLED (n={len(pool_idx)})")

# ---- selection by KCHS-internal criteria ONLY (no NHANES labels) ----
cand = {"SIMPLE": (simp, P_ho.mean(0)), "WEIGHTED": (wtd, (w[:,None]*P_ho).sum(0))}
sel = max(cand, key=lambda k: cand[k][0][0])  # pick higher internal AUROC
sel_int, sel_pre, sel_post = cand[sel][0]
print(f"\nSELECTED by internal AUROC: {sel}  internal={sel_int:.4f}")

eps = 0.001
beats = (sel_pre > pool[1] + eps) and (sel_post > pool[2] + eps)
print(f"\n=== RESULT ===")
print(f"baseline(pooled) pre={pool[1]:.4f} post={pool[2]:.4f} internal={pool[0]:.4f}")
print(f"method({sel})    pre={sel_pre:.4f} post={sel_post:.4f} internal={sel_int:.4f}")
print(f"beats_baseline (both pre&post > baseline+{eps}): {beats}")
print("DONE")
