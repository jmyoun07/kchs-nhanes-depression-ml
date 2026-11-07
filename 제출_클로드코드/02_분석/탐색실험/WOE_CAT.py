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

kchs = pd.read_parquet(KCHS, columns=FEATURES+["dep","W","year","strata","psu"])
pre  = pd.read_parquet(PRE,  columns=FEATURES+["dep","W","strata","psu"])
post = pd.read_parquet(POST, columns=FEATURES+["dep","W","strata","psu"])
print("shapes", kchs.shape, pre.shape, post.shape)

y = kchs["dep"].astype(int).values
tr, ho = train_test_split(np.arange(len(kchs)), train_size=500000, test_size=100000,
                          stratify=y, random_state=42)
print("train", len(tr), "holdout", len(ho), "prev", y[tr].mean().round(4), y[ho].mean().round(4))

def lgbm():
    return LGBMClassifier(num_leaves=31, min_child_samples=1000, learning_rate=0.01,
                          n_estimators=900, reg_alpha=1.0, reg_lambda=20.0,
                          subsample=0.6, colsample_bytree=0.6,
                          random_state=42, n_jobs=-1, verbose=-1)

# ============ MODEL A: one-hot baseline (recomputed in-run) ============
num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                     ("oh", OneHotEncoder(handle_unknown="ignore", drop="first"))])
ct = ColumnTransformer([("num", num_pipe, NUM), ("bin", "passthrough", BIN), ("cat", cat_pipe, CAT)])

Xtr = kchs.iloc[tr][FEATURES]
ct.fit(Xtr)
clfA = lgbm(); clfA.fit(ct.transform(Xtr), y[tr])
def probaA(df): return clfA.predict_proba(ct.transform(df[FEATURES]))[:,1]
baseA = dict(internal=roc_auc_score(y[ho], probaA(kchs.iloc[ho])),
             pre=roc_auc_score(pre["dep"].astype(int), probaA(pre), sample_weight=pre["W"].astype(float)),
             post=roc_auc_score(post["dep"].astype(int), probaA(post), sample_weight=post["W"].astype(float)))
print(f"[A one-hot] internal={baseA['internal']:.4f} pre={baseA['pre']:.4f} post={baseA['post']:.4f}")

# ============ WoE encoding (leakage-safe) ============
# WoE(level) = ln( P(level|dep=1) / P(level|dep=0) ), Laplace-smoothed. NaN -> its own level.
ALPHA = 0.5
def fit_woe(dfc, yv):
    """Return {col: {level: woe}} learned from given rows only. Unseen level -> 0."""
    mp = {}
    tot1 = float((yv == 1).sum()); tot0 = float((yv == 0).sum())
    for c in CAT:
        s = dfc[c].astype("object").where(dfc[c].notna(), "__NA__").astype(str).values
        K = len(np.unique(s))
        d = pd.DataFrame({"lv": s, "y": yv})
        g = d.groupby("lv")["y"].agg(["sum", "count"])
        n1 = g["sum"].astype(float); n0 = (g["count"] - g["sum"]).astype(float)
        woe = np.log(((n1 + ALPHA) / (tot1 + ALPHA * K)) / ((n0 + ALPHA) / (tot0 + ALPHA * K)))
        mp[c] = woe.to_dict()
    return mp

def apply_woe(df, mp):
    out = np.empty((len(df), len(CAT)), dtype=float)
    for j, c in enumerate(CAT):
        s = df[c].astype("object").where(df[c].notna(), "__NA__").astype(str).values
        m = mp[c]
        out[:, j] = np.array([m.get(v, 0.0) for v in s])
    return out

# NUM: fit imputer+scaler on TRAIN only (same as baseline treatment)
num_only = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
num_only.fit(kchs.iloc[tr][NUM])
def numbin(df):
    return np.hstack([num_only.transform(df[NUM]), df[BIN].astype(float).values])

# OOF WoE on TRAIN (5-fold; a row is never encoded with its own fold)
ytr = y[tr]
woe_tr = np.zeros((len(tr), len(CAT)))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
tr_df = kchs.iloc[tr].reset_index(drop=True)
for fold, (itr, ival) in enumerate(skf.split(np.arange(len(tr)), ytr)):
    mp = fit_woe(tr_df.iloc[itr][CAT], ytr[itr])
    woe_tr[ival] = apply_woe(tr_df.iloc[ival][CAT], mp)
print("OOF WoE train done. sample means per CAT:", np.round(woe_tr.mean(0), 3))

# Global mapping from ALL train -> applied to holdout & NHANES
mp_global = fit_woe(tr_df[CAT], ytr)

Xtr_woe = np.hstack([numbin(tr_df), woe_tr])
clfB = lgbm(); clfB.fit(Xtr_woe, ytr)
def probaB(df):
    return clfB.predict_proba(np.hstack([numbin(df), apply_woe(df, mp_global)]))[:,1]

baseB = dict(internal=roc_auc_score(y[ho], probaB(kchs.iloc[ho])),
             pre=roc_auc_score(pre["dep"].astype(int), probaB(pre), sample_weight=pre["W"].astype(float)),
             post=roc_auc_score(post["dep"].astype(int), probaB(post), sample_weight=post["W"].astype(float)))
print(f"[B WoE]     internal={baseB['internal']:.4f} pre={baseB['pre']:.4f} post={baseB['post']:.4f}")

print("--- DELTA (WoE - one-hot) ---")
for k in ["internal","pre","post"]:
    print(f"  {k}: {baseB[k]-baseA[k]:+.4f}")
# Selection is on KCHS-internal only:
selected = "WoE" if baseB["internal"] > baseA["internal"] else "one-hot"
print("SELECTED by internal:", selected)
print("DONE")
