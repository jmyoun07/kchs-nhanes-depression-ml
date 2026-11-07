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

kchs = pd.read_parquet(KCHS, columns=FEATURES+["dep","W","year","strata","psu"])
pre  = pd.read_parquet(PRE,  columns=FEATURES+["dep","W","strata","psu"])
post = pd.read_parquet(POST, columns=FEATURES+["dep","W","strata","psu"])
print("shapes", kchs.shape, pre.shape, post.shape)

y = kchs["dep"].astype(int).values
tr, ho = train_test_split(np.arange(len(kchs)), train_size=500000, test_size=100000,
                          stratify=y, random_state=42)
print("train", len(tr), "holdout", len(ho))

ypre  = pre["dep"].astype(int).values
ypost = post["dep"].astype(int).values
wpre  = pre["W"].astype(float).values
wpost = post["W"].astype(float).values

CLF = dict(num_leaves=31, min_child_samples=1000, learning_rate=0.01, n_estimators=900,
           reg_alpha=1.0, reg_lambda=20.0, subsample=0.6, colsample_bytree=0.6,
           random_state=42, n_jobs=-1, verbose=-1)

# ---- Select missing-indicator features from KCHS-TRAIN missingness only (LEAKAGE-FREE) ----
train_miss = kchs.iloc[tr][FEATURES].isna().mean()
IND_FEATS = sorted([c for c in FEATURES if train_miss[c] > 0.05])
print("KCHS-train missingness>5% (indicator feats):",
      {c: round(float(train_miss[c]),3) for c in IND_FEATS})

def num_pipe():
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
def cat_pipe():
    return Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                     ("oh", OneHotEncoder(handle_unknown="ignore", drop="first"))])
def bin_fill():
    return SimpleImputer(strategy="most_frequent")

def build_ct(variant):
    # BASELINE: BIN passthrough (NaN kept -> LightGBM tree-native)
    # B (explicit_fill): BIN explicitly filled with most_frequent (fit on train), no NaN to LGBM
    # C (fill_all_explicit): same explicit fill; kept distinct for clarity of the "all-explicit" design
    if variant in ("explicit_fill", "fill_indicator"):
        bin_trans = ("bin", bin_fill(), BIN)
    else:
        bin_trans = ("bin", "passthrough", BIN)
    return ColumnTransformer([("num", num_pipe(), NUM), bin_trans, ("cat", cat_pipe(), CAT)])

def add_indicators(df):
    # deterministic per-row isna -> no fitting -> leakage-free
    return np.column_stack([df[c].isna().astype(int).values for c in IND_FEATS])

def run(variant):
    ct = build_ct(variant)
    Xtr = kchs.iloc[tr][FEATURES]
    ct.fit(Xtr)  # fit transforms on KCHS TRAIN ONLY
    def feat(df):
        base = ct.transform(df[FEATURES])
        base = base.toarray() if hasattr(base, "toarray") else np.asarray(base)
        if variant in ("indicator", "fill_indicator"):
            base = np.hstack([base, add_indicators(df)])
        return base
    Xtr_t = feat(kchs.iloc[tr])
    clf = LGBMClassifier(**CLF)
    clf.fit(Xtr_t, y[tr])
    p_ho  = clf.predict_proba(feat(kchs.iloc[ho]))[:,1]
    p_pre = clf.predict_proba(feat(pre))[:,1]
    p_post= clf.predict_proba(feat(post))[:,1]
    internal = roc_auc_score(y[ho], p_ho)
    a_pre  = roc_auc_score(ypre,  p_pre,  sample_weight=wpre)
    a_post = roc_auc_score(ypost, p_post, sample_weight=wpost)
    return internal, a_pre, a_post

VARIANTS = [
    ("baseline",       "passthrough BIN (tree-native NaN); NUM median+scale; CAT mode+onehot"),
    ("indicator",      "baseline + binary missing-indicator cols for KCHS-train>5% feats"),
    ("explicit_fill",  "BIN explicitly mode-filled (no NaN to LGBM); rest as baseline"),
    ("fill_indicator", "explicit BIN fill + missing indicators (combined; all fit on train)"),
]

res = {}
for name, desc in VARIANTS:
    internal, a_pre, a_post = run(name)
    res[name] = (internal, a_pre, a_post)
    print(f"[{name:14s}] internal={internal:.4f}  pre={a_pre:.4f}  post={a_post:.4f}   ({desc})")

b_int, b_pre, b_post = res["baseline"]
print("\n=== vs baseline (delta, must be >0.001 on BOTH pre&post to 'beat') ===")
best = None
for name, _ in VARIANTS:
    if name == "baseline": continue
    i,p,q = res[name]
    dp, dq = p-b_pre, q-b_post
    beats = (dp > 0.001) and (dq > 0.001)
    print(f"[{name:14s}] dInternal={i-b_int:+.4f}  dPre={dp:+.4f}  dPost={dq:+.4f}  beats={beats}")
    if beats and (best is None or (p+q) > (res[best][1]+res[best][2])):
        best = name
print("BEST_BEATING:", best)
print("DONE")
