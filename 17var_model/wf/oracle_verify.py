import sys; sys.stdout.reconfigure(encoding="utf-8"); import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

PRE  = "C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model/data/ext_pre.parquet"
POST = "C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model/data/ext_post.parquet"

NUM = ["age","sleep_hrs","bmi","income_q","srh","oral_health"]
BIN = ["partnered","living_alone","current_drinker","active","htn","dm","employed","food_insecure"]
CAT = ["sex","edu","smoking"]
FEATURES = NUM + BIN + CAT

def make_pre(model):
    if model == "logit":
        # logistic needs imputed BIN too (no NaN); scale num
        num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
        bin_pipe = Pipeline([("imp", SimpleImputer(strategy="median"))])
        cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                             ("oh", OneHotEncoder(handle_unknown="ignore", drop="first"))])
        ct = ColumnTransformer([("num", num_pipe, NUM), ("bin", bin_pipe, BIN), ("cat", cat_pipe, CAT)])
        clf = LogisticRegression(penalty="l2", C=1.0, max_iter=2000, class_weight=None)
    else:  # lgbm — BIN passthrough, keep NaN
        num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
        cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                             ("oh", OneHotEncoder(handle_unknown="ignore", drop="first"))])
        ct = ColumnTransformer([("num", num_pipe, NUM), ("bin", "passthrough", BIN), ("cat", cat_pipe, CAT)])
        clf = LGBMClassifier(num_leaves=31, min_child_samples=1000, learning_rate=0.01, n_estimators=900,
                             reg_alpha=1.0, reg_lambda=20.0, subsample=0.6, colsample_bytree=0.6,
                             random_state=42, n_jobs=-1, verbose=-1)
    return Pipeline([("prep", ct), ("clf", clf)])

def cv_auroc(X, y, w, model):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_true, y_score, y_w = [], [], []
    for tr, te in skf.split(X, y):
        pipe = make_pre(model)
        pipe.fit(X.iloc[tr], y[tr])
        p = pipe.predict_proba(X.iloc[te])[:, 1]
        y_true.append(y[te]); y_score.append(p); y_w.append(w[te])
    yt = np.concatenate(y_true); ys = np.concatenate(y_score); yw = np.concatenate(y_w)
    return roc_auc_score(yt, ys, sample_weight=yw), yt, ys, yw

def load(path):
    df = pd.read_parquet(path, columns=FEATURES + ["dep","W"])
    y = df["dep"].astype(int).values
    w = df["W"].astype(float).values
    X = df[FEATURES].copy()
    for c in CAT: X[c] = X[c].astype(str)
    return X, y, w

pre_X, pre_y, pre_w = load(PRE)
post_X, post_y, post_w = load(POST)
pool_X = pd.concat([pre_X, post_X], ignore_index=True)
pool_y = np.concatenate([pre_y, post_y]); pool_w = np.concatenate([pre_w, post_w])

results = {}
for name, (X, y, w) in [("pre",(pre_X,pre_y,pre_w)), ("post",(post_X,post_y,post_w)), ("pool",(pool_X,pool_y,pool_w))]:
    a_lr,_,_,_ = cv_auroc(X, y, w, "logit")
    a_gb, yt, ys, yw = cv_auroc(X, y, w, "lgbm")
    results[name] = {"logit": a_lr, "lgbm": a_gb, "max": max(a_lr,a_gb)}
    if name == "pool":
        pool_oof = (yt, ys, yw)
    print(f"{name}: logit={a_lr:.4f}  lgbm={a_gb:.4f}  MAX={max(a_lr,a_gb):.4f}", flush=True)

# bootstrap 95% CI on pooled oracle (use the better model's OOF preds)
# recompute OOF for whichever model won on pool
yt, ys, yw = pool_oof  # lgbm oof
# if logit won on pool, recompute its oof
if results["pool"]["logit"] > results["pool"]["lgbm"]:
    _, yt, ys, yw = cv_auroc(pool_X, pool_y, pool_w, "logit")
rng = np.random.default_rng(42)
n = len(yt); boots = []
for _ in range(2000):
    idx = rng.integers(0, n, n)
    try:
        boots.append(roc_auc_score(yt[idx], ys[idx], sample_weight=yw[idx]))
    except ValueError:
        pass
lo, hi = np.percentile(boots, [2.5, 97.5])
print(f"POOL oracle max={results['pool']['max']:.4f}  95%CI=[{lo:.4f}, {hi:.4f}]  (n_boot={len(boots)})", flush=True)

print("\nSUMMARY")
for k,v in results.items():
    print(f"  {k}: logit={v['logit']:.4f} lgbm={v['lgbm']:.4f} max={v['max']:.4f}")
print(f"  pool_CI=[{lo:.4f},{hi:.4f}]")
