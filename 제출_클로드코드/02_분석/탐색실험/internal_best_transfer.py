import sys; sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, time
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

t0 = time.time()
KCHS = "C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model/data/ext_kchs.parquet"
PRE  = "C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model/data/ext_pre.parquet"
POST = "C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model/data/ext_post.parquet"

NUM = ["age","sleep_hrs","bmi","income_q","srh","oral_health"]
BIN = ["partnered","living_alone","current_drinker","active","htn","dm","employed","food_insecure"]
CAT = ["sex","edu","smoking"]
FEATURES = NUM + BIN + CAT

def load(p):
    return pd.read_parquet(p, columns=FEATURES+["dep","W","year","strata","psu"] if p==KCHS
                           else FEATURES+["dep","W"])

kchs = pd.read_parquet(KCHS, columns=FEATURES+["dep","W","year","strata","psu"])
pre  = pd.read_parquet(PRE,  columns=FEATURES+["dep","W"])
post = pd.read_parquet(POST, columns=FEATURES+["dep","W"])
# string categoricals -> ensure object dtype
for df in (kchs, pre, post):
    for c in CAT:
        df[c] = df[c].astype("object")
print(f"loaded kchs={len(kchs)} pre={len(pre)} post={len(post)}  {time.time()-t0:.0f}s", flush=True)

y = kchs["dep"].astype(int).values
tr, ho = train_test_split(np.arange(len(kchs)), train_size=500000, test_size=100000,
                          stratify=y, random_state=42)
print(f"split tr={len(tr)} ho={len(ho)} prevalence tr={y[tr].mean():.4f}", flush=True)

# 300k subsample of the TRAIN indices for CV-based selection (KCHS-internal only)
rng = np.random.RandomState(42)
sub = rng.permutation(tr)[:300000]
Xsub = kchs.iloc[sub][FEATURES]; ysub = y[sub]

def make_pre():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM),
        ("bin", "passthrough", BIN),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", drop="first"))]), CAT),
    ])

BASELINE = dict(num_leaves=31, min_child_samples=1000, learning_rate=0.01, n_estimators=900,
                reg_alpha=1.0, reg_lambda=20.0, subsample=0.6, colsample_bytree=0.6,
                random_state=42, n_jobs=-1, verbose=-1)

# ---- Build >=20 candidate configs (include baseline as config 0) ----
grid_nl   = [15, 31, 63, 127]
grid_mcs  = [200, 500, 1000, 2000]
grid_lr_ne= [(0.01,900),(0.02,600),(0.05,400),(0.03,700),(0.1,300)]
grid_reg  = [(0.0,0.0),(1.0,20.0),(5.0,5.0),(0.5,50.0),(2.0,100.0)]
grid_ss   = [(0.6,0.6),(0.8,0.8),(1.0,1.0),(0.7,0.5)]

cfg_rng = np.random.RandomState(7)
configs = [dict(BASELINE)]
seen = set()
while len(configs) < 24:
    nl = cfg_rng.choice(grid_nl); mcs = cfg_rng.choice(grid_mcs)
    lr, ne = grid_lr_ne[cfg_rng.randint(len(grid_lr_ne))]
    ra, rl = grid_reg[cfg_rng.randint(len(grid_reg))]
    ss, cs = grid_ss[cfg_rng.randint(len(grid_ss))]
    key = (int(nl),int(mcs),float(lr),int(ne),float(ra),float(rl),float(ss),float(cs))
    if key in seen: continue
    seen.add(key)
    configs.append(dict(num_leaves=int(nl), min_child_samples=int(mcs), learning_rate=float(lr),
                        n_estimators=int(ne), reg_alpha=float(ra), reg_lambda=float(rl),
                        subsample=float(ss), colsample_bytree=float(cs),
                        random_state=42, n_jobs=-1, verbose=-1))
print(f"n_configs={len(configs)}", flush=True)

# ---- 5-fold CV AUROC on 300k subsample (preprocessing fit inside each fold) ----
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(Xsub, ysub))
results = []
for i, cfg in enumerate(configs):
    fa = []
    for tri, vai in folds:
        pipe = Pipeline([("pre", make_pre()), ("clf", LGBMClassifier(**cfg))])
        pipe.fit(Xsub.iloc[tri], ysub[tri])
        p = pipe.predict_proba(Xsub.iloc[vai])[:,1]
        fa.append(roc_auc_score(ysub[vai], p))
    cv = float(np.mean(fa))
    results.append((i, cv, cfg))
    tag = "BASELINE" if i==0 else ""
    print(f"cfg{i:02d} cv={cv:.5f} {tag} {dict((k,cfg[k]) for k in ['num_leaves','min_child_samples','learning_rate','n_estimators','reg_alpha','reg_lambda','subsample','colsample_bytree'])}  [{time.time()-t0:.0f}s]", flush=True)

results.sort(key=lambda r: -r[1])
best_i, best_cv, best_cfg = results[0]
base_cv = [r[1] for r in results if r[0]==0][0]
print(f"\n=== INTERNAL-BEST cfg{best_i} cv={best_cv:.5f} (baseline cv={base_cv:.5f}) ===", flush=True)
print("BEST CONFIG:", {k:best_cfg[k] for k in ['num_leaves','min_child_samples','learning_rate','n_estimators','reg_alpha','reg_lambda','subsample','colsample_bytree']}, flush=True)

# ---- Retrain on full 500k train, evaluate hold-out + external for BOTH configs ----
Xtr = kchs.iloc[tr][FEATURES]; ytr = y[tr]
Xho = kchs.iloc[ho][FEATURES]; yho = y[ho]
Xpre = pre[FEATURES]; ypre = pre["dep"].astype(int).values; wpre = pre["W"].astype(float).values
Xpost= post[FEATURES]; ypost= post["dep"].astype(int).values; wpost= post["W"].astype(float).values

def eval_cfg(cfg, name):
    pipe = Pipeline([("pre", make_pre()), ("clf", LGBMClassifier(**cfg))])
    pipe.fit(Xtr, ytr)  # preprocessing fit on KCHS TRAIN only
    ph = pipe.predict_proba(Xho)[:,1]
    pp = pipe.predict_proba(Xpre)[:,1]
    pq = pipe.predict_proba(Xpost)[:,1]
    internal = roc_auc_score(yho, ph)
    pre_a = roc_auc_score(ypre, pp, sample_weight=wpre)
    post_a= roc_auc_score(ypost, pq, sample_weight=wpost)
    print(f"[{name}] internal={internal:.5f} pre={pre_a:.5f} post={post_a:.5f}  [{time.time()-t0:.0f}s]", flush=True)
    return internal, pre_a, post_a

base_int, base_pre, base_post = eval_cfg(BASELINE, "BASELINE @500k")
best_int, best_pre, best_post = eval_cfg(best_cfg, f"INTERNAL-BEST cfg{best_i} @500k")

print("\n=== SUMMARY ===")
print(f"baseline_internal_cv={base_cv:.5f}  best_internal_cv={best_cv:.5f}")
print(f"BASELINE     internal={base_int:.5f} pre={base_pre:.5f} post={base_post:.5f}")
print(f"INTERNAL-BEST internal={best_int:.5f} pre={best_pre:.5f} post={best_post:.5f}")
print(f"delta_external pre={best_pre-base_pre:+.5f} post={best_post-base_post:+.5f}")
print(f"beats_baseline_external={(best_pre>base_pre+0.001) and (best_post>base_post+0.001)}")
print(f"DONE {time.time()-t0:.0f}s")
