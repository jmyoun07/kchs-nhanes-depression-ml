import sys; sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, time, itertools, json
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

t0 = time.time()
kchs = pd.read_parquet(KCHS, columns=FEATURES+["dep","W","year","strata","psu"])
pre  = pd.read_parquet(PRE,  columns=FEATURES+["dep","W","strata","psu"])
post = pd.read_parquet(POST, columns=FEATURES+["dep","W","strata","psu"])
print("shapes", kchs.shape, pre.shape, post.shape, "load", round(time.time()-t0,1),"s")

y = kchs["dep"].astype(int).values
tr, ho = train_test_split(np.arange(len(kchs)), train_size=500000, test_size=100000,
                          stratify=y, random_state=42)
print("train", len(tr), "holdout", len(ho))

def make_ct():
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("oh", OneHotEncoder(handle_unknown="ignore", drop="first"))])
    return ColumnTransformer([("num", num_pipe, NUM),
                              ("bin", "passthrough", BIN),
                              ("cat", cat_pipe, CAT)])

BASE = dict(learning_rate=0.01, n_estimators=900, reg_alpha=1.0,
            subsample=0.6, colsample_bytree=0.6, random_state=42, n_jobs=-1, verbose=-1)

# ---------------- LOYO search (KCHS TRAIN indices ONLY; hold-out never touched) ----------------
# Precompute per-year folds ONCE: fit CT on the fold's (subsampled) training years, transform.
train_df = kchs.iloc[tr].reset_index(drop=True)
ytr = y[tr]
years = np.sort(train_df["year"].unique())
print("years:", list(years))
SUB = 200000  # ~200k total training rows per LOYO fold
rng = np.random.default_rng(42)

folds = {}   # year -> (Xtr_t, ytr_sub, Xhy_t, yhy, why)
for yr in years:
    tmask = (train_df["year"].values != yr)
    hmask = (train_df["year"].values == yr)
    pool = np.where(tmask)[0]
    if len(pool) > SUB:
        pool = rng.choice(pool, size=SUB, replace=False)
    ct = make_ct()
    Xtr_sub = train_df.iloc[pool][FEATURES]
    ct.fit(Xtr_sub)
    Xtr_t = ct.transform(Xtr_sub)
    hidx = np.where(hmask)[0]
    Xhy_t = ct.transform(train_df.iloc[hidx][FEATURES])
    folds[yr] = (Xtr_t, ytr[pool], Xhy_t,
                 ytr[hidx], train_df.iloc[hidx]["W"].astype(float).values)
print("folds prepared", round(time.time()-t0,1),"s")

grid_nl  = [15, 31, 63]
grid_mcs = [500, 1000, 4000]
grid_rl  = [5, 20, 50]
configs = list(itertools.product(grid_nl, grid_mcs, grid_rl))
print("n configs:", len(configs))

results = []
for ci,(nl,mcs,rl) in enumerate(configs):
    yr_aucs = []
    for yr in years:
        Xtr_t, yts, Xhy_t, yhy, why = folds[yr]
        clf = LGBMClassifier(num_leaves=nl, min_child_samples=mcs, reg_lambda=rl, **BASE)
        clf.fit(Xtr_t, yts)
        p = clf.predict_proba(Xhy_t)[:,1]
        yr_aucs.append(roc_auc_score(yhy, p, sample_weight=why))
    worst = float(np.min(yr_aucs)); mean = float(np.mean(yr_aucs))
    results.append(dict(nl=nl, mcs=mcs, rl=rl, worst=worst, mean=mean,
                        per_year={int(k):round(v,4) for k,v in zip(years,yr_aucs)}))
    print(f"[{ci+1:2d}/27] nl={nl:2d} mcs={mcs:4d} rl={rl:2d} worst={worst:.4f} mean={mean:.4f} t={round(time.time()-t0)}s")

results_sorted = sorted(results, key=lambda d: d["worst"], reverse=True)
best = results_sorted[0]
BASE_CFG = dict(nl=31, mcs=1000, rl=20)
base_loyo = next(r for r in results if (r["nl"],r["mcs"],r["rl"])==(31,1000,20))
print("\n=== LOYO ranking (top 5 by worst-year) ===")
for r in results_sorted[:5]:
    print(f"  nl={r['nl']:2d} mcs={r['mcs']:4d} rl={r['rl']:2d} worst={r['worst']:.4f} mean={r['mean']:.4f}")
print(f"BASELINE config LOYO: worst={base_loyo['worst']:.4f} mean={base_loyo['mean']:.4f}")
print(f"SELECTED config     : nl={best['nl']} mcs={best['mcs']} rl={best['rl']} worst={best['worst']:.4f}")

# ---------------- Final: train SELECTED and BASELINE on full 500k train; eval external ----------------
ct = make_ct()
Xtr_full = train_df[FEATURES]
ct.fit(Xtr_full)
Xtr_full_t = ct.transform(Xtr_full)
Xho_t  = ct.transform(kchs.iloc[ho][FEATURES])
Xpre_t = ct.transform(pre[FEATURES]); Xpost_t = ct.transform(post[FEATURES])
ypre = pre["dep"].astype(int).values; wpre = pre["W"].astype(float).values
ypost= post["dep"].astype(int).values; wpost= post["W"].astype(float).values
yho  = y[ho]

def train_eval(cfg, tag):
    clf = LGBMClassifier(num_leaves=cfg["nl"], min_child_samples=cfg["mcs"],
                         reg_lambda=cfg["rl"], **BASE)
    clf.fit(Xtr_full_t, ytr)
    internal = roc_auc_score(yho, clf.predict_proba(Xho_t)[:,1])
    pa = roc_auc_score(ypre,  clf.predict_proba(Xpre_t)[:,1],  sample_weight=wpre)
    poa= roc_auc_score(ypost, clf.predict_proba(Xpost_t)[:,1], sample_weight=wpost)
    print(f"{tag}: internal={internal:.4f} pre={pa:.4f} post={poa:.4f}")
    return internal, pa, poa

print("\n=== FINAL external eval (trained on 500k) ===")
b_int, b_pre, b_post = train_eval(BASE_CFG, "BASELINE(31/1000/20)")
s_int, s_pre, s_post = train_eval(best,     f"SELECTED({best['nl']}/{best['mcs']}/{best['rl']})")

beats = (s_pre > b_pre + 0.001) and (s_post > b_post + 0.001)
print(f"\nbeats_baseline (both pre&post > baseline+0.001): {beats}")
print(f"  d_pre={s_pre-b_pre:+.4f}  d_post={s_post-b_post:+.4f}")

summary = dict(selected=best, baseline_loyo=base_loyo,
               baseline_ext=dict(internal=b_int,pre=b_pre,post=b_post),
               selected_ext=dict(internal=s_int,pre=s_pre,post=s_post),
               beats=bool(beats))
with open("C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model/wf/LOYO_result.json","w") as f:
    json.dump(summary, f, indent=2)
print("DONE total", round(time.time()-t0,1),"s")
