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
print("train", len(tr), "holdout", len(ho), "prevalence tr/ho",
      y[tr].mean().round(4), y[ho].mean().round(4))

pre_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                     ("oh", OneHotEncoder(handle_unknown="ignore", drop="first"))])
ct = ColumnTransformer([("num", pre_pipe, NUM),
                        ("bin", "passthrough", BIN),
                        ("cat", cat_pipe, CAT)])

clf = LGBMClassifier(num_leaves=31, min_child_samples=1000, learning_rate=0.01,
                     n_estimators=900, reg_alpha=1.0, reg_lambda=20.0,
                     subsample=0.6, colsample_bytree=0.6,
                     random_state=42, n_jobs=-1, verbose=-1)

Xtr = kchs.iloc[tr][FEATURES]
ct.fit(Xtr)   # fit transforms on KCHS TRAIN only
Xtr_t = ct.transform(Xtr)
clf.fit(Xtr_t, y[tr])

def proba(df):
    return clf.predict_proba(ct.transform(df[FEATURES]))[:,1]

proba_ho = proba(kchs.iloc[ho])
proba_pre = proba(pre)
proba_post = proba(post)

internal = roc_auc_score(y[ho], proba_ho)
ypre = pre["dep"].astype(int).values
ypost = post["dep"].astype(int).values
pre_auroc  = roc_auc_score(ypre,  proba_pre,  sample_weight=pre["W"].astype(float))
post_auroc = roc_auc_score(ypost, proba_post, sample_weight=post["W"].astype(float))
print(f"internal(unweighted holdout) = {internal:.4f}")
print(f"pre  weighted AUROC = {pre_auroc:.4f}")
print(f"post weighted AUROC = {post_auroc:.4f}")

# ---- PSU-cluster bootstrap 95% CI (resample strata|psu clusters), 500 reps, weighted AUROC ----
def cluster_boot_ci(df, ytrue, p, w, n_rep=500, seed=42):
    d = pd.DataFrame({"y": ytrue, "p": p, "w": np.asarray(w, float),
                      "strata": df["strata"].values, "psu": df["psu"].values})
    # cluster = unique (strata, psu)
    d["clus"] = d["strata"].astype(str) + "||" + d["psu"].astype(str)
    groups = {k: v.values for k, v in d.groupby("clus").groups.items()}
    clus_ids = np.array(list(groups.keys()))
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_rep):
        pick = rng.choice(len(clus_ids), size=len(clus_ids), replace=True)
        idx = np.concatenate([groups[clus_ids[i]] for i in pick])
        yy = d["y"].values[idx]; pp = d["p"].values[idx]; ww = d["w"].values[idx]
        if yy.min() == yy.max():
            continue
        aucs.append(roc_auc_score(yy, pp, sample_weight=ww))
    aucs = np.array(aucs)
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5), aucs.mean(), len(aucs)

pre_lo, pre_hi, pre_mean, n1 = cluster_boot_ci(pre, ypre, proba_pre, pre["W"].astype(float).values)
post_lo, post_hi, post_mean, n2 = cluster_boot_ci(post, ypost, proba_post, post["W"].astype(float).values)
print(f"pre  95% CI = [{pre_lo:.4f}, {pre_hi:.4f}]  bootmean={pre_mean:.4f}  nreps={n1}")
print(f"post 95% CI = [{post_lo:.4f}, {post_hi:.4f}]  bootmean={post_mean:.4f}  nreps={n2}")
print("DONE")
