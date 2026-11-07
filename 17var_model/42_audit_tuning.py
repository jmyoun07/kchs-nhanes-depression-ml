# -*- coding: utf-8 -*-
"""42_audit_tuning.py — 원고 감사 지적사항 재계산 (B2·B5).

B2 알고리즘 비교 대등화: elastic net·random forest도 LightGBM과 '동일한 내부 CV 프로토콜'로
   하이퍼파라미터를 탐색한 뒤 최고 성능끼리 비교(기존은 LightGBM만 튜닝 상태로 비교).
B5 14변수 비교모델 자체 튜닝: 14변수 특징셋에 대해 roundF와 동일한 랜덤서치를 다시 수행하여
   각 모델이 '자기 최적' 설정을 갖도록 한 뒤 14→17 이득을 재추정(층별 군집 짝지음 부트스트랩).

출력: results/audit_algo_comparison.csv, results/audit_gain_retuned.csv
"""
import sys, time, warnings, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

D = r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model"; RES = f"{D}/results"; SEED = 42
kchs = pd.read_parquet(f"{D}/data/ext_kchs.parquet")
pre  = pd.read_parquet(f"{D}/data/ext_pre.parquet")
post = pd.read_parquet(f"{D}/data/ext_post.parquet")
NUM = ["age","sleep_hrs","bmi","income_q","srh","oral_health"]
BIN = ["partnered","living_alone","current_drinker","active","htn","dm","employed","food_insecure"]
CAT = ["sex","edu","smoking"]; FE = NUM+BIN+CAT
NUM14 = ["age","sleep_hrs","bmi","income_q","srh"]
BIN14 = ["partnered","living_alone","current_drinker","active","htn","dm"]
CAT14 = ["sex","edu","smoking"]; FE14 = NUM14+BIN14+CAT14
y = kchs["dep"].astype(int).values

def make_ct(NU, BI, CA, native=True):
    binp = "passthrough" if native else SimpleImputer(strategy="most_frequent", add_indicator=True)
    t = []
    if NU: t.append(("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), NU))
    if BI: t.append(("bin", binp, BI))
    if CA: t.append(("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                                      ("o", OneHotEncoder(handle_unknown="ignore", drop="first"))]), CA))
    return ColumnTransformer(t)

tr, ho = train_test_split(np.arange(len(kchs)), train_size=500000, test_size=100000, stratify=y, random_state=SEED)
Who, yho = kchs.iloc[ho]["W"].to_numpy(float), y[ho]
Wp, Ws = pre["W"].to_numpy(float), post["W"].to_numpy(float)
yp, ys = pre["dep"].astype(int).values, post["dep"].astype(int).values

# ══════════════════════════════════════════════════════════════════════════
# B2. 3개 알고리즘 동일 프로토콜 튜닝 (내부 5-fold CV, 학습셋 내 20만 부분표본)
# ══════════════════════════════════════════════════════════════════════════
print("="*84); print("[B2] 알고리즘 비교 대등화 — 세 알고리즘 모두 동일한 내부 CV로 탐색"); print("="*84, flush=True)
sub = np.random.RandomState(SEED).permutation(tr)[:200000]
Xs, ys_ = kchs.iloc[sub], y[sub]
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)

LGB_GRID = [
    dict(num_leaves=31, min_child_samples=1000, learning_rate=0.01, n_estimators=900,  reg_alpha=1.0, reg_lambda=20.0, subsample=0.6, colsample_bytree=0.6),  # 배포 설정(roundF cfg#0)
    dict(num_leaves=63, min_child_samples=500,  learning_rate=0.02, n_estimators=600,  reg_alpha=0.5, reg_lambda=5.0,  subsample=0.8, colsample_bytree=0.8),
    dict(num_leaves=15, min_child_samples=200,  learning_rate=0.03, n_estimators=900,  reg_alpha=0.0, reg_lambda=1.0,  subsample=1.0, colsample_bytree=0.8),
    dict(num_leaves=31, min_child_samples=3000, learning_rate=0.02, n_estimators=1500, reg_alpha=5.0, reg_lambda=20.0, subsample=0.6, colsample_bytree=1.0),
    dict(num_leaves=7,  min_child_samples=50,   learning_rate=0.05, n_estimators=400,  reg_alpha=0.0, reg_lambda=0.0,  subsample=0.8, colsample_bytree=0.6),
]
RF_GRID = [
    dict(n_estimators=300, max_depth=14,   min_samples_leaf=100),   # 기존 설정
    dict(n_estimators=300, max_depth=10,   min_samples_leaf=100),
    dict(n_estimators=300, max_depth=20,   min_samples_leaf=200),
    dict(n_estimators=300, max_depth=None, min_samples_leaf=500),
    dict(n_estimators=300, max_depth=25,   min_samples_leaf=50),
]
EN_GRID = [
    dict(C=1.0,  l1_ratio=0.5),   # 기존 설정
    dict(C=0.1,  l1_ratio=0.1),
    dict(C=0.1,  l1_ratio=0.9),
    dict(C=0.01, l1_ratio=0.5),
    dict(C=10.0, l1_ratio=0.5),
]

def cv_auc(mdl, feats=FE):
    oof = cross_val_predict(mdl, Xs[feats], ys_, cv=skf, method="predict_proba", n_jobs=-1)[:, 1]
    return roc_auc_score(ys_, oof)

rows = []
for algo, grid in [("lightgbm", LGB_GRID), ("randomforest", RF_GRID), ("elasticnet", EN_GRID)]:
    best = (-1, None)
    for i, g in enumerate(grid):
        t0 = time.time()
        if algo == "lightgbm":
            mdl = Pipeline([("pre", make_ct(NUM,BIN,CAT)), ("clf", lgb.LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, **g))])
        elif algo == "randomforest":
            mdl = Pipeline([("pre", make_ct(NUM,BIN,CAT)), ("clf", RandomForestClassifier(random_state=SEED, n_jobs=-1, **g))])
        else:
            mdl = Pipeline([("pre", make_ct(NUM,BIN,CAT, native=False)),
                            ("clf", LogisticRegression(penalty="elasticnet", solver="saga", max_iter=300, n_jobs=-1, **g))])
        a = cv_auc(mdl)
        rows.append(dict(algorithm=algo, cfg=i, params=str(g), cv_auroc=round(a,4)))
        print(f"   {algo:<13} cfg{i}  CV-AUROC={a:.4f}  ({time.time()-t0:.0f}s)  {g}", flush=True)
        if a > best[0]: best = (a, g)
    print(f"   ★ {algo}: 최고 CV-AUROC={best[0]:.4f}  {best[1]}\n", flush=True)
pd.DataFrame(rows).to_csv(f"{RES}/audit_algo_comparison.csv", index=False, encoding="utf-8-sig")
bestper = pd.DataFrame(rows).groupby("algorithm")["cv_auroc"].max().sort_values(ascending=False)
print("[B2 결론] 동일 프로토콜 최고 CV-AUROC:", {k: round(v,4) for k,v in bestper.items()}, flush=True)

# ══════════════════════════════════════════════════════════════════════════
# B5. 14변수 모델 자체 튜닝 후 14 vs 17 재비교
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*84); print("[B5] 14변수 모델 자체 튜닝 — roundF와 동일한 랜덤서치(24조합·4-fold 가중 CV·30만)"); print("="*84, flush=True)
rng = np.random.RandomState(SEED); GRID = []
for _ in range(24):
    GRID.append(dict(
        num_leaves=int(rng.choice([7,15,31,63])), min_child_samples=int(rng.choice([50,200,500,1000,3000])),
        learning_rate=float(rng.choice([0.01,0.02,0.03,0.05])), n_estimators=int(rng.choice([400,600,900,1500])),
        reg_alpha=float(rng.choice([0,0.5,1.0,5.0])), reg_lambda=float(rng.choice([0,1.0,5.0,20.0])),
        subsample=float(rng.choice([0.6,0.8,1.0])), colsample_bytree=float(rng.choice([0.6,0.8,1.0]))))

cvidx = np.random.RandomState(1).choice(tr, 300000, replace=False)
Xcv = kchs.iloc[cvidx]; ycv = y[cvidx]; Wcv = Xcv["W"].to_numpy(float)
skf4 = StratifiedKFold(4, shuffle=True, random_state=SEED)
res14 = []
for gi, g in enumerate(GRID):
    t0 = time.time(); oof = np.zeros(len(Xcv))
    for tri, vai in skf4.split(Xcv[FE14], ycv):
        m = Pipeline([("pre", make_ct(NUM14,BIN14,CAT14)), ("clf", lgb.LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, **g))]).fit(Xcv.iloc[tri][FE14], ycv[tri])
        oof[vai] = m.predict_proba(Xcv.iloc[vai][FE14])[:,1]
    a = roc_auc_score(ycv, oof, sample_weight=Wcv)
    res14.append((gi, a, g)); print(f"   14var cfg{gi:>2}  CV-AUROC={a:.4f}  ({time.time()-t0:.0f}s)", flush=True)
res14.sort(key=lambda r: -r[1])
gi_best, cv_best, g14 = res14[0]
print(f"\n   ★ 14변수 최적: cfg{gi_best}  CV-AUROC={cv_best:.4f}  {g14}", flush=True)

LGB17 = dict(num_leaves=31, min_child_samples=1000, learning_rate=0.01, n_estimators=900,
             reg_alpha=1.0, reg_lambda=20.0, subsample=0.6, colsample_bytree=0.6)
m17 = Pipeline([("pre", make_ct(NUM,BIN,CAT)),      ("clf", lgb.LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, **LGB17))]).fit(kchs.iloc[tr][FE], y[tr])
m14 = Pipeline([("pre", make_ct(NUM14,BIN14,CAT14)),("clf", lgb.LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, **g14))]).fit(kchs.iloc[tr][FE14], y[tr])
m14o= Pipeline([("pre", make_ct(NUM14,BIN14,CAT14)),("clf", lgb.LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, **LGB17))]).fit(kchs.iloc[tr][FE14], y[tr])

class ClusterBoot:
    def __init__(self, strata, psu):
        cl = pd.factorize(pd.Series(strata).astype(str) + "|" + pd.Series(psu).astype(str))[0]
        st = pd.factorize(pd.Series(strata).astype(str))[0]
        order = np.argsort(cl, kind="stable"); self.members = order
        counts = np.bincount(cl); self.counts = counts
        self.offs = np.concatenate([[0], np.cumsum(counts)])
        first = np.zeros(len(counts), dtype=int); first[cl[order]] = st[order]; self.cl_stratum = first
        o2 = np.argsort(self.cl_stratum, kind="stable"); self.cl_sorted = o2
        s_sorted = self.cl_stratum[o2]; sc = np.bincount(s_sorted)
        self.st_start = np.concatenate([[0], np.cumsum(sc)])[:-1]; self.st_n = sc
        self.slot_start = self.st_start[s_sorted]; self.slot_n = self.st_n[s_sorted]; self.C = len(counts)
    def draw(self, rng):
        u = rng.random_sample(self.C)
        cs = self.cl_sorted[self.slot_start + (u*self.slot_n).astype(int)]
        L = self.counts[cs]; tot = int(L.sum())
        starts = np.repeat(self.offs[cs], L)
        run = np.arange(tot) - np.repeat(np.concatenate([[0], np.cumsum(L)])[:-1], L)
        return self.members[starts + run]

def paired(d, yv, w, p17, p14, B=2000):
    cb = ClusterBoot(d["strata"].values, d["psu"].values); rng = np.random.RandomState(SEED); dfs = []
    for _ in range(B):
        ii = cb.draw(rng)
        try: dfs.append(roc_auc_score(yv[ii],p17[ii],sample_weight=w[ii]) - roc_auc_score(yv[ii],p14[ii],sample_weight=w[ii]))
        except Exception: pass
    dfs = np.array(dfs); lo, hi = np.percentile(dfs, [2.5, 97.5])
    return float(lo), float(hi), float(2*min((dfs<=0).mean(), (dfs>=0).mean()))

grows = []
print("\n   [14→17 이득 재추정 — 14변수도 자체 최적 설정 사용, 층별 군집 짝지음 부트스트랩]", flush=True)
i17 = roc_auc_score(yho, m17.predict_proba(kchs.iloc[ho][FE])[:,1], sample_weight=Who)
i14 = roc_auc_score(yho, m14.predict_proba(kchs.iloc[ho][FE14])[:,1], sample_weight=Who)
i14o= roc_auc_score(yho, m14o.predict_proba(kchs.iloc[ho][FE14])[:,1], sample_weight=Who)
print(f"   내부: 14(자체튜닝) {i14:.4f} / 14(17설정) {i14o:.4f} → 17 {i17:.4f}", flush=True)
grows.append(dict(set="Internal(hold-out)", AUROC_14_retuned=round(i14,4), AUROC_14_shared=round(i14o,4), AUROC_17=round(i17,4),
                  delta_retuned=round(i17-i14,4), CI_lo=None, CI_hi=None, p=None))
for nm, d, yv, w, f17, f14 in [("NHANES pre", pre, yp, Wp, FE, FE14), ("NHANES post", post, ys, Ws, FE, FE14)]:
    p17 = m17.predict_proba(d[f17])[:,1]; p14 = m14.predict_proba(d[f14])[:,1]; p14o = m14o.predict_proba(d[f14])[:,1]
    a17 = roc_auc_score(yv,p17,sample_weight=w); a14 = roc_auc_score(yv,p14,sample_weight=w); a14o = roc_auc_score(yv,p14o,sample_weight=w)
    lo, hi, pv = paired(d, yv, w, p17, p14)
    print(f"   {nm}: 14(자체튜닝) {a14:.4f} / 14(17설정) {a14o:.4f} → 17 {a17:.4f}  Δ={a17-a14:+.4f} 95%CI[{lo:+.4f},{hi:+.4f}] p≈{pv:.4f}", flush=True)
    grows.append(dict(set=nm, AUROC_14_retuned=round(a14,4), AUROC_14_shared=round(a14o,4), AUROC_17=round(a17,4),
                      delta_retuned=round(a17-a14,4), CI_lo=round(lo,4), CI_hi=round(hi,4), p=round(pv,4)))
pd.DataFrame(grows).to_csv(f"{RES}/audit_gain_retuned.csv", index=False, encoding="utf-8-sig")
pd.DataFrame([dict(featset="14var", cfg=gi_best, cv_auroc=round(cv_best,4), params=str(g14))]).to_csv(f"{RES}/audit_14var_tuning.csv", index=False, encoding="utf-8-sig")
print("\n저장: results/audit_algo_comparison.csv, audit_gain_retuned.csv, audit_14var_tuning.csv\n완료.", flush=True)
