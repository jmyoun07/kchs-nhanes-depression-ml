# -*- coding: utf-8 -*-
"""46_review_fixes.py — 2차 리뷰 지적사항 근거 산출.

(1) 하위군 보정 slope를 파일로 저장(기존엔 콘솔 출력만 → 추적 불가)
(2) 하위군 AUROC에 층별 군집 부트스트랩 CI 부여(§3.8이 CI 없이 "wide CIs"라고만 서술)
(3) NHANES 자기적합 오라클을 학습기별로 재산출 + 부트스트랩 CI(§3.5의 "0.765–0.775", "CI ≈0.75–0.79" 출처 부재)

출력: results/subgroup_calibration.csv, results/fairness_ci.csv, results/oracle_ci.csv
"""
import sys, warnings, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

D = r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model"; RES = f"{D}/results"; SEED = 42
kchs = pd.read_parquet(f"{D}/data/ext_kchs.parquet"); pre = pd.read_parquet(f"{D}/data/ext_pre.parquet"); post = pd.read_parquet(f"{D}/data/ext_post.parquet")
NUM = ["age","sleep_hrs","bmi","income_q","srh","oral_health"]
BIN = ["partnered","living_alone","current_drinker","active","htn","dm","employed","food_insecure"]
CAT = ["sex","edu","smoking"]; FE = NUM+BIN+CAT
y = kchs["dep"].astype(int).values
LGB = dict(num_leaves=31, min_child_samples=1000, learning_rate=0.01, n_estimators=900, reg_alpha=1.0,
           reg_lambda=20.0, subsample=0.6, colsample_bytree=0.6, random_state=SEED, n_jobs=-1, verbose=-1)
def make_ct(NU=NUM, BI=BIN, CA=CAT, native=True):
    t = [("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), NU),
         ("bin", "passthrough" if native else SimpleImputer(strategy="most_frequent"), BI),
         ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")), ("o", OneHotEncoder(handle_unknown="ignore", drop="first"))]), CA)]
    return ColumnTransformer(t)

class ClusterBoot:
    """층 내 PSU 복원추출(Rao-Wu)."""
    def __init__(self, strata, psu):
        cl = pd.factorize(pd.Series(strata).astype(str)+"|"+pd.Series(psu).astype(str))[0]
        st = pd.factorize(pd.Series(strata).astype(str))[0]
        o = np.argsort(cl, kind="stable"); self.members = o
        self.counts = np.bincount(cl); self.offs = np.concatenate([[0], np.cumsum(self.counts)])
        first = np.zeros(len(self.counts), dtype=int); first[cl[o]] = st[o]
        o2 = np.argsort(first, kind="stable"); self.cl_sorted = o2
        sc = np.bincount(first[o2]); start = np.concatenate([[0], np.cumsum(sc)])[:-1]
        self.slot_start = start[first[o2]]; self.slot_n = sc[first[o2]]; self.C = len(self.counts)
    def draw(self, rng):
        cs = self.cl_sorted[self.slot_start + (rng.random_sample(self.C)*self.slot_n).astype(int)]
        L = self.counts[cs]; tot = int(L.sum())
        return self.members[np.repeat(self.offs[cs], L) + np.arange(tot) - np.repeat(np.concatenate([[0], np.cumsum(L)])[:-1], L)]

def auc_ci(d, p, B=1000):
    yv = d["dep"].astype(int).values; w = d["W"].to_numpy(float)
    cb = ClusterBoot(d["strata"].values, d["psu"].values); rng = np.random.RandomState(SEED); a = []
    for _ in range(B):
        ii = cb.draw(rng)
        try: a.append(roc_auc_score(yv[ii], p[ii], sample_weight=w[ii]))
        except Exception: pass
    a = np.array(a); return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

def cal_slope(yv, p, w):
    p = np.clip(p, 1e-6, 1-1e-6); lp = np.log(p/(1-p)).reshape(-1,1)
    return float(LogisticRegression(penalty=None, solver="lbfgs").fit(lp, yv, sample_weight=w).coef_[0,0])

tr, ho = train_test_split(np.arange(len(kchs)), train_size=500000, test_size=100000, stratify=y, random_state=SEED)
model = Pipeline([("pre", make_ct()), ("clf", lgb.LGBMClassifier(**LGB))]).fit(kchs.iloc[tr][FE], y[tr])
dho = kchs.iloc[ho].copy(); dho["_p"] = model.predict_proba(dho[FE])[:,1]
prd = pre.copy(); prd["_p"] = model.predict_proba(pre[FE])[:,1]
age = pd.to_numeric(dho["age"])

# ══════════ (1)(2) 하위군: 보정 slope + AUROC CI ══════════
print("="*82); print("[1-2] 하위군 보정 slope + AUROC 층별 군집 부트스트랩 CI"); print("="*82, flush=True)
groups = [("Sex: female", dho, dho["sex"]=="F"), ("Sex: male", dho, dho["sex"]=="M"),
          ("Age 20-39", dho, age<40), ("Age 40-59", dho, (age>=40)&(age<60)), ("Age 60+", dho, age>=60),
          ("Education: college", dho, dho["edu"]=="college"), ("Education: high school", dho, dho["edu"]=="high"),
          ("Education: middle", dho, dho["edu"]=="middle"), ("Education: <=elementary", dho, dho["edu"]=="elem_low")]
RACE = {1:"Mexican American", 2:"Other Hispanic", 3:"Non-Hispanic White", 4:"Non-Hispanic Black", 6:"Non-Hispanic Asian", 7:"Other/multiracial"}
for code, lab in RACE.items():
    groups.append((f"NHANES pre race: {lab}", prd, prd["race"]==code))
rows = []
for lab, src, mask in groups:
    dd = src[mask]
    if dd["dep"].nunique() < 2 or len(dd) < 100: continue
    yv = dd["dep"].astype(int).values; w = dd["W"].to_numpy(float); p = dd["_p"].values
    a = roc_auc_score(yv, p, sample_weight=w); lo, hi = auc_ci(dd, p, B=1000)
    sl = cal_slope(yv, p, w)
    cohort = "KCHS hold-out" if src is dho else "NHANES pre"
    print(f"  {lab:<32} n={len(dd):>6,}  AUROC {a:.3f} [{lo:.3f},{hi:.3f}]  slope {sl:.3f}", flush=True)
    rows.append(dict(cohort=cohort, subgroup=lab, n=len(dd), AUROC=round(a,3), CI_lo=round(lo,3), CI_hi=round(hi,3), cal_slope=round(sl,3)))
sub = pd.DataFrame(rows)
sub.to_csv(f"{RES}/fairness_ci.csv", index=False, encoding="utf-8-sig")
sub[sub.cohort=="KCHS hold-out"][["subgroup","n","cal_slope"]].to_csv(f"{RES}/subgroup_calibration.csv", index=False, encoding="utf-8-sig")

# ══════════ (3) NHANES 자기적합 오라클: 학습기별 + 부트스트랩 CI ══════════
print("\n" + "="*82); print("[3] NHANES 자기적합 오라클(진단 목적) — 학습기별 5-fold CV + 부트스트랩 CI"); print("="*82, flush=True)
pooled = pd.concat([pre.assign(_period="pre"), post.assign(_period="post")], ignore_index=True)
pooled["strata"] = pooled["_period"].astype(str) + "|" + pooled["strata"].astype(str)   # 주기별 설계 분리
yo = pooled["dep"].astype(int).values; Wo = pooled["W"].to_numpy(float)
LEARNERS = {
    "Logistic regression": lambda: Pipeline([("pre", make_ct(native=False)), ("clf", LogisticRegression(max_iter=2000))]),
    "Penalized logistic (L2, C=0.1)": lambda: Pipeline([("pre", make_ct(native=False)), ("clf", LogisticRegression(C=0.1, max_iter=2000))]),
    "Histogram gradient boosting": lambda: Pipeline([("pre", make_ct()), ("clf", HistGradientBoostingClassifier(random_state=SEED, max_iter=300, learning_rate=0.05))]),
    "LightGBM (small)": lambda: Pipeline([("pre", make_ct()), ("clf", lgb.LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, num_leaves=7, min_child_samples=200, learning_rate=0.05, n_estimators=300))]),
}
orows = []; best_oof = None; best_auc = -1
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
for nm, mk in LEARNERS.items():
    oof = np.zeros(len(pooled))
    for tri, tei in skf.split(pooled[FE], yo):
        m = mk().fit(pooled.iloc[tri][FE], yo[tri])
        oof[tei] = m.predict_proba(pooled.iloc[tei][FE])[:,1]
    a = roc_auc_score(yo, oof, sample_weight=Wo)
    lo, hi = auc_ci(pooled, oof, B=1000)
    print(f"  {nm:<34} oracle AUROC {a:.4f}  95% CI [{lo:.3f},{hi:.3f}]", flush=True)
    orows.append(dict(learner=nm, oracle_AUROC=round(a,4), CI_lo=round(lo,3), CI_hi=round(hi,3)))
    if a > best_auc: best_auc, best_oof = a, oof
ap = LEARNERS["LightGBM (small)"]().fit(pooled[FE], yo)
appar = roc_auc_score(yo, ap.predict_proba(pooled[FE])[:,1], sample_weight=Wo)
orows.append(dict(learner="In-sample (apparent) upper bound", oracle_AUROC=round(appar,4), CI_lo=None, CI_hi=None))
print(f"  {'In-sample (apparent) upper bound':<34} {appar:.4f}", flush=True)
pd.DataFrame(orows).to_csv(f"{RES}/oracle_ci.csv", index=False, encoding="utf-8-sig")
lo_b, hi_b = auc_ci(pooled, best_oof, B=1000)
print(f"\n  → 학습기별 오라클 범위 {min(r['oracle_AUROC'] for r in orows[:-1]):.3f}–{max(r['oracle_AUROC'] for r in orows[:-1]):.3f} · 최고 학습기 CI [{lo_b:.3f},{hi_b:.3f}]", flush=True)
print("\n저장: results/fairness_ci.csv, subgroup_calibration.csv, oracle_ci.csv\n완료.", flush=True)
