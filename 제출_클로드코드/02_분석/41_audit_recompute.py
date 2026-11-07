# -*- coding: utf-8 -*-
"""41_audit_recompute.py — 원고 감사 지적사항 재계산 (B3·B4·B6·B8·B9).

B3 CITL: slope 고정(offset) 방식의 정식 calibration-in-the-large 재계산
B4 age>=30 민감도: 평가도 >=30으로 제한한 정식 버전 + 기존(전체 평가) 병기
B6 경량모델: 하드코딩 순서 → shap_importance.csv 기반 순서로 재계산
B8 부트스트랩: 층 내 PSU 복원추출(Rao-Wu 층별 군집 부트스트랩)로 CI 재계산
B9 컷오프>=5: 외부(pre/post) 미산출분 산출

출력: results/audit_recompute_{metrics,parsimonious,sensitivity}.csv
"""
import sys, warnings, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import lightgbm as lgb

D = r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model"; RES = f"{D}/results"; SEED = 42
kchs = pd.read_parquet(f"{D}/data/ext_kchs.parquet")
pre  = pd.read_parquet(f"{D}/data/ext_pre.parquet")
post = pd.read_parquet(f"{D}/data/ext_post.parquet")
NUM = ["age","sleep_hrs","bmi","income_q","srh","oral_health"]
BIN = ["partnered","living_alone","current_drinker","active","htn","dm","employed","food_insecure"]
CAT = ["sex","edu","smoking"]; FE = NUM+BIN+CAT
y = kchs["dep"].astype(int).values
LGB = dict(num_leaves=31, min_child_samples=1000, learning_rate=0.01, n_estimators=900,
           reg_alpha=1.0, reg_lambda=20.0, subsample=0.6, colsample_bytree=0.6,
           random_state=SEED, n_jobs=-1, verbose=-1)

def make_ct(NU=NUM, BI=BIN, CA=CAT):
    t = []
    if NU: t.append(("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), NU))
    if BI: t.append(("bin", "passthrough", BI))
    if CA: t.append(("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                                      ("o", OneHotEncoder(handle_unknown="ignore", drop="first"))]), CA))
    return ColumnTransformer(t)

tr, ho = train_test_split(np.arange(len(kchs)), train_size=500000, test_size=100000, stratify=y, random_state=SEED)
Who, yho = kchs.iloc[ho]["W"].to_numpy(float), y[ho]
print(f"[분할] 전체 {len(kchs):,} → 학습 {len(tr):,} / hold-out {len(ho):,} (미사용 {len(kchs)-len(tr)-len(ho):,})", flush=True)

model = Pipeline([("pre", make_ct()), ("clf", lgb.LGBMClassifier(**LGB))]).fit(kchs.iloc[tr][FE], y[tr])
def P(d): return model.predict_proba(d[FE])[:, 1]
pi, pp, ps = P(kchs.iloc[ho]), P(pre), P(post)
Wp, Ws = pre["W"].to_numpy(float), post["W"].to_numpy(float)
yp, ys = pre["dep"].astype(int).values, post["dep"].astype(int).values
print(f"[재현 확인] 내부 {roc_auc_score(yho,pi,sample_weight=Who):.4f} / pre {roc_auc_score(yp,pp,sample_weight=Wp):.4f} / post {roc_auc_score(ys,ps,sample_weight=Ws):.4f}", flush=True)

# ══════════════════════════════════════════════════════════════════
# B3. 정식 CITL — slope 고정(offset) vs 기존(slope 자유추정 절편)
# ══════════════════════════════════════════════════════════════════
def logit(p): p = np.clip(p, 1e-6, 1-1e-6); return np.log(p/(1-p))

def cal_slope_intercept(yv, p, w):
    """slope·intercept 동시 추정 (기존 방식). 절편은 'calibration intercept'."""
    c = LogisticRegression(penalty=None, solver="lbfgs").fit(logit(p).reshape(-1,1), yv, sample_weight=w)
    return float(c.coef_[0,0]), float(c.intercept_[0])

def citl_offset(yv, p, w):
    """정식 CITL: 선형예측자를 offset으로 고정(slope=1)한 절편만 추정."""
    lp = logit(p)
    # offset 로지스틱: logit(pi) = a + lp  →  뉴턴-랩슨으로 a 추정
    a = 0.0
    for _ in range(200):
        eta = a + lp; mu = 1/(1+np.exp(-eta))
        g = np.sum(w*(yv-mu)); h = np.sum(w*mu*(1-mu))
        if h <= 0: break
        step = g/h; a += step
        if abs(step) < 1e-10: break
    return float(a)

print("\n" + "="*78); print("[B3] CITL 재계산 — 정식(offset, slope=1 고정) vs 기존(slope 자유추정 절편)"); print("="*78, flush=True)
def recalib_cv(yv, p, w):
    """5-fold 교차적합 절편/기울기 재보정 (기존 파이프라인과 동일)."""
    lp = logit(p); oof = np.zeros_like(p)
    for tri, tei in StratifiedKFold(5, shuffle=True, random_state=SEED).split(lp, yv):
        c = LogisticRegression(penalty=None, solver="lbfgs").fit(lp[tri].reshape(-1,1), yv[tri], sample_weight=w[tri])
        oof[tei] = c.predict_proba(lp[tei].reshape(-1,1))[:,1]
    return oof

cal_rows = []
sets = [("Internal(hold-out)", yho, pi, Who), ("NHANES pre", yp, pp, Wp), ("NHANES post", ys, ps, Ws)]
for nm, yv, p, w in sets:
    sl, ic = cal_slope_intercept(yv, p, w); citl = citl_offset(yv, p, w)
    pr = recalib_cv(yv, p, w)
    sl_r, ic_r = cal_slope_intercept(yv, pr, w); citl_r = citl_offset(yv, pr, w)
    print(f"  {nm:<20} slope={sl:.3f}  절편(기존)={ic:+.3f}  CITL(정식)={citl:+.3f}   │ 재보정후 slope={sl_r:.3f} CITL={citl_r:+.3f}", flush=True)
    cal_rows.append(dict(set=nm, cal_slope=round(sl,3), cal_intercept_free=round(ic,3), CITL_offset=round(citl,3),
                         recal_slope=round(sl_r,3), recal_CITL_offset=round(citl_r,3),
                         AUROC=round(roc_auc_score(yv,p,sample_weight=w),4),
                         AUPRC=round(average_precision_score(yv,p,sample_weight=w),3),
                         Brier=round(brier_score_loss(yv,p,sample_weight=w),4),
                         recal_AUROC=round(roc_auc_score(yv,pr,sample_weight=w),4),
                         recal_AUPRC=round(average_precision_score(yv,pr,sample_weight=w),3),
                         recal_Brier=round(brier_score_loss(yv,pr,sample_weight=w),4)))

# ══════════════════════════════════════════════════════════════════
# B8. 층별 군집 부트스트랩 (층 내 PSU 복원추출) — 기존 풀링 방식과 병기
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*78); print("[B8] 부트스트랩 재계산 — 층 내 PSU 복원추출(Rao-Wu) vs 기존(층 무시 풀링)"); print("="*78, flush=True)

class ClusterBoot:
    """층별 군집 부트스트랩: 각 층 h에서 n_h개 PSU를 복원추출."""
    def __init__(self, strata, psu):
        cl = pd.factorize(pd.Series(strata).astype(str) + "|" + pd.Series(psu).astype(str))[0]
        st = pd.factorize(pd.Series(strata).astype(str))[0]
        order = np.argsort(cl, kind="stable")
        self.members = order
        counts = np.bincount(cl)
        self.offs = np.concatenate([[0], np.cumsum(counts)])
        self.counts = counts
        # 군집별 소속 층
        first = np.zeros(len(counts), dtype=int)
        first[cl[order]] = st[order]
        self.cl_stratum = first
        # 층별 군집 범위(층 기준 정렬)
        o2 = np.argsort(self.cl_stratum, kind="stable")
        self.cl_sorted = o2                      # 층순 정렬된 군집 인덱스
        s_sorted = self.cl_stratum[o2]
        sc = np.bincount(s_sorted)
        self.st_start = np.concatenate([[0], np.cumsum(sc)])[:-1]
        self.st_n = sc
        self.slot_start = self.st_start[s_sorted]  # 각 슬롯의 층 시작
        self.slot_n = self.st_n[s_sorted]
        self.C = len(counts)

    def draw(self, rng):
        u = rng.random_sample(self.C)
        pick = self.slot_start + (u*self.slot_n).astype(int)      # 층 내 균등 복원추출
        cs = self.cl_sorted[pick]
        L = self.counts[cs]; tot = int(L.sum())
        starts = np.repeat(self.offs[cs], L)
        run = np.arange(tot) - np.repeat(np.concatenate([[0], np.cumsum(L)])[:-1], L)
        return self.members[starts + run]

def boot_ci(d, p, yv, w, fn, B=1000, stratified=True):
    cb = ClusterBoot(d["strata"].values, d["psu"].values)
    rng = np.random.RandomState(SEED); a = []
    if stratified:
        for _ in range(B):
            ii = cb.draw(rng)
            try: a.append(fn(yv[ii], p[ii], w[ii]))
            except Exception: pass
    else:
        grp = (d["strata"].astype(str)+"|"+d["psu"].astype(str)).values
        uid = pd.unique(grp); idx = {u: np.where(grp==u)[0] for u in uid}
        for _ in range(B):
            ii = np.concatenate([idx[u] for u in rng.choice(uid, len(uid), replace=True)])
            try: a.append(fn(yv[ii], p[ii], w[ii]))
            except Exception: pass
    a = np.array(a); return float(np.percentile(a,2.5)), float(np.percentile(a,97.5))

AUC = lambda yv,p,w: roc_auc_score(yv,p,sample_weight=w)
APR = lambda yv,p,w: average_precision_score(yv,p,sample_weight=w)
boot_rows = []
for nm, d, yv, p, w in [("Internal(hold-out)", kchs.iloc[ho], yho, pi, Who), ("NHANES pre", pre, yp, pp, Wp), ("NHANES post", post, ys, ps, Ws)]:
    lo_s, hi_s = boot_ci(d, p, yv, w, AUC, stratified=True)
    lo_p, hi_p = boot_ci(d, p, yv, w, AUC, stratified=False)
    alo, ahi = boot_ci(d, p, yv, w, APR, stratified=True)
    print(f"  {nm:<20} AUROC 층별CI [{lo_s:.4f},{hi_s:.4f}]   (기존 풀링CI [{lo_p:.4f},{hi_p:.4f}])   AUPRC 층별CI [{alo:.3f},{ahi:.3f}]", flush=True)
    boot_rows.append(dict(set=nm, AUROC=round(AUC(yv,p,w),4), CI_lo_stratified=round(lo_s,4), CI_hi_stratified=round(hi_s,4),
                          CI_lo_pooled=round(lo_p,4), CI_hi_pooled=round(hi_p,4),
                          AUPRC=round(APR(yv,p,w),3), AUPRC_lo=round(alo,3), AUPRC_hi=round(ahi,3)))

# ══════════════════════════════════════════════════════════════════
# B6. 경량 모델 — SHAP 중요도 순서로 재계산
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*78); print("[B6] 경량모델 — shap_importance.csv 기반 순서로 재계산"); print("="*78, flush=True)
shp = pd.read_csv(f"{RES}/shap_importance.csv", index_col=0).iloc[:,0].sort_values(ascending=False)
order = [v for v in shp.index if v in FE]
print(f"  SHAP 순서: {order}", flush=True)
prows = []
for k in [3,4,5,8,12,17]:
    f = order[:k]
    NU=[c for c in NUM if c in f]; BI=[c for c in BIN if c in f]; CA=[c for c in CAT if c in f]; FEk=NU+BI+CA
    mk = Pipeline([("pre", make_ct(NU,BI,CA)), ("clf", lgb.LGBMClassifier(**LGB))]).fit(kchs.iloc[tr][FEk], y[tr])
    ik = roc_auc_score(yho, mk.predict_proba(kchs.iloc[ho][FEk])[:,1], sample_weight=Who)
    pk = roc_auc_score(yp, mk.predict_proba(pre[FEk])[:,1], sample_weight=Wp)
    sk = roc_auc_score(ys, mk.predict_proba(post[FEk])[:,1], sample_weight=Ws)
    print(f"  {k:>2}문항 내부 {ik:.4f}  pre {pk:.4f}  post {sk:.4f}   {f}", flush=True)
    prows.append(dict(k=k, vars=";".join(f), internal=round(ik,4), ext_pre=round(pk,4), ext_post=round(sk,4)))
pd.DataFrame(prows).to_csv(f"{RES}/audit_recompute_parsimonious.csv", index=False, encoding="utf-8-sig")

# ══════════════════════════════════════════════════════════════════
# B9 / B4. 컷오프≥5 외부 + 연령≥30 정식 민감도
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*78); print("[B9] 컷오프 ≥5 — 내부·외부 전부 산출 / [B4] 연령≥30 정식 민감도"); print("="*78, flush=True)
srows = []
y5 = (pd.to_numeric(kchs["phq_sum"]) >= 5).astype(int).values
y5p = (pd.to_numeric(pre["phq_sum"]) >= 5).astype(int).values
y5s = (pd.to_numeric(post["phq_sum"]) >= 5).astype(int).values
m5 = Pipeline([("pre", make_ct()), ("clf", lgb.LGBMClassifier(**LGB))]).fit(kchs.iloc[tr][FE], y5[tr])
i5 = roc_auc_score(y5[ho], m5.predict_proba(kchs.iloc[ho][FE])[:,1], sample_weight=Who)
p5 = roc_auc_score(y5p, m5.predict_proba(pre[FE])[:,1], sample_weight=Wp)
s5 = roc_auc_score(y5s, m5.predict_proba(post[FE])[:,1], sample_weight=Ws)
prev5 = [float(np.average(y5[ho],weights=Who)), float(np.average(y5p,weights=Wp)), float(np.average(y5s,weights=Ws))]
print(f"  컷오프≥5   내부 {i5:.4f}  pre {p5:.4f}  post {s5:.4f}   (가중 유병률 {prev5[0]:.3f}/{prev5[1]:.3f}/{prev5[2]:.3f})", flush=True)
srows.append(dict(cond="컷오프≥5(전체 산출)", internal=round(i5,4), ext_pre=round(p5,4), ext_post=round(s5,4)))

ageK = pd.to_numeric(kchs["age"]); ageP = pd.to_numeric(pre["age"]); ageS = pd.to_numeric(post["age"])
tr30 = np.intersect1d(tr, np.where(ageK.values >= 30)[0])
m30 = Pipeline([("pre", make_ct()), ("clf", lgb.LGBMClassifier(**LGB))]).fit(kchs.iloc[tr30][FE], y[tr30])
ho30 = ho[ageK.values[ho] >= 30]; mp30 = ageP.values >= 30; ms30 = ageS.values >= 30
i30 = roc_auc_score(y[ho30], m30.predict_proba(kchs.iloc[ho30][FE])[:,1], sample_weight=kchs.iloc[ho30]["W"].to_numpy(float))
p30 = roc_auc_score(yp[mp30], m30.predict_proba(pre[mp30][FE])[:,1], sample_weight=Wp[mp30])
s30 = roc_auc_score(ys[ms30], m30.predict_proba(post[ms30][FE])[:,1], sample_weight=Ws[ms30])
i30f = roc_auc_score(yho, m30.predict_proba(kchs.iloc[ho][FE])[:,1], sample_weight=Who)
p30f = roc_auc_score(yp, m30.predict_proba(pre[FE])[:,1], sample_weight=Wp)
s30f = roc_auc_score(ys, m30.predict_proba(post[FE])[:,1], sample_weight=Ws)
print(f"  연령≥30(학습·평가 모두 제한)  내부 {i30:.4f}  pre {p30:.4f}  post {s30:.4f}   [n: {len(ho30):,}/{int(mp30.sum())}/{int(ms30.sum())}]", flush=True)
print(f"  연령≥30(학습만 제한, 기존값)  내부 {i30f:.4f}  pre {p30f:.4f}  post {s30f:.4f}", flush=True)
srows.append(dict(cond="연령≥30(학습·평가 모두 제한)", internal=round(i30,4), ext_pre=round(p30,4), ext_post=round(s30,4)))
srows.append(dict(cond="연령≥30(학습만 제한)", internal=round(i30f,4), ext_pre=round(p30f,4), ext_post=round(s30f,4)))
pd.DataFrame(srows).to_csv(f"{RES}/audit_recompute_sensitivity.csv", index=False, encoding="utf-8-sig")

out = pd.DataFrame(cal_rows).merge(pd.DataFrame(boot_rows)[["set","CI_lo_stratified","CI_hi_stratified","CI_lo_pooled","CI_hi_pooled","AUPRC_lo","AUPRC_hi"]], on="set", how="left")
out.to_csv(f"{RES}/audit_recompute_metrics.csv", index=False, encoding="utf-8-sig")
print("\n저장: results/audit_recompute_{metrics,parsimonious,sensitivity}.csv\n완료.", flush=True)
