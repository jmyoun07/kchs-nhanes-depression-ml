# -*- coding: utf-8 -*-
"""
03_train.py (프로토콜 준거) — KCHS 2017-2025 개발셋에서 3모델 학습 + 내부검증(hold-out).
- 예측변수: Table 1 조화 14개(자가건강·신체활동 포함). income_q는 연도별 분위.
- 평가는 복합표본 가중(W) — 프로토콜 §2.8(비가중 학습 + 가중 평가). 비가중도 병기.
- 전처리는 train에서만 학습(파이프라인). income_q도 train에서만 분위 cutpoint 학습(코덱스 #1 누수제거).
- 주모델 선택: CV-AUROC→|slope-1|→간명성.
결과: models/*.joblib, results/internal_metrics.csv
"""
import os, warnings, numpy as np, pandas as pd, joblib
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import lightgbm as lgb

ROOT = r"C:/Users/rudtn/Desktop/claude code/지역사회건강"
PROC = os.path.join(ROOT, "data/processed"); MODELS = os.path.join(ROOT, "models"); RES = os.path.join(ROOT, "results")
os.makedirs(MODELS, exist_ok=True); SEED = 42; QC = []
def qc(s): QC.append(str(s)); print(s)
# 프로토콜 Table 1 전체 조화 예측변수(14개). 자가건강(srh=KCHS qoa_01z1↔NHANES HUQ010) 복원.
# PA·음주는 Table 1 "보수적 공통정의+민감도" 원칙대로 포함. race는 공정성 전용(제외).
NUM = ["age", "sleep_hrs", "bmi", "income_q", "srh"]
BIN = ["partnered", "living_alone", "current_drinker", "active", "htn", "dm"]
CAT = ["sex", "edu", "smoking"]
FEATURES = NUM + BIN + CAT

def make_pre(native_nan=False):
    """[버그수정#1-c] 이진변수에 most_frequent 대체를 걸면, NHANES pre에서 DMDHHSIZ 부재로 100% 결측인
    living_alone이 KCHS 최빈값 0으로 전량 채워져 7,888명 전원이 '비독거'로 **날조**된다.
    - native_nan=True (트리계열: LightGBM/RandomForest): 이진변수 결측을 그대로 두어 날조를 차단.
      (비교실험: 최빈값대체 pre 0.7492 vs 결측유지 pre 0.7526 — 성능도 결측유지가 우위)
    - native_nan=False (elastic net): 선형모델은 NaN을 못 받으므로 대체하되, **결측지표(add_indicator)**를
      함께 넣어 '진짜 0'과 '모름'을 모델이 구분하게 한다(무표시 상수 날조 방지)."""
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    binp = "passthrough" if native_nan else SimpleImputer(strategy="most_frequent", add_indicator=True)
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", drop="first"))])
    return ColumnTransformer([("num", num, NUM), ("bin", binp, BIN), ("cat", cat, CAT)])

def cal_si(y, p, w=None):
    p = np.clip(p, 1e-6, 1 - 1e-6); lp = np.log(p / (1 - p)).reshape(-1, 1)
    c = LogisticRegression(penalty=None, solver="lbfgs").fit(lp, y, sample_weight=w)
    return float(c.coef_[0, 0]), float(c.intercept_[0])
def metrics(y, p, w=None):
    sl, ic = cal_si(y, p, w)
    return dict(AUROC=roc_auc_score(y, p, sample_weight=w), AUPRC=average_precision_score(y, p, sample_weight=w),
                Brier=brier_score_loss(y, p, sample_weight=w), cal_slope=sl, cal_intercept=ic,
                obs=float(np.average(y, weights=w)), pred=float(np.average(p, weights=w)))

dev = pd.read_parquet(os.path.join(PROC, "kchs_dev.parquet"))
y, W = dev["dep"].astype(int).values, dev["W"].values
# 코덱스 #1: income_q 누수 제거 — 분할 후 train에서만 연도별 분위 cutpoint 학습→전체 적용
idx = np.arange(len(dev)); itr, iho = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
def _income_q_trainonly(df, is_tr):
    iq = np.full(len(df), np.nan); yr = df["year"].values; inc = df["income_raw"].to_numpy(float)
    for yv in np.unique(yr):
        m = yr == yv; v = inc[m & is_tr]; v = v[~np.isnan(v)]
        if len(v) < 50: continue
        e = np.unique(np.quantile(v, [0, .2, .4, .6, .8, 1.0]))
        if len(e) < 3: continue
        ii = np.where(m)[0]; b = np.clip(np.digitize(inc[ii], e[1:-1]), 0, len(e) - 2).astype(float)
        b[np.isnan(inc[ii])] = np.nan; iq[ii] = b
    return iq
is_tr = np.zeros(len(dev), bool); is_tr[itr] = True
dev = dev.copy(); dev["income_q"] = _income_q_trainonly(dev, is_tr)
X = dev[FEATURES]
Xtr, Xho, ytr, yho, Wtr, Who = X.iloc[itr], X.iloc[iho], y[itr], y[iho], W[itr], W[iho]
qc(f"개발셋 2017-2025 n={len(X)} 우울(비가중)={y.mean():.3%} | train={len(Xtr)} holdout={len(Xho)} | income_q=train-only(#1)")

models = {
    "elasticnet": Pipeline([("pre", make_pre()), ("clf", LogisticRegression(penalty="elasticnet", l1_ratio=0.5, C=1.0, solver="saga", max_iter=300, n_jobs=-1))]),
    "randomforest": Pipeline([("pre", make_pre(native_nan=True)), ("clf", RandomForestClassifier(n_estimators=300, max_depth=14, min_samples_leaf=100, n_jobs=-1, random_state=SEED))]),
    "lightgbm": Pipeline([("pre", make_pre(native_nan=True)), ("clf", lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1))]),
}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED); rows = []
for name, pipe in models.items():
    idx = np.random.RandomState(SEED).permutation(len(Xtr))[:200000]
    cvauc = cross_val_score(pipe, Xtr.iloc[idx], ytr[idx], cv=cv, scoring="roc_auc", n_jobs=-1).mean()
    # #3 수정: 선택용 보정slope를 CV OOF에서 산출(holdout 미사용)
    oof = cross_val_predict(pipe, Xtr.iloc[idx], ytr[idx], cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    cvsl, _ = cal_si(ytr[idx], oof)
    pipe.fit(Xtr, ytr)
    p = pipe.predict_proba(Xho)[:, 1]
    mw = metrics(yho, p, Who); mu = metrics(yho, p)          # 가중/비가중(보고용, 선택엔 미사용)
    joblib.dump(pipe, os.path.join(MODELS, f"{name}.joblib"))
    rows.append({"model": name, "cv_auroc": cvauc, "cv_slope": cvsl, **{f"w_{k}": v for k, v in mw.items()}, "u_AUROC": mu["AUROC"]})
    qc(f"[{name}] CV-AUROC={cvauc:.4f} CV-slope={cvsl:.3f} | (보고)가중 AUROC={mw['AUROC']:.4f} slope={mw['cal_slope']:.3f} Brier={mw['Brier']:.4f}")

res = pd.DataFrame(rows); res.to_csv(os.path.join(RES, "internal_metrics.csv"), index=False)
res["slope_gap"] = (res["cv_slope"] - 1).abs()   # #3: CV 기반 tiebreak
top = res[res["cv_auroc"] >= res["cv_auroc"].max() - 0.005].copy()
top["par"] = top["model"].map({"elasticnet": 0, "randomforest": 1, "lightgbm": 2})
primary = top.sort_values(["slope_gap", "par"]).iloc[0]["model"]
qc(f"\n★ 주 모델 = {primary}")
joblib.dump({"primary": primary, "features": FEATURES}, os.path.join(MODELS, "primary.joblib"))
open(os.path.join(RES, "qc_train.txt"), "w", encoding="utf-8").write("\n".join(QC))
