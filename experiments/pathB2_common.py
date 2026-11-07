# -*- coding: utf-8 -*-
"""
pathB2_common.py — 경로B(알고리즘·하이퍼파라미터 대탐색) 공용 모듈 v2.

§2.10 준수 규칙(코드 수준으로 강제):
  * 모델/하이퍼파라미터 선택은 오직 KCHS 내부 CV(가중 AUROC)로만 수행한다.
  * NHANES(pre/post)는 사전에 확정된 후보에 대해 최종 1회 평가만 한다.
  * NHANES 라벨은 학습/선택 어디에도 들어가지 않는다.
"""
import os, sys, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score

ROOT = r"C:/Users/rudtn/Desktop/claude code/지역사회건강"
PROC = os.path.join(ROOT, "data/processed")
RES = os.path.join(ROOT, "results")
SEED = 42

NUM = ["age", "sleep_hrs", "bmi", "income_q", "srh", "oral_health"]
BIN = ["partnered", "living_alone", "current_drinker", "active", "htn", "dm",
       "employed", "food_insecure"]
CAT = ["sex", "edu", "smoking"]
FEATURES = NUM + BIN + CAT          # 17개 확정 예측변수


# ----------------------------------------------------------------------------- data
def load():
    k = pd.read_parquet(os.path.join(PROC, "ext_kchs.parquet"))
    pre = pd.read_parquet(os.path.join(PROC, "ext_pre.parquet"))
    post = pd.read_parquet(os.path.join(PROC, "ext_post.parquet"))
    return k, pre, post


def income_q_trainonly(df, is_tr):
    """연도별 소득 5분위를 train 행만으로 산출(기준선 재현에 필요)."""
    iq = np.full(len(df), np.nan)
    yr = df["year"].values
    inc = df["income_raw"].to_numpy(float)
    for yv in np.unique(yr):
        m = yr == yv
        v = inc[m & is_tr]; v = v[~np.isnan(v)]
        if len(v) < 50:
            continue
        e = np.unique(np.quantile(v, [0, .2, .4, .6, .8, 1.0]))
        if len(e) < 3:
            continue
        ii = np.where(m)[0]
        b = np.clip(np.digitize(inc[ii], e[1:-1]), 0, len(e) - 2).astype(float)
        b[np.isnan(inc[ii])] = np.nan
        iq[ii] = b
    return iq


def baseline_split(k):
    """기준선 재현 프로토콜(V4): 80/20 stratified split + income_q train-only."""
    y = k["dep"].astype(int).values
    itr, iho = train_test_split(np.arange(len(k)), test_size=0.2, stratify=y,
                                random_state=SEED)
    is_tr = np.zeros(len(k), bool); is_tr[itr] = True
    kk = k.copy()
    kk["income_q"] = income_q_trainonly(kk, is_tr)
    return kk, itr, iho


# ----------------------------------------------------------------------------- preprocessing
def make_pre(scale=True, ohe=True):
    steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("sc", StandardScaler()))
    num = Pipeline(steps)
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore", drop="first"))])
    return ColumnTransformer([("num", num, NUM), ("bin", "passthrough", BIN),
                              ("cat", cat, CAT)])


def feat_names():
    """make_pre 출력 컬럼 순서(OneHot drop=first, KCHS 전체 레벨 기준)."""
    names = list(NUM) + list(BIN)
    lv = {"sex": ["F", "M"], "edu": ["college", "elem_low", "high", "middle"],
          "smoking": ["current", "former", "never"]}
    for c in CAT:
        names += [f"{c}={L}" for L in lv[c][1:]]
    return names


# 방향이 임상적으로 명확한 변수의 단조 제약(우울 위험 ↑ 방향 = +1)
#  srh 1좋음~5나쁨  -> 나쁠수록 위험 ↑  (+1)
#  oral_health 1~5  -> 나쁠수록 위험 ↑  (+1)
#  income_q 낮을수록 위험 ↑             (-1)
#  age  KCHS에서 비단조(U자)이므로 제약 없음 -> 아래 MONO_SETS에서 변형으로 검정
def mono_vector(spec):
    """spec: {feature_name: +1/-1} -> make_pre 출력 순서에 맞춘 리스트."""
    return [int(spec.get(n, 0)) for n in feat_names()]


MONO_SETS = {
    "mono_none":       {},
    "mono_srh":        {"srh": 1},
    "mono_srh_oral":   {"srh": 1, "oral_health": 1},
    "mono_srh_oral_inc": {"srh": 1, "oral_health": 1, "income_q": -1},
    "mono_core":       {"srh": 1, "oral_health": 1, "income_q": -1,
                        "living_alone": 1, "food_insecure": 1, "employed": -1},
    "mono_core_age":   {"srh": 1, "oral_health": 1, "income_q": -1,
                        "living_alone": 1, "food_insecure": 1, "employed": -1,
                        "age": 1},
    "mono_all17":      {"srh": 1, "oral_health": 1, "income_q": -1,
                        "living_alone": 1, "food_insecure": 1, "employed": -1,
                        "partnered": -1, "active": -1, "current_drinker": -1,
                        "htn": 1, "dm": 1},
}


# ----------------------------------------------------------------------------- metric
def wauc(y, p, w):
    return roc_auc_score(y, p, sample_weight=w)


# ----------------------------------------------------------------------------- CV (KCHS only)
def cv_indices(y, n_splits=5, seed=SEED):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def cv_score(make_model, X, y, W, folds, verbose=False):
    """KCHS 내부 5-fold 가중 AUROC. 외부 데이터는 절대 접근하지 않는다."""
    sc = []
    for f, (a, b) in enumerate(folds):
        m = make_model()
        m.fit(X.iloc[a], y[a])
        p = m.predict_proba(X.iloc[b])[:, 1]
        s = wauc(y[b], p, W[b])
        sc.append(s)
        if verbose:
            print(f"    fold{f} {s:.5f}", flush=True)
    return float(np.mean(sc)), float(np.std(sc))


def ext_eval(model, pre, post):
    """최종 외부평가(1회). 재적합 없음 — 이미 KCHS로 fit된 모델을 그대로 적용."""
    out = {}
    for nm, d in [("ext_pre", pre), ("ext_post", post)]:
        p = model.predict_proba(d[FEATURES])[:, 1]
        out[nm] = wauc(d["dep"].astype(int).values, p, d["W"].to_numpy(float))
    return out


# ----------------------------------------------------------------------------- model factory
def build_model(family, p, nj=22):
    """screen 단계에서 기록한 (family, params)로부터 동일 모델을 재구성."""
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    import lightgbm as lgb
    if family.startswith("lgbm"):
        q = {kk: vv for kk, vv in p.items() if kk != "mono"}
        if "mono" in p:
            q["monotone_constraints"] = mono_vector(MONO_SETS[p["mono"]])
        q.update(random_state=SEED, n_jobs=nj, verbose=-1)
        return lambda: Pipeline([("pre", make_pre()), ("clf", lgb.LGBMClassifier(**q))])
    if family.startswith("xgb"):
        import xgboost as xgb
        q = dict(n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.8,
                 colsample_bytree=0.8, tree_method="hist", eval_metric="auc",
                 random_state=SEED, n_jobs=nj)
        q.update({kk: vv for kk, vv in p.items() if kk != "mono"})
        if "mono" in p:
            q["monotone_constraints"] = tuple(mono_vector(MONO_SETS[p["mono"]]))
        return lambda: Pipeline([("pre", make_pre()), ("clf", xgb.XGBClassifier(**q))])
    if family.startswith("cat"):
        from catboost import CatBoostClassifier
        q = dict(iterations=800, learning_rate=0.05, depth=6, l2_leaf_reg=3,
                 random_seed=SEED, verbose=0, thread_count=nj, allow_writing_files=False)
        q.update(p)
        return lambda: Pipeline([("pre", make_pre()), ("clf", CatBoostClassifier(**q))])
    if family.startswith("histgb"):
        q = dict(max_iter=600, learning_rate=0.03, max_leaf_nodes=31, random_state=SEED)
        q.update({kk: vv for kk, vv in p.items() if kk != "mono"})
        if "mono" in p:
            q["monotonic_cst"] = mono_vector(MONO_SETS[p["mono"]])
        return lambda: Pipeline([("pre", make_pre()),
                                 ("clf", HistGradientBoostingClassifier(**q))])
    if family == "linear":
        q = dict(max_iter=3000)
        q.update(p)
        if q.get("penalty") in ("l1", "elasticnet"):
            q["solver"] = "saga"; q["n_jobs"] = nj
        else:
            q["solver"] = "lbfgs"
        return lambda: Pipeline([("pre", make_pre()), ("clf", LogisticRegression(**q))])
    raise ValueError(family)
