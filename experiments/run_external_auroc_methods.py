from __future__ import annotations

import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "experiments"
OUT.mkdir(exist_ok=True)

TARGET = "dep"
WEIGHT = "W"

NUM = ["age", "sleep_hrs", "bmi", "income_q", "srh"]
BIN = ["partnered", "living_alone", "current_drinker", "active", "htn", "dm"]
CAT = ["sex", "edu", "smoking"]
FULL14 = NUM + BIN + CAT

STRICT_TRANSPORT = [
    "age",
    "sex",
    "edu",
    "income_q",
    "srh",
    "partnered",
    "sleep_hrs",
    "bmi",
    "htn",
    "dm",
    "smoking",
]
HEALTH_CORE = ["age", "sex", "srh", "sleep_hrs", "bmi", "htn", "dm"]
LOW_BURDEN_4 = ["age", "sex", "srh", "sleep_hrs"]
NO_CONTEXT = ["age", "sex", "edu", "income_q", "srh", "sleep_hrs", "bmi", "htn", "dm", "smoking"]

LGBM_BASE = dict(
    n_estimators=600,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)
LGBM_ROBUST = dict(
    n_estimators=900,
    learning_rate=0.02,
    num_leaves=15,
    min_child_samples=500,
    reg_lambda=5.0,
    subsample=0.9,
    colsample_bytree=0.95,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)
LGBM_FAST_ERA = dict(
    n_estimators=450,
    learning_rate=0.035,
    num_leaves=31,
    min_child_samples=200,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=2.0,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)


@dataclass
class MethodResult:
    method: str
    family: str
    compliance: str
    nhanes_touch: str
    train_n: int
    internal_auroc: float
    nhanes_pre_auroc: float
    nhanes_post_auroc: float
    seconds: float
    notes: str


def log(msg: str) -> None:
    print(msg, flush=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    kchs = pd.read_parquet(PROC / "kchs_dev.parquet")
    pre = pd.read_parquet(PROC / "nhanes_pre.parquet")
    post = pd.read_parquet(PROC / "nhanes_post.parquet")
    return kchs, pre, post


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    age = out["age"]
    sleep = out["sleep_hrs"]
    bmi = out["bmi"]
    srh = out["srh"]
    income = out["income_q"]

    out["age_sq"] = age**2
    out["age_ge60"] = (age >= 60).astype(float)
    out["age_band"] = pd.cut(
        age,
        bins=[19, 29, 39, 49, 59, 69, 79, 120],
        labels=["20s", "30s", "40s", "50s", "60s", "70s", "80p"],
        include_lowest=True,
    ).astype("object")
    out["sleep_dev7"] = (sleep - 7.0).abs()
    out["short_sleep"] = (sleep < 6.0).astype(float)
    out["long_sleep"] = (sleep > 9.0).astype(float)
    out["sleep_cat"] = pd.cut(
        sleep,
        bins=[-np.inf, 5.999, 8.0, np.inf],
        labels=["short", "normal", "long"],
    ).astype("object")
    out["bmi_sq"] = bmi**2
    out["bmi_asian_cat"] = pd.cut(
        bmi,
        bins=[-np.inf, 18.5, 23, 25, 30, np.inf],
        labels=["under", "normal", "over", "obese1", "obese2"],
    ).astype("object")

    out["srh_sleep_dev"] = srh * out["sleep_dev7"]
    out["srh_age_ge60"] = srh * out["age_ge60"]
    out["srh_low_income"] = srh * (income <= 1).astype(float)
    out["female_sleep_dev"] = (out["sex"] == "F").astype(float) * out["sleep_dev7"]
    out["unpartnered_alone"] = (1 - out["partnered"].fillna(0)) * out["living_alone"].fillna(0)
    return out


def feature_types(features: list[str]) -> tuple[list[str], list[str], list[str]]:
    cat_extra = {"age_band", "sleep_cat", "bmi_asian_cat"}
    cat = [c for c in features if c in CAT or c in cat_extra]
    binary_like = {
        "partnered",
        "living_alone",
        "current_drinker",
        "active",
        "htn",
        "dm",
        "age_ge60",
        "short_sleep",
        "long_sleep",
        "unpartnered_alone",
    }
    bin_cols = [c for c in features if c in binary_like and c not in cat]
    num = [c for c in features if c not in set(cat) | set(bin_cols)]
    return num, bin_cols, cat


def make_preprocessor(features: list[str], scale_numeric: bool = False) -> ColumnTransformer:
    num, bin_cols, cat = feature_types(features)
    if scale_numeric:
        num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    else:
        num_pipe = Pipeline([("imp", SimpleImputer(strategy="median"))])
    bin_pipe = SimpleImputer(strategy="most_frequent")
    cat_pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]
    )
    blocks = []
    if num:
        blocks.append(("num", num_pipe, num))
    if bin_cols:
        blocks.append(("bin", bin_pipe, bin_cols))
    if cat:
        blocks.append(("cat", cat_pipe, cat))
    return ColumnTransformer(blocks)


def make_lgbm_pipeline(features: list[str], params: dict | None = None) -> Pipeline:
    return Pipeline(
        [
            ("pre", make_preprocessor(features, scale_numeric=False)),
            ("clf", lgb.LGBMClassifier(**(params or LGBM_BASE))),
        ]
    )


def make_elasticnet_pipeline(features: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("pre", make_preprocessor(features, scale_numeric=True)),
            (
                "clf",
                LogisticRegression(
                    penalty="elasticnet",
                    l1_ratio=0.25,
                    C=0.4,
                    solver="saga",
                    max_iter=600,
                    n_jobs=-1,
                    random_state=SEED,
                ),
            ),
        ]
    )


def weighted_auc(y: pd.Series | np.ndarray, p: np.ndarray, w: pd.Series | np.ndarray) -> float:
    return float(roc_auc_score(np.asarray(y, dtype=int), p, sample_weight=np.asarray(w, dtype=float)))


def eval_predictions(
    method: str,
    family: str,
    compliance: str,
    nhanes_touch: str,
    p_internal: np.ndarray,
    p_pre: np.ndarray,
    p_post: np.ndarray,
    yho: np.ndarray,
    who: np.ndarray,
    pre: pd.DataFrame,
    post: pd.DataFrame,
    seconds: float,
    train_n: int,
    notes: str,
) -> MethodResult:
    return MethodResult(
        method=method,
        family=family,
        compliance=compliance,
        nhanes_touch=nhanes_touch,
        train_n=int(train_n),
        internal_auroc=weighted_auc(yho, p_internal, who),
        nhanes_pre_auroc=weighted_auc(pre[TARGET], p_pre, pre[WEIGHT]),
        nhanes_post_auroc=weighted_auc(post[TARGET], p_post, post[WEIGHT]),
        seconds=float(seconds),
        notes=notes,
    )


def fit_eval_pipeline(
    method: str,
    family: str,
    compliance: str,
    nhanes_touch: str,
    pipe: Pipeline,
    features: list[str],
    Xtr: pd.DataFrame,
    ytr: np.ndarray,
    Xho: pd.DataFrame,
    yho: np.ndarray,
    who: np.ndarray,
    pre: pd.DataFrame,
    post: pd.DataFrame,
    sample_weight: np.ndarray | None = None,
    notes: str = "",
) -> tuple[MethodResult, dict[str, np.ndarray], Pipeline]:
    t0 = time.time()
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["clf__sample_weight"] = sample_weight
    pipe.fit(Xtr[features], ytr, **fit_kwargs)
    seconds = time.time() - t0
    preds = {
        "internal": pipe.predict_proba(Xho[features])[:, 1],
        "pre": pipe.predict_proba(pre[features])[:, 1],
        "post": pipe.predict_proba(post[features])[:, 1],
    }
    result = eval_predictions(
        method,
        family,
        compliance,
        nhanes_touch,
        preds["internal"],
        preds["pre"],
        preds["post"],
        yho,
        who,
        pre,
        post,
        seconds,
        len(Xtr),
        notes,
    )
    return result, preds, pipe


def normalized_training_weight(xtr: pd.DataFrame, ytr: np.ndarray, base_w: np.ndarray | None, mode: str) -> np.ndarray:
    if base_w is None:
        w = np.ones(len(xtr), dtype=float)
    else:
        w = np.asarray(base_w, dtype=float).copy()

    if "survey" in mode:
        w = w / np.nanmean(w)
    if "year" in mode:
        year = xtr["year"].to_numpy()
        for yr in np.unique(year):
            ix = year == yr
            w[ix] *= len(xtr) / (len(np.unique(year)) * ix.sum())
    if "class" in mode:
        y = np.asarray(ytr, dtype=int)
        for cls in [0, 1]:
            ix = y == cls
            w[ix] *= len(y) / (2 * ix.sum())
    return w / np.nanmean(w)


def encode_mono(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    edu_map = {"elem_low": 0, "middle": 1, "high": 2, "college": 3}
    smoking_map = {"never": 0, "former": 1, "current": 2}
    for c in features:
        if c == "sex":
            out["sex_M"] = (df[c] == "M").astype(float)
        elif c == "edu":
            out["edu_ord"] = df[c].map(edu_map).astype(float)
        elif c == "smoking":
            out["smoking_ord"] = df[c].map(smoking_map).astype(float)
        else:
            out[c] = pd.to_numeric(df[c], errors="coerce")
    return out


def fit_eval_monotonic(
    Xtr: pd.DataFrame,
    ytr: np.ndarray,
    Xho: pd.DataFrame,
    yho: np.ndarray,
    who: np.ndarray,
    pre: pd.DataFrame,
    post: pd.DataFrame,
) -> tuple[MethodResult, dict[str, np.ndarray]]:
    features = FULL14 + ["sleep_dev7", "short_sleep", "long_sleep"]
    etr = encode_mono(add_engineered(Xtr), features)
    eho = encode_mono(add_engineered(Xho), features)
    epre = encode_mono(add_engineered(pre), features)
    epost = encode_mono(add_engineered(post), features)

    imp = SimpleImputer(strategy="median")
    x_fit = imp.fit_transform(etr)
    x_ho = imp.transform(eho)
    x_pre = imp.transform(epre)
    x_post = imp.transform(epost)

    constraints = []
    for c in etr.columns:
        if c in {"srh", "sleep_dev7", "short_sleep", "long_sleep", "living_alone", "htn", "dm", "smoking_ord"}:
            constraints.append(1)
        elif c in {"income_q", "partnered", "active", "edu_ord"}:
            constraints.append(-1)
        else:
            constraints.append(0)

    clf = lgb.LGBMClassifier(**LGBM_ROBUST, monotone_constraints=constraints)
    t0 = time.time()
    clf.fit(x_fit, ytr)
    seconds = time.time() - t0
    preds = {
        "internal": clf.predict_proba(x_ho)[:, 1],
        "pre": clf.predict_proba(x_pre)[:, 1],
        "post": clf.predict_proba(x_post)[:, 1],
    }
    result = eval_predictions(
        "lgbm_monotonic_clinical_constraints",
        "monotonic_constrained_lgbm",
        "compliant §2.10",
        "none",
        preds["internal"],
        preds["pre"],
        preds["post"],
        yho,
        who,
        pre,
        post,
        seconds,
        len(Xtr),
        "Ordinal/raw encoding with clinically plausible monotone signs; no NHANES features/outcomes used for fitting.",
    )
    return result, preds


def fit_eval_era_ensemble(
    Xtr: pd.DataFrame,
    ytr: np.ndarray,
    Xho: pd.DataFrame,
    yho: np.ndarray,
    who: np.ndarray,
    pre: pd.DataFrame,
    post: pd.DataFrame,
) -> tuple[MethodResult, dict[str, np.ndarray]]:
    eras = {
        "2017_2019": [2017, 2018, 2019],
        "2020_2022": [2020, 2021, 2022],
        "2023_2025": [2023, 2024, 2025],
    }
    t0 = time.time()
    preds_int = []
    preds_pre = []
    preds_post = []
    for era_name, years in eras.items():
        ix = Xtr["year"].isin(years).to_numpy()
        pipe = make_lgbm_pipeline(FULL14, LGBM_FAST_ERA)
        pipe.fit(Xtr.loc[ix, FULL14], ytr[ix])
        preds_int.append(pipe.predict_proba(Xho[FULL14])[:, 1])
        preds_pre.append(pipe.predict_proba(pre[FULL14])[:, 1])
        preds_post.append(pipe.predict_proba(post[FULL14])[:, 1])
        log(f"  era model {era_name}: n={int(ix.sum())}")
    seconds = time.time() - t0
    preds = {
        "internal": np.mean(preds_int, axis=0),
        "pre": np.mean(preds_pre, axis=0),
        "post": np.mean(preds_post, axis=0),
    }
    result = eval_predictions(
        "lgbm_kchs_calendar_era_mean_ensemble",
        "kchs_transport_ensemble",
        "compliant §2.10",
        "none",
        preds["internal"],
        preds["pre"],
        preds["post"],
        yho,
        who,
        pre,
        post,
        seconds,
        len(Xtr),
        "Three KCHS calendar-era models averaged to reduce dependence on a single development-period distribution.",
    )
    return result, preds


def fit_eval_kchs_stack(
    Xtr: pd.DataFrame,
    ytr: np.ndarray,
    Xho: pd.DataFrame,
    yho: np.ndarray,
    who: np.ndarray,
    pre: pd.DataFrame,
    post: pd.DataFrame,
) -> tuple[MethodResult, dict[str, np.ndarray]]:
    x_fit, x_val, y_fit, y_val = train_test_split(
        Xtr, ytr, test_size=0.2, stratify=ytr, random_state=SEED + 10
    )
    base_specs = [
        ("full14", FULL14, LGBM_FAST_ERA),
        ("strict", STRICT_TRANSPORT, LGBM_FAST_ERA),
        ("engineered", FULL14 + engineered_features_only(), LGBM_FAST_ERA),
    ]
    val_cols = []
    ho_cols = []
    pre_cols = []
    post_cols = []
    t0 = time.time()
    for name, features, params in base_specs:
        train_df = add_engineered(x_fit) if any(f in engineered_features_only() for f in features) else x_fit
        val_df = add_engineered(x_val) if any(f in engineered_features_only() for f in features) else x_val
        ho_df = add_engineered(Xho) if any(f in engineered_features_only() for f in features) else Xho
        pre_df = add_engineered(pre) if any(f in engineered_features_only() for f in features) else pre
        post_df = add_engineered(post) if any(f in engineered_features_only() for f in features) else post
        pipe = make_lgbm_pipeline(features, params)
        pipe.fit(train_df[features], y_fit)
        val_cols.append(pipe.predict_proba(val_df[features])[:, 1])
        ho_cols.append(pipe.predict_proba(ho_df[features])[:, 1])
        pre_cols.append(pipe.predict_proba(pre_df[features])[:, 1])
        post_cols.append(pipe.predict_proba(post_df[features])[:, 1])
        log(f"  stack base {name}: fit n={len(x_fit)}")
    meta = LogisticRegression(penalty=None, solver="lbfgs")
    meta.fit(np.column_stack(val_cols), y_val)
    seconds = time.time() - t0
    preds = {
        "internal": meta.predict_proba(np.column_stack(ho_cols))[:, 1],
        "pre": meta.predict_proba(np.column_stack(pre_cols))[:, 1],
        "post": meta.predict_proba(np.column_stack(post_cols))[:, 1],
    }
    result = eval_predictions(
        "stack_lgbm_full_strict_engineered_kchs_only",
        "stacking_ensemble",
        "compliant §2.10",
        "none",
        preds["internal"],
        preds["pre"],
        preds["post"],
        yho,
        who,
        pre,
        post,
        seconds,
        len(x_fit),
        "Meta-learner trained only on a KCHS development sub-split; no NHANES feature/outcome use.",
    )
    return result, preds


def engineered_features_only() -> list[str]:
    return [
        "age_sq",
        "age_ge60",
        "age_band",
        "sleep_dev7",
        "short_sleep",
        "long_sleep",
        "sleep_cat",
        "bmi_sq",
        "bmi_asian_cat",
        "srh_sleep_dev",
        "srh_age_ge60",
        "srh_low_income",
        "female_sleep_dev",
        "unpartnered_alone",
    ]


def distribution_shift_scores(source: pd.DataFrame, target: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for f in features:
        a = source[f]
        b = target[f]
        if f in CAT:
            pa = a.value_counts(normalize=True, dropna=False)
            pb = b.value_counts(normalize=True, dropna=False)
            idx = pa.index.union(pb.index)
            score = 0.5 * float(np.abs(pa.reindex(idx, fill_value=0) - pb.reindex(idx, fill_value=0)).sum())
            metric = "categorical_total_variation"
        else:
            ax = pd.to_numeric(a, errors="coerce").dropna()
            bx = pd.to_numeric(b, errors="coerce").dropna()
            if len(ax) and len(bx):
                score = float(ks_2samp(ax, bx).statistic)
            else:
                score = 1.0
            metric = "ks_statistic"
        rows.append({"feature": f, "shift_score": score, "metric": metric})
    return pd.DataFrame(rows).sort_values("shift_score", ascending=False)


def make_domain_ratio_weights(
    Xtr: pd.DataFrame,
    target_df: pd.DataFrame,
    features: list[str],
    sample_n_source: int = 120_000,
    clip: tuple[float, float] = (0.15, 8.0),
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(SEED)
    source_ix = rng.choice(np.arange(len(Xtr)), size=min(sample_n_source, len(Xtr)), replace=False)
    source_sample = Xtr.iloc[source_ix][features].copy()
    target_sample = target_df[features].copy()
    domain_x = pd.concat([source_sample, target_sample], axis=0, ignore_index=True)
    domain_y = np.r_[np.zeros(len(source_sample)), np.ones(len(target_sample))]

    pipe = Pipeline(
        [
            ("pre", make_preprocessor(features, scale_numeric=True)),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=180,
                    learning_rate=0.05,
                    max_leaf_nodes=31,
                    l2_regularization=1.0,
                    random_state=SEED,
                ),
            ),
        ]
    )
    pipe.fit(domain_x, domain_y)
    domain_auc = roc_auc_score(domain_y, pipe.predict_proba(domain_x)[:, 1])

    pt = pipe.predict_proba(Xtr[features])[:, 1]
    pt = np.clip(pt, 1e-4, 1 - 1e-4)
    prior_odds = len(source_sample) / len(target_sample)
    ratio = (pt / (1 - pt)) * prior_odds
    ratio = np.clip(ratio, clip[0], clip[1])
    ratio = ratio / np.nanmean(ratio)
    return ratio.astype(float), float(domain_auc)


def run_gray_covariate_shift(
    label: str,
    target_df: pd.DataFrame,
    Xtr: pd.DataFrame,
    ytr: np.ndarray,
    Xho: pd.DataFrame,
    yho: np.ndarray,
    who: np.ndarray,
    pre: pd.DataFrame,
    post: pd.DataFrame,
) -> tuple[MethodResult, dict[str, np.ndarray]]:
    log(f"Computing gray-area covariate ratio weights for {label}")
    ratio, domain_auc = make_domain_ratio_weights(Xtr, target_df, FULL14)
    pipe = make_lgbm_pipeline(FULL14, LGBM_ROBUST)
    result, preds, _ = fit_eval_pipeline(
        f"gray_covariate_shift_weighted_to_{label}",
        "unlabeled_domain_adaptation",
        "gray-area: uses NHANES predictors only",
        f"NHANES {label} unlabeled predictors used for density-ratio weights; no NHANES outcomes in fitting",
        pipe,
        FULL14,
        Xtr,
        ytr,
        Xho,
        yho,
        who,
        pre,
        post,
        sample_weight=ratio,
        notes=f"Density-ratio weighting from domain classifier AUC={domain_auc:.3f}; weights clipped.",
    )
    return result, preds


def run() -> None:
    t_all = time.time()
    kchs, pre, post = load_data()
    log(f"Loaded kchs={kchs.shape}, pre={pre.shape}, post={post.shape}")

    y = kchs[TARGET].astype(int).to_numpy()
    X_train, X_hold, y_train, y_hold, w_train, w_hold = train_test_split(
        kchs,
        y,
        kchs[WEIGHT].to_numpy(float),
        test_size=0.2,
        stratify=y,
        random_state=SEED,
    )
    y_hold = y_hold.astype(int)
    w_hold = w_hold.astype(float)
    results: list[MethodResult] = []
    pred_bank: dict[str, dict[str, np.ndarray]] = {}

    def add_result(result: MethodResult, preds: dict[str, np.ndarray]) -> None:
        results.append(result)
        pred_bank[result.method] = preds
        log(
            f"{result.method}: internal={result.internal_auroc:.4f} "
            f"pre={result.nhanes_pre_auroc:.4f} post={result.nhanes_post_auroc:.4f} "
            f"[{result.compliance}]"
        )
        pd.DataFrame([r.__dict__ for r in results]).to_csv(OUT / "method_results_partial.csv", index=False)

    compliant_specs = [
        (
            "lgbm_full14_baseline",
            "baseline_lgbm",
            FULL14,
            LGBM_BASE,
            None,
            "Reference LightGBM using the current 14 harmonized predictors.",
        ),
        (
            "lgbm_full14_robust_low_complexity",
            "robust_lgbm",
            FULL14,
            LGBM_ROBUST,
            None,
            "Lower-complexity, high-min-child LightGBM selected without NHANES information.",
        ),
        (
            "lgbm_full14_survey_year_balanced_training",
            "transport_weighted_lgbm",
            FULL14,
            LGBM_ROBUST,
            "survey+year",
            "KCHS survey weights normalized and KCHS years balanced equally in training.",
        ),
        (
            "lgbm_full14_class_balanced_training",
            "imbalance_robust_lgbm",
            FULL14,
            LGBM_ROBUST,
            "class",
            "KCHS class-balanced training weights; evaluation remains weighted AUROC.",
        ),
        (
            "lgbm_strict_transport_no_living_active_drink",
            "a_priori_transport_subset",
            STRICT_TRANSPORT,
            LGBM_ROBUST,
            None,
            "A priori subset excluding living_alone, active, current_drinker due to known instrument/availability concerns.",
        ),
        (
            "lgbm_no_social_behavior_context",
            "a_priori_transport_subset",
            NO_CONTEXT,
            LGBM_ROBUST,
            None,
            "Drops social/behavior context variables with likely cultural or measurement drift.",
        ),
        (
            "lgbm_health_core_7",
            "parsimonious_subset",
            HEALTH_CORE,
            LGBM_ROBUST,
            None,
            "Clinical/self-report health core: age, sex, srh, sleep, BMI, HTN, DM.",
        ),
        (
            "lgbm_low_burden_4",
            "parsimonious_subset",
            LOW_BURDEN_4,
            LGBM_ROBUST,
            None,
            "Low-burden KCHS-only 4-predictor model.",
        ),
    ]

    for method, family, features, params, weight_mode, notes in compliant_specs:
        log(f"Running {method}")
        sw = None
        if weight_mode:
            base = w_train if "survey" in weight_mode else None
            sw = normalized_training_weight(X_train, y_train, base, weight_mode)
        result, preds, _ = fit_eval_pipeline(
            method,
            family,
            "compliant §2.10",
            "none",
            make_lgbm_pipeline(features, params),
            features,
            X_train,
            y_train,
            X_hold,
            y_hold,
            w_hold,
            pre,
            post,
            sample_weight=sw,
            notes=notes,
        )
        add_result(result, preds)

    engineered_features = FULL14 + engineered_features_only()
    X_train_eng = add_engineered(X_train)
    X_hold_eng = add_engineered(X_hold)
    pre_eng = add_engineered(pre)
    post_eng = add_engineered(post)

    log("Running lgbm_engineered_interactions")
    result, preds, _ = fit_eval_pipeline(
        "lgbm_engineered_interactions",
        "feature_engineering_lgbm",
        "compliant §2.10",
        "none",
        make_lgbm_pipeline(engineered_features, LGBM_ROBUST),
        engineered_features,
        X_train_eng,
        y_train,
        X_hold_eng,
        y_hold,
        w_hold,
        pre_eng,
        post_eng,
        notes="Age/sleep/BMI nonlinear terms, coarse bins, and selected interactions generated identically from harmonized predictors.",
    )
    add_result(result, preds)

    log("Running elasticnet_engineered_interactions")
    result, preds, _ = fit_eval_pipeline(
        "elasticnet_engineered_interactions",
        "feature_engineering_elasticnet",
        "compliant §2.10",
        "none",
        make_elasticnet_pipeline(engineered_features),
        engineered_features,
        X_train_eng,
        y_train,
        X_hold_eng,
        y_hold,
        w_hold,
        pre_eng,
        post_eng,
        notes="Elastic-net logistic model with nonlinear terms and interactions; fitted only in KCHS.",
    )
    add_result(result, preds)

    log("Running monotonic constraints")
    result, preds = fit_eval_monotonic(X_train, y_train, X_hold, y_hold, w_hold, pre, post)
    add_result(result, preds)

    log("Running KCHS calendar-era ensemble")
    result, preds = fit_eval_era_ensemble(X_train, y_train, X_hold, y_hold, w_hold, pre, post)
    add_result(result, preds)

    log("Running KCHS-only stacking ensemble")
    result, preds = fit_eval_kchs_stack(X_train, y_train, X_hold, y_hold, w_hold, pre, post)
    add_result(result, preds)

    equal_members = [
        "lgbm_full14_baseline",
        "lgbm_full14_robust_low_complexity",
        "lgbm_strict_transport_no_living_active_drink",
        "lgbm_engineered_interactions",
        "lgbm_monotonic_clinical_constraints",
        "lgbm_kchs_calendar_era_mean_ensemble",
    ]
    log("Evaluating equal-weight compliant ensemble")
    ensemble_preds = {
        key: np.mean([pred_bank[m][key] for m in equal_members if m in pred_bank], axis=0)
        for key in ["internal", "pre", "post"]
    }
    result = eval_predictions(
        "ensemble_equal_weight_diverse_compliant",
        "prediction_ensemble",
        "compliant §2.10",
        "none",
        ensemble_preds["internal"],
        ensemble_preds["pre"],
        ensemble_preds["post"],
        y_hold,
        w_hold,
        pre,
        post,
        0.0,
        len(X_train),
        "Equal-weight average of six diverse KCHS-only fitted models; no NHANES data used to choose weights.",
    )
    add_result(result, ensemble_preds)

    target_combined = pd.concat([pre, post], axis=0, ignore_index=True)
    shift = distribution_shift_scores(X_train, target_combined, FULL14)
    shift.to_csv(OUT / "gray_unlabeled_shift_scores.csv", index=False)
    drop_top = shift.head(4)["feature"].tolist()
    stable_features = [f for f in FULL14 if f not in drop_top]
    log(f"Gray-area shift subset drops {drop_top}")
    result, preds, _ = fit_eval_pipeline(
        "gray_unlabeled_shift_stable_subset_drop4",
        "unlabeled_domain_adaptation",
        "gray-area: uses NHANES predictors only",
        "NHANES pre/post unlabeled predictors used to rank distribution shift; no NHANES outcomes in fitting",
        make_lgbm_pipeline(stable_features, LGBM_ROBUST),
        stable_features,
        X_train,
        y_train,
        X_hold,
        y_hold,
        w_hold,
        pre,
        post,
        notes=f"Dropped top four source-target shifted predictors by KS/TV score: {', '.join(drop_top)}.",
    )
    add_result(result, preds)

    for label, target in [("pre", pre), ("post", post), ("combined", target_combined)]:
        result, preds = run_gray_covariate_shift(
            label, target, X_train, y_train, X_hold, y_hold, w_hold, pre, post
        )
        add_result(result, preds)

    final = pd.DataFrame([r.__dict__ for r in results]).sort_values(
        ["compliance", "nhanes_pre_auroc", "nhanes_post_auroc"], ascending=[True, False, False]
    )
    final.to_csv(OUT / "method_results.csv", index=False)

    summary = {
        "seed": SEED,
        "train_n": int(len(X_train)),
        "holdout_n": int(len(X_hold)),
        "nhanes_pre_n": int(len(pre)),
        "nhanes_post_n": int(len(post)),
        "elapsed_seconds": round(time.time() - t_all, 3),
        "note": "All writes are confined to experiments/. Gray-area methods use NHANES predictors but no NHANES outcomes for fitting.",
    }
    (OUT / "run_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log("Finished. Results written to experiments/method_results.csv")


if __name__ == "__main__":
    run()
