from __future__ import annotations

import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "kchs_dev.parquet"

BASE_NUMERIC = ["age", "income_q", "partnered", "sleep_hrs", "bmi", "htn", "dm"]
BASE_CATEGORICAL = ["sex", "edu", "smoking"]

STEP2_NUMERIC = BASE_NUMERIC + ["living_alone"]
STEP3_NUMERIC = STEP2_NUMERIC + ["active", "current_drinker"]
STRICT_ENGINEERED_CATEGORICAL = ["age_band_10y", "sleep_cat"]

DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "n_estimators": 100,
    "learning_rate": 0.1,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": -1,
    "force_col_wise": True,
}


@dataclass(frozen=True)
class FeatureSpec:
    numeric: list[str]
    categorical: list[str]

    @property
    def columns(self) -> list[str]:
        return self.numeric + self.categorical


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def make_preprocessor(spec: FeatureSpec) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, spec.numeric),
            ("cat", categorical_pipe, spec.categorical),
        ]
    )


def make_lgbm_pipeline(spec: FeatureSpec, params: dict[str, Any]) -> Pipeline:
    model_params = DEFAULT_LGBM_PARAMS | {k: v for k, v in params.items() if k != "name"}
    return Pipeline(
        [
            ("preprocess", make_preprocessor(spec)),
            ("model", LGBMClassifier(**model_params)),
        ]
    )


def make_elasticnet_pipeline(spec: FeatureSpec, params: dict[str, Any]) -> Pipeline:
    model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=params["l1_ratio"],
        C=params["C"],
        max_iter=params.get("max_iter", 500),
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocess", make_preprocessor(spec)), ("model", model)])


def make_rf_pipeline(spec: FeatureSpec, params: dict[str, Any]) -> Pipeline:
    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params.get("max_features", "sqrt"),
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight=params.get("class_weight"),
    )
    return Pipeline([("preprocess", make_preprocessor(spec)), ("model", model)])


def add_protocol_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    age = pd.to_numeric(out["age"], errors="coerce")
    decade = np.floor(age / 10) * 10
    age_band = pd.Series(np.nan, index=out.index, dtype="object")
    valid_age = decade.notna()
    age_band.loc[valid_age] = decade.loc[valid_age].astype(int).astype(str) + "s"
    age_band.loc[age >= 90] = "90+"
    out["age_band_10y"] = age_band

    sleep = pd.to_numeric(out["sleep_hrs"], errors="coerce")
    sleep_cat = pd.Series(np.nan, index=out.index, dtype="object")
    sleep_cat.loc[sleep < 7] = "short_lt7"
    sleep_cat.loc[(sleep >= 7) & (sleep <= 9)] = "normal_7to9"
    sleep_cat.loc[sleep > 9] = "long_gt9"
    out["sleep_cat"] = sleep_cat

    return out


def weighted_auc(model: Pipeline, x_df: pd.DataFrame, y: pd.Series, w: pd.Series) -> float:
    pred = model.predict_proba(x_df)[:, 1]
    return float(roc_auc_score(y, pred, sample_weight=w))


def fit_score_lgbm(
    step: str,
    spec: FeatureSpec,
    params: dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fit_sample_weight: bool = False,
) -> dict[str, Any]:
    start = time.time()
    model = make_lgbm_pipeline(spec, params)
    fit_kwargs: dict[str, Any] = {}
    if fit_sample_weight:
        fit_kwargs["model__sample_weight"] = train_df["W"].to_numpy()
    model.fit(train_df[spec.columns], train_df["dep"], **fit_kwargs)
    auc = weighted_auc(model, test_df[spec.columns], test_df["dep"], test_df["W"])
    model_params = DEFAULT_LGBM_PARAMS | {k: v for k, v in params.items() if k != "name"}
    return {
        "step": step,
        "model": "lightgbm",
        "tuning_config": params.get("name", "default"),
        "fit_sample_weight": fit_sample_weight,
        "weighted_holdout_auroc": auc,
        "raw_feature_count": len(spec.columns),
        "numeric_features": "|".join(spec.numeric),
        "categorical_features": "|".join(spec.categorical),
        "model_params": json.dumps(model_params, sort_keys=True),
        "seconds": round(time.time() - start, 3),
    }


def lgbm_grid(pos_weight: float) -> list[dict[str, Any]]:
    return [
        {"name": "g00_default"},
        {
            "name": "g01_small_leaves",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 100,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
        {
            "name": "g02_regularized",
            "n_estimators": 600,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_child_samples": 200,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.0,
            "reg_lambda": 2.0,
        },
        {
            "name": "g03_medium_leaves",
            "n_estimators": 600,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "min_child_samples": 200,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.0,
            "reg_lambda": 3.0,
        },
        {
            "name": "g04_heavy_regularization",
            "n_estimators": 800,
            "learning_rate": 0.02,
            "num_leaves": 63,
            "min_child_samples": 500,
            "subsample": 0.85,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.1,
            "reg_lambda": 5.0,
        },
        {
            "name": "g05_large_leaves_regularized",
            "n_estimators": 500,
            "learning_rate": 0.04,
            "num_leaves": 127,
            "min_child_samples": 500,
            "subsample": 0.8,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.1,
            "reg_lambda": 5.0,
        },
        {
            "name": "g06_balanced_class_weight",
            "n_estimators": 600,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "min_child_samples": 200,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.0,
            "reg_lambda": 3.0,
            "class_weight": "balanced",
        },
        {
            "name": "g07_shallow_long",
            "n_estimators": 1000,
            "learning_rate": 0.015,
            "num_leaves": 15,
            "min_child_samples": 500,
            "subsample": 0.9,
            "colsample_bytree": 0.95,
            "reg_alpha": 0.0,
            "reg_lambda": 3.0,
        },
        {
            "name": "g08_deeper_long",
            "n_estimators": 1000,
            "learning_rate": 0.02,
            "num_leaves": 127,
            "min_child_samples": 200,
            "subsample": 0.8,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.1,
            "reg_lambda": 5.0,
        },
        {
            "name": "g09_scale_pos_weight",
            "n_estimators": 800,
            "learning_rate": 0.02,
            "num_leaves": 63,
            "min_child_samples": 200,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.0,
            "reg_lambda": 3.0,
            "scale_pos_weight": pos_weight,
        },
    ]


def cv_tune_lgbm(
    spec: FeatureSpec,
    train_df: pd.DataFrame,
    params_grid: list[dict[str, Any]],
    n_splits: int = 3,
) -> tuple[dict[str, Any], pd.DataFrame]:
    y = train_df["dep"].to_numpy()
    weights = train_df["W"].to_numpy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    records: list[dict[str, Any]] = []

    for grid_index, params in enumerate(params_grid, start=1):
        fold_scores: list[float] = []
        fold_seconds: list[float] = []
        for fold_index, (fit_idx, valid_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
            fold_start = time.time()
            fit_df = train_df.iloc[fit_idx]
            valid_df = train_df.iloc[valid_idx]
            model = make_lgbm_pipeline(spec, params)
            model.fit(fit_df[spec.columns], fit_df["dep"])
            score = weighted_auc(
                model,
                valid_df[spec.columns],
                valid_df["dep"],
                pd.Series(weights[valid_idx]),
            )
            fold_scores.append(score)
            fold_seconds.append(time.time() - fold_start)
            print(
                f"CV {grid_index:02d}/{len(params_grid):02d} {params['name']} "
                f"fold {fold_index}/{n_splits}: weighted_auc={score:.6f} "
                f"seconds={fold_seconds[-1]:.1f}",
                flush=True,
            )
        records.append(
            {
                "name": params["name"],
                "mean_weighted_cv_auroc": float(np.mean(fold_scores)),
                "sd_weighted_cv_auroc": float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else math.nan,
                "fold_weighted_cv_aurocs": "|".join(f"{s:.8f}" for s in fold_scores),
                "mean_seconds": float(np.mean(fold_seconds)),
                "params": json.dumps({k: v for k, v in params.items() if k != "name"}, sort_keys=True),
            }
        )

    cv_results = pd.DataFrame(records).sort_values(
        ["mean_weighted_cv_auroc", "sd_weighted_cv_auroc"],
        ascending=[False, True],
    )
    best_name = str(cv_results.iloc[0]["name"])
    best_params = next(p for p in params_grid if p["name"] == best_name)
    return best_params, cv_results


def fit_score_alt_model(
    name: str,
    spec: FeatureSpec,
    params: dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    start = time.time()
    if name == "elasticnet":
        model = make_elasticnet_pipeline(spec, params)
    elif name == "randomforest":
        model = make_rf_pipeline(spec, params)
    else:
        raise ValueError(name)
    model.fit(train_df[spec.columns], train_df["dep"])
    auc = weighted_auc(model, test_df[spec.columns], test_df["dep"], test_df["W"])
    return {
        "step": "alt_model_on_strict_step4_features",
        "model": name,
        "weighted_holdout_auroc": auc,
        "raw_feature_count": len(spec.columns),
        "params": json.dumps(params, sort_keys=True),
        "seconds": round(time.time() - start, 3),
    }


def main() -> int:
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    run_start = time.time()

    required_cols = sorted(
        set(
            ["dep", "W"]
            + BASE_NUMERIC
            + BASE_CATEGORICAL
            + ["living_alone", "active", "current_drinker", "srh", "stress"]
        )
    )
    df = pd.read_parquet(DATA_PATH, columns=required_cols)
    original_n = len(df)
    df = df.loc[df["dep"].notna() & df["W"].notna() & (pd.to_numeric(df["W"], errors="coerce") > 0)].copy()
    df["dep"] = df["dep"].astype(int)
    df["W"] = pd.to_numeric(df["W"], errors="coerce").astype(float)
    df = add_protocol_engineered_features(df)

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        stratify=df["dep"],
        random_state=RANDOM_STATE,
    )
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    specs = {
        "1_baseline_current_10": FeatureSpec(BASE_NUMERIC, BASE_CATEGORICAL),
        "2_plus_living_alone": FeatureSpec(STEP2_NUMERIC, BASE_CATEGORICAL),
        "3_plus_active_current_drinker": FeatureSpec(STEP3_NUMERIC, BASE_CATEGORICAL),
        "4_plus_protocol_age_sleep_categories": FeatureSpec(
            STEP3_NUMERIC,
            BASE_CATEGORICAL + STRICT_ENGINEERED_CATEGORICAL,
        ),
        "6_gray_plus_self_rated_health": FeatureSpec(
            STEP3_NUMERIC + ["srh"],
            BASE_CATEGORICAL + STRICT_ENGINEERED_CATEGORICAL,
        ),
        "7_violation_plus_stress_upper_bound": FeatureSpec(
            STEP3_NUMERIC + ["srh", "stress"],
            BASE_CATEGORICAL + STRICT_ENGINEERED_CATEGORICAL,
        ),
    }

    records: list[dict[str, Any]] = []
    for step in [
        "1_baseline_current_10",
        "2_plus_living_alone",
        "3_plus_active_current_drinker",
        "4_plus_protocol_age_sleep_categories",
    ]:
        print(f"Fitting default LightGBM: {step}", flush=True)
        row = fit_score_lgbm(step, specs[step], {"name": "default"}, train_df, test_df)
        records.append(row)
        print(f"{step}: weighted_holdout_auroc={row['weighted_holdout_auroc']:.6f}", flush=True)

    train_pos = int(train_df["dep"].sum())
    train_neg = int(len(train_df) - train_pos)
    pos_weight = train_neg / train_pos
    grid = lgbm_grid(pos_weight=pos_weight)

    print("Running full-training 3-fold weighted-AUROC tuning for strict step-4 features...", flush=True)
    best_params, cv_results = cv_tune_lgbm(specs["4_plus_protocol_age_sleep_categories"], train_df, grid)
    cv_path = OUT_DIR / "strict_lgbm_cv_results.csv"
    cv_results.to_csv(cv_path, index=False)
    print(f"Best strict tuning config: {best_params['name']}", flush=True)

    for step in [
        "2_plus_living_alone",
        "3_plus_active_current_drinker",
        "4_plus_protocol_age_sleep_categories",
    ]:
        tuned_step = step.replace("plus", "tuned_plus") if step != "4_plus_protocol_age_sleep_categories" else "5_tuned_lgbm_strict_protocol_features"
        print(f"Fitting tuned LightGBM ({best_params['name']}): {tuned_step}", flush=True)
        row = fit_score_lgbm(tuned_step, specs[step], best_params, train_df, test_df)
        records.append(row)
        print(f"{tuned_step}: weighted_holdout_auroc={row['weighted_holdout_auroc']:.6f}", flush=True)

    for step in [
        "6_gray_plus_self_rated_health",
        "7_violation_plus_stress_upper_bound",
    ]:
        print(f"Fitting tuned LightGBM ({best_params['name']}): {step}", flush=True)
        row = fit_score_lgbm(step, specs[step], best_params, train_df, test_df)
        records.append(row)
        print(f"{step}: weighted_holdout_auroc={row['weighted_holdout_auroc']:.6f}", flush=True)

    alt_records: list[dict[str, Any]] = []
    alt_specs = specs["4_plus_protocol_age_sleep_categories"]
    alt_jobs = [
        ("elasticnet", {"C": 1.0, "l1_ratio": 0.5, "max_iter": 500}),
        ("randomforest", {"n_estimators": 300, "max_depth": 14, "min_samples_leaf": 100, "max_features": "sqrt"}),
    ]
    for model_name, params in alt_jobs:
        print(f"Fitting alternative model on strict step-4 features: {model_name}", flush=True)
        row = fit_score_alt_model(model_name, alt_specs, params, train_df, test_df)
        alt_records.append(row)
        print(f"{model_name}: weighted_holdout_auroc={row['weighted_holdout_auroc']:.6f}", flush=True)

    results = pd.DataFrame(records)
    results_path = OUT_DIR / "strict_auroc_ladder_results.csv"
    results.to_csv(results_path, index=False)

    alt_results = pd.DataFrame(alt_records)
    alt_path = OUT_DIR / "strict_alt_model_results.csv"
    alt_results.to_csv(alt_path, index=False)

    metadata = {
        "data_path": str(DATA_PATH.relative_to(ROOT)),
        "original_n": int(original_n),
        "analytic_n": int(len(df)),
        "train_n": int(len(train_df)),
        "test_n": int(len(test_df)),
        "train_events": train_pos,
        "test_events": int(test_df["dep"].sum()),
        "train_weight_sum": float(train_df["W"].sum()),
        "test_weight_sum": float(test_df["W"].sum()),
        "random_state": RANDOM_STATE,
        "test_size": 0.2,
        "stratify": "dep",
        "weighted_training_primary": False,
        "weighted_evaluation": True,
        "best_lgbm_config": best_params["name"],
        "best_lgbm_params": {k: v for k, v in best_params.items() if k != "name"},
        "pos_weight": float(pos_weight),
        "engineered_definitions": {
            "age_band_10y": "floor(age / 10) * 10 decade labels, with 90+ top code",
            "sleep_cat": "short <7 hours, normal 7-9 hours, long >9 hours",
        },
        "output_files": [
            str(results_path.relative_to(ROOT)),
            str(cv_path.relative_to(ROOT)),
            str(alt_path.relative_to(ROOT)),
            "experiments_codex/strict_run_metadata.json",
        ],
        "elapsed_seconds": round(time.time() - run_start, 3),
        "python": sys.version,
    }
    metadata_path = OUT_DIR / "strict_run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nStrict AUROC ladder:")
    print(results[["step", "model", "tuning_config", "weighted_holdout_auroc", "seconds"]].to_string(index=False))
    print("\nAlternative models:")
    print(alt_results[["model", "weighted_holdout_auroc", "seconds"]].to_string(index=False))
    print(
        f"\nWrote {results_path.relative_to(ROOT)}, {cv_path.relative_to(ROOT)}, "
        f"{alt_path.relative_to(ROOT)}, {metadata_path.relative_to(ROOT)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
