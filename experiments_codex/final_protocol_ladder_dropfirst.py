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
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "kchs_dev.parquet"
ALT_TUNING_MAX_N = 300_000

BASE_NUMERIC = ["age", "sleep_hrs", "bmi", "income_q", "partnered", "htn", "dm"]
BASE_CATEGORICAL = ["sex", "edu", "smoking"]

PLUS_A_NUMERIC = BASE_NUMERIC + ["living_alone", "active", "current_drinker"]
ENGINEERED_CATEGORICAL = ["age_band_10y", "sleep_cat", "bmi_cat"]
PLUS_B_CATEGORICAL = BASE_CATEGORICAL + ENGINEERED_CATEGORICAL

DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "n_estimators": 100,
    "learning_rate": 0.1,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": -1,
    "force_col_wise": True,
}

LGBM_GRID: list[dict[str, Any]] = [
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
]

ELASTICNET_GRID: list[dict[str, Any]] = [
    {"name": "en_C0.1_l1r0.0", "C": 0.1, "l1_ratio": 0.0, "max_iter": 700},
    {"name": "en_C0.3_l1r0.25", "C": 0.3, "l1_ratio": 0.25, "max_iter": 700},
    {"name": "en_C1_l1r0.5", "C": 1.0, "l1_ratio": 0.5, "max_iter": 700},
    {"name": "en_C3_l1r0.5", "C": 3.0, "l1_ratio": 0.5, "max_iter": 700},
]

RF_GRID: list[dict[str, Any]] = [
    {
        "name": "rf_depth12_leaf100",
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_leaf": 100,
        "max_features": "sqrt",
    },
    {
        "name": "rf_depth14_leaf100",
        "n_estimators": 300,
        "max_depth": 14,
        "min_samples_leaf": 100,
        "max_features": "sqrt",
    },
    {
        "name": "rf_depth16_leaf200",
        "n_estimators": 300,
        "max_depth": 16,
        "min_samples_leaf": 200,
        "max_features": "sqrt",
    },
]


@dataclass(frozen=True)
class FeatureSpec:
    numeric: list[str]
    categorical: list[str]

    @property
    def columns(self) -> list[str]:
        return self.numeric + self.categorical


def make_one_hot_encoder() -> OneHotEncoder:
    kwargs: dict[str, Any] = {"handle_unknown": "ignore", "drop": "first"}
    try:
        return OneHotEncoder(sparse_output=True, **kwargs)
    except TypeError:
        return OneHotEncoder(sparse=True, **kwargs)


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
        C=params["C"],
        l1_ratio=params["l1_ratio"],
        max_iter=params.get("max_iter", 700),
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    return Pipeline([("preprocess", make_preprocessor(spec)), ("model", model)])


def make_rf_pipeline(spec: FeatureSpec, params: dict[str, Any]) -> Pipeline:
    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params.get("max_features", "sqrt"),
        class_weight=params.get("class_weight"),
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocess", make_preprocessor(spec)), ("model", model)])


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
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

    bmi = pd.to_numeric(out["bmi"], errors="coerce")
    bmi_cat = pd.Series(np.nan, index=out.index, dtype="object")
    bmi_cat.loc[bmi < 18.5] = "underweight_lt18_5"
    bmi_cat.loc[(bmi >= 18.5) & (bmi < 25)] = "normal_18_5to24_9"
    bmi_cat.loc[(bmi >= 25) & (bmi < 30)] = "overweight_25to29_9"
    bmi_cat.loc[bmi >= 30] = "obesity_ge30"
    out["bmi_cat"] = bmi_cat

    return out


def weighted_auc(model: Pipeline, x_df: pd.DataFrame, y: pd.Series, w: pd.Series) -> float:
    pred = model.predict_proba(x_df)[:, 1]
    return float(roc_auc_score(y, pred, sample_weight=w))


def fit_lgbm(
    step: str,
    spec: FeatureSpec,
    params: dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    start = time.time()
    model = make_lgbm_pipeline(spec, params)
    model.fit(train_df[spec.columns], train_df["dep"])
    auc = weighted_auc(model, test_df[spec.columns], test_df["dep"], test_df["W"])
    model_params = DEFAULT_LGBM_PARAMS | {k: v for k, v in params.items() if k != "name"}
    return {
        "step": step,
        "model": "lightgbm",
        "tuning_config": params.get("name", "default"),
        "weighted_holdout_auroc": auc,
        "raw_feature_count": len(spec.columns),
        "numeric_features": "|".join(spec.numeric),
        "categorical_features": "|".join(spec.categorical),
        "model_params": json.dumps(model_params, sort_keys=True),
        "seconds": round(time.time() - start, 3),
    }


def cv_tune(
    model_kind: str,
    spec: FeatureSpec,
    train_df: pd.DataFrame,
    grid: list[dict[str, Any]],
    n_splits: int = 3,
) -> tuple[dict[str, Any], pd.DataFrame]:
    y = train_df["dep"].to_numpy()
    weights = train_df["W"].to_numpy()
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    records: list[dict[str, Any]] = []

    for grid_index, params in enumerate(grid, start=1):
        fold_scores: list[float] = []
        fold_seconds: list[float] = []
        for fold_index, (fit_idx, valid_idx) in enumerate(splitter.split(np.zeros(len(y)), y), start=1):
            fold_start = time.time()
            fit_df = train_df.iloc[fit_idx]
            valid_df = train_df.iloc[valid_idx]
            if model_kind == "lightgbm":
                model = make_lgbm_pipeline(spec, params)
            elif model_kind == "elasticnet":
                model = make_elasticnet_pipeline(spec, params)
            elif model_kind == "randomforest":
                model = make_rf_pipeline(spec, params)
            else:
                raise ValueError(model_kind)

            model.fit(fit_df[spec.columns], fit_df["dep"])
            score = weighted_auc(model, valid_df[spec.columns], valid_df["dep"], pd.Series(weights[valid_idx]))
            fold_scores.append(score)
            fold_seconds.append(time.time() - fold_start)
            print(
                f"CV {model_kind} {grid_index:02d}/{len(grid):02d} {params['name']} "
                f"fold {fold_index}/{n_splits}: weighted_auc={score:.6f} seconds={fold_seconds[-1]:.1f}",
                flush=True,
            )

        records.append(
            {
                "model": model_kind,
                "name": params["name"],
                "mean_weighted_cv_auroc": float(np.mean(fold_scores)),
                "sd_weighted_cv_auroc": float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else math.nan,
                "fold_weighted_cv_aurocs": "|".join(f"{score:.8f}" for score in fold_scores),
                "mean_seconds": float(np.mean(fold_seconds)),
                "params": json.dumps({k: v for k, v in params.items() if k != "name"}, sort_keys=True),
            }
        )

    cv_results = pd.DataFrame(records).sort_values(
        ["mean_weighted_cv_auroc", "sd_weighted_cv_auroc"],
        ascending=[False, True],
    )
    best_name = str(cv_results.iloc[0]["name"])
    best_params = next(params for params in grid if params["name"] == best_name)
    return best_params, cv_results


def stratified_tuning_frame(train_df: pd.DataFrame, max_n: int) -> pd.DataFrame:
    if len(train_df) <= max_n:
        return train_df
    sample_idx, _ = train_test_split(
        np.arange(len(train_df)),
        train_size=max_n,
        stratify=train_df["dep"],
        random_state=RANDOM_STATE,
    )
    return train_df.iloc[sample_idx].copy()


def fit_alt_holdout(
    model_kind: str,
    spec: FeatureSpec,
    params: dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    start = time.time()
    if model_kind == "elasticnet":
        model = make_elasticnet_pipeline(spec, params)
    elif model_kind == "randomforest":
        model = make_rf_pipeline(spec, params)
    else:
        raise ValueError(model_kind)
    model.fit(train_df[spec.columns], train_df["dep"])
    auc = weighted_auc(model, test_df[spec.columns], test_df["dep"], test_df["W"])
    return {
        "step": "best_compliant_A_plus_B_features",
        "model": model_kind,
        "tuning_config": params["name"],
        "weighted_holdout_auroc": auc,
        "raw_feature_count": len(spec.columns),
        "params": json.dumps({k: v for k, v in params.items() if k != "name"}, sort_keys=True),
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
    df = add_engineered_features(df)

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        stratify=df["dep"],
        random_state=RANDOM_STATE,
    )
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    specs = {
        "baseline": FeatureSpec(BASE_NUMERIC, BASE_CATEGORICAL),
        "plus_A": FeatureSpec(PLUS_A_NUMERIC, BASE_CATEGORICAL),
        "plus_A_B": FeatureSpec(PLUS_A_NUMERIC, PLUS_B_CATEGORICAL),
        "plus_A_B_D": FeatureSpec(PLUS_A_NUMERIC + ["srh"], PLUS_B_CATEGORICAL),
        "plus_A_B_D_E": FeatureSpec(PLUS_A_NUMERIC + ["srh", "stress"], PLUS_B_CATEGORICAL),
    }

    ladder_records: list[dict[str, Any]] = []
    for step, spec in [
        ("baseline", specs["baseline"]),
        ("plus_A", specs["plus_A"]),
        ("plus_A_B", specs["plus_A_B"]),
    ]:
        print(f"Fitting default LightGBM: {step}", flush=True)
        row = fit_lgbm(step, spec, {"name": "default"}, train_df, test_df)
        ladder_records.append(row)
        print(f"{step}: weighted_holdout_auroc={row['weighted_holdout_auroc']:.6f}", flush=True)

    print("Running 3-fold weighted-AUROC CV tuning for compliant A+B feature set...", flush=True)
    best_lgbm_params, lgbm_cv = cv_tune("lightgbm", specs["plus_A_B"], train_df, LGBM_GRID)
    lgbm_cv_path = OUT_DIR / "final_dropfirst_lgbm_cv_results.csv"
    lgbm_cv.to_csv(lgbm_cv_path, index=False)
    print(f"Best LightGBM config: {best_lgbm_params['name']}", flush=True)

    for step, spec in [
        ("plus_A_B_C", specs["plus_A_B"]),
        ("plus_A_B_C_D", specs["plus_A_B_D"]),
        ("plus_A_B_C_D_E", specs["plus_A_B_D_E"]),
    ]:
        print(f"Fitting tuned LightGBM ({best_lgbm_params['name']}): {step}", flush=True)
        row = fit_lgbm(step, spec, best_lgbm_params, train_df, test_df)
        ladder_records.append(row)
        print(f"{step}: weighted_holdout_auroc={row['weighted_holdout_auroc']:.6f}", flush=True)

    print("Running 3-fold weighted-AUROC CV tuning for elastic-net logistic on A+B features...", flush=True)
    alt_tune_df = stratified_tuning_frame(train_df, ALT_TUNING_MAX_N)
    print(
        f"Alternative-model tuning frame: n={len(alt_tune_df)} "
        f"events={int(alt_tune_df['dep'].sum())}",
        flush=True,
    )
    best_en_params, en_cv = cv_tune("elasticnet", specs["plus_A_B"], alt_tune_df, ELASTICNET_GRID)
    en_cv_path = OUT_DIR / "final_dropfirst_elasticnet_cv_results.csv"
    en_cv.to_csv(en_cv_path, index=False)
    print(f"Best elastic-net config: {best_en_params['name']}", flush=True)

    print("Running 3-fold weighted-AUROC CV tuning for random forest on A+B features...", flush=True)
    best_rf_params, rf_cv = cv_tune("randomforest", specs["plus_A_B"], alt_tune_df, RF_GRID)
    rf_cv_path = OUT_DIR / "final_dropfirst_rf_cv_results.csv"
    rf_cv.to_csv(rf_cv_path, index=False)
    print(f"Best random-forest config: {best_rf_params['name']}", flush=True)

    alt_records = [
        fit_alt_holdout("elasticnet", specs["plus_A_B"], best_en_params, train_df, test_df),
        fit_alt_holdout("randomforest", specs["plus_A_B"], best_rf_params, train_df, test_df),
    ]

    ladder_path = OUT_DIR / "final_dropfirst_ladder_results.csv"
    pd.DataFrame(ladder_records).to_csv(ladder_path, index=False)
    alt_path = OUT_DIR / "final_dropfirst_alt_model_results.csv"
    pd.DataFrame(alt_records).to_csv(alt_path, index=False)

    metadata = {
        "data_path": str(DATA_PATH.relative_to(ROOT)),
        "original_n": int(original_n),
        "analytic_n": int(len(df)),
        "train_n": int(len(train_df)),
        "test_n": int(len(test_df)),
        "train_events": int(train_df["dep"].sum()),
        "test_events": int(test_df["dep"].sum()),
        "train_weight_sum": float(train_df["W"].sum()),
        "test_weight_sum": float(test_df["W"].sum()),
        "random_state": RANDOM_STATE,
        "test_size": 0.2,
        "stratify": "dep",
        "categorical_encoding": "OneHotEncoder(drop='first', handle_unknown='ignore')",
        "numeric_preprocessing": "SimpleImputer(strategy='median') + StandardScaler()",
        "categorical_preprocessing": "SimpleImputer(strategy='most_frequent') + one-hot",
        "weighted_training_primary": False,
        "weighted_evaluation": True,
        "alt_model_tuning_n": int(len(alt_tune_df)),
        "alt_model_tuning_events": int(alt_tune_df["dep"].sum()),
        "alt_model_tuning_note": "Elastic-net and random-forest grids tuned on stratified training subsample; selected settings refit on full training split for holdout AUROC.",
        "best_lgbm_config": best_lgbm_params["name"],
        "best_lgbm_params": {k: v for k, v in best_lgbm_params.items() if k != "name"},
        "best_elasticnet_config": best_en_params["name"],
        "best_elasticnet_params": {k: v for k, v in best_en_params.items() if k != "name"},
        "best_rf_config": best_rf_params["name"],
        "best_rf_params": {k: v for k, v in best_rf_params.items() if k != "name"},
        "engineered_definitions": {
            "age_band_10y": "floor(age / 10) * 10 decade labels, with 90+ top code",
            "sleep_cat": "short <7 hours, normal 7-9 hours, long >9 hours",
            "bmi_cat": "WHO-style categories: <18.5, 18.5-<25, 25-<30, >=30",
        },
        "output_files": [
            str(ladder_path.relative_to(ROOT)),
            str(lgbm_cv_path.relative_to(ROOT)),
            str(en_cv_path.relative_to(ROOT)),
            str(rf_cv_path.relative_to(ROOT)),
            str(alt_path.relative_to(ROOT)),
            "experiments_codex/final_dropfirst_run_metadata.json",
        ],
        "elapsed_seconds": round(time.time() - run_start, 3),
        "python": sys.version,
    }
    metadata_path = OUT_DIR / "final_dropfirst_run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nFinal drop-first ladder:")
    print(pd.DataFrame(ladder_records)[["step", "model", "tuning_config", "weighted_holdout_auroc", "seconds"]].to_string(index=False))
    print("\nAlternative models on best compliant features:")
    print(pd.DataFrame(alt_records)[["model", "tuning_config", "weighted_holdout_auroc", "seconds"]].to_string(index=False))
    print(
        f"\nWrote {ladder_path.relative_to(ROOT)}, {lgbm_cv_path.relative_to(ROOT)}, "
        f"{en_cv_path.relative_to(ROOT)}, {rf_cv_path.relative_to(ROOT)}, "
        f"{alt_path.relative_to(ROOT)}, {metadata_path.relative_to(ROOT)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
