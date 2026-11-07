from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "kchs_dev.parquet"

BASE_NUMERIC = [
    "age",
    "sleep_hrs",
    "bmi",
    "income_q",
    "partnered",
    "htn",
    "dm",
]
BASE_CATEGORICAL = ["sex", "edu", "smoking"]

STRICT_STEP2_NUMERIC = BASE_NUMERIC + ["living_alone"]
STRICT_STEP3_NUMERIC = STRICT_STEP2_NUMERIC + ["active", "current_drinker"]
ENGINEERED_CATEGORICAL = ["age_band_10y", "sleep_cat", "bmi_cat"]
SRH_NUMERIC = STRICT_STEP3_NUMERIC + ["srh"]
STRESS_NUMERIC = SRH_NUMERIC + ["stress"]

DEFAULT_MODEL_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "n_estimators": 100,
    "learning_rate": 0.1,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": -1,
    "force_col_wise": True,
}

TUNING_GRID: list[dict[str, Any]] = [
    {
        "name": "g01_more_trees_small_leaves",
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
        "name": "g02_more_trees_regularized",
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
        "name": "g03_medium_leaves_regularized",
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
        "name": "g04_medium_leaves_heavy_regularization",
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
        "name": "g05_large_leaves_heavy_regularization",
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
]


@dataclass(frozen=True)
class FeatureSpec:
    numeric: list[str]
    categorical: list[str]


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def make_pipeline(spec: FeatureSpec, model_params: dict[str, Any]) -> Pipeline:
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
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipe, spec.numeric),
            ("cat", categorical_pipe, spec.categorical),
        ]
    )
    params = DEFAULT_MODEL_PARAMS | {k: v for k, v in model_params.items() if k != "name"}
    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", LGBMClassifier(**params)),
        ]
    )


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    age = pd.to_numeric(out["age"], errors="coerce")
    age_decade = np.floor(age / 10) * 10
    age_band = pd.Series(np.nan, index=out.index, dtype="object")
    valid_age = age_decade.notna()
    age_band.loc[valid_age] = age_decade.loc[valid_age].astype(int).astype(str) + "s"
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


def weighted_auc(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series, w_test: pd.Series) -> float:
    pred = model.predict_proba(x_test)[:, 1]
    return float(roc_auc_score(y_test, pred, sample_weight=w_test))


def fit_and_score(
    label: str,
    spec: FeatureSpec,
    params: dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    start = time.time()
    model = make_pipeline(spec, params)
    model.fit(train_df[spec.numeric + spec.categorical], train_df["dep"])
    auc = weighted_auc(
        model,
        test_df[spec.numeric + spec.categorical],
        test_df["dep"],
        test_df["W"],
    )
    return {
        "step": label,
        "weighted_holdout_auroc": auc,
        "n_features_raw": len(spec.numeric) + len(spec.categorical),
        "numeric_features": "|".join(spec.numeric),
        "categorical_features": "|".join(spec.categorical),
        "model_params": json.dumps(DEFAULT_MODEL_PARAMS | {k: v for k, v in params.items() if k != "name"}, sort_keys=True),
        "seconds": round(time.time() - start, 3),
    }


def cv_tune_lgbm(
    spec: FeatureSpec,
    train_df: pd.DataFrame,
    n_splits: int = 3,
) -> tuple[dict[str, Any], pd.DataFrame]:
    y = train_df["dep"].to_numpy()
    weights = train_df["W"].to_numpy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    records: list[dict[str, Any]] = []
    columns = spec.numeric + spec.categorical

    for grid_index, params in enumerate(TUNING_GRID, start=1):
        fold_scores: list[float] = []
        fold_seconds: list[float] = []
        for fold_index, (fit_idx, valid_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
            fold_start = time.time()
            fit_df = train_df.iloc[fit_idx]
            valid_df = train_df.iloc[valid_idx]
            model = make_pipeline(spec, params)
            model.fit(fit_df[columns], fit_df["dep"])
            score = weighted_auc(model, valid_df[columns], valid_df["dep"], pd.Series(weights[valid_idx]))
            fold_scores.append(score)
            fold_seconds.append(time.time() - fold_start)
            print(
                f"CV {grid_index}/{len(TUNING_GRID)} {params['name']} fold {fold_index}/{n_splits}: "
                f"weighted_auc={score:.6f} seconds={fold_seconds[-1]:.1f}",
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
    best_name = cv_results.iloc[0]["name"]
    best_params = next(p for p in TUNING_GRID if p["name"] == best_name)
    return best_params, cv_results


def main() -> int:
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    start = time.time()

    needed_columns = sorted(
        set(
            ["dep", "W"]
            + BASE_NUMERIC
            + BASE_CATEGORICAL
            + ["living_alone", "active", "current_drinker", "srh", "stress"]
        )
    )
    df = pd.read_parquet(DATA_PATH, columns=needed_columns)
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
        "1_baseline_current_10": FeatureSpec(BASE_NUMERIC, BASE_CATEGORICAL),
        "2_plus_living_alone": FeatureSpec(STRICT_STEP2_NUMERIC, BASE_CATEGORICAL),
        "3_plus_active_current_drinker": FeatureSpec(STRICT_STEP3_NUMERIC, BASE_CATEGORICAL),
        "4_plus_table1_engineered_categories": FeatureSpec(
            STRICT_STEP3_NUMERIC,
            BASE_CATEGORICAL + ENGINEERED_CATEGORICAL,
        ),
        "5_tuned_lgbm_strict_features": FeatureSpec(
            STRICT_STEP3_NUMERIC,
            BASE_CATEGORICAL + ENGINEERED_CATEGORICAL,
        ),
        "6_gray_plus_self_rated_health": FeatureSpec(
            SRH_NUMERIC,
            BASE_CATEGORICAL + ENGINEERED_CATEGORICAL,
        ),
        "7_deviation_plus_perceived_stress": FeatureSpec(
            STRESS_NUMERIC,
            BASE_CATEGORICAL + ENGINEERED_CATEGORICAL,
        ),
    }

    records: list[dict[str, Any]] = []
    for step in [
        "1_baseline_current_10",
        "2_plus_living_alone",
        "3_plus_active_current_drinker",
        "4_plus_table1_engineered_categories",
    ]:
        print(f"Fitting {step} with default LightGBM...", flush=True)
        records.append(fit_and_score(step, specs[step], {}, train_df, test_df))
        print(f"{step}: weighted_holdout_auroc={records[-1]['weighted_holdout_auroc']:.6f}", flush=True)

    print("Running 3-fold weighted-AUROC CV tuning on strict step-4 feature set...", flush=True)
    best_params, cv_results = cv_tune_lgbm(specs["5_tuned_lgbm_strict_features"], train_df)
    cv_path = OUT_DIR / "tuned_cv_results.csv"
    cv_results.to_csv(cv_path, index=False)
    print(f"Best tuning config: {best_params['name']}", flush=True)

    for step in [
        "5_tuned_lgbm_strict_features",
        "6_gray_plus_self_rated_health",
        "7_deviation_plus_perceived_stress",
    ]:
        print(f"Fitting {step} with tuned config {best_params['name']}...", flush=True)
        records.append(fit_and_score(step, specs[step], best_params, train_df, test_df))
        print(f"{step}: weighted_holdout_auroc={records[-1]['weighted_holdout_auroc']:.6f}", flush=True)

    results = pd.DataFrame(records)
    results_path = OUT_DIR / "auroc_ladder_results.csv"
    results.to_csv(results_path, index=False)

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
        "weighted_training": False,
        "weighted_evaluation": True,
        "best_tuning_config": best_params["name"],
        "best_tuning_params": {k: v for k, v in best_params.items() if k != "name"},
        "engineered_definitions": {
            "age_band_10y": "floor(age / 10) * 10 decade labels, with 90+ top code",
            "sleep_cat": "short <7 hours, normal 7-9 hours, long >9 hours",
            "bmi_cat": "WHO-style categories: <18.5, 18.5-<25, 25-<30, >=30",
        },
        "elapsed_seconds": round(time.time() - start, 3),
        "python": sys.version,
    }
    metadata_path = OUT_DIR / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nAUROC ladder:")
    print(results[["step", "weighted_holdout_auroc", "seconds"]].to_string(index=False))
    print(f"\nWrote {results_path.relative_to(ROOT)}, {cv_path.relative_to(ROOT)}, {metadata_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
