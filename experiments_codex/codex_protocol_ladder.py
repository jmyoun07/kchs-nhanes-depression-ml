from __future__ import annotations

import json
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
from sklearn.impute import SimpleImputer
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

BASE_NUM = ["age", "sleep_hrs", "bmi", "income_q"]
BASE_BIN = ["partnered", "htn", "dm"]
BASE_CAT = ["sex", "edu", "smoking"]

LEVER1_BIN = BASE_BIN + ["living_alone", "active", "current_drinker"]
ENGINEERED_CAT = ["age_band_10y", "sleep_cat", "bmi_asian_cat"]

SRC_STYLE_LGBM: dict[str, Any] = {
    "n_estimators": 600,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}

LGBM_GRID: list[dict[str, Any]] = [
    {"name": "src_style"},
    {
        "name": "regularized_31",
        "n_estimators": 600,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 200,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 2.0,
    },
    {
        "name": "long_shallow_15",
        "n_estimators": 1000,
        "learning_rate": 0.015,
        "num_leaves": 15,
        "min_child_samples": 500,
        "subsample": 0.9,
        "colsample_bytree": 0.95,
        "reg_lambda": 3.0,
    },
    {
        "name": "medium_63",
        "n_estimators": 700,
        "learning_rate": 0.025,
        "num_leaves": 63,
        "min_child_samples": 200,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 3.0,
    },
    {
        "name": "heavy_reg_63",
        "n_estimators": 900,
        "learning_rate": 0.02,
        "num_leaves": 63,
        "min_child_samples": 500,
        "subsample": 0.85,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.1,
        "reg_lambda": 5.0,
    },
    {
        "name": "deeper_127",
        "n_estimators": 600,
        "learning_rate": 0.03,
        "num_leaves": 127,
        "min_child_samples": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 5.0,
    },
    {
        "name": "balanced_weight",
        "n_estimators": 600,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_child_samples": 200,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 3.0,
        "class_weight": "balanced",
    },
]


@dataclass(frozen=True)
class FeatureSpec:
    num: list[str]
    bin: list[str]
    cat: list[str]

    @property
    def columns(self) -> list[str]:
        return self.num + self.bin + self.cat


def make_one_hot_encoder() -> OneHotEncoder:
    kwargs: dict[str, Any] = {"handle_unknown": "ignore", "drop": "first"}
    try:
        return OneHotEncoder(sparse_output=True, **kwargs)
    except TypeError:
        return OneHotEncoder(sparse=True, **kwargs)


def make_preprocessor(spec: FeatureSpec) -> ColumnTransformer:
    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    bin_pipe = SimpleImputer(strategy="most_frequent")
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        [
            ("num", num_pipe, spec.num),
            ("bin", bin_pipe, spec.bin),
            ("cat", cat_pipe, spec.cat),
        ]
    )


def lgbm_params(params: dict[str, Any]) -> dict[str, Any]:
    return SRC_STYLE_LGBM | {k: v for k, v in params.items() if k != "name"}


def make_lgbm(spec: FeatureSpec, params: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("pre", make_preprocessor(spec)),
            ("clf", LGBMClassifier(**lgbm_params(params))),
        ]
    )


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    age = pd.to_numeric(out["age"], errors="coerce")
    age_floor = np.floor(age / 10) * 10
    age_band = pd.Series(np.nan, index=out.index, dtype="object")
    valid_age = age_floor.notna()
    age_band.loc[valid_age] = age_floor.loc[valid_age].astype(int).astype(str) + "s"
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
    bmi_cat.loc[(bmi >= 18.5) & (bmi < 23.0)] = "normal_18_5to22_9"
    bmi_cat.loc[(bmi >= 23.0) & (bmi < 25.0)] = "overweight_23to24_9"
    bmi_cat.loc[(bmi >= 25.0) & (bmi < 30.0)] = "obese_I_25to29_9"
    bmi_cat.loc[bmi >= 30.0] = "obese_II_ge30"
    out["bmi_asian_cat"] = bmi_cat

    return out


def weighted_holdout_auc(model: Pipeline, test_df: pd.DataFrame, spec: FeatureSpec) -> float:
    pred = model.predict_proba(test_df[spec.columns])[:, 1]
    return float(roc_auc_score(test_df["dep"], pred, sample_weight=test_df["W"]))


def fit_eval(
    step: str,
    spec: FeatureSpec,
    params: dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    started = time.time()
    model = make_lgbm(spec, params)
    model.fit(train_df[spec.columns], train_df["dep"])
    auc = weighted_holdout_auc(model, test_df, spec)
    return {
        "step": step,
        "config": params.get("name", "src_style"),
        "weighted_holdout_auroc": auc,
        "n_raw_features": len(spec.columns),
        "num_features": "|".join(spec.num),
        "bin_features": "|".join(spec.bin),
        "cat_features": "|".join(spec.cat),
        "params": json.dumps(lgbm_params(params), sort_keys=True),
        "seconds": round(time.time() - started, 3),
    }


def stratified_subsample(train_df: pd.DataFrame, max_n: int) -> pd.DataFrame:
    if len(train_df) <= max_n:
        return train_df.copy()
    idx, _ = train_test_split(
        np.arange(len(train_df)),
        train_size=max_n,
        stratify=train_df["dep"],
        random_state=RANDOM_STATE,
    )
    return train_df.iloc[idx].copy()


def tune_lgbm_cv(
    train_df: pd.DataFrame,
    spec: FeatureSpec,
    max_n: int = 300_000,
    n_splits: int = 3,
) -> tuple[dict[str, Any], pd.DataFrame]:
    tune_df = stratified_subsample(train_df, max_n=max_n)
    y = tune_df["dep"].to_numpy()
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows: list[dict[str, Any]] = []
    print(f"CV tuning frame: n={len(tune_df)} events={int(tune_df['dep'].sum())}", flush=True)

    for grid_index, params in enumerate(LGBM_GRID, start=1):
        fold_scores: list[float] = []
        fold_seconds: list[float] = []
        for fold_index, (fit_idx, valid_idx) in enumerate(splitter.split(np.zeros(len(tune_df)), y), start=1):
            fold_started = time.time()
            fit_df = tune_df.iloc[fit_idx]
            valid_df = tune_df.iloc[valid_idx]
            model = make_lgbm(spec, params)
            model.fit(fit_df[spec.columns], fit_df["dep"])
            score = weighted_holdout_auc(model, valid_df, spec)
            fold_scores.append(score)
            fold_seconds.append(time.time() - fold_started)
            print(
                f"CV {grid_index:02d}/{len(LGBM_GRID):02d} {params['name']} "
                f"fold {fold_index}/{n_splits}: weighted_auc={score:.6f} "
                f"seconds={fold_seconds[-1]:.1f}",
                flush=True,
            )

        rows.append(
            {
                "config": params["name"],
                "mean_weighted_cv_auroc": float(np.mean(fold_scores)),
                "sd_weighted_cv_auroc": float(np.std(fold_scores, ddof=1)),
                "fold_weighted_cv_aurocs": "|".join(f"{score:.8f}" for score in fold_scores),
                "mean_fold_seconds": float(np.mean(fold_seconds)),
                "params": json.dumps(lgbm_params(params), sort_keys=True),
            }
        )

    cv = pd.DataFrame(rows).sort_values(
        ["mean_weighted_cv_auroc", "sd_weighted_cv_auroc"],
        ascending=[False, True],
    )
    best_name = str(cv.iloc[0]["config"])
    best_params = next(params for params in LGBM_GRID if params["name"] == best_name)
    return best_params, cv


def main() -> int:
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    started = time.time()

    needed = sorted(
        set(
            ["dep", "W"]
            + BASE_NUM
            + BASE_BIN
            + BASE_CAT
            + ["living_alone", "active", "current_drinker", "srh", "stress"]
        )
    )
    df = pd.read_parquet(DATA_PATH, columns=needed)
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
        "baseline": FeatureSpec(BASE_NUM, BASE_BIN, BASE_CAT),
        "lever1": FeatureSpec(BASE_NUM, LEVER1_BIN, BASE_CAT),
        "lever2": FeatureSpec(BASE_NUM, LEVER1_BIN, BASE_CAT + ENGINEERED_CAT),
        "lever4": FeatureSpec(BASE_NUM + ["srh"], LEVER1_BIN, BASE_CAT + ENGINEERED_CAT),
        "lever5": FeatureSpec(BASE_NUM + ["srh", "stress"], LEVER1_BIN, BASE_CAT + ENGINEERED_CAT),
    }

    print(
        f"KCHS analytic n={len(df)} train={len(train_df)} holdout={len(test_df)} "
        f"holdout_events={int(test_df['dep'].sum())}",
        flush=True,
    )

    ladder: list[dict[str, Any]] = []
    for step, key in [
        ("baseline_current_primary", "baseline"),
        ("lever1_reinclude_table1_drops", "lever1"),
        ("lever2_table1_feature_engineering", "lever2"),
    ]:
        print(f"Fitting source-style LightGBM: {step}", flush=True)
        row = fit_eval(step, specs[key], {"name": "src_style"}, train_df, test_df)
        ladder.append(row)
        print(f"{step}: weighted_holdout_auroc={row['weighted_holdout_auroc']:.6f}", flush=True)

    print("Running CV hyperparameter tuning for lever1+lever2 feature set", flush=True)
    best_params, cv = tune_lgbm_cv(train_df, specs["lever2"])
    cv_path = OUT_DIR / "codex_protocol_lgbm_cv.csv"
    cv.to_csv(cv_path, index=False)
    print(f"Best CV config: {best_params['name']}", flush=True)

    print("Fitting protocol-selected tuned LightGBM", flush=True)
    ladder.append(
        fit_eval(
            "lever3_cv_tuned_lightgbm",
            specs["lever2"],
            best_params,
            train_df,
            test_df,
        )
    )
    print(f"lever3_cv_tuned_lightgbm: weighted_holdout_auroc={ladder[-1]['weighted_holdout_auroc']:.6f}", flush=True)

    print("Fitting gray-area self-rated health model", flush=True)
    ladder.append(
        fit_eval(
            "lever4_gray_plus_self_rated_health",
            specs["lever4"],
            best_params,
            train_df,
            test_df,
        )
    )
    print(f"lever4_gray_plus_self_rated_health: weighted_holdout_auroc={ladder[-1]['weighted_holdout_auroc']:.6f}", flush=True)

    print("Fitting stress reference ceiling model", flush=True)
    ladder.append(
        fit_eval(
            "lever5_deviation_plus_perceived_stress",
            specs["lever5"],
            best_params,
            train_df,
            test_df,
        )
    )
    print(f"lever5_deviation_plus_perceived_stress: weighted_holdout_auroc={ladder[-1]['weighted_holdout_auroc']:.6f}", flush=True)

    print("Exploratory full-grid holdout scan for compliant feature set", flush=True)
    grid_rows: list[dict[str, Any]] = []
    for params in LGBM_GRID:
        row = fit_eval(
            f"exploratory_fullgrid_{params['name']}",
            specs["lever2"],
            params,
            train_df,
            test_df,
        )
        grid_rows.append(row)
        print(f"{params['name']}: full-train holdout weighted_auc={row['weighted_holdout_auroc']:.6f}", flush=True)

    ladder_path = OUT_DIR / "codex_protocol_ladder_results.csv"
    pd.DataFrame(ladder).to_csv(ladder_path, index=False)

    grid_path = OUT_DIR / "codex_protocol_fullgrid_holdout.csv"
    pd.DataFrame(grid_rows).to_csv(grid_path, index=False)

    metadata = {
        "data_path": str(DATA_PATH.relative_to(ROOT)),
        "original_n": int(original_n),
        "analytic_n": int(len(df)),
        "train_n": int(len(train_df)),
        "holdout_n": int(len(test_df)),
        "train_events": int(train_df["dep"].sum()),
        "holdout_events": int(test_df["dep"].sum()),
        "random_state": RANDOM_STATE,
        "test_size": 0.2,
        "split_stratified_by": "dep",
        "weighted_training": False,
        "weighted_evaluation": True,
        "preprocessing": {
            "numeric": "SimpleImputer(median) + StandardScaler",
            "binary": "SimpleImputer(most_frequent)",
            "categorical": "SimpleImputer(most_frequent) + OneHotEncoder(drop='first')",
        },
        "engineered_definitions": {
            "age_band_10y": "10-year bands from floor(age/10)*10; ages >=90 top-coded 90+",
            "sleep_cat": "short <7, normal 7-9, long >9 hours",
            "bmi_asian_cat": "<18.5, 18.5-<23, 23-<25, 25-<30, >=30 kg/m2",
        },
        "best_cv_config": best_params["name"],
        "best_cv_params": lgbm_params(best_params),
        "outputs": [
            str(ladder_path.relative_to(ROOT)),
            str(cv_path.relative_to(ROOT)),
            str(grid_path.relative_to(ROOT)),
            "experiments_codex/codex_protocol_ladder_metadata.json",
        ],
        "elapsed_seconds": round(time.time() - started, 3),
        "python": sys.version,
    }
    metadata_path = OUT_DIR / "codex_protocol_ladder_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nAUROC ladder")
    print(pd.DataFrame(ladder)[["step", "config", "weighted_holdout_auroc", "seconds"]].to_string(index=False))
    print("\nBest exploratory compliant full-grid holdout")
    grid_df = pd.DataFrame(grid_rows).sort_values("weighted_holdout_auroc", ascending=False)
    print(grid_df[["config", "weighted_holdout_auroc", "seconds"]].head(5).to_string(index=False))
    print(f"\nWrote {ladder_path}, {cv_path}, {grid_path}, {metadata_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
