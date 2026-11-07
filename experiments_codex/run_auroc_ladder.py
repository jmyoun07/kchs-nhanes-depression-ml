from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
DATA_PATH = Path("data/processed/kchs_dev.parquet")
OUT_PATH = Path("experiments_codex/auroc_ladder_results.json")

BASE_NUMERIC = ["age", "sleep_hrs", "bmi", "income_q", "partnered", "htn", "dm"]
BASE_CATEGORICAL = ["sex", "edu", "smoking"]

FEATURE_SETS = {
    "v0": {
        "label": "baseline current 10 variables",
        "numeric": BASE_NUMERIC,
        "categorical": BASE_CATEGORICAL,
        "engineered": False,
    },
    "v1": {
        "label": "v0 + living_alone",
        "numeric": BASE_NUMERIC + ["living_alone"],
        "categorical": BASE_CATEGORICAL,
        "engineered": False,
    },
    "v2": {
        "label": "v1 + active + current_drinker",
        "numeric": BASE_NUMERIC + ["living_alone", "active", "current_drinker"],
        "categorical": BASE_CATEGORICAL,
        "engineered": False,
    },
    "v3": {
        "label": "v2 + age/sleep/BMI categories",
        "numeric": BASE_NUMERIC + ["living_alone", "active", "current_drinker"],
        "categorical": BASE_CATEGORICAL + ["age_band_10yr", "sleep_cat", "bmi_cat"],
        "engineered": True,
    },
    "v5": {
        "label": "v4 tuned + self-rated health",
        "numeric": BASE_NUMERIC + ["living_alone", "active", "current_drinker", "srh"],
        "categorical": BASE_CATEGORICAL + ["age_band_10yr", "sleep_cat", "bmi_cat"],
        "engineered": True,
    },
    "v6": {
        "label": "v5 + stress",
        "numeric": BASE_NUMERIC + ["living_alone", "active", "current_drinker", "srh", "stress"],
        "categorical": BASE_CATEGORICAL + ["age_band_10yr", "sleep_cat", "bmi_cat"],
        "engineered": True,
    },
}

ALL_COLUMNS = sorted(
    set(
        [
            "dep",
            "W",
            *BASE_NUMERIC,
            *BASE_CATEGORICAL,
            "living_alone",
            "active",
            "current_drinker",
            "srh",
            "stress",
        ]
    )
)


@dataclass
class EvalResult:
    version: str
    label: str
    algorithm: str
    weighted_holdout_auroc: float
    n_train: int
    n_test: int
    n_features_after_preprocess: int
    fit_seconds: float
    params: dict[str, Any]
    notes: str = ""


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True, dtype=np.float32)


def add_engineered_features(x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    out["age_band_10yr"] = pd.cut(
        out["age"],
        bins=[-np.inf, 29, 39, 49, 59, 69, 79, np.inf],
        labels=["<30", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"],
    ).astype("object")
    out["sleep_cat"] = pd.cut(
        out["sleep_hrs"],
        bins=[-np.inf, 5.999999, 8.000001, np.inf],
        labels=["short_<6", "normal_6_8", "long_>8"],
    ).astype("object")
    out["bmi_cat"] = pd.cut(
        out["bmi"],
        bins=[-np.inf, 18.5, 25.0, 30.0, np.inf],
        right=False,
        labels=["underweight_<18.5", "normal_18.5_24.9", "overweight_25_29.9", "obese_30+"],
    ).astype("object")
    return out


def select_features(df: pd.DataFrame, version: str) -> pd.DataFrame:
    spec = FEATURE_SETS[version]
    cols = list(spec["numeric"]) + [c for c in spec["categorical"] if c in df.columns]
    x = df[[c for c in cols if c in df.columns]].copy()
    if spec["engineered"]:
        x = add_engineered_features(x)
    return x[list(spec["numeric"]) + list(spec["categorical"])]


def build_preprocessor(version: str) -> ColumnTransformer:
    spec = FEATURE_SETS[version]
    numeric = list(spec["numeric"])
    categorical = list(spec["categorical"])
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
            ("numeric", numeric_pipe, numeric),
            ("categorical", categorical_pipe, categorical),
        ],
        sparse_threshold=1.0,
    )


def ensure_csr(x: Any) -> Any:
    if sparse.issparse(x):
        return x.tocsr()
    return np.asarray(x, dtype=np.float32)


def scaled_weights(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    return w / np.mean(w)


def lgbm_default_params() -> dict[str, Any]:
    return {
        "objective": "binary",
        "boosting_type": "gbdt",
        "n_estimators": 700,
        "learning_rate": 0.04,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 200,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
    }


def lgbm_model(params: dict[str, Any]) -> LGBMClassifier:
    params = dict(params)
    params.setdefault("random_state", RANDOM_STATE)
    params.setdefault("n_jobs", -1)
    params.setdefault("verbose", -1)
    return LGBMClassifier(**params)


def fit_score_lgbm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    version: str,
    params: dict[str, Any],
    train_weight_mode: str = "none",
    notes: str = "",
) -> EvalResult:
    x_train = select_features(train_df, version)
    y_train = train_df["dep"].astype(int).to_numpy()
    w_train = scaled_weights(train_df["W"].to_numpy())
    x_test = select_features(test_df, version)
    y_test = test_df["dep"].astype(int).to_numpy()
    w_test = test_df["W"].to_numpy()

    preprocessor = build_preprocessor(version)
    t0 = time.perf_counter()
    x_train_t = ensure_csr(preprocessor.fit_transform(x_train))
    x_test_t = ensure_csr(preprocessor.transform(x_test))
    model = lgbm_model(params)
    fit_kwargs: dict[str, Any] = {}
    if train_weight_mode == "survey":
        fit_kwargs["sample_weight"] = w_train
    model.fit(x_train_t, y_train, **fit_kwargs)
    pred = model.predict_proba(x_test_t)[:, 1]
    auc = roc_auc_score(y_test, pred, sample_weight=w_test)
    fit_seconds = time.perf_counter() - t0
    result = EvalResult(
        version=version,
        label=FEATURE_SETS[version]["label"],
        algorithm="LightGBM",
        weighted_holdout_auroc=float(auc),
        n_train=int(len(train_df)),
        n_test=int(len(test_df)),
        n_features_after_preprocess=int(x_train_t.shape[1]),
        fit_seconds=float(fit_seconds),
        params={**params, "train_weight_mode": train_weight_mode},
        notes=notes,
    )
    del x_train, x_test, x_train_t, x_test_t, model, preprocessor, pred
    gc.collect()
    return result


def sample_training_for_tuning(train_df: pd.DataFrame, tune_n: int) -> pd.DataFrame:
    if tune_n <= 0 or tune_n >= len(train_df):
        return train_df.copy()
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=tune_n, random_state=RANDOM_STATE)
    y = train_df["dep"].astype(int).to_numpy()
    idx, _ = next(splitter.split(np.zeros(len(y)), y))
    return train_df.iloc[idx].copy()


def candidate_params(rng: np.random.Generator, n_iter: int) -> list[dict[str, Any]]:
    grid = {
        "learning_rate": [0.015, 0.025, 0.04, 0.06, 0.08],
        "n_estimators": [700, 1000, 1400, 1800, 2400],
        "num_leaves": [15, 31, 63, 127],
        "max_depth": [-1, 4, 6, 8, 10],
        "min_child_samples": [50, 100, 200, 500, 1000],
        "subsample": [0.65, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.65, 0.8, 0.9, 1.0],
        "reg_alpha": [0.0, 0.05, 0.2, 1.0, 3.0],
        "reg_lambda": [0.0, 0.5, 1.0, 3.0, 8.0, 15.0],
        "min_split_gain": [0.0, 0.01, 0.05],
        "class_weight": [None, "balanced"],
        "train_weight_mode": ["none", "survey"],
    }
    out: list[dict[str, Any]] = []
    default = lgbm_default_params()
    default["train_weight_mode"] = "none"
    out.append(default)
    for _ in range(max(0, n_iter - 1)):
        item = lgbm_default_params()
        for key, values in grid.items():
            item[key] = values[int(rng.integers(0, len(values)))]
        # Keep very large leaf counts paired with enough child samples.
        if item["num_leaves"] >= 127 and item["min_child_samples"] < 200:
            item["min_child_samples"] = 200
        out.append(item)
    return out


def tune_lgbm(
    train_df: pd.DataFrame,
    version: str,
    tune_n: int,
    n_iter: int,
    cv_splits: int,
    cv_repeats: int,
) -> dict[str, Any]:
    tune_df = sample_training_for_tuning(train_df, tune_n)
    y_all = tune_df["dep"].astype(int).to_numpy()
    w_all_raw = tune_df["W"].to_numpy()
    rng = np.random.default_rng(RANDOM_STATE)
    candidates = candidate_params(rng, n_iter)
    cv = RepeatedStratifiedKFold(
        n_splits=cv_splits,
        n_repeats=cv_repeats,
        random_state=RANDOM_STATE,
    )

    print(
        f"tuning version={version} rows={len(tune_df)} events={int(y_all.sum())} "
        f"candidates={len(candidates)} folds={cv_splits * cv_repeats}",
        flush=True,
    )
    history: list[dict[str, Any]] = []
    best_score = -math.inf
    best_params: dict[str, Any] | None = None

    for c_idx, candidate in enumerate(candidates, start=1):
        train_weight_mode = candidate.pop("train_weight_mode")
        fold_scores: list[float] = []
        fold_best_iterations: list[int] = []
        t0 = time.perf_counter()
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(np.zeros(len(y_all)), y_all), start=1):
            fold_train = tune_df.iloc[train_idx]
            fold_val = tune_df.iloc[val_idx]
            x_train = select_features(fold_train, version)
            x_val = select_features(fold_val, version)
            y_train = y_all[train_idx]
            y_val = y_all[val_idx]
            w_train = scaled_weights(w_all_raw[train_idx])
            w_val = scaled_weights(w_all_raw[val_idx])

            preprocessor = build_preprocessor(version)
            x_train_t = ensure_csr(preprocessor.fit_transform(x_train))
            x_val_t = ensure_csr(preprocessor.transform(x_val))
            model = lgbm_model(candidate)
            fit_kwargs: dict[str, Any] = {
                "eval_set": [(x_val_t, y_val)],
                "eval_metric": "auc",
                "eval_sample_weight": [w_val],
                "callbacks": [early_stopping(stopping_rounds=75, verbose=False), log_evaluation(period=0)],
            }
            if train_weight_mode == "survey":
                fit_kwargs["sample_weight"] = w_train
            model.fit(x_train_t, y_train, **fit_kwargs)
            pred = model.predict_proba(x_val_t)[:, 1]
            fold_scores.append(float(roc_auc_score(y_val, pred, sample_weight=w_all_raw[val_idx])))
            fold_best_iterations.append(int(getattr(model, "best_iteration_", 0) or candidate["n_estimators"]))
            del x_train, x_val, x_train_t, x_val_t, model, preprocessor, pred
            gc.collect()

        elapsed = time.perf_counter() - t0
        mean_score = float(np.mean(fold_scores))
        sd_score = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0
        final_n_estimators = max(50, int(np.ceil(np.median(fold_best_iterations))))
        record_params = dict(candidate)
        record_params["train_weight_mode"] = train_weight_mode
        record = {
            "candidate": c_idx,
            "mean_weighted_cv_auc": mean_score,
            "sd_weighted_cv_auc": sd_score,
            "fold_scores": fold_scores,
            "fold_best_iterations": fold_best_iterations,
            "elapsed_seconds": elapsed,
            "params": record_params,
            "final_n_estimators": final_n_estimators,
        }
        history.append(record)
        print(
            f"candidate {c_idx:02d}/{len(candidates)} auc={mean_score:.6f} "
            f"sd={sd_score:.6f} best_iter={final_n_estimators} "
            f"weight_mode={train_weight_mode} seconds={elapsed:.1f}",
            flush=True,
        )
        if mean_score > best_score:
            best_score = mean_score
            best_params = dict(candidate)
            best_params["n_estimators"] = final_n_estimators
            best_params["train_weight_mode"] = train_weight_mode

    if best_params is None:
        raise RuntimeError("No LightGBM tuning candidates were evaluated.")

    tune_summary = {
        "version": version,
        "tune_rows": int(len(tune_df)),
        "tune_events": int(y_all.sum()),
        "cv_splits": cv_splits,
        "cv_repeats": cv_repeats,
        "n_candidates": len(candidates),
        "best_weighted_cv_auc": best_score,
        "best_params": best_params,
        "history": history,
    }
    return tune_summary


def fit_score_logistic(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    version: str,
    max_train_rows: int,
) -> EvalResult:
    if 0 < max_train_rows < len(train_df):
        fit_df = sample_training_for_tuning(train_df, max_train_rows)
        notes = f"fit on stratified training subset n={len(fit_df)} for runtime"
    else:
        fit_df = train_df
        notes = "fit on full training split"
    x_train = select_features(fit_df, version)
    y_train = fit_df["dep"].astype(int).to_numpy()
    w_train = scaled_weights(fit_df["W"].to_numpy())
    x_test = select_features(test_df, version)
    y_test = test_df["dep"].astype(int).to_numpy()
    w_test = test_df["W"].to_numpy()
    preprocessor = build_preprocessor(version)
    params = {
        "penalty": "elasticnet",
        "solver": "saga",
        "l1_ratio": 0.2,
        "C": 0.5,
        "class_weight": None,
        "max_iter": 250,
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
        "tol": 1e-3,
    }
    t0 = time.perf_counter()
    x_train_t = ensure_csr(preprocessor.fit_transform(x_train))
    x_test_t = ensure_csr(preprocessor.transform(x_test))
    model = LogisticRegression(**params)
    model.fit(x_train_t, y_train, sample_weight=w_train)
    pred = model.predict_proba(x_test_t)[:, 1]
    auc = roc_auc_score(y_test, pred, sample_weight=w_test)
    fit_seconds = time.perf_counter() - t0
    result = EvalResult(
        version=version,
        label=FEATURE_SETS[version]["label"],
        algorithm="Penalized logistic regression",
        weighted_holdout_auroc=float(auc),
        n_train=int(len(fit_df)),
        n_test=int(len(test_df)),
        n_features_after_preprocess=int(x_train_t.shape[1]),
        fit_seconds=float(fit_seconds),
        params=params,
        notes=notes,
    )
    del x_train, x_test, x_train_t, x_test_t, model, preprocessor, pred
    gc.collect()
    return result


def fit_score_random_forest(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    version: str,
    max_train_rows: int,
) -> EvalResult:
    if 0 < max_train_rows < len(train_df):
        fit_df = sample_training_for_tuning(train_df, max_train_rows)
        notes = f"fit on stratified training subset n={len(fit_df)} for runtime"
    else:
        fit_df = train_df
        notes = "fit on full training split"
    x_train = select_features(fit_df, version)
    y_train = fit_df["dep"].astype(int).to_numpy()
    w_train = scaled_weights(fit_df["W"].to_numpy())
    x_test = select_features(test_df, version)
    y_test = test_df["dep"].astype(int).to_numpy()
    w_test = test_df["W"].to_numpy()
    preprocessor = build_preprocessor(version)
    params = {
        "n_estimators": 500,
        "max_depth": 16,
        "min_samples_leaf": 100,
        "max_features": "sqrt",
        "bootstrap": True,
        "class_weight": None,
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    }
    t0 = time.perf_counter()
    x_train_t = ensure_csr(preprocessor.fit_transform(x_train))
    x_test_t = ensure_csr(preprocessor.transform(x_test))
    model = RandomForestClassifier(**params)
    model.fit(x_train_t, y_train, sample_weight=w_train)
    pred = model.predict_proba(x_test_t)[:, 1]
    auc = roc_auc_score(y_test, pred, sample_weight=w_test)
    fit_seconds = time.perf_counter() - t0
    result = EvalResult(
        version=version,
        label=FEATURE_SETS[version]["label"],
        algorithm="Random forest",
        weighted_holdout_auroc=float(auc),
        n_train=int(len(fit_df)),
        n_test=int(len(test_df)),
        n_features_after_preprocess=int(x_train_t.shape[1]),
        fit_seconds=float(fit_seconds),
        params=params,
        notes=notes,
    )
    del x_train, x_test, x_train_t, x_test_t, model, preprocessor, pred
    gc.collect()
    return result


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH, columns=ALL_COLUMNS)
    df = df[df["dep"].notna() & df["W"].notna()].copy()
    df["dep"] = df["dep"].astype(int)
    return df


def save_results(payload: dict[str, Any]) -> None:
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune-n", type=int, default=400_000)
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--cv-repeats", type=int, default=1)
    parser.add_argument("--run-secondary-algorithms", action="store_true")
    parser.add_argument("--secondary-train-n", type=int, default=500_000)
    args = parser.parse_args()

    df = load_data()
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["dep"],
        random_state=RANDOM_STATE,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print(
        f"loaded rows={len(df)} train={len(train_df)} test={len(test_df)} "
        f"train_events={int(train_df['dep'].sum())} test_events={int(test_df['dep'].sum())}",
        flush=True,
    )

    payload: dict[str, Any] = {
        "data_path": str(DATA_PATH),
        "split": {
            "test_size": 0.2,
            "stratify": "dep",
            "random_state": RANDOM_STATE,
            "n_total": int(len(df)),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "train_events": int(train_df["dep"].sum()),
            "test_events": int(test_df["dep"].sum()),
            "weighted_dep_rate_train": float((train_df["dep"] * train_df["W"]).sum() / train_df["W"].sum()),
            "weighted_dep_rate_test": float((test_df["dep"] * test_df["W"]).sum() / test_df["W"].sum()),
        },
        "results": [],
        "tuning": None,
    }

    default_params = lgbm_default_params()
    for version in ["v0", "v1", "v2", "v3"]:
        print(f"fitting default LightGBM {version}: {FEATURE_SETS[version]['label']}", flush=True)
        result = fit_score_lgbm(train_df, test_df, version, default_params, train_weight_mode="none")
        payload["results"].append(asdict(result))
        save_results(payload)
        print(f"{version} LightGBM weighted_holdout_auroc={result.weighted_holdout_auroc:.6f}", flush=True)

    tune_summary = tune_lgbm(
        train_df=train_df,
        version="v3",
        tune_n=args.tune_n,
        n_iter=args.n_iter,
        cv_splits=args.cv_splits,
        cv_repeats=args.cv_repeats,
    )
    payload["tuning"] = tune_summary
    save_results(payload)
    best_params = dict(tune_summary["best_params"])
    best_weight_mode = best_params.pop("train_weight_mode")
    print("fitting tuned LightGBM v4 on full training split", flush=True)
    v4_result = fit_score_lgbm(
        train_df,
        test_df,
        "v3",
        best_params,
        train_weight_mode=best_weight_mode,
        notes="v4: v3 feature set with LightGBM parameters selected by training-only stratified CV",
    )
    v4_result.version = "v4"
    v4_result.label = "v3 + LightGBM CV hyperparameter tuning"
    payload["results"].append(asdict(v4_result))
    save_results(payload)
    print(f"v4 LightGBM weighted_holdout_auroc={v4_result.weighted_holdout_auroc:.6f}", flush=True)

    for version in ["v5", "v6"]:
        print(f"fitting tuned LightGBM {version}: {FEATURE_SETS[version]['label']}", flush=True)
        result = fit_score_lgbm(train_df, test_df, version, best_params, train_weight_mode=best_weight_mode)
        payload["results"].append(asdict(result))
        save_results(payload)
        print(f"{version} LightGBM weighted_holdout_auroc={result.weighted_holdout_auroc:.6f}", flush=True)

    if args.run_secondary_algorithms:
        for algorithm in ["logistic", "rf"]:
            for version in ["v4", "v5", "v6"]:
                actual_version = "v3" if version == "v4" else version
                print(f"fitting {algorithm} {version}", flush=True)
                if algorithm == "logistic":
                    result = fit_score_logistic(train_df, test_df, actual_version, args.secondary_train_n)
                else:
                    result = fit_score_random_forest(train_df, test_df, actual_version, args.secondary_train_n)
                result.version = version
                if version == "v4":
                    result.label = "v3 feature set, alternative algorithm"
                payload["results"].append(asdict(result))
                save_results(payload)
                print(
                    f"{version} {result.algorithm} weighted_holdout_auroc="
                    f"{result.weighted_holdout_auroc:.6f}",
                    flush=True,
                )

    save_results(payload)
    print(f"wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
