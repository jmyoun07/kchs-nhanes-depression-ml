from __future__ import annotations

import json
import time
import warnings
from dataclasses import asdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from run_external_auroc_methods import (
    CAT,
    FULL14,
    LGBM_ROBUST,
    OUT,
    PROC,
    SEED,
    TARGET,
    WEIGHT,
    MethodResult,
    distribution_shift_scores,
    eval_predictions,
    fit_eval_pipeline,
    make_lgbm_pipeline,
    make_preprocessor,
)


warnings.filterwarnings("ignore")

MAX_TRAIN = 500_000

FAST_LGBM = {
    **LGBM_ROBUST,
    "n_estimators": 520,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 250,
    "reg_lambda": 3.0,
    "n_jobs": 2,
}

DART_LGBM = {
    **FAST_LGBM,
    "boosting_type": "dart",
    "n_estimators": 360,
    "learning_rate": 0.035,
    "drop_rate": 0.08,
    "skip_drop": 0.55,
}

SHALLOW_LGBM = {
    **FAST_LGBM,
    "n_estimators": 650,
    "learning_rate": 0.025,
    "num_leaves": 7,
    "min_child_samples": 900,
    "reg_lambda": 12.0,
    "colsample_bytree": 0.9,
}


def log(msg: str) -> None:
    print(msg, flush=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_parquet(PROC / "kchs_dev.parquet"),
        pd.read_parquet(PROC / "nhanes_pre.parquet"),
        pd.read_parquet(PROC / "nhanes_post.parquet"),
    )


def cap_stratified(
    x: pd.DataFrame,
    y: np.ndarray,
    w: np.ndarray | None = None,
    max_n: int = MAX_TRAIN,
    seed: int = SEED,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
    y = np.asarray(y, dtype=int)
    if len(x) <= max_n:
        return x, y, None if w is None else np.asarray(w, dtype=float)
    idx = np.arange(len(x))
    take, _ = train_test_split(idx, train_size=max_n, stratify=y, random_state=seed)
    take = np.sort(take)
    return (
        x.iloc[take].copy(),
        y[take],
        None if w is None else np.asarray(w, dtype=float)[take],
    )


def normalized(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    return w / np.nanmean(w)


def group_balance_weights(x: pd.DataFrame) -> np.ndarray:
    age_band = pd.cut(
        x["age"],
        bins=[18, 29, 39, 49, 59, 69, 79, 120],
        include_lowest=True,
        labels=False,
    )
    srh = pd.to_numeric(x["srh"], errors="coerce").round().fillna(-1).astype(int)
    group = x["sex"].astype(str) + "_" + age_band.astype(str) + "_" + srh.astype(str)
    counts = group.value_counts()
    w = group.map(lambda g: len(x) / (len(counts) * counts[g])).to_numpy(float)
    return normalized(np.clip(w, 0.2, 5.0))


def phq_confidence_weights(x: pd.DataFrame) -> np.ndarray:
    # Downweight cases close to the PHQ-9 >= 10 threshold using KCHS labels only.
    distance = np.abs(pd.to_numeric(x["phq_sum"], errors="coerce").to_numpy(float) - 9.5)
    w = 0.35 + 0.65 * np.minimum(distance / 5.5, 1.0)
    return normalized(w)


def encoded_univariate_signal(df: pd.DataFrame, feature: str) -> pd.Series:
    if feature == "sex":
        return (df[feature] == "F").astype(float)
    if feature == "edu":
        return df[feature].map({"elem_low": 0, "middle": 1, "high": 2, "college": 3}).astype(float)
    if feature == "smoking":
        return df[feature].map({"never": 0, "former": 1, "current": 2}).astype(float)
    return pd.to_numeric(df[feature], errors="coerce")


def kchs_temporal_stable_features(x: pd.DataFrame, y: np.ndarray) -> tuple[list[str], pd.DataFrame]:
    rows = []
    y = np.asarray(y, dtype=int)
    for feature in FULL14:
        signs = []
        effects = []
        values = encoded_univariate_signal(x, feature)
        for year in sorted(x["year"].dropna().unique()):
            ix = (x["year"].to_numpy() == year) & values.notna().to_numpy()
            if ix.sum() < 500 or len(np.unique(y[ix])) < 2:
                continue
            v = values.to_numpy(dtype=float)[ix]
            yy = y[ix]
            denom = np.nanstd(v)
            if not np.isfinite(denom) or denom == 0:
                continue
            effect = (np.nanmean(v[yy == 1]) - np.nanmean(v[yy == 0])) / denom
            if np.isfinite(effect):
                effects.append(effect)
                signs.append(np.sign(effect))
        nonzero = [s for s in signs if s != 0]
        if nonzero:
            pos = np.mean(np.asarray(nonzero) > 0)
            sign_consistency = max(pos, 1 - pos)
        else:
            sign_consistency = 0.0
        mean_abs_effect = float(np.mean(np.abs(effects))) if effects else 0.0
        rows.append(
            {
                "feature": feature,
                "years": len(effects),
                "sign_consistency": sign_consistency,
                "mean_abs_std_effect": mean_abs_effect,
                "kept": bool(sign_consistency >= 0.78 and mean_abs_effect >= 0.015),
            }
        )
    table = pd.DataFrame(rows).sort_values(
        ["kept", "sign_consistency", "mean_abs_std_effect"], ascending=[True, True, True]
    )
    kept = table.loc[table["kept"], "feature"].tolist()
    if len(kept) < 5:
        kept = ["age", "sex", "edu", "income_q", "srh", "sleep_hrs", "smoking"]
    return kept, table


def fit_eval_isotonic(
    xtr: pd.DataFrame,
    ytr: np.ndarray,
    xho: pd.DataFrame,
    yho: np.ndarray,
    who: np.ndarray,
    pre: pd.DataFrame,
    post: pd.DataFrame,
) -> tuple[MethodResult, dict[str, np.ndarray]]:
    fit_x, cal_x, fit_y, cal_y, _, cal_w = train_test_split(
        xtr,
        ytr,
        np.ones(len(xtr), dtype=float),
        test_size=0.2,
        stratify=ytr,
        random_state=SEED + 111,
    )
    t0 = time.time()
    pipe = make_lgbm_pipeline(FULL14, FAST_LGBM)
    pipe.fit(fit_x[FULL14], fit_y)
    p_cal = pipe.predict_proba(cal_x[FULL14])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_cal, cal_y, sample_weight=cal_w)
    seconds = time.time() - t0

    preds = {
        "internal": iso.transform(pipe.predict_proba(xho[FULL14])[:, 1]),
        "pre": iso.transform(pipe.predict_proba(pre[FULL14])[:, 1]),
        "post": iso.transform(pipe.predict_proba(post[FULL14])[:, 1]),
    }
    result = eval_predictions(
        "lgbm_kchs_isotonic_calibrated",
        "calibrated_boosting",
        "COMPLIANT",
        "none",
        preds["internal"],
        preds["pre"],
        preds["post"],
        yho,
        who,
        pre,
        post,
        seconds,
        len(xtr),
        "LightGBM fitted on KCHS and isotonic calibration fitted on a KCHS-only calibration split.",
    )
    return result, preds


def logistic_domain_ratio_weights(
    xtr: pd.DataFrame,
    target_df: pd.DataFrame,
    sample_n_source: int = 80_000,
    clip: tuple[float, float] = (0.15, 8.0),
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(SEED)
    source_ix = rng.choice(np.arange(len(xtr)), size=min(sample_n_source, len(xtr)), replace=False)
    source_sample = xtr.iloc[source_ix][FULL14].copy()
    target_sample = target_df[FULL14].copy()
    domain_x = pd.concat([source_sample, target_sample], axis=0, ignore_index=True)
    domain_y = np.r_[np.zeros(len(source_sample), dtype=int), np.ones(len(target_sample), dtype=int)]
    pipe = Pipeline(
        [
            ("pre", make_preprocessor(FULL14, scale_numeric=True)),
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    C=0.7,
                    solver="lbfgs",
                    max_iter=500,
                    n_jobs=1,
                    random_state=SEED,
                ),
            ),
        ]
    )
    pipe.fit(domain_x, domain_y)
    domain_auc = float(roc_auc_score(domain_y, pipe.predict_proba(domain_x)[:, 1]))
    pt = np.clip(pipe.predict_proba(xtr[FULL14])[:, 1], 1e-4, 1 - 1e-4)
    prior_odds = len(source_sample) / len(target_sample)
    ratio = np.clip((pt / (1 - pt)) * prior_odds, clip[0], clip[1])
    return normalized(ratio), domain_auc


def fit_eval_gray_covariate_shift(
    label: str,
    target_df: pd.DataFrame,
    xtr: pd.DataFrame,
    ytr: np.ndarray,
    xho: pd.DataFrame,
    yho: np.ndarray,
    who: np.ndarray,
    pre: pd.DataFrame,
    post: pd.DataFrame,
) -> tuple[MethodResult, dict[str, np.ndarray]]:
    ratio, domain_auc = logistic_domain_ratio_weights(xtr, target_df)
    return fit_eval_pipeline(
        f"gray_logistic_covariate_shift_to_{label}",
        "unlabeled_domain_adaptation",
        "GRAY_UNLABELED_NHANES_FEATURES",
        f"NHANES {label} predictors used for density-ratio weights; NHANES outcomes not used for fitting",
        make_lgbm_pipeline(FULL14, FAST_LGBM),
        FULL14,
        xtr,
        ytr,
        xho,
        yho,
        who,
        pre,
        post,
        sample_weight=ratio,
        notes=f"Logistic domain classifier density-ratio weights, clipped 0.15-8.0; domain AUC={domain_auc:.3f}.",
    )[:2]


def support_mask(source: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    mask = np.ones(len(target), dtype=bool)
    # Restrict only on continuous/ordinal covariates. Applying central quantiles
    # to sparse binary indicators can silently define impossible support.
    for feature in ["age", "sleep_hrs", "bmi", "income_q", "srh"]:
        lo, hi = source[feature].quantile([0.001, 0.999])
        vals = pd.to_numeric(target[feature], errors="coerce")
        mask &= vals.between(lo, hi).fillna(False).to_numpy()
    for feature in CAT:
        allowed = set(source[feature].dropna().unique())
        mask &= target[feature].isin(allowed).to_numpy()
    return mask


def eval_support_trim(
    baseline_preds: dict[str, np.ndarray],
    baseline_result: MethodResult,
    source: pd.DataFrame,
    xho: pd.DataFrame,
    yho: np.ndarray,
    who: np.ndarray,
    pre: pd.DataFrame,
    post: pd.DataFrame,
) -> MethodResult:
    pre_mask = support_mask(source, pre)
    post_mask = support_mask(source, post)
    return MethodResult(
        method="estimand_shift_trim_nhanes_to_kchs_support",
        family="support_trim",
        compliance="ESTIMAND_DEVIATION",
        nhanes_touch="NHANES predictors used only to restrict validation to KCHS 0.5-99.5% covariate support",
        train_n=baseline_result.train_n,
        internal_auroc=baseline_result.internal_auroc,
        nhanes_pre_auroc=float(
            roc_auc_score(pre.loc[pre_mask, TARGET].astype(int), baseline_preds["pre"][pre_mask], sample_weight=pre.loc[pre_mask, WEIGHT])
        ),
        nhanes_post_auroc=float(
            roc_auc_score(post.loc[post_mask, TARGET].astype(int), baseline_preds["post"][post_mask], sample_weight=post.loc[post_mask, WEIGHT])
        ),
        seconds=0.0,
        notes=f"Frozen baseline model; pre retained {pre_mask.sum()}/{len(pre)}, post retained {post_mask.sum()}/{len(post)}. This changes the external-validation estimand.",
    )


def clean_existing_results(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["compliance"] = out["compliance"].astype(str)
    out.loc[out["compliance"].str.contains("compliant", case=False, na=False), "compliance"] = "COMPLIANT"
    out.loc[out["compliance"].str.contains("gray-area", case=False, na=False), "compliance"] = "GRAY_UNLABELED_NHANES_FEATURES"
    return out


def run() -> None:
    t_all = time.time()
    kchs, pre, post = load_data()
    y = kchs[TARGET].astype(int).to_numpy()
    x_train_full, x_hold, y_train_full, y_hold, w_train_full, w_hold = train_test_split(
        kchs,
        y,
        kchs[WEIGHT].to_numpy(float),
        test_size=0.2,
        stratify=y,
        random_state=SEED,
    )
    x_train, y_train, w_train = cap_stratified(x_train_full, y_train_full, w_train_full)
    assert w_train is not None
    log(
        f"Loaded kchs={kchs.shape}, NHANES pre={pre.shape}, post={post.shape}; "
        f"training cap={len(x_train)}, fixed holdout={len(x_hold)}"
    )

    results: list[MethodResult] = []
    pred_bank: dict[str, dict[str, np.ndarray]] = {}

    def add(result: MethodResult, preds: dict[str, np.ndarray] | None = None) -> None:
        results.append(result)
        if preds is not None:
            pred_bank[result.method] = preds
        log(
            f"{result.method}: internal={result.internal_auroc:.4f} "
            f"pre={result.nhanes_pre_auroc:.4f} post={result.nhanes_post_auroc:.4f} "
            f"[{result.compliance}]"
        )
        pd.DataFrame([asdict(r) for r in results]).to_csv(OUT / "method_results_supplement_partial.csv", index=False)

    specs = [
        (
            "sub_lgbm_full14_baseline_500k",
            "baseline_lgbm",
            x_train,
            y_train,
            None,
            FULL14,
            FAST_LGBM,
            "500k stratified KCHS training cap; same fixed KCHS holdout.",
        ),
        (
            "sub_lgbm_dart_regularized",
            "robust_lgbm",
            x_train,
            y_train,
            None,
            FULL14,
            DART_LGBM,
            "DART/dropout boosting trained only in KCHS.",
        ),
        (
            "sub_lgbm_extra_shallow",
            "robust_lgbm",
            x_train,
            y_train,
            None,
            FULL14,
            SHALLOW_LGBM,
            "Very shallow, high-min-child LightGBM to dampen country-specific interactions.",
        ),
    ]

    for method, family, xt, yt, sw, features, params, notes in specs:
        result, preds, _ = fit_eval_pipeline(
            method,
            family,
            "COMPLIANT",
            "none",
            make_lgbm_pipeline(features, params),
            features,
            xt,
            yt,
            x_hold,
            y_hold,
            w_hold,
            pre,
            post,
            sample_weight=sw,
            notes=notes,
        )
        add(result, preds)

    for label, year_mask, notes in [
        ("kchs_2017_2019_only", x_train_full["year"].isin([2017, 2018, 2019]).to_numpy(), "Train only on pre-pandemic KCHS years."),
        ("kchs_2021_2025_only", x_train_full["year"].isin([2021, 2022, 2023, 2024, 2025]).to_numpy(), "Train only on recent/pandemic-era KCHS years."),
    ]:
        xt, yt, _ = cap_stratified(x_train_full.loc[year_mask].copy(), y_train_full[year_mask], None)
        result, preds, _ = fit_eval_pipeline(
            f"sub_lgbm_{label}",
            "calendar_restricted_training",
            "COMPLIANT",
            "none",
            make_lgbm_pipeline(FULL14, FAST_LGBM),
            FULL14,
            xt,
            yt,
            x_hold,
            y_hold,
            w_hold,
            pre,
            post,
            notes=notes,
        )
        add(result, preds)

    clear_mask = ~x_train["phq_sum"].between(8, 11, inclusive="both").to_numpy()
    xt_clear, yt_clear, _ = cap_stratified(x_train.loc[clear_mask].copy(), y_train[clear_mask], None)
    result, preds, _ = fit_eval_pipeline(
        "sub_lgbm_drop_phq8_11_borderline",
        "label_noise_robust_training",
        "COMPLIANT",
        "none",
        make_lgbm_pipeline(FULL14, FAST_LGBM),
        FULL14,
        xt_clear,
        yt_clear,
        x_hold,
        y_hold,
        w_hold,
        pre,
        post,
        notes="Excluded KCHS training cases with PHQ-9 sums 8-11; evaluation still uses full KCHS holdout and NHANES.",
    )
    add(result, preds)

    for method, sw, notes in [
        ("sub_lgbm_phq_margin_weighted", phq_confidence_weights(x_train), "KCHS PHQ-margin weights downweight near-threshold labels."),
        ("sub_lgbm_age_sex_srh_group_balanced", group_balance_weights(x_train), "KCHS age-sex-self-rated-health strata balanced in training."),
        ("sub_lgbm_survey_weighted_500k", normalized(w_train), "KCHS survey weights used in model fitting; AUROC evaluated with fixed survey weights."),
    ]:
        result, preds, _ = fit_eval_pipeline(
            method,
            "weighted_training",
            "COMPLIANT",
            "none",
            make_lgbm_pipeline(FULL14, FAST_LGBM),
            FULL14,
            x_train,
            y_train,
            x_hold,
            y_hold,
            w_hold,
            pre,
            post,
            sample_weight=sw,
            notes=notes,
        )
        add(result, preds)

    stable_features, stability = kchs_temporal_stable_features(x_train, y_train)
    stability.to_csv(OUT / "kchs_temporal_feature_stability.csv", index=False)
    result, preds, _ = fit_eval_pipeline(
        "sub_lgbm_kchs_temporal_stable_features",
        "kchs_stability_feature_selection",
        "COMPLIANT",
        "none",
        make_lgbm_pipeline(stable_features, FAST_LGBM),
        stable_features,
        x_train,
        y_train,
        x_hold,
        y_hold,
        w_hold,
        pre,
        post,
        notes=f"KCHS-only year-to-year univariate sign-stability filter kept: {', '.join(stable_features)}.",
    )
    add(result, preds)

    result, preds = fit_eval_isotonic(x_train, y_train, x_hold, y_hold, w_hold, pre, post)
    add(result, preds)

    ensemble_members = [
        "sub_lgbm_full14_baseline_500k",
        "sub_lgbm_dart_regularized",
        "sub_lgbm_extra_shallow",
        "sub_lgbm_phq_margin_weighted",
        "sub_lgbm_kchs_temporal_stable_features",
    ]
    ensemble_preds = {
        key: np.mean([pred_bank[m][key] for m in ensemble_members], axis=0) for key in ["internal", "pre", "post"]
    }
    add(
        eval_predictions(
            "sub_ensemble_equal_weight_kchs_only",
            "prediction_ensemble",
            "COMPLIANT",
            "none",
            ensemble_preds["internal"],
            ensemble_preds["pre"],
            ensemble_preds["post"],
            y_hold,
            w_hold,
            pre,
            post,
            0.0,
            len(x_train),
            "Equal-weight average of five pre-specified KCHS-only models; weights not chosen with NHANES data.",
        ),
        ensemble_preds,
    )

    trim_result = eval_support_trim(
        pred_bank["sub_lgbm_full14_baseline_500k"],
        results[0],
        x_train,
        x_hold,
        y_hold,
        w_hold,
        pre,
        post,
    )
    add(trim_result, None)

    combined_target = pd.concat([pre, post], axis=0, ignore_index=True)
    shift = distribution_shift_scores(x_train, combined_target, FULL14)
    shift.to_csv(OUT / "supplement_gray_shift_scores.csv", index=False)
    drop_top = shift.head(6)["feature"].tolist()
    stable_by_shift = [f for f in FULL14 if f not in drop_top]
    result, preds, _ = fit_eval_pipeline(
        "gray_shift_drop_top6_features",
        "unlabeled_domain_adaptation",
        "GRAY_UNLABELED_NHANES_FEATURES",
        "NHANES predictors used to rank source-target distribution shift; outcomes not used for fitting",
        make_lgbm_pipeline(stable_by_shift, FAST_LGBM),
        stable_by_shift,
        x_train,
        y_train,
        x_hold,
        y_hold,
        w_hold,
        pre,
        post,
        notes=f"Dropped six most shifted predictors by KS/TV: {', '.join(drop_top)}.",
    )
    add(result, preds)

    for label, target_df in [("pre", pre), ("post", post), ("combined", combined_target)]:
        result, preds = fit_eval_gray_covariate_shift(
            label,
            target_df,
            x_train,
            y_train,
            x_hold,
            y_hold,
            w_hold,
            pre,
            post,
        )
        add(result, preds)

    supplement = pd.DataFrame([asdict(r) for r in results])
    supplement["external_mean_auroc"] = supplement[["nhanes_pre_auroc", "nhanes_post_auroc"]].mean(axis=1)
    supplement.sort_values("external_mean_auroc", ascending=False).to_csv(
        OUT / "method_results_supplement.csv", index=False
    )

    frames = []
    old_path = OUT / "method_results_partial.csv"
    if old_path.exists():
        frames.append(clean_existing_results(pd.read_csv(old_path)))
    frames.append(supplement)
    combined = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    combined["external_mean_auroc"] = combined[["nhanes_pre_auroc", "nhanes_post_auroc"]].mean(axis=1)
    combined.sort_values("external_mean_auroc", ascending=False).to_csv(
        OUT / "method_results_combined.csv", index=False
    )

    metadata = {
        "seed": SEED,
        "max_train": MAX_TRAIN,
        "train_cap_n": int(len(x_train)),
        "holdout_n": int(len(x_hold)),
        "nhanes_pre_n": int(len(pre)),
        "nhanes_post_n": int(len(post)),
        "elapsed_seconds": round(time.time() - t_all, 3),
        "notes": [
            "All fitting code and outputs are under experiments/.",
            "COMPLIANT methods fit predictor effects using KCHS labels only.",
            "GRAY_UNLABELED_NHANES_FEATURES methods use NHANES predictors but no NHANES outcomes for fitting.",
            "ESTIMAND_DEVIATION applies the frozen KCHS model to a restricted NHANES support subset.",
        ],
    }
    (OUT / "protocol_extension_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log("Finished. Wrote experiments/method_results_supplement.csv and experiments/method_results_combined.csv")


if __name__ == "__main__":
    run()
