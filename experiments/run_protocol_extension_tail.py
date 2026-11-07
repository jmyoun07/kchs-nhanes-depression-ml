from __future__ import annotations

import json
import time
from dataclasses import asdict

import pandas as pd
from sklearn.model_selection import train_test_split

from run_external_auroc_methods import (
    FULL14,
    OUT,
    SEED,
    TARGET,
    WEIGHT,
    distribution_shift_scores,
    fit_eval_pipeline,
    make_lgbm_pipeline,
)
from run_protocol_extension import (
    FAST_LGBM,
    cap_stratified,
    clean_existing_results,
    eval_support_trim,
    fit_eval_gray_covariate_shift,
    load_data,
)


def log(message: str) -> None:
    print(message, flush=True)


def run() -> None:
    t0 = time.time()
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
    x_train, y_train, _ = cap_stratified(x_train_full, y_train_full, w_train_full)
    log(f"Tail run loaded train cap={len(x_train)}, holdout={len(x_hold)}")

    result, baseline_preds, _ = fit_eval_pipeline(
        "sub_lgbm_full14_baseline_500k_refit_for_tail",
        "baseline_lgbm",
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
        notes="Refit only to recover baseline predictions for support-trim/gray tail evaluations.",
    )

    tail_results = []
    trim = eval_support_trim(baseline_preds, result, x_train, x_hold, y_hold, w_hold, pre, post)
    tail_results.append(trim)
    log(f"{trim.method}: pre={trim.nhanes_pre_auroc:.4f} post={trim.nhanes_post_auroc:.4f} [{trim.compliance}]")

    combined_target = pd.concat([pre, post], axis=0, ignore_index=True)
    shift = distribution_shift_scores(x_train, combined_target, FULL14)
    shift.to_csv(OUT / "supplement_gray_shift_scores.csv", index=False)
    drop_top = shift.head(6)["feature"].tolist()
    stable_by_shift = [f for f in FULL14 if f not in drop_top]
    gray_drop, _, _ = fit_eval_pipeline(
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
    tail_results.append(gray_drop)
    log(
        f"{gray_drop.method}: internal={gray_drop.internal_auroc:.4f} "
        f"pre={gray_drop.nhanes_pre_auroc:.4f} post={gray_drop.nhanes_post_auroc:.4f} "
        f"[{gray_drop.compliance}]"
    )

    for label, target_df in [("pre", pre), ("post", post), ("combined", combined_target)]:
        gray, _ = fit_eval_gray_covariate_shift(
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
        tail_results.append(gray)
        log(
            f"{gray.method}: internal={gray.internal_auroc:.4f} "
            f"pre={gray.nhanes_pre_auroc:.4f} post={gray.nhanes_post_auroc:.4f} "
            f"[{gray.compliance}]"
        )

    tail = pd.DataFrame([asdict(r) for r in tail_results])
    tail["external_mean_auroc"] = tail[["nhanes_pre_auroc", "nhanes_post_auroc"]].mean(axis=1)
    tail.sort_values("external_mean_auroc", ascending=False).to_csv(OUT / "method_results_tail.csv", index=False)

    partial_path = OUT / "method_results_supplement_partial.csv"
    frames = []
    if partial_path.exists():
        frames.append(pd.read_csv(partial_path))
    frames.append(tail)
    supplement = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    supplement["external_mean_auroc"] = supplement[["nhanes_pre_auroc", "nhanes_post_auroc"]].mean(axis=1)
    supplement = supplement.drop_duplicates("method", keep="last")
    supplement.sort_values("external_mean_auroc", ascending=False).to_csv(
        OUT / "method_results_supplement.csv", index=False
    )

    combined_frames = []
    old_path = OUT / "method_results_partial.csv"
    if old_path.exists():
        combined_frames.append(clean_existing_results(pd.read_csv(old_path)))
    combined_frames.append(supplement)
    combined = pd.concat(combined_frames, axis=0, ignore_index=True, sort=False)
    combined["external_mean_auroc"] = combined[["nhanes_pre_auroc", "nhanes_post_auroc"]].mean(axis=1)
    combined = combined.drop_duplicates("method", keep="last")
    combined.sort_values("external_mean_auroc", ascending=False).to_csv(
        OUT / "method_results_combined.csv", index=False
    )

    metadata = {
        "seed": SEED,
        "tail_elapsed_seconds": round(time.time() - t0, 3),
        "notes": "Tail run completes support-trim and gray-area experiments after the main supplement produced KCHS-only partial results.",
    }
    (OUT / "protocol_extension_tail_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log("Finished tail run.")


if __name__ == "__main__":
    run()
