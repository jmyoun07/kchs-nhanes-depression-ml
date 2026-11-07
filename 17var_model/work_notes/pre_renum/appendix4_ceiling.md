### Transportability-ceiling and calibration-interval analyses

Supports Results Testing Whether External Discrimination Can Be Raised Above ~0.77 (concept-shift ceiling) and Discussion Strengths and Limitations (development-survey coverage). Tables S11 to S15 concern the ceiling; Table S16, placed last so that the appendix tables run in numerical order, gives the calibration intervals that support External Validation Across Country and Pandemic Period. Selection of the primary model, and of every strategy not explicitly marked as externally screened, used KCHS-internal criteria only. NHANES labels are used throughout for evaluation, as in any external validation; what is quarantined to the oracle rows is the use of those labels for model *fitting* (a deliberate departure from the external-validation rule set out under External Validation and Sensitivity Analyses, for diagnosis only). Weighted AUROC throughout.

## Table S11. Attempts to exceed the external ceiling.

Except where a row is marked as externally screened, every learner and adaptation was trained on KCHS and applied to NHANES without refitting (unless noted). None improved both external cohorts once selection was restricted to KCHS-internal information. The baseline row is the exploration run in which these strategies were executed, not the primary model: its external values (0.772 pre-pandemic, 0.769 post-pandemic) therefore differ from the primary estimates of 0.765 and 0.771 reported in Table 2, and the comparisons in this table are internally consistent within the run rather than comparable in level with the main results. The same caution applies to Tables S12 and S13, which are anchored on a separate in-run baseline (0.7694 pre-pandemic, 0.7685 post-pandemic).

| Strategy | External pre | External post | Verdict |
|---|---|---|---|
| **LightGBM baseline (17 predictors, exploration run)** | **0.772** | **0.769** | reference |
| Add public-assistance receipt (welfare), externally screened | 0.772 | 0.769 | no change |
| Add welfare + 12-mo welfare, externally screened | 0.772 | 0.769 | no change |
| Add welfare + weekend sleep, externally screened | 0.771 | 0.769 | no change |
| Increase KCHS training 0.5M → 1.0M | 0.768 | 0.768 | lower |
| Increase KCHS training 0.5M → 2.0M | 0.767 | 0.767 | lower |
| 5-seed probability ensemble | 0.769 | 0.768 | ≈ |
| Logistic-regression transfer | 0.759 | 0.760 | lower |
| Elastic-net transfer | 0.759 | 0.760 | lower |
| LightGBM + logistic blend (w=0.5) | 0.768 | 0.771 | offsetting |
| Covariate-shift density-ratio reweighting | 0.766 | 0.758 | lower |
| Within-survey quantile normalization | 0.762 | 0.773 | offsetting |

Two adaptations moved pre and post in opposite directions (blend, quantile normalization); neither yields a net gain selectable without external labels.
LightGBM: light gradient-boosting machine.
KCHS: Korea Community Health Survey.

## Table S12. Pre-specified method search (independent re-run, 300k KCHS training).

An additional eight leakage-free strategies were run with an in-run same-sample baseline (pre 0.7694, post 0.7685). Values in this table and in Table S13 are given to four decimals, the precision recorded in the exploration run log, avoiding a second rounding. None improved both external cohorts. Notably, a leave-one-year-out (LOYO) transfer-robustness search re-selected the primary hyperparameters, and a transport-stability feature search retained all predictors, independent confirmations that the deployed model is already well specified. Maximizing internal cross-validation (the internal-CV-tuned configuration) raised pre but lowered post, illustrating that internal optimization does not transfer consistently.

| Method | External pre | External post | Verdict |
|---|---|---|---|
| Baseline (300k) | 0.7694 | 0.7685 | reference |
| Monotonic constraints (clinical direction) | 0.7610 | 0.7461 | lower |
| Transport-stable feature selection (PSI; internal-selected k=0) | 0.7694 | 0.7685 | no change |
| LOYO transfer-robust hyperparameters (re-selected baseline) | 0.7694 | 0.7685 | no change |
| KCHS per-year model ensemble | 0.7652 | 0.7658 | lower |
| Out-of-fold weight-of-evidence target encoding | 0.7686 | 0.7695 | no change |
| Missing-indicator augmentation | 0.7675 | 0.7685 | no change |
| Explicit imputation (no tree-native missing) | 0.7693 | 0.7684 | no change |
| Internal-CV-tuned configuration | 0.7719 | 0.7674 | offsetting (pre↑ post↓) |

KCHS: Korea Community Health Survey.
## Table S13. Deep tabular learners.

Four neural architectures were trained on KCHS and transferred without refitting, with model/architecture selection by KCHS-internal validation only. The gradient-boosting reference row is the 300,000-record baseline that anchors Table S12, and the caution stated under Table S11 applies to the values in this table. All fell below the gradient-boosting reference internally (0.8370) and none exceeded it on both external cohorts, consistent with the model-agnostic in-domain ceiling and with the established advantage of gradient boosting on modest-signal tabular data.

| Deep model | Internal | External pre | External post | Verdict |
|---|---|---|---|---|
| **Gradient boosting (exploration-run reference)** | 0.8370 | 0.7694 | 0.7685 | reference |
| Multilayer perceptron (internal-selected) | 0.8330 | 0.7677 | 0.7725 | not better |
| TabNet | 0.8290 | 0.7554 | 0.7656 | lower |
| Deep MLP (BatchNorm+Dropout, PyTorch) | 0.8350 | 0.7647 | 0.7756 | offsetting |
| FT-Transformer | 0.8360 | 0.7628 | 0.7722 | offsetting |

## Table S14. NHANES in-domain oracle by predictor set.

Fitted and cross-validated within NHANES on its own labels (5-fold, pooled pre+post) using a fixed penalized-logistic and gradient-boosting pair so that predictor sets are compared on a common footing; these values are not the learner maximum, which is reported separately in Table S15 (0.776 for the 17-predictor set against 0.765 here). Both are empirical benchmarks over the learners run, not proofs of a maximum.

| Predictor set | Pooled oracle AUROC | Δ vs 17 |
|---|---|---|
| 17 harmonized predictors | 0.765 |  |
| 17 + multimorbidity count | 0.774 | +0.009 |
| 17 + healthcare utilization (visits, hospitalization) | 0.771 | +0.006 |
| 17 + multimorbidity + utilization | **0.778** | **+0.013** |

The transported LightGBM model (pre 0.765, post 0.771) is at the level of the 17-predictor oracle, indicating the external number is not obviously below what these predictors support within NHANES itself.
AUROC: area under the receiver operating characteristic curve.

## Table S15. Oracle by learner, with confidence intervals.

Pooled pre+post NHANES, fitted and cross-validated on its own labels with folds split by primary sampling unit (PSU) within each cycle so that no cluster appears in both training and validation; 95% CIs from the rescaled stratified cluster bootstrap (1000 replicates, PSUs resampled within period-specific design strata). The final row refits the oracle inside each replicate so that model-fitting variability is included. An in-sample (apparent) bound is not reported, given that it is strongly learner-dependent and carries no information about the ceiling. Table S14 above uses a fixed penalized-logistic/gradient-boosting pair to compare *predictor sets* on a common footing; it is not the learner maximum reported here, and the two should not be read as one sequence.

| Learner | Oracle AUROC (5-fold CV) | 95% CI |
|---|---|---|
| Logistic regression | 0.764 | 0.739-0.788 |
| Penalized logistic (L2, C=0.1) | 0.764 | 0.740-0.788 |
| Histogram gradient boosting | 0.772 | 0.746-0.798 |
| LightGBM (small) | 0.776 | 0.753-0.799 |
| LightGBM (small), refitted inside each bootstrap replicate | 0.776 | 0.745-0.794 |







AUROC: area under the receiver operating characteristic curve.
LightGBM: light gradient-boosting machine.
### Interpretation
- The 17-predictor external ceiling (~0.77) held against every leakage-controlled attempt made, and the transported model already matches the best in-domain performance among the learners tested. Concept shift is the most consistent explanation, but it cannot be separated here from differences in how several predictors are measured, from development-survey coverage, or from complete-case selection; the single density-ratio correction applied shows only that the procedure did not help.
- Multimorbidity and healthcare utilization carry additional in-domain signal in NHANES (oracle 0.765→0.778). Table S14 is fitted and cross-validated entirely within pooled NHANES; this is therefore incremental discrimination inside that population rather than evidence that the signal would transport. The healthcare-visit item is carried as pre-cycle only, and its contribution cannot be separated from cycle differences. These domains are not collected by KCHS (only hypertension and diabetes are recorded as physician-diagnosed conditions across all survey years, both already included); hence those domains cannot enter a KCHS-developed transportable model. Part of the residual ceiling is therefore development-survey coverage, not concept shift alone.

*Multimorbidity count comprised physician-diagnosed asthma, arthritis, thyroid disorder, chronic obstructive pulmonary disease, liver disease, and cancer (circulatory conditions excluded per the pre-specified predictor policy; mental-health service contact excluded as tautological). Healthcare-visit count (HUQ051) is unavailable in the NHANES August 2021-2023 file and was carried as pre-cycle only.*

## Table S16. Calibration with cluster-bootstrap intervals (external cohorts).

Intervals are the 2.5th and 97.5th percentiles of 1000 rescaled stratified cluster-bootstrap replicates, the same replicates used for discrimination. Recalibration is intercept/slope updating applied out-of-sample by 5-fold cross-fitting grouped by primary sampling unit. Two recalibrated rows are given per cohort. The **nested** row refits the update inside every replicate; its interval therefore carries the uncertainty of the update itself and answers whether the procedure restores calibration when the whole analysis is repeated. The other row holds the update fixed at the one estimated on the full cohort, which is the position of a deployment that fits its update once; that interval is wider, since no refitting absorbs the replicate's own miscalibration. Slopes are given to three decimals here. Out-of-fold recalibration shrinks the slope marginally below 1 by construction, since each fold's map is estimated without the observations it is applied to; the nested pre-pandemic interval accordingly sits just under 1 (upper bound 0.998) rather than centered on it, and this reflects the cross-fitting rather than residual miscalibration. Expected calibration error (ECE) is given as a point estimate only: it is a non-negative absolute-deviation statistic that this replicate scheme biases upward, and its percentile interval failed to contain its own point estimate in one cell.

| Cohort | Probabilities | Calibration slope (95% CI) | CITL (95% CI) | ECE |
|---|---|---|---|---|
| NHANES pre | as transported | 0.651 (0.533-0.774) | +0.51 (+0.37 to +0.64) | 0.030 |
| NHANES pre | after intercept/slope recalibration | 0.998 (0.817-1.188) | 0.00 (-0.12 to +0.11) | 0.010 |
| NHANES pre | after intercept/slope recalibration (nested) | 0.998 (0.950-0.998) | 0.00 (-0.01 to +0.02) | 0.010 |
| NHANES post | as transported | 0.681 (0.603-0.764) | +1.13 (+0.98 to +1.26) | 0.068 |
| NHANES post | after intercept/slope recalibration | 0.996 (0.878-1.126) | 0.00 (-0.13 to +0.12) | 0.014 |
| NHANES post | after intercept/slope recalibration (nested) | 0.996 (0.965-1.005) | 0.00 (-0.02 to +0.01) | 0.014 |

External AUROC differed between cycles by +0.006 (95% CI -0.045 to +0.056; bootstrap P=.86); the interval was formed by differencing independently drawn replicate statistics from the two cohorts, the cycles being disjoint samples.
CITL: calibration-in-the-large.
ECE: expected calibration error.
NHANES: National Health and Nutrition Examination Survey.

### Discussion detail relocated from the manuscript (ceiling, oracle, and calibration)

Explanatory prose condensed out of the Discussion is preserved here in full. No new result is introduced.

The between-cycle difference interval was computed after the fact, no between-cycle test having been pre-specified, and its width leaves moderate change in either direction admissible; within that limit the data are consistent with no decay across the population-wide shock that motivated the analysis. The plateau is most consistent with shift in how the same predictors relate to the outcome, with residual contributions from differing measurement, development-survey coverage and complete-case selection.

Both prior Korean models identified perceived stress and subjective health as dominant predictors, and neither attempted external validation. The contribution of the present study over that work includes an explicit transportability ceiling characterized by an in-domain oracle and a catalogued set of over two thousand leakage-controlled runs.

The in-domain oracle is a deliberate departure from the external-validation rule, made for diagnosis only. That comparison is pooled across cycles while the transported estimates are cycle-specific, and the intervals overlap; it therefore locates the limit in the predictors rather than in the transfer only to the extent that the oracle is a fair upper bound.

Two specification analyses locate where the shortfall sits. Each analysis alters the target condition or the population as well as the gap; hence the pattern is suggestive rather than confirmatory.

The interpretability comparison is a statement about the fixed model's attribution and not about the external predictor-outcome associations, which the attribution never observes. Two caveats bound that conclusion. Several predictors are measured differently in the two surveys, and the external cohorts are complete-case samples with differential depression-module non-response; hence measurement and selection effects cannot be separated from concept shift with these data.

Local recalibration is required outside the development population since discrimination transported but absolute risk did not.

Further methodological strengths condensed out of the Discussion: validation covered two target periods as well as a second country, and recalibration was quantified rather than asserted; intervals came from a rescaled stratified cluster bootstrap validated against a closed-form design-based standard error; the predictor set was restricted to items both surveys measure in a form harmonizable to a common definition, with every inclusion decision documented and dated; and the three candidate algorithms were compared at each one's tuned best under one identical protocol, after a preceding gradient-boosting-only search that is reported in full.

### Results detail relocated from the manuscript (the third recalibration step)

Explanatory prose condensed out of the Results is preserved here in full. No new result is introduced.

Temperature scaling, the third of the three pre-specified recalibration steps applied out-of-sample, left less residual miscalibration than intercept-only updating: the calibration slope was 0.83 pre-pandemic and 1.17 post-pandemic, the post-pandemic value overshooting 1 rather than lying between the intercept-only (0.68) and intercept-plus-slope (1.00) results, with calibration-in-the-large +0.02 and -0.02. Intercept-plus-slope updating remains the recommended deployment update.

### Discussion detail relocated in revision (optimization search and predictor coverage)

Explanatory prose condensed out of the Discussion during revision is preserved here. No new result is introduced.

The catalogued search also covered alternative loss functions. Leakage is the failure mode that has driven reproducibility problems across machine-learning-based science; runs screened on external labels are identified as such in Table S11. That increasing the training set from 0.5 to 2 million records slightly lowered external discrimination is a pattern consistent with source-specific overfitting rather than with an estimation-limited ceiling.

The interpretability comparison is at least consistent with the shift-stability framing of deployment failure in health artificial intelligence.

The coverage analysis extracted candidate domains available in NHANES but not harmonizable to KCHS: physician-diagnosed multimorbidity (asthma, arthritis, thyroid, chronic obstructive pulmonary disease, liver disease, and cancer, excluding circulatory conditions) and healthcare utilization. KCHS records only hypertension and diabetes among physician-diagnosed conditions across more than one survey year, both already included; dyslipidemia, osteoporosis, cataract and arthritis are fielded in a single year and therefore cannot enter a model pooled over 2017-2025. KCHS collects no broader chronic-disease inventory, no hospitalization record and no visit counts, and these domains therefore cannot enter a KCHS-developed transportable model. Part of the residual external gap is thus attributable to development-survey coverage rather than to concept shift alone, and a development survey that captured multimorbidity and utilization might raise the attainable ceiling.
