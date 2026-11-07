## Part B. Clinical utility, sample characteristics, and sensitivity analyses

Estimates are complex-survey weighted unless a row or column is labeled unweighted; the depression-module non-response counts and the algorithm-selection cross-validated AUROCs are unweighted. Supports Results Participant Characteristics (sample characteristics in Table S4 and depression-module non-response in Table S8), Parsimonious Deployment Model (Table S6), Sensitivity Analyses (Table S5), and Interpretability, Fairness, and Clinical Utility (subgroup discrimination and calibration in Table S7; operating characteristics and net benefit in Tables S1 to S3). The model is positioned as a screening/triage aid, not a diagnostic instrument.

## Table S1. Operating characteristics at reference thresholds.

The Youden and prevalence points are computed over all observed scores rather than a coarse threshold grid, and both are selected and evaluated in the same cohort and are therefore apparent rather than out-of-sample. The development-fixed rows use the single threshold set once in the development data at 85% sensitivity and applied unchanged to every other set; those rows are therefore transported operating characteristics. PPV denotes positive predictive value and NPV negative predictive value.

| Set | Threshold rule | Probability cutoff | Sensitivity | Specificity | PPV | NPV |
|---|---|---|---|---|---|---|
| Internal | Youden (exact) | 0.0339 | 0.735 | 0.785 | 0.112 | 0.988 |
| Internal | prevalence | 0.1685 | 0.305 | 0.974 | 0.305 | 0.975 |
| NHANES pre | Youden (exact) | 0.0406 | 0.701 | 0.721 | 0.188 | 0.963 |
| NHANES pre | prevalence | 0.1888 | 0.309 | 0.936 | 0.309 | 0.936 |
| NHANES post | Youden (exact) | 0.0258 | 0.759 | 0.651 | 0.232 | 0.951 |
| NHANES post | prevalence | 0.1258 | 0.384 | 0.914 | 0.384 | 0.914 |
| Development set | development-fixed (sens 0.85) | 0.0205 | 0.850 | 0.664 | 0.085 | 0.992 |
| Internal | development-fixed (sens 0.85) | 0.0205 | 0.833 | 0.665 | 0.083 | 0.991 |
| NHANES pre | development-fixed (sens 0.85) | 0.0205 | 0.810 | 0.566 | 0.147 | 0.970 |
| NHANES post | development-fixed (sens 0.85) | 0.0205 | 0.791 | 0.596 | 0.214 | 0.953 |

PPV: positive predictive value; NPV: negative predictive value; NHANES: National Health and Nutrition Examination Survey.
NPV: negative predictive value.
NHANES: National Health and Nutrition Examination Survey.

## Table S2. Triage-oriented high-sensitivity operating points.

At sensitivity fixed to 0.85 and 0.90, the table gives specificity, positive and negative predictive value, the share of the population flagged, and the number needed to screen (NNS = 1/PPV). Thresholds are chosen and evaluated in the same cohort; hence these are apparent operating characteristics. The 6-item deployment model and the 7-item post hoc variant are both shown alongside the full 17-predictor model.

| Model | Set | Target | Specificity | PPV | NPV | Flag rate | NNS |
|---|---|---|---|---|---|---|---|
| 17-predictor | Internal | sens 0.85 | 0.625 | 0.077 | 0.991 | 0.392 | 13.0 |
| 17-predictor | Internal | sens 0.90 | 0.525 | 0.065 | 0.993 | 0.490 | 15.4 |
| 17-predictor | NHANES pre | sens 0.85 | 0.476 | 0.130 | 0.972 | 0.552 | 7.7 |
| 17-predictor | NHANES pre | sens 0.90 | 0.395 | 0.121 | 0.978 | 0.631 | 8.2 |
| 17-predictor | NHANES post | sens 0.85 | 0.490 | 0.188 | 0.959 | 0.551 | 5.3 |
| 17-predictor | NHANES post | sens 0.90 | 0.394 | 0.171 | 0.966 | 0.642 | 5.8 |
| 6-item | Internal | sens 0.85 | 0.603 | 0.073 | 0.991 | 0.413 | 13.7 |
| 6-item | Internal | sens 0.90 | 0.486 | 0.060 | 0.993 | 0.528 | 16.6 |
| 6-item | NHANES pre | sens 0.85 | 0.456 | 0.126 | 0.971 | 0.570 | 7.9 |
| 6-item | NHANES pre | sens 0.90 | 0.368 | 0.116 | 0.976 | 0.655 | 8.6 |
| 6-item | NHANES post | sens 0.85 | 0.448 | 0.176 | 0.956 | 0.588 | 5.7 |
| 6-item | NHANES post | sens 0.90 | 0.377 | 0.167 | 0.965 | 0.657 | 6.0 |
| 7-item | Internal | sens 0.85 | 0.615 | 0.075 | 0.991 | 0.401 | 13.3 |
| 7-item | Internal | sens 0.90 | 0.514 | 0.064 | 0.993 | 0.501 | 15.7 |
| 7-item | NHANES pre | sens 0.85 | 0.473 | 0.130 | 0.972 | 0.554 | 7.7 |
| 7-item | NHANES pre | sens 0.90 | 0.391 | 0.120 | 0.977 | 0.634 | 8.3 |
| 7-item | NHANES post | sens 0.85 | 0.448 | 0.176 | 0.956 | 0.589 | 5.7 |
| 7-item | NHANES post | sens 0.90 | 0.341 | 0.160 | 0.961 | 0.689 | 6.3 |

PPV: positive predictive value; NPV: negative predictive value; NNS: number needed to screen; NHANES: National Health and Nutrition Examination Survey.
NPV: negative predictive value.
NNS: number needed to screen.
NHANES: National Health and Nutrition Examination Survey.

## Table S3. Decision-curve net benefit.

Net benefit at threshold probabilities of 0.10 and 0.15. The two external cohorts use intercept/slope-recalibrated probabilities, with recalibration applied out-of-sample by 5-fold cross-fitting grouped by primary sampling unit; the internal hold-out uses the original probabilities, being the deployed model in its own data and already well calibrated there (calibration slope 1.03, CITL +0.10), so that no update is called for. Two default strategies are shown: screening everyone (screen-all), whose net benefit is negative at these thresholds in every cohort except NHANES post at 0.10, and screening no one (net benefit 0 by definition). A model is clinically useful when it exceeds both; the vs-better-default column gives the margin over whichever default is better.

| Model | Set | Threshold | Model net benefit | Screen-all NB | vs screen-all | vs better default | Flag rate |
|---|---|---|---|---|---|---|---|
| 17-predictor | Internal | 0.10 | +0.0099 | -0.0718 | +0.0817 | +0.0099 | 0.073 |
| 17-predictor | Internal | 0.15 | +0.0066 | -0.1348 | +0.1414 | +0.0066 | 0.042 |
| 17-predictor | NHANES pre | 0.10 | +0.0284 | -0.0172 | +0.0456 | +0.0284 | 0.268 |
| 17-predictor | NHANES pre | 0.15 | +0.0198 | -0.0771 | +0.0969 | +0.0198 | 0.164 |
| 17-predictor | NHANES post | 0.10 | +0.0580 | +0.0244 | +0.0335 | +0.0335 | 0.397 |
| 17-predictor | NHANES post | 0.15 | +0.0401 | -0.0329 | +0.0731 | +0.0401 | 0.260 |
| 6-item | Internal | 0.10 | +0.0087 | -0.0718 | +0.0805 | +0.0087 | 0.071 |
| 6-item | Internal | 0.15 | +0.0058 | -0.1348 | +0.1406 | +0.0058 | 0.039 |
| 6-item | NHANES pre | 0.10 | +0.0270 | -0.0172 | +0.0442 | +0.0270 | 0.269 |
| 6-item | NHANES pre | 0.15 | +0.0149 | -0.0771 | +0.0920 | +0.0149 | 0.152 |
| 6-item | NHANES post | 0.10 | +0.0553 | +0.0244 | +0.0308 | +0.0308 | 0.391 |
| 6-item | NHANES post | 0.15 | +0.0410 | -0.0329 | +0.0740 | +0.0410 | 0.257 |
| 7-item | Internal | 0.10 | +0.0091 | -0.0718 | +0.0809 | +0.0091 | 0.072 |
| 7-item | Internal | 0.15 | +0.0061 | -0.1348 | +0.1409 | +0.0061 | 0.041 |
| 7-item | NHANES pre | 0.10 | +0.0280 | -0.0172 | +0.0452 | +0.0280 | 0.265 |
| 7-item | NHANES pre | 0.15 | +0.0187 | -0.0771 | +0.0958 | +0.0187 | 0.160 |
| 7-item | NHANES post | 0.10 | +0.0544 | +0.0244 | +0.0299 | +0.0299 | 0.405 |
| 7-item | NHANES post | 0.15 | +0.0380 | -0.0329 | +0.0710 | +0.0380 | 0.265 |

NHANES: National Health and Nutrition Examination Survey.

## Interpretation

Tuned to a triage-appropriate 85% sensitivity, the full model flags roughly half of the external population (pre 55%, post 55%) at a PPV of 0.130 (pre) and 0.188 (post), ie, a number needed to screen of 7.7 and 5.3. The 6-item deployment model is close behind (PPV 0.126 and 0.176; NNS 7.9 and 5.7), as is the 7-item variant (NNS 7.7 and 5.7). For the two external cohorts after local recalibration, and for the internal hold-out on its original probabilities, net benefit is positive and exceeds both default strategies at threshold probabilities of 0.10-0.15 in every dataset and for every model size, supporting use as a low-burden digital triage aid rather than a diagnostic test.

## Table S4. Weighted sample characteristics (exact values for Figure 1).

Unweighted N; all other rows weighted and computed on **available cases**, ie, respondents with the item recorded. The last column gives the share missing in each cohort. Self-rated and oral health are scored 1 (best) to 5 (worst).

| Characteristic | KCHS | NHANES pre | NHANES post | Missing (KCHS / pre / post) |
|---|---|---|---|---|
| N (unweighted) | 2,048,510 | 7,888 | 5,210 | - |
| Depression PHQ-9 ≥10 | 3.6% | 8.4% | 12.2% | 0.0% / 0.0% / 0.0% |
| Female | 50.5% | 51.5% | 50.8% | 0.0% / 0.0% / 0.0% |
| Current smoker | 18.5% | 16.9% | 14.4% | 0.0% / 0.0% / 0.1% |
| Partnered | 62.7% | 62.4% | 60.1% | 0.1% / 0.1% / 0.1% |
| Living alone | 13.9% | not collected | 17.0% | 0.0% / 100.0% / 0.0% |
| Employed | 64.8% | 65.2% | 62.9% | 0.0% / 0.0% / 0.0% |
| Food insecure | 3.7% | 16.4% | 17.2% | 22.1% / 6.0% / 6.3% |
| Current drinker | 70.5% | 77.2% | 78.6% | 0.0% / 0.1% / 0.4% |
| Meets physical-activity recommendation | 34.1% | 34.9% | 46.6% | 11.0% / 0.0% / 0.7% |
| Hypertension | 22.5% | 32.7% | 31.0% | 0.0% / 0.1% / 0.0% |
| Diabetes | 9.7% | 11.5% | 11.2% | 0.0% / 0.0% / 0.0% |
| Education: elementary or less | 11.1% | 3.3% | 2.6% | 0.1% / 0.1% / 0.0% |
| Education: middle school | 8.0% | 7.0% | 5.9% | 0.1% / 0.1% / 0.0% |
| Education: high school | 29.1% | 27.1% | 25.6% | 0.1% / 0.1% / 0.0% |
| Education: college or more | 51.8% | 62.6% | 65.9% | 0.1% / 0.1% / 0.0% |
| Age (years) | 49.43 | 48.37 | 48.78 | 0.0% / 0.0% / 0.0% |
| BMI | 23.68 | 29.80 | 29.71 | 13.1% / 1.2% / 0.9% |
| Sleep hours | 6.57 | 7.55 | 7.67 | 11.3% / 0.8% / 1.0% |
| Self-rated health | 2.65 | 2.60 | 2.53 | 0.0% / 0.1% / 0.0% |
| Oral health | 2.98 | 2.85 | 2.82 | 0.0% / 0.1% / 0.2% |
| Household income quintile | 2.38 | 2.46 | 2.15 | 16.6% / 13.1% / 12.0% |

KCHS: Korea Community Health Survey; NHANES: National Health and Nutrition Examination Survey; PHQ-9: 9-item Patient Health Questionnaire; BMI: body mass index.
NHANES: National Health and Nutrition Examination Survey.
PHQ-9: 9-item Patient Health Questionnaire.
BMI: body mass index.

## Table S5. Sensitivity analyses - weighted AUROC (exact values for Figure 4).

All rows use the corrected primary data and the cluster-level split (Predictors and Harmonization, External Validation and Sensitivity Analyses). Every row also shares the primary model's hyperparameter configuration, so that the specifications are compared at one setting rather than at each variant's own optimum. The paired 17-versus-14-predictor comparison reported in the Results gives each model its own hyperparameter search and therefore returns marginally different values (0.745 pre-pandemic and 0.757 post-pandemic, against 0.745 and 0.758 here); the two answer different questions and should not be differenced.

| Specification | Internal | NHANES pre | NHANES post |
|---|---|---|---|
| Primary (17 predictors, corrected) | 0.836 | 0.765 | 0.771 |
| Food insecurity excluded (16) | 0.834 | 0.759 | 0.764 |
| Three added predictors excluded (14) | 0.827 | 0.745 | 0.758 |
| Weighted training | 0.836 | 0.764 | 0.768 |
| Missing-indicator imputation | 0.837 | 0.764 | 0.772 |
| Single chained-equations imputation | 0.837 | 0.763 | 0.770 |
| Age ≥30 (train and evaluate) | 0.838 | 0.781 | 0.787 |
| Age ≥30 (train only) | 0.836 | 0.770 | 0.769 |
| Outcome PHQ-9 ≥5 | 0.761 | 0.735 | 0.740 |

NHANES: National Health and Nutrition Examination Survey; PHQ-9: 9-item Patient Health Questionnaire.
PHQ-9: 9-item Patient Health Questionnaire.

## Table S6. Parsimonious model trajectory - weighted AUROC by item count.

Predictors entered in descending order of global SHAP importance computed on a 20,000-record subsample of the fitting partition; the item count was chosen on the selection set using the analysis plan's 0.02 margin (**6 items**, bold). Hold-out and NHANES columns are evaluation only.

| Items | Predictor added | Selection-set AUROC | Hold-out AUROC | NHANES pre | NHANES post |
|---|---|---|---|---|---|
| 1 | srh | 0.7412 | 0.7573 | 0.6966 | 0.6815 |
| 2 | age | 0.7745 | 0.7827 | 0.7078 | 0.7252 |
| 3 | sex | 0.7813 | 0.7898 | 0.7119 | 0.7214 |
| 4 | sleep_hrs | 0.8012 | 0.8097 | 0.7239 | 0.7404 |
| 5 | oral_health | 0.8130 | 0.8184 | 0.7372 | 0.7509 |
| **6** | **partnered** | **0.8176** | **0.8220** | **0.7476** | **0.7572** |
| 7 | smoking | 0.8242 | 0.8285 | 0.7570 | 0.7523 |
| 8 | employed | 0.8253 | 0.8311 | 0.7672 | 0.7560 |
| 9 | bmi | 0.8276 | 0.8307 | 0.7608 | 0.7582 |
| 10 | food_insecure | 0.8299 | 0.8335 | 0.7649 | 0.7669 |
| 11 | income_q | 0.8306 | 0.8346 | 0.7655 | 0.7685 |
| 12 | active | 0.8324 | 0.8356 | 0.7650 | 0.7694 |
| 13 | edu | 0.8326 | 0.8360 | 0.7652 | 0.7692 |
| 14 | living_alone | 0.8330 | 0.8363 | 0.7659 | 0.7709 |
| 15 | current_drinker | 0.8330 | 0.8363 | 0.7649 | 0.7689 |
| 16 | htn | 0.8337 | 0.8364 | 0.7654 | 0.7695 |
| 17 | dm | 0.8332 | 0.8366 | 0.7656 | 0.7690 |

AUROC: area under the receiver operating characteristic curve; NHANES: National Health and Nutrition Examination Survey.
NHANES: National Health and Nutrition Examination Survey.

## Table S7. Subgroup discrimination, calibration and error rates.

Weighted AUROC with 95% CIs from the rescaled stratified cluster bootstrap, applied as domain estimates from the full-cohort design. False-positive and false-negative rates are evaluated at the single development-fixed operating threshold. Income quintiles and error rates are included as the analysis plan requires.

| Cohort | Subgroup | n | Weighted AUROC | 95% CI | Calibration slope | CITL | FPR | FNR |
|---|---|---|---|---|---|---|---|---|
| KCHS hold-out | female | 54,584 | 0.826 | 0.814-0.837 | 1.030 | +0.108 | 0.421 | 0.128 |
| KCHS hold-out | male | 45,417 | 0.838 | 0.822-0.854 | 1.032 | +0.092 | 0.250 | 0.237 |
| KCHS hold-out | Age 20-39 | 20,862 | 0.808 | 0.790-0.827 | 1.052 | +0.039 | 0.434 | 0.151 |
| KCHS hold-out | Age 40-59 | 34,770 | 0.822 | 0.801-0.842 | 1.022 | +0.101 | 0.245 | 0.270 |
| KCHS hold-out | Age 60+ | 44,369 | 0.860 | 0.850-0.870 | 1.014 | +0.165 | 0.351 | 0.101 |
| KCHS hold-out | college | 38,778 | 0.811 | 0.793-0.829 | 1.055 | +0.077 | 0.305 | 0.232 |
| KCHS hold-out | high school | 28,658 | 0.839 | 0.821-0.855 | 1.000 | +0.108 | 0.313 | 0.180 |
| KCHS hold-out | middle | 11,013 | 0.882 | 0.861-0.902 | 1.128 | +0.018 | 0.352 | 0.100 |
| KCHS hold-out | <=elementary | 21,488 | 0.819 | 0.804-0.835 | 0.957 | +0.185 | 0.536 | 0.060 |
| KCHS hold-out | Income quintile 1 | 16,821 | 0.824 | 0.809-0.839 | 0.991 | +0.194 | 0.626 | 0.032 |
| KCHS hold-out | Income quintile 2 | 16,147 | 0.835 | 0.817-0.856 | 1.031 | +0.204 | 0.416 | 0.136 |
| KCHS hold-out | Income quintile 3 | 15,817 | 0.835 | 0.813-0.858 | 1.063 | +0.012 | 0.352 | 0.142 |
| KCHS hold-out | Income quintile 4 | 19,405 | 0.799 | 0.772-0.822 | 0.987 | +0.009 | 0.273 | 0.283 |
| KCHS hold-out | Income quintile 5 | 15,495 | 0.781 | 0.744-0.818 | 0.999 | -0.030 | 0.245 | 0.308 |
| NHANES pre | Mexican American | 914 | 0.693 | 0.633-0.763 | 0.584 | -0.095 | 0.677 | 0.104 |
| NHANES post | Mexican American | 330 | 0.692 | 0.533-0.802 | 0.576 | +1.024 | 0.520 | 0.189 |
| NHANES pre | Other Hispanic | 813 | 0.773 | 0.725-0.820 | 0.723 | +0.409 | 0.548 | 0.097 |
| NHANES post | Other Hispanic | 499 | 0.789 | 0.751-0.845 | 0.698 | +0.654 | 0.536 | 0.122 |
| NHANES pre | Non-Hispanic White | 2,798 | 0.769 | 0.715-0.818 | 0.645 | +0.705 | 0.362 | 0.234 |
| NHANES post | Non-Hispanic White | 3,215 | 0.773 | 0.726-0.815 | 0.699 | +1.321 | 0.342 | 0.254 |
| NHANES pre | Non-Hispanic Black | 2,096 | 0.762 | 0.727-0.805 | 0.731 | +0.076 | 0.609 | 0.122 |
| NHANES post | Non-Hispanic Black | 598 | 0.726 | 0.642-0.791 | 0.566 | +0.867 | 0.539 | 0.173 |
| NHANES pre | Non-Hispanic Asian | 886 | 0.807 | 0.716-0.874 | 0.782 | +0.357 | 0.296 | 0.278 |
| NHANES post | Non-Hispanic Asian | 228 | 0.845 | 0.656-0.987 | 1.194 | +0.699 | 0.299 | 0.218 |
| NHANES pre | Other/multiracial | 381 | 0.764 | 0.665-0.865 | 0.694 | +0.804 | 0.586 | 0.092 |
| NHANES post | Other/multiracial | 340 | 0.782 | 0.701-0.853 | 0.790 | +0.995 | 0.601 | 0.051 |

AUROC: area under the receiver operating characteristic curve; CITL: calibration-in-the-large; FPR: false positive rate; FNR: false negative rate; KCHS: Korea Community Health Survey; NHANES: National Health and Nutrition Examination Survey.
CITL: calibration-in-the-large.
FPR: false positive rate.
FNR: false negative rate.
KCHS: Korea Community Health Survey.
NHANES: National Health and Nutrition Examination Survey.

## Table S8. Depression-module (DPQ) non-response in NHANES.

The depression module is administered in the mobile examination center (MEC), and every weighted estimate in this study uses MEC weights; the MEC-examined sample is therefore the estimation frame and is the denominator used here. Rates over all interviewed adults are given alongside for comparison, but that denominator conflates module non-response with never having been examined, which the MEC weights already account for. Non-response is defined here as three or more of the nine PHQ-9 items unanswered.

| Stage | NHANES pre | NHANES post |
|---|---|---|
| Interviewed adults ≥20 y | 9,232 | 7,809 |
| MEC-examined adults ≥20 y (estimation frame) | 8,544 | 6,064 |
| Complete PHQ-9 within the MEC frame (analytic sample) | 7,888 (92.3%) | 5,210 (85.9%) |
| Lost to **any** item non-response within the MEC frame (the complete-case exclusion) | 7.7% | 14.1% |
| ≥3 items missing, MEC frame | 7.4% | 13.3% |
| ≥3 items missing, MEC frame, weighted | 5.6% | 13.2% |
| ≥3 items missing, all interviewed adults | 14.3% | 32.7% |
| Lost to **any** item non-response, all interviewed adults | 14.6% | 33.3% |
| Non-response (MEC frame), less than high school | 12.9% | 26.6% |
| Non-response (MEC frame), high school | 6.9% | 16.1% |
| Non-response (MEC frame), college or more | 5.7% | 9.8% |
| Non-response (MEC frame), non-Hispanic Asian | 14.3% | 31.6% |
| Non-response (MEC frame), non-Hispanic Black | 8.2% | 19.1% |
| Non-response (MEC frame), non-Hispanic White | 4.3% | 9.4% |
| Prevalence, complete case (unweighted) | 9.23% | 12.69% |
| Prevalence including 1-2 missing by person-mean | 9.27% | 12.78% |

Of respondents counted as non-respondents on the interview frame, 52.0% (pre-pandemic) and 68.4% (post-pandemic) were never MEC-examined and therefore carry zero MEC weight, lying outside the frame of every weighted estimate reported in this study. These figures are computed from the NHANES DEMO and DPQ files, and the derivation is released with the analysis code.
NHANES: National Health and Nutrition Examination Survey.
MEC: mobile examination center.
PHQ-9: 9-item Patient Health Questionnaire.

## Table S9. Disposition of the pre-specified selection rules.

The statistical analysis plan fixed the primary-model rule before analysis: highest cross-validated AUROC, ties (ΔAUROC < 0.005) broken by proximity of the calibration slope to 1, then parsimony. LightGBM and random forest fell inside the tie band, to which the tie-break applies. The pipeline as executed ranked the algorithms on cross-validated AUROC and then on the calibration slope without constructing the band; the plan's rule was therefore verified retrospectively, and it reproduces the model the pipeline selected.

| Algorithm | CV-AUROC | Calibration slope | \|slope - 1\| | Inside 0.005 tie band | Selected |
|---|---|---|---|---|---|
| LightGBM | 0.8396 | 1.0033 | 0.0033 | yes | **yes** |
| Random forest | 0.8350 | 1.2034 | 0.2034 | yes |  |
| Elastic net | 0.8220 | 0.9983 | 0.0017 | no |  |

CV-AUROC: cross-validated area under the receiver operating characteristic curve.
AUROC: area under the receiver operating characteristic curve.
LightGBM: light gradient-boosting machine.

Other pre-specified rules were applied as follows: the parsimony margin follows the plan's recommended 0.02, giving a 6-item deployment model; a stricter 0.01 margin giving 7 items is reported as an exploratory variant rather than the deployed model; on external data the seventh item does not help consistently, raising pre-pandemic AUROC from 0.748 to 0.757 while lowering post-pandemic AUROC from 0.757 to 0.752 (Parsimonious Deployment Model). The operating point is reported both at the recommended sensitivity ≥0.85 (Table S2) and at the alternative Youden point (Table S1). Unweighted training with weighted evaluation was used as the primary approach with weighted training as sensitivity, as recommended. The alcohol predictor was specified in the protocol as current drinking **plus at-risk drinking**, but the harmonized variable uses current drinking alone, given that the NHANES quantity item was not carried through harmonization; three alternative alcohol definitions are reported in the Results text and change external discrimination by ≤0.003. The planned self-reported-versus-measured BMI comparison and the prior-depression-diagnosis exclusion could not be run, given that KCHS records only self-reported BMI and no prior diagnosis (Strengths and Limitations). An alternative physical-activity definition was not available in the harmonized data, which carries only the binary aerobic-recommendation indicator, and income exists only as a within-survey quintile. The four interaction terms evaluated in the feature-representation search were specified when that search was written, not in any earlier dated document. The planned R components were not used; all analyses ran in Python (External Validation and Sensitivity Analyses).

Further points where the analysis departed from, or fully met, the plan's recommended defaults are recorded here for completeness; those also stated in the manuscript are repeated so that the disposition of every rule can be read in one place:

| Pre-specified default | What was done | Note |
|---|---|---|
| 80/20 stratified development / hold-out split (D4) | 500,000 training and 100,001 hold-out drawn from the 2,048,510 pooled records, the remainder unused (Study Design and Data Sources) | Training size fixed for tractability; a size experiment showed no external cost (Testing Whether External Discrimination Can Be Raised Above ~0.77) |
| 5-fold cross-validation repeated 5 times (D5) | single-pass 4-fold survey-weighted CV for the hyperparameter and feature-representation search, single-pass 5-fold CV for the algorithm comparison (Model Development and Evaluation) | Repeats were not run. This is a deviation, and the margin between the two leading algorithms (0.0046 AUROC) lies inside the plan's 0.005 tie band; repetition could therefore have changed which algorithm led on AUROC alone, whereas the tie-break on calibration slope, which decides the choice inside that band, separates the two by 0.20 and is not close |
| Class weighting for outcome imbalance, with synthetic minority oversampling technique (SMOTE) as sensitivity (D6) | no imbalance handling; the learner was fitted on the natural 3.6% event rate | Tree-based boosting with a large event count (~18,000) trained stably without reweighting; not tested against the planned alternatives |
| Recalibration in three steps - intercept only, then intercept+slope, then Platt/temperature comparison (Methods) | all three variants run and reported, each cross-fitted by primary sampling unit (External Validation and Sensitivity Analyses) | Plan followed in full; no deviation |
| Fairness flag when a subgroup AUROC differs from the overall by more than 0.05 (D13) | rule applied post hoc in Interpretability, Fairness, and Clinical Utility | the highest income quintile in KCHS (-0.056); Mexican American in NHANES pre-pandemic (-0.072); Mexican American in NHANES post-pandemic (-0.078); non-Hispanic Asian in NHANES post-pandemic (+0.074) |

The analysis was not registered in a public registry, contrary to the plan's intention; the protocol, plan and amendment are dated documents released with the code (Study Design and Data Sources).

## Table S10. Harmonization-correction sensitivity.

Each row applies **one** correction to the analytic data and refits the primary model; all other settings are unchanged. Three corrections are not tabulated separately: living alone fabricated to a constant and post-pandemic physical activity parsed to zero, both NHANES-side, since in the uncorrected data the affected variable is constant rather than merely mismeasured; and the KCHS 77777 income refusal code, which is a missing-value declaration rather than a change of definition. The six rows shown address the questionnaire-driven discontinuities inside KCHS (2020 drinking recode, 2018 activity fallback, weekday-versus-seven-day sleep), the NHANES age top-code, unmasked 2017 income refusal codes, and the income-quintile collapse in the post-pandemic cohort (Predictors and Harmonization, External Validation and Sensitivity Analyses). No correction changes weighted AUROC by more than 0.003 in any cohort.

| Specification | Internal | NHANES pre | NHANES post | Δ internal | Δ pre | Δ post |
|---|---|---|---|---|---|---|
| Baseline (uncorrected data, before the six corrections below) | 0.8315 | 0.7718 | 0.7688 | +0.0000 | +0.0000 | +0.0000 |
| KCHS current drinking: 2020+ code change reconciled | 0.8309 | 0.7710 | 0.7687 | -0.0006 | -0.0008 | -0.0001 |
| KCHS 2018 physical activity set to missing (fallback definition) | 0.8321 | 0.7706 | 0.7692 | +0.0006 | -0.0012 | +0.0003 |
| KCHS sleep aligned to weekday-only, as in NHANES | 0.8310 | 0.7707 | 0.7716 | -0.0005 | -0.0011 | +0.0027 |
| Age top-code aligned (all cohorts capped at 80) | 0.8315 | 0.7710 | 0.7688 | -0.0000 | -0.0008 | -0.0000 |
| KCHS 2017 income refusal codes (77/99) set to missing | 0.8315 | 0.7713 | 0.7688 | -0.0001 | -0.0005 | -0.0001 |
| Income quintile recomputed as rank-based five levels | 0.8317 | 0.7722 | 0.7692 | +0.0002 | +0.0004 | +0.0003 |

NHANES: National Health and Nutrition Examination Survey.
KCHS: Korea Community Health Survey.
### Discussion detail relocated from the manuscript (utility, deployment, and limitations)

Explanatory prose condensed out of the Discussion is preserved here in full. No new result is introduced.

Operating characteristics support the usefulness claim, which rests on a decision-analytic criterion rather than on a nominal AUROC cut-off. At population prevalence the positive predictive value is low, and the appropriate reading of a positive score is a prompt for confirmatory assessment rather than an action in itself.

The mobile-delivery evidence base cited in Principal Findings concerns intervention delivery to defined patient groups rather than the delivery of screening, and no equivalent trial evidence exists for a transported screening algorithm.

The six items of the deployment model are self-rated health, age, sex, sleep duration, self-rated oral health, and partnership. All six require no examination, no laboratory value and no clinician; none is a symptom item, which keeps the triage step distinct from the full PHQ-9 it is meant to prioritize. Item reduction should also be re-evaluated whenever the deployment population differs materially from the development population, given that the ranking of predictors that supports a six-item form is itself population-dependent.

Intercept-plus-slope updating restored calibration in both external cohorts, each using that cohort's full sample, and is the recommended update; the local sample size needed for a stable update was not examined here.

Equity requires explicit monitoring rather than a single fairness statement. Among the subgroup estimates, the highest-income KCHS estimate sits clear of the cohort interval (Table S7).

The limitations set out in Strengths and Limitations are given in descending order of consequence. Depression-module non-response is also the reason the PROBAST analysis domain is rated unclear for the external cohorts (Table S8). The amendment that added three of the seventeen predictors was written while the study was under way; it is dated and summarized in English in Multimedia Appendix 1, Part A, with the unmodified Korean original released alongside the code.

### Results detail relocated from the manuscript (sample composition, module non-response, predictor attribution, subgroup detail, and the six-item operating point)

Explanatory prose condensed out of the Results is preserved here in full. No new result is introduced.

The contrast between the development and external samples in body-mass index, food insecurity and outcome frequency is consistent with genuinely distinct populations and, for food insecurity, with the single-item-versus-module measurement difference set out in Part A.

Denominator choice governs the apparent size of depression-module non-response. Counted on the mobile examination center frame, which is where the module is administered and on which every weighted estimate in this study is computed, the three-or-more-item non-response rates are 7.4% pre-pandemic and 13.3% post-pandemic. Counted over all interviewed adults the same rates are 14.3% and 32.7%; that second denominator conflates module non-response with never having been examined, which the examination-center weights already account for. Non-response was higher among respondents with less than high-school education, at 12.9% pre-pandemic and 26.6% post-pandemic on the examination-center frame.

Respondents missing only one or two depression items can be recovered by filling the missing items with the person-mean. That change moved the observed prevalence by no more than 0.1 percentage points in either cycle (unweighted: 9.23% to 9.27% pre-pandemic, 12.69% to 12.78% post-pandemic), indicating that the complete-case restriction does not itself shift the observed outcome frequency. Item-level counts are in Table S8.

Mean absolute SHAP values for the predictors ranked immediately below self-rated health in the development fitting partition were age 0.37, sex 0.28, sleep duration 0.26, self-rated oral health 0.23, partnership 0.19 and smoking 0.17. Food insecurity ranked 10th in KCHS but 8th in NHANES pre-pandemic; the item is a single question in KCHS against the 10-item module in NHANES.

Analytic subgroup sizes for the external race and ethnicity estimates were, pre-pandemic, Mexican American n=914, non-Hispanic Black n=2,096, other or multiracial n=381, non-Hispanic White n=2,798, other Hispanic n=813 and non-Hispanic Asian n=886; and post-pandemic, Mexican American n=330, non-Hispanic Black n=598, non-Hispanic White n=3,215, other or multiracial n=340, other Hispanic n=499 and non-Hispanic Asian n=228.

Within KCHS a few subgroup intervals sit clear of the cohort interval: higher in adults aged ≥60 years (0.850-0.870) and in middle-school-educated respondents (0.861-0.902); lower in income quintile 4 (0.772-0.822) and in the highest income quintile (0.744-0.818). Subgroup discrimination is therefore not uniform in the development population. That the lowest external point estimate falls in Mexican American respondents in both cycles warrants monitoring as an equity caveat even though the external intervals are mutually overlapping.

Internal subgroup calibration slopes ran 0.96-1.13 across sex, age, education and income strata, with subgroup calibration-in-the-large from -0.03 to +0.20. External subgroup slopes were below 1 in every subgroup except non-Hispanic Asian respondents post-pandemic (0.57-0.79; 1.19 in that subgroup), and external subgroup calibration-in-the-large was positive in every group except Mexican American respondents pre-pandemic, reaching +1.32. Exact values with intervals are in Table S7.

At a within-cohort operating point fixed to 85% sensitivity (apparent values, Table S2), the six-item deployment form required 7.9 pre-pandemic and 5.7 post-pandemic respondents screened per case detected, against 7.7 and 5.3 for the 17-predictor model, and its net benefit exceeded both default strategies in every cohort at threshold probabilities of 0.10 and 0.15 (Table S3).

Replacing current drinking with drinking at least twice weekly, with binge drinking, or with all three indicators together reached 0.769-0.771 pre-pandemic and 0.770 post-pandemic against an in-run baseline of 0.772 and 0.769; that run predates the data-integrity corrections and the cluster-level split, hence its levels are not comparable with the primary estimates or with Table S5. No single tabulated harmonization correction, the age top-code alignment and the rank-based income-quintile rebuild included, changed weighted AUROC by more than 0.003 in any cohort (Table S10).

Under the common configuration of Table S5, in which the 17-, 16- and 14-predictor models share one setting and the 14-predictor value is 0.758 post-pandemic, removing food insecurity alone accounts for 28% of the 17-versus-14 gap pre-pandemic and 50% post-pandemic.

### Discussion detail relocated in revision (utility, deployment, and internal split)

Explanatory prose condensed out of the Discussion during revision is preserved here. No new result is introduced.

The parsimonious form places a low-burden self-report instrument within reach for digital triage. Whether triage delivered this way improves depression outcomes was not evaluated here; the mobile-delivery evidence base cited in Principal Findings [40] concerns intervention delivery to defined patient groups rather than the delivery of screening, and no equivalent trial evidence exists for a transported screening algorithm.

At the development-fixed operating point, weighted net benefit exceeded both the screen-all and the screen-none default strategy across the clinically relevant threshold range in every cohort (Table S3).

Development used a 500,000-record training set and a 100,001-record hold-out drawn from the 2,048,510 pooled records rather than the full pooled sample; a training-set size experiment showed this did not cost external discrimination, and internal estimates are therefore hold-out rather than fully cross-validated.

Equity requires explicit and continuing monitoring rather than an assumption that the development-time estimate holds, as recommended for machine learning intended to advance rather than erode health equity [45]. Three quantities warrant tracking: the subgroup AUROC, the subgroup calibration slope, and the false-negative rate at the operating threshold (Table S7).
