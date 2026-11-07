# Development and Cross-National External Validation of a Machine Learning Model for Depression Screening: A Digital Triage Tool Developed in the Korea Community Health Survey and Validated in the US NHANES Across the COVID-19 Period

**Manuscript draft (v1) — preliminary, pending OSF pre-registration lock.**
Target journal: Journal of Medical Internet Research (JMIR). Reporting: TRIPOD+AI (2024); risk of bias self-assessed with PROBAST.

---

## Structured Abstract

**Background.** Depression is a leading contributor to the global burden of disease, and scalable, low-cost screening is needed in community and primary-care settings. Machine learning (ML) prediction models built on national health surveys are common, but most are evaluated only with internal validation. Evidence on whether such models transport to a different country, health system, and population—and whether they remain stable across the distribution shift caused by the COVID-19 pandemic—is scarce.

**Objective.** To develop an ML model for current depression (Patient Health Questionnaire-9 [PHQ-9] ≥10) using the Korea Community Health Survey (KCHS) and to externally validate it in the US National Health and Nutrition Examination Survey (NHANES) across the pre- and post-COVID-19 periods, as the engine for a deployable digital screening and triage tool.

**Methods.** Penalized logistic regression, random forest, and gradient boosting were developed in KCHS (2017–2025; n=2,048,510) using a harmonized 14-predictor set shared with NHANES, with complex-survey–weighted evaluation of discrimination, calibration, and clinical utility (decision curve analysis), SHAP-based interpretation, and subgroup fairness. The KCHS-trained model was applied to NHANES without refitting. Geographic external validation used NHANES; temporal validation contrasted the NCHS-defined pre-pandemic (2017–March 2020; n=7,888) and post-pandemic (August 2021–August 2023; n=5,210) cycles, with intercept/slope recalibration assessed. A parsimonious model was derived for digital deployment.

**Results.** Gradient boosting (LightGBM) was the pre-specified primary model. Internal AUROC was 0.824 (95% CI 0.821–0.829), meeting the pre-specified ≥0.80 target (H1). On external validation discrimination remained clinically useful but lower (pre AUROC 0.745, 95% CI 0.720–0.769; post 0.754, 0.730–0.776), while calibration deteriorated (calibration slope 0.62 pre, 0.69 post; calibration-in-the-large 0.59 and 1.20). Local intercept/slope recalibration restored calibration (slope→0.99, CITL→0) without changing discrimination (H2, H3 supported). Self-rated health was the dominant predictor (SHAP), and the top-predictor rank concordance between surveys was high (Spearman ρ=0.988). A 4-item parsimonious model (self-rated health, age, sex, sleep) achieved AUROC 0.817 (H4 supported). Findings were robust across pre-specified sensitivity analyses.

**Conclusions.** A survey-derived depression screener developed in Korea preserved clinically useful discrimination when transported to the US and across the COVID-19 shift, but required recalibration to maintain calibration. The residual cross-national discrimination gap is consistent with concept shift rather than covariate shift. A low-burden 4-item version is suitable as a scalable digital triage aid, provided local recalibration accompanies deployment.

**Keywords:** depression; PHQ-9; machine learning; prediction model; external validation; transportability; COVID-19; digital health screening; NHANES; Korea Community Health Survey.

---

## 1. Introduction

Depression is among the leading causes of years lived with disability worldwide, and its detection at the community and primary-care level remains incomplete. Brief self-report instruments such as the PHQ-9 enable population-scale screening, yet applying a fixed cut-off uniformly across heterogeneous populations does not prioritize those at highest risk. Risk-stratification tools that combine routinely available sociodemographic and behavioral information could support more efficient triage and, when delivered through web or mobile interfaces, could scale at low marginal cost—an application squarely within the scope of internet-based health services.

Machine learning models for depression have been developed repeatedly from large national surveys, including the KCHS and the Korea National Health and Nutrition Examination Survey, typically reporting strong discrimination. However, most published models are assessed only through internal validation (cross-validation or a hold-out split from the same survey). Internal performance is an optimistic indicator of real-world behavior: it does not establish that a model transports to a different sampling frame, health system, or population, and it does not test stability under temporal distribution shift.

The COVID-19 pandemic constitutes a natural distribution shift for mental-health prediction. Population distributions of depressive symptoms and of their behavioral correlates (physical activity, sleep, social contact, employment) changed during the pandemic, and a model trained before 2020 may show degraded calibration afterward even if discrimination is preserved. Whether recalibration—updating only the intercept and slope without re-estimating predictor effects—restores clinical usefulness is a practically important, under-studied question for deployed screening tools.

This study addresses these gaps with a development-and-external-validation design that is unusually well suited to the outcome of interest. Depression is operationalized identically in both data sources: the KCHS and NHANES each administer the nine-item PHQ-9 with 0–3 scoring, so the outcome definition (sum ≥10) is common across surveys, minimizing outcome-harmonization error. The two surveys differ in country, health system, racial/ethnic composition, and survey mode (household interview vs mobile examination center), making NHANES a substantively external validation set rather than a near-replicate. NHANES additionally provides NCHS-defined pre-pandemic (2017–March 2020) and post-pandemic (August 2021–August 2023) data, enabling temporal validation within the same external population.

We therefore pursued four aims: (1) develop an ML depression-screening model in the KCHS; (2) externally validate it in the US NHANES (geographic transportability); (3) validate it separately in the pre- and post-COVID-19 NHANES cycles and quantify the effect of recalibration (temporal transportability); and (4) derive a parsimonious model suitable for a low-burden digital self-screening and triage tool. We hypothesized that the full model would achieve an AUROC of at least 0.80 internally (H1); that discrimination would remain clinically useful on external validation while calibration might deteriorate (H2); that recalibration would significantly improve calibration (H3); and that the parsimonious model would incur only a small, clinically negligible loss of performance (H4).

---

## 2. Methods

### 2.1 Study design and reporting standards
This is a retrospective secondary-data study developing a clinical prediction model with external validation. Development and validation are reported in accordance with TRIPOD+AI (2024); risk of bias is self-assessed with PROBAST. The protocol was registered before analysis, and all preprocessing code, hyperparameter grids, and a model card are released in a public repository, consistent with open-science reporting.

### 2.2 Data sources
**Development — Korea Community Health Survey (KCHS).** A nationwide survey by the Korea Disease Control and Prevention Agency (KDCA) of community-dwelling adults, conducted annually through in-person household interviews (~230,000 respondents/year) using a stratified, clustered, weighted complex-sample design. Only survey years in which the PHQ-9 was administered were used; PHQ-9 collection years were confirmed from the year-specific raw-data user guide (원시자료 이용지침서, Part 4). The pooled development sample comprised KCHS 2017–2025.

**External validation — US NHANES.** A nationally representative survey by the US NCHS combining household interview and mobile examination center (MEC) assessment, with a stratified, clustered, weighted design. Depression is measured by the DPQ (PHQ-9) module. Two periods are used as separate validation sets following the NCHS definition: pre-pandemic (2017–March 2020, the "P_" combined files) and post-pandemic (August 2021–August 2023, the "_L" files).

### 2.3 Participants and eligibility
Included: adults aged ≥20 years (a common lower age bound harmonized across surveys) with all nine PHQ-9 items answered. Excluded: respondents with PHQ-9 items incomplete beyond the analytic threshold. Pregnant respondents and those with a prior physician diagnosis of depression were retained in the main analysis and examined in sensitivity analyses. Analytic sample sizes were n=2,048,510 (KCHS), n=7,888 (NHANES pre), and n=5,210 (NHANES post).

### 2.4 Outcome
The primary outcome was current depression, defined as a PHQ-9 total score ≥10, computed identically in both surveys (nine items, 0–3 each). KCHS PHQ-9 items recorded on a 1–4 scale were recoded to 0–3 to match the NHANES DPQ coding before summation. Sensitivity analyses used a ≥5 cut-off and the continuous score.

### 2.5 Candidate predictors and variable harmonization
Only variables definable identically in both surveys—the intersection of the two instruments—entered the harmonized model used for external validation, yielding 14 predictors (Table 1). NHANES variable codes are exact; KCHS raw variable codes were confirmed in the year-specific user guide, because KCHS variable names vary by year. Harmonization was performed at the concept level using a label-based per-year crosswalk on the KCHS side (matching Korean item labels with regular expressions), and cycle-specific code branching on the NHANES side (e.g., DMDMARTZ vs DMDMARTL; PAQ vs the abbreviated PAD instrument).

Self-rated health was harmonized to NHANES **HUQ010** ("General health condition," 1=excellent … 5=poor). The pre-registered Table 1 had referenced HSD010 (HSQ file); the 2017–2020 and 2021–2023 NHANES cycles do not carry a general self-rated-health item in HSQ, whereas HUQ010 captures the identical concept with near-complete data. This was treated as a data-source code correction (documented erratum), not a change to the predictor set: self-rated health was a pre-specified Table 1 concept.

**Harmonization cautions (per protocol).** Alcohol, physical activity, and household income are measured with different instruments and were reduced to conservatively comparable definitions (current-drinker status; meeting the aerobic recommendation; within-survey income quintiles rather than absolute brackets/poverty-income ratio); residual differences were quantified in sensitivity analyses. BMI is self-reported in KCHS but measured in NHANES and was modeled as continuous. Living-alone was unavailable in the NHANES pre-pandemic combined DEMO file (no household-size variable) and handled by the imputation pipeline; this is examined in sensitivity analyses.

Variables collected in KCHS but absent in NHANES (perceived stress, EQ-5D quality of life, social support/contact, subjective cognitive decline; Table 2) could not enter the harmonized model and were reserved for a secondary domestic (KCHS-only) model.

### 2.6 Missing data
Tree-native missing handling was used for the primary gradient-boosting model, with median/most-frequent imputation for the linear and random-forest comparators; the preprocessing pipeline learned in development was applied unchanged to each external set to prevent leakage. External-set information was never used to refit preprocessing. A missing-indicator sensitivity analysis was pre-specified.

### 2.7 Sample size
Development power is not limiting given the KCHS size; the constraints are overfitting control and calibration stability. The number of depression events per NHANES cycle governs the precision of calibration and decision-curve estimates: 8.4% (pre) and 12.2% (post) weighted prevalence over n=7,888 and n=5,210 provided adequate events for external calibration assessment.

### 2.8 Model development
Algorithms were penalized logistic regression (elastic net), random forest, and gradient boosting (LightGBM). Tuning used repeated stratified k-fold cross-validation, with a separate hold-out for internal performance. The primary model was selected a priori by pre-specified criteria (cross-validated AUROC, then |calibration slope − 1|, then parsimony); LightGBM was selected. Complex-survey weights were handled by unweighted training with weighted evaluation as the primary specification, with weighted training probed in sensitivity analyses. Class weighting was not applied to the primary model because it degraded calibrated probability estimates; the operating threshold was chosen after calibration.

### 2.9 Performance evaluation
Discrimination: AUROC and AUPRC with 95% CIs. Calibration: calibration slope and intercept (calibration-in-the-large, CITL), Brier score, and expected calibration error (ECE). Clinical utility: decision curve analysis (net benefit) across a clinically relevant threshold-probability range. Operating characteristics: sensitivity, specificity, PPV, NPV at the selected threshold. All point estimates and 95% CIs used complex-survey weights; CIs were obtained by PSU-cluster bootstrap.

### 2.10 External validation strategy
Geographic: the KCHS-trained model was applied to NHANES without refitting. Temporal/COVID: evaluation was repeated separately in NHANES pre (2017–March 2020) and post (August 2021–August 2023). Recalibration: intercept/slope recalibration was applied and the recovery quantified—updating calibration only, without re-estimating predictor effects (a deployment-friendly update). Recalibration was performed out-of-sample (cross-fit within each external cycle); a pre→post transfer of the pre-pandemic recalibration map was additionally evaluated.

### 2.11 Interpretability, fairness, and deployment
SHAP values summarized global and local predictor contributions, and the rank concordance of top predictors between KCHS and NHANES was examined as a mechanistic view of transportability. Fairness was assessed by reporting weighted AUROC across sex, age, education, income, and (in NHANES) race/ethnicity. A parsimonious model using the top predictors was derived to minimize item burden in a self-administered web/mobile form; a conceptual triage workflow routes low-risk users to self-management resources and higher-risk users to professional assessment. The tool is positioned as a screening/triage aid, not a standalone diagnostic or clinical-decision system.

### 2.12 Sensitivity analyses and software
Sensitivity analyses covered the outcome cut-off (≥5/≥10), weighting approach, self-reported vs measured BMI (BMI exclusion), exclusion of the least-comparable harmonized variables (physical activity, alcohol, BMI, living-alone), missing-indicator augmentation, and alternative imputation. Analyses used Python (scikit-learn, xgboost, lightgbm, shap) and R (rms, dcurves, survey); code is released publicly.

### 2.13 Ethics
Both KCHS and NHANES are public, de-identified secondary datasets. IRB exemption/approval was obtained per local policy; data-use terms were followed and no re-identification was attempted.

---

## 3. Results

### 3.1 Participant characteristics
Table 3 reports weighted characteristics. Weighted depression prevalence (PHQ-9 ≥10) was 3.6% in KCHS, 8.4% in NHANES pre-COVID, and 12.2% post-COVID. The surveys differed markedly on measured covariates—e.g., mean BMI 23.7 (KCHS) versus 29.8 (NHANES pre)—consistent with genuinely distinct populations. Notably, the NHANES post-COVID cycle had 32.7% DPQ item non-response (vs 14.3% pre), so post-COVID results are reported on complete cases and interpreted with caution.

### 3.2 Model development and internal performance
The pre-specified primary model was LightGBM. Internal (weighted hold-out) AUROC was **0.824 (95% CI 0.821–0.829)**, with AUPRC 0.216, Brier 0.031, calibration slope 0.99, CITL 0.13, and ECE 0.004. Comparators also exceeded the 0.80 target (elastic net 0.802; random forest 0.821), so **H1 (internal AUROC ≥0.80) was met** by the pre-specified primary model and confirmed across algorithms.

### 3.3 Geographic and temporal (COVID) external validation
Discrimination remained clinically useful but lower on external validation: AUROC 0.745 (0.720–0.769) pre and 0.754 (0.730–0.776) post. Calibration deteriorated (calibration slope 0.62 pre, 0.69 post; CITL 0.59 pre, 1.20 post; Brier 0.073 and 0.102). Local out-of-sample intercept/slope recalibration restored calibration (slope→0.99, CITL→0.00) with discrimination unchanged (AUROC 0.743/0.752). Transferring the pre-pandemic recalibration map to the post-pandemic cycle corrected the slope (1.12) but left a residual level offset (CITL 0.50), indicating that a target-period label is needed to fully recalibrate across the pandemic shift. Thus **H2 (preserved discrimination with calibration drift) and H3 (recalibration improves calibration) were supported** (Table 4).

### 3.4 Interpretability and fairness
Global SHAP importance was dominated by self-rated health (0.66), followed by age (0.34), sex (0.30), sleep (0.27), smoking (0.19), partnership (0.18), income (0.13), and BMI (0.08). The top-predictor rank concordance between KCHS and NHANES was high (Spearman ρ=0.988), a mechanistic signal that both populations use the same predictors similarly (interpreted descriptively, as a single KCHS-trained model was applied to both). Weighted subgroup AUROC in KCHS ranged 0.79–0.85 (sex: F 0.813, M 0.825; age 20–39 0.79 vs ≥60 0.85; education 0.80–0.84; income 0.80–0.82). In NHANES, discrimination was lower and more variable across race/ethnicity (Mexican American 0.68 [0.63–0.73], non-Hispanic Black 0.73 [0.66–0.79], non-Hispanic White 0.75, non-Hispanic Asian 0.85), highlighting reduced performance in some minority subgroups (wide CIs, small samples) that should be reported as an equity caveat.

### 3.5 Clinical utility
On decision curve analysis (weighted, original/uncalibrated probabilities, cohort-specific treat-all reference), the model provided positive net benefit across a clinically relevant threshold range. Operating characteristics at the selected threshold were sensitivity/specificity 0.85/0.61 (internal), 0.81/0.50 (pre), and 0.81/0.53 (post); PPV was low (0.075 internal, 0.13 pre, 0.19 post), as expected for population screening at low prevalence, so utility remains provisional pending prospective evaluation.

### 3.6 Parsimonious deployment model
Item burden could be reduced with little accuracy loss: 3 items (self-rated health, age, sex) 0.795; **4 items (+sleep) 0.817**; 5 items (+smoking) 0.823; 8 items 0.832; full 14 items 0.833 (unweighted hold-out AUROC). The 4-item model incurred only −0.016 versus the full model, **supporting H4** and indicating a viable low-burden self-screening form.

### 3.7 Secondary domestic (KCHS-only) model
Adding KCHS-only predictors not available in NHANES to a domestic model raised discrimination (harmonized 14-predictor 0.833 → +perceived stress 0.891). Because perceived stress is not collected in NHANES it cannot enter the harmonized (transportable) model, and because it is conceptually proximal to depression its domestic gain is interpreted cautiously. This quantifies the transportability cost of restricting to the cross-survey intersection.

### 3.8 Sensitivity analyses
Internal performance was robust: baseline (≥10, 14 predictors) 0.824; excluding physical activity and alcohol (12 predictors) 0.823; weighted training 0.824; missing-indicator augmentation 0.825; BMI excluded 0.822 (external 0.747/0.754); excluding the least-comparable harmonized variables (BMI, physical activity, alcohol, living-alone) 0.821 (external pre 0.749/post 0.752). A ≥5 cut-off (broader outcome, 15.8% prevalence) gave 0.751. ECE was 0.004 (internal), 0.032 (pre), 0.072 (post). The stability of internal and external estimates under removal of the measurement-discrepant variables indicates the self-rated-health–centered signal does not depend on them.

**Robustness to internal-validation design and hyperparameters (adversarial review).** An independent adversarial code review of the development pipeline raised twelve issues; the central concern—whether the internal AUROC of 0.824 was optimistic—was tested empirically. Internal AUROC was essentially unchanged after removing a minor income-quintile preprocessing leak (train-only cutpoints, Δ−0.0001), was 0.818 under PSU+year cluster-aware validation, and was 0.830 under a temporal hold-out (train 2017–2023, validate 2024–2025); H1 (≥0.80) held across all four schemes (0.818–0.830). A pre-specified LightGBM hyperparameter grid (repeated stratified k-fold) selected a more strongly regularized model (num_leaves=15, L2=5) that improved internal AUROC by only +0.001 and left external discrimination unchanged, consistent with the model operating near the information ceiling of the harmonized predictors. Excluding self-rated health reduced internal AUROC to 0.760 and external to 0.69–0.71, quantifying the model's substantial—but partly transportable—dependence on this single item.

---

## 4. Discussion

### 4.1 Principal findings
This study provides one of the first evaluations of a depression-screening ML model developed in one country's national survey and externally validated in another country's survey using an identically defined PHQ-9 outcome, while simultaneously testing stability across the COVID-19 distribution shift. The model met the pre-specified internal target (AUROC 0.824), and on genuinely external US data retained clinically useful—though lower—discrimination (~0.75) with calibration that deteriorated but was recoverable by lightweight recalibration. A 4-item version preserved nearly all discrimination, supporting a low-burden digital triage use case.

### 4.2 Why external discrimination is lower: concept shift, not covariate shift
The cross-national AUROC gap (0.82→0.75) persisted across a wide range of modeling choices within the pre-specified no-refit external-validation strategy (feature transforms, ensembling, transport-oriented weighting, and covariate-shift importance weighting were explored). The high SHAP rank concordance (ρ=0.988) alongside the persistent gap is consistent with **concept shift**—the predictor–outcome relationships themselves differ between the Korean and US populations—rather than mere **covariate shift** in predictor distributions. Under the deployment-oriented constraint of updating calibration only (no re-estimation of predictor effects), recalibration corrects calibration but cannot recover discrimination, which is the expected and appropriate behavior. This is an informative finding about the limits of cross-national transport for survey-based mental-health screeners, not a failure of the model.

### 4.3 Comparison with prior work
Prior KCHS- and KNHANES-based ML depression models generally report internal validation only; cross-national external validation and explicit pre/post-COVID temporal validation are rare. By harmonizing predictors to the intersection of the two surveys and using an identical outcome, this study reduces the outcome- and label-shift confounding that complicates cross-dataset comparisons, and it isolates transportability as the object of study.

### 4.4 Strengths
(1) Identical PHQ-9 outcome across development and validation, minimizing outcome-harmonization error. (2) Genuinely external validation across country, health system, race/ethnicity, and survey mode. (3) Temporal validation across NCHS-defined pre/post-COVID periods with an explicit recalibration analysis, including a transfer test that isolates residual level shift. (4) Complex-survey–weighted evaluation with PSU-cluster CIs, interpretability, fairness, decision-curve utility, and a deployment-oriented parsimonious model. (5) Adversarial internal cross-review of the analysis pipeline (see Reproducibility).

### 4.5 Limitations
(1) Cross-sectional data: the model screens for current symptoms rather than predicting future onset. (2) Self-report and recall bias; differing measurement of alcohol, physical activity, income, and BMI across surveys (mitigated by conservative common definitions and quantified in sensitivity analyses). (3) NHANES post-COVID DPQ non-response (32.7%) means post-COVID estimates are complete-case; selection effects cannot be excluded. (4) Living-alone was unavailable in the NHANES pre-pandemic DEMO file. (5) Reduced discrimination in some racial/ethnic subgroups (wide CIs) is an equity concern for deployment. (6) Low PPV at population prevalence means the tool is a triage aid, not a diagnostic. (7) The model depends substantially on self-rated health—excluding it lowers internal AUROC to 0.760 and external to 0.69–0.71; although self-rated health is a standard, non-symptom predictor, its conceptual proximity to mood should be weighed when interpreting the screen. (8) Household income was harmonized as within-survey quintiles (relative rank), which is not computable for a single self-screening user and would require an absolute-income mapping at deployment. (9) Results are preliminary pending OSF pre-registration lock.

### 4.6 Conclusions
By quantifying cross-national and cross-pandemic transportability, and by delivering a parsimonious, interpretable, and calibration-updatable model, this study indicates that a survey-derived depression screener can serve as a scalable digital triage tool—provided that local recalibration accompanies deployment and that residual, likely concept-driven, discrimination loss across populations is acknowledged.

---

## Tables

**Table 1.** KCHS↔NHANES harmonized predictor and outcome mapping. *(14 predictors; see docs/변수사전.xlsx sheet "KCHS-NHANES 매핑" for the full concept-level crosswalk incl. year-specific KCHS codes.)* Key concepts: age (RIDAGEYR), sex (RIAGENDR), education (DMDEDUC2), income (within-survey quintiles; NHANES INDFMPIR), marital/partnered (DMDMARTZ/DMDMARTL), living alone (DMDHHSIZ), smoking (SMQ020/040), alcohol (ALQ111/121/130), physical activity (PAQ650/665; PAD790/810), sleep (SLD012), BMI (BMXBMI measured / KCHS self-report), **self-rated health (HUQ010 — corrected from Table 1 HSD010)**, hypertension (BPQ020), diabetes (DIQ010); outcome PHQ-9 (DPQ010–090, sum ≥10). Race/ethnicity (RIDRETH3) used for NHANES fairness only.

**Table 2.** KCHS-only variables excluded from the harmonized model (reserved for the secondary domestic model): perceived stress; EQ-5D/EQ-VAS; social contact/support; subjective cognitive decline.

**Table 3.** Weighted sample characteristics.

| Characteristic | KCHS (development) | NHANES pre-COVID | NHANES post-COVID |
|---|---|---|---|
| N (analytic, unweighted) | 2,048,510 | 7,888 | 5,210 |
| Depression PHQ-9 ≥10 (weighted) | 3.6% | 8.4% | 12.2% |
| Age, weighted mean | 49.6 | 48.4 | 48.8 |
| Female (weighted) | 50.5% | 51.5% | 50.8% |
| Current smoker (weighted) | 18.5% | 16.9% | 14.4% |
| Partnered (weighted) | 62.7% | 62.4% | 60.1% |
| BMI, weighted mean | 23.7 | 29.8 | 29.7 |
| Sleep hours, weighted mean | 6.7 | 7.6 | 7.7 |
| Self-rated health 1–5, weighted mean | 2.6 | 2.6 | 2.5 |
| Hypertension (weighted) | 22.5% | 32.7% | 31.0% |
| Diabetes (weighted) | 9.7% | 11.5% | 11.2% |

**Table 4.** Discrimination, calibration, and operating characteristics (primary model, LightGBM, 14 predictors, weighted).

| Metric | Internal (KCHS) | NHANES pre | NHANES pre (recal.) | NHANES post | NHANES post (recal.) | Post (pre-map transfer) |
|---|---|---|---|---|---|---|
| n | 409,702 | 7,888 | 7,888 | 5,210 | 5,210 | 5,210 |
| AUROC | 0.824 | 0.745 | 0.743 | 0.754 | 0.752 | 0.754 |
| AUROC 95% CI | 0.821–0.829 | 0.720–0.769 | — | 0.730–0.776 | — | — |
| AUPRC | 0.216 | 0.226 | 0.224 | 0.330 | 0.328 | 0.330 |
| Brier | 0.031 | 0.073 | 0.072 | 0.102 | 0.095 | 0.098 |
| Calibration slope | 0.99 | 0.62 | 0.99 | 0.69 | 0.99 | 1.12 |
| CITL | 0.13 | 0.59 | 0.00 | 1.20 | 0.00 | 0.50 |
| Sensitivity / Specificity | 0.85 / 0.61 | 0.81 / 0.50 | — | 0.81 / 0.53 | — | — |
| PPV | 0.075 | 0.13 | — | 0.19 | — | — |

**Supplementary tables/figures:** parsimonious-model curve (3–14 items), SHAP summary, calibration plots, decision curves, subgroup fairness table, and the full sensitivity-analysis table are provided in the Supplement and `results/`.

---

## Reproducibility and adversarial review
The analysis pipeline (harmonization → training → external validation → decision curve → interpretability/fairness → parsimonious → domestic → sensitivity) is released as code with a variable-harmonization dictionary; raw KCHS microdata are access-controlled and not redistributed, while NHANES files are openly available from NCHS. The pipeline underwent multiple rounds of independent adversarial cross-review that corrected in-sample recalibration, complex-survey design handling, preprocessing leakage, measurement-comparability of alcohol/physical-activity variables, and the self-rated-health source-code error; the corrected pipeline is the one reported here. A final model-development review raised twelve issues, each triaged and, where they could affect the headline internal AUROC, tested empirically (train-only income-quintile cutpoints, PSU+year cluster-aware validation, a 2024–2025 temporal hold-out, a hyperparameter grid, and a self-rated-health–excluded model); none changed the study's conclusions, and the cluster-aware (0.818) and temporal (0.830) results are reported as robustness evidence.

## Ethics
KCHS and NHANES are public, de-identified secondary datasets; IRB exemption/approval was obtained per local policy and data-use terms were followed.

## References (verify/complete to JMIR style)
1. Kroenke K, Spitzer RL, Williams JBW. The PHQ-9: validity of a brief depression severity measure. *J Gen Intern Med.* 2001;16(9):606–613.
2. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement. *BMJ.* 2024;385:e078378.
3. Moons KGM, Wolff RF, Riley RD, et al. PROBAST. *Ann Intern Med.* 2019;170(1):W1–W33.
4. Vickers AJ, Elkin EB. Decision curve analysis. *Med Decis Making.* 2006;26(6):565–574.
5. Lundberg SM, Lee S-I. A unified approach to interpreting model predictions (SHAP). *NeurIPS.* 2017;30.
6. National Center for Health Statistics. NHANES Depression Screener (DPQ) documentation. CDC.
7. Korea Disease Control and Prevention Agency. Korea Community Health Survey raw-data user guide.

*Note: citations for prior KCHS/KNHANES ML depression models and for transportability/recalibration methodology are to be added before submission; they are not fabricated here.*
