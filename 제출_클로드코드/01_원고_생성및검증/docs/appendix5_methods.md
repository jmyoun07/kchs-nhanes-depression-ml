### Study design, model development, and analysis detail

This section states what a reader needs in order to reproduce the analysis, the main text stating what a reader needs in order to judge it. Estimates are complex-survey weighted unless a row or column is labeled unweighted; nothing here restates a result reported in the manuscript, and where a number appears in both, the manuscript is the reporting site and this section gives the derivation.

### Prior-model reviews cited in the Introduction

Supports the Introduction. The Introduction states that most published prediction models are assessed through internal validation alone, in depression modeling and in other clinical domains alike, and cites two reviews for that claim [7,8]; the detailed findings of both are recorded here. A systematic review of machine-learning depression models built on electronic health records [7] found reported discrimination clustered near 0.80, with generalizability and interpretability repeatedly identified as unresolved concerns. A review of hundreds of prediction models published during the COVID-19 pandemic [8] reached the same conclusion in a different clinical domain: almost none of those models were validated in data independent of the development source, and most carried a high risk of bias. Convergence of the two reviews indicates that internal-validation-only reporting is not confined to depression modeling. The near-0.80 clustering is the benchmark against which the internal estimate of 0.836 obtained here is read under Comparison With Prior Work in the Discussion.


### Study design, split procedure, and risk-of-bias assessment

Supports the Methods section Study Design and Data Sources.

The PROBAST self-assessment rated the participants, predictors and outcome domains at low risk of bias in both the development and the validation data, and the analysis domain at low risk for the development data. The analysis domain was rated unclear for the external cohorts, given that depression-module non-response was substantial and differential and was handled as complete-case (Table S1), which gives an overall low-to-unclear rating driven by that non-response.

The internal split was constructed as follows. From the pooled KCHS 2017-2025 sample of 2,048,510 records, clusters defined as survey year crossed with design stratum crossed with primary sampling unit were randomly permuted and assigned whole to the hold-out until 100,001 records had accumulated; the training set is a random 500,000-record draw from the records remaining in the clusters not assigned to the hold-out. The two sides share 0 clusters, and the hold-out spans 9,636 clusters. The internal estimate therefore refers to clusters the model never saw and not merely to new individuals within sampled clusters. The training-set size was fixed for computational tractability before any external evaluation, and a subsequent check confirmed that expanding it toward the full pooled sample did not improve, and slightly reduced, external discrimination (Table S11 in Multimedia Appendix 1).

On the NHANES side, the pre-pandemic combined file and the 2021-2023 cycle carry separate NCHS documentation and weight variables, which are named in the complex-survey design-variables table above.

The unmodified original protocol, statistical analysis plan, and amendment document are available with the analysis code, hyperparameter grids, and model card (see Data Availability). This appendix additionally provides the year-specific harmonization crosswalk together with an English rendition of the locked amendment.


### Participant eligibility and analytic sample derivation

This section supports the Methods subsection Participants and Outcome and Table S1. Eligibility in both surveys was restricted to adults aged 20 years or older with all nine PHQ-9 items answered.

In NHANES the depression module is administered in the mobile examination center, and the estimation frame is therefore the examined sample. Of 9,232 interviewed adults aged 20 years or older in the pre-pandemic file, 8,544 were examined and 7,888 answered all nine items; the corresponding post-pandemic counts were 7,809, 6,064 and 5,210. Table S1 gives the same three stages together with item-level non-response and its education-specific and race-specific rates.

A prior physician diagnosis of depression is not recorded in the KCHS 2017-2025 questionnaires and therefore could not be used for exclusion or sensitivity analysis; this is noted as a limitation (Strengths and Limitations, and Table S2).


### Harmonization resolution, scale differences, and preprocessing

Supports the Methods section Predictors and Harmonization and complements the year-by-year crosswalk in Multimedia Appendix 1. 
Concept-level harmonization used a label-based per-year crosswalk on the KCHS side and cycle-specific code branching on the NHANES side. Self-rated health was harmonized to NHANES HUQ010, which corrects a file-code error in the pre-specification document, where the variable had been named HSD010/HSQ. The variables excluded a priori as circular with or contaminated by the outcome are the PHQ-9 items, suicidal ideation, the EQ-5D anxiety/depression item, the happiness index, perceived stress, and mental-health service use, the last including NHANES HUQ090 on the external side; the table of variables excluded as circular with the outcome lists each exclusion with its reason.

Five further harmonizable items surfaced by the exhaustive extraction were screened after the primary analysis by comparing the direction of the adjusted association with depression across the three cohorts and then testing each item in the model. Three reversed direction between countries and the remainder added no external discrimination. That screening inspected NHANES outcome associations and is therefore a post-hoc diagnostic; no item was added, and the predictor set is unchanged.

Living alone requires a caveat that the tree learner's native missing support does not remove. Living alone has no missing values anywhere in the KCHS training data, hence the learner never observes a missing value for the item and never learns a missing direction for it. At prediction time the missing values are treated as zero and follow the same branch as "not living alone", and the pre-pandemic cohort is thus scored as if no respondent lived alone.

Of the three predictors reduced to conservatively comparable definitions, only alcohol could be varied in sensitivity analysis: the harmonized data carry a single binary physical-activity indicator and a within-survey income quintile, hence no alternative definition of those two was available. The protocol specified alcohol as current drinking plus at-risk drinking, whereas the harmonized variable uses current drinking alone, given that the NHANES quantity item was not carried through harmonization; this is recorded as a definition-level deviation in Table S2. Body mass index is self-reported in KCHS and measured in NHANES.

Two scale differences arise from NHANES top-coding. Age is capped at 80 years in NHANES, whereas 3.8% of the weighted KCHS sample is older (up to 110); the same cap was applied to KCHS before fitting, so that the model cannot separate respondents above 80 years in either survey and that part of the Korean age range is compressed. NHANES sleep duration is bounded at 2-14 hours while KCHS admits 0-23 hours, though implausible KCHS values account for only 0.01% of respondents.

Three further discontinuities are internal to KCHS. The current-drinking items were recoded in 2020, and the weighted prevalence rises from 72.5-75.4% before that year to 81.8-86.8% after it. The 2018 physical-activity module lacks the moderate-activity day count. Sleep duration is a weekday-only figure in 2017-2019 but a weekday-weighted seven-day average from 2020, whereas NHANES asks about weekdays. These three, together with the age top-code, the 2017 income refusal codes and the income quintile, were corrected in the analytic data before the primary analysis rather than carried as limitations; Table S7 quantifies singly each of the six tabulated corrections.

The preprocessing pipeline was fitted in the development training set and applied unchanged to the internal hold-out and to each external set. For the linear comparator, continuous predictors were median-imputed, categorical predictors mode-imputed, and binary predictors mode-imputed with an added missing indicator. The three ordinal scores (household income quintile, self-rated health and self-rated oral health) were median-imputed and standardized. In the post-pandemic cohort the income-to-poverty ratio is top-coded at 5.00 for 23.8% of respondents, hence the income quintile would collapse to four levels if the raw ratio were binned directly; the primary analysis instead recomputes a rank-based five-level score within each cohort, which restores five levels. The two alternative strategies examined in sensitivity analysis were missing indicators added for the continuous predictors, and single chained-equations imputation of the continuous predictors; weighted AUROC for both is given in Table S6. A check for silently omitted predictor columns was added to the pipeline after the data-integrity audit.

The amendment is summarized in English, with contextual notes marked as such, and the unmodified Korean original is released with the code, together with the crosswalk.


### Model development, tuning, and resampling detail

This section carries the reproduction-level detail of the development, tuning and interval-construction procedures summarized in the Methods subsection Model Development and Evaluation. The external event counts also govern the precision of the external calibration estimates and of the decision-curve analysis.

The feature-representation and hyperparameter search compared 24 LightGBM configurations under single-pass 4-fold survey-weighted cross-validation in a 300,000-record subsample of the training set, crossed with two representations: main effects only, and main effects plus four candidate interaction terms (age×self-rated health, self-rated health×oral health, age×employment, income×employment). Those four terms were specified when the search was written and are not named in any earlier document (Table S2). The algorithm comparison then tuned elastic net, random forest and LightGBM over comparable 5-configuration grids under one protocol, single-pass 5-fold stratified cross-validation in a 200,000-record subsample of the training set scored by unweighted cross-validated AUROC, and compared each algorithm at its best configuration. Inside the plan's tie band the decision passes to the calibration slope, and the parsimony level of the rule is therefore never reached; the cross-validated values, the tie-band membership of each algorithm and the retrospective reconstruction of the rule are given in Table S2.

The primary model was refitted on the full 500,000-record training set with num_leaves 31, min_child_samples 1000, learning_rate 0.01, n_estimators 900, reg_alpha 1.0, reg_lambda 20, colsample_bytree 0.6, and subsample 0.6 with subsample_freq at its default of 0. The selected row-subsampling fraction is inactive in that configuration, given that the subsampling frequency was left at zero; the fitted model therefore uses all rows at every boosting iteration. The full search grids are specified in the released analysis code, and the selected configuration is recorded in the model card.

Calibration-in-the-large is estimated as the intercept of a weighted logistic model with the linear predictor entered as an offset, ie, with the slope fixed at 1; a positive value indicates that observed risk exceeds predicted risk. For the recalibrated columns of Table 2 the interval is obtained by refitting the intercept and slope inside every bootstrap replicate, so that the uncertainty of the update itself is carried; the wider fixed-update alternative is tabulated alongside it in Table S3. The interval for the difference in external AUROC between the two cycles is formed by differencing independently drawn replicate statistics rather than by pairing replicate indices, since the two cycles are disjoint samples. Bootstrap P values are read off the replicate distribution as twice the smaller tail proportion, and with 1000 replicates the smallest attainable value is reported as P<.001 rather than as an equality.

Bootstrap intervals are taken as the 2.5th and 97.5th percentiles of the replicate distribution. Within each design stratum, *n*ₕ-1 primary sampling units are resampled with replacement and the survey weights rescaled by *n*ₕ/(*n*ₕ-1) times the selection count, following the replication approach cited in the manuscript. Rescaling is necessary given that most NHANES strata contain only two primary sampling units, for which an ordinary within-stratum cluster bootstrap understates variance. The implementation was verified against a closed-form design-based (linearized ultimate-cluster) standard error for weighted depression prevalence: bootstrap 0.00466 vs design 0.00471 in the pre-pandemic cycle and 0.00802 vs 0.00796 in the post-pandemic cycle, ie, agreement to within about 1.1%. For KCHS the stratum is defined as survey year crossed with the design stratum, owing to weight normalization within year and to the recurrence of stratum codes across years. Subgroup intervals are domain estimates: replicate weights are generated once on the full cohort design and then applied to each subgroup, rather than rebuilding the design after subsetting. Strata left with a single primary sampling unit are pooled into one pseudo-stratum, ensuring that the variance contribution of those strata is retained.

Applying the tie-break rule, the two candidate slopes differ by 0.20 (1.003 for LightGBM against 1.203 for random forest), which is not a close call, and the rule selects LightGBM.
### External validation, recalibration, parsimony selection, and the data-integrity audit

This section carries the procedural detail underlying the manuscript subsection External Validation and Sensitivity Analyses. Nothing here alters any reported estimate.

Each of the three recalibration steps was cross-fitted over five folds grouped by primary sampling unit, so that no cluster contributes both to fitting an update and to evaluating it. The two-parameter intercept-plus-slope step is the Platt-type update.

Cross-survey rank concordance of predictor importance was undertaken as a mechanistic view of transportability. The fairness analysis was undertaken since algorithmic performance can differ systematically across groups even when overall accuracy is high.

For the parsimonious model, global Shapley additive explanations (SHAP) importance was computed on a random 20,000-record subsample of the 450,000-record fitting set. The search ran over every item count below 17 and did not impose the analysis plan's preferred 5-10 item range, which the selected count nonetheless satisfies. A variant using a stricter 0.01 margin is reported alongside the primary selection and is flagged as post hoc. Table S13 gives the full item-count trajectory.

The model is not a diagnostic instrument and does not replace clinical assessment. Since the household income predictor is a within-cohort rank, the 17-predictor model cannot be scored from one respondent's answers alone. The six-item form carries no such variable and no measured value; every one of its items is self-reportable in a web or mobile form, which is what makes it, rather than the full model, the version suited to remote and mobile delivery.

The self-report (KCHS) versus measured (NHANES) difference in body mass index is inherent to the data sources and is addressed in the harmonization cautions under Predictors and Harmonization and under Strengths and Limitations rather than by swapping measurements.

Complex-survey weighting, the rescaled stratified cluster bootstrap, the calibration measures, and decision-curve net benefit were implemented directly rather than through a survey-analysis package, and the implementations are released with the analysis code.

The adversarial data-integrity audit corrected three silent data-processing defects before the final analysis: living alone fabricated to a constant in NHANES pre-pandemic; post-pandemic physical activity coded 0 for every respondent through a unit-field parsing error; and an income refusal code (77777) not treated as missing, which had placed non-respondents in the top income quintile.

Peer review identified four further defects, all corrected in the primary analysis rather than carried forward as limitations: the income quintile collapsed to four levels post-pandemic through top-coding of the income-to-poverty ratio; the 2017 income refusal codes entered the highest quintile; post-2020 annual non-drinkers were coded as current drinkers (17.9% weighted of 2020-2025); and the 2018 physical-activity indicator used a different construct from adjacent years.

The remedies were: post-2020 annual non-drinkers recoded as non-drinkers; the 2018 activity indicator set to missing rather than silently redefined; the 2017 refusal codes masked and the income quintile recomputed as a rank-based five-level score within each survey year, which repairs both the four-level collapse and the scale mismatch between the 2017 ordinal field and the later monetary fields; sleep duration expressed on the weekday basis NHANES uses in every KCHS year; the age top-code aligned at 80 years in all cohorts; and the internal split rebuilt at cluster level.

Table S7 quantifies six of these corrections one at a time: the three questionnaire-driven KCHS discontinuities (the 2020 drinking recode, the 2018 activity fallback and the sleep alignment), the age alignment, the 2017 KCHS income refusal codes, and the income-quintile rebuild, the last of which also removes the four-level collapse in the post-pandemic cohort. Three corrections are not tabulated separately: the living-alone and physical-activity audit corrections, since in the uncorrected data the affected variable is constant rather than mismeasured, and the KCHS 77777 income refusal code, which is a missing-value declaration rather than a change of definition.


### Ethics approvals, consent, and data governance

Supports the Methods section Ethical Considerations. The manuscript states the approvals, the exemption grounds and the consent position in condensed form; the full record is set out here.

The two data sources are governed differently. NHANES is publicly released by the NCHS, and NHANES data collection was approved by the NCHS Research Ethics Review Board. KCHS microdata are access-controlled: the records analyzed here were obtained under a KDCA data-use agreement, and KCHS is conducted under the authority of the KDCA. Both sources as analyzed are de-identified secondary survey data, and the terms governing access to each are given in the manuscript section Data Availability.

The Institutional Review Board of Daejeon University reviewed the present secondary analysis and granted an exemption from review (approval number [to be inserted]) on the grounds that the study analyzes de-identified secondary survey data obtained under the data-collecting agencies' release conditions, collects and records no personal identifiers, and poses no more than minimal risk (no participant contact, intervention, or biological sampling).

Written informed consent was obtained from every participant by the data-collecting agencies at the time of the original surveys. The present analysis of de-identified records required no additional consent, and none was sought. No re-identification was attempted, no compensation was involved, and no individually identifiable information is reported. Patients and members of the public were not involved in the design, conduct, reporting or dissemination plans of this secondary analysis of existing survey data.

### Crosswalk coverage, the food-insecurity measure, and predictor coding

Supports the Methods section Predictors and Harmonization.

The crosswalk available when the protocol amendment was written and locked held a 175-variable extract, of which the seventeen predictors are every eligible concept; the predictor set is thus exhaustive with respect to that extract rather than with respect to the full crosswalk. A later exhaustive extraction identified 265 variables common to both NHANES cycles, and the five further harmonizable items screened post hoc against NHANES outcome associations came from that extraction.

A direct Korea-US comparison using the same 18-item household food-security module found food insecurity associated with higher odds of PHQ-9 depression in both countries, with a larger association in Korea [31]. The single-item KCHS measure used here is therefore the weakest link in the harmonized set rather than a difference in the underlying construct.

The three ordinal scores were modeled as numeric variables rather than as unordered categories, an ordinal coding rather than a set of indicator columns. Recomputation of the income quintile within each dataset and, inside KCHS, within each survey year is the one preprocessing step not carried over from the development training set unchanged, the underlying income instruments not being on a common scale across the two surveys or across KCHS years.

### Comparability of the two algorithm searches and the parsimony procedure

Supports the Methods sections Model Development and Evaluation and External Validation and Sensitivity Analyses.

The training set's approximately 1,000 events per candidate predictor is far above conventional events-per-predictor requirements. Only gradient boosting received the preceding feature-representation and hyperparameter search, hence the algorithm comparison is like-for-like at each algorithm's tuned best configuration rather than equal in cumulative search effort. The four candidate interaction terms did not improve cross-validated discrimination.

Both algorithm searches used individual-level cross-validation, and the selection step may therefore be mildly optimistic; the reported performance rests instead on the cluster-level separation of the internal hold-out and of the parsimony selection set and is unaffected.

Predictor order and item count were both fixed inside the development training set, split into a 450,000-record fitting set and a 50,000-record selection set. Predictors were entered in descending order of global SHAP importance, the model was refitted at every item count from 1 to 17, and the count was chosen on selection-set AUROC by the analysis plan's margin, the smallest set within 0.02 AUROC of the full model.

### Decision-curve construction, intended use, and deployment requirements

Supports the Methods section External Validation and Sensitivity Analyses.

Each dataset carries its own screen-all and screen-none default strategies in the decision-curve analysis, and the reported net benefit is therefore internal to each cohort (Table S5 in Multimedia Appendix 1).

The model is intended as a triage aid for population-level screening programs and web or mobile self-assessment services, operated by public-health and primary-care services rather than by individuals acting alone, to prioritize who receives a full PHQ-9 and clinical follow-up.

Deployment presupposes local recalibration and, the household income predictor being a within-cohort rank, a local income distribution to rank against.
