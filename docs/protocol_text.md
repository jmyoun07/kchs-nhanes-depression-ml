Development and Cross-National External Validation of a Machine Learning Model for Depression Screening: A Digital Triage Tool Developed in the Korea Community Health Survey and Validated in the US NHANES Across the COVID-19 Period
Study Protocol / Research Design
Target journal: Journal of Medical Internet Research (JMIR). Alternatives: JMIR Public Health and Surveillance, JMIR Medical Informatics, JMIR Mental Health.
Reporting guideline: TRIPOD+AI (2024); risk of bias self-assessed with PROBAST. Protocol pre-registered (e.g., OSF) before analysis.
[Heading1] Structured Abstract
Background. Depression is a leading contributor to the global burden of disease, and scalable, low-cost screening is needed in community and primary-care settings. Machine learning (ML) prediction models built on national health surveys are common, but most are evaluated only with internal validation. Evidence on whether such models transport to a different country, health system, and population, and whether they remain stable across the distribution shift caused by the COVID-19 pandemic, is scarce.
Objective. To develop an ML model for current depression (Patient Health Questionnaire-9 [PHQ-9] ≥10) using the Korea Community Health Survey (KCHS) and to externally validate it in the US National Health and Nutrition Examination Survey (NHANES) across the pre- and post-COVID-19 periods, as the engine for a deployable digital screening and triage tool.
Methods. Penalized logistic regression, random forest, and gradient-boosting models will be developed in KCHS using a harmonized predictor set shared with NHANES and evaluated for discrimination, calibration, and clinical utility (decision curve analysis), with SHAP-based interpretation and subgroup fairness. Geographic external validation will be performed in NHANES; temporal validation will contrast the NCHS-defined pre-pandemic (2017–March 2020) and pandemic/post-pandemic (August 2021–August 2023) cycles, with recalibration assessed. A parsimonious model will be derived for digital deployment.
Results. This is a protocol; results are pending. Planned tables and figures, with pre-specified metrics and placeholders, are provided.
Conclusions. The study will quantify the cross-national and cross-pandemic transportability of a survey-based depression screening model and assess its suitability as a low-burden digital triage tool.
Keywords: depression; PHQ-9; machine learning; prediction model; external validation; transportability; COVID-19; digital health screening; NHANES; Korea Community Health Survey.

[Heading1] 1. Introduction
Depression is among the leading causes of years lived with disability worldwide, and its detection at the community and primary-care level remains incomplete. Brief self-report instruments such as the PHQ-9 enable population-scale screening, yet applying a fixed cut-off uniformly across heterogeneous populations does not prioritize those at highest risk. Risk-stratification tools that combine routinely available sociodemographic and behavioral information could support more efficient triage, and, when delivered through web or mobile interfaces, could scale at low marginal cost—an application squarely within the scope of internet-based health services.
Machine learning models for depression have been developed repeatedly from large national surveys, including the KCHS and the Korea National Health and Nutrition Examination Survey, typically reporting strong discrimination. However, most published models are assessed only through internal validation (cross-validation or a hold-out split from the same survey). Internal performance is an optimistic indicator of real-world behavior: it does not establish that a model transports to a different sampling frame, health system, or population, and it does not test stability under temporal distribution shift.
The COVID-19 pandemic constitutes a natural distribution shift for mental-health prediction. Population distributions of depressive symptoms and of their behavioral correlates (physical activity, sleep, social contact, employment) changed during the pandemic, and a model trained before 2020 may show degraded calibration afterward even if discrimination is preserved. Whether recalibration—updating only the intercept and slope without re-estimating predictor effects—restores clinical usefulness is a practically important, under-studied question for deployed screening tools.
This protocol addresses these gaps with a development-and-external-validation design that is unusually well suited to the outcome of interest. Depression is operationalized identically in both data sources: the KCHS and NHANES each administer the nine-item PHQ-9 with 0–3 scoring, so the outcome definition (sum ≥10) is common across surveys, minimizing outcome harmonization error. The two surveys differ in country, health system, racial/ethnic composition, and survey mode (household interview vs mobile examination center), making NHANES a substantively external validation set rather than a near-replicate. NHANES additionally provides National Center for Health Statistics-defined pre-pandemic (2017–March 2020) and pandemic/post-pandemic (August 2021–August 2023) data, enabling temporal validation within the same external population.
We therefore pursue four aims: (1) develop an ML depression-screening model in the KCHS; (2) externally validate it in the US NHANES (geographic transportability); (3) validate it separately in the pre- and post-COVID-19 NHANES cycles and quantify the effect of recalibration (temporal transportability); and (4) derive a parsimonious model suitable for a low-burden digital self-screening and triage tool. We hypothesize that the full model will achieve an AUROC of at least 0.80 internally; that discrimination will remain clinically useful on external validation while calibration may deteriorate; that recalibration will significantly improve calibration; and that the parsimonious model will incur only a small, clinically negligible loss of performance.

[Heading1] 2. Methods
[Heading2] 2.1 Study design and reporting standards
This is a retrospective secondary-data study developing a clinical prediction model with external validation. Development and validation are reported in accordance with TRIPOD+AI (2024); risk of bias is self-assessed with PROBAST. The protocol is registered before analysis, and all preprocessing code, hyperparameter grids, and a model card are released in a public repository, consistent with open-science reporting.
[Heading2] 2.2 Data sources
Development — Korea Community Health Survey (KCHS). A nationwide survey by the Korea Disease Control and Prevention Agency (KDCA) of community-dwelling adults aged ≥19 years, conducted annually through in-person household interviews (~230,000 respondents/year) using a stratified, clustered, weighted complex-sample design. Only survey years in which the PHQ-9 is administered will be used; the exact PHQ-9 collection years must be confirmed from the year-specific raw-data user guide (원시자료 이용지침서, Part 4: Variable Descriptions) before finalizing the development sample.
External validation — US NHANES. A nationally representative survey by the US National Center for Health Statistics (NCHS) combining household interview and mobile examination center assessment, with a stratified, clustered, weighted design. Depression is measured by the DPQ (PHQ-9) module (adults analyzed). Two periods are used as separate validation sets following the NCHS definition: pre-pandemic (2017–March 2020, the ‘P_’ combined files) and pandemic/post-pandemic (August 2021–August 2023, the ‘_L’ files).
[Heading2] 2.3 Participants and eligibility
[ListParagraph] Included: adults meeting a common lower age bound harmonized across surveys, with all nine PHQ-9 items answered.
[ListParagraph] Excluded: PHQ-9 items incomplete beyond the analytic threshold. Pregnant respondents and those with a prior physician diagnosis of depression are retained in the main analysis and examined in sensitivity analyses.
[Heading2] 2.4 Outcome
The primary outcome is current depression, defined as a PHQ-9 total score ≥10, computed identically in both surveys (nine items, 0–3 each). Sensitivity analyses use a ≥5 cut-off and the continuous score.
[Heading2] 2.5 Candidate predictors and variable harmonization
Only variables that can be defined identically in both surveys—the intersection of the two instruments—enter the harmonized model used for external validation. Table 1 gives the concept-level mapping with NHANES file and variable codes and the harmonized modeling definition. NHANES variable codes are exact; KCHS entries are specified at the questionnaire-item / published-indicator level and the raw variable code must be confirmed in the year-specific user guide (variable names vary by year).
Table 1. KCHS↔NHANES harmonized predictor and outcome mapping.
Concept / domain
KCHS (item · published indicator)
NHANES (file · variable)
Harmonized definition (modeling)
Age
Age in completed years
DEMO · RIDAGEYR
Continuous + 10-year bands
Sex
Sex
DEMO · RIAGENDR
Binary (male/female)
Education
Education level item
DEMO · DMDEDUC2
4 levels: ≤elementary / middle / high / ≥college
Household income
Monthly household income bracket
DEMO · INDFMPIR (poverty-income ratio)
Within-survey quintiles (relative rank)
Marital / partnership
Marital status item
DEMO · DMDMARTZ (P_, L cycles) / DMDMARTL (≤2018)
Partnered vs not partnered
Living alone
Household type / one-person household
DEMO · DMDHHSIZ
Living alone (yes/no)
Race / ethnicity
Not collected (used for context only)
DEMO · RIDRETH3
Validation-set fairness subgroup ONLY
Smoking
Current smoking item
SMQ · SMQ020, SMQ040
Never / former / current
Alcohol use
Monthly drinking / high-risk drinking
ALQ · ALQ111, ALQ121, ALQ130
Current drinker + at-risk (see caution)
Physical activity
Walking / moderate activity practice
PAQ · PAQ650, PAQ665 (+PAQ605, PAQ620)
Meets aerobic recommendation (yes/no)
Sleep duration
Sleep duration item
SLQ · SLD012
Hours (continuous) + short/normal/long
BMI / obesity
Self-reported height & weight → BMI
BMX · BMXBMI (measured)
BMI continuous; self-report vs measured (see caution)
Self-rated health
Subjective health level
HSQ · HSD010
5-level ordinal
Hypertension
Physician diagnosis of hypertension
BPQ · BPQ020
Self-reported diagnosis (yes/no)
Diabetes
Physician diagnosis of diabetes
DIQ · DIQ010
Self-reported diagnosis (yes/no)
OUTCOME — Depression
PHQ-9, 9 items (0–3 each)
DPQ · DPQ010–DPQ090
Sum score ≥ 10 = current depression

Harmonization cautions: Alcohol, physical activity, and income are measured with different instruments and must be reduced to conservatively comparable definitions; residual differences are quantified in sensitivity analyses. BMI is self-reported in KCHS but measured in NHANES. Marital and alcohol variable names are cycle-dependent in NHANES (e.g., DMDMARTZ vs DMDMARTL). Asian-specific BMI thresholds differ from US categories and are handled by using continuous BMI as primary.

Variables collected in KCHS but absent in NHANES cannot enter the harmonized model; they are reserved for a secondary domestic (KCHS-only) model and sensitivity analyses (Table 2).
Table 2. KCHS-only variables excluded from the harmonized model.
KCHS variable (no NHANES equivalent)
Handling
Perceived stress (스트레스 인지)
Excluded from harmonized model; used only in secondary domestic (KCHS-only) model & sensitivity analysis
EQ-5D / EQ-VAS quality of life
Excluded from harmonized model (not collected in NHANES)
Social contact / social support items
Excluded from harmonized model; domestic model only
Subjective cognitive decline (older adults)
Excluded from harmonized model; domestic model only

[Heading2] 2.6 Missing data
Multiple imputation (or tree-native handling) is applied within the development data; the imputation and preprocessing pipeline learned in development is applied unchanged to each external set to prevent leakage. External-set information is never used to refit preprocessing.
[Heading2] 2.7 Sample size
Development power is not limiting given the KCHS size; the constraints are overfitting control and calibration stability. The number of depression events per NHANES cycle governs the precision of calibration and decision-curve estimates and is computed a priori to confirm adequacy for external evaluation.
[Heading2] 2.8 Model development
[ListParagraph] Algorithms: penalized logistic regression (elastic net; interpretable reference), random forest, and gradient boosting (XGBoost/LightGBM); optionally a shallow neural network.
[ListParagraph] Tuning: repeated stratified k-fold cross-validation for tuning; a separate hold-out for internal performance.
[ListParagraph] Imbalance: class weighting / focal loss / (if needed) SMOTE applied within training folds only; the operating threshold is chosen after calibration on clinical criteria.
[ListParagraph] Weights: the treatment of complex-survey weights in ML (weighted training vs unweighted training with weighted evaluation) is pre-specified and probed in sensitivity analyses.
[Heading2] 2.9 Performance evaluation
[ListParagraph] Discrimination: AUROC and AUPRC with 95% CIs.
[ListParagraph] Calibration: calibration plot, calibration slope and intercept, Brier score, expected calibration error—the central external-validation metrics.
[ListParagraph] Clinical utility: decision curve analysis (net benefit) across a clinically relevant threshold-probability range.
[ListParagraph] Operating characteristics: sensitivity, specificity, PPV, NPV at the selected threshold.
[Heading2] 2.10 External validation strategy
[ListParagraph] Geographic: the KCHS-trained model is applied to NHANES without refitting.
[ListParagraph] Temporal/COVID: evaluation is repeated separately in NHANES pre (2017–March 2020) and post (August 2021–August 2023).
[ListParagraph] Recalibration: if calibration degrades, intercept/slope recalibration (or Platt/temperature scaling) is applied and the recovery quantified—updating calibration only, without re-estimating predictor effects (a deployment-friendly update).
[Heading2] 2.11 Interpretability, fairness, and deployment
SHAP values summarize global and local predictor contributions, and the rank concordance of top predictors between KCHS and NHANES is examined as a mechanistic view of transportability. Fairness is assessed by reporting AUROC, calibration, and error rates across sex, age, education, income, and (in NHANES) race/ethnicity. For deployment, a parsimonious model using the top 5–10 predictors is derived to minimize item burden in a self-administered web/mobile form; a conceptual triage workflow routes low-risk users to self-management resources and higher-risk users to professional assessment. The tool is positioned as a screening/triage aid, not a standalone diagnostic or clinical-decision system.
[Heading2] 2.12 Sensitivity analyses and software
Sensitivity analyses cover the outcome cut-off (≥5/≥10), weighting approach, self-reported vs measured BMI, exclusion of prior-diagnosis respondents, the harmonized age bound, and alternative imputation. Analyses use Python (scikit-learn, xgboost, lightgbm, shap) and R (rms, dcurves, survey); code is released publicly.
[Heading2] 2.13 Ethics
Both KCHS and NHANES are public, de-identified secondary datasets. Institutional review board exemption or approval is obtained per local policy; data-use terms are followed and no re-identification is attempted.

[Heading1] 3. Results (Planned — to be populated)
Note: This is a protocol. The section below is a results template with pre-specified tables/figures and bracketed placeholders [ ]; no empirical results are reported.

[Heading2] 3.1 Participant flow and characteristics
The analytic samples for KCHS (development) and for the two NHANES periods will be reported with a flow diagram (Figure 1) documenting exclusions. Baseline characteristics and depression prevalence are summarized in Table 3.
Table 3. Sample characteristics (shell).
Characteristic
KCHS (development)
NHANES pre-COVID
NHANES post-COVID
N (analytic)
[ ]
[ ]
[ ]
Depression, PHQ-9 ≥10, n (%)
[ ]
[ ]
[ ]
Age, mean (SD)
[ ]
[ ]
[ ]
Female, n (%)
[ ]
[ ]
[ ]
Current smoker, n (%)
[ ]
[ ]
[ ]
Living alone, n (%)
[ ]
[ ]
[ ]
Poor self-rated health, n (%)
[ ]
[ ]
[ ]

[Heading2] 3.2 Model development and internal performance
Candidate algorithms will be compared by cross-validated AUROC/AUPRC and calibration; the primary model will be selected a priori by pre-specified criteria. Internal discrimination is expected in the AUROC ≈ [0.80–0.88] range based on prior survey-based models. Reported: internal AUROC [ ], AUPRC [ ], calibration slope [ ], Brier [ ].
[Heading2] 3.3 Geographic and temporal (COVID) external validation
Discrimination, calibration, and clinical utility in NHANES pre- and post-COVID cycles, before and after recalibration, are reported in Table 4, with calibration plots (Figure 2) and decision curves (Figure 3). We anticipate preserved discrimination with possible calibration drift that recalibration substantially corrects.
Table 4. Discrimination, calibration, and operating characteristics (shell).
Metric
Internal (KCHS)
NHANES pre
NHANES post
NHANES post (recalibrated)
AUROC (95% CI)
[ ]
[ ]
[ ]
[ ]
AUPRC
[ ]
[ ]
[ ]
[ ]
Calibration slope
[ ]
[ ]
[ ]
[ ]
Calibration intercept
[ ]
[ ]
[ ]
[ ]
Brier score
[ ]
[ ]
[ ]
[ ]
Sensitivity / Specificity
[ ] / [ ]
[ ] / [ ]
[ ] / [ ]
[ ] / [ ]

[Heading2] 3.4 Interpretability and fairness
Global SHAP importance and top-predictor rank concordance between surveys are reported (Figure 4). Subgroup discrimination/calibration/error rates by sex, age, education, income, and race/ethnicity are tabulated to characterize equity of performance.
[Heading2] 3.5 Parsimonious deployment model
Performance of the 5–10-item model versus the full model is reported to quantify the accuracy–burden trade-off relevant to a digital self-screening form.

[Heading1] 4. Discussion
[Heading2] 4.1 Principal findings (anticipated)
This study will provide, to our knowledge, one of the first evaluations of a depression-screening ML model that is developed in one country’s national survey and externally validated in another country’s survey using an identically defined PHQ-9 outcome, while simultaneously testing stability across the COVID-19 distribution shift. The design directly targets the transportability question that internal validation cannot answer, and it clarifies whether lightweight recalibration is sufficient to maintain clinical usefulness after a population-level shock.
[Heading2] 4.2 Comparison with prior work
Prior KCHS- and KNHANES-based ML depression models generally report internal validation only; cross-national external validation and explicit pre/post-COVID temporal validation are rare. By harmonizing predictors to the intersection of the two surveys and using an identical outcome, this protocol reduces the outcome- and label-shift confounding that complicates cross-dataset comparisons.
[Heading2] 4.3 Strengths
[ListParagraph] identical PHQ-9 outcome across development and validation, minimizing outcome harmonization error.
[ListParagraph] genuinely external validation across country, health system, race/ethnicity, and survey mode.
[ListParagraph] temporal validation across NCHS-defined pre/post-COVID periods with recalibration analysis.
[ListParagraph] interpretability, fairness, clinical-utility (DCA), and a deployment-oriented parsimonious model, aligned with digital-health translation.
[Heading2] 4.4 Limitations
[ListParagraph] cross-sectional data: the model screens for current symptoms rather than predicting future onset.
[ListParagraph] self-report and recall bias; differing measurement of alcohol, physical activity, income, and BMI across surveys.
[ListParagraph] residual population differences (race/ethnicity, health system) may limit transportability.
[ListParagraph] dependence on the KCHS years in which PHQ-9 is collected, which constrains development-sample size and calendar coverage.
[Heading2] 4.5 Conclusions
By quantifying cross-national and cross-pandemic transportability and by delivering a parsimonious, interpretable, and calibration-updatable model, this study will indicate whether a survey-derived depression screener can serve as a scalable digital triage tool and how it should be maintained when populations shift.

[Heading1] Appendix A. Pre-analysis confirmations
[ListParagraph] Confirm the exact KCHS survey years in which PHQ-9 is administered (year-specific user guide, Part 4).
[ListParagraph] Confirm year-specific KCHS raw variable codes for every Table 1 item.
[ListParagraph] Compute depression event counts per NHANES period to confirm calibration/DCA precision.
[ListParagraph] Finalize the common operational definitions for alcohol, physical activity, and income.
[ListParagraph] Confirm NHANES cycle-specific variable names (e.g., DMDMARTZ vs DMDMARTL; ALQ items).
[Heading1] Appendix B. Key references (verify formatting to JMIR style)
Kroenke K, Spitzer RL, Williams JBW. The PHQ-9: validity of a brief depression severity measure. J Gen Intern Med. 2001;16(9):606–613.
Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. 2024;385:e078378.
Moons KGM, Wolff RF, Riley RD, et al. PROBAST: assessing the risk of bias and applicability of prediction model studies. Ann Intern Med. 2019;170(1):W1–W33.
Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. Med Decis Making. 2006;26(6):565–574.
Lundberg SM, Lee S-I. A unified approach to interpreting model predictions (SHAP). Adv Neural Inf Process Syst. 2017;30.
National Center for Health Statistics. NHANES Mental Health – Depression Screener (DPQ) documentation. Centers for Disease Control and Prevention.
Korea Disease Control and Prevention Agency. Korea Community Health Survey raw-data user guide (지역사회건강조사 원시자료 이용지침서).

[Heading1] Appendix C. Data availability and access
Both datasets are de-identified and free of charge, but their access procedures differ: KCHS microdata are released only after an application and review, whereas NHANES data are openly downloadable. Raw KCHS microdata may not be redistributed; only approved users may obtain them.
[Heading2] C.1 Korea Community Health Survey (KCHS)
Provider / portal. Korea Disease Control and Prevention Agency (KDCA). Portal: https://chs.kdca.go.kr (지역사회건강조사). Microdata are provided under the KDCA regulation on raw-data release (질병관리청 예규 제124호) through a request-and-review process, not open download.
Procedure:
[ListParagraph] Create an account on the portal and agree to the data-use pledge (이용서약).
[ListParagraph] Submit a raw-data request (원시자료 자료요청) specifying the survey years, target variables, and research purpose.
[ListParagraph] KDCA reviews the request (심의; typically ~2–3 business days).
[ListParagraph] On approval, download the microdata (download link valid for 1 year from the request date).
Materials provided. Raw microdata plus the raw-data user guide (원시자료 이용지침서, including Part 4 variable descriptions) and the questionnaires (조사표) — these are needed to confirm year-specific variable codes and the exact years in which the PHQ-9 was administered. A separate open statistics/indicator viewer exists on the portal, but analytic microdata require the request above.
[Heading2] C.2 US National Health and Nutrition Examination Survey (NHANES)
Provider / portal. US National Center for Health Statistics (NCHS), Centers for Disease Control and Prevention. Portal: https://wwwn.cdc.gov/nchs/nhanes/. Data are open access with no application required.
Procedure:
[ListParagraph] Browse data by cycle and component: Demographics, Questionnaire, Examination, and Laboratory; each file has an online documentation/codebook.
[ListParagraph] Download the SAS transport (.XPT) files directly. For this study: pre-pandemic 2017–March 2020 uses the ‘P_’ prefix (e.g., P_DEMO, P_DPQ, P_SMQ, P_ALQ, P_PAQ, P_SLQ, P_BMX, P_HSQ, P_BPQ, P_DIQ); August 2021–August 2023 uses the ‘_L’ suffix (e.g., DEMO_L, DPQ_L, SMQ_L, ALQ_L, PAQ_L, SLQ_L, BMX_L, HSQ_L, BPQ_L, DIQ_L).
[ListParagraph] Merge components within a cycle by the respondent sequence number (SEQN), then apply the survey design variables (strata, PSU) and the appropriate MEC/interview weights following the NHANES Analytic Guidelines.
[Heading2] C.3 Reproducibility note
Because KCHS microdata are access-controlled and non-redistributable, we will release the analysis code and a variable-harmonization dictionary (not the raw KCHS data) so that approved users can reproduce the pipeline; NHANES files are openly available and can be re-obtained directly from NCHS. Exact URLs and access dates will be recorded in the final Data Availability statement.

===== STATS =====
paragraphs: 278
chars: 24941
tables: 6
