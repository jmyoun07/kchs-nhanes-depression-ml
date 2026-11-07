# Cover Letter: Submission to the Journal of Medical Internet Research (JMIR)

*[Date] · To the Editor-in-Chief, Journal of Medical Internet Research*

Dear Editor,

Please consider the enclosed manuscript, "A Survey-Based Machine Learning Depression Screener Developed in Korea and Externally Validated in the United States Across the COVID-19 Period: Algorithm Development and Validation Study," for publication in JMIR.

**What the study does.** A machine-learning depression screener (PHQ-9 ≥10) was developed in the Korea Community Health Survey (n=2,048,510) and validated, without refitting, in the US National Health and Nutrition Examination Survey across National Center for Health Statistics-defined pre- and post-COVID-19 cycles. To the authors' knowledge this is among the first depression screeners tested across country, health system, and a pandemic distribution shift using an identically defined outcome in both surveys.

**Why it matters for JMIR's readership.** Most published survey-based depression models report internal validation only and, in Korean data, reach internal AUROCs near 0.90 while relying on predictors close to tautological with the outcome. This study asks instead the question that decides whether a digital screener is deployable: does it transport? Discrimination remained clinically useful but lower externally (AUROC 0.765 and 0.771 against 0.836 internally), calibration drifted yet was recovered by intercept-and-slope updating, and that external level withstood every leakage-controlled attempt to raise it, including matched sample size and covariate-shift correction. An in-domain oracle fitted on NHANES labels reached only 0.764-0.776, placing the limit in the transportable predictor set rather than in the modeling. A six-item version retains most of the external discrimination, supporting a low-burden web or mobile triage tool.

**Rigor and transparency.** Reporting follows TRIPOD+AI (2024), with an item-by-item mapping enclosed as Multimedia Appendix 2 and PROBAST risk-of-bias self-assessment. Three of the seventeen predictors entered through a dated protocol amendment, written while the study was under way and locked before any model using the amended set was fitted; the contribution of those three is reported as a robustness analysis rather than a primary result. Model fitting, tuning and algorithm selection used development-data cross-validation only, the in-domain oracle being a declared diagnostic exception. A data-integrity audit and subsequent peer review identified seven defects, all corrected before the primary analysis rather than carried forward as limitations.

**Declarations.** The manuscript is original, is not under consideration elsewhere, and all authors approve submission. The authors declare no competing interests. The corresponding author acknowledges the applicable article processing fee should the manuscript be accepted after peer review. The study analyzes de-identified secondary survey data; NHANES is openly released by the National Center for Health Statistics, while KCHS microdata are access-controlled and not redistributable. Generative AI tools assisted with code and drafting under full author verification, as stated in the manuscript.

Reviewers with expertise in clinical prediction modeling, survey epidemiology, or digital mental health would be well suited; no reviewers are excluded.

Thank you for your consideration.

Sincerely,
[Corresponding author name, degrees, affiliation, email: to complete]
on behalf of all authors
