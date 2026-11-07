## Supplementary Methods

This appendix carries the material needed to reproduce and interpret the analysis but too detailed for the main text: the year-specific KCHS-NHANES harmonization crosswalk, the predictor-set amendment, the study-design and model-development detail relocated from the Methods, and the supplementary results tabulated as Tables S4 to S3. Estimates are complex-survey weighted unless a row or column is labeled unweighted.

Supports the Methods sections Predictors and Harmonization, and Study Design and Data Sources. Section A documents how each harmonized variable was resolved in every KCHS survey year and in each NHANES cycle. Section B is an annotated English summary of the predictor-set amendment, with contextual notes added where marked; the original Korean document (dated 2026-07-15) is released unchanged with the code and is the authoritative version. The amendment was locked before any 17-predictor model was fitted, but after the 14-predictor model had been developed and transferred (see §B.5).

### Harmonization crosswalk
#### Outcome

| Concept | KCHS | NHANES pre (`P_`) | NHANES post (`_L`) | Harmonized definition |
|---|---|---|---|---|
| Current depression | `mtb_07a1`-`mtb_07i1` (9 items) | `DPQ010`-`DPQ090` | `DPQ010`-`DPQ090` | Nine items scored 0-3, summed; ≥10 = case. KCHS response codes 1-4 recoded to 0-3. Respondents missing any item excluded. |

#### The 17 harmonized predictors

KCHS variable names are not stable across survey years; the pipeline resolves each concept per year, first by variable name and then by a label regular expression, and writes the resolved name to a crosswalk log (`results/qc_crosswalk.txt`). Predictors 15-17 are the amendment additions. The locked document lists these three as food insecurity, employment and oral health; the present table reorders the entries for readability without changing the set.

| # | Concept | KCHS variable(s), by year | NHANES pre | NHANES post | Harmonized definition |
|---|---|---|---|---|---|
| 1 | Age | `age` (all years) | `RIDAGEYR` | `RIDAGEYR` | Continuous, top-coded at 80 y in every cohort; analytic samples restricted to ≥20 y |
| 2 | Sex | `sex` (all years) | `RIAGENDR` | `RIAGENDR` | Male / female |
| 3 | Education | `sob_01z1` (all years) | `DMDEDUC2` | `DMDEDUC2` | 4 levels (≤elementary, middle, high, college+) |
| 4 | Household income | `fma_24z1` (2017); `fma_14z1` (2018-2025) | `INDFMPIR` | `INDFMPIR` | Within-survey quintiles (rank-based, rendering the differing instruments comparable in relative terms) |
| 5 | Self-rated health | `qoa_01z1` (all years) | `HUQ010` | `HUQ010` | 5-level ordinal, 1 best - 5 worst |
| 6 | Partnered | `sod_02z2` (2017-2019); `sod_02z3` (2020-2025) | `DMDMARTZ` | `DMDMARTZ` | Married/cohabiting vs not |
| 7 | Living alone | `fma_01z1` (2017-2019); `mbhld_co` (2020-2025) | `DMDHHSIZ` (**absent from the pre-pandemic DEMO file**) | `DMDHHSIZ` | Household size = 1. Retained as missing in NHANES pre rather than imputed |
| 8 | Smoking | `sma_03z2` (current smoking status) | `SMQ020`, `SMQ040` | `SMQ020`, `SMQ040` | Never / former / current |
| 9 | Alcohol | `dra_01z1`; frequency `drb_01z2` (2017-2019), `drb_01z3` (2020-2025) | `ALQ111`, `ALQ121` | same | Current drinker (conservative common definition; `ALQ130` was extracted but not used in the harmonized definition) |
| 10 | Physical activity | `pha_04z1`/`pha_05z1`/`pha_06z1` (vigorous), `pha_07z1`/`pha_08z1`/`pha_09z1` (moderate), `phb_01z1` (walking); `pha_07z1` not fielded in 2018 (set to missing in the primary analysis; the module lacks the moderate-activity day count) | `PAQ650`/`PAQ655`/`PAD660` (vigorous), `PAQ665`/`PAQ670`/`PAD675` (moderate) | `PAD790Q`/`PAD790U`/`PAD800` (moderate), `PAD810Q`/`PAD810U`/`PAD820` (vigorous) | Meets aerobic recommendation (yes/no). Instruments differ between NHANES cycles; the binary recommendation threshold is the conservative common definition |
| 11 | Sleep duration | `mtc_01z1` (2017-2019); `mtc_17z1` (2020-2022, 2024, 2025); **not fielded in 2023** | `SLD012` | `SLD012` | Hours per night (continuous) |
| 12 | BMI | `oba_bmi` when present (2018, 2020, 2023, 2024); otherwise derived from `oba_02z1` (height) and `oba_03z1` (weight); **neither available in 2019** | `BMXBMI` | `BMXBMI` | Continuous. Self-reported in KCHS, measured in NHANES |
| 13 | Hypertension | `hya_04z1` (all years) | `BPQ020` | `BPQ020` | Physician-diagnosed, yes/no |
| 14 | Diabetes | `dia_04z1` (all years) | `DIQ010` | `DIQ010` | Physician-diagnosed, yes/no |
| **15** | **Employment** | `soa_01z1` (9/9 years) | `OCD150` (P_OCQ) | `OCD150` (OCQ_L) | Binary: KCHS 1 = employed; NHANES 1-2 (working, or has a job but not at work) = employed, 3-4 = not |
| **16** | **Self-rated oral health** | `ora_01z1` (9/9 years) | `OHQ845` (P_OHQ) | `OHQ845` (OHQ_L) | 5-level ordinal, 1 best - 5 worst (same construction as self-rated health) |
| **17** | **Food insecurity** | `nue_01z1` (7/9 years; **not fielded 2020-2021**) | `FSDAD` (P_FSQ) | `FSDAD` (FSQ_L) | Binary: KCHS categories 3-4 (sometimes/often insufficient) = insecure; NHANES adult food-security categories 3-4 (low/very low) = insecure. Single item vs 10-item module - the most weakly harmonized predictor |
|  | Race/ethnicity (fairness analysis only, never a predictor) | not collected | `RIDRETH3` | `RIDRETH3` | NHANES subgroup analysis only |

#### Complex-survey design variables

| Element | KCHS | NHANES pre | NHANES post |
|---|---|---|---|
| Weight | `wt` (2017-2018); `wt_p` (2019-2025) | `WTMECPRP` (P_DEMO) | `WTMEC2YR` (DEMO_L) |
| Stratum | `kstrata` | `SDMVSTRA` | `SDMVSTRA` |
| Primary sampling unit | `jijum_cd` (2017-2018); `spot_no` (2019-2021); `SPOT_NO` (2022-2025) | `SDMVPSU` | `SDMVPSU` |
| Design scale | 5,729 design strata; crossed with survey year as analyzed, 50,537 strata and 199,546 stratum-PSU combinations | 24 strata; 49 PSUs | 15 strata; 30 PSUs |

#### Analytic sample derivation

| Cohort | Source records | With complete PHQ-9, aged ≥20 y | Unweighted prevalence | Weighted prevalence |
|---|---|---|---|---|
| KCHS 2017-2025 | 2,071,211 (before the ≥20 y and complete-PHQ-9 filters) | **2,048,510** | 3.607% | 3.58% |
| NHANES pre-pandemic | 15,560 (all ages) → 9,232 adults ≥20 y | **7,888** | 9.229% | 8.45% |
| NHANES post-pandemic | 11,933 (all ages) → 7,809 adults ≥20 y | **5,210** | 12.687% | 12.20% |

Per-year KCHS counts and structural gaps are logged in `results/qc_crosswalk.txt`. Depression-module non-response in NHANES (three or more PHQ-9 items unanswered) was 7.4% pre and 13.3% post on the mobile examination center estimation frame (14.3% and 32.7% over all interviewed adults) and was differential by education and race/ethnicity; see Participant Characteristics in the Results and Strengths and Limitations in the Discussion.

#### Variables excluded as circular with the outcome

The following variables were pre-specified for exclusion, given that each re-measures the outcome construct rather than predicting it:

| Excluded | Reason |
|---|---|
| KCHS `mtb_07a1`-`mtb_07i1` | the PHQ-9 items constituting the outcome |
| KCHS `mtb_01z1` (depressed-mood experience) | a subset of the PHQ-9 construct |
| KCHS `mtb_02z1`, `mte_01z1` (mental-health consultation) | consequence/proxy of depression |
| KCHS `mtd_01z1`-`mtd_06z1` (suicidal ideation, attempt, counseling) | overlaps PHQ-9 item 9 |
| KCHS `qoc_05z1` (EQ-5D anxiety/depression) | directly measures depression |
| KCHS `qoc_07z1` (happiness index) | overlapping affective construct |
| KCHS `mta_01z1` (perceived stress) | outcome-proximal; also 100% missing in NHANES |
| NHANES `HUQ090` (saw a mental-health professional) | consequence/proxy; excluded on the external side as well |

---

### Predictor-set amendment (English rendition of the locked document)

Status: written 2026-07-15 and locked before any 17-predictor model was fitted. The original states that the document is not to be revised after seeing results. It was, however, written after the 14-predictor model had been developed and applied to NHANES: §B.5 below reproduces the expectations it records, which reference the then-known internal (0.824) and external (~0.75) performance.

#### Rationale

The original protocol pre-specified 14 harmonized predictors (protocol Table 1). The supervising investigator approved a re-selection of predictors; the investigators elected to keep the external-validation design unchanged (develop in KCHS, apply to NHANES without refitting) and to expand only the set of variables that both surveys share. The protocol sections covering the outcome definition, model development and the external-validation strategy were therefore left unchanged; only the predictor list was amended.

#### Eligibility criteria

Candidates were drawn from a crosswalk of all 616 KCHS concepts against the NHANES codebook for both cycles (a 175-variable extract of candidate items at the time of the amendment; a later, exhaustive extraction identified 265 variables common to both cycles). A candidate was eligible only if it was (i) available in both NHANES cycles, (ii) collected in more than one KCHS survey year, and (iii) not circular with the PHQ-9 outcome. These criteria use survey documentation and variable availability only; no NHANES outcome data were consulted.

#### Predictors added

Employment (`soa_01z1` ↔ `OCD150`), self-rated oral health (`ora_01z1` ↔ `OHQ845`), and household food insecurity (`nue_01z1` ↔ `FSDAD`) were the three concepts adopted. Harmonized definitions are given in the crosswalk above.

Candidates rejected at this stage, with reasons: additional chronic conditions in KCHS (dyslipidemia, osteoporosis, cataract, arthritis) are fielded in a single year only; stroke and myocardial infarction are recorded in KCHS as *awareness of early warning signs* rather than physician diagnosis and hence do not match the NHANES concept; functional limitation exists only in the NHANES post-pandemic cycle.

#### Analysis plan (unchanged from the protocol)

Outcome PHQ-9 ≥10 in both surveys, with ≥5 as a sensitivity analysis. Development in KCHS 2017-2025 with three algorithms (elastic net, random forest, LightGBM); the primary model selected on internal cross-validation only, by CV-AUROC, then |calibration slope - 1|, then parsimony; external performance is never used to choose a model. Preprocessing fitted in the development data and applied unchanged externally. Complex-survey-weighted evaluation with cluster bootstrap CIs. The KCHS-trained model applied to NHANES without refitting, evaluated separately in the pre- and post-pandemic cycles, with intercept/slope recalibration. The 17-predictor and the original 14-predictor models are reported side by side, internally and externally, so that the net effect of the expansion is visible.

#### Expectations declared in advance

The three additions were expected to be moderate-strength predictors, raising internal AUROC only slightly (roughly 0.824 → 0.83). External AUROC of 0.80 was explicitly not expected, since earlier work had already localized the external gap to Korea-US concept shift rather than to a shortage of predictors. The document states that if the expansion does not improve external performance, that result is to be reported as it stands, since an absence of improvement is itself evidence that the gap is a transportability problem rather than a predictor-count problem.

#### Deviation log

Deviations from the locked document and the parent protocol are reported here and, rule by rule, in Multimedia Appendix 1 Table S2. These comprise: the post-hoc candidate screening described below and reported under Predictors and Harmonization in the Methods; a parsimony margin reported at both the planned 0.02 and a stricter 0.01; an alcohol variable operationalized as current drinking alone rather than current plus at-risk drinking; the planned BMI self-report-versus-measured comparison, prior-diagnosis exclusion and alternative physical-activity definition, none of which the data support; the planned R components, which were not used; and public registration, which was not completed. After the primary analysis was complete, the remaining harmonizable-but-unused items (public-assistance receipt, alcohol frequency and quantity, binge drinking, weekend sleep duration) were screened by comparing the direction of the adjusted associations with depression across the three cohorts. This screening inspected NHANES outcome associations and thus lies outside the locked plan; it is reported under Predictors and Harmonization in the Methods as a post-hoc diagnostic. It did not change the predictor set: alcohol quantity, binge drinking and weekend sleep duration reversed direction between countries; alcohol frequency and public-assistance receipt were directionally consistent and added no external discrimination, public-assistance receipt leaving external AUROC unchanged (Table S11 in Multimedia Appendix 1). No item was added.
