### Transportability-ceiling and calibration-interval analyses

Supports Results Testing Whether External Discrimination Can Be Raised Above ~0.77 (concept-shift ceiling) and Discussion Strengths and Limitations (development-survey coverage). Tables S2, S7, S11, S9 and S3 concern the ceiling; Table S3, placed last so that the appendix tables run in numerical order, gives the calibration intervals that support External Validation Across Country and Pandemic Period. Selection of the primary model, and of every strategy not explicitly marked as externally screened, used KCHS-internal criteria only. NHANES labels are used throughout for evaluation, as in any external validation; what is quarantined to the oracle rows is the use of those labels for model *fitting* (a deliberate departure from the external-validation rule set out under External Validation and Sensitivity Analyses, for diagnosis only). Weighted AUROC throughout.

### Interpretation
- The 17-predictor external ceiling (~0.77) held against every leakage-controlled attempt made, and the transported model already matches the best in-domain performance among the learners tested. Concept shift is the most consistent explanation, but it cannot be separated here from differences in how several predictors are measured, from development-survey coverage, or from complete-case selection; the single density-ratio correction applied shows only that the procedure did not help.
- Multimorbidity and healthcare utilization carry additional in-domain signal in NHANES (oracle 0.765→0.778). Table S16 is fitted and cross-validated entirely within pooled NHANES; this is therefore incremental discrimination inside that population rather than evidence that the signal would transport. The healthcare-visit item is carried as pre-cycle only, and its contribution cannot be separated from cycle differences. These domains are not collected by KCHS (only hypertension and diabetes are recorded as physician-diagnosed conditions across all survey years, both already included); hence those domains cannot enter a KCHS-developed transportable model. Part of the residual ceiling is therefore development-survey coverage, not concept shift alone.

*Multimorbidity count comprised physician-diagnosed asthma, arthritis, thyroid disorder, chronic obstructive pulmonary disease, liver disease, and cancer (circulatory conditions excluded per the pre-specified predictor policy; mental-health service contact excluded as tautological). Healthcare-visit count (HUQ051) is unavailable in the NHANES August 2021-2023 file and was carried as pre-cycle only.*

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

