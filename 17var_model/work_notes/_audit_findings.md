# 독립감사 확정 결함 50건 (기각 14건)

### [1] MAJOR — 17var_model/results/primary2_calibration_ci.csv
QUOTE: NHANES pre,after intercept/slope recalibration,0.998299,0.816666,1.187912,-0.000276,-0.124354,0.109213,0.009572,0.009653,0.022768

PROBLEM: The reported ECE point estimate for NHANES pre after recalibration is 0.009572 but the reported 95% lower bound is 0.009653 — the interval does not contain its own point estimate. This is not a typo: ECE is a non-negative, upward-biased statistic, and the Rao-Wu replicates (which zero out roughly half the clusters) inflate it systematically, so the percentile interval sits entirely above the point estimate. The same upward shift contaminates every ECE interval in the table, so the percentile method as applied is not a valid interval procedure for this statistic without bias correction. Table S16 prints the row as '0.010 (0.010–0.023)', which conceals the inversion only through rounding; the released CSV, which is the artifact reviewers will open, shows it plainly.

CORRECT: A 95% CI must contain the estimate it accompanies. The ECE intervals need a bias-corrected method (BCa, or bootstrap-bias-corrected/basic intervals), or the ECE column of Table S16 must be reported as a point estimate only, as Table 2 already does.

EVIDENCE: Recomputed the ECE bootstrap over the same 1000 replicates: NHANES pre after recalibration, point 0.009572, replicate mean 0.016209, 2.5th pct 0.009653; NHANES post after recalibration, point 0.014488, replicate mean 0.018787; NHANES pre as transported, point 0.030223, replicate mean 0.031654. All four replicate means exceed their point estimates. 1000/1000 replicates were valid in every case (no silent drops), so the inversion is not a filtering artifact.

---

### [2] MINOR — 17var_model/92_crossref_audit.py
QUOTE: S_CITED = set(int(x) for x in re.findall(r"Table S(\d+)", M + "".join(OTHERS.values())))

PROBLEM: The new mechanical cross-reference check cannot detect an uncited appendix table. S_DEFINED is built at line 41 from the headings `^#{2,3} Table S(\d+)\.` inside OTHERS; S_CITED is then built at line 42 from a corpus that INCLUDES those same OTHERS, so each table's own heading ('## Table S16. Calibration with cluster-bootstrap intervals (external cohorts)') matches the citation regex. S_DEFINED is therefore a subset of S_CITED by construction, and the guard at line 44 — `need(not (S_DEFINED - S_CITED), f"[전체] 인용되지 않은 부록 표: ...")` — can never fire. This is the check that was introduced to stop exactly this class of defect, and it gives false assurance: it prints '[2] 부록 표 S1–S16 정의·인용 일치' and exits 0.

CORRECT: Citations must be counted from a corpus with the definition headings stripped (or S_CITED built from `M` plus OTHERS minus the heading lines). Doing so shows Tables S3, S11 and S16 are cited nowhere in any document.

EVIDENCE: Ran the script: it passes with '상호참조 검사 통과'. Reimplemented the same check with heading lines removed by `re.sub(r"^#{2,3} Table S\d+\..*$", "", corpus, flags=re.M)`: TRUE cited = [1,2,4,5,6,7,8,9,10,12,13,14,15]; TRULY UNCITED = [3, 11, 16]. 92_crossref_audit.py lines 41-44.

---

### [3] MAJOR — 17var_model/docs/manuscript_17var.md
QUOTE: Ninety-five percent CIs are reported for AUROC and AUPRC, and, in the two external cohorts, for the calibration slope, CITL and ECE both as transported and after recalibration; the difference in external AUROC between the two cycles is also given an interval.

PROBLEM: The Methods promise ECE confidence intervals for both external cohorts, as transported and after recalibration. Those intervals exist only in Table S16, and Table S16 is never cited — the string 'Table S16' occurs exactly once in the entire document set, in its own heading at docs/appendix3_utility.md:226. The manuscript body never points the reader anywhere, so the promised ECE intervals are unreachable, and the new appendix table (which also carries the only appendix statement of the pre-vs-post AUROC difference) is orphaned. Every other appendix table the manuscript relies on is cited by number in the text (S4, S5, S7, S8, S9, S10).

CORRECT: Table S16 must be cited from the manuscript — at minimum from the Performance Evaluation sentence that promises the ECE intervals, and from the External Validation Results paragraph where the recalibrated slope/CITL intervals are first given.

EVIDENCE: grep 'Table S16' over docs/*.md returns a single hit, docs/appendix3_utility.md:226 ('## Table S16. Calibration with cluster-bootstrap intervals (external cohorts)'). Set difference of defined vs cited appendix tables, computed with heading lines stripped: {3, 11, 16} uncited.

---

### [4] MINOR — 17var_model/90_delta_and_calib_ci.py
QUOTE: pval = 2 * min((diff <= 0).mean(), (diff >= 0).mean())

PROBLEM: This is an uncentred, uncorrected doubled-tail percentile P: the bootstrap distribution is not shifted to the null, and there is no (1+·)/(B+1) finite-B correction, so the statistic can return exactly 0. The paper's other bootstrap contrast of the same kind uses a different estimator — 70_primary_corrected.py lines 325-326: `cen=ds-ds.mean()` then `pv=(1+np.sum(np.abs(cen)>=abs(obs)))/(len(ds)+1)` — which is null-centred and finite-B corrected. Both surface in the manuscript under the same unqualified label: Results and Discussion report 'bootstrap P=0.86' for the pre-vs-post difference, and the sensitivity paragraph reports '*P*=0.001 and 0.001; paired rescaled cluster bootstrap' for the 14-vs-17 contrast. Two mutually inconsistent P-value definitions are presented as one quantity, and the Methods never define either: the Performance Evaluation paragraph specifies the interval construction in full (2.5th/97.5th percentiles, n_h−1 PSU resampling, rescaling) but says nothing about how a P value is computed, and the strings 'P value', 'P-value' and 'two-sided' appear zero times in any document.

CORRECT: Use one definition for both contrasts (the null-centred, (1+·)/(B+1)-corrected form already implemented in 70_primary_corrected.py) and state it in the Performance Evaluation paragraph alongside the interval definition.

EVIDENCE: 90_delta_and_calib_ci.py:50 vs 70_primary_corrected.py:325-326. Manuscript lines 127 and 175 ('bootstrap P=0.86'), line 159 ('*P*=0.001 and 0.001'). `grep -i -E 'P value|P-value|two-sided|two sided' docs/*.md` returns nothing.

---

### [5] MINOR — 17var_model/90_delta_and_calib_ci.py
QUOTE:                    P=round(float(max(pval, 1 / K)), 4),

PROBLEM: The reported P floor is not the attainable resolution, and it is wrong by a factor of two even on the code's own terms. `pval` is two times a tail proportion computed out of K=20000 resampled pairs, so its smallest nonzero value is 2/K = 0.0001; when the tail count is zero the code substitutes 1/K = 0.00005, a value the statistic can never take. Separately, the honest resolution is governed by the B=1000 independent replicates drawn per cohort, not by the 20,000 pairs resampled from them — the pairs are a Monte Carlo subsample of the 1000x1000 convolution, not additional information. The defect is latent for this submission (the observed P is 0.8628, so the floor never binds) but the code as written would print P=0.0001 on a decisive contrast while resting on 1000 replicates per arm.

CORRECT: The floor should be 2/K if it is to describe the doubled-tail statistic, and the honest statement of resolution is the replicate count B=1000 per cohort. Better still: enumerate the exact 1000x1000 convolution, which is cheap and removes the Monte Carlo layer entirely.

EVIDENCE: 90_delta_and_calib_ci.py lines 47, 50, 53 (K=20000). Verified by recomputation that the exact 10^6-pair convolution gives CI [-0.044593, +0.056270] and P=0.8612 versus the K=20000 subsample's [-0.044569, +0.056146] and P=0.8628 — i.e. the subsampling contributes ~1e-4 of endpoint jitter and 0.0016 of P jitter that the exact enumeration would remove.

---

### [6] MAJOR — docs/manuscript_17var.md
QUOTE: Intervals are computed for AUROC and AUPRC; calibration, operating characteristics and net benefit are reported as point estimates, as stated under Performance Evaluation.

PROBLEM: This is the last sentence of Methods "Sample size" (line 105). It survives unedited from before the calibration intervals were computed, and it now (a) states a falsehood, (b) contradicts the rewritten Performance Evaluation paragraph, Table 2, the Abstract, the Discussion and Table S16, (c) contradicts its own paragraph two sentences earlier ("the achieved precision is reported as confidence intervals throughout"), and (d) falsely attributes the claim to another section — Performance Evaluation says the exact opposite. 91_add_intervals.py rewrote only the Performance Evaluation string ("Calibration measures and operating characteristics are point estimates.") and never touched this duplicate claim in the Sample-size paragraph.

CORRECT: Calibration slope, CITL and ECE all carry 95% bootstrap intervals in both external cohorts, as transported and after recalibration; only operating characteristics and net benefit are point estimates.

EVIDENCE: docs/manuscript_17var.md:105 (the quoted sentence) vs line 98 "Ninety-five percent CIs are reported for AUROC and AUPRC, and, in the two external cohorts, for the calibration slope, CITL and ECE both as transported and after recalibration"; vs Table 2 lines 140 and 142 ("| Calibration slope 95% CI | — | 0.53–0.77 | 0.82–1.19 | 0.60–0.76 | 0.88–1.13 |", "| CITL 95% CI | ..."); vs Abstract line 19 ("slope 1.00 (95% CI 0.82–1.19)"); vs Discussion lines 175 and 201; vs docs/appendix3_utility.md:230-235 (Table S16); vs results/primary2_calibration_ci.csv, which contains slope_lo/slope_hi, CITL_lo/CITL_hi and ECE_lo/ECE_hi for all four cohort×probability combinations. 91_add_intervals.py:30-31 shows only one of the two "point estimates" sentences was targeted.

---

### [7] MAJOR — docs/manuscript_17var.md
QUOTE: Supplementary methods and results: the year-specific KCHS↔NHANES harmonization crosswalk and the predictor-set amendment; clinical utility, sample characteristics and sensitivity analyses (Tables S1–S10); and the transportability-ceiling analyses (Tables S11–S15).

PROBLEM: The new Table S16 is an orphan. `grep -c "Table S16" docs/manuscript_17var.md` returns 0 — it is never cited anywhere in the main text — and the Multimedia Appendix 1 contents statement (line 231) still stops at S15. It is also out of sequence in the assembled appendix: 45_make_appendix_docx.py concatenates appendix2 → appendix3 → appendix4, and S16 sits at the end of appendix3 (after S10), so the reader meets S1–S10, then S16, then S11–S15. This matters substantively because Table S16 is the ONLY place the ECE confidence intervals appear, and Methods line 98 asserts that ECE CIs are reported (Table 2 has an ECE row with no CI row) — so that Methods claim has no locatable home in the reader-visible document. The new mechanical checker cannot catch this: 92_crossref_audit.py builds S_CITED from `M + "".join(OTHERS.values())`, so "## Table S16." is matched by its own heading regex `Table S(\d+)`, making the `S_DEFINED - S_CITED` test at line 44 vacuous for every table. Running it prints "부록 표 S1–S16 정의·인용 일치" and passes.

CORRECT: Table S16 must be cited in the main text (natural anchor: Table 2's footnote or the Results "External Validation Across Country and Pandemic Period" paragraph), listed in the Multimedia Appendix 1 contents, and renumbered/relocated so appendix table numbers run in document order.

EVIDENCE: grep -rn "S16" docs/ returns exactly one hit, docs/appendix3_utility.md:226 (the heading itself); docs/manuscript_17var.md:231 lists only S1–S10 and S11–S15; 45_make_appendix_docx.py:24 orders [appendix2, appendix3, appendix4]; docs/appendix3_utility.md headings run S1…S10 then S16 while docs/appendix4_ceiling.md:67 holds S15; 92_crossref_audit.py:42-44 (self-matching regex) and its passing run output.

---

### [8] MAJOR — docs/manuscript_17var.md
QUOTE: the recalibrated slope intervals reported here (0.82–1.19 pre and 0.88–1.13 post) are correspondingly wide; the update is therefore shown to remove the bias without being pinned down precisely.

PROBLEM: These intervals cannot support that claim, because the recalibration map is held fixed across the bootstrap. In 90_delta_and_calib_ci.py the recalibrated probability vector is built once on the full cohort (`pr = recal(y, p, w, "slope", groups=grp(d))`, line 61) and then resampled (line 63), so the 1000 replicates carry only evaluation-sample variability and none of the uncertainty in estimating the update's own intercept and slope — the very quantity the sentence claims to be characterising, and the quantity Riley et al [45] is cited for. The same defect makes the Discussion claim "the intervals are wide, as expected at these sample sizes, and both include 1" near-tautological: the update is fitted on the same cohort it is evaluated in, so slope≈1 by construction, and covering 1 is not evidence of anything. The manuscript applies the opposite (correct) standard to the oracle in the same paper and discloses it there, but discloses nothing here — neither Table 2's footnote ("CIs are from a rescaled stratified cluster bootstrap (1000 replicates)"), nor Methods line 98, nor Table S16's note says the update is held fixed.

CORRECT: Either re-estimate the recalibration map inside each bootstrap replicate (as was done for the oracle) and report the widened interval, or state explicitly that the recalibrated slope/CITL intervals condition on a fixed, once-estimated update and therefore understate the uncertainty of the update itself.

EVIDENCE: 90_delta_and_calib_ci.py:61-63 — `pr = recal(...)` is outside the replicate loop and `boot(y, q, R, cal_slope)` resamples the fixed `pr`; contrast docs/manuscript_17var.md:153, which for the oracle reports "widening to 0.745–0.794 when the oracle is refitted inside each bootstrap replicate so that model-fitting variability is included"; the undisclosed footnotes are docs/manuscript_17var.md:144, :98 and docs/appendix3_utility.md:228.

---

### [9] MINOR — docs/manuscript_17var.md
QUOTE: the difference in external AUROC between the two cycles is also given an interval.

PROBLEM: The post-hoc status of the between-cycle interval is disclosed in Results (line 127, "an interval computed after the fact, since no between-cycle test was pre-specified") and in Discussion Principal Findings (line 175), but not in the three other places the quantity is introduced or used: Methods Performance Evaluation (this sentence), the Abstract Results (line 19, "difference +0.006, 95% CI −0.045 to +0.056"), and the Table S16 note (docs/appendix3_utility.md:237). Methods is where pre-specification is established, and this sentence sits inside a paragraph otherwise describing the planned evaluation apparatus, so an unflagged mention reads as pre-specified. This also breaks the manuscript's own consistent convention of flagging post-hoc items inside Methods.

CORRECT: The between-cycle AUROC difference and its interval were not pre-specified; the manuscript's convention is to say so at the point of introduction in Methods.

EVIDENCE: docs/manuscript_17var.md:98 (quoted, no flag), :19 (Abstract, no flag), docs/appendix3_utility.md:237 (no flag), vs :127 and :175 where it is flagged. Convention established at :87 ("it is reported as a post-hoc diagnostic") and :103 ("is reported alongside it and flagged as post hoc"), both inside Methods.

---

### [10] MINOR — docs/manuscript_17var.md
QUOTE: the difference is +0.006 (95% CI −0.045 to +0.056; bootstrap P=0.86)

PROBLEM: A bootstrap P value is reported three times (Results line 127, Discussion line 175, appendix Table S16 note) but no Methods sentence anywhere states that a between-cycle test was performed or how the P value is defined. Methods line 98 says only that the difference "is also given an interval". The value is in fact a two-sided bootstrap tail probability, doubled and floored at 1/K, computed over 20,000 independently paired replicate draws — none of which the reader can reconstruct from the Methods.

CORRECT: P=0.8628 is 2 × min(Pr(diff ≤ 0), Pr(diff ≥ 0)) over 20,000 random pairs drawn independently from the two 1000-replicate AUROC distributions, floored at 1/20000; this definition should appear in Methods alongside the interval.

EVIDENCE: 90_delta_and_calib_ci.py:46-53 (`K = 20000`; `pval = 2 * min((diff <= 0).mean(), (diff >= 0).mean())`; `P=round(float(max(pval, 1 / K)), 4)`); results/primary2_delta_auroc.csv gives P=0.8628, pairs=20000; grep of docs/manuscript_17var.md finds "bootstrap P" only at lines 127 and 175 and no definition in Methods.

---

### [11] MINOR — docs/cover_letter.md
QUOTE: calibration drifts yet is recoverable, on point estimates, by lightweight intercept/slope recalibration

PROBLEM: The hedge "on point estimates" was inserted in an earlier round precisely because no calibration interval existed. It was never updated when the calibration CIs were computed, so the cover letter now tells the editor the calibration claim rests on point estimates only, while the Abstract, Table 2, Table S16 and the Discussion all report 95% CIs for the recalibrated slope and CITL. Since the cover letter is the first thing a JMIR editor reads, this understates the submission's own evidence.

CORRECT: Recalibration restored the calibration slope to 1.00 (95% CI 0.82–1.19) pre-pandemic and 1.00 (0.88–1.13) post-pandemic, with CITL 0.00 (−0.12 to +0.11 and −0.13 to +0.12); the "on point estimates" qualifier is obsolete.

EVIDENCE: docs/cover_letter.md:11 (quoted) vs docs/manuscript_17var.md:19, :140, :142, :175, :201 and docs/appendix3_utility.md:233,235; source values in results/primary2_calibration_ci.csv rows "after intercept/slope recalibration" (slope_lo 0.816666 / slope_hi 1.187912; 0.8779 / 1.126015).

---

### [12] MAJOR — docs/manuscript_17var.md
QUOTE: Intervals are computed for AUROC and AUPRC; calibration, operating characteristics and net benefit are reported as point estimates, as stated under Performance Evaluation.

PROBLEM: Direct internal contradiction and a false description of the paper's own Table 2. The Sample size paragraph (line 105) says calibration is reported as point estimates and attributes that to Performance Evaluation, but Performance Evaluation (line 98) says the opposite, and Table 2 carries two explicit CI rows ("Calibration slope 95% CI", "CITL 95% CI"). The Abstract, Results and Discussion all quote those calibration intervals. This is a leftover sentence from before the interval work was added and was never reconciled.

CORRECT: Intervals are computed for AUROC, AUPRC and, in the two external cohorts, for the calibration slope, CITL and ECE; only operating characteristics and net benefit are point estimates.

EVIDENCE: docs/manuscript_17var.md line 105 vs line 98 ("Ninety-five percent CIs are reported for AUROC and AUPRC, and, in the two external cohorts, for the calibration slope, CITL and ECE both as transported and after recalibration") and Table 2 rows at lines 140 and 142 ("| Calibration slope 95% CI | — | 0.53–0.77 | 0.82–1.19 | 0.60–0.76 | 0.88–1.13 |"; "| CITL 95% CI | — | +0.37–+0.64 | −0.12–+0.11 | +0.98–+1.26 | −0.13–+0.12 |"). Values verified against results/primary2_calibration_ci.csv.

---

### [13] MINOR — docs/manuscript_17var.md
QUOTE: Ninety-five percent CIs are reported for AUROC and AUPRC, and, in the two external cohorts, for the calibration slope, CITL and ECE both as transported and after recalibration

PROBLEM: The ECE intervals promised in Methods appear nowhere in the manuscript. Table 2 has an ECE row (0.003 / 0.030 / 0.010 / 0.068 / 0.014) but no ECE CI row, unlike the calibration slope and CITL, which each got a CI row. The reader is told an interval exists and cannot find it.

CORRECT: ECE intervals were computed (pre as-transported 0.030, 95% CI 0.025–0.039; pre recalibrated 0.010, 0.010–0.023; post as-transported 0.068, 0.056–0.080; post recalibrated 0.014, 0.011–0.028) but exist only in the uncited Table S16; either add an ECE CI row to Table 2 with a pointer, or drop ECE from the Methods sentence.

EVIDENCE: results/primary2_calibration_ci.csv columns ECE_lo/ECE_hi; docs/manuscript_17var.md Table 2 (lines 131-142) contains rows "| ECE |" at line 138 with no accompanying CI row; grep of the manuscript for "S16" returns 0 hits.

---

### [14] MAJOR — docs/manuscript_17var.md
QUOTE: Supplementary methods and results: the year-specific KCHS↔NHANES harmonization crosswalk and the predictor-set amendment; clinical utility, sample characteristics and sensitivity analyses (Tables S1–S10); and the transportability-ceiling analyses (Tables S11–S15).

PROBLEM: Table S16 exists in Multimedia Appendix 1 (docs/appendix3_utility.md line 226, "Table S16. Calibration with cluster-bootstrap intervals (external cohorts)") but is missing from this inventory and is never cited anywhere in the manuscript or any other document. It is a wholly orphaned supplementary table, which JMIR production will flag. The mechanical checker 92_crossref_audit.py passes it because its citation regex `Table S(\d+)` matches the table's own heading inside the appendix, so a table always counts as citing itself and an orphan can never be detected.

CORRECT: Multimedia Appendix 1 contains Tables S1–S16; Table S16 must be listed in the inventory and cited from the text (the natural anchor is the Performance Evaluation sentence promising calibration/ECE intervals, or the External Validation Results paragraph).

EVIDENCE: docs/appendix3_utility.md line 226 defines Table S16; `python -c` scan shows 0 occurrences of "S16" in docs/manuscript_17var.md, appendix2_crosswalk.md, appendix4_ceiling.md, TRIPOD_AI_checklist.md, model_card_17var.md and cover_letter.md; 92_crossref_audit.py line with `S_CITED = set(... re.findall(r"Table S(\d+)", M + "".join(OTHERS.values())))` searches the appendices themselves, so self-headings satisfy the check and the script prints "부록 표 S1–S16 정의·인용 일치".

---

### [15] MINOR — docs/manuscript_17var.md
QUOTE: intercept-plus-slope updating restored both without re-estimating any predictor effect, to slope 1.00 (0.82–1.19) and 1.00 (0.88–1.13); the intervals are wide, as expected at these sample sizes, and both include 1.

PROBLEM: The recalibrated calibration-slope and CITL intervals do not propagate the uncertainty of the recalibration map itself. In 90_delta_and_calib_ci.py the recalibrated probabilities are computed once on the full cohort and then held fixed while the bootstrap resamples PSUs, so every replicate re-uses a map that was fitted to make the full-sample slope equal 1. The statement "both include 1" is therefore close to guaranteed by construction and carries little evidential weight, and the intervals are narrower than a correct nested bootstrap would give. The paper demonstrates it knows the distinction: for the oracle it explicitly reports the widening from refitting inside each replicate (0.753–0.799 to 0.745–0.794), but the same treatment was not applied here.

CORRECT: The recalibration map must be re-estimated inside each bootstrap replicate (as was done for the oracle), or the intervals must be labelled as conditional on the full-sample recalibration map and the phrase "both include 1" removed as non-informative.

EVIDENCE: 90_delta_and_calib_ci.py line 61 `pr = recal(y, p, w, "slope", groups=grp(d))` sits outside the replicate loop; lines 62-70 then call `boot(y, q, R, cal_slope)` on the fixed `pr`. Contrast docs/manuscript_17var.md line 153, which reports the nested version for the oracle: "widening to 0.745–0.794 when the oracle is refitted inside each bootstrap replicate so that model-fitting variability is included".

---

### [16] MAJOR — docs/manuscript_17var.md
QUOTE: covariate-shift correction by density-ratio reweighting of the KCHS training sample toward the NHANES covariate distribution (labels unused) *reduced* external AUROC (to 0.766/0.758) ... public-assistance receipt alone left external AUROC unchanged (0.772/0.769), weekend sleep duration moved it by 0.001 (0.771/0.769)

PROBLEM: These absolute AUROCs come from the superseded pre-correction exploration pipeline, whose baseline is 0.772 pre / 0.769 post, not from the primary analysis, whose external values are 0.765 pre / 0.771 post (Table 2). Read against Table 2, the text says the covariate-shift correction "reduced" external AUROC to a pre-pandemic value (0.766) higher than the primary estimate (0.765), and says adding public assistance left AUROC "unchanged" at 0.772, i.e. 0.007 above the primary pre-pandemic value. The exploration-run provenance is disclosed in Multimedia Appendix 1 (Table S11 heading "LightGBM baseline (17 predictors, exploration run)"; Table S13 "its external values (0.7694/0.7685) therefore differ from the primary model's 0.765/0.771") but nowhere in the manuscript body, and the qualifier "in the exploration run" is attached only to the last of the three clauses.

CORRECT: These runs were executed on the pre-correction pipeline against an in-run baseline of 0.772 pre / 0.769 post; either report them as deltas from that in-run baseline, or state explicitly in the manuscript that the whole clause is on the exploration-run scale and is not comparable with Table 2.

EVIDENCE: results/ceiling_experiments.csv row "Baseline 17-predictor transfer (LightGBM, KCHS 0.5M)" = 0.772/0.769; rows "+ welfare (18)" = 0.772/0.769, "+ welfare + sleep_weekend (19)" = 0.771/0.769, "Covariate-shift reweight (clip 3)" = 0.766/0.758. results/primary2_metrics.csv gives the primary external values 0.765028 / 0.770591. docs/appendix4_ceiling.md Table S13 states the exploration reference "should not be read against the primary estimates"; docs/manuscript_17var.md line 153 carries no such caveat.

---

### [17] MINOR — docs/manuscript_17var.md
QUOTE: changed discrimination by at most 0.003 externally (pre 0.769–0.772, post 0.769–0.770)

PROBLEM: The alternative-alcohol-definition ranges are absolute AUROCs on the uncorrected baseline (0.7718 pre / 0.7688 post), not on the primary analysis (0.765 pre / 0.771 post in Table 2). Presented in the Sensitivity Analyses paragraph next to the primary numbers, and with no "on that baseline" qualifier (which the manuscript does supply two sentences later for the income quintile), the range 0.769-0.772 reads as showing that every alternative alcohol definition beats the primary specification pre-pandemically. The Δ≤0.003 statement is correct; only the absolute anchors are on the wrong scale.

CORRECT: The alcohol alternatives were run against the uncorrected baseline of 0.7718 pre / 0.7688 post; report them as deltas, or add the same "on that baseline" qualifier used for the age top-code and income-quintile sentences.

EVIDENCE: results/sap_alcohol_alt.csv: baseline (현재 음주 여부) 0.8315/0.7718/0.7688; alternatives 0.7711/0.7703, 0.7687/0.7704, 0.7687/0.7703. results/harmonization_sensitivity.csv row "Baseline (as analysed)" = 0.83154/0.7718/0.76883 confirms this is the pre-correction pipeline. results/primary2_metrics.csv gives the primary 0.765028/0.770591.

---

### [18] MINOR — docs/manuscript_17var.md
QUOTE: Applying the margin set in the analysis plan (the smallest set within 0.02 AUROC of the full model, searched over every count below 17 without imposing the plan's preferred 5–10 item range) selects six items

PROBLEM: The six-item selection clears the pre-specified 0.02 margin by 0.00024 AUROC. The full 17-item selection-set AUROC is 0.833235, giving a cutoff of 0.813235; the five-item model scores 0.812995, missing the cutoff by 0.00024. A trivially different seed, fold assignment or fitting/selection cluster split would have selected five items, and the deployment model that anchors H4, the abstract and the Conclusions would be different. The manuscript reports the trajectory to three decimals ("5 items 0.813; 6 items 0.818"), at which resolution the margin is invisible, and never discloses the fragility.

CORRECT: The k=5 selection-set AUROC is 0.812995 against a cutoff of 0.813235; the decision between a five- and six-item deployment model turns on 0.00024 AUROC and should be reported as such.

EVIDENCE: results/primary2_parsimonious.csv: k=5 selection_AUROC 0.812995, k=6 0.817610, k=17 0.833235; 0.833235 − 0.02 = 0.813235. Rule implemented at 70_primary_corrected.py: `cand=pp[(pp.k<17)&(pp.selection_AUROC>=fullsel-0.02)]; kstar=int(cand.k.min())`.

---

### [19] MINOR — docs/manuscript_17var.md
QUOTE: 14.3% of the pre-pandemic and 32.7% of the post-pandemic respondents left three or more PHQ-9 items unanswered, yielding the complete-case analytic samples of 7,888 and 5,210

PROBLEM: The stated exclusion does not produce the stated sample sizes. Excluding only respondents with three or more missing items leaves about 7,910 pre and 5,258 post; the reported analytic samples of 7,888 and 5,210 additionally exclude the 0.2% and 0.6% with one or two items missing, consistent with the Participants criterion "Adults ≥20 years with all nine PHQ-9 items answered" and with the very next sentence, which speaks of "adding respondents missing only one or two items". The "yielding" link is therefore wrong.

CORRECT: 14.3% pre and 32.7% post left three or more items unanswered; the analytic samples of 7,888 and 5,210 follow from excluding any missing item (14.6% pre and 33.3% post of the 9,232 and 7,809 adults ≥20 years).

EVIDENCE: docs/appendix3_utility.md Table S8: adults ≥20 y = 9,232 pre / 7,809 post; ≥3 missing 14.3%/32.7%; 1–2 missing 0.2%/0.6%; analytic 7,888 (85.4%) / 5,210 (66.7%). Recomputed from results/primary2_dpq_nonresponse.csv race rows: pre Σn=9,232, Σ(n×nonresponse_ge3)=1,322 → 7,910, Σ(n×any_missing)=1,344 → 7,888; post Σn=7,809, ge3 → 5,258, any_missing → 5,211.

---

### [20] MINOR — docs/manuscript_17var.md
QUOTE: the difference is +0.006 (95% CI −0.045 to +0.056; bootstrap P=0.86)

PROBLEM: Two reporting problems with the bootstrap P values. (1) The Methods describe interval estimation only ("the difference in external AUROC between the two cycles is also given an interval") and never specify a hypothesis-testing procedure, yet the Results report bootstrap P values here and again for the 14-vs-17 comparison. (2) The P=0.001 values reported for the 14-vs-17 gain are the floor of the procedure, not estimates: with 1000 replicates the smallest attainable value is (1+999)/(999+1)... in practice 1/(len(ds)+1) ≈ 0.001, so "P=0.001 and 0.001" should be "P<.001".

CORRECT: Add the bootstrap P-value definition to Performance Evaluation (two-sided proportion of centred replicate differences at least as extreme as observed), and report the gain comparison as P<.001 rather than P=0.001.

EVIDENCE: docs/manuscript_17var.md line 98 Methods mentions only intervals; line 159 reports "(*P*=0.001 and 0.001; paired rescaled cluster bootstrap...)". 70_primary_corrected.py line ~326: `pv=(1+np.sum(np.abs(cen)>=abs(obs)))/(len(ds)+1)` with ~1000 replicates, so the minimum is ≈0.001; results/primary2_gain.csv records P=0.001 for both cohorts. 90_delta_and_calib_ci.py line 53 similarly floors at `max(pval, 1/K)`.

---

### [21] MINOR — docs/manuscript_17var.md
QUOTE: Removing food insecurity alone leaves most of that difference intact.

PROBLEM: "Most" is not supported for the post-pandemic cohort. Removing food insecurity alone accounts for 0.0065 of the 0.0131 total 17-vs-14 external gap post-pandemically, i.e. exactly half, and 0.0055 of 0.0200 pre-pandemically. Whichever way the sentence is read (food insecurity reproduces most of the gap, or the gap mostly survives its removal), the post-pandemic split is 50/50 and "most" overstates it.

CORRECT: Removing food insecurity alone accounts for about a quarter of the pre-pandemic gap and about half of the post-pandemic gap; the remainder is carried by employment and self-rated oral health.

EVIDENCE: results/primary2_sensitivity.csv: primary 0.765028/0.770591; "Food insecurity excluded (16)" 0.759495/0.764047 (Δ 0.00553 / 0.00654); "Three added predictors excluded (14)" 0.745047/0.757501 (Δ 0.01998 / 0.01309). 0.00553/0.01998 = 27.7%; 0.00654/0.01309 = 50.0%.

---

### [22] MINOR — docs/manuscript_17var.md
QUOTE: lowers external discrimination to 0.745 (pre) and 0.757 (post) when each model is given its own hyperparameter search, a difference of Δ+0.020, 95% CI +0.010 to +0.030 and Δ+0.014, 95% CI +0.006 to +0.020 (*P*=0.001 and 0.001; paired rescaled cluster bootstrap, each model given its own hyperparameter search)

PROBLEM: The clause "each model is given its own hyperparameter search" appears twice inside the same sentence, once in the main clause and once in the parenthesis.

CORRECT: Delete one of the two occurrences; the parenthesis should read "(P<.001 for both; paired rescaled cluster bootstrap)".

EVIDENCE: docs/manuscript_17var.md line 159, verbatim.

---

### [23] MINOR — docs/manuscript_17var.md
QUOTE: the 500,000-record training set contains approximately 18,000 events (about 1,000 events per candidate predictor) and the 100,000-record hold-out approximately 3,600 events

PROBLEM: The internal hold-out is given as 100,000 records here but as 100,001 in the Abstract, in Data sources, in Sample size and in Limitations.

CORRECT: The hold-out contains 100,001 records (results/primary2_design.csv holdout_n = 100001).

EVIDENCE: results/primary2_design.csv: train_n 500000, holdout_n 100001. docs/manuscript_17var.md lines 17, 48, 105 and 201 all use "100,001"; line 94 uses "100,000".

---

### [24] MINOR — docs/manuscript_17var.md
QUOTE: The predictors carrying most of the signal are therefore the general-health, demographic and sleep items that both surveys measure identically

PROBLEM: "Measure identically" is contradicted by the paper's own Harmonization cautions. Sleep duration is one of the top four predictors and required an explicit definitional realignment (KCHS is weekday-only in 2017–2019 but a weekday-weighted seven-day average from 2020, corrected to the NHANES weekday basis) and sits on different admissible ranges (NHANES 2–14 hours vs KCHS 0–23). Age, also in the top group, required top-coding at 80 to match NHANES.

CORRECT: The high-importance predictors are those harmonizable to a common definition with the least residual instrument difference, not items the two surveys measure identically; sleep duration and age both required alignment before fitting.

EVIDENCE: docs/manuscript_17var.md line 89 (Harmonization cautions): "sleep duration is a weekday-only figure in 2017–2019 but a weekday-weighted seven-day average from 2020, whereas NHANES asks about weekdays" and "NHANES sleep duration is bounded at 2–14 hours while KCHS admits 0–23"; results/primary2_shap.csv ranks sleep_hrs 4th (0.2610) and age 2nd (0.3734).

---

### [25] MAJOR — docs/manuscript_17var.md
QUOTE: Intervals are computed for AUROC and AUPRC; calibration, operating characteristics and net benefit are reported as point estimates, as stated under Performance Evaluation.

PROBLEM: The Sample size paragraph (line 105) is a leftover from before the calibration intervals existed. It states that calibration is reported as point estimates and attributes that statement to Performance Evaluation, which now says the opposite. It also contradicts Table 2, which carries two calibration-interval rows, and Table S16, which tabulates the same intervals. 91_add_intervals.py updated the Abstract, Performance Evaluation, Results, Discussion and Limitations but never touched this sentence.

CORRECT: 95% CIs are reported for AUROC, AUPRC and, in both external cohorts, for the calibration slope, CITL and ECE, as transported and after recalibration; only operating characteristics and net benefit remain point estimates.

EVIDENCE: docs/manuscript_17var.md line 98 (Performance evaluation): "Ninety-five percent CIs are reported for AUROC and AUPRC, and, in the two external cohorts, for the calibration slope, CITL and ECE both as transported and after recalibration ... Operating characteristics and net benefit are point estimates." Table 2 (lines 140, 142) has rows "| Calibration slope 95% CI | — | 0.53–0.77 | 0.82–1.19 | 0.60–0.76 | 0.88–1.13 |" and "| CITL 95% CI | — | +0.37–+0.64 | −0.12–+0.11 | +0.98–+1.26 | −0.13–+0.12 |". 91_add_intervals.py MS list replaces only "Calibration measures and operating characteristics are point estimates."

---

### [26] MAJOR — docs/manuscript_17var.md
QUOTE: Supplementary methods and results: the year-specific KCHS↔NHANES harmonization crosswalk and the predictor-set amendment; clinical utility, sample characteristics and sensitivity analyses (Tables S1–S10); and the transportability-ceiling analyses (Tables S11–S15).

PROBLEM: Table S16 is defined in docs/appendix3_utility.md but is never cited anywhere in the manuscript, and the Multimedia Appendix 1 contents statement (line 231) enumerates only S1–S10 and S11–S15, so the reader is told the supplement stops at S15. The roadmap paragraph at the head of Part B (appendix3_utility.md line 3) likewise lists what Part B supports and does not mention Table S16 or the External Validation Across Country and Pandemic Period section it belongs to. The new mechanical check cannot catch this: 92_crossref_audit.py builds S_CITED from `M + "".join(OTHERS.values())`, so each "## Table SN." heading matches its own `Table S(\d+)` citation pattern and the uncited-table assertion can never fail (it prints "부록 표 S1–S16 정의·인용 일치").

CORRECT: The manuscript contains zero occurrences of "Table S16"; the table exists at docs/appendix3_utility.md line 226 and carries the calibration intervals that Performance Evaluation and Table 2 rely on (including the ECE intervals, which appear nowhere else).

EVIDENCE: `grep -c "Table S16" docs/manuscript_17var.md` returns 0; `grep -n "Table S"` on the manuscript yields only S4, S5, S6, S7, S8, S9, S10 and "Tables S1, S2 and S3". 92_crossref_audit.py lines 41-44.

---

### [27] MINOR — docs/appendix3_utility.md
QUOTE: ## Table S16. Calibration with cluster-bootstrap intervals (external cohorts)

PROBLEM: Table S16 was appended to the end of Part B (after Table S10), but Tables S11–S15 live in Part C (appendix4_ceiling.md), which is concatenated after Part B when Multimedia Appendix 1 is assembled. The supplement therefore presents its tables in the order S1…S10, S16, S11…S15. The table is also thematically misfiled: Part B is declared as "Clinical utility, sample characteristics, and sensitivity analyses", while S16 is an external-validation calibration table. It should have been placed with the calibration material and the later tables renumbered, or numbered to follow its position.

CORRECT: In the generated submission file the paragraph order is Table S10 → Table S16 → Table S11 … Table S15, i.e. the supplementary numbering is out of sequence.

EVIDENCE: 45_make_appendix_docx.py JOBS concatenates appendix2_crosswalk.md, appendix3_utility.md, appendix4_ceiling.md into 부록1_보충방법및결과.docx. Extracting paragraph text from that docx gives heading order at paragraph indices 1161 (Table S10), 1219 (Table S16), 1249 (Table S11), 1304 (S12), 1346 (S13), 1378 (S14), 1396 (S15). 91_add_intervals.py writes the table with `s.rstrip() + "\n\n" + "\n".join(tab)`, i.e. plain append.

---

### [28] MINOR — 90_delta_and_calib_ci.py
QUOTE: Intervals are the 2.5th and 97.5th percentiles of 1000 rescaled stratified cluster-bootstrap replicates, the same replicates used for discrimination. Recalibration is intercept/slope updating applied out-of-sample by 5-fold cross-fitting grouped by primary sampling unit.

PROBLEM: The "same replicates" claim itself checks out (90 re-executes the prefix of 70_primary_corrected.py, and reps_for reseeds np.random.RandomState(SEED) on every call, so the replicate weight vectors are byte-identical to those used for the Table 2 AUROC CIs). What is not disclosed is that the recalibration map is estimated once on the full external cohort and then held fixed across all 1000 replicates: `pr = recal(y, p, w, "slope", groups=grp(d))` sits outside the replicate loop, and `boot(y, q, R, cal_slope)` merely reweights those fixed out-of-fold probabilities. The "after intercept/slope recalibration" intervals in Table S16, in Table 2 and in the Abstract therefore condition on one estimated calibration map and do not propagate its estimation uncertainty. Neither Table S16's note nor Table 2's note says so, and the paper is explicit about exactly this distinction elsewhere.

CORRECT: The intervals for the recalibrated slope (0.82–1.19 pre, 0.88–1.13 post), recalibrated CITL and recalibrated ECE are conditional on a recalibration map fitted once on the full cohort; the cross-fitting is not redone inside the bootstrap.

EVIDENCE: 90_delta_and_calib_ci.py lines 59-63: the `pr = recal(...)` assignment precedes `for lab, q in [("as transported", p), ("after intercept/slope recalibration", pr)]`, so `pr` is constant across the 1000 replicates. Contrast docs/appendix4_ceiling.md Table S15 note: "The final row refits the oracle inside each replicate so that model-fitting variability is included" (and the resulting interval widens from 0.753–0.799 to 0.745–0.794).

---

### [29] MAJOR — docs/model_card_17var.md
QUOTE: the model was then fitted on 500,000 records drawn at random from the internal metrics computed on that hold-out.

PROBLEM: The sentence is garbled and, read literally, says the 500,000 training records were drawn from the hold-out — the opposite of the cluster-level split the paper's leakage argument depends on. A blind string replacement mangled it: 85_review10_fixes.py wrote "...drawn at random from the remaining clusters", and 86_review11_fixes.py line 98 then applied the pair ("remaining clusters used for internal metrics", "internal metrics computed on that hold-out").

CORRECT: The hold-out clusters are removed first, and the 500,000 training records are drawn at random from the remaining (non-hold-out) clusters; internal metrics are computed on the 100,001-record hold-out.

EVIDENCE: 70_primary_corrected.py lines 86-90: `ho_cl = set(order[:n_ho].tolist()); is_ho = np.isin(cl, list(ho_cl)); ho = np.where(is_ho)[0]; tr_pool = np.where(~is_ho)[0]; tr = rng.permutation(tr_pool)[:500000]`, with `shared` printed as 0 and results/primary2_design.csv recording shared_clusters=0. 86_review11_fixes.py line 98.

---

### [30] MINOR — docs/appendix3_utility.md
QUOTE: External AUROC differed between cycles by +0.006 (95% CI −0.045 to +0.056; bootstrap P=0.86), formed by differencing independently drawn replicate statistics from the two cohorts, the cycles being disjoint samples.

PROBLEM: This paragraph sits immediately after the Table S16 body, and the docx converter treats any paragraph following a table as that table's note: 45_make_appendix_docx.py's collect_notes()/letter_notes() turn it into footnote "a" and stamp the superscript anchor onto the first header cell. In the submitted appendix the calibration table's "Cohort" header therefore carries a footnote marker whose text is about discrimination, not calibration. The numbers themselves are correct, but the delta result appears in the supplement only as a mislabelled footnote to a calibration table, with no table or heading of its own.

CORRECT: In 부록1_보충방법및결과.docx the Table S16 header cell renders as "Cohorta" and the corresponding note paragraph is "aExternal AUROC differed between cycles by +0.006 ..."; the delta values match results/primary2_delta_auroc.csv (delta 0.005563, CI −0.044569 to 0.056146, P 0.8628).

EVIDENCE: 45_make_appendix_docx.py lines 57-90 (collect_notes, letter_notes, anchor on first non-empty header cell); extracted docx paragraph 1246 = "aExternal AUROC differed between cycles by +0.006 …" and the Table S16 header row first cell text = "Cohorta".

---

### [31] MINOR — docs/appendix3_utility.md
QUOTE: | NHANES pre | after intercept/slope recalibration | 1.00 (0.82–1.19) | 0.00 (−0.12–+0.11) | 0.010 (0.010–0.023) |

PROBLEM: The reported ECE point estimate lies below the lower limit of its own interval; three-decimal rounding conceals it. This is a symptom of percentile-bootstrapping a non-negative absolute-deviation statistic: the replicate ECE distribution is shifted upward relative to the full-sample value, so the interval is not centred on the estimate in any of the four rows (e.g. post recalibrated 0.014 with 0.011–0.028). A reviewer who opens results/primary2_calibration_ci.csv will see the point estimate outside the CI.

CORRECT: results/primary2_calibration_ci.csv, NHANES pre / after intercept/slope recalibration: ECE=0.009572, ECE_lo=0.009653, ECE_hi=0.022768, i.e. ECE < ECE_lo.

EVIDENCE: results/primary2_calibration_ci.csv row 3: "NHANES pre,after intercept/slope recalibration,0.998299,0.816666,1.187912,-0.000276,-0.124354,0.109213,0.009572,0.009653,0.022768"; 90_delta_and_calib_ci.py lines 63-70 take np.percentile(ec,2.5) directly with no bias correction.

---

### [32] MINOR — docs/cover_letter.md
QUOTE: calibration drifts yet is recoverable, on point estimates, by lightweight intercept/slope recalibration

PROBLEM: The hedge "on point estimates" was inserted specifically because no calibration interval existed at the time (83_review8_fixes.py line 92). Calibration intervals now exist and both recalibrated slope intervals include 1, so the cover letter understates the manuscript's own evidence and is inconsistent with Table 2, Table S16 and the Discussion. 91_add_intervals.py updated the manuscript and the TRIPOD checklist but not the cover letter.

CORRECT: Recalibration restores the slope to 1.00 (95% CI 0.82–1.19) pre-pandemic and 1.00 (0.88–1.13) post-pandemic, with CITL 0.00; the recovery is now supported by intervals, not only by point estimates.

EVIDENCE: 83_review8_fixes.py line 92 inserts the phrase; 91_add_intervals.py touches only docs/manuscript_17var.md, docs/TRIPOD_AI_checklist.md and docs/appendix3_utility.md. Manuscript line 175: "to slope 1.00 (0.82–1.19) and 1.00 (0.88–1.13); the intervals are wide ... and both include 1."

---

### [33] MINOR — docs/TRIPOD_AI_checklist.md
QUOTE: | 3b | Objectives, including intended use and stakeholders | ✅ | Introduction (¶4, aims); Intended use and users, under Model Development and Evaluation |

PROBLEM: The checklist's stated purpose is to locate each item in the manuscript ("'Section' refers to the manuscript section where each item is addressed"), but the Intended use and users subsection is not under Model Development and Evaluation. A reviewer following the map will look in the wrong Methods subsection.

CORRECT: The "**Intended use and users.**" paragraph sits under "### External Validation and Sensitivity Analyses".

EVIDENCE: docs/manuscript_17var.md line 93 = "### Model Development and Evaluation"; line 100 = "### External Validation and Sensitivity Analyses"; line 107 = "**Intended use and users.** The model is intended as a triage aid ...". The next heading after line 100 is "### Ethical Considerations" at line 113.

---

### [34] MAJOR — 17var_model/60_verify_manuscript.py
QUOTE: for var in re.findall(r"\b([A-Z]{2,4}[0-9]{3}[A-Z]?|[a-z]{3}_[0-9]{2}[a-z][0-9]|oba_bmi|mbhld_co|RID[A-Z]+|DMD[A-Z]+|IND[A-Z]+|BMX[A-Z]+|SLD[0-9]+|HUQ[0-9]+|FSDAD|OCD150|OHQ845)\b", t1):

PROBLEM: The two sequences that render as \b in line 300 are NOT the two characters backslash+b. Each is a single literal BACKSPACE byte (0x08) embedded in the raw string. The compiled pattern therefore requires a literal control character on both sides of every variable code, matches nothing in the manuscript, and the whole loop executes ZERO times. Section '5. 변수명 실재성 — 표에 적은 변수가 실제 조화 코드에 존재하는지' verifies nothing: a fabricated NHANES or KCHS variable code in Table 1 would pass silently.

CORRECT: The pattern must use the regex word-boundary \b (backslash + letter b). With that fixed the loop runs 33 checks against src/01_harmonize.py + src/15_ext_harmonize.py.

EVIDENCE: Byte scan: `b.count(b'\x08')` == 2, both on line 300 (`b'for var in re.findall(r"\x08([A-Z]{2,4}...'`). Instrumenting need() to record caller line numbers shows NO entries for line 301 — the 579 recorded need() calls contain 0 from this loop. Re-running the same regex with real \b on docs/manuscript_17var.md yields 33 matches (RIDAGEYR, sob_01z1, DMDEDUC, INDFMPIR, HUQ010, …). Only mtb_07a1 and DPQ010 are absent from the harmonization source, and those are built programmatically (src/01_harmonize.py:28 `PHQ_K = [f"mtb_07{c}1" for c in "abcdefghi"]`, :147 `[f"DPQ0{i}0" for i in range(1, 10)]`), so no Table 1 code is currently wrong — the check is simply dead.

---

### [35] MAJOR — 17var_model/docs/manuscript_17var.md
QUOTE: Discrimination remained clinically useful but lower externally: pre 0.765 (0.724–0.804), post 0.771 (0.739–0.801); the difference is +0.006 (95% CI −0.045 to +0.056; bootstrap P=0.86)

PROBLEM: The headline external-AUROC point estimates and their CIs in Results (External Validation Across Country and Pandemic Period) and in Discussion (Principal Findings) are hand-written. 61_sync_numbers.py owns only the '…the difference is …' clause of these sentences (pattern at 61_sync:409); 60_verify_manuscript.py has no membership test for either sentence. If primary2_metrics.csv changes, Table 2, the Abstract and the model card update while these two sentences keep the old numbers, and the verifier still prints '실패 0건'.

CORRECT: The values are currently correct (primary2_metrics.csv: NHANES pre AUROC 0.765028, CI 0.724149–0.803709; NHANES post 0.770591, CI 0.738854–0.800925), but nothing enforces that.

EVIDENCE: Mutation test in a sandbox copy: added +0.011 to AUROC/CI_lo/CI_hi in results/primary2_metrics.csv, ran 61_sync_numbers.py then 60_verify_manuscript.py. Table 2 and the Abstract became 0.776 (0.735–0.815) / 0.782 (0.750–0.812) while both prose sentences still read 0.765 (0.724–0.804) / 0.771 (0.739–0.801); 60_verify reported only 3 docx-timestamp failures and zero content failures. Same result for the Abstract's 'calibration deteriorated (slope 0.65–0.68; … [CITL] +0.51 and +1.13' and 'selected a 6-item model (0.822 internal, 0.748 pre, 0.757 post'.

---

### [36] MINOR — 17var_model/92_crossref_audit.py
QUOTE: SKIP = {"KCHS", "NHANES", "PHQ", "US", "AI", "OR", "CI", "SD", "II", "III", "IV", "NCHS", "KDCA", "COVID",
        "LOYO", "PSI", "WoE", "MLP", "FT", "GBM", "DPQ", "SHAP", "ECE", "CITL", "AUROC", "AUPRC",
        "BMI", "PROBAST", "TRIPOD", "ML", "NNS", "PSU", "IRB", "DOI", "PDF", "CSV", "JMIR", "USPSTF"}

PROBLEM: SKIP contains 15 of the 19 abbreviations that are actually in the manuscript's Abbreviations list (AUPRC, AUROC, BMI, CI, CITL, DPQ, ECE, KCHS, KDCA, ML, NCHS, NHANES, PROBAST, PSU, SHAP). Since the test is `missing = CAND - SKIP - set(listed)`, subtracting SKIP exempts exactly the abbreviations the audit exists to police: deleting any of them from the Abbreviations list is undetectable. KNHANES (7 letters) is also invisible because CAND is `[A-Z]{2,6}`. The `>= 2` occurrence threshold additionally means an abbreviation used once and never defined is never flagged. The complementary direction (listed-but-unused) is checked, so only the direction that matters for JMIR first-use definition is disabled.

CORRECT: SKIP should contain only tokens that are genuinely not abbreviations (variable-code stems, roman numerals), never entries that duplicate the Abbreviations list.

EVIDENCE: Replaying the audit's own logic on docs/manuscript_17var.md with 'AUROC: area under the receiver operating characteristic curve', 'CITL: …' and 'SHAP: …' deleted from the Abbreviations section: missing=[] , unused=[] , verdict PASS — despite AUROC appearing 31 times in the body. Same logic on the unmodified file also gives PASS, so the check has identical output either way.

---

### [37] MINOR — None
QUOTE: 

PROBLEM: 

CORRECT: 

EVIDENCE: 

---

### [38] MINOR — 17var_model/87_make_ledger.py
QUOTE: table("경량모델 궤적 (primary2_parsimonious.csv, 표 S6)",
      ["k", "added", "selection_AUROC", "holdout_AUROC", "ext_pre", "ext_post"],
      PARS[["k", "added", "selection_AUROC", "holdout_AUROC", "ext_pre", "ext_post"]].values.tolist(),

PROBLEM: Two silent losses. (a) The hard-coded column list drops the `vars` column of primary2_parsimonious.csv — the only column that records which six items the deployment model contains, and the exact column 61_sync_numbers.py reads at line 269 to name them in the manuscript and model card. The ledger therefore cannot be used to audit that item list. (b) 8 of the 20 results/primary2_*.csv files are never emitted at all, including BOTH new never-reviewed files (primary2_calibration_ci.csv, primary2_delta_auroc.csv) plus primary2_operating.csv, primary2_shap.csv, primary2_triage.csv, primary2_netbenefit.csv, primary2_cut5_prevalence.csv and primary2_nonresponse_ipw.csv — while the generated preamble asserts the ledger is produced from results/primary2_*.csv.

CORRECT: primary2_parsimonious.csv has 7 columns (k, added, vars, selection_AUROC, holdout_AUROC, ext_pre, ext_post); the ledger table has 6. 12 of 20 primary2_* files appear in docs/evidence_ledger.md; 8 do not.

EVIDENCE: Row/column comparison of every ledger table against its source CSV: all 11 tables match on rows and columns except parsimonious (csv 17x7 vs ledger 17x6). Filename membership test over glob('results/primary2_*.csv') against docs/evidence_ledger.md: MISSING = calibration_ci, cut5_prevalence, delta_auroc, netbenefit, nonresponse_ipw, operating, shap, triage. Contradicted by the file's own generated line 22: '이 문서는 `87_make_ledger.py` 가 `results/primary2_*.csv` 에서 자동 생성한다.'

---

### [39] MINOR — 17var_model/60_verify_manuscript.py
QUOTE: al = rd("sap_alcohol_alt.csv")

PROBLEM: sap_alcohol_alt.csv is loaded and never used — the identifier `al` appears exactly once in the whole file (its assignment). The same is true of `ov = rd("primary2_design.csv").iloc[0]` on line 57 (line 95 re-reads the file for KSTAR). The dead load implies a check that was intended and does not exist: the alcohol-alternative sensitivity numbers quoted in the manuscript are verified by neither 60_verify nor 61_sync.

CORRECT: The manuscript sentence 'changed discrimination by at most 0.003 externally (pre 0.769–0.772, post 0.769–0.770)' is currently correct against sap_alcohol_alt.csv (ext_pre range 0.7687–0.7718, ext_post 0.7688–0.7704, max shift 0.0031) but is hand-written and unchecked.

EVIDENCE: Occurrence count of each loaded frame name in 60_verify_manuscript.py: al=1, ov=1 (assignment only); every other loaded frame is used 2–7 times. Span-coverage analysis places the sentence outside all 61_sync-owned regions and outside all 60_verify membership tokens.

---

### [40] MINOR — 17var_model/docs/manuscript_17var.md
QUOTE: First, external discrimination was 0.765 (95% CI 0.724-0.804) before the pandemic and 0.771 (0.739-0.801) after it

PROBLEM: These are the only numeric ranges in the manuscript body written with an ASCII hyphen-minus instead of an en dash; the same paragraph also contains 'reached just 0.764-0.776'. The document uses an en dash in 43 other numeric ranges. All three offenders sit in the Discussion paragraph rewritten after the last review, which is exactly the prose neither 61_sync nor 60_verify owns — corroborating that it was typed by hand rather than injected.

CORRECT: House style throughout the manuscript is an en dash (U+2013) for numeric ranges: '0.724–0.804', '0.739–0.801', '0.764–0.776'.

EVIDENCE: Regex counts over docs/manuscript_17var.md up to '## References': `0\.\d{3}–0\.\d{3}` matches 43 times; `0\.\d{3}-0\.\d{3}` matches 3 times — ['0.724-0.804', '0.739-0.801', '0.764-0.776'], all in Discussion > Principal Findings and Plausible Mechanisms for the External Ceiling.

---

### [41] MAJOR — C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model/docs/manuscript_17var.md
QUOTE: among adults ≥20 years, 14.3% of the pre-pandemic and 32.7% of the post-pandemic respondents left three or more PHQ-9 items unanswered, yielding the complete-case analytic samples of 7,888 and 5,210

PROBLEM: The headline non-response statistic is computed on the wrong denominator and mis-describes the mechanism. 82_dpq_subgroup_fix.py takes ALL interviewed adults >=20 from DEMO and left-merges DPQ, so respondents who were interviewed but never attended the mobile examination center (MEC) count as having 'left three or more PHQ-9 items unanswered'. Those persons carry MEC weight 0/NaN and are outside the estimation frame of every weighted estimate in the paper (the analysis correctly uses WTMECPRP/WTMEC2YR, src/01_harmonize.py:145), and NCHS has already non-response-adjusted and post-stratified the MEC weights for exactly this attrition. Of the 2,551 post-pandemic 'non-respondents', 1,745 (68%) are simply not MEC-examined; of the 1,322 pre-pandemic, 688 (52%). This mischaracterization propagates to Table S8, to the differential-non-response figures, to the Limitations, and to the PROBAST 'unclear' rating for the analysis domain, which the paper says is 'driven by that non-response'. It errs against the paper (the true problem is roughly half as large), but it is a false description of how the data arose and an NHANES-literate reviewer will read it as a design misunderstanding in the paper's only external dataset.

CORRECT: Depression-module non-response within the MEC-weighted analytic frame is 7.4% pre (634/8,544) and 13.3% post (806/6,064), not 14.3% and 32.7%. Analytic-sample retention is 7,888/8,544 = 92.3% pre and 5,210/6,064 = 85.9% post, not 85.4% and 66.7%. Differential rates are also roughly halved: <HS 12.9% pre / 26.6% post (not 20.2/45.3); non-Hispanic Asian 14.3% / 31.6% (not 20.6/45.7); non-Hispanic Black 8.2% / 19.1% (not 14.5/38.8); non-Hispanic White 4.3% / 9.4% (not 12.8/29.0).

EVIDENCE: Recomputed directly from data/raw/nhanes/{pre_2017_2020/P_DEMO.xpt,P_DPQ.xpt} and {post_2021_2023/DEMO_L.xpt,DPQ_L.xpt}. Restricting to RIDAGEYR>=20 with MEC weight >0: pre n=8,544, 634 with >=3 DPQ items missing (7.4%), 7,888 with 0 missing (exactly the analytic sample); post n=6,064, 806 with >=3 missing (13.3%), 5,210 with 0 missing (exactly the analytic sample). The paper's denominators (9,232 and 7,809) are all interviewed adults; 688 pre and 1,745 post have no DPQ record at all and MEC weight <=0. Code that produces the wrong figures: 17var_model/82_dpq_subgroup_fix.py lines 23-26 (demo.merge(dpq, how='left') with no MEC-weight restriction).

---

### [42] MAJOR — C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model/90_delta_and_calib_ci.py
QUOTE:     pr = recal(y, p, w, "slope", groups=grp(d))          # 표 2 와 같은 재보정
    for lab, q in [("as transported", p), ("after intercept/slope recalibration", pr)]:
        sl = boot(y, q, R, cal_slope); ct = boot(y, q, R, citl); ec = boot(y, q, R, ece)

PROBLEM: The recalibration map is estimated once on the full external cohort (line 61, outside the replicate loop) and then held fixed while the 1000 cluster-bootstrap replicates resample the already-recalibrated probabilities. The resulting intervals for the recalibrated slope, CITL and ECE are therefore conditional on one estimated intercept/slope pair and omit the uncertainty of estimating that update - which is exactly the quantity H3 is about and exactly what the Limitations paragraph claims to be quantifying. This is never disclosed: the Table 2 footnote and the Table S16 header describe the intervals only as 'a rescaled stratified cluster bootstrap (1000 replicates)'. The team demonstrably knows the correct construction, because Table S15 reports a nested version for the oracle ('refitted inside each bootstrap replicate so that model-fitting variability is included', which widened 0.753-0.799 to 0.745-0.794).

CORRECT: The recalibrated slope intervals (0.82-1.19 pre, 0.88-1.13 post) and recalibrated CITL/ECE intervals are fixed-map intervals and understate the true uncertainty of a locally re-estimated update. A nested bootstrap that refits the intercept/slope inside each replicate is required, or the conditioning must be stated explicitly in the Table 2 footnote and in Table S16.

EVIDENCE: 17var_model/90_delta_and_calib_ci.py lines 59-70: recal() is called once per cohort at line 61; boot() at line 63 iterates REPS over the fixed vector pr. Contrast 17var_model/results/review4_oracle_grouped.csv, which carries a 'grouped CV, nested bootstrap with refit' row precisely to avoid this. Affected published numbers: docs/manuscript_17var.md Table 2 rows 'Calibration slope 95% CI' and 'CITL 95% CI' (recalibrated columns), and docs/appendix3_utility.md Table S16 rows 2 and 4.

---

### [43] MAJOR — C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model/docs/manuscript_17var.md
QUOTE: Intervals are computed for AUROC and AUPRC; calibration, operating characteristics and net benefit are reported as point estimates, as stated under Performance Evaluation.

PROBLEM: Direct self-contradiction introduced by the unreviewed interval edits. The Methods 'Sample size' paragraph (line 105) states that calibration is reported without intervals and attributes that statement to the Performance Evaluation subsection - but Performance Evaluation (line 98) now says the opposite, and Table 2 and Table S16 print calibration-slope, CITL and ECE intervals. The sentence also survives in the same paragraph as 'the achieved precision is reported as confidence intervals throughout'. A reviewer reading the Methods will conclude the calibration intervals in Table 2 are undocumented.

CORRECT: Performance Evaluation now reads: 'Ninety-five percent CIs are reported for AUROC and AUPRC, and, in the two external cohorts, for the calibration slope, CITL and ECE both as transported and after recalibration'. Table 2 reports 'Calibration slope 95% CI' 0.53-0.77 / 0.82-1.19 / 0.60-0.76 / 0.88-1.13 and 'CITL 95% CI' rows. The line-105 sentence should read that intervals are computed for AUROC, AUPRC and, externally, the calibration measures, while operating characteristics and net benefit remain point estimates.

EVIDENCE: docs/manuscript_17var.md line 105 vs line 98 and Table 2 (lines 140, 142); docs/appendix3_utility.md Table S16; results/primary2_calibration_ci.csv.

---

### [44] MINOR — C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model/docs/manuscript_17var.md
QUOTE: Applying the margin set in the analysis plan (the smallest set within 0.02 AUROC of the full model, searched over every count below 17 without imposing the plan's preferred 5–10 item range) selects six items: self-rated health, age, sex, sleep duration, self-rated oral health, partnership.

PROBLEM: The six-item form is a headline deliverable (Abstract, Conclusions, Table S2, Table S3, Figure 7), but its item count survives the prespecified rule by 0.00024 AUROC on a 50,000-record selection set - far inside the noise of that estimate. A five-item model (dropping partnership) fails the same rule by two ten-thousandths. The paper reports the stricter-0.01-margin seven-item variant as an exploratory comparison but never discloses that the boundary on the other side is essentially a coin flip, so a reader cannot see how fragile the deployed instrument's composition is. Table S6 puts the two numbers adjacent in the printed appendix (0.8130 vs 0.8176) but the threshold itself is never printed, so the margin is invisible.

CORRECT: Full-model selection-set AUROC is 0.833235, so the prespecified 0.02 threshold is 0.813235. k=5 scores 0.812995, missing by 0.00024; k=6 scores 0.817610. The selection of six rather than five items is not robust and should be reported as such, with the 5-item external performance (0.7372 pre, 0.7509 post) shown alongside the 6-item values.

EVIDENCE: results/primary2_parsimonious.csv rows k=5 (selection_AUROC 0.812995, ext_pre 0.737231, ext_post 0.750909), k=6 (0.817610), k=17 (0.833235); selection code 70_primary_corrected.py line 304: fullsel=pp.loc[pp.k==17,'selection_AUROC']; cand=pp[(pp.k<17)&(pp.selection_AUROC>=fullsel-0.02)].

---

### [45] MAJOR — C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model/docs/manuscript_17var.md
QUOTE: living alone is the exception, being unavailable in the pre-pandemic NHANES file and therefore entirely missing there and handled natively by the tree learner.

PROBLEM: Two things this wording conceals. First, 'handled natively by the tree learner' implies a data-driven default; but living alone has ZERO missing values in the KCHS training data, so LightGBM never learned a missing direction for it and assigns the entire pre-pandemic cohort to an arbitrary default branch at every split on that feature. That is an uncontrolled constant offset applied to all 7,888 pre-pandemic predictions, and it contributes to the pre-pandemic CITL of +0.51. Second, and more damaging for the title claim, the pre-vs-post contrast that carries 'transports across the pandemic' compares two cohorts that differ in period AND in whether a model predictor is observed at all (100% missing pre, 0% missing post). The paper never names this confound and never runs the obvious control - dropping living alone from both cohorts and re-running the +0.006 contrast. Nothing in Table S5 does this; only the nested SHAP-order trajectory in Table S6 incidentally bounds it.

CORRECT: living_alone missingness is 0.000% in ext_kchs (n=2,048,510), 100.0% in ext_pre (n=7,888), 0.000% in ext_post (n=5,210). Table S6 implies the confound is small (k=13 without living alone: pre 0.765206 / post 0.769224; k=14 with it: 0.765860 / 0.770931, i.e. <=0.002), but that bound is never invoked and a same-predictor-set pre-vs-post contrast is never reported. The delta and its interval (+0.006, -0.045 to +0.056) should be recomputed on the 16 predictors observed in both cycles.

EVIDENCE: Recomputed from 17var_model/data/{ext_kchs,ext_pre,ext_post}.parquet (the exact files loaded at 70_primary_corrected.py lines 71-73): living_alone NaN share 0.0 / 100.0 / 0.0. Table S6 k=13 vs k=14 rows in results/primary2_parsimonious.csv. No living-alone-removed row exists in results/primary2_sensitivity.csv.

---

### [46] MAJOR — C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model/docs/appendix3_utility.md
QUOTE: | Characteristic | KCHS | NHANES pre | NHANES post | Missing (KCHS / pre / post) |

PROBLEM: Table S4 is the study's only participant-characteristics table and it omits 5 of the 17 model predictors entirely: education, household income quintile, living alone, alcohol (current drinker) and physical activity. Neither their distributions nor their missingness appear anywhere in the manuscript or appendices. Three of these are exactly the items a reviewer would want to check. Income quintile is missing for 16.5% of KCHS records - the second-highest missing rate of any predictor, higher than the BMI (13.1%) and sleep (11.3%) rates the table does report - and income handling was the subject of three separate corrected data-integrity defects. Physical activity was silently coded 0 for the entire post-pandemic cohort until the audit, and the reader has no distribution with which to verify the fix. Living alone's 100% missingness in NHANES pre never appears in any table. TRIPOD+AI requires predictor distributions and missing-data counts for development and validation data.

CORRECT: Predictor missingness not reported anywhere: income_q 16.52% (KCHS) / 13.11% (pre) / 11.98% (post); living_alone 0.0 / 100.0 / 0.0; active 0.038 / 0.0 / 0.749; edu 0.082 / 0.076 / 0.038; current_drinker 0.002 / 0.051 / 0.441. Table S4 should carry a row per predictor.

EVIDENCE: Recomputed from 17var_model/data/{ext_kchs,ext_pre,ext_post}.parquet. Table S4 (docs/appendix3_utility.md lines 105-121) lists only depression, female, current smoker, partnered, employed, food insecure, hypertension, diabetes, age, BMI, sleep, self-rated health, oral health; results/primary2_characteristics.csv contains the same 13 rows and no others.

---

### [47] MAJOR — C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model/docs/manuscript_17var.md
QUOTE: given that non-response was differential by education and by race and ethnicity, the external estimates, and particularly the post-pandemic subgroup results, may be biased in an unknown direction.

PROBLEM: The paper's entire response to its largest acknowledged threat is a statement that the direction is unknown - no quantitative bias analysis, no respondent-vs-non-respondent comparison on predictors, no tipping-point or E-value analysis on prevalence or calibration. This is what makes the PROBAST analysis domain 'unclear' by the paper's own account. But the analysis pipeline already computes an inverse-probability-of-response reweighted external AUROC and it shows the discrimination estimates are essentially unaffected. That result appears in no manuscript or appendix section and in no table; it is present only as an orphan CSV. Leaving it out converts a defensible, quantified sensitivity into an unaddressed limitation, and a reviewer who requests exactly this analysis will get a major revision out of work that has already been done.

CORRECT: Response-propensity inverse-probability reweighting (education-based, from the Table S8 non-response rates) changes external AUROC by at most 0.001: NHANES pre 0.7651 vs 0.7650 base; NHANES post 0.7694 vs 0.7706 base. This should be reported in the Sensitivity Analyses section and in Table S5 or S8. It also does not cover calibration, prevalence or the subgroup estimates, which remain unquantified.

EVIDENCE: 17var_model/results/primary2_nonresponse_ipw.csv (AUROC_ipw 0.765111 / 0.769430 vs AUROC_base 0.765028 / 0.770591); generating code 17var_model/70_primary_corrected.py lines 356-369. grep for 'IPW|inverse.probability|inverse probability' across 17var_model/docs/ returns zero hits.

---

### [48] MINOR — C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model/docs/appendix3_utility.md
QUOTE: | NHANES pre | after intercept/slope recalibration | 1.00 (0.82–1.19) | 0.00 (−0.12–+0.11) | 0.010 (0.010–0.023) |

PROBLEM: The percentile bootstrap interval for the recalibrated pre-pandemic ECE excludes its own point estimate: the point value is 0.009572 but the 2.5th percentile of the replicate distribution is 0.009653. Rounding to three decimals hides this ('0.010 (0.010-0.023)'). ECE is a positively biased statistic under resampling (bin-level noise inflates it), so a naive percentile interval is the wrong construction for it; a bias-corrected interval or a stated caveat is needed. Minor for the paper's conclusions, but a statistical reviewer who requests the underlying numbers will see an interval that cannot contain the estimate.

CORRECT: NHANES pre, after intercept/slope recalibration: ECE = 0.009572, percentile CI 0.009653-0.022768. The lower bound exceeds the point estimate by 0.000081. The other three ECE rows do contain their point estimates.

EVIDENCE: 17var_model/results/primary2_calibration_ci.csv row 2: ECE 0.009572, ECE_lo 0.009653, ECE_hi 0.022768. Interval construction: 90_delta_and_calib_ci.py lines 63-70 (np.percentile(ec, 2.5)).

---

### [49] MINOR — C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model/docs/manuscript_17var.md
QUOTE: reducing the Korean data to the NHANES sample size and cross-validating within it still yielded 0.823 at n=7,888

PROBLEM: These matched-sample-size figures come from the pre-audit exploration pipeline, which used the uncorrected harmonized data (before the six/seven KCHS drinking, 2018 activity, weekday-sleep, age top-code, income-refusal and income-quintile corrections) and a different LightGBM configuration. The manuscript places them in the same sentence as corrected primary numbers with no provenance note, whereas Table S13 carries exactly such a note for the deep-learner table ('its external values (0.7694/0.7685) therefore differ from the primary model's 0.765/0.771; the comparison is internally consistent within the run and should not be read against the primary estimates'). The caveating is inconsistent between two exploration-run tables that are used for the same argument.

CORRECT: The matched-n figures derive from experiments/pathG_02b_ceiling_rest.py operating on the exploration parquet files with LGB_PARAMS (n_estimators=600, lr=0.03, num_leaves=31, no min_child_samples), not from the corrected primary pipeline or its deployed configuration (num_leaves=31, min_child_samples=1000, lr=0.01, n_estimators=900). They need the same provenance note as Table S13. Table S10 bounds the effect of the corrections at <=0.003 AUROC, so the substantive conclusion is unlikely to change.

EVIDENCE: C:/Users/rudtn/Desktop/claude code/지역사회건강/experiments/pathG_common.py lines 29-38 (FEATURES, LGB_PARAMS) and load() reading data/processed/ext_*.parquet; deployed configuration in docs/manuscript_17var.md line 96; Table S13 caveat in docs/appendix4_ceiling.md line 44.

---

### [50] MINOR — C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model/docs/manuscript_17var.md
QUOTE: LightGBM (0.840) and random forest (0.835) differ by 0.0046 and therefore fall inside the tie band, to which the plan's tie-break applies; it selects LightGBM, whose cross-validated calibration slope was 1.003 against 1.203 for random forest.

PROBLEM: The main text never states that the cross-validated AUROCs producing this 0.0046 margin came from a single-pass 5-fold cross-validation, whereas the analysis plan specified 5-fold cross-validation repeated 5 times. That deviation is material precisely here: it is the deviation that makes the AUROC ranking of the two leading algorithms unstable, and hence makes primary-model selection contingent on the calibration-slope tie-break. The manuscript states other plan deviations inline (for example the class-weighting deviation, 'which departs from the plan's recommended class weighting'), so the omission of this one is asymmetric. It is disclosed, fully and honestly, only in appendix Table S9.

CORRECT: Table S9 records: prespecified '5-fold cross-validation repeated 5 times (D5)'; what was done was 'single-pass 4-fold survey-weighted CV for the hyperparameter and feature-representation search, single-pass 5-fold CV for the algorithm comparison', with the note 'Repeats were not run. This is a deviation, and the margin between the two leading algorithms (0.0046 AUROC) lies inside the plan's 0.005 tie band'. One clause to that effect belongs in the Model Development and Evaluation subsection.

EVIDENCE: docs/appendix3_utility.md Table S9 deviations table, row 'D5'; docs/manuscript_17var.md line 96 (no mention of repeats); grep -i 'repeat' over the manuscript returns only unrelated Introduction text.


---

# 종합

## Synthesis — 50 surviving defects, ranked

I re-verified the three claims that could actually move the verdict (`living_alone` missingness across the three parquet files; LightGBM's NaN behaviour when no missing was seen in training; the calibration/delta CSVs). All three confirm. Everything below is grouped by underlying cause, most severe first.

---

### 1. The living-alone predictor is silently constant-imputed, the Methods say it is not, and it confounds the pandemic contrast — #45
This is the only item that touches a central claim. I confirmed `living_alone` is 0.000% missing in KCHS (n=2,048,510), **100.0%** missing in NHANES pre, 0.000% in NHANES post. I then confirmed empirically that LightGBM, having seen no missing values for a feature in training, maps NaN to the **zero branch** at prediction time — a 3-feature test gave byte-identical probabilities for `NaN` and `0.0` (0.00239881 both) versus 0.99594 for `1.0`. Two consequences:
- `manuscript_17var.md:91` states "structurally missing binary items such as living-alone in NHANES pre are therefore **never imputed to a constant**." That is a **false statement of fact about your own pipeline**, sitting in the Missing Data paragraph, in the same document as a data-integrity audit that claims to have caught exactly this class of bug.
- The headline pre-vs-post contrast (+0.006, the "transports across the pandemic" claim) compares cohorts that differ in period **and** in whether a predictor exists at all. You never run the obvious control.

**Must change:** delete/replace the false parenthetical, and recompute the pre-vs-post delta and its interval on the 16 predictors observed in both cycles (Table S6's k=13/k=14 rows suggest the effect is ≤0.002, so this almost certainly costs you nothing — but it must be *run and reported*, not inferred).

### 2. The NHANES non-response statistic is on the wrong frame — #41 (and #19)
14.3% / 32.7% are computed over all interviewed adults ≥20, but 68% of the post-pandemic "non-respondents" were simply never MEC-examined and carry MEC weight 0 — they are outside the estimation frame of every weighted number in the paper. On the MEC frame the rates are 7.4% and 13.3%, retention 92.3% and 85.9%. It errs *against* you, but it is a design misunderstanding of your only external dataset, and NHANES-literate reviewers are exactly who JMIR will assign. #19 is the same paragraph: "yielding the complete-case analytic samples of 7,888 and 5,210" is arithmetically wrong (≥3-missing leaves 7,910/5,258; the samples come from excluding *any* missing item).

**Must change:** recompute Table S8, the differential subgroup rates, the Limitations sentence and the PROBAST rationale on the MEC-weighted frame, and fix the "yielding" link.

### 3. You already ran the bias analysis your Limitations says is impossible — #47
`results/primary2_nonresponse_ipw.csv` shows response-propensity IPW moves external AUROC by ≤0.0012 (pre 0.7651 vs 0.7650; post 0.7694 vs 0.7706). Zero hits for "IPW"/"inverse probability" across `docs/`. Your Limitations currently says the bias direction is "unknown"; a reviewer will demand this analysis, and you will be doing a major revision to hand over a CSV that already exists.

**Must change:** report the IPW result in Sensitivity Analyses and Table S5/S8 (after recomputing it on the corrected denominator from item 2).

### 4. The interval edit only half-landed: a Methods sentence now says the opposite of the rest of the paper, and the table holding the new intervals is an orphan — #6/#12/#25/#43 + #3/#7/#13/#14/#26/#27/#30 + #11/#32
One cause, many symptoms. `91_add_intervals.py` rewrote some strings and missed others.
- `manuscript_17var.md:105` still reads "**calibration, operating characteristics and net benefit are reported as point estimates, as stated under Performance Evaluation**" — contradicted by line 98, by Table 2's two CI rows, by the Abstract, by the Discussion, and it misattributes itself to the section that says the opposite. Four independent reviewers found this; it is the single most embarrassing line in the file.
- Table S16 is cited **nowhere**. `Table S16` occurs once in all of `docs/` — its own heading. The appendix inventory (line 231) stops at S15. In the built docx it sits physically between S10 and S11, and the delta-AUROC paragraph after it is auto-converted into footnote "a" anchored to the *Cohort* column of a **calibration** table.
- Consequently the Methods promise of ECE CIs has no reachable home: Table 2 has an ECE row and no ECE CI row.
- `cover_letter.md:11` still hedges "recoverable, **on point estimates**" — the first thing the editor reads, understating your own evidence.

**Must change:** delete the stale clause at line 105; add an ECE CI row to Table 2 (or drop ECE from the Methods promise); cite S16 from Methods and Results; add it to the inventory; renumber/relocate it so appendix order is monotone; drop the obsolete hedge from the cover letter.

### 5. The recalibrated intervals hold the recalibration map fixed, and nothing says so — #8/#15/#28/#42
`90_delta_and_calib_ci.py:61` fits `recal()` once on the full cohort, outside the replicate loop; the 1000 replicates then reweight a fixed probability vector. So 0.82–1.19 and 0.88–1.13 are conditional on one estimated update and exclude the uncertainty in the very quantity H3 is about. You apply and *disclose* the correct nested construction for the oracle (Table S15, widening 0.753–0.799 → 0.745–0.794), so a statistical reviewer will see you know the difference and ask why not here.

**Must change:** refit the intercept/slope inside each replicate and report the widened interval, or state the fixed-map conditioning explicitly in the Table 2 footnote and Table S16 note.

### 6. Table S4 omits 5 of the 17 predictors — #46
No distribution and no missingness anywhere for education, income quintile, living alone, alcohol, physical activity. Income quintile is 16.5% missing in KCHS — the **highest** rate of any predictor, above BMI (13.1%) and sleep (11.3%), both of which you do report. Income handling and post-pandemic physical activity were both subjects of corrected integrity defects, so these are precisely the rows a reviewer wants to audit the fixes. TRIPOD+AI requires them.

**Must change:** one row per predictor in Table S4.

### 7. Exploration-run numbers are printed next to primary numbers with no scale flag — #16 (also #17, #49)
The text says covariate-shift correction "*reduced*" external AUROC **to 0.766 pre** — higher than your own primary pre-pandemic 0.765 — and that adding public assistance left it "unchanged" at 0.772, 0.007 above primary. Same problem for the alcohol ranges (0.769–0.772) and the matched-n 0.823. The provenance note exists in Table S13 and is applied to Table S11; it is not applied in the body, where it matters most. As written the manuscript appears to contradict itself numerically.

**Must change:** report these as deltas from the in-run baseline (0.772/0.769), or attach the Table S13 caveat to the body clause.

### 8. Your verification infrastructure does not verify the things it claims to — #35, #34, #2/#36, #38/#39, #29
Independent reproduction of the mutation test: adding +0.011 to `primary2_metrics.csv` and re-running `61_sync` then `60_verify` updated the Abstract and Table 2 to 0.776/0.782 while the Results and Discussion sentences kept 0.765/0.771 — and `60_verify` reported **zero content failures**. Meanwhile `model_card_17var.md:43` states "documents are populated from result files, **never by hand**." That is a submitted document asserting a property the headline external result does not have. Supporting rot: `60_verify_manuscript.py:300` contains two literal **backspace bytes (0x08)** where `\b` should be, so the variable-existence check runs zero times (33 checks with the fix); `92_crossref_audit.py`'s uncited-table guard can never fire; its abbreviation guard exempts 15 of the 19 abbreviations it exists to police; two dead loads in `60_verify`. Also #29: `model_card_17var.md:30` is garbled by a blind string replacement and now reads that the 500,000 training records were "drawn at random from the internal metrics computed on that hold-out" — literally inverting the split your leakage argument rests on.

**Must change:** fix the model card sentence; either make `61_sync` own the prose sentences or delete the "never by hand" claim; fix the backspace bytes and the two dead guards.

### 9. P values appear three times with no Methods definition, two incompatible estimators, and a floor reported as an estimate — #4/#5/#10/#20
`90_delta...:50` is uncentred and uncorrected; `70_primary_corrected.py:325-326` is null-centred and (1+·)/(B+1)-corrected. "P value", "P-value" and "two-sided" appear zero times in `docs/`. And "*P*=0.001 and 0.001" **is the floor** of a 1000-replicate procedure — it must be `P<.001` (also JMIR style). The `max(pval, 1/K)` floor is off by a factor of two on the code's own terms, though latent at P=0.86.

**Must change:** define the bootstrap P once in Performance Evaluation; report `P<.001`; unify the estimator.

### 10. The six-item deployment model clears its own rule by 0.00024 AUROC — #18/#44
k=17 selection AUROC 0.833235 → threshold 0.813235; k=5 scores 0.812995. Worse for reporting: the manuscript prints "5 items 0.813 … 17 items 0.833", at which precision a reader applying your stated rule computes 0.813 and selects **five** items. The reported trajectory does not reproduce the reported decision.

**Must change:** one sentence stating the margin and that the 5-vs-6 boundary is not robust, with k=5 external values (0.7372/0.7509) alongside.

### 11. Residual factual/wording items
"Removing food insecurity alone leaves **most** of that difference intact" — it is exactly 50% post-pandemically (#21). "items that both surveys **measure identically**" — sleep and age both required definitional realignment per your own Harmonization cautions (#24). 100,000 vs 100,001 hold-out (#23). Duplicated clause in line 159 (#22). Post-hoc flag missing from Methods/Abstract for the between-cycle interval (#9). Five hyphen-minus decimal ranges plus mixed year ranges (#40). TRIPOD checklist points 3b at the wrong subsection (#33). The D5 repeats deviation — the one that makes the LightGBM-vs-RF ranking unstable and forces the tie-break — is disclosed only in Table S9 while other deviations are stated inline (#50).

---

## Verdict

**Not submittable as-is.** But nothing here invalidates the model, the headline AUROCs, Table 2, or the transportability conclusion.

One item threatens a central claim: **#1 (living alone)**. The paper's title-level claim is that the model transports across a pandemic boundary, and the pre-vs-post contrast that carries it is confounded by a predictor that is 100% present on one side and 100% absent on the other — and the Methods contain a false sentence denying exactly that mechanism. Your own Table S6 suggests the confound is ≤0.002, so this is very likely a one-run fix, but until you run it the claim is unsupported and the sentence is untrue. Do not submit before that sentence is corrected and that control is run.

Two more will draw a major revision on their own: **#2 (non-response frame)** and **#5 (fixed recalibration map)**. Both are cheap — one recomputation, one caveat or one nested bootstrap. **#3 (orphan IPW)** is free evidence you are currently throwing away.

**#4 is the real indictment of process.** Ten rounds of hostile review, 178 defects fixed, and the *unreviewed* interval edit introduced a Methods sentence that flatly contradicts the section it cites, orphaned the table holding the new numbers, and left the cover letter hedging against evidence the paper now has. Four independent reviewers found line 105. An editor will find it in the first ten minutes.

And be honest with yourself about **#8**: your verification suite reported "실패 0건" on a manuscript whose Abstract and Results disagreed by 0.011, while the model card told the world the documents are never populated by hand. The confidence you have been drawing from "60_verify passes" is not warranted for the headline external result.

**Sequence:** fix #1 (sentence + control run) → #2/#3 (recompute non-response on the MEC frame, report IPW) → #4 (delete line 105 clause, cite and relocate S16, add the ECE CI row, fix the cover letter) → #5 (nested bootstrap or explicit caveat) → #6/#7 → the rest as copyedit. After those, submit. It is a strong paper being let down by an unreviewed final edit and a verification harness that stopped verifying.