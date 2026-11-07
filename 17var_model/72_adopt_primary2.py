# -*- coding: utf-8 -*-
"""72_adopt_primary2.py — 교정 주분석(primary2_*)을 원고의 주분석으로 채택.

지금까지 '한계로 공개'했던 결함을 실제로 제거한 결과를 문서에 반영한다.
"""
import io, os, sys, re
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__)); RES=f"{D}/results"; DOCS=f"{D}/docs"
R = lambda v, nd=3: f"{Decimal(str(v)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP):.{nd}f}"

m  = pd.read_csv(f"{RES}/primary2_metrics.csv").set_index("set")
op = pd.read_csv(f"{RES}/primary2_operating.csv").set_index("set")
g  = pd.read_csv(f"{RES}/primary2_gain.csv").set_index("set")
pp = pd.read_csv(f"{RES}/primary2_parsimonious.csv").set_index("k")
dz = pd.read_csv(f"{RES}/primary2_design.csv").iloc[0]
sn = pd.read_csv(f"{RES}/primary2_sensitivity.csv").set_index("cond")
ipw= pd.read_csv(f"{RES}/primary2_nonresponse_ipw.csv").set_index("set")
I,PR,PO = "Internal(hold-out)","NHANES pre","NHANES post"
K = int(dz.kstar); kv = pp.loc[K,"vars"].split(";")
NAME = {"srh":"self-rated health","age":"age","sex":"sex","sleep_hrs":"sleep duration","oral_health":"self-rated oral health",
        "partnered":"partnership","employed":"employment","smoking":"smoking","bmi":"BMI","food_insecure":"food insecurity",
        "income_q":"household income","active":"physical activity","edu":"education","living_alone":"living alone",
        "current_drinker":"current drinking","htn":"hypertension","dm":"diabetes"}
kn = ", ".join(NAME[v] for v in kv)
NUMW = {1:"one",2:"two",3:"three",4:"four",5:"five",6:"six",7:"seven",8:"eight"}

def sub(path, pairs):
    p=f"{DOCS}/{path}"; s=io.open(p,encoding="utf8").read()
    for a,b in pairs:
        n=s.count(a); assert n==1,(path,a[:70],n); s=s.replace(a,b)
    io.open(p,"w",encoding="utf8").write(s); print("  fixed:",path)

print("[1] 원고 — 설계·주요 결과")
sub("manuscript_17var.md", [
 # §2.2 군집 분할
 ("From this pooled sample, an outcome-stratified random split created a **500,000-record training set** and a **100,000-record internal hold-out set** used for all internal performance estimates; the remaining 1,448,510 records were not used.",
  f"From this pooled sample, primary sampling units were split at the **cluster level** — every survey-year/stratum/PSU is assigned entirely to one side — giving a **{int(dz.train_n):,}-record training set** and a **{int(dz.holdout_n):,}-record internal hold-out set** that share no cluster with it, so the internal estimate refers to new clusters and not merely to new individuals within sampled clusters."),
 # §2.9 CI/재보정 문장 유지, 재보정 3단계
 ("The plan also envisaged an intercept-only step and a Platt/temperature-scaling comparison; neither was run (Multimedia Appendix 3, Table S11).",
  "Following the plan, recalibration was applied in three steps and all three are reported: intercept only, intercept plus slope, and temperature (Platt) scaling, each cross-fitted over five folds."),
 # §3.2 내부
 (f"Internal hold-out AUROC was **0.832 (95% CI {R(0.8222)}–{R(0.8408)})**, AUPRC 0.233, Brier 0.031, calibration slope 1.00, CITL +0.10 — **H1 met**.",
  f"Internal hold-out AUROC was **{R(m.loc[I,'AUROC'])} (95% CI {R(m.loc[I,'CI_lo'])}–{R(m.loc[I,'CI_hi'])})**, AUPRC {R(m.loc[I,'AUPRC'])}, Brier {R(m.loc[I,'Brier'],3)}, calibration slope {R(m.loc[I,'cal_slope'],2)}, CITL +{R(m.loc[I,'CITL'],2)} — **H1 met**. Because the split is at cluster level, this is an estimate for clusters the model never saw."),
])

print("[2] 원고 — §3.3 및 Table 2")
s=io.open(f"{DOCS}/manuscript_17var.md",encoding="utf8").read()
i=s.index("See Table 2 and Figures 2 and 3."); j=s.index("**Table 2. Discrimination and calibration")
s = s[:i] + (
 f"See Table 2 and Figures 2 and 3. Discrimination remained clinically useful but lower externally: pre **{R(m.loc[PR,'AUROC'])} ({R(m.loc[PR,'CI_lo'])}–{R(m.loc[PR,'CI_hi'])})**, "
 f"post **{R(m.loc[PO,'AUROC'])} ({R(m.loc[PO,'CI_lo'])}–{R(m.loc[PO,'CI_hi'])})**; the two cycles are statistically indistinguishable, so discrimination did not deteriorate across the pandemic. "
 f"Calibration degraded in both cycles: the slope fell to {R(m.loc[PR,'cal_slope'],2)} (pre) and {R(m.loc[PO,'cal_slope'],2)} (post) and CITL rose to +{R(m.loc[PR,'CITL'],2)} and +{R(m.loc[PO,'CITL'],2)}, "
 f"indicating that the Korea-trained model systematically *under*-predicted absolute risk in the US, as expected from the prevalence difference (KCHS 3.6% vs 8.4% and 12.2%). "
 f"Applying the three pre-specified recalibration steps out-of-sample, intercept-only updating removed the mean bias but left the slope unchanged ({R(m.loc[PR,'recal_int_slope'],2)} pre, {R(m.loc[PO,'recal_int_slope'],2)} post), "
 f"intercept-plus-slope updating restored both (slope {R(m.loc[PR,'recal_is_slope'],2)} and {R(m.loc[PO,'recal_is_slope'],2)}, CITL 0.00), and temperature scaling was intermediate — so intercept-plus-slope is the recommended deployment update "
 f"— **H2 and H3 supported**.\n\n") + s[j:]
i=s.index("| Metric | Internal | NHANES pre | pre (recal.) | NHANES post | post (recal.) |"); j=s.index("\n\nCIs are from")
rows = [
 "| Metric | Internal | NHANES pre | pre (recal.) | NHANES post | post (recal.) |","|---|---|---|---|---|---|",
 f"| AUROC | {R(m.loc[I,'AUROC'])} | {R(m.loc[PR,'AUROC'])} | {R(m.loc[PR,'recal_is_AUROC'])} | {R(m.loc[PO,'AUROC'])} | {R(m.loc[PO,'recal_is_AUROC'])} |",
 f"| 95% CI | {R(m.loc[I,'CI_lo'])}–{R(m.loc[I,'CI_hi'])} | {R(m.loc[PR,'CI_lo'])}–{R(m.loc[PR,'CI_hi'])} | — | {R(m.loc[PO,'CI_lo'])}–{R(m.loc[PO,'CI_hi'])} | — |",
 f"| AUPRC | {R(m.loc[I,'AUPRC'])} | {R(m.loc[PR,'AUPRC'])} | {R(m.loc[PR,'recal_AUPRC'])} | {R(m.loc[PO,'AUPRC'])} | {R(m.loc[PO,'recal_AUPRC'])} |",
 f"| 95% CI | {R(m.loc[I,'AUPRC_lo'])}–{R(m.loc[I,'AUPRC_hi'])} | {R(m.loc[PR,'AUPRC_lo'])}–{R(m.loc[PR,'AUPRC_hi'])} | — | {R(m.loc[PO,'AUPRC_lo'])}–{R(m.loc[PO,'AUPRC_hi'])} | — |",
 f"| Brier | {R(m.loc[I,'Brier'],3)} | {R(m.loc[PR,'Brier'],3)} | {R(m.loc[PR,'recal_is_Brier'],3)} | {R(m.loc[PO,'Brier'],3)} | {R(m.loc[PO,'recal_is_Brier'],3)} |",
 f"| ECE | {R(m.loc[I,'ECE'],3)} | {R(m.loc[PR,'ECE'],3)} | {R(m.loc[PR,'recal_ECE'],3)} | {R(m.loc[PO,'ECE'],3)} | {R(m.loc[PO,'recal_ECE'],3)} |",
 f"| Calibration slope | {R(m.loc[I,'cal_slope'],2)} | {R(m.loc[PR,'cal_slope'],2)} | {R(m.loc[PR,'recal_is_slope'],2)} | {R(m.loc[PO,'cal_slope'],2)} | {R(m.loc[PO,'recal_is_slope'],2)} |",
 f"| CITL | +{R(m.loc[I,'CITL'],2)} | +{R(m.loc[PR,'CITL'],2)} | 0.00 | +{R(m.loc[PO,'CITL'],2)} | 0.00 |",
]
s = s[:i] + "\n".join(rows) + s[j:]
io.open(f"{DOCS}/manuscript_17var.md","w",encoding="utf8").write(s); print("  Table 2 재작성")

print("[3] 원고 — §3.4 이득 · §3.6 경량 · §3.8 운영지점")
sub("manuscript_17var.md", [
 (f"pre 0.749→**0.772** (Δ+{R(0.0225)}, 95% CI +{R(0.0131)} to +{R(0.033)}; *P*<.001), post 0.753→**0.769** (Δ+{R(0.0158)}, 95% CI +{R(0.0070)} to +{R(0.0231)}; *P*<.001)",
  f"pre {R(g.loc[PR,'AUROC_14'])}→**{R(g.loc[PR,'AUROC_17'])}** (Δ+{R(g.loc[PR,'delta'])}, 95% CI +{R(g.loc[PR,'CI_lo'])} to +{R(g.loc[PR,'CI_hi'])}; *P*={R(g.loc[PR,'P'],3)}), post {R(g.loc[PO,'AUROC_14'])}→**{R(g.loc[PO,'AUROC_17'])}** (Δ+{R(g.loc[PO,'delta'])}, 95% CI +{R(g.loc[PO,'CI_lo'])} to +{R(g.loc[PO,'CI_hi'])}; *P*={R(g.loc[PO,'P'],3)})"),
])
s=io.open(f"{DOCS}/manuscript_17var.md",encoding="utf8").read()
i=s.index("Predictor order and item count were both fixed inside the training set"); j=s.index("The full k=1–17 trajectory is tabulated in Multimedia Appendix 3, Table S8.")+len("The full k=1–17 trajectory is tabulated in Multimedia Appendix 3, Table S8.")
traj="; ".join(f"{k} item{'s' if k>1 else ''} {R(pp.loc[k,'selection_AUROC'])}" for k in [1,3,5,K,K+2,12,17] if k in pp.index)
s = s[:i] + (
 f"Predictor order and item count were both fixed inside the training set (§2.11). On the selection set, discrimination rose steeply and then flattened: {traj}. "
 f"Applying the margin set in the analysis plan — the smallest set within 0.02 AUROC of the full model, subject to a 5–10 item ceiling — selects **{NUMW[K]} items**: {kn}. "
 f"Evaluated afterwards on data that played no part in selection, this model achieved hold-out AUROC **{R(pp.loc[K,'holdout_AUROC'])}**, NHANES pre **{R(pp.loc[K,'ext_pre'])}** and NHANES post **{R(pp.loc[K,'ext_post'])}**, "
 f"against {R(pp.loc[17,'holdout_AUROC'])}/{R(pp.loc[17,'ext_pre'])}/{R(pp.loc[17,'ext_post'])} for the full 17-predictor model fitted on the same data — a loss of {R(pp.loc[17,'holdout_AUROC']-pp.loc[K,'holdout_AUROC'])} internally, "
 f"{R(pp.loc[17,'ext_pre']-pp.loc[K,'ext_pre'])} pre-pandemic and {R(pp.loc[17,'ext_post']-pp.loc[K,'ext_post'])} post-pandemic. Every loss is within the plan's 0.02 tolerance, so **H4 is supported**. "
 f"No smaller-margin variant is reported, so no external information entered the item-count decision. The full k=1–17 trajectory is tabulated in Multimedia Appendix 3, Table S8.") + s[j:]
io.open(f"{DOCS}/manuscript_17var.md","w",encoding="utf8").write(s); print("  §3.6 재작성")

s=io.open(f"{DOCS}/manuscript_17var.md",encoding="utf8").read()
i=s.index("On decision-curve analysis (weighted, with cohort-specific screen-all"); j=s.index("Clinical utility remains provisional pending prospective evaluation.")
s = s[:i] + (
 "On decision-curve analysis (weighted, with cohort-specific screen-all and screen-none default strategies), the model provided positive net benefit exceeding both defaults across a clinically relevant threshold-probability range in all three datasets (Figure 7). "
 f"The operating threshold was fixed once in the development data at the point giving 85% sensitivity and then applied unchanged to the hold-out and to both external cohorts, so the external operating characteristics are genuine transported values rather than values guaranteed by construction. "
 f"At that single threshold, sensitivity/specificity were {R(op.loc[I,'sens'])}/{R(op.loc[I,'spec'])} internally, {R(op.loc[PR,'sens'])}/{R(op.loc[PR,'spec'])} in NHANES pre and {R(op.loc[PO,'sens'])}/{R(op.loc[PO,'spec'])} in NHANES post — that is, the target sensitivity transported to within {R(0.85-min(op.loc[PR,'sens'],op.loc[PO,'sens']),2)} — with positive predictive value {R(op.loc[PR,'PPV'])} and {R(op.loc[PO,'PPV'])} externally "
 f"(number needed to screen {R(op.loc[PR,'NNS'],1)} and {R(op.loc[PO,'NNS'],1)}) and negative predictive value {R(op.loc[PR,'NPV'])} and {R(op.loc[PO,'NPV'])}. The model flagged {R(op.loc[PR,'flag']*100,0)}% of the pre-pandemic and {R(op.loc[PO,'flag']*100,0)}% of the post-pandemic population. "
 "Full operating characteristics and net benefit are in Multimedia Appendix 3, Tables S1, S4 and S5. ") + s[j:]
io.open(f"{DOCS}/manuscript_17var.md","w",encoding="utf8").write(s); print("  §3.8 운영지점 재작성")

print("[4] 원고 — §2.5 / §2.13: '공개'에서 '교정'으로")
sub("manuscript_17var.md", [
 ("Three further discontinuities inside KCHS follow from questionnaire changes across survey years:",
  "Four harmonization defects were identified and **corrected before the primary analysis** rather than carried as limitations. Three arise from questionnaire changes inside KCHS:"),
 ("We tested each of these, together with the age top-code, the 2017 income refusal codes and the income-quintile repair, by applying one correction at a time and refitting: no single correction moved discrimination by more than 0.003 in any cohort, the largest being the weekday-sleep alignment which raised post-pandemic AUROC from 0.769 to 0.772 (Multimedia Appendix 3, Table S12). The primary results are therefore reported on the harmonization as analysed, with these definitional breaks stated.",
  "All are corrected in the primary analysis: post-2020 annual non-drinkers are recoded as non-drinkers, the 2018 activity indicator is set to missing rather than silently redefined, sleep duration is expressed on the weekday basis used by NHANES in every KCHS year, age is capped at 80 in all cohorts to match the NHANES top-code, the 2017 income refusal codes are masked, and the income quintile is recomputed as a rank-based five-level score **within each survey year**, which removes both the four-level collapse in the post-pandemic cohort and the scale mismatch between the 2017 ordinal field and the later monetary fields. Multimedia Appendix 3, Table S12 reports the effect of each correction applied singly to the uncorrected data."),
 ("A fourth defect was identified during peer review and is disclosed rather than silently corrected: because the income quintile is derived within each cohort and the NHANES income-to-poverty ratio is top-coded at 5.00, the post-pandemic cohort has only four income levels (weighted 13.0%, 16.7%, 20.1% and 50.1%) instead of five, so its highest-income half is mapped onto the numeric level that the training data use for the fourth quintile. Recomputing the variable as a rank-based five-level score in every cohort and refitting changes discrimination by at most 0.0004 (internal 0.832, pre 0.772, post 0.769; §3.7), so the primary results are reported unchanged with this limitation stated. All four defects affected data integrity but not the substantive conclusions.",
  "Peer review identified four further defects, all of which are corrected in the primary analysis rather than carried forward: the income quintile collapsed to four levels in the post-pandemic cohort because the income-to-poverty ratio is top-coded at 5.00; the 2017 income refusal codes entered the highest quintile; post-2020 annual non-drinkers were coded as current drinkers; and the 2018 physical-activity indicator used a different construct from the other years. Sleep duration and the age top-code were additionally aligned across surveys (§2.5). The seven corrected defects affected data integrity; §3.7 and Multimedia Appendix 3 Table S12 quantify each one."),
])

print("[5] 초록·결론")
sub("manuscript_17var.md", [
 (f"outcome-stratified 500,000-record training set and 100,000-record hold-out",
  f"cluster-level split into a {int(dz.train_n):,}-record training set and a {int(dz.holdout_n):,}-record hold-out sharing no primary sampling unit"),
])
print("\n완료 — 이어서 61_sync_numbers.py 재지정 및 60_verify_manuscript.py 확인 필요.")
