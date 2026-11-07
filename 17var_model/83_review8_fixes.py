# -*- coding: utf-8 -*-
"""83_review8_fixes.py — 코덱스 8차 교차검증 지적의 교정.

수치가 바뀌는 곳(DPQ 하위군 미응답률)은 82_dpq_subgroup_fix.py 가 다시 계산한
results/primary2_dpq_nonresponse.csv 를 근거로 삼는다.
"""
import io, os, sys, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__))

nr = pd.read_csv(f"{D}/results/primary2_dpq_nonresponse.csv")
def v(period, group):
    r = nr[(nr.period == period) & (nr.group == group)]
    return f"{float(r.nonresponse_ge3_pct.iloc[0]):.1f}"
E = {g: (v("pre", g), v("post", g)) for g in ["<HS", "HS", "College", "Asian", "Black", "White"]}
print("표 정의(≥3문항 결측) 기준 미응답률:", E)

MS = [
 # ⑤ 하위군 미응답률은 표 정의(≥3문항)로 다시 계산한다
 ("higher among respondents with less than high-school education (20.6% pre, 47.3% post)",
  f"higher among respondents with less than high-school education ({E['<HS'][0]}% pre, {E['<HS'][1]}% post)"),
 # ③ '권고 충족' 은 중강도 150분 또는 고강도 75분 단독 기준이다 (혼합 환산 없음)
 ("| 10 | Physical activity | pha_07z1 | PAQ650/665; PAD790/810 | Meets aerobic recommendation |",
  "| 10 | Physical activity | pha_07z1 | PAQ650/665; PAD790/810 | Moderate ≥150 min/week or vigorous ≥75 min/week, each assessed separately |"),
 # ④ 나이 상한을 표 1 정의와 맞춘다 (부록에도 동일 적용)
 # ⑪ PHQ-9 ≥5 는 경도 이상 증상이지 '어떤 증상이든' 이 아니다
 ("any symptom burden; notably, the Korea–US gap narrows", "mild or greater symptom burden; notably, the Korea–US gap narrows"),
 # ⑫ 모델카드에는 선택된 설정만 있고 탐색 격자는 코드에 있다
 ("full search grids are reported in the model card, which is available with the code (see Data Availability).",
  "full search grids are specified in the released analysis code, and the selected configuration is recorded in the model card (see Data Availability)."),
 # ⑬ KCHS 원자료는 공개자료가 아니다
 ("KCHS and NHANES are public, de-identified secondary datasets.",
  "Both analyses use de-identified secondary survey data: NHANES is publicly released by the NCHS, and KCHS microdata are access-controlled and were obtained under a KDCA data-use agreement."),
 # ⑮ 동의 취득 상태 (TRIPOD+AI 17)
 ("No re-identification was attempted, no compensation was involved, and no individually identifiable information is reported.",
  "Written informed consent was obtained from every participant by the data-collecting agencies at the time of the original surveys; the present analysis of de-identified records required no additional consent, and none was sought. No re-identification was attempted, no compensation was involved, and no individually identifiable information is reported."),
 # ⑲ 환자·공중 참여 (TRIPOD+AI 19)
 ("The present secondary analysis was reviewed by the Institutional Review Board of Daejeon University,",
  "Patients and members of the public were not involved in the design, conduct, reporting or dissemination plans of this secondary analysis of existing survey data. The present secondary analysis was reviewed by the Institutional Review Board of Daejeon University,"),
 # ⑱ 적합된 모형 객체의 이용 가능성 (TRIPOD+AI 12g·22)
 ("Analysis code, the variable-harmonization dictionary, the model card, and the protocol, statistical analysis plan and amendment documents are available from the corresponding author upon reasonable request,",
  "Analysis code, the variable-harmonization dictionary, the model card, the fitted LightGBM model object together with the preprocessing pipeline needed to score a new individual, and the protocol, statistical analysis plan and amendment documents are available from the corresponding author upon reasonable request,"),
]
MS_ANCHOR = [
 # ⑯ 표본 크기의 근거 (TRIPOD+AI 10)
 ("**Planned sensitivity analyses.**",
  "**Sample size.** No a priori sample-size calculation was performed. The development sample is the full pooled KCHS 2017-2025 cohort and the external samples are the complete NHANES cycles, that is, all available data rather than a target computed in advance. What that sample supplies is reported instead: the 500,000-record training set carries approximately 18,000 events, about 1,000 per candidate predictor and far above conventional events-per-predictor requirements, and the achieved precision is reported as confidence intervals throughout. The external cohorts are the binding constraint on precision, which is why interval widths, and not point estimates alone, are reported for every external quantity, and why subgroup estimates are presented with intervals and read as exploratory.\n\n**Intended use and users.** The model is intended as a triage aid for population-level screening programmes and web or mobile self-assessment services, to be operated by public-health screening programmes and primary-care services rather than by individuals acting alone, and to prioritize who receives a full PHQ-9 and clinical follow-up. It is not a diagnostic instrument and does not replace clinical assessment. Deployment presupposes local recalibration and access to the 17 (or 6) harmonized items.\n\n**Planned sensitivity analyses.**"),
]
A2 = [
 # ① 나이 상한
 ("| 1 | Age | `age` (all years) | `RIDAGEYR` | `RIDAGEYR` | Continuous; analytic samples restricted to ≥20 y |",
  "| 1 | Age | `age` (all years) | `RIDAGEYR` | `RIDAGEYR` | Continuous, top-coded at 80 y in every cohort; analytic samples restricted to ≥20 y |"),
 # ② 음주 문항 코드
 ("| 9 | Alcohol | `dra_01z1` (+ `drb_01z3`) |",
  "| 9 | Alcohol | `dra_01z1`; frequency `drb_01z2` (2017-2019), `drb_01z3` (2020-2025) |"),
 # ④ 2,071,211 은 연령·PHQ 필터 이전의 원자료 행수다
 ("| Cohort | Adults ≥20 y | With complete PHQ-9 | Unweighted prevalence | Weighted prevalence |",
  "| Cohort | Source records | With complete PHQ-9, aged ≥20 y | Unweighted prevalence | Weighted prevalence |"),
 ("| KCHS 2017–2025 | 2,071,211 | **2,048,510** |", "| KCHS 2017–2025 | 2,071,211 (all ages) | **2,048,510** |"),
]
A3 = [
 # ⑤ 표 S8 하위군 행
 ("| Non-response, less than high school | 20.6% | 47.3% |", f"| Non-response, less than high school | {E['<HS'][0]}% | {E['<HS'][1]}% |"),
 ("| Non-response, high school | 14.1% | 38.8% |", f"| Non-response, high school | {E['HS'][0]}% | {E['HS'][1]}% |"),
 ("| Non-response, college or more | 12.6% | 28.1% |", f"| Non-response, college or more | {E['College'][0]}% | {E['College'][1]}% |"),
 ("| Non-response, non-Hispanic Asian | 20.9% | 46.0% |", f"| Non-response, non-Hispanic Asian | {E['Asian'][0]}% | {E['Asian'][1]}% |"),
 ("| Non-response, non-Hispanic Black | 14.8% | 39.9% |", f"| Non-response, non-Hispanic Black | {E['Black'][0]}% | {E['Black'][1]}% |"),
 ("| Non-response, non-Hispanic White | 13.0% | 29.4% |", f"| Non-response, non-Hispanic White | {E['White'][0]}% | {E['White'][1]}% |"),
 ("Computed from the NHANES DPQ item files; the derivation is released with the analysis code.",
  "Computed from the NHANES DPQ item files under the same ≥3-item definition used for the overall rates; the derivation is released with the analysis code."),
 # ⑥ 7항목 변형은 외부 증거로 선호된 것이 아니다 (pre 는 오르고 post 는 내린다)
 ("a stricter 0.01 margin giving 7 items is reported as an exploratory variant; it was preferred only on external evidence and is therefore not the deployed model (Parsimonious Deployment Model).",
  "a stricter 0.01 margin giving 7 items is reported as an exploratory variant rather than the deployed model, the plan's margin being 0.02; on external data the seventh item does not help consistently, raising pre-pandemic AUROC from 0.748 to 0.757 while lowering post-pandemic AUROC from 0.757 to 0.752 (Parsimonious Deployment Model)."),
 # ⑦ 이 표의 기준행은 교정 전 자료다
 ("| Baseline (as analysed) | 0.8315 | 0.7718 | 0.7688 |", "| Baseline (uncorrected data, before the seven corrections) | 0.8315 | 0.7718 | 0.7688 |"),
]
CL = [
 # ⑧ '동일하게 측정' 은 사실이 아니다
 ("The harmonized predictor set is restricted to items both surveys measure identically.",
  "The harmonized predictor set is restricted to items both surveys measure in a form reducible to a common definition, and the instrument differences that remain are reported item by item."),
 # ⑨ 공통 설정으로 비교한 표가 있다
 ("and every predictor-set comparison was given its own hyperparameter search, so that no comparison is confounded by tuning effort.",
  "and the inferential 14-versus-17-predictor comparison gave each model its own hyperparameter search, so that it is not confounded by tuning effort, while the predictor-set variants tabulated in the appendix share one common configuration."),
 # ⑥ 커버레터도 같은 서술로
 ("a smaller-margin seven-item variant is reported but was preferred on external evidence and is labelled exploratory,",
  "a smaller-margin seven-item variant is reported and labelled exploratory, since it does not improve both external cohorts,"),
 # ⑩ 오라클의 지위를 본문과 맞춘다
 ("The ceiling therefore reflects a limit of the transportable predictor set rather than a modeling failure. That conclusion is established with an in-domain oracle (fitting NHANES on its own labels caps at ~0.77, whereas the same predictors reach ~0.83 within Korea)",
  "The ceiling is therefore best read as a limit of the transportable predictor set rather than as a modeling failure. That reading rests on an in-domain oracle (fitting NHANES on its own labels reaches only ~0.77 across the learners tested, whereas the same predictors reach ~0.83 within Korea), which is an empirical benchmark rather than a proof of a maximum,"),
 # ① 보정 회복도 점추정 기준이다
 ("calibration drifts yet is fully recoverable by lightweight intercept/slope recalibration;",
  "calibration drifts yet is recoverable, on point estimates, by lightweight intercept/slope recalibration;"),
 # ⑬ 공개자료 서술
 ("The study uses public, de-identified secondary data; raw KCHS microdata are access-controlled and not redistributable, while NHANES data are openly available.",
  "The study uses de-identified secondary survey data; NHANES is openly available, while raw KCHS microdata are access-controlled and not redistributable."),
]
CK = [
 # ⑳ 체크리스트의 자기지시 문장을 진술로 바꾼다
 ("*Note: item numbering here is organized to the TRIPOD+AI 2024 domains; the official numbered checklist form should be completed and uploaded at submission. All substantive reporting elements are present in the manuscript as mapped above.*",
  "*Note: this appendix is a completed item-by-item TRIPOD+AI 2024 checklist, with items grouped by reporting domain rather than by the form's running numbers; the journal's official checklist form is uploaded separately with the submission. Every substantive reporting element is located in the manuscript or appendices as mapped above.*"),
]

def edit(path, pairs, anchor=False):
    p = f"{D}/{path}"; s = io.open(p, encoding="utf8").read(); n = 0
    for a, b in pairs:
        if a not in s: print(f"  [미일치] {path}: {a[:76]}"); continue
        s = s.replace(a, b, 1); n += 1
    io.open(p, "w", encoding="utf8").write(s); print(f"  {path}: {n}/{len(pairs)}건")

print("코덱스 8차 지적 교정")
edit("docs/manuscript_17var.md", MS + MS_ANCHOR)
edit("docs/appendix2_crosswalk.md", A2)
edit("docs/appendix3_utility.md", A3)
edit("docs/cover_letter.md", CL)
edit("docs/TRIPOD_AI_checklist.md", CK)
