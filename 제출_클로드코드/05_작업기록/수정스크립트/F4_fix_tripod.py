# -*- coding: utf-8 -*-
"""TRIPOD 체크리스트의 위치 서술을 고친다.

원고를 39% 잘라낸 뒤라 '본문 어느 절에 있다' 고 적어 둔 항목의 상당수가 빈 곳을 가리킨다.
방침: 내용이 실제로 있는 곳을 적는다. 부록으로 옮겨간 것은 Part D 를 가리킨다.
원본은 work_notes/tripod_map.json 이고 B8 이 체크리스트를 다시 만든다.
"""
import io, json, os, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")

P = "work_notes/tripod_map.json"
E = json.loads(io.open(P, encoding="utf8").read())
BY = {e["item"]: e for e in E}

NEW = {
 # [15] 개정이 '분석 전에 작성' 이라고 적혀 있었다. 논문이 하지 않는 주장이다
 "18c": ("Methods, Study Design and Data Sources and Data Availability: dated protocol and statistical analysis "
         "plan written before any analysis, plus a dated predictor-set amendment written after the 14-predictor "
         "external result was known but locked before any model using the amended set was fitted; all released "
         "with the analysis code.", "addressed"),
 # [16] 축소 후 본문에 결측 처리 방법이 남아 있지 않다
 "11": ("Methods, Predictors and Harmonization: one preprocessing pipeline fitted in the training set and applied "
        "unchanged, the income quintile excepted; the imputation scheme and the two alternative schemes tested in "
        "sensitivity analysis are in Multimedia Appendix 1, Part D and Table S5.", "addressed"),
 # [17] 네 항목 중 둘이 본문에 없다
 "12b": ("Table 1 (age top-coded at 80 years) and Methods, Predictors and Harmonization (income recomputed as a "
         "within-cohort rank quintile) and Model Development and Evaluation (main-effects representation selected "
         "over four candidate interaction terms); ordinal scores modeled as numeric and standardized, and "
         "continuous predictors median-imputed, in Multimedia Appendix 1, Part D.", "addressed"),
 # [18] 튜닝 부표본과 450,000/50,000 분할이 본문에 없다
 "21": ("Methods, Model Development and Evaluation: training 500,000 (~18,000 events), hold-out 100,001 (~3,600), "
        "external cohorts 7,888 (728 events) and 5,210 (661); the 300,000-record and 200,000-record tuning "
        "subsamples and the 450,000/50,000 parsimony split are in Multimedia Appendix 1, Part D.", "addressed"),
 # [19] 초매개변수 값이 본문에 하나도 없다. 재현성을 묻는 항목이라 가장 중요하다
 "22": ("Methods, Model Development and Evaluation (algorithm, search and selection rule), with the full LightGBM "
        "hyperparameter configuration in Multimedia Appendix 1, Part D and in the model card; Results, "
        "Parsimonious Deployment Model: the six items; Data Availability: preprocessing specification and "
        "refitting code, no public model object.", "addressed"),
 # [20] 의도된 사용자가 지목한 절에 없다
 "3b": ("Introduction (triage rationale for community and primary-care detection), Discussion, Clinical and Policy "
        "Implications (intended use is triage, not diagnosis), and Multimedia Appendix 1, Part D: a triage aid for "
        "population screening programs and web or mobile self-assessment, operated by public-health and "
        "primary-care services rather than by individuals acting alone.", "addressed"),
 "27b": ("Discussion, Clinical and Policy Implications, and Multimedia Appendix 1, Part D: intended users are "
         "public-health and primary-care services operating a screening program, not individuals acting alone; "
         "the model outputs a triage probability and does not diagnose.", "addressed"),
 # [22] 공정성 근거와 소득분위
 "14": ("Methods, External Validation and Sensitivity Analyses: weighted subgroup AUROC and calibration slope "
        "across sex, age, education, household income quintile and, in NHANES, race and ethnicity, with the "
        "rationale stated and a 0.05 AUROC deviation from the cohort value pre-specified as the flag; results in "
        "Results, Interpretability, Fairness, and Clinical Utility and Table S7.", "addressed"),
 # [23] 초록에 연령범위가 없다
 "1": ("Title: identifies development plus external validation, the outcome and the two national-survey "
       "populations; PHQ-9 >=10 is specified in the Abstract and Methods, and the adult age range (20 years or "
       "older) in Methods, Participants and Outcome.", "addressed"),
 # [24] 사전 진단 기록 언급은 Methods 가 아니라 Discussion 에 있다
 "6c": ("Not applicable; no treatments were administered in these observational surveys and no treatment variable "
        "was modeled. Discussion, Strengths and Limitations, and Multimedia Appendix 1, Part D note that a prior "
        "physician diagnosis of depression is not recorded in KCHS, precluding a diagnosis-based exclusion.",
        "not_applicable"),
 # [14 리뷰] 환자·대중 참여 진술을 본문에 넣었다
 "19": ("Methods, Ethical Considerations: patients and members of the public were not involved in the design, "
        "conduct, reporting or dissemination plans of this secondary analysis of existing survey data.",
        "addressed"),
}

n = 0
for k, (loc, st) in NEW.items():
    if k not in BY:
        print(f"  [경고] 항목 {k} 가 매핑에 없다"); continue
    BY[k]["location"] = loc; BY[k]["status"] = st; n += 1
io.open(P, "w", encoding="utf8").write(json.dumps(E, ensure_ascii=False, indent=1))
print(f"tripod_map.json: {n}개 항목 위치 갱신")

# ── [28] 체크리스트 머리말의 TRIPOD+AI 중복 정의 ────────────────────────
q = "B8_tripod_rebuild.py"
s = io.open(q, encoding="utf8").read()
a = ('L.append(f"Abbreviations: TRIPOD+AI, Transparent Reporting of a multivariable prediction model for "\n'
     '         f"Individual Prognosis Or Diagnosis + Artificial Intelligence; {abbr}.")')
b = ('# TRIPOD+AI 는 원고 약어 목록에도 있다. 여기서 또 쓰면 한 문서에 두 가지 확장형이 생긴다.\n'
     'L.append(f"Abbreviations: {abbr}.")')
if a in s:
    io.open(q, "w", encoding="utf8").write(s.replace(a, b, 1))
    print("B8: 머리말의 TRIPOD+AI 중복 정의 제거")
else:
    print("B8: 머리말 앵커 형태가 달라 수동 확인 필요")
