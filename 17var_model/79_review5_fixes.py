# -*- coding: utf-8 -*-
"""79_review5_fixes.py — 코덱스 5차 교차검증 지적의 교정."""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__))

MS = [
 # ⑤ living_alone 은 NHANES pre 에 아예 없다 — '두 조사가 모두 측정' 은 사실이 아니다
 ("All 17 predictors are harmonized to a common definition in both surveys, and the harmonized definition is the one used in the model;",
  "Sixteen of the 17 predictors are harmonized to a common definition in both surveys and in both NHANES cycles; living alone is the exception, being unavailable in the pre-pandemic NHANES file and therefore entirely missing there and handled natively by the tree learner. The harmonized definition is the one used in the model;"),
 ("comprising the sociodemographic, behavioural and health items that both surveys measure in a form harmonizable to a common definition",
  "comprising sociodemographic, behavioural and health items harmonizable to a common definition, one of which (living alone) is unavailable in the pre-pandemic NHANES file"),
 # ⑥ 전처리를 '그대로 적용' 했다는 서술과 소득분위 재계산이 어긋난다
 ("Preprocessing fitted in the development data and applied unchanged externally",
  "Imputation and scaling were fitted in the development data and applied unchanged externally; the household income quintile is the exception, being a within-cohort rank score recomputed in each dataset, since the underlying income instruments are not on a common scale"),
 # ⑦ 오라클은 NHANES 라벨로 적합한다 — 절대 진술을 주모형으로 한정
 ("**No NHANES data were used in any model-fitting, hyperparameter-tuning or algorithm-selection step.**",
  "**No NHANES data were used in fitting, tuning or selecting the primary model.** The in-domain oracle reported under Why External Discrimination Plateaus at ~0.77 is fitted on NHANES labels by design, as a diagnostic upper bound rather than as a candidate model."),
 # ⑧ 같은 문단 안의 상충
 ("covariate imbalance, sample size, algorithm class and hyperparameters were tested and none accounts for the gap",
  "covariate imbalance, sample size, algorithm class and hyperparameters were tested and none closed the gap"),
 # ④ 14변수 사후값의 출처를 분명히
 ("lowers external discrimination to 0.745 (pre) and 0.757 (post)",
  "lowers external discrimination to 0.745 (pre) and 0.757 (post) when each model is given its own hyperparameter search"),
 ("Removing food insecurity alone leaves most of that difference intact (Multimedia Appendix 1, Table S5).",
  "Removing food insecurity alone leaves most of that difference intact. Table S5 in Multimedia Appendix 1 reports the same predictor-set variants under one common configuration, where the 14-predictor value is 0.758 post-pandemic."),
]
A2 = [
 # ⑬ 절 이름 치환이 만든 깨진 문장
 ("Supports Methods the Methods and the Methods.",
  "Supports the Methods sections Predictors and Harmonization and Study Design and Data Sources."),
 # ② 설계 규모는 분석에 쓰인 연도×층 기준으로
 ("| Design scale | 5,729 strata; 197,117 stratum–PSU combinations | 24 strata; 49 PSUs | 15 strata; 30 PSUs |",
  "| Design scale | 5,729 design strata; crossed with survey year as analysed, 50,537 strata and 199,546 stratum–PSU combinations | 24 strata; 49 PSUs | 15 strata; 30 PSUs |"),
]
A3 = [
 # ③ hold-out 은 100,001
 ("500,000 training and 100,000 hold-out drawn from the 2,048,510 pooled records",
  "500,000 training and 100,001 hold-out drawn from the 2,048,510 pooled records"),
 # ⑨ '모든 추정치가 가중' 은 사실이 아니다
 ("All estimates complex-survey weighted.",
  "Estimates are complex-survey weighted unless a row or column is labelled unweighted; the depression-module non-response counts and the algorithm-selection cross-validated AUROCs are unweighted."),
 # ⑪ '다섯 가지 미공개 이탈' 서술이 표 내용과 맞지 않는다
 ("Five further departures from the plan's recommended defaults are recorded here for completeness, none of which was previously disclosed in the manuscript text:",
  "Further points where the analysis departed from, or fully met, the plan's recommended defaults are recorded here for completeness; those also stated in the manuscript are repeated so that the disposition of every rule can be read in one place:"),
]
A4 = [
 # ⑩ 표에는 여덟 가지가 있다
 ("An additional seven leakage-free strategies were run",
  "An additional eight leakage-free strategies were run"),
]
CK = [
 # ⑫ 기존 모형 문단이 인용하는 참고문헌
 ("Background, rationale, references to existing models", "Background, rationale, references to existing models (refs 5-8)"),
]

def edit(path, pairs):
    p = f"{D}/{path}"; s = io.open(p, encoding="utf8").read(); n = 0
    for a, b in pairs:
        if a not in s: print(f"  [미일치] {path}: {a[:74]}"); continue
        s = s.replace(a, b); n += 1
    io.open(p, "w", encoding="utf8").write(s); print(f"  {path}: {n}/{len(pairs)}건")

print("코덱스 5차 지적 교정")
for f, pairs in [("docs/manuscript_17var.md", MS), ("docs/appendix2_crosswalk.md", A2),
                 ("docs/appendix3_utility.md", A3), ("docs/appendix4_ceiling.md", A4),
                 ("docs/TRIPOD_AI_checklist.md", CK)]:
    edit(f, pairs)

# ① 그림 7: 내부 패널의 범례·제목이 재보정이라고 잘못 말한다
p = f"{D}/73_figures_primary2.py"; s = io.open(p, encoding="utf8").read()
s = s.replace('ax.plot(ts, nb, color=c, lw=1.8, label="Model (recalibrated)")',
              'ax.plot(ts, nb, color=c, lw=1.8,\n            label=("Model (original)" if nm.startswith("KCHS") else "Model (recalibrated)"))', 1)
s = s.replace('fig.suptitle("Decision-curve analysis (weighted, cross-fit recalibration)", fontsize=11, fontweight="bold")',
              'fig.suptitle("Decision-curve analysis (weighted; external cohorts after cross-fit recalibration)",\n             fontsize=11, fontweight="bold")', 1)
io.open(p, "w", encoding="utf8").write(s)
compile(io.open(p, encoding="utf8").read(), p, "exec")
print("  73_figures_primary2.py: 그림 7 범례·제목 정정")

# 원고의 그림 7 범례도 같은 기준으로
P = f"{D}/docs/manuscript_17var.md"; t = io.open(P, encoding="utf8").read()
t = t.replace("Probabilities are recalibrated out-of-sample by 5-fold cross-fitting. The model is clinically useful",
              "The two external cohorts use probabilities recalibrated out-of-sample by 5-fold cross-fitting grouped by primary sampling unit; the internal hold-out uses the original probabilities. The model is clinically useful")
t = t.replace("decision-curve net benefit exceeded both screen-all and screen-none defaults in every dataset after local recalibration",
              "decision-curve net benefit exceeded both screen-all and screen-none defaults in every dataset, with the external cohorts taken after local recalibration")
io.open(P, "w", encoding="utf8").write(t)
print("  docs/manuscript_17var.md: 그림 7 범례·본문 서술 정정")
