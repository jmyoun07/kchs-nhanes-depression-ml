# -*- coding: utf-8 -*-
"""89_review14_fixes.py — 코덱스 14차 교차검증 지적의 교정 (8건)."""
import io, os, sys
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__))

MS = [
 # ① '기준을 만족하는 개념을 모두 보유' 는 사후 전수추출 결과와 어긋난다
 ("retaining every concept that is (i) available in NHANES, (ii) collected in more than one KCHS survey year, and (iii) not circular with the PHQ-9 outcome. The crosswalk available when the amendment was written held a 175-variable extract; an exhaustive extraction afterwards identified 265 variables common to both cycles, and the items it added beyond the seventeen are examined in the post-hoc screening described below.",
  "retaining concepts that are (i) available in NHANES, (ii) collected in more than one KCHS survey year, and (iii) not circular with the PHQ-9 outcome. The crosswalk available when the amendment was written held a 175-variable extract, and the seventeen predictors are every eligible concept in that extract. An exhaustive extraction afterwards identified 265 variables common to both cycles; the further eligible items it surfaced are listed and tested under Post-hoc candidate screening below, and none entered the model, so the deployed set is exhaustive with respect to the extract available at the time of the amendment rather than with respect to the full crosswalk."),
 # ② 부록1 은 주석을 단 영문 요약이다
 ("it is dated, released unmodified (Multimedia Appendix 1), and its contribution is reported as a robustness analysis rather than a primary result.",
  "it is dated, summarized in English in Multimedia Appendix 1 with the unmodified Korean original released alongside the code, and its contribution is reported as a robustness analysis rather than a primary result."),
 # ⑤ 사전 탐색이 LightGBM 에만 있었으므로 '동일 노력' 이라 단정할 수 없다
 ("no algorithm was advantaged by differential tuning effort, an equal-effort design motivated by evidence that machine learning shows no systematic advantage over regression once validation bias is controlled [32].",
  "the three algorithms were compared under one identical protocol. That protocol equalizes the final comparison but not the whole development history, since only gradient boosting received the preceding 24-configuration search; the comparison is therefore reported as a like-for-like contest at each algorithm's tuned best rather than as equal cumulative effort. The design is motivated by evidence that machine learning shows no systematic advantage over regression once validation bias is controlled [32]."),
 # ④ 표 S10 이 무엇을 담고 무엇을 담지 않는지 분명히
 ("Multimedia Appendix 1 Table S10 quantifies each correction applied singly to the uncorrected data.",
  "Multimedia Appendix 1 Table S10 quantifies, one at a time, the six corrections that can be applied singly to the KCHS analytic data: the three questionnaire-driven ones above, the sleep and age alignments, and the income-quintile rebuild. The two NHANES-side audit corrections are not tabulated separately, since without them the affected variable is constant rather than mismeasured, and the 2017 income refusal codes are shown as their own row."),
]
CK = [
 # ③ 오라클 예외를 체크리스트에도
 ("Model fitting, tuning and algorithm selection used development data only; the predictor-set amendment is the documented exception and is dated |",
  "Fitting, tuning and algorithm selection for the primary model used development data only; the predictor-set amendment is the documented exception and is dated, and the in-domain oracle is fitted on NHANES labels by design as a diagnostic rather than as a candidate model |"),
]
A3 = [
 # ⑥ 내부 코호트는 재보정하지 않는다
 ("After local recalibration, net bene",
  "For the two external cohorts after local recalibration, and for the internal hold-out on its original probabilities, net bene"),
]
A4 = [
 # ⑧ 한쪽 코호트에서 기준선을 넘은 행은 있다
 ("None exceeded the pre-specified LightGBM baseline once selection was restricted to KCHS-internal information.",
  "None improved both external cohorts once selection was restricted to KCHS-internal information; a few rows exceed the baseline in one cohort while falling below it in the other, which is not a gain selectable without external labels."),
 # ⑦ 같은 문단에서 '학습기 최대가 아니다' 와 '최고 성능' 이 부딪힌다
 ("these values are not the learner maximum reported in Table S15. This estimates the best in-domain performance obtained by the learners tested when fitted directly to NHANES. It is an empirical benchmark over those learners, not a proof of a maximum.",
  "these values are not the learner maximum, which is reported separately in Table S15 (0.776 for the 17-predictor set against 0.765 here). This table therefore compares predictor sets at a fixed learner pair, and Table S15 gives the best in-domain performance obtained by any learner tested; both are empirical benchmarks over the learners run, not proofs of a maximum."),
]

def edit(path, pairs):
    p = f"{D}/{path}"; s = io.open(p, encoding="utf8").read(); n = 0
    for a, b in pairs:
        if a not in s: print(f"  [미일치] {path}: {a[:76]}"); continue
        s = s.replace(a, b, 1); n += 1
    io.open(p, "w", encoding="utf8").write(s); print(f"  {path}: {n}/{len(pairs)}건")

print("코덱스 14차 지적 교정")
for f, pairs in [("docs/manuscript_17var.md", MS), ("docs/TRIPOD_AI_checklist.md", CK),
                 ("docs/appendix3_utility.md", A3), ("docs/appendix4_ceiling.md", A4)]:
    edit(f, pairs)
