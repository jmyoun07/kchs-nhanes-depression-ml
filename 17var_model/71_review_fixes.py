# -*- coding: utf-8 -*-
"""71_review_fixes.py — 코덱스 교차검증(리뷰어 시점)에서 나온 지적의 본문 교정.

번호는 코덱스 보고서의 항목 번호다. 계산이 필요한 13·15번(재보정 교차적합의 군집 분리,
경량모델 선택셋의 군집 분리)은 별도 재계산 스크립트에서 다룬다.
"""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__))

MS = [
 # ① ECE 값이 정본과 불일치 (0.0032/0.0302/0.0676)
 ("ECE: expected calibration error; excellent internally (0.004) with drift on external application (0.032 pre, 0.067 post), recoverable by recalibration.",
  "ECE: expected calibration error; excellent internally (0.003) with drift on external application (0.030 pre, 0.068 post), recoverable by recalibration."),
 # ② 재보정 전후 AUROC 차이는 최대 0.004 (pre 0.7650 → 0.7612)
 ("the recalibrated AUROC may differ trivially (by ≤0.003) from the original",
  "the recalibrated AUROC may differ trivially (by ≤0.002) from the original"),
 # ④ 소득분위 4단계 붕괴는 교정 전 상태
 ("hence the top two quintiles collapse and the variable takes only four levels there.",
  "hence the top two quintiles would collapse to four levels if the raw ratio were binned directly; the primary analysis instead recomputes a rank-based five-level score within each cohort, which restores five levels."),
 # ⑤ '완전히 동일하게 측정' 은 본문의 측정 차이 서술과 모순
 ("All 17 predictors are defined identically in both surveys; the KCHS column gives one representative variable name per concept",
  "All 17 predictors are harmonized to a common definition in both surveys, and the harmonized definition is the one used in the model; the instruments behind alcohol, physical activity, income, body mass index and food insecurity differ between surveys, as detailed under Harmonization cautions. The KCHS column gives one representative variable name per concept"),
 ("comprising the sociodemographic, behavioural and health items that both surveys measure identically",
  "comprising the sociodemographic, behavioural and health items that both surveys measure in a form harmonizable to a common definition"),
 ("a predictor set restricted to items both surveys measure identically",
  "a predictor set restricted to items both surveys measure in a form harmonizable to a common definition"),
 # ⑥ 방법과 결과의 직접 모순
 ("No smaller-margin variant is reported; hence no external information entered the item-count decision.",
  "The stricter-margin seven-item variant is reported only as an exploratory comparison (Multimedia Appendix 1, Table S6); no external information entered the item-count decision itself."),
 # ⑦ 보정곡선 방향이 반대
 ("Points below the diagonal indicate under-prediction of absolute risk.",
  "Points above the diagonal indicate under-prediction of absolute risk, that is, observed risk higher than predicted."),
 # ⑧ 두 주기 차이에 대한 검정을 하지 않았다
 ("post 0.771, 0.739–0.801, statistically indistinguishable)",
  "post 0.771, 0.739–0.801, with substantially overlapping intervals)"),
 ("the two cycles are statistically indistinguishable; hence discrimination did not deteriorate across the pandemic.",
  "the two intervals overlap substantially; no formal test of the between-cycle difference was pre-specified; the data are therefore consistent with no deterioration across the pandemic but cannot exclude a modest change."),
 ("the two cycles are statistically indistinguishable; hence the model did not decay across the population-wide shock that motivated the temporal analysis",
  "the intervals overlap substantially, and no between-cycle test was pre-specified; the data are therefore consistent with no decay across the population-wide shock that motivated the temporal analysis"),
 # ⑨ 초록 결론이 본문의 유보보다 강하다
 ("and is therefore a property of the transportable predictor set rather than a modeling failure",
  "and is most consistent with a property of the transportable predictor set rather than a modeling failure"),
 # ⑪ 절단점 비교로 '심한 쪽에 집중' 을 단정할 수 없다
 ("suggesting that the transportability loss is concentrated at the more severe end of the symptom distribution",
  "which is consistent with, but does not establish, a larger cross-national difference at the more severe end of the symptom distribution"),
 # ⑫ 미사용 항목 추가 검사는 외부 라벨을 본 진단이다
 ("and (iii) adding the remaining harmonized-but-unused survey items did not help:",
  "and (iii) adding the remaining harmonized-but-unused survey items did not help, a check evaluated on external data and therefore reported as a diagnostic rather than as leakage-controlled evidence:"),
 # ⑭ 재보정 '완전 복구' 는 점추정만의 판단
 ("and was fully restored by intercept/slope recalibration (slope→0.99, CITL→0.00; H2, H3 supported)",
  "and was restored to slope 0.99 and CITL 0.00 by intercept/slope recalibration, on point estimates, since no interval was computed for the calibration measures (H2, H3 supported)"),
 ("intercept-plus-slope updating restored both without re-estimating any predictor effect",
  "intercept-plus-slope updating restored both on point estimates, without re-estimating any predictor effect; no interval was computed for the calibration measures"),
 # ⑯ 표 번호 인용 오류 (재번호 때 놓침)
 ("Full operating characteristics and net benefit are in Multimedia Appendix 1, Tables S1, S4 and S5.",
  "Full operating characteristics and net benefit are in Multimedia Appendix 1, Tables S1, S2 and S3."),
 # ⑱ 사전등록 여부를 본문에 명시
 ("The study protocol and statistical analysis plan, including the four hypotheses, were written and dated before any analysis.",
  "The study protocol and statistical analysis plan, including the four hypotheses, were written and dated before any analysis. The study was not registered in a public registry; the protocol, the analysis plan and the amendment are dated documents released with the analysis code."),
 # ⑲ 눈가림·측정 시점
 ("Adults ≥20 years with all nine PHQ-9 items answered.",
  "Adults ≥20 years with all nine PHQ-9 items answered. Predictors and outcome were recorded in the same survey contact; hence predictor measurement was contemporaneous with outcome measurement; blinding of outcome or predictor assessment does not apply to secondary analysis of already-collected survey responses."),
 # ⑳ 최종 모형 사양을 본문에 보고
 ("the selected hyperparameters and the full search grids are reported in the model card, which is available with the code (see Data Availability).",
  "the selected configuration was num_leaves 31, min_child_samples 1000, learning_rate 0.01, n_estimators 900, reg_alpha 1.0, reg_lambda 20, colsample_bytree 0.6, and subsample 0.6 with subsample_freq at its default of 0, which leaves row subsampling inactive; the full search grids are reported in the model card, which is available with the code (see Data Availability)."),
]
A3 = [
 # ③ 경량모델은 6항목 (0.02 여유 규칙)
 ("The 5-item deployment model and the 7-item post hoc variant are both shown alongside the full 17-predictor model.",
  "The 6-item deployment model and the 7-item post hoc variant are both shown alongside the full 17-predictor model."),
 ("the parsimony margin follows the plan's recommended 0.02, giving a 5-item deployment model",
  "the parsimony margin follows the plan's recommended 0.02, giving a 6-item deployment model"),
 # ⑰ 세 가지 재보정은 모두 실행되었다
 ("The intercept-only and Platt/temperature variants were not run",
  "All three variants were run and are reported"),
]

def edit(path, pairs):
    p = f"{D}/{path}"; s = io.open(p, encoding="utf8").read(); n = 0
    for a, b in pairs:
        if a not in s: print(f"  [미일치] {path}: {a[:78]}"); continue
        s = s.replace(a, b); n += 1
    io.open(p, "w", encoding="utf8").write(s); print(f"  {path}: {n}/{len(pairs)}건")

print("코덱스 지적 교정")
edit("docs/manuscript_17var.md", MS)
edit("docs/appendix3_utility.md", A3)
