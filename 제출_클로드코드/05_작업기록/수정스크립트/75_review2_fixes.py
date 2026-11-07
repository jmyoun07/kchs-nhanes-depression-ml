# -*- coding: utf-8 -*-
"""75_review2_fixes.py — 코덱스 2차 교차검증 지적의 교정.

가장 중요한 것: 61_sync_numbers.py 의 경량모델 궤적 정규식이 소수점을 문장 끝으로 오인해
문장을 이어붙여 망가뜨렸다(지적 2). 여기서 문장을 복구하고 정규식을 고친다.
"""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__))
P = f"{D}/docs/manuscript_17var.md"
s = io.open(P, encoding="utf8").read()

# ── 지적 2: 깨진 궤적 문장 복구 (이후 61이 정본값으로 다시 채운다) ──
m = re.search(r"On the selection set, discrimination rose steeply and then flattened:.*?(?= Applying the margin set)", s, re.S)
assert m, "궤적 문장을 찾지 못함"
s = s[:m.start()] + "On the selection set, discrimination rose steeply and then flattened: 1 item 0.000; 3 items 0.000; 5 items 0.000; 6 items 0.000; 8 items 0.000; 12 items 0.000; 17 items 0.000." + s[m.end():]
print("2) 깨진 궤적 문장 복구 (자리표시자 → 61 이 정본값 주입)")

FIX = [
 # 1) 초록의 경량모델 외부 성능
 ("selected a 6-item model (0.822 internal, 0.747 pre, 0.760 post; H4 supported), with no smaller-margin variant reported so that no external information entered the choice",
  "selected a 6-item model (0.822 internal, 0.748 pre, 0.757 post; H4 supported); the stricter-margin seven-item variant is reported only as an exploratory comparison, and no external information entered the item-count decision"),
 # 5) 측정이 동일하다는 서술이 남아 있던 곳
 ("The predictor set was restricted to items both surveys measure identically, with every inclusion decision documented and dated",
  "The predictor set was restricted to items both surveys measure in a form harmonizable to a common definition, with every inclusion decision documented and dated"),
 # 8) 분할은 군집 무작위이지 결과층화가 아니다
 ("outcome-stratified 500,000-record training set and 100,000-record hold-out",
  "cluster-level split into a 500,000-record training set and a 100,001-record hold-out sharing no primary sampling unit"),
 # 12) 결측지시자는 연속형에만 붙는다
 ("Alternative strategies (missing indicators added for continuous and binary predictors;",
  "Alternative strategies (missing indicators added for the continuous predictors;"),
 # 13) 온도조정과 Platt 조정을 혼동
 ("intercept only, intercept plus slope, and temperature (Platt) scaling, each cross-fitted over five folds.",
  "intercept only, intercept plus slope (the two-parameter Platt-type update), and one-parameter temperature scaling with the intercept fixed at zero, each cross-fitted over five folds."),
 # 6) 교차조사 SHAP 은 외부 라벨을 쓰지 않으므로 '관계의 구조' 를 말할 수 없다
 ("indicating that the two populations use the predictors in a similar order; this suggests the transportability limit lies in the *strength*, not the *structure*, of predictor–outcome associations.",
  "indicating that the transported model draws on the predictors in a similar order in both populations. This is a statement about how the fixed model attributes risk, not about the external predictor–outcome associations, since the attribution uses no external outcome; it therefore constrains, without establishing, the interpretation that the limit lies in the strength rather than the structure of those associations."),
 ("a pattern consistent with the shift-stability framing of deployment failure in health artificial intelligence [",
  "a pattern that is at least consistent with the shift-stability framing of deployment failure in health artificial intelligence ["),
 # 7) 오라클 해석의 과장
 ("The transported model therefore already performs at the level that the best achievable model on those predictors attains inside the external population, which locates the limit in the predictors rather than in the transfer.",
  "The transported model therefore already performs at the level reached by the best model that could be fitted on those predictors inside the external population, among the learners evaluated. That comparison is pooled across cycles while the transported estimates are cycle-specific, and the intervals overlap; it therefore locates the limit in the predictors rather than in the transfer only to the extent that the oracle is a fair upper bound."),
 # 9) '진정한' 외부검증이라는 표현
 ("this study provides what prior work did not: genuine cross-national",
  "this study provides what prior work did not: cross-national"),
 # 10) 알고리즘 선택 교차검증은 군집 단위가 아니다
 ("Second, all three algorithms were tuned over comparable 5-configuration grids under an identical protocol",
  "Algorithm and hyperparameter selection used individual-level cross-validation within the training set; the cluster-level separation applies to the hold-out and to the parsimony selection set; the selection step may therefore be mildly optimistic while the reported performance is not. Second, all three algorithms were tuned over comparable 5-configuration grids under an identical protocol"),
]
n = 0
for a, b in FIX:
    if a not in s: print(f"  [미일치] {a[:74]}"); continue
    s = s.replace(a, b); n += 1
io.open(P, "w", encoding="utf8").write(s)
print(f"본문 {n}/{len(FIX)}건 교정")

# ── 부록·체크리스트 ──
A = {
 "docs/appendix2_crosswalk.md": [
  ("2018 (activity derived from the remaining components)",
   "2018 (set to missing in the primary analysis; the module lacks the moderate-activity day count)")],
 "docs/appendix3_utility.md": [
  ("intercept+slope only, cross-fitted", "all three variants run and reported, each cross-fitted by primary sampling unit")],
 "docs/TRIPOD_AI_checklist.md": [
  ("Tables S1, S4–S5", "Tables S1–S3"),
  ("Tables S6 and S10", "Table S4; depression-module non-response in Table S8 and Multimedia Appendix 1 Part A"),
  ("| 33 | Use of generative AI disclosed | ✅ | Acknowledgments |", "| 33 | Use of generative AI disclosed | ✅ | Use of Generative AI Tools |")],
}
for f, pairs in A.items():
    p = f"{D}/{f}"; t = io.open(p, encoding="utf8").read(); k = 0
    for a, b in pairs:
        if a not in t: print(f"  [미일치] {f}: {a[:60]}"); continue
        t = t.replace(a, b); k += 1
    io.open(p, "w", encoding="utf8").write(t); print(f"  {f}: {k}/{len(pairs)}건")
