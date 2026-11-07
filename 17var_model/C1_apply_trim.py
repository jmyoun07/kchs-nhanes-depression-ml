# -*- coding: utf-8 -*-
"""축소안을 적용한다. 감사 지적 25건을 반영한 뒤 한 번에 넣는다.

지적 25건의 성격
  · 배치 12건 - 전부 같은 원인. 이전 내용을 Part B(임상유용성)에 넣으려 했으나
    Part B 는 범위 문장이 Results 특정 절만 지원한다고 명시한다. Part D 로 보낸다.
  · 문장 13건 - 축소하면서 좁혀지거나 사라진 것들. 감사자가 준 문구로 되돌린다.

에이전트를 다시 부르지 않는다. 지적 내용을 이미 받아뒀으므로 여기서 끝낸다.
"""
import io, json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")

J = json.loads(io.open("work_notes/trim.json", encoding="utf8").read())
SEC = {s["key"]: s for s in J["sections"]}

# ══ 1. 문장 복원 13건 ═══════════════════════════════════════════════════
# (섹션, 대상필드, 찾을 문자열, 바꿀 문자열, 지적번호)
FIX = [
 # ── intro ──
 ("intro", "replacement",
  "Most such models are assessed through internal validation alone, in depression modeling and in other clinical domains alike [7,8].",
  "Most published prediction models are assessed through internal validation alone, in depression modeling and in other clinical domains alike [7,8].",
  "intro-3 주어가 우울 모형으로 좁혀졌다. [8] 을 여기 인용하는 근거는 그 폭이다"),
 ("intro", "replacement",
  "Separating the two requires more than a single external estimate.",
  "Separating the two requires more than a single external estimate. It requires an estimate of what the same "
  "predictors can achieve when a model is fitted directly inside the external population, together with a modeling "
  "search in which no external label is used for selection.",
  "intro-5 다섯째 문단의 방법론적 논증이 통째로 사라졌다"),
 ("intro", "appendix_addition",
  "A review of the several hundred prediction models published for the diagnosis and prognosis of COVID-19 [8]",
  "A review of hundreds of prediction models published during the COVID-19 pandemic [8]",
  "intro-1,2 [8] 의 대상 집합과 수량이 바뀌어 Discussion 과 어긋난다"),
 ("intro", "appendix_addition",
  "The Introduction states that most published depression prediction models are assessed through internal validation alone",
  "The Introduction states that most published prediction models are assessed through internal validation alone",
  "intro-4 부록의 재진술이 본문보다 좁다"),
 # ── design ──
 ("design", "appendix_addition", "(Table S11)", "(Multimedia Appendix 1, Table S11)",
  "design-3 Table S11 은 Part C 에 있다. 다른 Part 의 표는 전체 이름으로 인용한다"),
 # ── predictors ──
 ("predictors", "replacement",
  "a concept was retained when available in NHANES, collected in more than one KCHS survey year, and neither circular with nor contaminated by the PHQ-9 outcome, an exclusion fixed a priori.",
  "concepts were retained that are (i) available in NHANES, (ii) collected in more than one KCHS survey year, and "
  "(iii) not circular with the PHQ-9 outcome. Circular or outcome-contaminated variables were excluded a priori.",
  "predictors-4 잠긴 개정문서의 기준 (iii) 보다 넓은 기준을 개정문서에 귀속시킨다"),
 ("predictors", "replacement", "265 common variables", "265 variables common to both cycles",
  "predictors-5 '무엇에 공통인지' 가 빠지면 265 가 무엇을 센 값인지 알 수 없다"),
 ("predictors", "appendix_addition",
  "Estimates quoted here are complex-survey weighted unless labeled otherwise.\n", "",
  "predictors-6 원문이 가중이라 하지 않은 0.01%·23.8% 까지 가중 추정치로 바꾼다"),
 # ── model ──
 ("model", "appendix_addition",
  "are named in no earlier dated document", "are not named in any earlier document",
  "model-1 'dated' 를 넣으면 '날짜 없는 문서에는 있었을 수 있다' 가 되어 사전명시 주장이 약해진다"),
 ("model", "replacement",
  "as a diagnostic rather than a candidate model", "as a diagnostic upper bound rather than a candidate model",
  "model-2 오라클의 '상한' 성격이 빠지면 천장 논증의 근거가 사라진다"),
 # ── external ──
 ("external", "replacement",
  "the count was chosen on selection-set AUROC by the analysis plan's margin (the smallest set within 0.02 AUROC of the full model).",
  "the count was chosen on selection-set AUROC by the analysis plan's margin (the smallest set within 0.02 AUROC of "
  "the full model); a variant using a stricter 0.01 margin is flagged as post hoc (Multimedia Appendix 1, Part D).",
  "external-1 0.01 마진이 본문에서 사라져 Results 의 'stricter-margin' 을 해석할 수 없다"),
 ("external", "replacement",
  "All seven were corrected in the primary analysis rather than carried forward as limitations;",
  "The four peer-review defects were corrected in the primary analysis rather than carried forward as limitations;",
  "external-2 감사 3건과 동료검토 4건의 교정 시점 구분이 뭉개진다"),
]

applied, missing = [], []
for key, field, a, b, why in FIX:
    t = SEC[key][field]
    if a in t:
        SEC[key][field] = t.replace(a, b, 1); applied.append((key, why))
    else:
        missing.append((key, field, a[:70], why))

# ── predictors: 사라진 두 문장을 되살린다(치환이 아니라 추가) ──────────────
RESTORE = [
 ("predictors", "replacement", "Three harmonization cautions attach to the set.",
  "Three harmonization cautions attach to the set. BMI is self-reported in KCHS, measured in NHANES.",
  "predictors-1 Sensitivity Analyses 가 '조화 주의사항의 BMI' 를 가리키는데 그 문장이 사라졌다"),
 ("predictors", "appendix_addition", None,
  "\n\nThe amendment is summarized in English, with contextual notes marked as such, and the unmodified Korean "
  "original is released with the code, together with the crosswalk.",
  "predictors-2 영문본에 '표시된 추가 주석' 이 있다는 투명성 진술이 어디에도 남지 않았다"),
]
for key, field, a, b, why in RESTORE:
    t = SEC[key][field]
    if a is None:
        SEC[key][field] = t.rstrip() + b; applied.append((key, why)); continue
    if a in t and b.split(". ")[1][:20] not in t:
        SEC[key][field] = t.replace(a, b, 1); applied.append((key, why))
    else:
        missing.append((key, field, str(a)[:70], why))

# ── predictors-3: 순서형 3변수 진술을 한정 없이 되돌린다 ──────────────────
_p = SEC["predictors"]["replacement"]
_m = re.search(r"in the gradient-boosting primary model[^.]*ordinal scores[^.]*unordered categories\.", _p)
if _m:
    SEC["predictors"]["replacement"] = _p[:_m.start()] + (
        "Household income (within-survey quintile), self-rated health and self-rated oral health are ordinal scores "
        "and were modeled as numeric variables rather than as unordered categories, and were median-imputed and "
        "standardized.") + _p[_m.end():]
    applied.append(("predictors", "predictors-3 순서형 3변수가 '경사부스팅에서만'·'선형비교모형에서만' 으로 쪼개졌다"))
else:
    missing.append(("predictors", "replacement", "순서형 3변수 문장", "predictors-3"))

print(f"문장 복원 적용 {len(applied)}건 · 앵커 불일치 {len(missing)}건\n")
for k, w in applied: print(f"  [{k:<10}] {w}")
if missing:
    print("\n── 앵커를 못 찾은 것 (직접 확인 필요) ──")
    for k, f, a, w in missing: print(f"  [{k}/{f}] {w}\n      찾던 문자열: {a}")

io.open("work_notes/trim_fixed.json", "w", encoding="utf8").write(
    json.dumps({"sections": list(SEC.values())}, ensure_ascii=False, indent=1))
print("\nwork_notes/trim_fixed.json 저장 - 적용은 C2 에서")
