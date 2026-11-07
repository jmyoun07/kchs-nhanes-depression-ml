# -*- coding: utf-8 -*-
"""88_review13_fixes.py — 코덱스 13차 교차검증 지적의 교정 (5건)."""
import io, os, sys
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__))

MS = [
 # ① 계측기가 다른 항목이 있으므로 '동일하게 정의된다' 는 과함
 ("Only variables definable identically in both surveys entered the harmonized model.",
  "Only variables reducible to a common definition in both surveys entered the harmonized model; the instruments behind several of them differ, as set out under Harmonization cautions."),
 # ② 개정 시점의 추출은 175변수였고, 이후 전수 추출에서 조화 가능 항목이 더 나왔다
 ("The harmonized set was assembled by crosswalking all 616 KCHS concepts against the NHANES codebook for both cycles and retaining every concept that is (i) available in NHANES, (ii) collected in more than one KCHS survey year, and (iii) not circular with the PHQ-9 outcome.",
  "The harmonized set was assembled by crosswalking all 616 KCHS concepts against the NHANES codebook for both cycles and retaining every concept that is (i) available in NHANES, (ii) collected in more than one KCHS survey year, and (iii) not circular with the PHQ-9 outcome. The crosswalk available when the amendment was written held a 175-variable extract; an exhaustive extraction afterwards identified 265 variables common to both cycles, and the items it added beyond the seventeen are examined in the post-hoc screening described below."),
 # ③ 부록1 은 주석을 단 영문판이고 원본은 코드와 함께 공개한다
 ("The amendment is released unmodified as Multimedia Appendix 1",
  "The amendment is summarized in English, with contextual notes marked as such, in Multimedia Appendix 1, and the unmodified Korean original is released with the code"),
 # ④ 오라클은 NHANES 라벨로 적합한다 — 초록의 절대 진술을 주모형으로 한정
 ("All model fitting, tuning and algorithm selection used KCHS-internal cross-validation only;",
  "Fitting, tuning and algorithm selection for the primary model used KCHS-internal cross-validation only, the in-domain oracle being a deliberate diagnostic exception;"),
]
CL = [
 ("All model fitting, tuning and algorithm selection used development-data cross-validation only, and the deployment item count follows the analysis plan's margin applied inside the development data;",
  "Fitting, tuning and algorithm selection for the primary model used development-data cross-validation only, the in-domain oracle being a deliberate diagnostic exception, and the deployment item count follows the analysis plan's margin applied inside the development data;"),
 ("the amendment is released unmodified as Multimedia Appendix 1",
  "the amendment is summarized in English in Multimedia Appendix 1, with the unmodified Korean original released with the code"),
]
A4 = [
 # ⑤ 외부 라벨로 선별된 행이 있다
 ("All model selection used KCHS-internal criteria only.",
  "Selection of the primary model, and of every strategy not explicitly marked as externally screened, used KCHS-internal criteria only."),
]

def edit(path, pairs):
    p = f"{D}/{path}"; s = io.open(p, encoding="utf8").read(); n = 0
    for a, b in pairs:
        if a not in s: print(f"  [미일치] {path}: {a[:76]}"); continue
        s = s.replace(a, b, 1); n += 1
    io.open(p, "w", encoding="utf8").write(s); print(f"  {path}: {n}/{len(pairs)}건")

print("코덱스 13차 지적 교정")
for f, pairs in [("docs/manuscript_17var.md", MS), ("docs/cover_letter.md", CL), ("docs/appendix4_ceiling.md", A4)]:
    edit(f, pairs)
