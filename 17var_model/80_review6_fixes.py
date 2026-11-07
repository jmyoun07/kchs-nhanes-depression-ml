# -*- coding: utf-8 -*-
"""80_review6_fixes.py — 코덱스 6차 교차검증 지적의 교정 (9건, 모두 코드·자료로 재확인)."""
import io, os, sys
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__))

MS = [
 # ② 나이 상한 80 은 KCHS 에도 적용된다 (70_primary_corrected.py 의 correct() 가 두 자료 모두에 clip)
 ("age is capped at 80 years in NHANES, whereas 3.8% of the weighted KCHS sample is older (up to 110); hence the model learns splits above 80 that no external respondent can reach;",
  "age is capped at 80 years in NHANES, whereas 3.8% of the weighted KCHS sample is older (up to 110); the same cap was therefore applied to KCHS before fitting, so that the model cannot separate respondents above 80 in either survey and that part of the Korean age range is compressed;"),
 # ③ 결측표시자는 연속변수에만 붙는다 (72_sensitivity_primary2.py 의 ct("indicator"))
 ("alternative imputation (missing indicators for continuous and binary predictors; single chained-equations imputation of continuous predictors)",
  "alternative imputation (missing indicators added to the continuous predictors, with binary predictors left to the learner's native missing handling; single chained-equations imputation of continuous predictors)"),
 # ④ 5-10 항목 상한은 탐색에 강제되지 않았다 (70_primary_corrected.py: k<17 전체를 탐색)
 ("The item count follows the analysis plan's margin (the smallest set within 0.02 AUROC of the full model, subject to a 5-10 item ceiling);",
  "The item count follows the analysis plan's margin (the smallest set within 0.02 AUROC of the full model). The search ran over every count below 17 and did not impose the plan's preferred 5-10 item range, which the selected count nonetheless satisfies;"),
 ("Applying the margin set in the analysis plan (the smallest set within 0.02 AUROC of the full model, subject to a 5-10 item ceiling) selects six items:",
  "Applying the margin set in the analysis plan (the smallest set within 0.02 AUROC of the full model, searched over every count below 17 without imposing the plan's preferred 5-10 item range) selects six items:"),
 # ⑤ '두 주기 모두에서 이용 가능한 개념을 모두 보유' 는 living alone 에서 성립하지 않는다
 ("retaining every concept that is (i) available in both NHANES cycles, (ii) collected in more than one KCHS survey year, and (iii) not circular with the PHQ-9 outcome.",
  "retaining every concept that is (i) available in NHANES, (ii) collected in more than one KCHS survey year, and (iii) not circular with the PHQ-9 outcome. Living alone is the single retained item not present in both cycles, being absent from the pre-pandemic file."),
 # ⑧ KCHS 는 이상지질혈증 등도 묻는다 — 단일 연도라 쓸 수 없을 뿐이다
 ("KCHS, however, records only hypertension and diabetes among physician-diagnosed conditions, both already included,",
  "KCHS, however, records only hypertension and diabetes among physician-diagnosed conditions across more than one survey year, both already included (dyslipidemia, osteoporosis, cataract and arthritis are fielded in a single year and therefore cannot enter a model pooled over 2017-2025),"),
 # ⑨ 인용 7편은 7개 영역을 뒷받침할 뿐 17개 변수 전부를 뒷받침하지 않는다
 ("Every harmonized predictor has an established association with depression in the epidemiological literature: socioeconomic position [21], sleep duration [22], physical activity [23], body mass index [24], smoking [25], living alone [26] and partnership status [27].",
  "The harmonized set is drawn from domains with established associations with depression in the epidemiological literature: socioeconomic position [21], sleep duration [22], physical activity [23], body mass index [24], smoking [25], living alone [26] and partnership status [27]."),
]
A3 = [
 # ⑥ 사후 확인이었다는 본문 서술과 맞춘다
 ("LightGBM and random forest fell inside the tie band; hence the tie-break decided the choice.",
  "LightGBM and random forest fell inside the tie band, to which the tie-break applies. The pipeline as executed ranked the algorithms on cross-validated AUROC and then on the calibration slope without constructing the band; the plan's rule was therefore verified retrospectively, and it reproduces the model the pipeline selected."),
]
A4 = [
 # ① 표 S13 의 기준행은 주모형이 아니라 탐색 실행의 값이다 (정본은 0.765/0.771)
 ("| Gradient boosting (reference) | 0.837 | 0.769 | 0.769 | reference |",
  "| **Gradient boosting (exploration-run reference)** | 0.837 | 0.769 | 0.769 | reference |"),
 ("Four neural architectures were trained on KCHS and transferred without refitting, with model/architecture selection by KCHS-internal validation only.",
  "Four neural architectures were trained on KCHS and transferred without refitting, with model/architecture selection by KCHS-internal validation only. The gradient-boosting reference row is the same exploration run that anchors Table S11, and its external values (0.769/0.769) therefore differ slightly from the primary model's 0.765/0.771; the comparison is internally consistent within the run and should not be read against the primary estimates."),
 # ⑦ 오라클은 NHANES 안에서 적합·교차검증되므로 '전이 가능한 신호' 를 증명하지 않는다
 ("Multimorbidity and healthcare utilization carry additional, transportable signal in NHANES (oracle 0.765→0.778).",
  "Multimorbidity and healthcare utilization carry additional in-domain signal in NHANES (oracle 0.765→0.778). Table S14 is fitted and cross-validated entirely within pooled NHANES; this is therefore incremental discrimination inside that population rather than evidence that the signal would transport, the healthcare-visit item is also carried as pre-cycle only, and its contribution cannot be separated from cycle differences."),
]
CK = [
 # ⑥ 체크리스트도 같은 표현으로
 ("Model Development (3 algorithms, each tuned under one identical internal-CV protocol; a-priori selection)",
  "Model Development (3 algorithms, each tuned under one identical internal-CV protocol; selection rule pre-specified, with the tie band verified retrospectively)"),
]

def edit(path, pairs):
    p = f"{D}/{path}"; s = io.open(p, encoding="utf8").read(); n = 0
    for a, b in pairs:
        a = a.replace("5-10", "5–10"); b = b.replace("5-10", "5–10")
        if a not in s: print(f"  [미일치] {path}: {a[:76]}"); continue
        s = s.replace(a, b); n += 1
    io.open(p, "w", encoding="utf8").write(s); print(f"  {path}: {n}/{len(pairs)}건")

print("코덱스 6차 지적 교정")
for f, pairs in [("docs/manuscript_17var.md", MS), ("docs/appendix3_utility.md", A3),
                 ("docs/appendix4_ceiling.md", A4), ("docs/TRIPOD_AI_checklist.md", CK)]:
    edit(f, pairs)
