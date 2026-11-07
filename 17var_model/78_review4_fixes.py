# -*- coding: utf-8 -*-
"""78_review4_fixes.py — 코덱스 4차 교차검증 지적의 교정."""
import io, os, sys
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__))

MS = [
 # ② NHANES 정보를 전혀 쓰지 않았다는 서술은 개정 시점과 어긋난다
 ("**No NHANES data were used in any selection or tuning step.**",
  "**No NHANES data were used in any model-fitting, hyperparameter-tuning or algorithm-selection step.** "
  "The predictor set is the one exception: three of the seventeen items entered through a dated amendment "
  "written after the external result for the earlier set was known, as set out under Provenance of the "
  "Harmonized Predictor Set."),
 # ⑤ Table S5 는 이 문장이 나열한 분석을 모두 담고 있지 않다
 ("**Planned sensitivity analyses.** Pre-specified and additional analyses (exact values in Multimedia Appendix 1, Table S5):",
  "**Planned sensitivity analyses.** Pre-specified and additional analyses, of which the specifications that alter the model, "
  "the predictor set or the analytic population are tabulated in Multimedia Appendix 1, Table S5, and the remainder are reported "
  "in the Results text:"),
 # ⑦ 참고문헌 18 은 2015-2018 만 뒷받침한다
 ("the sample design and estimation procedures are documented for the 2015-2018 cycles [18], and the same framework applies to the pre-pandemic file and the 2021-2023 cycle used here;",
  "the sample design and estimation procedures for the 2015-2018 cycles are documented by NCHS [18]; the pre-pandemic combined file and the 2021-2023 cycle carry separate NCHS documentation and weight variables, which are named in Multimedia Appendix 1;"),
]
A3 = [
 # ⑥ 부록 머리 문장·본문의 표 번호
 ("Interpretability, Fairness, and Clinical Utility (fairness and clinical utility, Tables S1, S4, S5 and S9)",
  "Interpretability, Fairness, and Clinical Utility (subgroup discrimination and calibration in Table S7; operating characteristics and net benefit in Tables S1 to S3)"),
 ("The operating point is reported both at the recommended sensitivity ≥0.85 and at the alternative Youden point (Tables S1 and S4).",
  "The operating point is reported both at the recommended sensitivity ≥0.85 (Table S2) and at the alternative Youden point (Table S1)."),
 ("three alternative alcohol definitions are reported in Table S5 and change external discrimination by ≤0.003",
  "three alternative alcohol definitions are reported in the Results text and change external discrimination by ≤0.003"),
 # ④ Table S3 캡션: 내부 코호트는 재보정하지 않는다
 ("Net benefit at threshold probabilities of 0.10 and 0.15 using intercept/slope–recalibrated probabilities, with recalibration applied **out-of-sample by 5-fold cross-fitting**, as specified in External Validation and Sensitivity Analyses.",
  "Net benefit at threshold probabilities of 0.10 and 0.15. The two external cohorts use intercept/slope-recalibrated probabilities, with recalibration applied out-of-sample by 5-fold cross-fitting grouped by primary sampling unit; the internal hold-out uses the original probabilities, since recalibrating on its own labels would be optimistic."),
]
A4 = [
 # ③ 외부 라벨을 본 행을 실제로 표시
 ("| Add public-assistance receipt (welfare) |", "| Add public-assistance receipt (welfare), externally screened |"),
 ("| Add welfare + 12-mo welfare |", "| Add welfare + 12-mo welfare, externally screened |"),
 ("| Add welfare + weekend sleep |", "| Add welfare + weekend sleep, externally screened |"),
]
CK = [
 # ① hold-out 은 100,001
 ("(500,000 training / 100,000 hold-out from 2,048,510 pooled)", "(500,000 training / 100,001 hold-out from 2,048,510 pooled)"),
 # ② 누수 통제 체크의 한정
 ("| 18 | Leakage control / avoidance of optimistic selection | ✅ | Model Development, Provenance of the Harmonized Predictor Set, Parsimonious Deployment Model |",
  "| 18 | Leakage control / avoidance of optimistic selection | ✅ | Model Development, Provenance of the Harmonized Predictor Set, Parsimonious Deployment Model. Model fitting, tuning and algorithm selection used development data only; the predictor-set amendment is the documented exception and is dated |"),
]

def edit(path, pairs):
    p = f"{D}/{path}"; s = io.open(p, encoding="utf8").read(); n = 0
    for a, b in pairs:
        if a not in s: print(f"  [미일치] {path}: {a[:74]}"); continue
        s = s.replace(a, b); n += 1
    io.open(p, "w", encoding="utf8").write(s); print(f"  {path}: {n}/{len(pairs)}건")

print("코덱스 4차 지적 교정")
edit("docs/manuscript_17var.md", MS)
edit("docs/appendix3_utility.md", A3)
edit("docs/appendix4_ceiling.md", A4)
edit("docs/TRIPOD_AI_checklist.md", CK)

# ④ 그림 7 도 내부 코호트는 재보정하지 않는다 (표 S3 과 같은 기준)
p = f"{D}/73_figures_primary2.py"; s = io.open(p, encoding="utf8").read()
a = "    pr = recal_is(yv, p, w, groups=psu_groups(d)); W = w.sum(); prev = w[yv == 1].sum()/W"
b = ("    # 내부 hold-out 은 재보정하지 않는다(자기 라벨 재보정은 낙관적, 표 S3·70 과 같은 기준)\n"
     "    pr = p if nm.startswith(\"KCHS\") else recal_is(yv, p, w, groups=psu_groups(d))\n"
     "    W = w.sum(); prev = w[yv == 1].sum()/W")
assert a in s, "그림7 재보정 호출부 미일치"
io.open(p, "w", encoding="utf8").write(s.replace(a, b, 1))
compile(io.open(p, encoding="utf8").read(), p, "exec")
print("  73_figures_primary2.py: 그림 7 내부 코호트 재보정 제외")
