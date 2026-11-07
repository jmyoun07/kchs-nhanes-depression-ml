# -*- coding: utf-8 -*-
"""91_add_intervals.py — 새로 계산한 두 구간을 문서에 반영한다.

 · 사후−사전 AUROC 차의 구간(primary2_delta_auroc.csv): 주변구간 겹침은 동일성의 근거가 못 된다.
 · 외부 코호트 보정지표의 구간(primary2_calibration_ci.csv): 보정은 1차 주장인데 점추정만 있었다.
"""
import io, os, sys, pandas as pd
from decimal import Decimal, ROUND_HALF_UP
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__)); RES = f"{D}/results"
def R(v, nd=3): return f"{Decimal(str(v)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP):.{nd}f}"
def sgn(v, nd=2):
    s = R(abs(v), nd); return ("+" if float(v) >= 0 else "−") + s

DA = pd.read_csv(f"{RES}/primary2_delta_auroc.csv").iloc[0]
CC = pd.read_csv(f"{RES}/primary2_calibration_ci.csv")
def c(setname, kind, col):
    r = CC[(CC.set == setname) & (CC.probabilities == kind)].iloc[0]; return r[col]
PRE_T, PRE_R = ("NHANES pre", "as transported"), ("NHANES pre", "after intercept/slope recalibration")
POST_T, POST_R = ("NHANES post", "as transported"), ("NHANES post", "after intercept/slope recalibration")

MS = [
 # 초록 — 겹침 대신 차이의 구간을, 그리고 재보정 기울기의 구간을
 ("post 0.771, 0.739–0.801, with overlapping intervals)",
  f"post 0.771, 0.739–0.801; difference {sgn(DA.delta,3)}, 95% CI {sgn(DA.CI_lo,3)} to {sgn(DA.CI_hi,3)})"),
 ("and was restored to slope 1.00 and CITL 0.00 by intercept/slope recalibration, on point estimates; no calibration interval was computed (H2, H3 supported)",
  f"and was restored by intercept/slope recalibration to slope {R(c(*PRE_R,'cal_slope'),2)} (95% CI {R(c(*PRE_R,'slope_lo'),2)}–{R(c(*PRE_R,'slope_hi'),2)}) and "
  f"{R(c(*POST_R,'cal_slope'),2)} ({R(c(*POST_R,'slope_lo'),2)}–{R(c(*POST_R,'slope_hi'),2)}) with CITL 0.00 (H2, H3 supported)"),
 # 방법 — 무엇에 구간이 붙는지
 ("Calibration measures and operating characteristics are point estimates.",
  "Operating characteristics and net benefit are point estimates. Calibration slope, calibration-in-the-large and expected calibration error carry bootstrap intervals in the two external cohorts, both as transported and after recalibration, from the same replicate weights; the difference in external AUROC between the two cycles is likewise given an interval, formed from the difference of the two independently generated replicate distributions."),
 # 고찰 — 점추정만이라는 서술을 갱신
 ("and intercept-plus-slope updating restored both on point estimates, without re-estimating any predictor effect; no interval was computed for the calibration measures.",
  f"and intercept-plus-slope updating restored both without re-estimating any predictor effect, to slope {R(c(*PRE_R,'cal_slope'),2)} ({R(c(*PRE_R,'slope_lo'),2)}–{R(c(*PRE_R,'slope_hi'),2)}) and "
  f"{R(c(*POST_R,'cal_slope'),2)} ({R(c(*POST_R,'slope_lo'),2)}–{R(c(*POST_R,'slope_hi'),2)}); the intervals are wide, as expected at these sample sizes, and both include 1."),
 # 한계 — 보정 기울기 구간이 이제 있다
 ("is derived [45]; no interval was computed for the calibration slope here, so that constraint is stated as a design consideration rather than as something the present intervals demonstrate.",
  f"is derived [45]; the recalibrated slope intervals reported here ({R(c(*PRE_R,'slope_lo'),2)}–{R(c(*PRE_R,'slope_hi'),2)} pre and {R(c(*POST_R,'slope_lo'),2)}–{R(c(*POST_R,'slope_hi'),2)} post) are correspondingly wide, so the update is shown to remove the bias without being pinned down precisely."),
]

TAB2_ADD = (f"| Calibration slope 95% CI | — | {R(c(*PRE_T,'slope_lo'),2)}–{R(c(*PRE_T,'slope_hi'),2)} | {R(c(*PRE_R,'slope_lo'),2)}–{R(c(*PRE_R,'slope_hi'),2)} | "
            f"{R(c(*POST_T,'slope_lo'),2)}–{R(c(*POST_T,'slope_hi'),2)} | {R(c(*POST_R,'slope_lo'),2)}–{R(c(*POST_R,'slope_hi'),2)} |\n")
TAB2_ADD2 = (f"| CITL 95% CI | — | {sgn(c(*PRE_T,'CITL_lo'))}–{sgn(c(*PRE_T,'CITL_hi'))} | {sgn(c(*PRE_R,'CITL_lo'))}–{sgn(c(*PRE_R,'CITL_hi'))} | "
             f"{sgn(c(*POST_T,'CITL_lo'))}–{sgn(c(*POST_T,'CITL_hi'))} | {sgn(c(*POST_R,'CITL_lo'))}–{sgn(c(*POST_R,'CITL_hi'))} |\n")

p = f"{D}/docs/manuscript_17var.md"; s = io.open(p, encoding="utf8").read(); n = 0
for a, b in MS:
    if a not in s: print(f"  [미일치] {a[:76]}"); continue
    s = s.replace(a, b, 1); n += 1
# 표 2 에 보정 구간 행을 넣는다
i = s.index("| Calibration slope |"); j = s.index("\n", i) + 1
k = s.index("| CITL |", j); m = s.index("\n", k) + 1
s = s[:j] + TAB2_ADD + s[j:m] + TAB2_ADD2 + s[m:]
# 결과 본문에 두 코호트 차이를 명시
a = "the intervals overlap"
b = (f"the difference is {sgn(DA.delta,3)} (95% CI {sgn(DA.CI_lo,3)} to {sgn(DA.CI_hi,3)}; bootstrap P={DA.P:.2f}), so the two cycles are not distinguishable at this sample size; the intervals overlap")
if a in s: s = s.replace(a, b, 1); n += 1
io.open(p, "w", encoding="utf8").write(s)
print(f"원고: {n}건 + 표 2 보정 구간 2행")

CK = [("| 21 | Model performance (internal + external, with CIs) | ✅ |",
       "| 21 | Model performance (discrimination and calibration, internal + external, with CIs) | ✅ |")]
p = f"{D}/docs/TRIPOD_AI_checklist.md"; s = io.open(p, encoding="utf8").read()
for a, b in CK:
    if a in s: s = s.replace(a, b, 1)
io.open(p, "w", encoding="utf8").write(s); print("체크리스트 갱신")

# 부록에 표를 하나 더
tab = ["## Table S16. Calibration with cluster-bootstrap intervals (external cohorts)", "",
       "Intervals are the 2.5th and 97.5th percentiles of 1000 rescaled stratified cluster-bootstrap replicates, "
       "the same replicates used for discrimination. Recalibration is intercept/slope updating applied out-of-sample "
       "by 5-fold cross-fitting grouped by primary sampling unit.", "",
       "| Cohort | Probabilities | Calibration slope (95% CI) | CITL (95% CI) | ECE (95% CI) |", "|---|---|---|---|---|"]
for _, r in CC.iterrows():
    tab.append(f"| {r['set']} | {r['probabilities']} | {R(r.cal_slope,2)} ({R(r.slope_lo,2)}–{R(r.slope_hi,2)}) | "
               f"{sgn(r.CITL)} ({sgn(r.CITL_lo)}–{sgn(r.CITL_hi)}) | {R(r.ECE)} ({R(r.ECE_lo)}–{R(r.ECE_hi)}) |")
tab += ["", f"External AUROC differed between cycles by {sgn(DA.delta,3)} (95% CI {sgn(DA.CI_lo,3)} to {sgn(DA.CI_hi,3)}; "
        f"bootstrap P={DA.P:.2f}), formed as the difference of the two independently generated replicate distributions.", ""]
# 이 스크립트는 이미 적용됐다. 표 S16 은 그 뒤 Part C(부록4)로 옮겨졌고 유지는 61_sync_numbers.py 가 한다.
raise SystemExit("적용 완료된 일회성 스크립트 — 다시 실행하지 않는다")
if "Table S16" not in s:
    io.open(p, "w", encoding="utf8").write(s.rstrip() + "\n\n" + "\n".join(tab))
    print("부록3: 표 S16 추가")
