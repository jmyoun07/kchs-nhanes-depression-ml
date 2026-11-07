# -*- coding: utf-8 -*-
"""96_audit_fixes_B.py — living_alone 통제 결과와 중첩 재보정 구간을 문서에 싣는다.

중첩 구간이 고정 구간보다 **좁다**. 이유를 숨기지 않고 쓴다: 재표본마다 갱신을 다시 적합하면
갱신이 그 표본의 편차를 스스로 흡수하므로, 절차를 통째로 반복했을 때의 기울기 분포는 1 근처로
모인다. 두 구간은 서로 다른 질문에 답하므로 둘 다 싣고 무엇에 조건부인지 밝힌다.

ECE 는 비음수·상향편의 통계라 백분위 구간이 점추정을 포함하지 않는 경우가 생겼다(사전 재보정
후 점추정 0.0096 < 하한 0.0097). 구간을 싣지 않고 점추정만 보고한다.
"""
import io, os, sys, pandas as pd
from decimal import Decimal, ROUND_HALF_UP
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__)); RES = f"{D}/results"
def R(v, nd=3): return f"{Decimal(str(v)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP):.{nd}f}"
def sgn(v, nd=2):
    return ("+" if float(v) >= 0 else "−") + R(abs(v), nd)

LA = pd.read_csv(f"{RES}/primary2_livingalone_control.csv")
l17, l16 = LA.iloc[0], LA.iloc[1]
CC = pd.read_csv(f"{RES}/primary2_calibration_ci.csv")
def row(setname, kind): return CC[(CC.set == setname) & (CC.probabilities == kind)].iloc[0]
NEST = "after intercept/slope recalibration (nested)"
FIX = "after intercept/slope recalibration"
np_, ns_ = row("NHANES pre", NEST), row("NHANES post", NEST)
fp_, fs_ = row("NHANES pre", FIX), row("NHANES post", FIX)

LA_TXT = (
 f"**Living alone as a cycle-specific predictor.** Living alone is recorded in every KCHS year and in the "
 f"post-pandemic NHANES cycle but is absent from the pre-pandemic file, and it has no missing values in the "
 f"training data, so the learner routes the missing pre-pandemic column to the same branch as \"not living alone\" "
 f"(Predictors and Harmonization). The pandemic-period contrast therefore mixes a period difference with a "
 f"predictor-availability difference. Refitting the model on the 16 predictors observed in both cycles removes "
 f"that difference and leaves the contrast unchanged: external AUROC {R(l16.ext_pre)} pre-pandemic and "
 f"{R(l16.ext_post)} post-pandemic, a difference of {sgn(l16.delta,3)} (95% CI {sgn(l16.CI_lo,3)} to "
 f"{sgn(l16.CI_hi,3)}) against {sgn(l17.delta,3)} ({sgn(l17.CI_lo,3)} to {sgn(l17.CI_hi,3)}) for the "
 f"17-predictor model, with internal discrimination {R(l16.internal)} against {R(0.836454)}. The availability "
 f"difference therefore does not account for the cross-cycle result.")

MS = [
 # 중첩 구간을 초록에
 (f"restored by intercept/slope recalibration to slope {R(fp_.cal_slope,2)} (95% CI {R(fp_.slope_lo,2)}–{R(fp_.slope_hi,2)}) and {R(fs_.cal_slope,2)} ({R(fs_.slope_lo,2)}–{R(fs_.slope_hi,2)}) with CITL 0.00",
  f"restored by intercept/slope recalibration to slope {R(np_.cal_slope,2)} (95% CI {R(np_.slope_lo,2)}–{R(np_.slope_hi,2)}) and {R(ns_.cal_slope,2)} ({R(ns_.slope_lo,2)}–{R(ns_.slope_hi,2)}) with CITL 0.00"),
 # 방법: ECE 구간 약속을 거둔다 + 두 구간의 의미를 밝힌다
 ("Ninety-five percent CIs are reported for AUROC and AUPRC, and, in the two external cohorts, for the calibration slope, CITL and ECE both as transported and after recalibration;",
  "Ninety-five percent CIs are reported for AUROC and AUPRC and, in the two external cohorts, for the calibration slope and CITL both as transported and after recalibration. Expected calibration error is reported as a point estimate only: it is a non-negative absolute-deviation statistic that the replicate scheme biases upward, so a percentile interval for it is not trustworthy and in one cell does not even contain its own point estimate. For the recalibrated columns the interval is obtained by refitting the intercept and slope inside every replicate, so that the uncertainty of the update itself is included; that interval is narrower than one computed with the update held fixed, since refitting lets each replicate absorb its own miscalibration, and both are given in Multimedia Appendix 1, Table S16."),
 # 고찰의 재보정 구간
 (f"to slope {R(fp_.cal_slope,2)} ({R(fp_.slope_lo,2)}–{R(fp_.slope_hi,2)}) and {R(fs_.cal_slope,2)} ({R(fs_.slope_lo,2)}–{R(fs_.slope_hi,2)}); the intervals are wide, as expected at these sample sizes, and both include 1.",
  f"to slope {R(np_.cal_slope,2)} ({R(np_.slope_lo,2)}–{R(np_.slope_hi,2)}) and {R(ns_.cal_slope,2)} ({R(ns_.slope_lo,2)}–{R(ns_.slope_hi,2)}), with the update refitted inside every replicate so that its own uncertainty is carried."),
 # 한계의 구간 문장
 (f"the recalibrated slope intervals reported here ({R(fp_.slope_lo,2)}–{R(fp_.slope_hi,2)} pre and {R(fs_.slope_lo,2)}–{R(fs_.slope_hi,2)} post) are correspondingly wide; the update is therefore shown to remove the bias without being pinned down precisely.",
  f"conditioning on a single fitted update rather than refitting it gives visibly wider slope intervals ({R(fp_.slope_lo,2)}–{R(fp_.slope_hi,2)} pre and {R(fs_.slope_lo,2)}–{R(fs_.slope_hi,2)} post, Multimedia Appendix 1 Table S16); a deployment that fits its update once on a modest local sample is in that wider regime."),
 # 표 S16 을 실제로 인용 + 부록 목록에 넣는다
 ("(Tables S11–S15)", "(Tables S11–S15); and the calibration intervals (Table S16)"),
]

def edit(path, pairs):
    p = f"{D}/{path}"; s = io.open(p, encoding="utf8").read(); n = 0
    for a, b in pairs:
        if a not in s: print(f"  [미일치] {path}: {a[:90]}"); continue
        s = s.replace(a, b, 1); n += 1
    io.open(p, "w", encoding="utf8").write(s); print(f"  {path}: {n}/{len(pairs)}건")

print("독립감사 교정 B (통제 대조 · 중첩 구간)")
edit("docs/manuscript_17var.md", MS)

# living_alone 통제를 민감도 절에 넣는다
p = f"{D}/docs/manuscript_17var.md"; s = io.open(p, encoding="utf8").read()
if "Living alone as a cycle-specific predictor" not in s:
    a = "Response-propensity weighting gave the same answer"
    i = s.index(a)
    s = s[:i] + LA_TXT + "\n\n" + s[i:]
    io.open(p, "w", encoding="utf8").write(s); print("  living_alone 통제 대조 문단 추가")
