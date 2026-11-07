# -*- coding: utf-8 -*-
"""인용되지 않은 부록 표 넷을 본문에 연결한다.

JMIR 규정: 부록 항목은 '본문 등장 순서'로 번호를 매긴다. 등장하지 않는 표는
번호를 매길 근거가 없고, 심사자가 '왜 있는지' 를 묻는 자리가 된다.
  S2·S6 - 39% 축소 때 인용이 함께 지워졌다(축소 전 원고에는 있었다)
  S1·S12 - 처음부터 인용이 없었다
"""
import io, os, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")
p = "docs/manuscript_17var.md"
s = io.open(p, encoding="utf8").read()

FIX = [
 # S1 - 기준 임계값에서의 성능. 운영점을 처음 서술하는 자리에 붙인다.
 ("A single operating threshold, the predicted probability giving 85% sensitivity (0.02051), was fixed once "
  "in the development data and applied unchanged to the hold-out and both external cohorts",
  "A single operating threshold, the predicted probability giving 85% sensitivity (0.02051), was fixed once "
  "in the development data and applied unchanged to the hold-out and both external cohorts; the Youden and "
  "prevalence-matched reference points are given for comparison in Table S1 in Multimedia Appendix 1",
  "S1 기준 임계값 비교"),
 # S6 - 항목수별 궤적. 항목수를 훈련셋 안에서 고정했다는 문장에 붙인다.
 ("Predictor order and item count were both fixed inside the training set, and the analysis plan's margin",
  "Predictor order and item count were both fixed inside the training set, the full item-count trajectory "
  "being given in Table S6 in Multimedia Appendix 1, and the analysis plan's margin",
  "S6 항목수별 궤적"),
 # S12 - 사전명시 방법탐색 재실행. 누출통제 최적화 문장에 붙인다.
 ("Leakage-controlled optimization (2,199 recorded runs across 117 experiment files",
  "Leakage-controlled optimization, whose pre-specified method search was additionally re-run independently "
  "(Table S12 in Multimedia Appendix 1), covered 2,199 recorded runs across 117 experiment files",
  "S12 사전명시 방법탐색"),
]
n = 0
for a, b, why in FIX:
    if a in s:
        s = s.replace(a, b, 1); n += 1; print(f"  연결: {why}")
    else:
        print(f"  [앵커 없음] {why}: {a[:70]}")

# S2 - 고민감도 운영점. 축소 전 문장을 되살려 붙인다.
A2 = ("respondents screened per case identified, and the model flagged 47% and 45% of the population for "
      "confirmatory assessment")
B2 = ("respondents screened per case identified, and the model flagged 47% and 45% of the population for "
      "confirmatory assessment; operating points tuned to fixed sensitivity targets of 0.85 and 0.90 are "
      "tabulated in Table S2 in Multimedia Appendix 1")
if A2 in s:
    s = s.replace(A2, B2, 1); n += 1; print("  연결: S2 고민감도 운영점")
else:
    print(f"  [앵커 없음] S2: {A2[:70]}")

io.open(p, "w", encoding="utf8").write(s)
print(f"\n{n}/4 연결 완료")

import re
body = io.open(p, encoding="utf8").read().split("## Figure Legends")[0]
order = []
for m in re.finditer(r"Table S(\d+)", body):
    k = int(m.group(1))
    if k not in order: order.append(k)
print("본문 최초 인용 순서:", order)
print("아직 인용 안 된 표:", sorted(set(range(1, 17)) - set(order)))
