# -*- coding: utf-8 -*-
"""A0_round4_fixes.py — 4차 감사 13건(중복 제외 11건)의 교정."""
import io, os, re, sys, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__)); RES = f"{D}/results"
NR = pd.read_csv(f"{RES}/primary2_dpq_nonresponse.csv")
o = lambda p_, c: NR[(NR.period == p_) & (NR.factor == "__overall__")][c].iloc[0]

MS = [
 # ①⑦ 붙여쓰기로 명사구가 중복됐다
 ("and the transportability-ceiling analyses and the transportability-ceiling and calibration-interval analyses (Tables S11–S16).",
  "and the transportability-ceiling and calibration-interval analyses (Tables S11–S16)."),
 # ② MAJOR — 거짓 출처. 0.823/0.810/0.830 은 KCHS 내부 표본크기 실험 값이다
 ("rising to 0.830 only at n=200,000 (best of three learners, 5-fold weighted AUROC, in the same transfer-optimization run).",
  "rising to 0.830 only at n=200,000 (best of three learners, 5-fold weighted AUROC, in a separate KCHS-internal sample-size experiment)."),
 # ③ 한 실행을 두 이름으로 — 표 S11 자신의 이름(exploration run)으로 통일
 ("against the baseline of the transfer-optimization run that anchors Multimedia Appendix 1 Table S11 (0.772 pre, 0.769 post), showing",
  "against the baseline of the exploration run that anchors Multimedia Appendix 1 Table S11 (0.772 pre, 0.769 post), showing"),
 ("These are changes against the in-run baseline of a separate exploration run (0.772 pre, 0.769 post; Multimedia Appendix 1, Table S11) and are not comparable in level with the primary estimates.",
  "These are changes against the in-run baseline of that same exploration run (0.772 pre, 0.769 post) and are not comparable in level with the primary estimates."),
 # ⑧ 결과의 서술과 표현이 같아 서로 다른 통계가 같은 것으로 읽힌다
 (f"cost 7.7% of the pre-pandemic and 14.1% of the post-pandemic sample ({o('pre','anymiss_interview_pct'):.1f}% and {o('post','anymiss_interview_pct'):.1f}% if counted over all interviewed adults)",
  f"cost 7.7% of the pre-pandemic and 14.1% of the post-pandemic sample, counting any unanswered item ({o('pre','anymiss_interview_pct'):.1f}% and {o('post','anymiss_interview_pct'):.1f}% on the wider interview frame; Multimedia Appendix 1, Table S8)"),
 # ⑩ 한 문장 안에서 같은 말을 두 번
 ("and compared at each algorithm's best configuration, so that the three algorithms were compared under one identical protocol. That protocol equalizes",
  "and compared at each algorithm's best configuration. That protocol equalizes"),
 # ⑪ 영국식 철자 (약어표는 center 로 정의돼 있다)
 ("mobile examination centre", "mobile examination center"),
 ("mobile examination centre", "mobile examination center"),
]
A3 = [
 # ⑨ 절 제목을 소문자로 되풀이한 괄호
 ("Parsimonious Deployment Model (parsimonious deployment model, Table S6)", "Parsimonious Deployment Model (Table S6)"),
]
A4 = [
 # ⑬ 표 S15 만 H3 (다른 표는 전부 H2)
 ("### Table S15.", "## Table S15."),
]

def edit(path, pairs):
    p = f"{D}/{path}"; s = io.open(p, encoding="utf8").read(); n = 0
    for a, b in pairs:
        if a not in s: print(f"  [미일치] {path}: {a[:80]}"); continue
        s = s.replace(a, b, 1); n += 1
    io.open(p, "w", encoding="utf8").write(s); print(f"  {path}: {n}/{len(pairs)}건")

print("4차 감사 교정")
edit("docs/manuscript_17var.md", MS)
edit("docs/appendix3_utility.md", A3)
edit("docs/appendix4_ceiling.md", A4)

# 영국식 철자 잔여 일괄 (centre/organisation 등)
BR = [("centre", "center"), ("Centre", "Center"), ("organisation", "organization"),
      ("standardised", "standardized"), ("recognised", "recognized"), ("utilise", "utilize"),
      ("modelling", "modeling"), ("labelled", "labeled")]
import glob
for f in glob.glob(f"{D}/docs/*.md"):
    if "_audit" in f: continue
    s = io.open(f, encoding="utf8").read(); o0 = s
    refs = s.split("## References")[1].split("## Abbreviations")[0] if "## References" in s else ""
    for a, b in BR: s = s.replace(a, b)
    if refs:  # 참고문헌 제목은 원문 그대로 두어야 한다
        s2 = s.split("## References")
        s = s2[0] + "## References" + refs + "## Abbreviations" + s2[1].split("## Abbreviations", 1)[1]
    if s != o0:
        io.open(f, "w", encoding="utf8").write(s); print(f"  {os.path.basename(f)}: 영국식 철자 정리")

# ④ 제출본 README 의 Part C 범위
p = f"{D}/46_build_submission.py"; s = io.open(p, encoding="utf8").read()
a = "Part C 전이 천장 분석(Table S11–S15)"
b = "Part C 전이 천장 분석·보정 구간(Table S11–S16)"
if a in s:
    io.open(p, "w", encoding="utf8").write(s.replace(a, b, 1))
    compile(io.open(p, encoding="utf8").read(), p, "exec")
    print("  46_build_submission.py: Part C 범위 S11–S16")

# ⑤ 일회성 스크립트가 표 S16 을 부록3 에서 찾는다 (이미 옮겨졌다)
p = f"{D}/91_add_intervals.py"; s = io.open(p, encoding="utf8").read()
if 'p = f"{D}/docs/appendix3_utility.md"' in s and "Table S16" in s:
    s = s.replace('p = f"{D}/docs/appendix3_utility.md"', 'p = f"{D}/docs/appendix4_ceiling.md"  # 표 S16 은 Part C 로 이동했다')
    io.open(p, "w", encoding="utf8").write(s); print("  91_add_intervals.py: 표 S16 위치 갱신")
