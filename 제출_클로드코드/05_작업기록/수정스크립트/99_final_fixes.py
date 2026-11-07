# -*- coding: utf-8 -*-
"""99_final_fixes.py — 최종 감사 13건(중복 제외 9건)의 교정."""
import io, os, re, sys, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__)); RES = f"{D}/results"
NR = pd.read_csv(f"{RES}/primary2_dpq_nonresponse.csv")
o = lambda p_, c: NR[(NR.period == p_) & (NR.factor == "__overall__")][c].iloc[0]
AM_PRE, AM_POST = o("pre", "anymiss_interview_pct"), o("post", "anymiss_interview_pct")
print(f"any-item 면접프레임: pre {AM_PRE}% · post {AM_POST}%")

MS = [
 # ① MAJOR — 같은 통계여야 하는데 정의가 달랐다 (any-item 대 ≥3문항)
 (f"cost 7.7% of the pre-pandemic and 14.1% of the post-pandemic sample ({o('pre','nr3_interview_pct'):.1f}% and {o('post','nr3_interview_pct'):.1f}% if counted over all interviewed adults)",
  f"cost 7.7% of the pre-pandemic and 14.1% of the post-pandemic sample ({AM_PRE:.1f}% and {AM_POST:.1f}% if counted over all interviewed adults)"),
 # ② 한 문장 안에 같은 단서가 두 번
 ("(*P*<.001 for both; paired rescaled cluster bootstrap, each model given its own hyperparameter search)",
  "(*P*<.001 for both; paired rescaled cluster bootstrap)"),
 # ③ 연속한 두 문장이 같은 설명을 반복
 ("Under the common configuration of Table S5, where the 17-, 16- and 14-predictor models share one setting, removing food insecurity alone accounts for 28% of the 17-versus-14 gap pre-pandemic and 50% post-pandemic. Table S5 in Multimedia Appendix 1 reports the same predictor-set variants under one common configuration, where the 14-predictor value is 0.758 post-pandemic.",
  "Under the common configuration of Multimedia Appendix 1 Table S5, where the 17-, 16- and 14-predictor models share one setting and the 14-predictor value is 0.758 post-pandemic, removing food insecurity alone accounts for 28% of the 17-versus-14 gap pre-pandemic and 50% post-pandemic."),
 # ④ 한 문장 안에서 같은 실행을 두 이름으로
 ("still yielded 0.823 at n=7,888 (the pre-pandemic cohort size) and 0.810 at n=13,098 (the pooled oracle size), rising to 0.830 only at n=200,000 (best of three learners, 5-fold weighted AUROC, an exploration run separate from the primary analysis).",
  "still yielded 0.823 at n=7,888 (the pre-pandemic cohort size) and 0.810 at n=13,098 (the pooled oracle size), rising to 0.830 only at n=200,000 (best of three learners, 5-fold weighted AUROC, in the same transfer-optimization run)."),
 # ⑦ 파일코드 정정은 사전지정 문서에 대한 것이지 표 1 에 대한 것이 아니다
 ("self-rated health was harmonized to NHANES HUQ010 (correcting a Table 1 file-code error for HSD010/HSQ)",
  "self-rated health was harmonized to NHANES HUQ010 (correcting a file-code error in the pre-specification document, which had named HSD010/HSQ)"),
 # ⑬ TRIPOD 3b 의 상위 절 이름
 ("Intended use and users, under Model Development and Evaluation",
  "Intended use and users, under External Validation and Sensitivity Analyses"),
]

def edit(path, pairs):
    p = f"{D}/{path}"; s = io.open(p, encoding="utf8").read(); n = 0
    for a, b in pairs:
        if a not in s: print(f"  [미일치] {path}: {a[:80]}"); continue
        s = s.replace(a, b, 1); n += 1
    io.open(p, "w", encoding="utf8").write(s); print(f"  {path}: {n}/{len(pairs)}건")

print("최종 교정")
edit("docs/manuscript_17var.md", MS[:5])
edit("docs/TRIPOD_AI_checklist.md", MS[5:])

# ⑩⑪ 철자·약어 표기를 미국식 / AMA 로 통일
SP = [("behavioural", "behavioral"), ("Behavioural", "Behavioral"),
      ("i.e., ", "ie, "), ("i.e. ", "ie, "), ("e.g., ", "eg, "), ("e.g. ", "eg, ")]
for f in ("docs/manuscript_17var.md", "docs/appendix2_crosswalk.md", "docs/appendix3_utility.md",
          "docs/appendix4_ceiling.md", "docs/cover_letter.md", "docs/model_card_17var.md",
          "docs/TRIPOD_AI_checklist.md"):
    p = f"{D}/{f}"; s = io.open(p, encoding="utf8").read(); o0 = s
    for a, b in SP: s = s.replace(a, b)
    if s != o0:
        io.open(p, "w", encoding="utf8").write(s)
        print(f"  {f}: 철자·약어 표기 통일")

# ⑤ 근거대장 주석이 없는 열 이름을 가리킨다
p = f"{D}/87_make_ledger.py"; s = io.open(p, encoding="utf8").read()
a = '"`nonresponse_ge3_pct` 가 표 정의(≥3문항 결측)이고, `any_missing_pct` 는 참고용이다."'
b = ('"`nr3_*` 열이 ≥3문항 결측률(면접·MEC·MEC가중 세 프레임), `anymiss_interview_pct` 는 1문항 이상 결측률이다. "\n'
     '      "표 S8 의 완전례 제외 행(7.7%/14.1%)은 MEC 프레임의 1문항 이상 결측이다."')
if a in s:
    io.open(p, "w", encoding="utf8").write(s.replace(a, b, 1))
    compile(io.open(p, encoding="utf8").read(), p, "exec")
    print("  87_make_ledger.py: 열 설명 정정")

# ⑫ 조립된 부록에서 표 번호가 S10 → S16 → S11 순이 된다. S16 을 Part C 끝으로 옮긴다.
p3 = f"{D}/docs/appendix3_utility.md"; p4 = f"{D}/docs/appendix4_ceiling.md"
s3 = io.open(p3, encoding="utf8").read()
if "## Table S16." in s3:
    i = s3.index("## Table S16.")
    block = s3[i:].rstrip() + "\n"
    io.open(p3, "w", encoding="utf8").write(s3[:i].rstrip() + "\n")
    s4 = io.open(p4, encoding="utf8").read().rstrip() + "\n\n" + block
    io.open(p4, "w", encoding="utf8").write(s4)
    print("  표 S16 을 부록4(Part C) 끝으로 이동 — 조립 순서가 S1–S16 이 된다")
