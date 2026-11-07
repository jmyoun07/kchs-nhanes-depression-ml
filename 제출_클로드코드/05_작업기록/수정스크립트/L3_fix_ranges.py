# -*- coding: utf-8 -*-
"""재번호가 망가뜨린 범위 표기를 고치고, 남은 편집 이력 표현을 지운다.

 ① 'Tables S1 to S16'(전체 표를 가리킴)이 재번호로 'Tables S4 to S3' 이 됐다.
    범위의 양끝만 치환했기 때문이다. 전체를 가리키는 표기이므로 다시 S1-S16 으로 쓴다.
 ② 부록 머리말에 'relocated from the Methods' 가 남았다. 심사자가 읽을 표현이 아니다.
"""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")
D = "docs"

# ── ① 역순이 된 범위 ────────────────────────────────────────────────────
n = 0
for f in ("manuscript_17var.md", "appendix2_crosswalk.md", "appendix3_utility.md"):
    p = f"{D}/{f}"; s = io.open(p, encoding="utf8").read()
    s2 = re.sub(r"Tables S4 to S3\b", "Tables S1 to S16", s)
    if s2 != s:
        io.open(p, "w", encoding="utf8").write(s2)
        n += 1; print(f"  {f}: 'Tables S4 to S3' -> 'Tables S1 to S16'")
print(f"① 범위 표기 {n}곳 복구")

# ── ② 편집 이력 표현 ────────────────────────────────────────────────────
SUB = [
 ("the study-design and model-development detail relocated from the Methods",
  "the study-design and model-development detail supporting the Methods"),
 ("Detail relocated from the manuscript Methods so that the main text states what a reader needs "
  "in order to judge the study and this section states what a reader needs in order to reproduce it.",
  "This section states what a reader needs in order to reproduce the analysis, the main text stating "
  "what a reader needs in order to judge it."),
 ("This section carries detail relocated from the Methods subsection Participants and Outcome and supports",
  "This section supports the Methods subsection Participants and Outcome and"),
]
m = 0
for f in ("appendix2_crosswalk.md", "appendix5_methods.md", "appendix3_utility.md", "appendix4_ceiling.md"):
    p = f"{D}/{f}"; s = io.open(p, encoding="utf8").read(); k = 0
    for a, b in SUB:
        if a in s: s = s.replace(a, b); k += 1
    if k: io.open(p, "w", encoding="utf8").write(s); m += k; print(f"  {f}: {k}곳")
print(f"② 편집 이력 표현 {m}곳 정리")

BAD = re.compile(r"relocat|trimmed Methods|condensed out of|in revision|preserved here", re.I)
left = [(p, mm.group(0)) for p in ("appendix2_crosswalk.md", "appendix3_utility.md",
                                   "appendix4_ceiling.md", "appendix5_methods.md")
        for mm in BAD.finditer(io.open(f"{D}/{p}", encoding="utf8").read())]
print(f"\n잔존: {left if left else '없음'}")
