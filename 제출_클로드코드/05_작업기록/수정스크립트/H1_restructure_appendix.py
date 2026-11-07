# -*- coding: utf-8 -*-
"""부록을 예시파일 형식으로 재구성한다.

지금은 Part A~D 와 A.1~B.6 절번호로 굵은 제목이 51개다. 예시파일은 2개다.
예시는 '보충 방법' 산문을 앞에 두고 Table S1 부터 번호순으로 죽 잇는다.

바꾸는 것
 · Part A~D 껍데기 제거, A.1~B.6 절번호 제거
 · 큰 제목 둘만 남긴다: Supplementary Methods / Supplementary Results
 · 산문(크로스워크·개정문서·보충방법)을 앞에, 표 S1~S16 을 번호순으로 뒤에
 · 'Multimedia Appendix 1, Part X' 참조를 'Multimedia Appendix 1' 로 통일
"""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")
D = "docs"

def rd(f): return io.open(f"{D}/{f}", encoding="utf8").read()
def wr(f, s): io.open(f"{D}/{f}", "w", encoding="utf8").write(s)

# ── ① 크로스워크·개정문서: Part A 껍데기와 절번호 제거 ───────────────────
s = rd("appendix2_crosswalk.md")
head = ("## Supplementary Methods\n\n"
        "This appendix carries the material needed to reproduce and interpret the analysis but too detailed "
        "for the main text: the year-specific KCHS-NHANES harmonization crosswalk, the predictor-set amendment, "
        "the study-design and model-development detail relocated from the Methods, and the supplementary "
        "results tabulated as Tables S1 to S16. Estimates are complex-survey weighted unless a row or column "
        "is labeled unweighted.\n\n"
        "### Harmonization crosswalk\n")
s = re.sub(r"^## Part A\..*?\n", "", s, count=1, flags=re.M)
s = re.sub(r"^## A\. Harmonization crosswalk\s*$", head.rstrip(), s, count=1, flags=re.M)
s = re.sub(r"^### [AB]\.\d+ ", "#### ", s, flags=re.M)
s = re.sub(r"^## B\. Predictor-set amendment", "### Predictor-set amendment", s, count=1, flags=re.M)
wr("appendix2_crosswalk.md", s.lstrip("\n"))
print("① 크로스워크: Part A 제거 · 절번호 A.1~B.6 제거 · 'Supplementary Methods' 로 시작")

# ── ② 보충방법: Part D 껍데기 제거, 한 급 내려 Methods 아래로 ────────────
s = rd("appendix5_methods.md")
s = re.sub(r"^## Part D\. Supplementary methods\s*\n+", "", s, count=1, flags=re.M)
# 첫 문단(범위 설명)은 위 머리말과 겹치므로 뺀다
s = re.sub(r"^Detail relocated from the manuscript Methods so that.*?\n\n", "", s, count=1, flags=re.S)
s = re.sub(r"^### ", "### ", s, flags=re.M)
wr("appendix5_methods.md", "### Study design, model development, and analysis detail\n\n" + s.lstrip("\n"))
print("② 보충방법: Part D 제거 · 중복 머리말 제거")

# ── ③ 임상유용성: Part B 껍데기 -> Supplementary Results ────────────────
s = rd("appendix3_utility.md")
s = re.sub(r"^## Part B\..*?\n", "", s, count=1, flags=re.M)
s = ("## Supplementary Results\n\n"
     "Tables S1 to S16 give the exact values behind the figures and the analyses summarized in the main text. "
     + s.lstrip("\n"))
s = re.sub(r"^## Interpretation\s*$", "### Interpretation", s, flags=re.M)
wr("appendix3_utility.md", s)
print("③ 임상유용성: Part B 제거 · 'Supplementary Results' 로 시작")

# ── ④ 천장분석: Part C 껍데기 제거(표가 이어진다) ────────────────────────
s = rd("appendix4_ceiling.md")
s = re.sub(r"^## Part C\..*?\n\n?", "", s, count=1, flags=re.M)
s = re.sub(r"^## Interpretation\s*$", "### Interpretation", s, flags=re.M)
wr("appendix4_ceiling.md", s.lstrip("\n"))
print("④ 천장분석: Part C 제거 (표 S11~S16 이 이어짐)")

# ── ⑤ 모든 문서의 Part 참조를 통일 ──────────────────────────────────────
FILES = ["manuscript_17var.md", "TRIPOD_AI_checklist.md", "model_card_17var.md", "cover_letter.md",
         "appendix2_crosswalk.md", "appendix3_utility.md", "appendix4_ceiling.md", "appendix5_methods.md"]
SUB = [
    (r"Multimedia Appendix 1,? Parts? [A-D](?: and [A-D])?", "Multimedia Appendix 1"),
    (r"Multimedia Appendix 1 Parts? [A-D](?: and [A-D])?", "Multimedia Appendix 1"),
    (r"\bsection [AB]\.\d+ above\b", "the crosswalk above"),
    (r"\bsection [AB]\.\d+\b", "the crosswalk"),
    (r"(?<![A-Za-z])Parts? [A-D](?= (?:additionally|lists|provides|holds|gives|carries))", "The crosswalk"),
    (r"\bin Part [A-D]\b", "in Multimedia Appendix 1"),
    (r"\bthe Part [A-D] preamble\b", "the preamble"),
]
tot = 0
for f in FILES:
    s = rd(f); n = 0
    for pat, rep in SUB:
        s, k = re.subn(pat, rep, s); n += k
    # 같은 문장에 'Multimedia Appendix 1' 이 두 번 남는 경우 정리
    s = re.sub(r"Multimedia Appendix 1 and Multimedia Appendix 1", "Multimedia Appendix 1", s)
    if n: wr(f, s); print(f"   {f}: 참조 {n}곳 통일"); tot += n
print(f"⑤ Part 참조 {tot}곳 정리")

left = {f: len(re.findall(r"Part [A-D]\b", rd(f))) for f in FILES}
print("\n남은 'Part X':", {k: v for k, v in left.items() if v})
