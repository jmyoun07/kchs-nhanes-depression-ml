# -*- coding: utf-8 -*-
"""축소본을 원고에 넣고, 이전된 상세는 Part D 로 모은다."""
import io, json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")

S = {s["key"]: s for s in json.loads(io.open("work_notes/trim_fixed.json", encoding="utf8").read())["sections"]}
HEAD = {"intro": "## Introduction", "design": "### Study Design and Data Sources",
        "parts": "### Participants and Outcome", "predictors": "### Predictors and Harmonization",
        "model": "### Model Development and Evaluation",
        "external": "### External Validation and Sensitivity Analyses",
        "ethics": "### Ethical Considerations"}
ORDER = ["intro", "design", "parts", "predictors", "model", "external", "ethics"]

p = "docs/manuscript_17var.md"
src = io.open(p, encoding="utf8").read()
lines = src.split("\n")

def bounds(h):
    """소제목 다음 줄부터, 같은 급 이상의 다음 제목 직전까지."""
    i = lines.index(h)
    lvl = len(h) - len(h.lstrip("#"))
    j = i + 1
    while j < len(lines):
        m = re.match(r"^(#{1,6}) ", lines[j])
        if m and len(m.group(1)) <= lvl: break
        j += 1
    return i + 1, j

before = {}
for k in reversed(ORDER):          # 뒤에서부터 갈아끼워야 앞 줄번호가 안 밀린다
    h = HEAD[k]
    assert h in lines, f"제목 없음: {h}"
    a, b = bounds(h)
    before[k] = len(" ".join(lines[a:b]).split())
    body = S[k]["replacement"].strip("\n")
    lines[a:b] = [""] + body.split("\n") + [""]

src = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))

# ── Multimedia Appendix 1 설명에 Part D 를 알린다 ─────────────────────────
a1_old = ("Supplementary methods and results: the year-specific KCHS\u2194NHANES harmonization crosswalk and the "
          "predictor-set amendment; clinical utility, sample characteristics and sensitivity analyses (Tables S1-S10); "
          "and the transportability-ceiling and calibration-interval analyses (Tables S11-S16).")
a1_new = ("Supplementary methods and results, in four parts: Part A, the year-specific KCHS\u2194NHANES harmonization "
          "crosswalk and the predictor-set amendment; Part B, clinical utility, sample characteristics and sensitivity "
          "analyses (Tables S1-S10); Part C, the transportability-ceiling and calibration-interval analyses "
          "(Tables S11-S16); and Part D, supplementary methods, which carries the study-design, participant, "
          "harmonization, model-development, external-validation and ethics detail relocated from the Methods.")
assert a1_old in src, "Multimedia Appendix 1 설명 앵커 없음"
src = src.replace(a1_old, a1_new, 1)
io.open(p, "w", encoding="utf8").write(src)

# ── Part D 조립 ──────────────────────────────────────────────────────────
TITLE = {"intro": "Prior-model reviews cited in the Introduction",
         "design": "Study design, split procedure, and risk-of-bias assessment",
         "parts": "Participant eligibility and analytic sample derivation",
         "predictors": "Harmonization resolution, scale differences, and preprocessing",
         "model": "Model development, tuning, and resampling detail",
         "external": "External validation, recalibration, parsimony selection, and the data-integrity audit",
         "ethics": "Ethics approvals, consent, and data governance"}
d = io.open("docs/appendix5_methods.md", encoding="utf8").read().rstrip("\n")
head_re = re.compile(r"^#{2,3} .*$", re.M)
added = 0
for k in ORDER:
    blk = S[k]["appendix_addition"].strip()
    if not blk: continue
    blk = head_re.sub(lambda m: "#### " + m.group(0).lstrip("# "), blk)   # Part D 아래로 한 급 내린다
    if not blk.lstrip().startswith("####"):
        blk = f"#### {TITLE[k]}\n\n" + blk
    else:                                    # 첫 제목을 우리 제목으로 통일
        blk = head_re.sub("", blk, count=0)
        blk = re.sub(r"^#### .*$", f"### {TITLE[k]}", blk, count=1, flags=re.M)
    blk = re.sub(r"^#### ", "### ", blk, count=1, flags=re.M)
    d += "\n\n" + blk.strip() + "\n"
    added += 1
io.open("docs/appendix5_methods.md", "w", encoding="utf8").write(d + "\n")

print(f"{'섹션':<12}{'이전':>7}{'이후':>7}")
after = {}
new_lines = io.open(p, encoding="utf8").read().split("\n")
lines = new_lines
for k in ORDER:
    a, b = bounds(HEAD[k]); after[k] = len(" ".join(lines[a:b]).split())
    print(f"{k:<12}{before[k]:>7}{after[k]:>7}")
print(f"{'합계':<12}{sum(before.values()):>7}{sum(after.values()):>7}")
print(f"\nPart D: {added}개 블록 · {len(io.open('docs/appendix5_methods.md',encoding='utf8').read().split())}단어")
