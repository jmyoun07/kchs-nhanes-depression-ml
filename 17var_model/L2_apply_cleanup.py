# -*- coding: utf-8 -*-
"""'relocated' 덩어리 정리안을 적용한다. 감사 지적 3건(전부 범위·포인터)을 먼저 반영한다.

제목이 편집 이력('relocated in revision', 'the trimmed Methods')을 드러내던 것을
주제 이름으로 바꾸고, 다섯 블록에 반복되던 상투구와 본문과 겹치는 문장을 걷어낸다.
"""
import io, json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")
J = json.loads(io.open("work_notes/relocated_cleanup.json", encoding="utf8").read())

# ── 감사 지적 3건 반영 ──────────────────────────────────────────────────
FIX = [
 ("docs/appendix3_utility.md",
  "measurement difference recorded in the harmonization crosswalk.",
  "measurement difference recorded in the harmonization crosswalk in Multimedia Appendix 1.",
  "크로스워크가 단독 항목으로 인용될 수 없어 부록 이름을 붙임"),
 ("docs/appendix5_methods.md",
  "Supports the Methods section Model Development and Evaluation.\n\nThe training set's approximately",
  "Supports the Methods sections Model Development and Evaluation and External Validation and Sensitivity "
  "Analyses.\n\nThe training set's approximately",
  "경량모형 절차는 External Validation 절 소관이라 범위에 추가"),
 ("docs/appendix4_ceiling.md",
  "Supports Discussion Plausible Mechanisms for the External Ceiling.",
  "Supports Discussion Principal Findings and Discussion Plausible Mechanisms for the External Ceiling.",
  "주기간 차이 구간은 Principal Findings 소관"),
]
for f in J["files"]:
    for tf, a, b, why in FIX:
        if f["file"] == tf and a in f["replacement"]:
            f["replacement"] = f["replacement"].replace(a, b, 1)
            print(f"  감사 반영: {why}")

# ── 파일에 적용 ─────────────────────────────────────────────────────────
for f in J["files"]:
    p = f["file"]
    s = io.open(p, encoding="utf8").read()
    a = f["start_heading"].strip()
    assert a in s, f"{p}: 시작 제목 없음 {a[:50]}"
    i = s.index(a)
    end = f["end_heading"].strip()
    j = s.index(end, i) if end and end in s else len(s)
    io.open(p, "w", encoding="utf8").write(
        (s[:i] + f["replacement"].rstrip("\n") + "\n" + s[j:]).rstrip("\n") + "\n")
    print(f"{p.split('/')[-1]:<24} {f['words_before']:>5} -> {f['words_after']:<5}단어")

# ── 편집 이력 표현이 남지 않았는지 ──────────────────────────────────────
BAD = re.compile(r"relocat|trimmed|condensed out of|in revision|preserved here", re.I)
left = []
for p in ("docs/appendix2_crosswalk.md", "docs/appendix3_utility.md",
          "docs/appendix4_ceiling.md", "docs/appendix5_methods.md"):
    for m in BAD.finditer(io.open(p, encoding="utf8").read()):
        left.append((p.split("/")[-1], m.group(0)))
print(f"\n편집 이력 표현 잔존: {left if left else '없음'}")
