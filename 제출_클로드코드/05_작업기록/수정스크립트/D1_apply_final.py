# -*- coding: utf-8 -*-
"""최종 축소본에 감사 지적 22건을 반영해 원고에 넣는다.

지적은 전부 '줄이면서 한정어가 사라진' 유형이다. 감사자가 준 정정 문장으로 되돌린다.
"""
import io, json, os, re, sys
from difflib import SequenceMatcher
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")

J = json.loads(io.open("work_notes/trim_final.json", encoding="utf8").read())

def sentences(t):
    return [x for x in re.split(r"(?<=[.;])\s+(?=[A-Z(\u201c])", t) if x.strip()]

def apply_fix(text, fix):
    if fix.strip() in text: return text, 1.0, "이미 있음"
    best, sc = None, 0.0
    for s in sentences(text):
        r = SequenceMatcher(None, s.lower(), fix.lower()).ratio()
        if r > sc: best, sc = s, r
    if best and sc >= 0.32:
        return text.replace(best, fix.strip(), 1), sc, "치환"
    return text, sc, "미적용"

tot_ok = tot_miss = 0
for sec in J["sections"]:
    print(f"\n── {sec['key']} ({len(sec['problems'])}건)")
    for i, pr in enumerate(sec["problems"], 1):
        fix = pr["fix"].strip()
        if fix and not fix.endswith((".", ";", '"')): fix += "."
        hd = pr["heading"].strip("# ").split(" - ")[0].strip().lower()
        target = None
        for sub in sec["subsections"]:
            if hd and hd in sub["heading"].lower(): target = sub; break
        if target is None:                       # 제목 매칭 실패 시 본문 전체에서 가장 닮은 곳
            best, sc0 = None, 0.0
            for sub in sec["subsections"]:
                _, r, _ = apply_fix(sub["replacement"], fix)
                if r > sc0: best, sc0 = sub, r
            target = best
        if target is None:
            print(f"  [{i}] 대상 없음: {pr['heading'][:44]}"); tot_miss += 1; continue
        new, sc, how = apply_fix(target["replacement"], fix)
        target["replacement"] = new
        print(f"  [{i}] {how} ({sc:.2f}) {target['heading'][:42]}")
        tot_ok += (how != "미적용"); tot_miss += (how == "미적용")

print(f"\n적용 {tot_ok} · 미적용 {tot_miss}")
io.open("work_notes/trim_final_fixed.json", "w", encoding="utf8").write(json.dumps(J, ensure_ascii=False, indent=1))

# ── 원고에 넣는다 ────────────────────────────────────────────────────────
p = "docs/manuscript_17var.md"
lines = io.open(p, encoding="utf8").read().split("\n")
def bounds(h):
    i = lines.index(h); lvl = len(h) - len(h.lstrip("#")); j = i + 1
    while j < len(lines):
        m = re.match(r"^(#{1,6}) ", lines[j])
        if m and len(m.group(1)) <= lvl: break
        j += 1
    return i + 1, j

allsubs = [s for sec in J["sections"] for s in sec["subsections"]]
pos = []
for s in allsubs:
    h = s["heading"].rstrip()
    assert h in lines, f"제목 없음: {h}"
    pos.append((lines.index(h), s))
for _, s in sorted(pos, key=lambda x: -x[0]):     # 뒤에서부터
    a, b = bounds(s["heading"].rstrip())
    lines[a:b] = [""] + s["replacement"].strip("\n").split("\n") + [""]
src = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
io.open(p, "w", encoding="utf8").write(src)

# ── 부록 추가분 ──────────────────────────────────────────────────────────
for sec in J["sections"]:
    for ad in sec.get("appendix_additions", []):
        f = ad["target_file"]; t = io.open(f, encoding="utf8").read().rstrip("\n")
        blk = re.sub(r"(?m)^## (?!Part )", "### ", ad["markdown"].strip())
        if blk[:70] in t: print(f"  {f}: 이미 있음"); continue
        io.open(f, "w", encoding="utf8").write(t + "\n\n" + blk + "\n")
        print(f"  {f}: +{len(blk.split())}단어")

lines = src.split("\n")
for sec in J["sections"]:
    w = sum(len(" ".join(lines[slice(*bounds(s['heading'].rstrip()))]).split()) for s in sec["subsections"])
    print(f"{sec['key']:<12} 최종 {w}단어")
