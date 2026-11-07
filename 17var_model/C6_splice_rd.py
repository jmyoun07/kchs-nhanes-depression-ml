# -*- coding: utf-8 -*-
"""Discussion 축소본(지적 반영본)을 원고에 넣고, 이전된 상세는 지정 부록으로 보낸다."""
import io, json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")

D = json.loads(io.open("work_notes/trim_rd_fixed.json", encoding="utf8").read())["sections"][0]
p = "docs/manuscript_17var.md"
lines = io.open(p, encoding="utf8").read().split("\n")

def bounds(h):
    i = lines.index(h)
    lvl = len(h) - len(h.lstrip("#"))
    j = i + 1
    while j < len(lines):
        m = re.match(r"^(#{1,6}) ", lines[j])
        if m and len(m.group(1)) <= lvl: break
        j += 1
    return i + 1, j

before = after = 0
for sub in reversed(D["subsections"]):          # 뒤에서부터 갈아끼운다
    h = sub["heading"].rstrip()
    assert h in lines, f"제목 없음: {h}"
    a, b = bounds(h)
    before += len(" ".join(lines[a:b]).split())
    body = sub["replacement"].strip("\n")
    lines[a:b] = [""] + body.split("\n") + [""]

src = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
io.open(p, "w", encoding="utf8").write(src)
lines = src.split("\n")
for sub in D["subsections"]:
    a, b = bounds(sub["heading"].rstrip()); after += len(" ".join(lines[a:b]).split())

# 이전된 상세를 각 부록에 붙인다
for ad in D["appendix_additions"]:
    f = ad["target_file"]
    t = io.open(f, encoding="utf8").read().rstrip("\n")
    blk = ad["markdown"].strip()
    blk = re.sub(r"(?m)^## (?!Part )", "### ", blk)      # Part 제목과 같은 급이 되지 않게
    if blk[:60] in t:
        print(f"  {f}: 이미 있음, 건너뜀"); continue
    io.open(f, "w", encoding="utf8").write(t + "\n\n" + blk + "\n")
    print(f"  {f}: +{len(blk.split())}단어")

print(f"\nDiscussion {before} -> {after}단어")
