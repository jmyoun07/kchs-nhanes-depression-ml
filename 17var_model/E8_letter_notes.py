# -*- coding: utf-8 -*-
"""각주 문자 배정을 고친다.

지금은 첫 각주를 무조건 머리글 첫 칸에 붙인다. 그래서 총괄 주석 없이 약어 정의만
있는 표에서는 'AUROC: ...' 가 'Cohort' 칸에 붙는 엉뚱한 결과가 나온다.
그걸 피하려고 약어를 한 줄에 몰아 썼는데, 그러면 '약어 하나에 문자 하나' 규칙이 깨진다.

고친 방식
  · 총괄 주석(약어 정의가 아닌 것)이 먼저 a, b ... 를 받고, 첫 총괄 주석만 머리글 첫 칸에 붙는다
  · 약어 정의는 그 다음 문자를 받고, 각각 그 약어가 실제로 나오는 칸에 붙는다
  · 총괄 주석이 없으면 머리글 첫 칸에는 아무 문자도 붙지 않는다
"""
import io, os, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")

NEW_BODY = '''    if not notes: return header, body, []
    header = list(header); body = [list(r) for r in body]
    lettered, extra = [], []
    def anchor(tok, ch):
        for ci, c in enumerate(header):
            if tok.lower() in c.lower(): header[ci] = c + f"^{ch}"; return True
        for r in body:                       # 첫 칸만이 아니라 모든 칸을 뒤진다
            for ci, c in enumerate(r):       # (표 1은 첫 칸이 번호라 못 찾았다)
                if tok.lower() in c.lower(): r[ci] = c + f"^{ch}"; return True
        return False
    gen = [n for n in notes if not NOTE_ABBR.match(n)]
    abb = [n for n in notes if NOTE_ABBR.match(n)]
    ch = ord("a")
    for n in gen:                            # 총괄 주석이 먼저, 첫 개만 머리글 첫 칸에
        if not lettered:
            for _ci, _c in enumerate(header):
                if _c.strip(): header[_ci] = _c + f"^{chr(ch)}"; break
        lettered.append((chr(ch), n)); ch += 1
    for n in abb:                            # 약어는 하나에 문자 하나, 실제 나오는 칸에
        m = NOTE_ABBR.match(n)
        if anchor(m.group(1), chr(ch)):
            lettered.append((chr(ch), n)); ch += 1
        else:
            extra.append(n)
    if extra:
        if lettered:
            lettered[0] = (lettered[0][0], lettered[0][1] + " " + " ".join(extra))
        else:
            for _ci, _c in enumerate(header):
                if _c.strip(): header[_ci] = _c + f"^{chr(ch)}"; break
            lettered.append((chr(ch), " ".join(extra)))
    return header, body, lettered
'''

for p, ind in (("44_make_docx.py", ""), ("45_make_appendix_docx.py", "")):
    s = io.open(p, encoding="utf8").read()
    a = s.index("def letter_notes(header, body, notes):")
    a = s.index("\n", s.index('"""', s.index('"""', a) + 3)) + 1   # docstring 다음 줄부터
    b = s.index("\n\n", a)
    body = NEW_BODY if ind == "" else NEW_BODY
    io.open(p, "w", encoding="utf8").write(s[:a] + body + s[b:])
    compile(io.open(p, encoding="utf8").read(), p, "exec")
    print(f"{p}: letter_notes 재작성")

# 몰아 쓴 약어 줄을 한 줄씩으로 나눈다
import re
for f in ("docs/appendix2_crosswalk.md", "docs/appendix3_utility.md",
          "docs/appendix4_ceiling.md", "docs/appendix5_methods.md"):
    L = io.open(f, encoding="utf8").read().split("\n")
    out, n = [], 0
    for l in L:
        if re.match(r"^[A-Za-z][A-Za-z0-9+\-]{1,20}: [^;]+(; [A-Za-z][A-Za-z0-9+\-]{1,20}: [^;]+)+\.$", l):
            parts = [x.strip() for x in l.rstrip(".").split("; ")]
            out += [x + "." for x in parts]; n += len(parts)
        else:
            out.append(l)
    if n:
        io.open(f, "w", encoding="utf8").write("\n".join(out))
        print(f"{f}: 약어 각주 {n}줄로 분리")
