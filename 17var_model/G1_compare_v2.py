# -*- coding: utf-8 -*-
"""작업용_v2(사용자 수정본) 와 내 생성본을 정밀 대조한다.

손으로 하신 서식 조정을 생성기에 옮겨야 다시 만들 때마다 유지된다.
문단 속성·표 여백·행 높이·칸 여백·그림까지 본다.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
from docx import Document
from docx.oxml.ns import qn

R = r"C:/Users/rudtn/Desktop/claude code/지역사회건강"
A = Document(f"{R}/17var_model/논문원고_우울ML_17변수.docx")   # 내 생성본
B = Document(f"{R}/제출본/01_원고/작업용_v2.docx")             # 사용자 수정본
EMU = 914400.0

print(f"문단  생성본 {len(A.paragraphs)} · v2 {len(B.paragraphs)}")
print(f"표    생성본 {len(A.tables)} · v2 {len(B.tables)}")

# ── 문단 속성 ────────────────────────────────────────────────────────────
def pf(p):
    f = p.paragraph_format
    return dict(first=f.first_line_indent, left=f.left_indent, ls=f.line_spacing,
                sb=f.space_before, sa=f.space_after, pbb=f.page_break_before,
                kwn=f.keep_with_next, kt=f.keep_together, al=str(f.alignment))
ia = {}
for p in A.paragraphs:
    t = " ".join(p.text.split())
    if t: ia.setdefault(t, pf(p))
print("\n=== 문단 서식 차이 ===")
n = 0
for p in B.paragraphs:
    t = " ".join(p.text.split())
    if not t or t not in ia: continue
    x, y = ia[t], pf(p)
    d = {k: (x[k], y[k]) for k in x if x[k] != y[k]}
    if d:
        n += 1
        print(f"  {t[:56]!r}")
        for k, (o, v) in d.items():
            fo = round(o/EMU, 3) if isinstance(o, int) and abs(o or 0) > 500 else o
            fv = round(v/EMU, 3) if isinstance(v, int) and abs(v or 0) > 500 else v
            print(f"      {k}: {fo} -> {fv}")
        if n > 20: print("   ..."); break
if n == 0: print("  (동일)")

# ── 그림 문단 ────────────────────────────────────────────────────────────
print("\n=== 그림 문단 속성 ===")
def figpars(d):
    out = []
    for p in d.paragraphs:
        if p._p.findall(".//" + qn("w:drawing")):
            f = p.paragraph_format
            ext = p._p.find(".//" + qn("wp:extent"))
            out.append(dict(kt=f.keep_together, kwn=f.keep_with_next, ls=f.line_spacing,
                            w=round(int(ext.get("cx"))/EMU, 3) if ext is not None else None,
                            h=round(int(ext.get("cy"))/EMU, 3) if ext is not None else None))
    return out
fa, fb = figpars(A), figpars(B)
for i, (x, y) in enumerate(zip(fa, fb), 1):
    d = {k: (x[k], y[k]) for k in x if x[k] != y[k]}
    print(f"  그림{i}: {'동일' if not d else d}")

# ── 표: 칸 여백·행 높이·너비 ─────────────────────────────────────────────
print("\n=== 표 속성 ===")
def tinfo(t):
    tblPr = t._tbl.tblPr
    mar = tblPr.find(qn("w:tblCellMar"))
    m = {}
    if mar is not None:
        for side in ("top", "bottom", "left", "right"):
            e = mar.find(qn("w:" + side))
            if e is not None: m[side] = round(int(e.get(qn("w:w")))/1440, 3)
    hs = []
    for r in t.rows:
        trPr = r._tr.find(qn("w:trPr"))
        h = trPr.find(qn("w:trHeight")) if trPr is not None else None
        hs.append(round(int(h.get(qn("w:val")))/1440, 3) if h is not None else None)
    return m, hs, [round(c.width/EMU, 3) if c.width else None for c in t.rows[0].cells]
for i, (t1, t2) in enumerate(zip(A.tables, B.tables), 1):
    m1, h1, w1 = tinfo(t1); m2, h2, w2 = tinfo(t2)
    print(f"  표{i}")
    print(f"    칸여백 생성본 {m1 or '(기본)'} -> v2 {m2 or '(기본)'}")
    print(f"    행높이 생성본 {sorted(set(x for x in h1 if x)) or '(지정없음)'} -> v2 {sorted(set(x for x in h2 if x)) or '(지정없음)'}")
    if w1 != w2:
        print(f"    너비 생성본 {w1}")
        print(f"         v2     {w2}")
    # 셀 내부 문단 간격
    def cellsp(t):
        s = set()
        for r in t.rows:
            for c in r.cells:
                for p in c.paragraphs:
                    s.add((p.paragraph_format.space_before, p.paragraph_format.space_after,
                           p.paragraph_format.line_spacing))
        return s
    c1, c2 = cellsp(t1), cellsp(t2)
    if c1 != c2:
        print(f"    셀 문단간격 생성본 {c1}")
        print(f"                 v2   {c2}")
