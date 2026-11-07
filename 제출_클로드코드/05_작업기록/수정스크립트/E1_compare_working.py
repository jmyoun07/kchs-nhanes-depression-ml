# -*- coding: utf-8 -*-
"""작업용.docx 와 내가 생성한 Manuscript.docx 를 대조한다.

사용자가 손으로 조정한 것이 무엇인지 정확히 알아야, 그 조정을 생성기에 반영해
다시 만들 때마다 유지할 수 있다. 문단 서식·표 너비·본문을 모두 본다.
"""
import os, sys
sys.stdout.reconfigure(encoding="utf-8")
from docx import Document
from docx.shared import Emu

R = r"C:/Users/rudtn/Desktop/claude code/지역사회건강/제출본/01_원고"
A = Document(f"{R}/Manuscript.docx")     # 내가 생성한 것
B = Document(f"{R}/작업용.docx")          # 사용자가 조정한 것

def pinfo(p):
    f = p.paragraph_format
    return dict(text=" ".join(p.text.split()),
                align=str(f.alignment), first=f.first_line_indent, left=f.left_indent,
                space_b=f.space_before, space_a=f.space_after, ls=f.line_spacing,
                pbb=f.page_break_before, kwn=f.keep_with_next, kt=f.keep_together)

pa = [pinfo(p) for p in A.paragraphs]
pb = [pinfo(p) for p in B.paragraphs]
print(f"문단 수  생성본 {len(pa)} · 작업용 {len(pb)}")
print(f"표 수    생성본 {len(A.tables)} · 작업용 {len(B.tables)}")
print(f"섹션 여백 생성본 {[round(s.left_margin/914400,2) for s in A.sections]} · "
      f"작업용 {[round(s.left_margin/914400,2) for s in B.sections]}")

# ── 본문이 바뀐 곳 ────────────────────────────────────────────────────────
ta = [x["text"] for x in pa if x["text"]]
tb = [x["text"] for x in pb if x["text"]]
import difflib
sm = difflib.SequenceMatcher(None, ta, tb)
print("\n=== 본문 차이 ===")
nd = 0
for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag == "equal": continue
    nd += 1
    print(f"  [{tag}] 생성본 {i1}-{i2} / 작업용 {j1}-{j2}")
    for x in ta[i1:i2][:3]: print(f"      - {x[:96]}")
    for x in tb[j1:j2][:3]: print(f"      + {x[:96]}")
    if nd > 12: print("   ..."); break
if nd == 0: print("  (본문 텍스트는 동일)")

# ── 문단 서식이 바뀐 곳 ──────────────────────────────────────────────────
print("\n=== 문단 서식 차이 ===")
idx = {x["text"]: x for x in pa if x["text"]}
n = 0
for x in pb:
    if not x["text"] or x["text"] not in idx: continue
    y = idx[x["text"]]
    diff = {k: (y[k], x[k]) for k in ("align","first","left","space_b","space_a","ls","pbb","kwn","kt")
            if y[k] != x[k]}
    if diff:
        n += 1
        print(f"  {x['text'][:52]!r}")
        for k, (o, v) in diff.items():
            fo = round(o/914400,3) if isinstance(o,(int,)) and abs(o or 0)>1000 else o
            fv = round(v/914400,3) if isinstance(v,(int,)) and abs(v or 0)>1000 else v
            print(f"      {k}: {fo} -> {fv}")
        if n > 25: print("   ..."); break
if n == 0: print("  (문단 서식 동일)")

# ── 표 너비가 바뀐 곳 ────────────────────────────────────────────────────
print("\n=== 표 너비 차이 (인치) ===")
for i, (t1, t2) in enumerate(zip(A.tables, B.tables)):
    w1 = [round(c.width/914400, 3) if c.width else None for c in t1.rows[0].cells]
    w2 = [round(c.width/914400, 3) if c.width else None for c in t2.rows[0].cells]
    if w1 != w2:
        print(f"  표{i+1}")
        print(f"      생성본 {w1}  합 {round(sum(x for x in w1 if x),2)}")
        print(f"      작업용 {w2}  합 {round(sum(x for x in w2 if x),2)}")
        print(f"      차이   {[round((b or 0)-(a or 0),3) for a,b in zip(w1,w2)]}")
    else:
        print(f"  표{i+1} 동일 {w1}")
