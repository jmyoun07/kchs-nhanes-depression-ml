# -*- coding: utf-8 -*-
"""작업용.docx 와 생성본의 그림을 대조한다.

앞선 대조는 문단·표만 봤고 그림은 보지 않았다. 크기·위치·정렬을 모두 본다.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
from docx import Document
from docx.oxml.ns import qn

R = r"C:/Users/rudtn/Desktop/claude code/지역사회건강/제출본/01_원고"
EMU = 914400.0

def figs(path):
    d = Document(path)
    out = []
    for i, p in enumerate(d.paragraphs):
        for dr in p._p.findall(".//" + qn("w:drawing")):
            ext = dr.find(".//" + qn("wp:extent"))
            cx = int(ext.get("cx")) / EMU if ext is not None else None
            cy = int(ext.get("cy")) / EMU if ext is not None else None
            inline = dr.find(qn("wp:inline")) is not None
            out.append(dict(par=i, w=round(cx, 3), h=round(cy, 3),
                            inline=inline,
                            align=str(p.paragraph_format.alignment),
                            before=" ".join(d.paragraphs[i-1].text.split())[:58] if i else ""))
    return out, d

fa, da = figs(f"{R}/Manuscript.docx")
fb, db = figs(f"{R}/작업용.docx")
print(f"그림 수  생성본 {len(fa)} · 작업용 {len(fb)}\n")
print(f"{'#':<3}{'생성본 폭x높이':>18}{'작업용 폭x높이':>18}{'비율':>8}  {'정렬':<22}앞 문단")
for i in range(max(len(fa), len(fb))):
    a = fa[i] if i < len(fa) else None
    b = fb[i] if i < len(fb) else None
    if a and b:
        r = round(b["w"] / a["w"], 3) if a["w"] else None
        mark = "" if abs((b["w"] or 0) - (a["w"] or 0)) < 0.005 and abs((b["h"] or 0) - (a["h"] or 0)) < 0.005 else "  <-- 다름"
        al = a["align"] if a["align"] == b["align"] else f"{a['align']}->{b['align']}"
        print(f"{i+1:<3}{a['w']:>9.3f}x{a['h']:<8.3f}{b['w']:>9.3f}x{b['h']:<8.3f}{r:>8}  {al:<22}{b['before'][:46]}{mark}")
    else:
        print(f"{i+1:<3}  {'(없음)' if not a else a}  {'(없음)' if not b else b}")

# 그림이 놓인 위치(앞 문단)가 바뀌었는지
print("\n=== 그림 앞 문단 비교 ===")
for i in range(min(len(fa), len(fb))):
    if fa[i]["before"] != fb[i]["before"]:
        print(f"  그림{i+1}")
        print(f"    생성본 뒤: {fa[i]['before']}")
        print(f"    작업용 뒤: {fb[i]['before']}")
    if fa[i]["par"] != fb[i]["par"]:
        print(f"  그림{i+1} 문단 위치 {fa[i]['par']} -> {fb[i]['par']}")
