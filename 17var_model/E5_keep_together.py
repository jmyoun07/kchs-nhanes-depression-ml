# -*- coding: utf-8 -*-
"""캡션이 그림·표에서 떨어지지 않게 한다.

그림 크기를 손으로 줄이신 이유가 '캡션과 그림이 페이지 경계에서 갈라져서' 였다.
크기로 맞출 일이 아니라 워드의 문단 속성으로 묶는 것이 정답이다.

  · 캡션 문단      keepNext  - 바로 다음 것(그림 또는 표 첫 행)과 같은 쪽에 둔다
  · 그림 문단      keepLines - 그림 자체가 쪼개지지 않게
  · 표의 각 행     cantSplit - 한 행이 쪽 경계에서 갈라지지 않게

표 전체를 한 쪽에 묶지는 않는다. 28행짜리 부록 표까지 묶으면 앞 쪽에 큰 빈칸이 생긴다.
"""
import io, os, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")

HELPER = '''
def keep_next(p):
    """이 문단을 다음 것과 같은 쪽에 둔다(캡션이 그림·표에서 떨어지지 않게)."""
    p.paragraph_format.keep_with_next = True
    return p

def rows_cant_split(tbl):
    """행이 쪽 경계에서 갈라지지 않게 한다."""
    for r in tbl.rows:
        trPr = r._tr.get_or_add_trPr()
        e = OxmlElement("w:cantSplit"); trPr.append(e)

'''

# ── 44: 본문 ─────────────────────────────────────────────────────────────
p = "44_make_docx.py"
s = io.open(p, encoding="utf8").read()
if "def keep_next" not in s:
    s = s.replace("def set_borders(tbl):", HELPER.lstrip("\n") + "def set_borders(tbl):", 1)

# 그림: 캡션 문단과 그림 문단
a = '''        pic = doc.add_paragraph(); pic.paragraph_format.alignment = AL.CENTER'''
assert a in s, "[44] 그림 문단 앵커 없음"
s = s.replace(a, a + '''
        pic.paragraph_format.keep_together = True''', 1)

# 표: 캡션 keepNext + 행 cantSplit
b = "        header_underline(t)"
assert b in s, "[44] header_underline 앵커 없음"
s = s.replace(b, b + "\n        rows_cant_split(t)", 1)
io.open(p, "w", encoding="utf8").write(s)
compile(io.open(p, encoding="utf8").read(), p, "exec")
print("44_make_docx.py: 그림 문단 keepLines · 표 행 cantSplit")

# ── 45: 부록 ─────────────────────────────────────────────────────────────
q = "45_make_appendix_docx.py"
d = io.open(q, encoding="utf8").read()
if "def rows_cant_split" not in d:
    d = d.replace("def set_borders(tbl):", '''def rows_cant_split(tbl):
    """행이 쪽 경계에서 갈라지지 않게 한다."""
    for r in tbl.rows:
        trPr = r._tr.get_or_add_trPr()
        e = OxmlElement("w:cantSplit"); trPr.append(e)

def set_borders(tbl):''', 1)
c = "            header_underline(t)"
assert c in d, "[45] header_underline 앵커 없음"
d = d.replace(c, c + "\n            rows_cant_split(t)", 1)
io.open(q, "w", encoding="utf8").write(d)
compile(io.open(q, encoding="utf8").read(), q, "exec")
print("45_make_appendix_docx.py: 표 행 cantSplit")
