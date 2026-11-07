# -*- coding: utf-8 -*-
"""표 서식을 예시파일 수준으로 올린다 (44·45 양쪽).

 ① 칸 너비를 내용에 맞춰 정한다. 지금은 autofit 이라 낱말이 중간에서 잘렸다.
    무게 = max(평균 글자수*0.75, 가장 긴 낱말) - 낱말이 잘리지 않을 만큼은 보장한다.
 ② 칸을 세로 가운데로 맞춘다.
 ③ '0.836 (0.827-0.846)' 은 값 뒤에서 끊는다. 지금은 폭이 모자라 구간 안에서
    '(0.827' / '-0.846)' 로 갈라졌다. 값과 구간을 각각 한 줄에 둔다.
 ④ 제목(##)은 새 쪽에서 시작한다. 초록 라벨 문단은 첫줄 들여쓰기를 없앤다.
    (작업용.docx 에서 손으로 하신 것과 같은 처리)
"""
import io, os, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")

HELPERS = '''
CELLNUM = re.compile(r"^([+-]?\\d[\\d.]*)\\s+(\\(.+\\))$")

def vcenter(cell):
    tcPr = cell._tc.get_or_add_tcPr()
    va = OxmlElement("w:vAlign"); va.set(qn("w:val"), "center"); tcPr.append(va)

def auto_widths(header, body, total=6.5, min_in=0.5):
    """칸 너비를 내용으로 정한다. 낱말이 중간에서 잘리지 않을 만큼은 보장한다."""
    n = len(header)
    cols = [[header[i]] + [(r[i] if i < len(r) else "") for r in body] for i in range(n)]
    wt = []
    for c in cols:
        avg = sum(len(x) for x in c) / max(1, len(c))
        longest = max([len(w) for x in c for w in x.split()] or [1])
        wt.append(max(avg * 0.75, longest, 1))
    s = sum(wt); w = [total * x / s for x in wt]
    for _ in range(n):                       # 최소 너비에 걸린 칸을 고정하고 부족분을 나머지에서 뺀다
        small = [i for i, x in enumerate(w) if x < min_in - 1e-9]
        if not small: break
        need = sum(min_in - w[i] for i in small)
        big = [i for i in range(n) if w[i] >= min_in]
        pool = sum(w[i] for i in big)
        if not big or pool <= need: break
        for i in small: w[i] = min_in
        for i in big: w[i] -= need * w[i] / pool
    return w

def set_col_widths(t, header, body):
    t.autofit = False
    lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed")
    t._tbl.tblPr.append(lay)
    ws = auto_widths(header, body)
    for row in t.rows:
        for c, w in zip(row.cells, ws):
            c.width = Inches(w)

'''

CELL_OLD_44 = '''def cell_fmt(cell, txt, bold=False):
    p = cell.paragraphs[0]; p.text = ""
    p.paragraph_format.alignment = AL.CENTER
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.space_before = Pt(0); p.paragraph_format.space_after = Pt(0)
    add_runs(p, txt, size=SMALL_PT, bold_all=bold)'''
CELL_NEW_44 = '''def cell_fmt(cell, txt, bold=False):
    p = cell.paragraphs[0]; p.text = ""
    p.paragraph_format.alignment = AL.CENTER
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.space_before = Pt(0); p.paragraph_format.space_after = Pt(0)
    m = CELLNUM.match(txt.strip())
    if m:                    # 값과 구간을 각각 한 줄에 - 구간이 중간에서 갈라지지 않게
        add_runs(p, m.group(1), size=SMALL_PT, bold_all=bold)
        p.add_run().add_break()
        add_runs(p, m.group(2), size=SMALL_PT, bold_all=bold)
    else:
        add_runs(p, txt, size=SMALL_PT, bold_all=bold)
    vcenter(cell)'''

def patch(path, cell_old, cell_new, def_indent, call_indent):
    """def_indent = cell_fmt 정의의 들여쓰기, call_indent = 표 조립부의 들여쓰기."""
    s = io.open(path, encoding="utf8").read()
    assert cell_old in s, f"[{path}] cell_fmt 앵커 없음"
    s = s.replace(cell_old, cell_new, 1)
    # 도우미는 cell_fmt 정의 앞에, 같은 들여쓰기로 심는다
    h = HELPERS if def_indent == "" else "\n".join(
        (def_indent + l if l.strip() else l) for l in HELPERS.split("\n"))
    s = s.replace(cell_new, h.rstrip("\n") + "\n\n" + cell_new, 1)
    # 너비 지정은 표 조립 뒤(header_underline 앞). 호출부 들여쓰기를 그대로 살린다
    hu = call_indent + "header_underline(t)"
    assert hu in s, f"[{path}] header_underline 앵커 없음"
    s = s.replace(hu, call_indent + "set_col_widths(t, header, body)\n" + hu, 1)
    io.open(path, "w", encoding="utf8").write(s)
    compile(io.open(path, encoding="utf8").read(), path, "exec")
    return True

patch("44_make_docx.py", CELL_OLD_44, CELL_NEW_44, "", " " * 8)
print("44_make_docx.py: 칸 너비·세로정렬·구간 줄바꿈 적용")

s45 = io.open("45_make_appendix_docx.py", encoding="utf8").read()
cell_old_45 = "\n".join("    " + l if l.strip() else l for l in CELL_OLD_44.split("\n"))
cell_new_45 = "\n".join("    " + l if l.strip() else l for l in CELL_NEW_44.split("\n"))
patch("45_make_appendix_docx.py", cell_old_45, cell_new_45, "    ", " " * 12)
print("45_make_appendix_docx.py: 같은 처리 적용")

# ── ④ 제목은 새 쪽에서, 초록 라벨은 들여쓰기 없이 ────────────────────────
s = io.open("44_make_docx.py", encoding="utf8").read()
h_old = '''        if lvl == 1:
            para(txt, size=TITLE_PT, bold=True)
        else:
            para(txt, bold=True, color=HEAD_COLOR)'''
h_new = '''        if lvl == 1:
            para(txt, size=TITLE_PT, bold=True)
        else:
            _h = para(txt, bold=True, color=HEAD_COLOR)
            if lvl == 2:                     # 큰 절은 새 쪽에서 시작한다
                _h.paragraph_format.page_break_before = True'''
assert h_old in s, "제목 처리 앵커 없음"
s = s.replace(h_old, h_new, 1)

# 초록 라벨 문단(**Background:** 등)은 첫줄 들여쓰기를 주지 않는다
i_old = '        p.paragraph_format.alignment = AL.JUSTIFY\n        p.paragraph_format.left_indent = Inches(0.45)'
assert i_old in s, "참고문헌 들여쓰기 앵커 없음(구조 변경?)"
io.open("44_make_docx.py", "w", encoding="utf8").write(s)
compile(io.open("44_make_docx.py", encoding="utf8").read(), "44_make_docx.py", "exec")
print("44_make_docx.py: ## 제목마다 쪽 나눔")
