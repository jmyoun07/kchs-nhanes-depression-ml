# -*- coding: utf-8 -*-
"""B5 가 중간에 멈춰서 남은 두 가지를 마저 적용한다.

 ① 45_make_appendix_docx.py 에도 같은 표 서식(너비·세로정렬·구간 줄바꿈)
 ② 44_make_docx.py: ## 절 제목마다 새 쪽에서 시작
"""
import io, os, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")

# ── ① 부록 생성기 ────────────────────────────────────────────────────────
p = "45_make_appendix_docx.py"
s = io.open(p, encoding="utf8").read()

HELP = '''    CELLNUM = re.compile(r"^([+-]?\\d[\\d.]*)\\s+(\\(.+\\))$")

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
        t = sum(wt); w = [total * x / t for x in wt]
        for _ in range(n):
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
        for row in t.rows:
            for c, w in zip(row.cells, auto_widths(header, body)):
                c.width = Inches(w)

'''

CELL_OLD = '''    def cell_fmt(cell, txt, bold=False):
        p = cell.paragraphs[0]; p.text = ""
        p.paragraph_format.alignment = AL.CENTER
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.space_before = Pt(0); p.paragraph_format.space_after = Pt(0)
        add_runs(p, txt, size=SMALL_PT, bold_all=bold)'''
CELL_NEW = '''    def cell_fmt(cell, txt, bold=False):
        p = cell.paragraphs[0]; p.text = ""
        p.paragraph_format.alignment = AL.CENTER
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.space_before = Pt(0); p.paragraph_format.space_after = Pt(0)
        m = CELLNUM.match(txt.strip())
        if m:                # 값과 구간을 각각 한 줄에 - 구간이 중간에서 갈라지지 않게
            add_runs(p, m.group(1), size=SMALL_PT, bold_all=bold)
            p.add_run().add_break()
            add_runs(p, m.group(2), size=SMALL_PT, bold_all=bold)
        else:
            add_runs(p, txt, size=SMALL_PT, bold_all=bold)
        vcenter(cell)'''
assert CELL_OLD in s, "[45] cell_fmt 앵커 없음"
s = s.replace(CELL_OLD, HELP + CELL_NEW, 1)

HU = " " * 12 + "header_underline(t)"
assert HU in s, "[45] header_underline 앵커 없음"
s = s.replace(HU, " " * 12 + "set_col_widths(t, header, body)\n" + HU, 1)

# 부록 생성기에도 collect_notes 버그가 그대로 있다
C_OLD = '''def collect_notes(lines, i):
    """표 바로 뒤에 붙는 주석 문단을 모은다(제목·표·굵은 라벨을 만나면 멈춘다)."""
    notes = []
    while i < len(lines):
        t = lines[i].strip()
        if not t: i += 1; continue
        if t.startswith("#") or t.startswith("|") or t.startswith("**") or t == "---": break
        notes.append(t); i += 1
    return notes, i'''
C_NEW = '''def collect_notes(lines, i):
    """표 바로 뒤 '첫 문단 하나'만 주석으로 본다(44 와 같은 기준).

    빈 줄을 건너뛰며 계속 읽으면 표 뒤에 오는 본문 문단까지 주석이 된다.
    """
    notes = []
    while i < len(lines) and not lines[i].strip():
        i += 1
    while i < len(lines):
        t = lines[i].strip()
        if not t: break
        if t.startswith("#") or t.startswith("|") or t.startswith("**") or t == "---": break
        notes.append(t); i += 1
    return notes, i'''
assert C_OLD in s, "[45] collect_notes 앵커 없음"
s = s.replace(C_OLD, C_NEW, 1)

if "from docx.shared import" in s and "Inches" not in s.split("from docx.shared import")[1].split("\n")[0]:
    _l = s.split("from docx.shared import")[1].split("\n")[0]
    s = s.replace("from docx.shared import" + _l, "from docx.shared import" + _l + ", Inches", 1)

io.open(p, "w", encoding="utf8").write(s)
compile(io.open(p, encoding="utf8").read(), p, "exec")
print("45_make_appendix_docx.py: 표 서식 + 각주 버그 수정 적용")

# ── ② 절 제목마다 새 쪽 ──────────────────────────────────────────────────
q = "44_make_docx.py"
d = io.open(q, encoding="utf8").read()
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
assert h_old in d, "[44] 제목 처리 앵커 없음"
d = d.replace(h_old, h_new, 1)
io.open(q, "w", encoding="utf8").write(d)
compile(io.open(q, encoding="utf8").read(), q, "exec")
print("44_make_docx.py: ## 절 제목마다 쪽 나눔")
