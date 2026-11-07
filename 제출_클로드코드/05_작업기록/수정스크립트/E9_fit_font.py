# -*- coding: utf-8 -*-
"""지면에 안 들어가는 넓은 표는 글자를 줄여 맞춘다.

9열짜리 부록 표(하위군 공정성)는 10pt 로는 6.5인치에 물리적으로 들어가지 않는다.
지금은 최소너비를 비례 축소해서 낱말이 갈라진 채로 나온다.
글자를 줄이면 갈라지지 않고 들어간다. 예시파일도 넓은 표는 작은 글자를 쓴다.
"""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")

FIT = '''
def fit_pt(header, body, total=6.5, pad=0.17, base=10.0, floor=7.0):
    """줄바꿈 단위가 갈라지지 않고 들어갈 글자 크기를 찾는다.

    넓은 표는 10pt 로 지면을 넘는다. 그때는 최소너비를 눌러 낱말을 자르는 대신
    글자를 줄인다. 표 하나가 조금 작아지는 편이 낱말이 갈라지는 것보다 낫다.
    """
    _SEP = chr(1)
    n = len(header)
    cols = [[header[i]] + [(r[i] if i < len(r) else "") for r in body] for i in range(n)]
    longest = []
    for c in cols:
        b = 0.0
        for x in c:
            _x = re.sub(r"(?<=\\d)-(?=\\d)", _SEP, x)
            for w in re.split(r"[\\s/\\-]+", _x):
                if w: b = max(b, text_width(w.replace(_SEP, "-"), base))
        longest.append(b)
    room = total - n * pad
    if room <= 0 or sum(longest) <= room: return base
    return max(floor, base * room / sum(longest))

'''

for p, ind in (("44_make_docx.py", ""), ("45_make_appendix_docx.py", "    ")):
    s = io.open(p, encoding="utf8").read()
    if "def fit_pt(" in s:
        print(f"{p}: 이미 적용됨"); continue
    blk = FIT if ind == "" else "\n".join((ind + l if l.strip() else l) for l in FIT.split("\n"))
    a = s.index(ind + "def auto_widths(")
    s = s[:a] + blk.lstrip("\n") + s[a:]

    # auto_widths 와 cell_fmt 가 글자 크기를 받도록
    s = s.replace(ind + "def auto_widths(header, body, total=6.5, pad=0.17):",
                  ind + "def auto_widths(header, body, total=6.5, pad=0.17, pt=10.0):", 1)
    s = s.replace(ind + "        avg = sum(text_width(x) for x in c) / max(1, len(c))",
                  ind + "        avg = sum(text_width(x, pt) for x in c) / max(1, len(c))", 1)
    s = s.replace('if w: longest = max(longest, text_width(w.replace(_SEP, "-")))',
                  'if w: longest = max(longest, text_width(w.replace(_SEP, "-"), pt))', 1)
    s = s.replace(ind + "def set_col_widths(t, header, body):",
                  ind + "def set_col_widths(t, header, body, pt=10.0):", 1)
    s = s.replace(ind + "    ws = auto_widths(header, body)",
                  ind + "    ws = auto_widths(header, body, pt=pt)", 1)
    s = s.replace(ind + "        for c, w in zip(row.cells, auto_widths(header, body)):",
                  ind + "        for c, w in zip(row.cells, auto_widths(header, body, pt=pt)):", 1)
    s = s.replace(ind + "def cell_fmt(cell, txt, bold=False):",
                  ind + "def cell_fmt(cell, txt, bold=False, pt=None):", 1)
    s = s.replace("add_runs(p, m.group(1), size=SMALL_PT, bold_all=bold)",
                  "add_runs(p, m.group(1), size=(pt or SMALL_PT), bold_all=bold)", 1)
    s = s.replace("add_runs(p, m.group(2), size=SMALL_PT, bold_all=bold)",
                  "add_runs(p, m.group(2), size=(pt or SMALL_PT), bold_all=bold)", 1)
    s = s.replace(ind + "        add_runs(p, txt, size=SMALL_PT, bold_all=bold)\n" + ind + "    vcenter(cell)",
                  ind + "        add_runs(p, txt, size=(pt or SMALL_PT), bold_all=bold)\n" + ind + "    vcenter(cell)", 1)
    io.open(p, "w", encoding="utf8").write(s)
    compile(io.open(p, encoding="utf8").read(), p, "exec")
    print(f"{p}: fit_pt 추가 · auto_widths/set_col_widths/cell_fmt 가 글자 크기를 받음")
