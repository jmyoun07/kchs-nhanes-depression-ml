# -*- coding: utf-8 -*-
"""45_make_appendix_docx.py - Multimedia Appendix 1~4를 docx로 생성.

서식은 JMIR 예시파일(윤수 예시파일/Multimedia_Appendix_1_supplementary_methods_results_final.docx)을 따른다.
본문 docx(44_make_docx.py)와 동일한 규칙: Times New Roman 11pt · 양쪽맞춤 · 문단 앞뒤 공백 0 ·
표 캡션은 표 **위**, 주석은 표 **아래**, 표는 위·아래와 머리글 아래에만 1pt 실선.
부록은 표가 많아 줄간격 1.5를 쓴다(본문은 2.0).

원본은 항상 docs/*.md 이고, docx는 이 스크립트로만 재생성한다.
"""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH as AL
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

def _clear_props(doc):
    """문서 속성을 비운다. 저자란을 비워 둔 논문인데 워드 속성에 이름이 남으면
    익명 심사에서 신원이 샌다. python-docx 는 기본값으로 자기 이름을 넣는다."""
    cp = doc.core_properties
    for _a in ("author", "last_modified_by", "comments", "category", "keywords",
               "subject", "title", "identifier", "content_status"):
        try: setattr(cp, _a, "")
        except Exception: pass
    cp.revision = 1

from docx.oxml import OxmlElement

D = os.path.dirname(os.path.abspath(__file__))
# 예시파일처럼 부록은 2개로 낸다.
# 부록1 = 보충 방법·결과(크로스워크 → 임상유용성 → 천장분석)를 한 문서로, 부록2 = TRIPOD+AI 체크리스트
JOBS = [
    (["docs/appendix2_crosswalk.md", "docs/appendix5_methods.md",
      "docs/appendix3_utility.md", "docs/appendix4_ceiling.md"],
     "부록1_보충방법및결과.docx",
     "# Multimedia Appendix 1. Supplementary Methods and Results"),
    (["docs/TRIPOD_AI_checklist.md"], "부록2_TRIPOD체크리스트.docx", None),
]
BODY_PT, SMALL_PT, TITLE_PT = Pt(11), Pt(10), Pt(14)
INDENT = Pt(18)
HEAD_COLOR = RGBColor(0x2E, 0x74, 0xB5)   # 예시파일 제목 색

TOKEN = re.compile(r"(\*\*.+?\*\*|`[^`\n]+`|\^[a-z](?![a-z])|(?<![\w*])\*[^*\n]+?\*(?![\w*]))")
def add_runs(par, text, size=None, bold_all=False):
    for piece in TOKEN.split(text):
        if not piece: continue
        if piece.startswith("**") and piece.endswith("**"):
            r = par.add_run(piece[2:-2]); r.bold = True
        elif piece.startswith("`") and piece.endswith("`"):
            r = par.add_run(piece[1:-1]); r.font.name = "Consolas"
        elif piece.startswith("^") and len(piece) == 2:
            r = par.add_run(piece[1]); r.font.superscript = True
        elif piece.startswith("*") and piece.endswith("*"):
            r = par.add_run(piece[1:-1]); r.italic = True
        else:
            r = par.add_run(piece)
        if bold_all: r.bold = True
        if size: r.font.size = size

def split_row(l):
    """셀 분리. 마크다운에서 이스케이프한 파이프(\\|)는 셀 구분자가 아니다."""
    cells = re.split(r"(?<!\\)\|", l.strip().strip("|"))
    return [c.strip().replace("\\|", "|") for c in cells]

# ── 표 주석 위첨자 문자 (예시파일 방식) ──────────────────────────────
# 두 낱말짜리 용어(Brier score)도 개별 문자를 받도록 한 낱말을 더 허용한다
NOTE_ABBR = re.compile(r"^([A-Za-z][A-Za-z0-9+\-]{1,13}(?: [a-z]+)?):\s")
def collect_notes(lines, i):
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
    return notes, i

def letter_notes(header, body, notes):
    """주석에 문자를 붙이고, 표 안에 앵커를 심는다. (header, body, [(문자, 본문)]) 반환."""
    if not notes: return header, body, []
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



ROW_MIN_IN = 0.19     # 작업용_v2 에서 늘리신 표 세로 간격과 같은 기준

def set_row_min_height(tbl, inches=ROW_MIN_IN):
    """행이 눌리지 않게 최소 높이를 준다. 내용이 더 크면 알아서 늘어난다."""
    for r in tbl.rows:
        trPr = r._tr.get_or_add_trPr()
        h = OxmlElement("w:trHeight")
        h.set(qn("w:val"), str(int(inches * 1440)))
        h.set(qn("w:hRule"), "atLeast")
        trPr.append(h)

def rows_cant_split(tbl):
    """행이 쪽 경계에서 갈라지지 않게 한다."""
    for r in tbl.rows:
        trPr = r._tr.get_or_add_trPr()
        e = OxmlElement("w:cantSplit"); trPr.append(e)

def set_borders(tbl):
    b = OxmlElement("w:tblBorders")
    for edge, val in [("top", "single"), ("left", "nil"), ("bottom", "single"),
                      ("right", "nil"), ("insideH", "nil"), ("insideV", "nil")]:
        e = OxmlElement(f"w:{edge}"); e.set(qn("w:val"), val)
        if val == "single":
            e.set(qn("w:sz"), "8"); e.set(qn("w:space"), "0"); e.set(qn("w:color"), "auto")
        b.append(e)
    tbl._tbl.tblPr.append(b)

def header_underline(tbl):
    for c in tbl.rows[0].cells:
        tcPr = c._tc.get_or_add_tcPr()
        bd = OxmlElement("w:tcBorders"); e = OxmlElement("w:bottom")
        e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "8"); e.set(qn("w:space"), "0"); e.set(qn("w:color"), "auto")
        bd.append(e); tcPr.append(bd)

def convert(srcs, out, banner=None):
    md = ("\n\n".join(io.open(x, encoding="utf-8").read() for x in srcs)).replace("\r\n", "\n")
    if banner: md = banner + "\n\n" + md
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"; st.font.size = BODY_PT
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    pf = st.paragraph_format
    pf.line_spacing = 1.5; pf.space_before = Pt(0); pf.space_after = Pt(0)
    for sec in doc.sections:
        sec.top_margin = sec.bottom_margin = sec.left_margin = sec.right_margin = Inches(1)

    def para(text="", *, size=None, bold=False, indent=None, spacing=None, color=None):
        p = doc.add_paragraph()
        p.paragraph_format.alignment = AL.JUSTIFY
        p.paragraph_format.first_line_indent = indent
        if spacing is not None: p.paragraph_format.line_spacing = spacing
        if text: add_runs(p, text, size=size, bold_all=bold)
        if color:
            for _r in p.runs: _r.font.color.rgb = color
        return p

    CELLNUM = re.compile(r"^([+-]?\d[\d.]*)\s+(\(.+\))$")

    def vcenter(cell):
        tcPr = cell._tc.get_or_add_tcPr()
        va = OxmlElement("w:vAlign"); va.set(qn("w:val"), "center"); tcPr.append(va)

    def _adv(ch):
        """Times New Roman 의 글자 폭(em). 숫자는 0.5em 고정폭이다."""
        if ch.isdigit(): return 0.500
        if ch in ".,;:'": return 0.250
        if ch in "-()/": return 0.320
        if ch in "+=<>": return 0.564
        if ch == "%": return 0.833
        if ch == " ": return 0.250
        if ch.isupper(): return 0.722
        return 0.480

    def text_width(s, pt=10.0):
        """문자열이 차지하는 폭(인치)."""
        return sum(_adv(c) for c in s) * pt / 72.0

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
                _x = re.sub(r"(?<=\d)-(?=\d)", _SEP, x)
                for w in re.split(r"[\s/\-]+", _x):
                    if w: b = max(b, text_width(w.replace(_SEP, "-"), base))
            longest.append(b)
        room = total - n * pad
        if room <= 0 or sum(longest) <= room: return base
        return max(floor, base * room / sum(longest))

    def auto_widths(header, body, total=6.5, pad=0.17, pt=10.0):
        """칸 너비를 내용으로 정한다. 줄바꿈 단위가 잘리지 않을 만큼은 보장한다.

        워드는 공백·'/'·하이픈 뒤에서 줄을 바꾼다. 다만 수치 구간(0.814-0.837)에서
        갈라지면 보기 흉하므로 그 하이픈은 끊기지 않는 것으로 본다.
        """
        _SEP = chr(1)
        n = len(header)
        cols = [[header[i]] + [(r[i] if i < len(r) else "") for r in body] for i in range(n)]
        wt, mins = [], []
        for c in cols:
            avg = sum(text_width(x, pt) for x in c) / max(1, len(c))
            longest = 0.0
            for x in c:
                _x = re.sub(r"(?<=\d)-(?=\d)", _SEP, x)
                for w in re.split(r"[\s/\-]+", _x):
                    if w: longest = max(longest, text_width(w.replace(_SEP, "-"), pt))
            wt.append(max(avg * 0.75, longest, 0.1))
            mins.append(longest + pad)
        if sum(mins) > total:                      # 지면이 모자라면 최소치를 비례 축소한다
            k = total / sum(mins); mins = [m * k for m in mins]
        s = sum(wt); w = [total * x / s for x in wt]
        for _ in range(n * 2):
            small = [i for i, x in enumerate(w) if x < mins[i] - 1e-9]
            if not small: break
            need = sum(mins[i] - w[i] for i in small)
            big = [i for i in range(n) if w[i] > mins[i] + 1e-9]
            pool = sum(w[i] - mins[i] for i in big)
            if not big or pool < need - 1e-9:      # 더 뺄 곳이 없으면 최소치로 균등 정착
                return [m * total / sum(mins) for m in mins]
            for i in small: w[i] = mins[i]
            for i in big: w[i] -= need * (w[i] - mins[i]) / pool
        # 같은 성격의 수치 열은 같은 폭으로 맞춘다(작업용 파일에서 손으로 하신 조정).
        # 내용대로만 계산하면 1.06/1.12 처럼 들쭉날쭉해서 표가 정돈돼 보이지 않는다.
        def _numericish(col):
            vals = [x for x in col[1:] if x.strip()]
            if len(vals) < 2: return False
            hit = sum(1 for v in vals if re.fullmatch(r"[+-]?[\d.,]+(\s*\([^)]*\))?", v.strip()))
            return hit / len(vals) >= 0.7
        _num = [i for i, c in enumerate(cols) if _numericish(c)]
        if len(_num) >= 2:
            _avg = sum(w[i] for i in _num) / len(_num)
            if _avg >= max(mins[i] for i in _num):        # 균일화해도 아무 칸도 좁아지지 않을 때만
                for i in _num: w[i] = _avg
        return w
    def set_col_widths(t, header, body, pt=10.0):
        t.autofit = False
        lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed")
        t._tbl.tblPr.append(lay)
        for row in t.rows:
            for c, w in zip(row.cells, auto_widths(header, body, pt=pt)):
                c.width = Inches(w)

    def cell_fmt(cell, txt, bold=False, pt=None):
        p = cell.paragraphs[0]; p.text = ""
        p.paragraph_format.alignment = AL.CENTER
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.space_before = Pt(0); p.paragraph_format.space_after = Pt(0)
        m = CELLNUM.match(txt.strip())
        if m:                # 값과 구간을 각각 한 줄에 - 구간이 중간에서 갈라지지 않게
            add_runs(p, m.group(1), size=(pt or SMALL_PT), bold_all=bold)
            p.add_run().add_break()
            add_runs(p, m.group(2), size=(pt or SMALL_PT), bold_all=bold)
        else:
            add_runs(p, txt, size=(pt or SMALL_PT), bold_all=bold)
        vcenter(cell)

    lines = md.split("\n"); i = 0; prev_table = False
    while i < len(lines):
        ln = lines[i].rstrip()
        if ln.strip() in ("", "---"):
            i += 1; continue
        if ln.lstrip().startswith("|") and i + 1 < len(lines) \
           and set(lines[i+1].replace("|", "").replace(" ", "")) <= set("-:") and "-" in lines[i+1]:
            header = split_row(ln); body = []
            i += 2
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                body.append(split_row(lines[i])); i += 1
            t = doc.add_table(rows=1, cols=len(header))
            t.alignment = WD_TABLE_ALIGNMENT.CENTER
            set_borders(t)
            for c, txt in zip(t.rows[0].cells, header): cell_fmt(c, txt, bold=True)
            for row in body:
                cells = t.add_row().cells
                for c, txt in zip(cells, row + [""] * (len(header) - len(row))): cell_fmt(c, txt)
            notes, i = collect_notes(lines, i)
            header, body, lettered = letter_notes(header, body, notes)
            _pt = Pt(fit_pt(header, body))   # 지면에 안 들어가면 글자를 줄인다
            for c, txt in zip(t.rows[0].cells, header): cell_fmt(c, txt, bold=True, pt=_pt)
            for ri, row in enumerate(body):
                for c, txt in zip(t.rows[ri + 1].cells, row + [""] * (len(header) - len(row))): cell_fmt(c, txt, pt=_pt)
            set_col_widths(t, header, body, pt=_pt.pt)
            header_underline(t)
            rows_cant_split(t)
            set_row_min_height(t)
            for _ch, _tx in lettered:
                para(f"^{_ch}{_tx}", size=SMALL_PT, spacing=1.0)
            prev_table = False
            continue
        if ln.startswith("#"):
            lvl = len(ln) - len(ln.lstrip("#")); txt = ln[lvl:].strip()
            # 'Table SN.' 은 절 제목이 아니라 표 캡션이다. 예시파일은 캡션을
            # 검은 굵은 10pt 로 쓰고 파란색은 절 제목에만 쓴다.
            if re.match(r"^Table S\d+\.", txt):
                if not txt.endswith("."): txt += "."
                # 캡션은 표와 같은 쪽에 둔다(쪽 경계에서 갈라지지 않게)
                para(txt, size=SMALL_PT, bold=True)
            else:
                # 예시파일은 큰 절(##)만 파랑이고 그 아래는 검정이다. 전부 파랑으로 하면
                # 제목 서른 개가 모두 같은 무게로 보여 구조가 읽히지 않는다.
                para(txt, size=(TITLE_PT if lvl == 1 else (None if lvl <= 3 else SMALL_PT)),
                     bold=True, color=(HEAD_COLOR if lvl == 2 else None))
            prev_table = False; i += 1; continue
        txt = ln.strip()
        if ln.startswith("> "):
            p = para(ln[2:].strip(), size=SMALL_PT, spacing=1.0)
            p.paragraph_format.left_indent = Pt(18)
        elif txt.startswith("- "):
            para(txt, spacing=1.0)
        elif prev_table:
            para(txt, size=SMALL_PT, spacing=1.0)      # 표 주석은 표 아래·10pt
        else:
            para(txt, indent=(None if txt.startswith("**") and txt.endswith("**") else INDENT))
            prev_table = False
        if not (prev_table and not txt.startswith("**")): prev_table = prev_table and not txt.startswith("**")
        i += 1
    path = os.path.join(D, out)
    try:
        _clear_props(doc); doc.save(path)
    except PermissionError:
        path = path.replace(".docx", "_rev1.docx"); _clear_props(doc); doc.save(path)
        print("  [알림] 기존 파일이 열려 있어 새 이름으로 저장")
    print(f"  → {os.path.basename(path)}  (표 {len(doc.tables)})")

for srcs, out, banner in JOBS:
    print(f"[{out}]")
    convert([os.path.join(D, x) for x in srcs], out, banner)
print("완료.")
