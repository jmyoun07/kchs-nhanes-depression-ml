# -*- coding: utf-8 -*-
"""44_make_docx.py - manuscript_17var.md → 제출용 docx.

서식은 JMIR 예시파일(윤수 예시파일/JMIR Adolescent Depressed Mood and Suicidality.docx)을 그대로 따른다.

  · 여백 1인치 · Times New Roman 11pt · 줄간격 2.0 · 양쪽맞춤 · 문단 앞뒤 공백 0
  · 본문 문단 첫줄 들여쓰기 0.25"(18pt) / 제목은 들여쓰기 없이 굵게
  · 표: 캡션이 표 **위**, 표 위·아래와 머리글 아래에만 1pt 실선, 머리글 반복 없음, 표 안 10pt·줄간격 1.0
  · 표 주석: 표 **아래**, 10pt·줄간격 1.0
  · 그림: 캡션이 그림 **위**, 10pt, 첫 인용 문단 바로 뒤에 배치, 가로 6.1인치
  · 참고문헌: 'N.<탭>본문', 내어쓰기, 10pt, 줄간격 1.5
  · 뒷단 순서: Acknowledgments → Funding → Data Availability → Authors' Contributions →
    Conflicts of Interest → Use of Generative AI Tools → Multimedia Appendix 2..N → References → Abbreviations

원고 본문의 유일한 원본은 docs/manuscript_17var.md 이고, docx는 항상 이 스크립트로 재생성한다.
"""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH as AL, WD_BREAK
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
SRC = f"{D}/docs/manuscript_17var.md"
OUT = f"{D}/논문원고_우울ML_17변수.docx"
FIG = f"{D}/results/figures"
FIGURES = [
    ("Figure 1", "characteristics.png",  "Weighted sample characteristics across cohorts: panel A, prevalence/proportion; panel B, means (exact values in Multimedia Appendix 1, Table S4)."),
    ("Figure 2", "calibration.png",      "Calibration plots (internal, NHANES pre, NHANES post; original probabilities)."),
    ("Figure 3", "roc_curves.png",       "Receiver operating characteristic curves (weighted) for internal, NHANES pre, and NHANES post."),
    ("Figure 4", "sensitivity.png",      "Sensitivity analyses: weighted AUROC across specifications for internal, NHANES pre, and NHANES post (exact values in Multimedia Appendix 1, Table S5)."),
    ("Figure 5", "shap_importance.png",  "Global SHAP importance of the 17 predictors (KCHS)."),
    ("Figure 6", "subgroup_forest.png",  "Subgroup discrimination (weighted AUROC with 95% CIs) by sex, age, education and household income quintile (KCHS hold-out) and by race/ethnicity (NHANES pre and post). Dotted vertical lines mark the corresponding cohort-level AUROC."),
    ("Figure 7", "dca.png",              "Decision curves (weighted net benefit) for the three datasets, each with its own screen-all and screen-none default strategies; original, non-recalibrated probabilities."),
]
BODY_PT, SMALL_PT, TITLE_PT = Pt(11), Pt(10), Pt(14)
INDENT = Pt(18)
HEAD_COLOR = RGBColor(0x2E, 0x74, 0xB5)   # 예시파일 제목 색

doc = Document()
st = doc.styles["Normal"]
st.font.name = "Times New Roman"; st.font.size = BODY_PT
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
_pf = st.paragraph_format
_pf.line_spacing = 2.0; _pf.space_before = Pt(0); _pf.space_after = Pt(0)   # 정렬은 문단마다 지정(예시와 동일)
for sec in doc.sections:
    sec.top_margin = sec.bottom_margin = sec.left_margin = sec.right_margin = Inches(1)

TOKEN = re.compile(r"(\*\*.+?\*\*|\^[a-z](?![a-z])|(?<![\w*])\*[^*\n]+?\*(?![\w*]))")
def add_runs(par, text, size=None, bold_all=False):
    for piece in TOKEN.split(text):
        if not piece: continue
        if piece.startswith("**") and piece.endswith("**"):
            r = par.add_run(piece[2:-2]); r.bold = True
        elif piece.startswith("^") and len(piece) == 2:
            r = par.add_run(piece[1]); r.font.superscript = True
        elif piece.startswith("*") and piece.endswith("*"):
            r = par.add_run(piece[1:-1]); r.italic = True
        else:
            r = par.add_run(piece)
        if bold_all: r.bold = True
        if size: r.font.size = size
    return par

def para(text="", *, size=None, bold=False, indent=None, spacing=None, align=AL.JUSTIFY, color=None):
    p = doc.add_paragraph()
    p.paragraph_format.alignment = align
    p.paragraph_format.first_line_indent = indent
    if spacing is not None: p.paragraph_format.line_spacing = spacing
    if text: add_runs(p, text, size=size, bold_all=bold)
    if color:
        for _r in p.runs: _r.font.color.rgb = color
    return p

# ── 손으로 맞춘 배치값 (작업용_v2 에서 조정하신 것) ─────────────────────
# 자동 계산보다 이 값을 우선한다. 다시 조정하시면 여기만 고치면 된다.
FIG_SIZE = {          # 그림 번호: (가로, 세로) 인치
    1: (6.300, 2.945), 2: (6.298, 3.325), 3: (6.299, 4.052), 4: (6.300, 3.187),
    5: (6.298, 2.727), 6: (6.299, 4.805), 7: (6.442, 3.337),
}
COL_W = {             # 표 번호: 칸 너비(인치). 없으면 자동 계산
    1: [0.370, 1.431, 1.053, 1.255, 2.390],
}
ROW_MIN_IN = 0.19     # 표 행의 최소 높이. v2 에서 늘리신 세로 간격

def set_row_min_height(tbl, inches=ROW_MIN_IN):
    """행이 눌리지 않게 최소 높이를 준다. 내용이 더 크면 알아서 늘어난다."""
    for r in tbl.rows:
        trPr = r._tr.get_or_add_trPr()
        h = OxmlElement("w:trHeight")
        h.set(qn("w:val"), str(int(inches * 1440)))
        h.set(qn("w:hRule"), "atLeast")
        trPr.append(h)

def keep_next(p):
    """이 문단을 다음 것과 같은 쪽에 둔다(캡션이 그림·표에서 떨어지지 않게)."""
    p.paragraph_format.keep_with_next = True
    return p

def rows_cant_split(tbl):
    """행이 쪽 경계에서 갈라지지 않게 한다."""
    for r in tbl.rows:
        trPr = r._tr.get_or_add_trPr()
        e = OxmlElement("w:cantSplit"); trPr.append(e)

def set_borders(tbl):
    """표 위·아래만 1pt 실선, 좌우·내부는 없음 (예시파일과 동일)."""
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

def set_col_widths(t, header, body, pt=10.0, override=None):
    t.autofit = False
    lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed")
    t._tbl.tblPr.append(lay)
    ws = override if override and len(override) == len(header) else auto_widths(header, body, pt=pt)
    for row in t.rows:
        for c, w in zip(row.cells, ws):
            c.width = Inches(w)

def cell_fmt(cell, txt, bold=False, pt=None):
    p = cell.paragraphs[0]; p.text = ""
    p.paragraph_format.alignment = AL.CENTER
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.space_before = Pt(0); p.paragraph_format.space_after = Pt(0)
    m = CELLNUM.match(txt.strip())
    if m:                    # 값과 구간을 각각 한 줄에 - 구간이 중간에서 갈라지지 않게
        add_runs(p, m.group(1), size=(pt or SMALL_PT), bold_all=bold)
        p.add_run().add_break()
        add_runs(p, m.group(2), size=(pt or SMALL_PT), bold_all=bold)
    else:
        add_runs(p, txt, size=(pt or SMALL_PT), bold_all=bold)
    vcenter(cell)

md = io.open(SRC, encoding="utf-8").read().replace("\r\n", "\n")

# 캡션의 단일 원본은 원고 말미의 'Figure legends' 블록이다.
# docx 에서는 별도 절로 넣지 않고 각 그림 위에 붙인다(예시파일과 동일).
LEG = {}
_lm = re.search(r"^##\s+Figure Legends\s*\n+(.*?)(?=\n##\s|\Z)", md, re.S | re.M)
if _lm:
    for _m in re.finditer(r"Figure (\d+): (.+?)(?=\s*Figure \d+:|\s*Tables:|$)", _lm.group(1), re.S):
        LEG[int(_m.group(1))] = " ".join(_m.group(2).split())
    md = md[:_lm.start()] + md[_lm.end():]
if len(LEG) != len(FIGURES):
    raise SystemExit(f"[44] 원고 'Figure Legends' 절에서 캡션 {len(LEG)}/{len(FIGURES)}개만 회수 - 캡션 단일원본이 끊겼습니다.")

lines = md.split("\n")
def split_row(l):
    """셀 분리. 마크다운에서 이스케이프한 파이프(\\|)는 셀 구분자가 아니다."""
    cells = re.split(r"(?<!\\)\|", l.strip().strip("|"))
    return [c.strip().replace("\\|", "|") for c in cells]

# ── 표 주석 위첨자 문자 (예시파일 방식) ──────────────────────────────
# 두 낱말짜리 용어(Brier score)도 개별 문자를 받도록 한 낱말을 더 허용한다
NOTE_ABBR = re.compile(r"^([A-Za-z][A-Za-z0-9+\-]{1,13}(?: [a-z]+)?):\s")
def collect_notes(lines, i):
    """표 바로 뒤 '첫 문단 하나'만 주석으로 본다.

    예전에는 빈 줄을 건너뛰며 계속 읽어서 표 뒤에 오는 Methods 본문 문단까지
    주석으로 만들었다. 표 1 밑에 본문 1000단어가 붙어 있던 원인이다.
    """
    notes = []
    while i < len(lines) and not lines[i].strip():   # 표와 주석 사이의 빈 줄
        i += 1
    while i < len(lines):
        t = lines[i].strip()
        if not t: break                              # 빈 줄 = 주석 끝, 이후는 본문
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



fig_done = set()
def cited_figures(text):
    """'Figure 4' · 'Figures 2 and 3' · 'Figures 1, 2 and 3' 을 모두 인식."""
    out = []
    for m in re.finditer(r"Figures?\s+((?:\d+)(?:\s*(?:,|and)\s*\d+)*)", text):
        out += [int(x) for x in re.findall(r"\d+", m.group(1))]
    return out

def maybe_figure(text):
    """문단이 Figure N 을 처음 인용하면 그 뒤에 캡션과 그림을 넣는다(캡션이 그림 위)."""
    nums = cited_figures(text)
    for n, (label, fn, cap) in enumerate(FIGURES, 1):
        cap = LEG.get(n, cap)
        if label in fig_done or n not in nums: continue
        path = f"{FIG}/{fn}"
        if not os.path.exists(path): continue
        para(f"**{label}.** {cap}", size=SMALL_PT, spacing=1.0)
        pic = doc.add_paragraph(); pic.paragraph_format.alignment = AL.CENTER
        # 단락을 나누지 않음(keep_together)은 그림 왼쪽에 표식을 만든다. 쓰지 않는다.
        pic.paragraph_format.line_spacing = 1.0
        _w, _hh = FIG_SIZE.get(n, (6.3, None))
        if _hh:
            pic.add_run().add_picture(path, width=Inches(_w), height=Inches(_hh))
        else:
            pic.add_run().add_picture(path, width=Inches(_w))
        para("", spacing=1.0)
        fig_done.add(label)

i, in_refs, in_abbr, prev_table = 0, False, False, False
while i < len(lines):
    ln = lines[i].rstrip()
    if ln.strip() in ("", "---"):
        i += 1; continue

    # ── 표 ──
    if ln.lstrip().startswith("|") and i + 1 < len(lines) \
       and set(lines[i+1].replace("|", "").replace(" ", "")) <= set("-:") and "-" in lines[i+1]:
        header = split_row(ln); body = []
        i += 2
        while i < len(lines) and lines[i].lstrip().startswith("|"):
            body.append(split_row(lines[i])); i += 1
        _tbl_no = len(doc.tables) + 1
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
        _tno = len([x for x in doc.tables]) if False else None
        set_col_widths(t, header, body, pt=_pt.pt, override=COL_W.get(_tbl_no))
        set_row_min_height(t)
        header_underline(t)
        rows_cant_split(t)
        for _ch, _tx in lettered:
            para(f"^{_ch}{_tx}", size=SMALL_PT, spacing=1.0)
        prev_table = False
        continue

    # ── 제목 ──
    if ln.startswith("#"):
        lvl = len(ln) - len(ln.lstrip("#")); txt = ln[lvl:].strip()
        if lvl == 1:
            para(txt, size=TITLE_PT, bold=True)
        else:
            # 본문 큰 절만 새 쪽에서 시작한다. 문단 속성(page_break_before) 대신
            # 삽입된 페이지 나누기 문자를 쓴다. 속성으로 하면 워드가 문단 왼쪽에
            # 검은 사각형 표식을 띄우고, 그 표식이 지워지지 않는다는 지적을 받았다.
            # 문자로 넣으면 표식이 없고 백스페이스로 지울 수 있다.
            if lvl == 2 and txt in ("Abstract", "Introduction", "Methods", "Results", "Discussion"):
                _bp = doc.add_paragraph()
                _bp.paragraph_format.line_spacing = 1.0
                _bp.add_run().add_break(WD_BREAK.PAGE)
            # 색은 큰 절(##)에만, 그 급은 예외 없이 전부. 소제목까지 파랗게 하면
            # 제목 서른 개가 같은 무게로 보여 구조가 읽히지 않는다.
            # (예시파일은 Funding 등 다섯 개를 검정으로 내지만, 같은 급을 다르게 칠하지 않는다.)
            _h = para(txt, bold=True, color=(HEAD_COLOR if lvl == 2 else None))
        in_refs = (txt == "References")
        in_abbr = (txt == "Abbreviations")
        prev_table = False
        i += 1; continue

    # ── 참고문헌 ──
    m = re.match(r"^(\d+)\.\s+(.*)$", ln.strip())
    if in_refs and m:
        p = doc.add_paragraph()
        p.paragraph_format.alignment = AL.JUSTIFY
        p.paragraph_format.line_spacing = 1.5
        p.paragraph_format.left_indent = Inches(0.45)
        p.paragraph_format.first_line_indent = Inches(-0.45)
        add_runs(p, f"{m.group(1)}.\t{m.group(2)}", size=SMALL_PT)
        prev_table = False
        i += 1; continue

    # ── 일반 문단 ──
    txt = ln.strip()
    caption = bool(re.match(r"^\*\*Table \d+\.", txt))
    note = prev_table and not caption
    if caption or note:
        para(txt, size=SMALL_PT, spacing=1.0)
    else:
        # 굵은 라벨로 시작하는 문단(초록의 Background: 등)은 첫줄을 들여쓰지 않는다.
        # 문단 속 굵은 라벨은 문체 규칙으로 금지돼 있으므로, 여기 걸리는 것은 초록 라벨과 표제줄뿐이다.
        indent = None if in_abbr or txt.startswith("**") or txt.startswith("- ") else INDENT
        para(txt, indent=indent)
        maybe_figure(txt)
    prev_table = note
    i += 1

missing = [l for l, _, _ in FIGURES if l not in fig_done]
if missing: raise SystemExit(f"[44] 본문에서 인용 지점을 찾지 못한 그림 {missing} - docx 에 그림과 범례가 빠집니다.")
try:
    _clear_props(doc); doc.save(OUT)
except PermissionError:
    OUT = OUT.replace(".docx", "_rev1.docx")
    doc.save(OUT)
    print("  [알림] 기존 docx가 열려 있어 새 파일로 저장했습니다.")
print(f"저장: {OUT}")
print(f"  문단 {len(doc.paragraphs)} · 표 {len(doc.tables)} · 그림 {len(fig_done)}")
