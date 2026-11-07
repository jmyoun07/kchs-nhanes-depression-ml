# -*- coding: utf-8 -*-
"""작업용_v2 에서 손으로 하신 배치 조정을 생성기에 옮긴다.

옮기는 것
 ① 그림 문단의 '단락을 나누지 않음'(keep_together) 해제 - 그림 왼쪽 점의 원인
 ② 그림별 크기 - v2 에서 맞추신 값 그대로
 ③ 표 행의 최소 높이 0.19in - v2 에서 늘리신 세로 간격
 ④ 표 1 의 칸 너비 - v2 에서 옮기신 값

②④ 는 손으로 맞춘 값이라 자동 계산 대신 '덮어쓰기 표' 에 적어 둔다. 나중에 다시
조정하시면 이 표만 고치면 되고, 다시 만들어도 유지된다.
"""
import io, os, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")

p = "44_make_docx.py"
s = io.open(p, encoding="utf8").read()

# ── 덮어쓰기 표를 파일 앞쪽에 심는다 ─────────────────────────────────────
OVR = '''
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

'''
anchor = "def keep_next(p):"
assert anchor in s, "삽입 위치 앵커 없음"
if "FIG_SIZE" not in s:
    s = s.replace(anchor, OVR.lstrip("\n") + anchor, 1)

# ── ① 그림 keep_together 해제 + ② 그림별 크기 ──────────────────────────
a = '''        pic = doc.add_paragraph(); pic.paragraph_format.alignment = AL.CENTER
        pic.paragraph_format.keep_together = True
        pic.paragraph_format.line_spacing = 1.0
        pic.add_run().add_picture(path, width=Inches(6.3))   # 작업용 파일에서 맞추신 폭'''
b = '''        pic = doc.add_paragraph(); pic.paragraph_format.alignment = AL.CENTER
        # 단락을 나누지 않음(keep_together)은 그림 왼쪽에 표식을 만든다. 쓰지 않는다.
        pic.paragraph_format.line_spacing = 1.0
        _w, _hh = FIG_SIZE.get(n, (6.3, None))
        if _hh:
            pic.add_run().add_picture(path, width=Inches(_w), height=Inches(_hh))
        else:
            pic.add_run().add_picture(path, width=Inches(_w))'''
assert a in s, "그림 삽입 앵커 없음"
s = s.replace(a, b, 1)

# ── ③ 행 최소 높이 + ④ 표별 칸 너비 ────────────────────────────────────
a2 = "        set_col_widths(t, header, body, pt=_pt.pt)"
b2 = ('        _tno = len([x for x in doc.tables]) if False else None\n'
      '        set_col_widths(t, header, body, pt=_pt.pt, override=COL_W.get(_tbl_no))\n'
      '        set_row_min_height(t)')
assert a2 in s, "표 조립 앵커 없음"
s = s.replace(a2, b2, 1)

# 표 번호를 센다
a3 = "        t = doc.add_table(rows=1, cols=len(header))"
b3 = ("        _tbl_no = len(doc.tables) + 1\n"
      "        t = doc.add_table(rows=1, cols=len(header))")
assert a3 in s, "표 생성 앵커 없음"
s = s.replace(a3, b3, 1)

# set_col_widths 가 덮어쓰기를 받도록
a4 = '''def set_col_widths(t, header, body, pt=10.0):
    t.autofit = False
    lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed")
    t._tbl.tblPr.append(lay)
    ws = auto_widths(header, body, pt=pt)'''
b4 = '''def set_col_widths(t, header, body, pt=10.0, override=None):
    t.autofit = False
    lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed")
    t._tbl.tblPr.append(lay)
    ws = override if override and len(override) == len(header) else auto_widths(header, body, pt=pt)'''
assert a4 in s, "set_col_widths 앵커 없음"
s = s.replace(a4, b4, 1)

io.open(p, "w", encoding="utf8").write(s)
compile(io.open(p, encoding="utf8").read(), p, "exec")
print("44_make_docx.py 갱신")
print("  ① 그림 keep_together 해제 (그림 왼쪽 점 제거)")
print("  ② 그림 7종 크기를 v2 값으로 고정")
print("  ③ 표 행 최소 높이 0.19in (세로 간격)")
print("  ④ 표 1 칸 너비를 v2 값으로 고정")
