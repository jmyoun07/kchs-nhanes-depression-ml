# -*- coding: utf-8 -*-
"""표 칸 너비 검사를 60_verify_manuscript.py 에 넣는다.

칸이 줄바꿈 단위보다 좁으면 '0.814-' / '0.837' 처럼 낱말·수치구간이 갈라진다.
눈으로 확인하면 다음 편집에서 되돌아가므로 검사로 만든다.
세 docx 의 모든 표를 검사한다.
"""
import io, os, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")

p = "60_verify_manuscript.py"
s = io.open(p, encoding="utf8").read()
if "표 칸 너비" in s:
    print("이미 적용됨"); raise SystemExit

GUARD = '''
# ══════════════════════════════════════════════════════════════════════
# 표 칸 너비: 칸이 줄바꿈 단위보다 좁으면 낱말이나 수치구간이 갈라진다.
# ══════════════════════════════════════════════════════════════════════
try:
    from docx import Document as _Doc
    _gsrc = io.open(f"{D}/44_make_docx.py", encoding="utf8").read()
    _gg = {"re": re}
    exec(compile(_gsrc[_gsrc.index("def _adv"):_gsrc.index("def auto_widths")], "w", "exec"), _gg)
    _tw = _gg["text_width"]; _S = chr(1)
    def _worst(v):
        _x = re.sub(r"(?<=\\d)-(?=\\d)", _S, v)
        _b, _w = "", 0.0
        for _t in re.split(r"[\\s/\\-]+", _x):
            if not _t: continue
            _t = _t.replace(_S, "-")
            if _tw(_t) > _w: _b, _w = _t, _tw(_t)
        return _b, _w
    for _f in ("논문원고_우울ML_17변수.docx", "부록1_보충방법및결과.docx", "부록2_TRIPOD체크리스트.docx"):
        _pth = f"{D}/{_f}"
        if not os.path.exists(_pth): continue
        for _ti, _t in enumerate(_Doc(_pth).tables):
            _ws = [c.width / 914400 if c.width else 0 for c in _t.rows[0].cells]
            for _k, _wd in enumerate(_ws):
                _bb, _bw = "", 0.0
                for _r in _t.rows:
                    if _k >= len(_r.cells): continue
                    _b2, _w2 = _worst(_r.cells[_k].text)
                    if _w2 > _bw: _bb, _bw = _b2, _w2
                need(_bw + 0.17 <= _wd + 0.02,
                     f"[{_f}] 표{_ti+1} 칸{_k+1} 너비 {_wd:.2f}\\" < 필요 {_bw+0.17:.2f}\\" ({_bb!r} 가 갈라진다)")
except ImportError:
    pass

'''

anchor = [l for l in s.split("\n") if l.startswith("print(") and "검증 항목" in l][0]
i = s.index(anchor)
io.open(p, "w", encoding="utf8").write(s[:i] + GUARD.replace("\\\\", "\\") + "\n" + s[i:])
compile(io.open(p, encoding="utf8").read(), p, "exec")
print("60_verify_manuscript.py: 표 칸 너비 검사 추가 (세 docx 전체)")
