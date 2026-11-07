# -*- coding: utf-8 -*-
"""① 생성기 소스의 제어문자를 chr(1) 표기로 바꾼다.
   ② 세 docx 의 모든 표에서 '한 줄에 못 들어가는 칸' 을 찾는다.

워드는 공백·'/'·하이픈 뒤에서 줄을 바꾼다. 다만 수치 구간(0.814-0.837)의 하이픈에서
갈라지면 보기 흉하므로, 그 구간은 통째로 한 줄에 들어가야 한다.
"""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")

SEP = chr(1)
for p in ("44_make_docx.py", "45_make_appendix_docx.py"):
    s = io.open(p, encoding="utf8").read()
    if SEP in s:
        s = s.replace(f'"{SEP}"', "_SEP")
        s = s.replace("        toks = []\n", "        _SEP = chr(1)\n        toks = []\n", 1)
        s = s.replace("            toks = []\n", "            _SEP = chr(1)\n            toks = []\n", 1)
        io.open(p, "w", encoding="utf8").write(s)
        compile(io.open(p, encoding="utf8").read(), p, "exec")
        print(f"{p}: 소스의 제어문자를 chr(1) 표기로 교체")
    else:
        print(f"{p}: 제어문자 없음")

# ── 폭 점검 ──────────────────────────────────────────────────────────────
from docx import Document
def tokens(v):
    """워드가 줄을 바꿀 수 있는 조각. 수치 구간의 하이픈은 끊기지 않는 것으로 본다."""
    x = re.sub(r"(?<=\d)-(?=\d)", SEP, v)
    return [w.replace(SEP, "-") for w in re.split(r"[\s/\-]+", x) if w]

CHAR = 0.085      # 10pt Times 평균 글자폭(인치)
PAD  = 0.12
bad = 0
for f in ("논문원고_우울ML_17변수.docx", "부록1_보충방법및결과.docx", "부록2_TRIPOD체크리스트.docx"):
    d = Document(f)
    for i, t in enumerate(d.tables):
        ws = [c.width / 914400 if c.width else 0 for c in t.rows[0].cells]
        for k, w in enumerate(ws):
            worst, worstv = 0, ""
            for r in range(len(t.rows)):
                if k >= len(t.rows[r].cells): continue
                for tk in tokens(t.rows[r].cells[k].text):
                    if len(tk) > worst: worst, worstv = len(tk), tk
            need = worst * CHAR + PAD
            if need > w + 0.03:
                bad += 1
                print(f"  {f[:16]:<18} 표{i+1:>2} 칸{k+1} 너비 {w:.2f}\" < 필요 {need:.2f}\"  최장 {worstv!r}")
print(f"\n한 줄에 못 들어가는 칸 {bad}개")
