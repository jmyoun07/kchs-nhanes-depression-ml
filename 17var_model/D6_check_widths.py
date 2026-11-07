# -*- coding: utf-8 -*-
"""세 docx 의 모든 표에서 '줄바꿈 단위가 칸에 안 들어가는 곳' 을 찾는다.
생성기와 같은 글자폭 모델을 쓴다."""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")
from docx import Document

src = io.open("44_make_docx.py", encoding="utf8").read()
g = {"re": re}
exec(compile(src[src.index("def _adv"):src.index("def auto_widths")], "w", "exec"), g)
text_width = g["text_width"]
SEP = chr(1)

def worst_token(v):
    x = re.sub(r"(?<=\d)-(?=\d)", SEP, v)
    best, bw = "", 0.0
    for w in re.split(r"[\s/\-]+", x):
        if not w: continue
        w = w.replace(SEP, "-")
        if text_width(w) > bw: best, bw = w, text_width(w)
    return best, bw

bad = 0
for f in ("논문원고_우울ML_17변수.docx", "부록1_보충방법및결과.docx", "부록2_TRIPOD체크리스트.docx"):
    d = Document(f)
    for i, t in enumerate(d.tables):
        ws = [c.width / 914400 if c.width else 0 for c in t.rows[0].cells]
        for k, w in enumerate(ws):
            best, bw = "", 0.0
            for r in range(len(t.rows)):
                if k >= len(t.rows[r].cells): continue
                b2, w2 = worst_token(t.rows[r].cells[k].text)
                if w2 > bw: best, bw = b2, w2
            if bw + 0.17 > w + 0.02:
                bad += 1
                print(f"  {f[:16]:<18} 표{i+1:>2} 칸{k+1}  너비 {w:.2f}\" 필요 {bw+0.17:.2f}\"  최장 {best!r}")
    print(f"{f[:20]:<22} 표 {len(d.tables)}개")
print(f"\n칸에 안 들어가는 곳 {bad}개")
