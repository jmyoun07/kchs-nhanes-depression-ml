# -*- coding: utf-8 -*-
"""절마다 '검증기가 실제로 요구하는 문자열' 목록을 뽑는다.

'숫자 든 문장은 전부 동결' 이 두 번 축소를 막았다(Results -170단어, Methods -264단어).
깨지는 것은 60_verify 의 in_doc 토큰과 61_sync 의 정규식 구간뿐이다. 그것만 준다.
"""
import io, os, re, sys
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")

def R(v, nd=3): return f"{Decimal(str(v)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP):.{nd}f}"

# 60_verify 의 in_doc 토큰을 기록하며 실행
src = io.open("60_verify_manuscript.py", encoding="utf8").read()
src = src.replace('    lab = label or token\n', '    lab = label or token\n    _REC.append((token, tuple(where)))\n', 1)
src = re.sub(r"(?m)^sys\.exit\(.*$", "pass", src)
g = {"__name__": "rec", "_REC": [], "__file__": os.path.abspath("60_verify_manuscript.py")}
try: exec(compile(src, "60_rec", "exec"), g)
except SystemExit: pass
TOK = [t for t, w in g["_REC"] if "manuscript" in w]

# 하위군 CI 는 in_doc 을 거치지 않고 직접 검사한다
fa = pd.read_csv("results/primary2_fairness.csv")
TOK += [f"{R(r.CI_lo)}-{R(r.CI_hi)}" for _, r in fa.iterrows()
        if not any(x in str(r.subgroup) for x in ("Education", "Income quintile"))]

# 61_sync 가 원고에서 찾는 정규식
sy = io.open("61_sync_numbers.py", encoding="utf8").read()
PATS = []
for b in re.findall(r'patch\("manuscript_17var\.md",\s*\[(.*?)\]\s*\)', sy, re.S):
    PATS += re.findall(r'r"((?:[^"\\]|\\.)+)"', b)

M = io.open("docs/manuscript_17var.md", encoding="utf8").read()
SECS = {"methods": M.split("\n## Methods\n")[-1].split("\n## Results")[0],
        "results": M.split("\n## Results\n")[-1].split("\n## Discussion")[0],
        "discussion": M.split("\n## Discussion\n")[-1].split("\n## Figure Legends")[0]}

for name, body in SECS.items():
    fr = {t for t in TOK if t in body}
    for pt in PATS:
        try:
            for m in re.finditer(pt, body): fr.add(m.group(0))
        except re.error: pass
    out = sorted(fr, key=len, reverse=True)
    io.open(f"work_notes/frozen_{name}.txt", "w", encoding="utf8").write("\n".join(out))
    print(f"{name:<11} 본문 {len(body.split()):>5}단어 · 동결 {len(out):>3}개 / {sum(len(x.split()) for x in out):>4}단어 "
          f"({100*sum(len(x.split()) for x in out)/max(1,len(body.split())):.0f}%)")
