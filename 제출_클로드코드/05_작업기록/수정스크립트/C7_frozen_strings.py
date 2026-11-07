# -*- coding: utf-8 -*-
"""검증기가 원고에 요구하는 '정확한 문자열' 목록을 뽑는다.

'숫자 든 문장은 전부 동결' 은 너무 넓은 규칙이었다. Results 는 거의 모든 문장에
숫자가 있어서 통째로 얼어붙었고, 3304 -> 3134 로 170단어밖에 못 줄였다.
실제로 깨지는 것은 60_verify 의 in_doc 토큰과 61_sync 의 정규식뿐이다. 그것만 동결한다.
"""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")

# ── 60_verify 의 in_doc 토큰을 기록하면서 실행 ────────────────────────────
src = io.open("60_verify_manuscript.py", encoding="utf8").read()
a = '    lab = label or token\n'
assert a in src, "in_doc 본문 앵커 없음"
src = src.replace(a, a + '    _REC.append((token, tuple(where)))\n', 1)
src = re.sub(r"(?m)^sys\.exit\(.*$", "pass", src)

g = {"__name__": "rec", "_REC": [], "__file__": os.path.abspath("60_verify_manuscript.py")}
try:
    exec(compile(src, "60_rec", "exec"), g)
except SystemExit:
    pass
REC = g["_REC"]

M = io.open("docs/manuscript_17var.md", encoding="utf8").read()
RES = M.split("\n## Results\n")[-1].split("\n## Discussion")[0]

man = [t for t, w in REC if "manuscript" in w]
in_res = sorted({t for t in man if t in RES}, key=len, reverse=True)

# 직접 need(... in M ...) 로 검사하는 하위군 CI 도 동결 대상이다
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
def R(v, nd=3): return f"{Decimal(str(v)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP):.{nd}f}"
fa = pd.read_csv("results/primary2_fairness.csv")
ci = [f"{R(r.CI_lo)}-{R(r.CI_hi)}" for _, r in fa.iterrows()
      if not any(t in str(r.subgroup) for t in ("Education", "Income quintile"))]
in_res += [c for c in ci if c in RES]

# ── 61_sync 가 원고에서 찾는 정규식 ──────────────────────────────────────
sy = io.open("61_sync_numbers.py", encoding="utf8").read()
blocks = re.findall(r'patch\("manuscript_17var\.md",\s*\[(.*?)\]\s*\)', sy, re.S)
pats = []
for b in blocks:
    pats += [m for m in re.findall(r'r"((?:[^"\\]|\\.)+)"', b)]
sync_hits = []
for pt in pats:
    try:
        m = re.search(pt, RES)
        if m: sync_hits.append(m.group(0))
    except re.error:
        pass

frozen = sorted(set(in_res) | set(sync_hits), key=len, reverse=True)
io.open("work_notes/frozen_results.txt", "w", encoding="utf8").write("\n".join(frozen))
print(f"in_doc 토큰 전체 {len(man)} · Results 안 {len(set(in_res))}")
print(f"61_sync 정규식 {len(pats)} · Results 에서 일치 {len(sync_hits)}")
print(f"동결 문자열 {len(frozen)}개 -> work_notes/frozen_results.txt")
print(f"Results 현재 {len(RES.split())}단어 · 동결분 합계 {sum(len(f.split()) for f in frozen)}단어")
print("\n샘플 8개:")
for f in frozen[:8]: print("  ", f[:110])
