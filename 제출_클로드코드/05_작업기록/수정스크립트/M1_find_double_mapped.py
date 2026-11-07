# -*- coding: utf-8 -*-
"""재번호 매핑이 두 번 이상 적용된 표 참조를 전수 탐지한다.

K2_renumber.py 의 결함: 범위·나열을 1단계에서 평문 새 번호로 써 넣었는데, 2단계의
낱개 치환이 그 결과를 다시 매핑했다. 낱개는 자리표시자로 보호돼 무사했지만
범위·나열은 전부 뚫렸다. 일부는 세 번까지 적용됐다.

방법: 재번호 전 백업에서 같은 자리의 참조를 찾아 매핑을 '한 번만' 적용한 값과 비교한다.
"""
import io, os, re, sys
from difflib import SequenceMatcher
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")

MAP = {1:4, 2:15, 3:5, 4:8, 5:6, 6:13, 7:14, 8:1, 9:2, 10:7, 11:11, 12:9, 13:12, 14:16, 15:10, 16:3}
FILES = ["manuscript_17var.md", "TRIPOD_AI_checklist.md", "model_card_17var.md", "cover_letter.md",
         "appendix2_crosswalk.md", "appendix3_utility.md", "appendix4_ceiling.md", "appendix5_methods.md"]
BAK = "work_notes/pre_renum2"

# 여러 표를 한꺼번에 가리키는 표현
MULTI = re.compile(r"Tables? S\d+(?:\s*(?:to|-|,|and|;)\s*S?\d+)+")

def nums(t): return [int(x) for x in re.findall(r"S(\d+)", t)]
def expand(t):
    """'Tables S11 to S15' 를 낱개 목록으로."""
    ns = nums(t)
    if re.search(r"\bto\b|-", t) and len(ns) == 2 and ns[0] < ns[1]:
        return list(range(ns[0], ns[1] + 1))
    return ns

rows = []
for f in FILES:
    cur = io.open(f"docs/{f}", encoding="utf8").read()
    bak_p = f"{BAK}/{f}"
    if not os.path.exists(bak_p): continue
    bak = io.open(bak_p, encoding="utf8").read()
    bak_multi = [(m.start(), m.group(0), bak[max(0, m.start()-70):m.start()]) for m in MULTI.finditer(bak)]
    for m in MULTI.finditer(cur):
        got = expand(m.group(0))
        ctx = cur[max(0, m.start()-70):m.start()]
        # 백업에서 문맥이 가장 닮은 것을 찾는다
        best, sc = None, 0.0
        for _, txt, bctx in bak_multi:
            r = SequenceMatcher(None, ctx.lower(), bctx.lower()).ratio()
            if r > sc: best, sc = txt, r
        if not best or sc < 0.55: continue
        want = [MAP[x] for x in expand(best)]
        if sorted(got) != sorted(want):
            rows.append((f, m.group(0), best, want, sorted(got), sc))

print(f"{'파일':<26}{'지금':<34}{'백업(재번호 전)':<26}{'옳은 값'}")
for f, cur_t, bak_t, want, got, sc in rows:
    print(f"{f:<26}{cur_t[:33]:<34}{bak_t[:25]:<26}{sorted(want)}")
print(f"\n중복 적용된 참조 {len(rows)}곳")
io.open("work_notes/double_mapped.txt", "w", encoding="utf8").write(
    "\n".join(f"{f}\t{c}\t{b}\t{sorted(w)}" for f, c, b, w, g, s in rows))
