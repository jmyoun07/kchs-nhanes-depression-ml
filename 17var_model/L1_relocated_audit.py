# -*- coding: utf-8 -*-
"""부록의 'relocated' 덩어리들을 살핀다.

축소 라운드마다 '원고에서 옮긴 상세' 블록이 쌓였다. 제목이 편집 과정을 그대로 드러내고
('relocated in revision'), 서로 중복될 가능성이 높다. 제출 전에 정리해야 한다.
"""
import io, os, re, sys
from difflib import SequenceMatcher
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")
D = "docs"
rd = lambda f: io.open(f"{D}/{f}", encoding="utf8").read()
FILES = ["appendix2_crosswalk.md", "appendix5_methods.md", "appendix3_utility.md", "appendix4_ceiling.md"]

blocks = []
for f in FILES:
    s = rd(f)
    heads = [(m.start(), m.group(0).strip()) for m in re.finditer(r"(?m)^#{2,4}\s+.+$", s)]
    for k, (pos, h) in enumerate(heads):
        end = heads[k + 1][0] if k + 1 < len(heads) else len(s)
        blocks.append((f, h, s[pos:end]))

REL = [b for b in blocks if re.search(r"relocat|Additional detail|trimmed", b[1], re.I)]
print(f"=== 'relocated' 계열 {len(REL)}개 ===")
for f, h, b in REL:
    print(f"  [{f.replace('appendix','').replace('.md',''):<12}] {len(b.split()):>4}단어  {h[:78]}")

print("\n=== 문장 단위 중복 (블록 사이) ===")
def sents(t):
    t = re.sub(r"(?m)^#{2,4}.*$", "", t)
    return [x.strip() for x in re.split(r"(?<=[.;])\s+(?=[A-Z(])", t) if len(x.split()) >= 10]
S = [(i, s) for i, (f, h, b) in enumerate(REL) for s in sents(b)]
dupe = 0
for a in range(len(S)):
    for c in range(a + 1, len(S)):
        if S[a][0] == S[c][0]: continue
        r = SequenceMatcher(None, S[a][1].lower(), S[c][1].lower()).ratio()
        if r >= 0.80:
            dupe += 1
            print(f"  블록{S[a][0]+1} vs 블록{S[c][0]+1}  유사도 {r:.2f}")
            print(f"     A: {S[a][1][:118]}")
            print(f"     B: {S[c][1][:118]}")
            if dupe >= 10: break
    if dupe >= 10: break
if not dupe: print("  (블록 사이 중복 없음)")

print("\n=== 원고 본문과의 중복 ===")
BODY = rd("manuscript_17var.md").split("## Figure Legends")[0]
bs = sents(BODY)
hit = 0
for i, s in S:
    for t in bs:
        if SequenceMatcher(None, s.lower(), t.lower()).ratio() >= 0.82:
            hit += 1
            print(f"  블록{i+1} 문장이 본문과 거의 같음")
            print(f"     부록: {s[:110]}")
            print(f"     본문: {t[:110]}")
            break
    if hit >= 6: break
if not hit: print("  (본문과의 중복 없음)")
