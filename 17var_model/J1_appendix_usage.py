# -*- coding: utf-8 -*-
"""부록의 각 덩어리가 실제로 쓰이는지 전수 조사한다.

기준
  · 표      - 원고 본문에서 인용되는가, 어느 자리에서 처음 인용되는가
  · 산문 절 - 원고나 TRIPOD 체크리스트가 그 내용을 가리키는가
쓰이지 않는 덩어리는 부록에 있을 이유가 없다.
"""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")
D = "docs"
rd = lambda f: io.open(f"{D}/{f}", encoding="utf8").read()
M = rd("manuscript_17var.md")
BODY = M.split("## Figure Legends")[0]
TRIPOD = rd("TRIPOD_AI_checklist.md")
APPX = ["appendix2_crosswalk.md", "appendix5_methods.md", "appendix3_utility.md", "appendix4_ceiling.md"]

# ── 표: 본문 최초 인용 순서 ─────────────────────────────────────────────
order, seen = [], set()
for m in re.finditer(r"Table S(\d+)", BODY):
    n = int(m.group(1))
    if n not in seen: seen.add(n); order.append(n)
CAP = {}
for f in APPX:
    for m in re.finditer(r"(?m)^#{2,4}\s+Table S(\d+)\.\s*(.*)$", rd(f)):
        CAP[int(m.group(1))] = (f, m.group(2)[:62])

print("=== 표: 본문 등장 순서 대 현재 번호 ===")
print(f"{'새번호':>6}{'현재':>6}  {'파일':<24}캡션")
for i, n in enumerate(order, 1):
    f, c = CAP.get(n, ("?", "?"))
    mark = "" if i == n else "  ← 바뀜"
    print(f"{i:>6}{('S'+str(n)):>6}  {f.replace('appendix','').replace('.md',''):<24}{c}{mark}")
missing = sorted(set(CAP) - set(order))
if missing: print(f"\n본문 미인용 표: {['S'+str(x) for x in missing]}")

# ── 산문 절: 참조 여부 ──────────────────────────────────────────────────
print("\n=== 부록 산문 절: 원고·체크리스트가 가리키는가 ===")
CORPUS = BODY + "\n" + TRIPOD
for f in APPX:
    s = rd(f)
    heads = [(m.start(), m.group(1), m.group(2).strip())
             for m in re.finditer(r"(?m)^(#{2,4})\s+(?!Table S)(.+?)\s*$", s)]
    print(f"\n[{f}]")
    for k, (pos, lvl, h) in enumerate(heads):
        end = heads[k + 1][0] if k + 1 < len(heads) else len(s)
        blk = s[pos:end]
        words = len(blk.split())
        tabs = sorted({int(x) for x in re.findall(r"Table S(\d+)", blk)})
        # 이 절의 제목이나 특징 문구가 원고에서 언급되는가
        key = re.sub(r"[^a-z ]", "", h.lower()).split()
        hit = any(w in CORPUS.lower() for w in key if len(w) > 7)
        print(f"  {lvl:<4} {h[:58]:<60} {words:>5}단어  표{tabs if tabs else '-'}  "
              f"{'제목어 본문에 있음' if hit else '← 본문에서 안 가리킴'}")
