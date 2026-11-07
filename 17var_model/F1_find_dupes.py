# -*- coding: utf-8 -*-
"""원고의 중복 문장과 떠 있는 줄을 기계적으로 찾는다.

내가 빠진 수치를 되살릴 때 유사도로 자리를 찾았고, 축소본이 같은 내용을 다른 문장으로
이미 갖고 있으면 '없음' 으로 판정해 새로 끼워 넣었다. 그 결과가 중복과 조각이다.
같은 실수를 반복하지 않도록, 고치기 전에 전부 찾아 놓는다.
"""
import io, os, re, sys
from difflib import SequenceMatcher
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")

L = io.open("docs/manuscript_17var.md", encoding="utf8").read().split("\n")

print("=== 1) 선두 공백으로 떠 있는 줄 (문단에 안 붙은 삽입) ===")
for i, t in enumerate(L, 1):
    if t[:1] == " " and t.strip() and not t.startswith("|"):
        nxt = L[i].strip() if i < len(L) else ""
        print(f"  {i:>4}  다음줄={'제목' if nxt.startswith('#') else ('빈줄' if not nxt else '본문')}  {t.strip()[:110]}")

print("\n=== 2) 중복 문장 (본문 전체) ===")
SENT = re.compile(r"(?<=[.;])\s+(?=[A-Z(])")
sents = []
for i, t in enumerate(L, 1):
    if not t.strip() or t.startswith(("|", "#")): continue
    for s in SENT.split(t):
        s = s.strip()
        if len(s.split()) >= 8: sents.append((i, s))
seen = {}
for i, s in sents:
    key = " ".join(re.sub(r"[^\w\s]", "", s.lower()).split())[:90]
    seen.setdefault(key, []).append((i, s))
for key, group in seen.items():
    if len(group) > 1:
        print(f"  줄 {[g[0] for g in group]}")
        for _, s in group: print(f"      {s[:130]}")

print("\n=== 3) 거의 같은 문장 (유사도 0.75 이상) ===")
n = 0
for a in range(len(sents)):
    for b in range(a + 1, len(sents)):
        if sents[a][0] == sents[b][0]: continue
        r = SequenceMatcher(None, sents[a][1].lower(), sents[b][1].lower()).ratio()
        if r >= 0.75:
            n += 1
            print(f"  줄 {sents[a][0]} vs {sents[b][0]}  유사도 {r:.2f}")
            print(f"      A: {sents[a][1][:125]}")
            print(f"      B: {sents[b][1][:125]}")
            if n > 12: break
    if n > 12: break

print("\n=== 4) 문장 안에 남은 조각 (닫는 괄호 뒤 소수점 시작) ===")
for i, t in enumerate(L, 1):
    for m in re.finditer(r"\)\.?\d", t):
        print(f"  {i:>4}  ...{t[max(0,m.start()-70):m.start()+70]}...")
