# -*- coding: utf-8 -*-
"""B1_loop_apply.py — 수렴 루프의 라운드 결과를 적용한다.

사용: python B1_loop_apply.py <감사결과 json 경로>
문서(docs/*.md)와 스크립트(*.py) 양쪽을 대상으로 한다. 생성기 수정이 빠지면
문서만 고쳐 놓고 다음 실행에서 되돌아가므로 반드시 함께 적용한다.
"""
import io, os, re, sys, json
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1]
UP = json.loads(io.open(SRC, encoding="utf-8").read())["result"]["confirmed"]

BAN = re.compile(r"\b(we|We|our|Our|ours|they|They|their|Their|them|Them|themselves)\b"
                 r"|(?<![A-Za-z0-9])us(?![A-Za-z])|\bbecause\b|(?<![A-Za-z])so (?!that\b)"
                 r"|[–—−]|i\.e\.|e\.g\.|behavioural")
def dec(t):
    for a, b in (("&lt;", "<"), ("&gt;", ">"), ("&amp;", "&"), ("&#39;", "'"), ("&quot;", '"')):
        t = t.replace(a, b)
    return t

def resolve(f):
    f = (f or "").replace("\\", "/").split("/")[-1]
    if not f or not f.endswith((".md", ".py")): return None
    for cand in (f"{D}/docs/{f}", f"{D}/{f}"):
        if os.path.exists(cand): return cand
    return None

byfile = {}
for u in UP:
    byfile.setdefault(u.get("file", ""), []).append(u)

tot = miss = bad = 0
for f, ups in byfile.items():
    p = resolve(f)
    if not p:
        print(f"  [없음] {f}"); continue
    s = io.open(p, encoding="utf8").read()
    ok = m0 = b0 = 0
    for u in ups:
        old, new = dec(u.get("old_text", "")), dec(u.get("new_text", ""))
        if not old: continue
        if old not in s:
            m0 += 1; print(f"    [미일치] {os.path.basename(p)}: {old[:80]}"); continue
        # 파이썬 소스에는 문체 규칙을 적용하지 않는다(주석·정규식이 걸린다)
        if p.endswith(".md") and new and [x.group(0) for x in BAN.finditer(new)]:
            b0 += 1; print(f"    [규칙보류] {[x.group(0) for x in BAN.finditer(new)][:3]}"); continue
        s = s.replace(old, new, 1); ok += 1
    io.open(p, "w", encoding="utf8").write(s)
    if p.endswith(".py"):
        try: compile(io.open(p, encoding="utf8").read(), p, "exec")
        except SyntaxError as e:
            print(f"    [문법오류] {os.path.basename(p)}: {e}"); raise
    print(f"  {os.path.basename(p):<28} 적용 {ok:>2} · 미일치 {m0} · 규칙보류 {b0}")
    tot += ok; miss += m0; bad += b0
print(f"\n합계 적용 {tot} · 미일치 {miss} · 규칙보류 {bad}")
