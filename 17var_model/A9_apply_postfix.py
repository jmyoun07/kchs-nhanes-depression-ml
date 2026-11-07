# -*- coding: utf-8 -*-
"""A9_apply_postfix.py — 사후 감사 31건 적용."""
import io, os, re, sys, json
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__))
SRC = r"C:/Users/rudtn/AppData/Local/Temp/claude/C--Users-rudtn-Desktop-claude-code-------/dffa918b-2ff7-4931-9dc0-4af59c5ddb26/tasks/wo3ylehcq.output"
UP = json.loads(io.open(SRC, encoding="utf-8").read())["result"]["accepted"]

BAN = re.compile(r"\b(we|We|our|Our|ours|they|They|their|Their|them|Them|themselves)\b"
                 r"|(?<![A-Za-z0-9])us(?![A-Za-z])|\bbecause\b|(?<![A-Za-z])so (?!that\b)"
                 r"|[\u2013\u2014\u2212]|i\.e\.|e\.g\.|behavioural|\bP\s*[<=>]\s*0\.")
def dec(t):
    for a, b in (("&lt;", "<"), ("&gt;", ">"), ("&amp;", "&"), ("&#39;", "'"), ("&quot;", '"')):
        t = t.replace(a, b)
    return t

byfile = {}
for u in UP:
    byfile.setdefault((u.get("file") or "").replace("\\", "/").split("/")[-1], []).append(u)

tot = miss = bad = 0
for f, ups in byfile.items():
    p = f"{D}/docs/{f}"
    if not os.path.exists(p):
        print(f"  [없음] {f}"); continue
    s = io.open(p, encoding="utf8").read()
    ok = m0 = b0 = 0
    for u in ups:
        old, new = dec(u.get("old_text", "")), dec(u.get("new_text", ""))
        if not old: continue
        if old not in s:
            m0 += 1; print(f"    [미일치] {old[:88]}"); continue
        if new and [x.group(0) for x in BAN.finditer(new)]:
            b0 += 1; print(f"    [규칙보류] {[x.group(0) for x in BAN.finditer(new)][:3]} ← {old[:60]}"); continue
        s = s.replace(old, new, 1); ok += 1
    io.open(p, "w", encoding="utf8").write(s)
    print(f"  {f:<28} 적용 {ok:>2} · 미일치 {m0} · 규칙보류 {b0}")
    tot += ok; miss += m0; bad += b0
print(f"\n합계 적용 {tot} · 미일치 {miss} · 규칙보류 {bad}")

# 남은 굵은 라벨·연속 빈줄 정리
for f in ("manuscript_17var.md", "appendix3_utility.md", "cover_letter.md"):
    p = f"{D}/docs/{f}"; s = io.open(p, encoding="utf8").read(); o = s
    s = s.replace("## Interpretation\n\n**Interpretation.** ", "## Interpretation\n\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    if s != o: io.open(p, "w", encoding="utf8").write(s); print(f"  {f}: 잔재 정리")
