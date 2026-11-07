# -*- coding: utf-8 -*-
"""A8_apply_trim_audit.py — 압축 감사에서 편집자 검증을 통과한 12건을 적용한다."""
import io, os, re, sys, json
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__))
SRC = r"C:/Users/rudtn/AppData/Local/Temp/claude/C--Users-rudtn-Desktop-claude-code-------/dffa918b-2ff7-4931-9dc0-4af59c5ddb26/tasks/w801qj66s.output"
UP = json.loads(io.open(SRC, encoding="utf-8").read())["result"]["accepted"]

BAN = re.compile(r"\b(we|We|our|Our|ours|they|They|their|Their|them|Them|themselves)\b"
                 r"|(?<![A-Za-z0-9])us(?![A-Za-z])|\bbecause\b|(?<![A-Za-z])so (?!that\b)"
                 r"|[\u2013\u2014\u2212]|i\.e\.|e\.g\.|behavioural|\bP\s*[<=>]\s*0\.")

byfile = {}
for u in UP:
    f = (u.get("file") or "").replace("\\", "/").split("/")[-1]
    byfile.setdefault(f, []).append(u)

tot = miss = bad = 0
for f, ups in byfile.items():
    p = f"{D}/docs/{f}"
    if not os.path.exists(p):
        print(f"  [없음] {f}"); continue
    s = io.open(p, encoding="utf8").read()
    ok = m0 = b0 = 0
    for u in ups:
        old, new = u.get("old_text", ""), u.get("new_text", "")
        # 감사 출력은 HTML 엔티티로 인코딩된 부등호를 쓴다
        for a, b in (("&lt;", "<"), ("&gt;", ">"), ("&amp;", "&")):
            old = old.replace(a, b); new = new.replace(a, b)
        if not old: continue
        if old not in s: m0 += 1; print(f"    [미일치] {old[:80]}"); continue
        if new and [x.group(0) for x in BAN.finditer(new)]: b0 += 1; continue
        s = s.replace(old, new, 1); ok += 1
    io.open(p, "w", encoding="utf8").write(s)
    print(f"  {f:<28} 적용 {ok} · 미일치 {m0} · 규칙보류 {b0}")
    tot += ok; miss += m0; bad += b0
print(f"\n합계 적용 {tot} · 미일치 {miss} · 규칙보류 {bad}")

# 반복된 굵은 라벨 잔재 정리
p = f"{D}/docs/appendix3_utility.md"; s = io.open(p, encoding="utf8").read()
n = 0
while "**Interpretation.** **Interpretation.**" in s or "**Interpretation.**" in s:
    s2 = s.replace("**Interpretation.** ", "").replace("**Interpretation.**", "")
    if s2 == s: break
    s = s2; n += 1
if n: io.open(p, "w", encoding="utf8").write(s); print(f"  appendix3: 반복된 굵은 라벨 제거")

M = io.open(f"{D}/docs/manuscript_17var.md", encoding="utf8").read()
b = M[M.index("## Introduction"):M.index("## Acknowledgments")]
L = [l for l in b.split("\n") if not l.strip().startswith("|") and not l.strip().startswith("**Table")]
print("서술 본문:", len(re.sub(r"[*#\[\]]", "", re.sub(r"## Figure Legends.*", "", "\n".join(L), flags=re.S)).split()), "단어")
