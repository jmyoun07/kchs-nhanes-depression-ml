# -*- coding: utf-8 -*-
"""A6_apply_language.py — 최종 문법·문체 검사에서 편집자 검증을 통과한 수정을 적용한다."""
import io, os, re, sys, json
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__))
SRC = r"C:/Users/rudtn/AppData/Local/Temp/claude/C--Users-rudtn-Desktop-claude-code-------/dffa918b-2ff7-4931-9dc0-4af59c5ddb26/tasks/wdg8mtgmr.output"
UP = json.loads(io.open(SRC, encoding="utf-8").read())["result"]["accepted"]

BAN = re.compile(r"\b(we|We|our|Our|ours|they|They|their|Their|them|Them|themselves)\b"
                 r"|(?<![A-Za-z0-9])us(?![A-Za-z])|\bbecause\b|(?<![A-Za-z])so (?!that\b)"
                 r"|[\u2013\u2014\u2212]|i\.e\.|e\.g\.|behavioural|\bP\s*[<=>]\s*0\.")

# 파일별로 모아서 적용 (같은 파일 안에서 순서대로)
byfile = {}
for u in UP:
    f = (u.get("file") or "").replace("\\", "/").split("/17var_model/")[-1]
    if not f.startswith("docs/"): continue
    byfile.setdefault(f, []).append(u)

tot_ok = tot_miss = tot_bad = 0
for f, ups in byfile.items():
    p = f"{D}/{f}"
    if not os.path.exists(p):
        print(f"  [없음] {f}"); continue
    s = io.open(p, encoding="utf8").read()
    ok = miss = bad = 0
    for u in ups:
        old, new = u.get("old_text", ""), u.get("new_text", "")
        if not old: continue
        if old not in s: miss += 1; continue
        if new and [m.group(0) for m in BAN.finditer(new)]: bad += 1; continue
        s = s.replace(old, new, 1); ok += 1
    io.open(p, "w", encoding="utf8").write(s)
    print(f"  {os.path.basename(f):<28} 적용 {ok:>3} · 미일치 {miss:>2} · 규칙보류 {bad}")
    tot_ok += ok; tot_miss += miss; tot_bad += bad

print(f"\n합계 적용 {tot_ok} · 미일치 {tot_miss} · 규칙보류 {tot_bad}")
M = io.open(f"{D}/docs/manuscript_17var.md", encoding="utf8").read()
ab = M[M.index("## Abstract"):M.index("## Introduction")]
print("초록 단어수:", len(re.sub(r"[*#\[\]]", "", ab).split()) - 1)
b = M[M.index("## Introduction"):M.index("## Acknowledgments")]
L = [l for l in b.split("\n") if not l.strip().startswith("|") and not l.strip().startswith("**Table")]
t = re.sub(r"\*\*Figure legends\.\*\*.*", "", "\n".join(L), flags=re.S)
print("서술 본문:", len(re.sub(r"[*#\[\]]", "", t).split()), "단어")
