# -*- coding: utf-8 -*-
"""A3_dash_normalize.py — 대시 표기를 예시논문과 동일하게 맞춘다.

예시논문(윤수 예시파일/JMIR ...docx) 실측:
  em dash(—) 0회 · en dash(–) 0회 · minus(−) 0회 · 하이픈만 사용
  숫자 범위: 0.072-0.104, 2020-2024   음수: -0.104   표의 빈칸: 공백(— 아님)

따라서 문서와 **생성기 양쪽**에서 —, –, − 를 하이픈으로 바꾸고,
표의 '해당 없음' 칸(— 단독)은 빈칸으로 만든다. 생성기를 함께 고치지 않으면 되돌아온다.
"""
import io, os, re, sys, glob
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__))

DASHES = {"—": "-", "–": "-", "−": "-"}

def norm(text, table_blank=True):
    if table_blank:
        # 표의 단독 — 칸을 빈칸으로 (예시논문은 빈칸을 쓴다)
        prev = None
        while prev != text:
            prev = text
            text = re.sub(r"\|\s*—\s*(?=\|)", "|  ", text)
    for a, b in DASHES.items():
        text = text.replace(a, b)
    return text

print("[1] 문서")
for f in sorted(glob.glob(f"{D}/docs/*.md")):
    if "_audit" in f: continue
    s = io.open(f, encoding="utf8").read()
    before = {k: s.count(k) for k in DASHES}
    s2 = norm(s)
    if s2 != s:
        io.open(f, "w", encoding="utf8").write(s2)
        print(f"  {os.path.basename(f)}: {before}")

print("[2] 생성기·검증기")
for f in ["61_sync_numbers.py", "60_verify_manuscript.py", "44_make_docx.py",
          "45_make_appendix_docx.py", "46_build_submission.py", "87_make_ledger.py",
          "92_crossref_audit.py", "73_figures_primary2.py"]:
    p = f"{D}/{f}"
    if not os.path.exists(p): continue
    s = io.open(p, encoding="utf8").read()
    before = {k: s.count(k) for k in DASHES if s.count(k)}
    if not before: continue
    s2 = norm(s, table_blank=False)
    io.open(p, "w", encoding="utf8").write(s2)
    compile(io.open(p, encoding="utf8").read(), p, "exec")
    print(f"  {f}: {before}")

print("\n[3] 확인")
tot = 0
for f in sorted(glob.glob(f"{D}/docs/*.md")) + [f"{D}/61_sync_numbers.py", f"{D}/60_verify_manuscript.py"]:
    if "_audit" in f: continue
    s = io.open(f, encoding="utf8").read()
    c = sum(s.count(k) for k in DASHES)
    tot += c
    if c: print(f"  잔존 {os.path.basename(f)}: {c}")
print(f"  총 잔존 {tot}곳")
