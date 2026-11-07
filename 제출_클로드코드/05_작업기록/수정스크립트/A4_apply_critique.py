# -*- coding: utf-8 -*-
"""A4_apply_critique.py — 섹션별 리뷰에서 편집자 검증을 통과한 업그레이드를 적용한다.

적용 원칙:
  · old_text 가 파일에 정확히 있을 때만 바꾼다(없으면 건너뛰고 기록).
  · 초록은 450단어 상한이 있으므로 적용 후 단어수를 확인한다.
  · 대시류·금지 대명사·because·결과의 so 가 새로 들어오면 즉시 잡아 되돌린다.
"""
import io, os, re, sys, json, glob
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__))
SRC = r"C:/Users/rudtn/AppData/Local/Temp/claude/C--Users-rudtn-Desktop-claude-code-------/dffa918b-2ff7-4931-9dc0-4af59c5ddb26/tasks/wasuvjgcq.output"
UP = json.loads(io.open(SRC, encoding="utf-8").read())["result"]["accepted"]

BAN = re.compile(r"\b(we|We|our|Our|ours|they|They|their|Their|them|Them|themselves)\b"
                 r"|(?<![A-Za-z0-9])us(?![A-Za-z])|\bbecause\b|(?<![A-Za-z])so (?!that\b)"
                 r"|[\u2013\u2014\u2212]|i\.e\.|e\.g\.|behavioural|\bP\s*[<=>]\s*0\.")

TARGET = f"{D}/docs/manuscript_17var.md"
s = io.open(TARGET, encoding="utf8").read()
applied, missing, unsafe = 0, [], []
for u in UP:
    old, new = u.get("old_text", ""), u.get("new_text", "")
    # 리뷰는 대시 정규화 이전 원문 기준이라 같은 정규화를 걸어 맞춘다
    for _a, _b in (("—", "-"), ("–", "-"), ("−", "-")):
        old = old.replace(_a, _b); new = new.replace(_a, _b)
    if not old or not new: continue
    if old not in s:
        missing.append((u.get("section", "?"), u.get("importance", "?"), old[:90])); continue
    bad = [m.group(0) for m in BAN.finditer(new)]
    if bad:
        unsafe.append((u.get("section", "?"), old[:70], bad[:3])); continue
    s = s.replace(old, new, 1); applied += 1
io.open(TARGET, "w", encoding="utf8").write(s)

print(f"적용 {applied} · old_text 미일치 {len(missing)} · 규칙 위반으로 보류 {len(unsafe)}")
if unsafe:
    print("\n[규칙 위반 보류]")
    for sec, o, b in unsafe[:12]: print(f"  {sec}: {b} ← {o}")
if missing:
    print("\n[미일치 — 앞선 적용으로 원문이 바뀐 것들]")
    for sec, imp, o in missing[:12]: print(f"  {sec}/{imp}: {o}")

ab = s[s.index("## Abstract"):s.index("## Introduction")]
print(f"\n초록 단어수: {len(re.sub(r'[*#\[\]]','',ab).split())-1}")
