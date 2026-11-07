# -*- coding: utf-8 -*-
"""부록을 원고와 같은 기준으로 훑는다.

기준: 캡션 한 문장 · 각주는 약어 하나에 알파벳 하나 · 표 안 약어는 각주에 정의 ·
      제목 구조 일관 · 긴 각주 없음
"""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")

FILES = [("Part A", "docs/appendix2_crosswalk.md"), ("Part B", "docs/appendix3_utility.md"),
         ("Part C", "docs/appendix4_ceiling.md"), ("Part D", "docs/appendix5_methods.md")]
ABBR = re.compile(r"\b(AUROC|AUPRC|CITL|ECE|SHAP|PPV|NPV|NNS|FPR|FNR|CI|SD|BMI|PHQ-9|KCHS|NHANES|"
                  r"MEC|DPQ|PSU|CV|LightGBM|PROBAST|TRIPOD\+AI|KDCA|NCHS|IQR|OR)\b")
issues = []

for pn, f in FILES:
    L = io.open(f, encoding="utf8").read().split("\n")
    i = 0
    while i < len(L):
        m = re.match(r"^#{2,4}\s+(Table S\d+)\.?\s*(.*)$", L[i])
        if not m: i += 1; continue
        tab, cap = m.group(1), m.group(2)
        # 표 본문
        j = i + 1
        while j < len(L) and not L[j].startswith("|"): j += 1
        k = j
        rows = []
        while k < len(L) and L[k].startswith("|"): rows.append(L[k]); k += 1
        # 각주
        n = k
        while n < len(L) and not L[n].strip(): n += 1
        notes = []
        while n < len(L) and L[n].strip() and not L[n].startswith(("|", "#")):
            notes.append(L[n]); n += 1

        if not cap.endswith("."): issues.append((pn, tab, "캡션 끝에 마침표 없음", cap[:52]))
        if len(cap.split()) > 30: issues.append((pn, tab, f"캡션 {len(cap.split())}단어", cap[:52]))
        if notes:
            w0 = len(notes[0].split())
            if w0 > 110: issues.append((pn, tab, f"총괄 각주 {w0}단어", notes[0][:52]))
            for nt in notes[1:]:
                if not re.match(r"^[A-Za-z][A-Za-z0-9+\- ]{1,20}: ", nt):
                    issues.append((pn, tab, "각주 형식이 'ABBR: 풀이.' 아님", nt[:52]))
        # 표 안에 쓰인 약어가 각주에 정의돼 있는지
        used = set()
        for r in rows[:2] + rows[2:]:
            used |= set(ABBR.findall(r))
        defined = set()
        for nt in notes:
            mm = re.match(r"^([A-Za-z][A-Za-z0-9+\-]{1,20}):", nt)
            if mm: defined.add(mm.group(1))
        # 캡션에서 풀어 쓴 것은 정의된 것으로 본다
        for a in list(used):
            if f"({a})" in cap: defined.add(a)
        miss = sorted(used - defined - {"CI", "CV", "SD", "OR", "IQR"})
        if miss: issues.append((pn, tab, "각주에 정의 없는 약어", ", ".join(miss)))
        i = k

print(f"{'Part':<8}{'표':<10}문제")
for pn, tab, kind, det in issues:
    print(f"{pn:<8}{tab:<10}{kind}  |  {det}")
print(f"\n총 {len(issues)}건")
