# -*- coding: utf-8 -*-
"""Discussion 축소본에 감사 지적 9건을 반영해 적용한다.

지적은 전부 실질이다. 인용번호가 든 문장을 다시 쓰면서 내용이 함께 빠졌다.
감사자가 준 정정 문장으로 되돌린 뒤 원고에 넣는다.
"""
import io, json, os, re, sys
from difflib import SequenceMatcher
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")

J = json.loads(io.open("work_notes/trim_rd.json", encoding="utf8").read())
D = [s for s in J["sections"] if s["key"] == "discussion"][0]

def sentences(t):
    return [x for x in re.split(r"(?<=[.;])\s+(?=[A-Z(])", t) if x.strip()]

def apply_fix(text, fix):
    """정정 문장과 가장 닮은 문장을 찾아 갈아끼운다."""
    if fix.strip() in text: return text, 1.0, "이미 있음"
    best, score = None, 0.0
    for s in sentences(text):
        r = SequenceMatcher(None, s.lower(), fix.lower()).ratio()
        if r > score: best, score = s, r
    if best and score >= 0.35:
        return text.replace(best, fix.strip(), 1), score, "치환"
    return text, score, "미적용"

applied = miss = 0
for i, pr in enumerate(D["problems"], 1):
    fix = pr["fix"].strip()
    if not fix.endswith((".", ";")): fix += "."          # 잘려 온 정정 문장 마무리
    done = False
    for sub in D["subsections"]:
        if pr["heading"].strip("# ").lower() not in sub["heading"].lower(): continue
        new, sc, how = apply_fix(sub["replacement"], fix)
        sub["replacement"] = new
        print(f"[{i}] {how} (유사도 {sc:.2f}) {sub['heading'][:44]}")
        done = True; applied += (how != "미적용"); miss += (how == "미적용")
        break
    if not done:                                          # 부록 블록 대상
        for a in D["appendix_additions"]:
            new, sc, how = apply_fix(a["markdown"], fix)
            a["markdown"] = new
            print(f"[{i}] {how} (유사도 {sc:.2f}) 부록 {a['target_file']}")
            applied += (how != "미적용"); miss += (how == "미적용")
            done = True; break
    if not done: print(f"[{i}] 대상 없음: {pr['heading']}"); miss += 1

print(f"\n적용 {applied} · 미적용 {miss}")

# Part D -> Part A 교정 (개정문서는 Part A 에 있다)
for a in D["appendix_additions"]:
    n = a["markdown"].count("Multimedia Appendix 1, Part D")
    if n:
        a["markdown"] = a["markdown"].replace("Multimedia Appendix 1, Part D", "Multimedia Appendix 1, Part A")
        print(f"부록 인용 Part D -> Part A: {n}건")
for sub in D["subsections"]:
    n = sub["replacement"].count("Multimedia Appendix 1, Part D")
    if n and "amendment" in sub["replacement"]:
        sub["replacement"] = sub["replacement"].replace(
            "amendment that added three of the seventeen predictors was written while the study was under way; it is dated and summarized in English in Multimedia Appendix 1, Part D",
            "amendment that added three of the seventeen predictors was written while the study was under way; it is dated and summarized in English in Multimedia Appendix 1, Part A")

io.open("work_notes/trim_rd_fixed.json", "w", encoding="utf8").write(
    json.dumps({"sections": [D]}, ensure_ascii=False, indent=1))
tot = sum(len(s["replacement"].split()) for s in D["subsections"])
print(f"\nDiscussion 최종 {tot}단어 (축소안 {D['word_count']}, 원본 2610)")
