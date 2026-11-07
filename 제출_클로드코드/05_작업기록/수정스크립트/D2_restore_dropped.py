# -*- coding: utf-8 -*-
"""최종 축소가 지운 검사 대상 문장을 되살린다.

동결 목록을 in_doc 토큰으로만 만들어서, 직접 need(...) 로 검사하는 두 항목이 빠졌다.
  · 공정성 플래그 하위군 (편차값이 원고에 있어야 한다)
  · SHAP 상위 5개 값
둘 다 이전본에서 그대로 되살린다. 그리고 부록에 새로 생긴 결과의 so 를 고친다.
"""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")

B = io.open("work_notes/manuscript_pre_final.bak.md", encoding="utf8").read()
p = "docs/manuscript_17var.md"
s = io.open(p, encoding="utf8").read()

def grab(start, end):
    i = B.index(start); j = B.index(end, i) + len(end)
    return " ".join(B[i:j].split())

SHAP = grab("Global SHAP importance, computed in the development fitting partition",
            "Food insecurity contributed modestly (0.06, ranked 10th).")
FAIR = grab("Applying the fairness flag set in the analysis plan",
            "non-Hispanic Asian in NHANES post-pandemic (+0.074).")

sec = "### Interpretability, Fairness, and Clinical Utility"
a = s.index(sec) + len(sec)
b = s.index("\n### ", a) if "\n### " in s[a:] else s.index("\n## ", a)
body = s[a:b]

# SHAP 순위: 절 첫 문단 앞에 놓는다(61_sync 가 이 문장에 값을 주입한다)
if "ranked the predictors:" not in body:
    body = "\n\n" + SHAP + body.lstrip("\n")
    print(f"SHAP 순위 문장 복원 ({len(SHAP.split())}단어)")

# 공정성 플래그: 하위군 나열 문단 뒤에 놓는다
if "flags the highest income quintile" not in body:
    m = re.search(r"non-Hispanic Asian 0\.\d{3} \([^)]*\)\.", body)
    if m:
        body = body[:m.end()] + " " + FAIR + body[m.end():]
    else:
        body = body.rstrip() + "\n\n" + FAIR + "\n"
    print(f"공정성 플래그 문장 복원 ({len(FAIR.split())}단어)")

s = s[:a] + body + s[b:]
io.open(p, "w", encoding="utf8").write(s)

# 부록에 새로 생긴 결과의 so
q = "docs/appendix3_utility.md"
t = io.open(q, encoding="utf8").read()
old = "Doing so changed the observed prevalence by no more than 0.1 percenta"
if old in t:
    t = t.replace("Doing so changed the observed prevalence", "That change moved the observed prevalence", 1)
    io.open(q, "w", encoding="utf8").write(t)
    print("appendix3: 결과의 'Doing so' 제거")
