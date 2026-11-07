# -*- coding: utf-8 -*-
"""부록 표에 약어 각주를 채우고 캡션에 마침표를 붙인다.

예시파일은 표마다 그 표에서 쓴 약어를 각주에 정의한다(KYRBS·YRBSS 도 표마다 반복).
우리 부록에는 그게 없어서, 표만 떼어 보면 약어를 알 수 없었다.

각주 붙이는 방식
  · 이미 총괄 각주가 있는 표  -> 약어마다 한 줄씩 덧붙인다(b, c, d ... 개별 알파벳)
  · 각주가 하나도 없는 표      -> 한 줄에 세미콜론으로 모아 쓴다(예시파일과 동일)
    (첫 각주는 무조건 'a' 가 되어 머리글 첫 칸에 붙으므로, 그 자리에 약어 정의를 몰아 둔다)
"""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")

DEF = {
    "AUROC": "area under the receiver operating characteristic curve",
    "AUPRC": "area under the precision-recall curve",
    "CITL": "calibration-in-the-large",
    "ECE": "expected calibration error",
    "SHAP": "Shapley additive explanations",
    "PPV": "positive predictive value",
    "NPV": "negative predictive value",
    "NNS": "number needed to screen",
    "FPR": "false positive rate",
    "FNR": "false negative rate",
    "BMI": "body mass index",
    "PHQ-9": "9-item Patient Health Questionnaire",
    "KCHS": "Korea Community Health Survey",
    "NHANES": "National Health and Nutrition Examination Survey",
    "MEC": "mobile examination center",
    "DPQ": "Depression Screener",
    "LightGBM": "light gradient-boosting machine",
    "PROBAST": "Prediction model Risk Of Bias Assessment Tool",
}
TOK = re.compile(r"\b(" + "|".join(re.escape(k) for k in sorted(DEF, key=len, reverse=True)) + r")\b")

FILES = ["docs/appendix2_crosswalk.md", "docs/appendix3_utility.md",
         "docs/appendix4_ceiling.md", "docs/appendix5_methods.md"]

n_cap = n_note = 0
for f in FILES:
    L = io.open(f, encoding="utf8").read().split("\n")
    out, i = [], 0
    while i < len(L):
        m = re.match(r"^(#{2,4}\s+Table S\d+)\.?\s*(.*)$", L[i])
        if not m:
            out.append(L[i]); i += 1; continue
        head, cap = m.group(1), m.group(2).rstrip()
        if cap and not cap.endswith("."): cap += "."; n_cap += 1
        out.append(f"{head}. {cap}".rstrip())
        i += 1
        # 표까지 그대로 옮긴다
        while i < len(L) and not L[i].startswith("|"): out.append(L[i]); i += 1
        rows = []
        while i < len(L) and L[i].startswith("|"): rows.append(L[i]); out.append(L[i]); i += 1
        # 기존 각주 블록
        blanks = []
        while i < len(L) and not L[i].strip(): blanks.append(L[i]); i += 1
        notes = []
        while i < len(L) and L[i].strip() and not L[i].startswith(("|", "#")):
            notes.append(L[i]); i += 1

        used, seen = [], set()
        for r in rows:
            for t in TOK.findall(r):
                if t not in seen: seen.add(t); used.append(t)
        # 캡션에서 이미 풀어 쓴 것과 이미 정의된 것은 뺀다
        already = {mm.group(1) for nt in notes for mm in [re.match(r"^([A-Za-z][A-Za-z0-9+\-]{1,20}):", nt)] if mm}
        need = [a for a in used if a not in already and f"({a})" not in cap]

        if need:
            if notes:                       # 총괄 각주가 있으면 약어마다 한 줄씩
                notes += [f"{a}: {DEF[a]}." for a in need]
            else:                           # 없으면 한 줄에 모아 쓴다(예시파일 방식)
                notes = ["; ".join(f"{a}: {DEF[a]}" for a in need) + "."]
            n_note += len(need)
        out += blanks + notes
    io.open(f, "w", encoding="utf8").write("\n".join(out))

print(f"캡션 마침표 {n_cap}건 · 약어 각주 {n_note}건 추가")
