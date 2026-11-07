# -*- coding: utf-8 -*-
"""76_refs_expand2.py — 참고문헌 2차 확장 (42 → 45편). 방식은 70_refs_expand.py 와 같다.

새로 넣는 3편은 코덱스가 지적한 서술을 실제로 받쳐 주는 문헌이다:
  · 보정의 수준(moderate calibration) 정의 — 우리가 보고하는 기울기·CITL 이 어느 수준인지 명시
  · 순편익에 신뢰구간을 붙이지 않는 이유 — 결정분석 양에 대한 추론의 한계
  · 같은 18문항 모듈로 한국·미국을 직접 비교한 식품불안정 연구 — 조화 주의사항의 근거
"""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__)); P = f"{D}/docs/manuscript_17var.md"

ADD = {
 "26772608": "Van Calster B, Nieboer D, Vergouwe Y, De Cock B, Pencina MJ, Steyerberg EW. A calibration hierarchy for risk models was defined: from utopia to empirical data. J Clin Epidemiol. Jan 06, 2016;74:167-76. [doi: 10.1016/j.jclinepi.2015.12.005] [Medline: 26772608]",
 "37277840": "Vickers AJ, Van Calster B, Wynants L, Steyerberg EW. Decision curve analysis: confidence intervals and hypothesis testing for net benefit. Diagn Progn Res. Jun 06, 2023;7(1):11. [doi: 10.1186/s41512-023-00148-y] [Medline: 37277840]",
 "33435492": "Lee SJ, Lee KW, Cho MS. Association of Food Insecurity with Nutrient Intake and Depression among Korean and US Adults: Data from the 2014 Korea and the 2013-2014 US National Health and Nutrition Examination Surveys. Int J Environ Res Public Health. Jan 09, 2021;18(2):506. [doi: 10.3390/ijerph18020506] [Medline: 33435492]",
}
INS = [
 ("Calibration was summarized by the calibration slope and by calibration-in-the-large (CITL)",
  "Calibration was assessed at the moderate level of the calibration hierarchy [@26772608] and summarized by the calibration slope and by calibration-in-the-large (CITL)"),
 ("net benefit were implemented directly rather than through a survey-analysis package, and the implementations are released with the code.",
  "net benefit were implemented directly rather than through a survey-analysis package, and the implementations are released with the code. Intervals are not reported for net benefit, since inference on a decision-analytic quantity answers a different question from the decision itself [@37277840]."),
 ("**Harmonization cautions.** Alcohol, physical activity, and income use different instruments",
  "**Harmonization cautions.** A direct Korea-US comparison using the same 18-item household module found food insecurity associated with higher odds of PHQ-9 depression in both countries, with a larger association in Korea [@33435492]; the single-item KCHS measure used here is therefore the weakest link in the harmonized set rather than a difference in the underlying construct. Alcohol, physical activity, and income use different instruments"),
]

s = io.open(P, encoding="utf8").read()
head, refsec = s.split("\n## References", 1)
body_refs, tail = refsec.split("\n## Abbreviations", 1)
cur = {}
for m in re.finditer(r"^(\d+)\.\s+(.*)$", body_refs, re.M):
    n, txt = int(m.group(1)), m.group(2)
    pm = re.search(r"\[Medline: (\d+)\]", txt)
    cur[n] = (pm.group(1) if pm else ("SHAP" if "papers.nips.cc" in txt else "LUMLEY"), txt)
REFS = {k: t for k, t in cur.values()}; REFS.update(ADD)

def to_key(m):
    ns = [int(x) for x in re.findall(r"\d+", m.group(1))]
    if any(n not in cur for n in ns): return m.group(0)
    return "[" + ",".join("@" + cur[n][0] for n in ns) + "]"
head = re.sub(r"\[(\d+(?:[,\-]\d+)*)\]", to_key, head)
for a, b in INS:
    assert a in head, f"[앵커 없음] {a[:80]}"
    head = head.replace(a, b, 1)

order, seen = [], set()
for m in re.finditer(r"@([A-Z0-9]+)", head):
    k = m.group(1)
    if k not in seen: seen.add(k); order.append(k)
missing = [k for k in REFS if k not in seen]
assert not missing, f"인용되지 않은 문헌: {missing}"
num = {k: i + 1 for i, k in enumerate(order)}
def fmt(m):
    ns = sorted(num[k] for k in re.findall(r"@([A-Z0-9]+)", m.group(0)))
    out, i = [], 0
    while i < len(ns):
        j = i
        while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1: j += 1
        out.append(f"{ns[i]}-{ns[j]}" if j - i >= 2 else ",".join(str(x) for x in ns[i:j + 1]))
        i = j + 1
    return "[" + ",".join(out) + "]"
head = re.sub(r"\[@[A-Z0-9]+(?:,@[A-Z0-9]+)*\]", fmt, head)
io.open(P, "w", encoding="utf8").write(head + "\n## References\n\n"
    + "\n".join(f"{num[k]}. {REFS[k]}" for k in order) + "\n\n## Abbreviations" + tail)
print(f"참고문헌 {len(cur)} → {len(order)}편 · 신규 인용 {len(INS)}지점")
