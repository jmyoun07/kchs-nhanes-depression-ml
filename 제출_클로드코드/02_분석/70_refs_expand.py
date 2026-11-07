# -*- coding: utf-8 -*-
"""70_refs_expand.py — 참고문헌 확장. 본문에 실제 근거 지점을 만들어 인용하고 등장순으로 재번호한다.

방식: 현재 번호 → PMID 로 되돌린 뒤(@키), 새 인용을 앵커에 심고, 첫 등장 순서로 다시 번호를 매긴다.
새로 넣는 문헌은 모두 PubMed 메타데이터로 확인했고, 채워 넣기가 아니라 본문 주장에 대응한다.
"""
import io, json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__)); P = f"{D}/docs/manuscript_17var.md"
ADD = json.load(io.open(r"C:/Users/rudtn/AppData/Local/Temp/claude/C--Users-rudtn-Desktop-claude-code-------/dffa918b-2ff7-4931-9dc0-4af59c5ddb26/scratchpad/refs_add.json", encoding="utf8"))
for drop in ["29851860", "40457490", "30144211"]:      # 정오표·주제 불일치·대체됨
    ADD.pop(drop, None)

s = io.open(P, encoding="utf8").read()
head, refsec = s.split("\n## References", 1)
body_refs, tail = refsec.split("\n## Abbreviations", 1)

# ── 1. 현재 목록 → {번호: (키, 서지문자열)} ──
cur = {}
for m in re.finditer(r"^(\d+)\.\s+(.*)$", body_refs, re.M):
    n, txt = int(m.group(1)), m.group(2)
    pm = re.search(r"\[Medline: (\d+)\]", txt)
    key = pm.group(1) if pm else ("SHAP" if "papers.nips.cc" in txt else "LUMLEY")
    cur[n] = (key, txt)
REFS = {k: t for k, t in cur.values()}
REFS.update(ADD)
print(f"1) 기존 {len(cur)}편 + 신규 {len(ADD)}편 = {len(REFS)}편")

# ── 2. 본문 번호 → @키 ──
def to_key(m):
    ns = [int(x) for x in re.findall(r"\d+", m.group(1))]
    if any(n not in cur for n in ns): return m.group(0)
    return "[" + ",".join("@" + cur[n][0] for n in ns) + "]"
head = re.sub(r"\[(\d+(?:[,\-]\d+)*)\]", to_key, head)

# ── 3. 신규 인용 삽입 (근거 지점) ──
INS = [
 ("yet applying a fixed cut-off uniformly across heterogeneous populations does not prioritize those at highest risk",
  "yet applying a fixed cut-off uniformly across heterogeneous populations does not prioritize those at highest risk, "
  "even where universal screening of adults is recommended [@37338872]"),
 ("The COVID-19 pandemic constitutes a natural distribution shift for mental-health prediction [@34634250,@34260843].",
  "The COVID-19 pandemic constitutes a natural distribution shift for mental-health prediction [@34634250,@34260843]. "
  "In Korea, national surveillance recorded a rise in depressive symptoms and suicide attempts over the pandemic years [@37633158]."),
 ("**Data sources.** **Development — KCHS.** Nationwide Korea Disease Control and Prevention Agency (KDCA) survey",
  "**Data sources.** **Development — KCHS.** Nationwide Korea Disease Control and Prevention Agency (KDCA) survey [@26430619]"),
 ("**External validation — US NHANES.** NCHS survey combining household interview and mobile examination center assessment",
  "**External validation — US NHANES.** NCHS survey combining household interview and mobile examination center assessment [@33663649]"),
 ("the ≥10 cut-off is the conventional screening threshold and has been evaluated against semi-structured diagnostic interviews in individual-participant-data meta-analyses [@30967483,@31593971]",
  "the ≥10 cut-off is the conventional screening threshold and has been evaluated against semi-structured diagnostic interviews in "
  "individual-participant-data meta-analyses [@30967483,@31593971] and validated in a Korean population [@18243897]"),
 ("body mass index [@20194822], smoking [@27199385] and living alone [@36111305]",
  "body mass index [@20194822], smoking [@27199385], living alone [@36111305] and partnership status [@34085616]"),
 ("and compared at each algorithm's best configuration, so that no algorithm was advantaged by differential tuning effort.",
  "and compared at each algorithm's best configuration, so that no algorithm was advantaged by differential tuning effort, "
  "an equal-effort design motivated by evidence that machine learning shows no systematic advantage over regression once "
  "validation bias is controlled [@30763612]."),
 ("Following the plan, recalibration was applied in three steps and all three are reported",
  "Updating the intercept and slope rather than refitting is the conventional first step when a model is moved to a new "
  "population [@32308897]. Following the plan, recalibration was applied in three steps and all three are reported"),
 ("apparent gains from externally tuned variants did not survive leakage control",
  "apparent gains from externally tuned variants did not survive leakage control, the failure mode that has driven "
  "reproducibility problems across machine-learning-based science [@37720327]"),
 ("what differs is the strength of the predictor-outcome associations, not the structure of those associations",
  "what differs is the strength of the predictor-outcome associations, not the structure of those associations, a pattern "
  "consistent with the shift-stability framing of deployment failure in health artificial intelligence [@31742354]"),
 ("but a deployed screener should report subgroup performance continuously rather than assume the development-time estimate holds",
  "but a deployed screener should report subgroup performance continuously rather than assume the development-time estimate "
  "holds, as recommended for machine learning intended to advance rather than erode health equity [@30508424]"),
 ("which places a low-burden self-report instrument within reach for digital triage",
  "which places a low-burden self-report instrument within reach for digital triage, the delivery mode in which mobile "
  "interventions for depressive symptoms have shown benefit [@39407174]"),
]
for a, b in INS:
    assert a in head, f"[앵커 없음] {a[:80]}"
    head = head.replace(a, b, 1)
print(f"2) 신규 인용 {len(INS)}개 지점 삽입")

# ── 4. 등장순 재번호 ──
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

new_refs = "\n## References\n\n" + "\n".join(f"{num[k]}. {REFS[k]}" for k in order) + "\n\n## Abbreviations" + tail
io.open(P, "w", encoding="utf8").write(head + new_refs)
print(f"3) 재번호 완료 — 총 {len(order)}편")
