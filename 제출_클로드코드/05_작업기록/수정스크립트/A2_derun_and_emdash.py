# -*- coding: utf-8 -*-
"""A2_derun_and_emdash.py — 예시논문 조판에 맞춘다.

확인한 사실(윤수 예시파일/JMIR ...docx):
  · 굵게 처리되는 것은 제목, 초록 라벨(Background: 등), 절 제목, 표·그림 캡션뿐이다.
    문단 안의 굵은 라벨은 하나도 없다.
  · em dash(—) 사용 0회.

따라서 문단 내 굵은 라벨 16개를 없애고 그 정보를 문장에 녹이며, em dash 를 전부 없앤다.
초록 라벨·표/그림 캡션·논문유형 표기는 예시와 같으므로 남긴다.
"""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__))
P = f"{D}/docs/manuscript_17var.md"

# (라벨 포함 원문, 교체문) — 라벨을 지우고 첫 문장이 스스로 서게 만든다
REPL = [
 ("**Data sources.** **Development — KCHS.** Nationwide Korea Disease Control and Prevention Agency (KDCA) survey [17] of community-dwelling adults,",
  "Development used the Korea Community Health Survey, a nationwide Korea Disease Control and Prevention Agency (KDCA) survey [17] of community-dwelling adults,"),
 ("**External validation — US NHANES.** NCHS survey combining household interview and mobile examination center assessment;",
  "External validation used NHANES, an NCHS survey combining household interview and mobile examination center assessment;"),
 ("**Outcome.** Current depression = PHQ-9 total ≥10, computed identically in both surveys (nine items, 0–3);",
  "The outcome is current depression, defined as a PHQ-9 total of at least 10 and computed identically in both surveys (nine items, scored 0 to 3);"),
 ("**Provenance of the harmonized predictor set.** The harmonized set was assembled",
  "The harmonized set was assembled"),
 ("**Post-hoc candidate screening (reported for completeness).** A later exhaustive extraction surfaced",
  "A later exhaustive extraction, reported here for completeness, surfaced"),
 ("**Harmonization cautions.** A direct Korea-US comparison",
  "Three harmonization cautions attach to the set. A direct Korea-US comparison"),
 ("**Missing data.** All missing-data handling", "All missing-data handling"),
 ("**Algorithms and selection.** Algorithms: penalized logistic regression (elastic net), random forest, and gradient boosting (LightGBM). Development proceeded",
  "Three algorithms were considered: penalized logistic regression (elastic net), random forest, and gradient boosting (LightGBM). Development proceeded"),
 ("**Performance evaluation.** Discrimination (AUROC and area under the precision–recall curve [AUPRC]), calibration, and operating characteristics, all complex-survey–weighted.",
  "Performance was summarized by discrimination (AUROC and area under the precision-recall curve [AUPRC]), calibration, and operating characteristics, all complex-survey weighted."),
 ("**Interpretability, fairness, and deployment.** Global importance", "Interpretability and fairness were examined through global importance"),
 ("**Sample size.** No a priori sample-size calculation was performed.", "No a priori sample-size calculation was performed."),
 ("**Intended use and users.** The model is intended", "The model is intended"),
 ("**Planned sensitivity analyses.** Pre-specified and additional analyses,",
  "Sensitivity analyses were pre-specified, with further analyses added afterwards,"),
 ("**Data-integrity audit.** An adversarial audit corrected", "An adversarial data-integrity audit corrected"),
 ("**Living alone as a cycle-specific predictor.** Living alone is recorded", "Living alone is recorded"),
]

s = io.open(P, encoding="utf8").read()
n = 0
for a, b in REPL:
    if a not in s:
        print(f"  [미일치] {a[:78]}"); continue
    s = s.replace(a, b, 1); n += 1
io.open(P, "w", encoding="utf8").write(s)
print(f"문단 내 굵은 라벨 {n}/{len(REPL)}개 제거")

# ── em dash 제거 ────────────────────────────────────────────────────────────
EM = [
 ("annual in-person household interviews (~230,000/year), stratified/clustered/weighted complex-sample design",
  "annual in-person household interviews (~230,000/year) and a stratified, clustered, weighted complex-sample design"),
]
s = io.open(P, encoding="utf8").read()
for a, b in EM:
    if a in s: s = s.replace(a, b, 1)
io.open(P, "w", encoding="utf8").write(s)

s = io.open(P, encoding="utf8").read()
rem = [m.group(0) for m in re.finditer(r"[^\n]{0,60}—[^\n]{0,60}", s)]
print(f"\n남은 em dash {len(rem)}곳:")
for r in rem: print("  …", r.strip()[:130])
