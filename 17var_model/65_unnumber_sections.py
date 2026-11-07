# -*- coding: utf-8 -*-
"""65_unnumber_sections.py — 절 번호를 없애고 §상호참조를 정리한다 (JMIR 예시파일 방식).

예시 논문(윤수 예시파일)에는 절 번호도, 절을 가리키는 상호참조도 **한 건도 없다**.
부록 표만 'Table S9 in Multimedia Appendix 1' 형태로 가리킨다. 그 관행을 그대로 따른다.

  · 원고: 제목에서 번호 제거 + Title Case, 본문 §참조는 삭제(뜻이 사라지는 곳만 문장을 고쳐 씀)
  · Strengths + Limitations → 예시처럼 하나의 'Strengths and Limitations'
  · Conclusions 제목 제거(예시는 마지막 문단을 제목 없이 둔다)
  · 부록1(TRIPOD 체크리스트): '어디에 보고했는가'가 목적이므로 §2.5 → 절 이름으로 치환
  · 그 밖의 문서: §2.x→Methods, §3.x→Results, §4.x→Discussion
"""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__)); DOCS = f"{D}/docs"

NAME = {
 "2.1": "Study Design and Reporting Standards", "2.2": "Data Sources", "2.3": "Participants",
 "2.4": "Outcome", "2.5": "Candidate Predictors and Harmonization", "2.5.1": "Provenance of the Harmonized Predictor Set",
 "2.6": "Missing Data", "2.7": "Sample Size", "2.8": "Model Development", "2.9": "Performance Evaluation",
 "2.10": "External Validation Strategy", "2.11": "Interpretability, Fairness, and Deployment",
 "2.12": "Planned Sensitivity Analyses", "2.13": "Data-Integrity Audit", "2.14": "Ethical Considerations",
 "3.1": "Participant Characteristics", "3.2": "Model Development and Internal Performance",
 "3.3": "Geographic and Temporal External Validation", "3.4": "Why External Discrimination Plateaus at ~0.77",
 "3.5": "Parsimonious Deployment Model", "3.6": "Sensitivity Analyses",
 "3.7": "Interpretability, Fairness, and Clinical Utility",
 "4.1": "Principal Findings", "4.2": "Comparison With Prior Work",
 "4.3": "Strengths and Limitations", "4.4": "Strengths and Limitations", "4.5": "Conclusions",
}
TOP = {"1": "Introduction", "2": "Methods", "3": "Results", "4": "Discussion"}

# ── 1. 원고: 제목 ──
P = f"{DOCS}/manuscript_17var.md"
s = io.open(P, encoding="utf8").read()

def head(m):
    hashes, num, title = m.group(1), m.group(2), m.group(3).strip()
    if num in TOP: return f"{hashes} {TOP[num]}"
    if num == "4.4": return "\x00DROP\x00"          # Strengths 와 합친다
    if num == "4.5": return "\x00DROP\x00"          # 결론은 제목 없이 (예시와 동일)
    return f"{hashes} {NAME.get(num, title)}"
s = re.sub(r"^(#{2,4}) (\d+(?:\.\d+)*)\.? (.+)$", head, s, flags=re.M)
s = re.sub(r"\x00DROP\x00\n+", "", s)
s = s.replace("**2.5.1 Provenance of the harmonized predictor set.**", "**Provenance of the harmonized predictor set.**")

# ── 2. 원고: 뜻이 사라지는 §참조는 문장을 고쳐 쓴다 ──
REWRITE = [
 ("The predictor set is documented in §2.5, including a dated protocol amendment covering three of the seventeen items (§2.5.1).",
  "The predictor set, including a dated protocol amendment covering three of the seventeen items, is documented below."),
 ("The three predictors covered by the protocol amendment are documented in §2.5.1.",
  "The three predictors covered by the protocol amendment are documented below."),
 ("A silent-column-omission check was added after a data-integrity audit (see §2.13).",
  "A silent-column-omission check was added after a data-integrity audit."),
 ("is addressed under harmonization cautions (§2.5) and Limitations rather than",
  "is addressed under Harmonization cautions and in the Limitations rather than"),
 ("(a deliberate §2.10 violation, for diagnosis only)",
  "(a deliberate departure from the external-validation rule, for diagnosis only)"),
 ("released unmodified (Multimedia Appendix 2, §2.5.1)", "released unmodified (Multimedia Appendix 2)"),
]
for a, b in REWRITE:
    if a not in s: raise SystemExit(f"[실패] 원고 문장 미일치: {a[:80]}")
    s = s.replace(a, b)

# ── 3. 원고: 남은 §참조 삭제 ──
s = re.sub(r"\s*\(§[\d.]+(?:\s*,\s*§[\d.]+)*\)", "", s)          # " (§2.5)" · " (§2.2, §2.5)"
s = re.sub(r"[;,]\s*§[\d.]+(?=\))", "", s)                        # "(1000 replicates; §2.9)"
s = re.sub(r"\s*§[\d.]+", "", s)                                  # 남은 것
left = re.findall(r"§", s)
if left: raise SystemExit(f"[실패] 원고에 §가 {len(left)}개 남음")
io.open(P, "w", encoding="utf8").write(s)
print(f"원고: 제목 번호 제거 · §참조 정리 완료 (제목 {len(re.findall(r'^#{2,4} ', s, re.M))}개)")

# ── 4. 부록1(TRIPOD): §번호 → 절 이름 ──
def to_name(m):
    n = m.group(1)
    if n in NAME: return NAME[n]
    return TOP.get(n.split(".")[0], "Methods")
p = f"{DOCS}/TRIPOD_AI_checklist.md"; t = io.open(p, encoding="utf8").read()
t2 = re.sub(r"§(\d+(?:\.\d+)*)", to_name, t)
io.open(p, "w", encoding="utf8").write(t2)
print(f"부록1: §참조 {len(re.findall(r'§', t))}건 → 절 이름")

# ── 5. 나머지 문서: §2.x→Methods, §3.x→Results, §4.x→Discussion ──
COARSE = {"1": "the Introduction", "2": "the Methods", "3": "the Results", "4": "the Discussion"}
for fn in ["appendix2_crosswalk.md", "appendix3_utility.md", "appendix4_ceiling.md",
           "model_card_17var.md", "cover_letter.md", "evidence_ledger.md"]:
    p = f"{DOCS}/{fn}"; t = io.open(p, encoding="utf8").read(); n = len(re.findall(r"§", t))
    if not n: continue
    t = re.sub(r"\(§(\d+)(?:\.\d+)*(?:\s*,\s*§[\d.]+)*\)", lambda m: f"({COARSE[m.group(1)]})", t)
    t = re.sub(r"§(\d+)(?:\.\d+)*", lambda m: COARSE[m.group(1)], t)
    io.open(p, "w", encoding="utf8").write(t)
    print(f"{fn}: §참조 {n}건 → 대분류 이름")
print("\n완료. 이어서 61 → 44/45 → 60 순으로 실행하세요.")
