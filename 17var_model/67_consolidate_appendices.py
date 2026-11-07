# -*- coding: utf-8 -*-
"""67_consolidate_appendices.py — 부록을 JMIR 예시처럼 2개로 합치고 표 번호를 연속되게 한다.

예시파일은 부록이 2개다: Multimedia Appendix 1(보충 방법·결과, Table S1~), Multimedia Appendix 2(TRIPOD+AI 체크리스트).
우리는 4개였고 표 번호도 S1, S4~S12(부록3) / S2a, S2a-2, S2a-3, S3, S3b(부록4) 로 어긋나 있었다.

  · 원본 md 는 4개로 유지한다(관리·검증이 쉽다). docx 출력만 2개로 묶는다.
  · 표 번호를 문서 순서대로 S1~S15 로 다시 매기고, 이를 인용하는 모든 문서를 함께 고친다.
  · 부록 번호: TRIPOD → Multimedia Appendix 2, 나머지(크로스워크·임상유용성·천장분석) → Multimedia Appendix 1
"""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__)); DOCS = f"{D}/docs"

# 옛 번호 → 새 번호 (최종 문서 순서: 임상유용성 → 천장분석)
TMAP = [("S12", "S10"), ("S11", "S9"), ("S10", "S8"), ("S9", "S7"), ("S8", "S6"),
        ("S7", "S5"), ("S6", "S4"), ("S5", "S3"), ("S4", "S2"), ("S1", "S1"),
        ("S2a-2", "S12"), ("S2a-3", "S13"), ("S2a", "S11"), ("S3b", "S15"), ("S3", "S14")]
# 부록 번호 재배치
AMAP = {"Multimedia Appendix 1": "Multimedia Appendix 2",   # TRIPOD 체크리스트
        "Multimedia Appendix 2": "Multimedia Appendix 1",   # 크로스워크·개정
        "Multimedia Appendix 3": "Multimedia Appendix 1",   # 임상유용성
        "Multimedia Appendix 4": "Multimedia Appendix 1"}   # 천장분석

TARGETS = ["docs/manuscript_17var.md", "docs/TRIPOD_AI_checklist.md", "docs/appendix2_crosswalk.md",
           "docs/appendix3_utility.md", "docs/appendix4_ceiling.md", "docs/model_card_17var.md",
           "docs/cover_letter.md", "docs/evidence_ledger.md",
           "61_sync_numbers.py", "60_verify_manuscript.py", "44_make_docx.py"]

def remap(text):
    # 1) 표 번호: 긴 이름(S2a-2)부터, 자리표시자를 거쳐 충돌을 막는다
    for old, new in TMAP:
        text = re.sub(rf"Table {re.escape(old)}\b", f"Table \x00{new}\x00", text)
    text = re.sub(r"\x00(S\d+)\x00", r"\1", text)
    # 2) 부록 번호
    for old, new in AMAP.items():
        text = text.replace(old, f"\x01{new[-1]}\x01")
    text = re.sub(r"\x01(\d)\x01", r"Multimedia Appendix \1", text)
    return text

changed = []
for rel in TARGETS:
    p = f"{D}/{rel}"
    if not os.path.exists(p): continue
    s = io.open(p, encoding="utf8").read()
    s2 = remap(s)
    if s2 != s:
        io.open(p, "w", encoding="utf8").write(s2); changed.append(rel)
print("표·부록 번호 재매핑:", ", ".join(os.path.basename(x) for x in changed))

# ── 원고 뒷단: 부록 항목 4개 → 2개 ──
P = f"{DOCS}/manuscript_17var.md"; s = io.open(P, encoding="utf8").read()
i = s.index("## Multimedia Appendix")
j = s.index("## References")
new = ("## Multimedia Appendix 1\n"
       "Supplementary methods and results: the year-specific KCHS↔NHANES harmonization crosswalk and the "
       "predictor-set amendment; clinical utility, sample characteristics and sensitivity analyses (Tables S1–S10); "
       "and the transportability-ceiling analyses (Tables S11–S15).\n\n"
       "## Multimedia Appendix 2\n"
       "TRIPOD+AI (2024) reporting checklist.\n\n")
io.open(P, "w", encoding="utf8").write(s[:i] + new + s[j:])
print("원고 뒷단: 부록 항목 2개로 정리")

# ── 부록 문서 제목을 합본 구조에 맞게 ──
for fn, title in [("TRIPOD_AI_checklist.md", "# Multimedia Appendix 2. TRIPOD+AI (2024) Reporting Checklist"),
                  ("appendix2_crosswalk.md", "## Part A. Harmonization crosswalk and predictor-set amendment"),
                  ("appendix3_utility.md",   "## Part B. Clinical utility, sample characteristics, and sensitivity analyses"),
                  ("appendix4_ceiling.md",   "## Part C. Transportability-ceiling analyses")]:
    p = f"{DOCS}/{fn}"; t = io.open(p, encoding="utf8").read()
    t = re.sub(r"^#{1,2} .*$", title, t, count=1, flags=re.M)
    io.open(p, "w", encoding="utf8").write(t)
print("부록 문서 제목 정리 (Part A/B/C + 체크리스트)")
