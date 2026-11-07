# -*- coding: utf-8 -*-
"""재구성 검증에서 확정된 15건을 고친다.

가장 큰 것은 재번호 매핑의 중복 적용이다. 범위·나열 표기가 두세 번 매핑되어
전혀 다른 표를 가리키고 있었다. 재번호 전 백업에서 한 번만 매핑해 올바른 값을 넣는다.
"""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")
D = "docs"
rd = lambda f: io.open(f"{D}/{f}", encoding="utf8").read()
wr = lambda f, s: io.open(f"{D}/{f}", "w", encoding="utf8").write(s)
n_ok = n_miss = 0

def fix(f, a, b, why):
    global n_ok, n_miss
    s = rd(f)
    if a not in s:
        print(f"  [앵커없음] {f}: {why}"); n_miss += 1; return
    wr(f, s.replace(a, b, 1)); n_ok += 1
    print(f"  {f.split('_')[0]:<12} {why}")

print("=== 1. 매핑 중복 적용 (표 참조가 엉뚱한 표를 가리킴) ===")
fix("manuscript_17var.md", "(Tables S2 and S11 in Multimedia Appendix 1)",
    "(Tables S9 and S11 in Multimedia Appendix 1)", "누출통제 → S9·S11")
fix("manuscript_17var.md", "Tables S6, S12 and S10", "Tables S5, S13 and S15",
    "운영점·순편익·경량모형 → S5·S13·S15")
fix("manuscript_17var.md", "Tables S13 to S14", "Tables S6 and S7",
    "알코올 정의·조화 보정 민감도 → S6·S7")
fix("manuscript_17var.md", "Tables S8, S6 and S10", "Tables S4, S5 and S15",
    "기준 임계값·순편익·고민감도 운영점 → S4·S5·S15")
fix("appendix3_utility.md", "operating characteristics and net benefit in Tables S13, S1 and S7",
    "operating characteristics and net benefit in Tables S4, S5 and S15",
    "범위 문장 → S4·S5·S15")
fix("appendix4_ceiling.md",
    "Tables S2, S7, S11, S9 and S3 concern the ceiling; Table S3, placed last so that the appendix "
    "tables run in numerical order, gives the calibration intervals",
    "Tables S9, S10, S11, S12 and S16 concern the ceiling; Table S3 gives the calibration intervals",
    "천장 표 목록 → S9·S10·S11·S12·S16, 'placed last' 삭제")
fix("appendix3_utility.md", "The same caution applies to Tables S2 and S9,",
    "The same caution applies to Tables S9 and S12,", "탐색런 기준선 표 → S9·S12")

print("\n=== 2. 모델카드 중복 항목 ===")
p = "model_card_17var.md"
s = rd(p); lines = s.split("\n")
seen, out, dropped = set(), [], 0
for l in lines:
    k = l.strip()
    if k.startswith("- ") and k in seen and len(k) > 25:
        dropped += 1; continue
    if k.startswith("- ") and len(k) > 25: seen.add(k)
    out.append(l)
if dropped:
    wr(p, "\n".join(out)); print(f"  model_card    같은 항목 {dropped}줄 제거")
else: print("  model_card    중복 없음")

print("\n=== 3. 죽은 포인터 (Part 구조 제거로 사라진 라벨) ===")
DEAD = [
 ("TRIPOD_AI_checklist.md", "Tables A.4", "the analytic sample derivation in Multimedia Appendix 1"),
 ("TRIPOD_AI_checklist.md", "sections B.2-B.3", "the predictor-set amendment in Multimedia Appendix 1"),
 ("appendix2_crosswalk.md", "\u00a7B.5 below reproduces", "the expectations section below reproduces"),
]
for f, a, b in DEAD:
    s = rd(f)
    if a in s: wr(f, s.replace(a, b)); print(f"  {f.split('_')[0]:<12} {a} → {b[:44]}")
    else: print(f"  [없음] {f}: {a}")
for f in ("appendix2_crosswalk.md",):
    s = rd(f); k = 0
    for a, b in (("Section A documents", "The crosswalk below documents"),
                 ("Section B is an English", "The amendment section is an English"),
                 ("Section A", "the crosswalk"), ("Section B", "the amendment section")):
        if a in s: s = s.replace(a, b); k += 1
    if k: wr(f, s); print(f"  {f.split('_')[0]:<12} 'Section A/B' 표기 {k}종 정리")

print("\n=== 4. 'above/below' 방향 ===")
fix("appendix3_utility.md", "the oracle by predictor set above", "the oracle by predictor set in Table S16")

print(f"\n적용 {n_ok} · 앵커 없음 {n_miss}")
