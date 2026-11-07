# -*- coding: utf-8 -*-
"""A1_refs_pubmed_sci.py — PubMed 등재 SCI급 저널 문헌만 남긴다.

교체 대상 3편(모두 PubMed 미등재이거나 학회록):
  34 Lumley, J Stat Softw          → Rust & Rao, Stat Methods Med Res (PMID 8931197)
                                      복합표본 복제 기반 분산추정의 정본 문헌. 인용 맥락(재척도 군집 부트스트랩)에 더 정확하다.
  36 Davis et al, AMIA Annu Symp   → Vergouwe et al, Stat Med (PMID 27891652)
                                      절편만/절편+기울기/전면 재적합의 3단계를 다루므로 본문의 3단계 재보정과 정확히 대응한다.
  37 Lundberg & Lee, NIPS 학회록    → Lundberg et al, Nat Mach Intell (PMID 32607472)
                                      TreeSHAP 논문. 이 연구가 실제로 쓰는 알고리즘이라 원 인용보다 정확하다.

출처: PubMed.
"""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__))
P = f"{D}/docs/manuscript_17var.md"

REPL = {
 34: ("Rust KF, Rao JN. Variance estimation for complex surveys using replication techniques. "
      "Stat Methods Med Res. Sep 1996;5(3):283-310. [doi: 10.1177/096228029600500305] [Medline: 8931197]"),
 36: ("Vergouwe Y, Nieboer D, Oostenbrink R, Debray TPA, Murray GD, Kattan MW, et al. "
      "A closed testing procedure to select an appropriate method for updating prediction models. "
      "Stat Med. 2017;36(28):4529-4539. [doi: 10.1002/sim.7179] [Medline: 27891652]"),
 37: ("Lundberg SM, Erion G, Chen H, DeGrave A, Prutkin JM, Nair B, et al. "
      "From local explanations to global understanding with explainable AI for trees. "
      "Nat Mach Intell. Jan 17, 2020;2(1):56-67. [doi: 10.1038/s42256-019-0138-9] [Medline: 32607472]"),
}

s = io.open(P, encoding="utf8").read()
head, rest = s.split("## References", 1)
reflines, tail = rest.split("## Abbreviations", 1)
lines = [l for l in reflines.strip().split("\n") if l.strip()]
out, n = [], 0
for l in lines:
    m = re.match(r"^(\d+)\.\s+(.*)$", l.strip())
    k = int(m.group(1))
    if k in REPL:
        print(f"  [{k}] 교체")
        print(f"      전: {m.group(2)[:110]}")
        print(f"      후: {REPL[k][:110]}")
        out.append(f"{k}. {REPL[k]}"); n += 1
    else:
        out.append(l.strip())
io.open(P, "w", encoding="utf8").write(head + "## References\n\n" + "\n".join(out) + "\n\n## Abbreviations" + tail)
print(f"\n{n}편 교체 · 총 {len(out)}편")

# 본문의 SHAP 인용 문구를 TreeSHAP 에 맞춘다
s = io.open(P, encoding="utf8").read()
a = "SHAP global importance [37]"
b = "Global importance from tree-based Shapley additive explanations [37]"
if a in s:
    io.open(P, "w", encoding="utf8").write(s.replace(a, b, 1)); print("본문: SHAP 인용 문구를 TreeSHAP 에 맞춤")

# 전수 검증
s = io.open(P, encoding="utf8").read()
refs = [l for l in s.split("## References")[1].split("## Abbreviations")[0].strip().split("\n") if l.strip()]
nomed = [r for r in refs if "[Medline:" not in r]
print(f"\n검증: {len(refs)}편 · Medline ID 없는 항목 {len(nomed)}건")
for r in nomed: print("   ", r[:150])
