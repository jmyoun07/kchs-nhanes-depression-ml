# -*- coding: utf-8 -*-
"""A5_merge_seam_check.py — 기계적 병합이 남긴 이음매를 찾는다.

81건의 대체 문장을 기계로 끼워 넣었으므로 사람이 읽어야 보이는 흠이 남을 수 있다.
에이전트를 못 쓰는 동안 규칙으로 잡을 수 있는 것만이라도 전수 검사한다.
"""
import io, os, re, sys, glob
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__))
FILES = [f for f in sorted(glob.glob(f"{D}/docs/*.md")) if not any(x in f for x in ("_audit","_ms_before","STYLE_RULES","evidence_ledger","prespecification"))]

TRANS = r"(?:However|Therefore|Moreover|Furthermore|In contrast|By contrast|Nevertheless|Thus|Hence|In addition|Also|Conversely)"
hits = []
def rep(f, kind, txt):
    hits.append((os.path.basename(f), kind, txt.replace("\n", " ")[:150]))

for f in FILES:
    s = io.open(f, encoding="utf8").read()
    prose = "\n".join(l for l in s.split("\n") if not l.strip().startswith("|"))

    # ① 연속 전환어 (…. However, Therefore, …)
    for m in re.finditer(rf"{TRANS},\s+{TRANS},", prose):
        rep(f, "전환어 중복", prose[max(0, m.start()-60):m.end()+50])
    # ② 문장 끝 구두점 누락 (소문자로 끝나고 다음이 대문자 문장 시작)
    for m in re.finditer(r"[a-z]{3}\s+[A-Z][a-z]{3,}\s+(?:was|were|is|are|has|have)\b", prose):
        seg = prose[max(0, m.start()-70):m.end()+20]
        if "." not in seg[-90:-20] and "," not in seg[-30:]:
            rep(f, "마침표 누락 의심", seg)
    # ③ 이중 구두점·공백
    for pat, nm in ((r"[.,;:]{2,}", "구두점 중복"), (r"[ \t]{2,}(?!\|)", "공백 중복"),
                    (r"\(\s|\s\)", "괄호 공백"), (r"\s+[.,;]", "구두점 앞 공백")):
        for m in re.finditer(pat, prose):
            rep(f, nm, prose[max(0, m.start()-55):m.end()+55])
    # ④ 인접 문장 중복 (같은 문단 안에서 40자 이상 겹침)
    for para in prose.split("\n\n"):
        sents = re.split(r"(?<=[.;])\s+", para)
        for a, b in zip(sents, sents[1:]):
            a, b = a.strip(), b.strip()
            if len(a) < 45 or len(b) < 45: continue
            aw, bw = set(re.findall(r"[a-z]{5,}", a.lower())), set(re.findall(r"[a-z]{5,}", b.lower()))
            if aw and bw and len(aw & bw) / min(len(aw), len(bw)) > 0.72:
                rep(f, "인접 문장 중복 의심", a[:70] + " ‖ " + b[:70])
    # ⑤ 같은 문장 안에서 같은 구절 반복
    for m in re.finditer(r"([a-z]{4,}(?:\s+[a-z]{3,}){2,4})(?=[^.;]{0,80}\1)", prose):
        rep(f, "문장 내 구절 반복", prose[max(0, m.start()-40):m.end()+90])

# ⑥ 약어: 첫 사용보다 정의가 뒤인지
M = io.open(f"{D}/docs/manuscript_17var.md", encoding="utf8").read()
body = M.split("## References")[0]
for m in re.finditer(r"\(([A-Z]{2,7}(?:\+[A-Z]{2})?)\)", body):
    ab = m.group(1)
    first = re.search(rf"\b{re.escape(ab)}\b", body)
    if first and first.start() < m.start() - len(ab) - 2:
        seg = body[max(0, first.start()-60):first.end()+50]
        hits.append(("manuscript_17var.md", "약어 정의가 첫 사용보다 뒤", f"{ab}: {seg}".replace("\n", " ")[:150]))

c = Counter(k for _, k, _ in hits)
print(f"이음매 의심 {len(hits)}건 · 유형 {dict(c)}\n")
for fn, k, t in hits[:40]:
    print(f"[{fn}] {k}\n    {t}\n")
