# -*- coding: utf-8 -*-
"""부록 표를 본문 등장 순서로 재번호하고 물리적으로 재배열한다.

JMIR 규정: "Each appendix must number items as S1, S2, S3... in order of appearance."
현재 등장 순서는 S8 → S9 → S16 → S1 → S3 … 이라 규정과 어긋난다.

하는 일
 ① 원고 본문의 최초 인용 순서로 새 번호를 매긴다
 ② 모든 문서·생성기의 'Table S<옛>' 을 한꺼번에 바꾼다(자리표시자 2단계로 충돌 방지)
 ③ 표 덩어리를 새 번호 순으로 한 파일에 모은다
 ④ 61_sync 의 구간 끝 앵커도 새 번호로 맞춘다
"""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")
D = "docs"
rd = lambda f: io.open(f"{D}/{f}", encoding="utf8").read()
wr = lambda f, s: io.open(f"{D}/{f}", "w", encoding="utf8").write(s)

# ── ① 새 번호 ───────────────────────────────────────────────────────────
BODY = rd("manuscript_17var.md").split("## Figure Legends")[0]
order, seen = [], set()
for m in re.finditer(r"Table S(\d+)", BODY):
    n = int(m.group(1))
    if n not in seen: seen.add(n); order.append(n)
MAP = {old: new for new, old in enumerate(order, 1)}
assert len(MAP) == 16, f"표 {len(MAP)}개만 매핑됨"
print("재번호:", " ".join(f"S{o}->S{n}" for o, n in sorted(MAP.items())))

# ── ② 모든 문서·생성기 치환 ────────────────────────────────────────────
DOCS = ["manuscript_17var.md", "TRIPOD_AI_checklist.md", "model_card_17var.md", "cover_letter.md",
        "appendix2_crosswalk.md", "appendix3_utility.md", "appendix4_ceiling.md", "appendix5_methods.md"]
GENS = ["61_sync_numbers.py", "60_verify_manuscript.py", "92_crossref_audit.py"]
SEP = "\uE000"                                    # 사용자 영역 문자 - 원문에 나올 리 없다
def renum(s):
    s = re.sub(r"Table S(\d+)\b", lambda m: f"Table {SEP}{MAP[int(m.group(1))]}{SEP}", s)
    s = re.sub(r"Tables S(\d+)\b", lambda m: f"Tables {SEP}{MAP[int(m.group(1))]}{SEP}", s)
    s = re.sub(r"(?<=[,和] )S(\d+)\b", lambda m: f"{SEP}{MAP[int(m.group(1))]}{SEP}", s)
    s = re.sub(r"(?<=and )S(\d+)\b", lambda m: f"{SEP}{MAP[int(m.group(1))]}{SEP}", s)
    return s.replace(SEP + SEP, "").replace(SEP, "S", 1) if False else re.sub(
        SEP + r"(\d+)" + SEP, lambda m: "S" + m.group(1), s)

for f in DOCS:
    wr(f, renum(rd(f)))
for g in GENS:
    s = io.open(g, encoding="utf8").read()
    io.open(g, "w", encoding="utf8").write(renum(s))
    compile(io.open(g, encoding="utf8").read(), g, "exec")
print(f"② 문서 {len(DOCS)}개 · 생성기 {len(GENS)}개 치환")

# ── ③ 표 덩어리를 새 번호 순으로 한 파일에 모은다 ───────────────────────
def split_blocks(s):
    """(제목없는 머리부분, [(번호, 덩어리)], [(제목, 덩어리)]) 로 나눈다."""
    heads = [(m.start(), m.group(0)) for m in re.finditer(r"(?m)^#{2,4}\s+.+$", s)]
    if not heads: return s, [], []
    pre = s[:heads[0][0]]
    tabs, prose = [], []
    for k, (pos, h) in enumerate(heads):
        end = heads[k + 1][0] if k + 1 < len(heads) else len(s)
        blk = s[pos:end]
        m = re.match(r"^#{2,4}\s+Table S(\d+)\.", h)
        (tabs if m else prose).append((int(m.group(1)) if m else h, blk))
    return pre, tabs, prose

pre3, tab3, pro3 = split_blocks(rd("appendix3_utility.md"))
pre4, tab4, pro4 = split_blocks(rd("appendix4_ceiling.md"))
alltabs = sorted(tab3 + tab4, key=lambda x: x[0])
nums = [n for n, _ in alltabs]
assert nums == list(range(1, 17)), f"표 번호가 1..16 연속이 아님: {nums}"

# 표는 모두 Supplementary Results 아래로, 산문은 그 뒤에
res = pre3.rstrip() + "\n\n" + "\n".join(b.rstrip() + "\n" for _, b in alltabs)
res += "\n" + "\n".join(b.rstrip() + "\n" for _, b in pro3)
wr("appendix3_utility.md", re.sub(r"\n{3,}", "\n\n", res) + "\n")
cei = pre4.rstrip() + "\n\n" + "\n".join(b.rstrip() + "\n" for _, b in pro4)
wr("appendix4_ceiling.md", re.sub(r"\n{3,}", "\n\n", cei).lstrip("\n") + "\n")
print(f"③ 표 16개를 Supplementary Results 로 모음 (천장분석 파일은 산문 {len(pro4)}절만 남김)")

# ── ④ 61_sync 끝 앵커 점검 ─────────────────────────────────────────────
s = io.open("61_sync_numbers.py", encoding="utf8").read()
bad = []
for m in re.finditer(r'splice\("(appendix[345][^"]*)",\s*"([^"]*)",\s*"((?:[^"\\]|\\.)*)"', s):
    f, start, end = m.group(1), m.group(2), m.group(3).replace("\\n", "\n")
    src = rd(f)
    if start not in src or end.lstrip("\n") not in src: bad.append((f, start[:40], end.strip()[:40]))
print(f"④ splice 앵커 점검: {'모두 유효' if not bad else bad}")
