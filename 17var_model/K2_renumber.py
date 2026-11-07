# -*- coding: utf-8 -*-
"""2단계: 부록 표 번호를 본문 등장 순서로 바꾼다. 물리적 이동은 하지 않는다.

JMIR 규정: 부록 항목은 본문 등장 순서로 S1, S2, ... 를 매긴다.
이 단계에서 검증이 통과해야 3단계(재배열)로 간다.

범위 표기 주의: 'Tables S11 to S15' 는 재번호하면 연속이 아니게 된다.
먼저 낱개로 풀고, 재번호한 뒤 다시 연속인 것만 범위로 합친다.
"""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")
D = "docs"

BODY = io.open(f"{D}/manuscript_17var.md", encoding="utf8").read().split("## Figure Legends")[0]
order, seen = [], set()
for m in re.finditer(r"Table S(\d+)", BODY):
    n = int(m.group(1))
    if n not in seen: seen.add(n); order.append(n)
MAP = {old: new for new, old in enumerate(order, 1)}
assert sorted(MAP) == list(range(1, 17)), f"매핑 불완전: {sorted(MAP)}"
print("재번호:", "  ".join(f"S{o}→S{MAP[o]}" for o in sorted(MAP)))

FILES = [f"{D}/{x}" for x in ("manuscript_17var.md", "TRIPOD_AI_checklist.md", "model_card_17var.md",
                              "cover_letter.md", "evidence_ledger.md", "appendix2_crosswalk.md",
                              "appendix3_utility.md", "appendix4_ceiling.md", "appendix5_methods.md")] \
        + ["61_sync_numbers.py", "60_verify_manuscript.py", "92_crossref_audit.py"]

SEP = ""

def fmt(ns, word):
    """번호 목록을 'Tables S1 to S4' / 'Tables S1, S3 and S7' 로 되돌린다.

    반드시 자리표시자로 감싸서 돌려준다. 평문 'S4' 로 돌려주면 뒤따르는 낱개 치환이
    이 결과를 다시 매핑한다. 실제로 그렇게 범위·나열이 두세 번 매핑되어 전혀 다른 표를
    가리키게 됐고, 기계 검증 1,042건이 전부 통과하는데도 심사자만 알아볼 결함이 남았다.
    """
    ns = sorted(set(ns))
    if len(ns) == 1: return f"Table {SEP}{ns[0]}{SEP}"
    if ns == list(range(ns[0], ns[-1] + 1)):
        return f"{word} {SEP}{ns[0]}{SEP} to {SEP}{ns[-1]}{SEP}"
    if len(ns) == 2: return f"{word} {SEP}{ns[0]}{SEP} and {SEP}{ns[1]}{SEP}"
    return f"{word} " + ", ".join(f"{SEP}{x}{SEP}" for x in ns[:-1]) + f" and {SEP}{ns[-1]}{SEP}"

SEP = "\uE000"
tot = 0
for f in FILES:
    s = io.open(f, encoding="utf8").read(); before = s

    # ① 범위·나열을 새 번호 목록으로 통째 교체
    def rng(m):
        word = m.group(1)
        a, b = int(m.group(2)), int(m.group(3))
        old = list(range(a, b + 1)) if m.group(0).find(" and ") < 0 else [a, b]
        return fmt([MAP[x] for x in old], word).replace("S", SEP, 0)
    s = re.sub(r"(Tables?) S(\d+)\s*(?:to|-|–)\s*S?(\d+)", rng, s)
    s = re.sub(r"(Tables?) S(\d+) and S(\d+)", rng, s)
    s = re.sub(r"(Tables?) S(\d+), S(\d+),? and S(\d+)",
               lambda m: fmt([MAP[int(m.group(i))] for i in (2, 3, 4)], m.group(1)), s)

    # ② 남은 낱개 번호
    s = re.sub(r"(?<![A-Za-z0-9])S(\d+)(?![0-9])",
               lambda m: f"{SEP}{MAP[int(m.group(1))]}{SEP}" if int(m.group(1)) in MAP else m.group(0), s)
    s = re.sub(SEP + r"(\d+)" + SEP, lambda m: "S" + m.group(1), s)

    if s != before:
        io.open(f, "w", encoding="utf8").write(s); tot += 1
        if f.endswith(".py"): compile(io.open(f, encoding="utf8").read(), f, "exec")
print(f"\n{tot}개 파일 치환")

# 확인: 정의된 표 번호와 본문 등장 순서
app = "\n".join(io.open(f"{D}/{x}", encoding="utf8").read()
                for x in ("appendix3_utility.md", "appendix4_ceiling.md"))
defined = sorted(int(x) for x in re.findall(r"(?m)^#{2,4}\s+Table S(\d+)\.", app))
BODY2 = io.open(f"{D}/manuscript_17var.md", encoding="utf8").read().split("## Figure Legends")[0]
o2, s2 = [], set()
for m in re.finditer(r"Table S(\d+)", BODY2):
    n = int(m.group(1))
    if n not in s2: s2.add(n); o2.append(n)
print(f"정의된 표: {defined}")
print(f"본문 등장 순서: {o2}")
print("등장 순서가 1..16 인가:", o2 == list(range(1, 17)))
