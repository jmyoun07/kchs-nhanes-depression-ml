# -*- coding: utf-8 -*-
"""인용 문장과 그 표의 캡션이 서로 맞는지 본다.

이번에 놓친 결함은 전부 '문장이 엉뚱한 표를 가리킴' 이었다. 기존 검사는 'SN 이라는 표가
있는가' 만 봤기 때문에 전부 통과했다. 문장 쪽 낱말과 캡션 낱말이 하나도 안 겹치면 의심한다.

먼저 잡음 수준을 재 본다. 오탐이 많으면 검사로 넣을 수 없다.
"""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")
D = "docs"
rd = lambda f: io.open(f"{D}/{f}", encoding="utf8").read()

APPX = ["appendix2_crosswalk.md", "appendix3_utility.md", "appendix4_ceiling.md", "appendix5_methods.md"]
CAP = {}
for f in APPX:
    for m in re.finditer(r"(?m)^#{2,4}\s+Table S(\d+)\.\s*(.+)$", rd(f)):
        CAP[int(m.group(1))] = m.group(2)

STOP = set("""the a an and or of in to for by with from on at as is are was were be been that this these those
it its their our we they he she has have had not no than then when where which who whom whose all any both
each few more most other some such only own same so nor too very can will just should now table tables
multimedia appendix results result values value given gives give shown show exact detail details full
above below here reported report see also using used use per across between within into over under""".split())
def words(t):
    return {w for w in re.findall(r"[a-z][a-z\-]{3,}", t.lower()) if w not in STOP}

rows = []
for f in ["manuscript_17var.md"] + APPX:
    s = rd(f)
    for m in re.finditer(r"Tables? S(\d+)((?:[,\s]+(?:and\s+)?S\d+)*)", s):
        ns = [int(m.group(1))] + [int(x) for x in re.findall(r"S(\d+)", m.group(2) or "")]
        # 문장 안에서 인용 앞 구절
        st = s.rfind(".", 0, m.start()) + 1
        ctx = s[max(st, m.start() - 260): m.start()]
        cw = words(ctx)
        for n in ns:
            if n not in CAP: continue
            kw = words(CAP[n])
            if kw and cw and not (kw & cw):
                rows.append((f, n, CAP[n][:52], " ".join(ctx.split())[-92:]))

print(f"캡션과 낱말이 하나도 안 겹치는 인용: {len(rows)}건\n")
for f, n, cap, ctx in rows[:18]:
    print(f"  [{f.split('_')[0]:<11}] S{n:<3} 캡션: {cap}")
    print(f"                 문장: …{ctx}")
