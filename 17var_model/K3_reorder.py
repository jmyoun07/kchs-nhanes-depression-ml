# -*- coding: utf-8 -*-
"""3단계: 표를 번호 순으로 물리적으로 재배열한다.

지난 시도에서 여기서 표 11개가 사라졌다. 원인은 두 가지였다.
  · 절 제목('## Supplementary Results')을 표와 같은 급으로 보고 표 뒤로 밀어냈다
  · 조립 결과를 세지 않고 그대로 덮어썼다
이번에는 (1) 제목 급을 구분하고, (2) 조립 전후로 표 개수를 세고, (3) 16개가 아니면 쓰지 않는다.
"""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")
D = "docs"
rd = lambda f: io.open(f"{D}/{f}", encoding="utf8").read()

def blocks(s):
    """제목 단위로 쪼갠다. (제목줄, 덩어리) 목록과 머리부분을 돌려준다."""
    heads = [m.start() for m in re.finditer(r"(?m)^#{2,4}\s+.+$", s)]
    if not heads: return s, []
    out = []
    for k, pos in enumerate(heads):
        end = heads[k + 1] if k + 1 < len(heads) else len(s)
        out.append((s[pos:s.index("\n", pos)] if "\n" in s[pos:] else s[pos:], s[pos:end]))
    return s[:heads[0]], out

pre3, b3 = blocks(rd("appendix3_utility.md"))
pre4, b4 = blocks(rd("appendix4_ceiling.md"))
TAB = re.compile(r"^#{2,4}\s+Table S(\d+)\.")

tables, prose3, prose4, sec_head = [], [], [], None
for h, blk in b3:
    m = TAB.match(h)
    if m: tables.append((int(m.group(1)), blk))
    elif h.strip() == "## Supplementary Results": sec_head = blk
    else: prose3.append(blk)
for h, blk in b4:
    m = TAB.match(h)
    if m: tables.append((int(m.group(1)), blk))
    else: prose4.append(blk)

nums = sorted(n for n, _ in tables)
assert nums == list(range(1, 17)), f"표가 1..16 이 아님: {nums}"
assert sec_head, "'## Supplementary Results' 제목을 못 찾음"
print(f"수집: 표 {len(tables)}개 · 결과절 산문 {len(prose3)} · 천장절 산문 {len(prose4)}")

# 조립: 절 제목 → 표 1..16 → 산문
tables.sort(key=lambda x: x[0])
body = sec_head.rstrip() + "\n\n" + "\n".join(b.rstrip() + "\n" for _, b in tables)
body += "\n" + "\n".join(b.rstrip() + "\n" for b in prose3)
body = re.sub(r"\n{3,}", "\n\n", body) + "\n"

cei = pre4.rstrip() + "\n\n" + "\n".join(b.rstrip() + "\n" for b in prose4)
cei = re.sub(r"\n{3,}", "\n\n", cei).lstrip("\n") + "\n"

# 쓰기 전에 센다
n_new = len(re.findall(r"(?m)^#{2,4}\s+Table S\d+\.", body))
assert n_new == 16, f"조립 결과에 표가 {n_new}개뿐"
assert not re.search(r"(?m)^#{2,4}\s+Table S\d+\.", cei), "천장절에 표가 남음"
seq = [int(x) for x in re.findall(r"(?m)^#{2,4}\s+Table S(\d+)\.", body)]
assert seq == list(range(1, 17)), f"물리 순서가 1..16 이 아님: {seq}"

io.open(f"{D}/appendix3_utility.md", "w", encoding="utf8").write(body)
io.open(f"{D}/appendix4_ceiling.md", "w", encoding="utf8").write(cei)
print(f"조립 완료: 결과절에 표 {n_new}개 (물리 순서 1..16) · 천장절은 산문만")
