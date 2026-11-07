# -*- coding: utf-8 -*-
"""패키지에 자료를 붙인다.

KCHS 원자료는 12.2GB 라 클라우드에 한 번에 올리기 어렵고, 분석이 실제로 읽는 것은
조화가 끝난 parquet 이다. 그래서 세 층으로 나눈다.
  · 처리자료(parquet)  - 이것만 있으면 70번 이후 전 과정을 다시 돌릴 수 있다
  · NHANES 원자료      - 105MB, 공개 자료라 그대로 넣는다
  · KCHS 원자료        - 목록과 해시만 넣고 파일은 따로 전달한다
"""
import hashlib, io, os, shutil, sys
sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"C:/Users/rudtn/Desktop/claude code/지역사회건강"
OUT = f"{ROOT}/제출_클로드코드"

def mk(p): os.makedirs(f"{OUT}/{p}", exist_ok=True); return f"{OUT}/{p}"
def mb(p): return os.path.getsize(p) / 1048576

# ── 처리자료: 분석이 실제로 읽는 것 ────────────────────────────────────
d = mk("06_자료/처리자료_parquet")
n = 0
for f in sorted(os.listdir(f"{ROOT}/data/processed")):
    if f.endswith(".parquet"):
        shutil.copy2(f"{ROOT}/data/processed/{f}", f"{d}/{f}"); n += 1
print(f"처리자료 {n}개 · {sum(mb(f'{d}/{x}') for x in os.listdir(d)):.0f}MB")

# ── NHANES 원자료: 공개 자료 ───────────────────────────────────────────
src = f"{ROOT}/data/raw/nhanes"
if os.path.isdir(src):
    shutil.copytree(src, f"{OUT}/06_자료/NHANES_원자료")
    s = sum(os.path.getsize(os.path.join(r, x)) for r, _, fs in os.walk(f"{OUT}/06_자료/NHANES_원자료") for x in fs)
    print(f"NHANES 원자료 {sum(len(fs) for _,_,fs in os.walk(src))}개 · {s/1048576:.0f}MB")

# ── KCHS 원자료: 목록만 (12.2GB, 별도 전달) ────────────────────────────
rows = ["# KCHS 원자료 목록", "",
        "12.2GB 라 이 패키지에 넣지 않았다. 아래 목록대로 별도 전달한다.",
        "자료이용협약상 연구팀 밖으로는 재배포할 수 없다.", "",
        "| 연도 | 파일 | 크기(MB) |", "|---|---|---|"]
tot = cnt = 0
base = f"{ROOT}/data/raw/kchs"
for yr in sorted(os.listdir(base)):
    p = f"{base}/{yr}"
    if not os.path.isdir(p): continue
    for r, _, fs in os.walk(p):
        for f in sorted(fs):
            fp = os.path.join(r, f)
            sz = os.path.getsize(fp) / 1048576
            if sz < 1: continue
            rows.append(f"| {yr} | {os.path.relpath(fp, base)} | {sz:.0f} |")
            tot += sz; cnt += 1
rows += ["", f"1MB 이상 {cnt}개 · 합계 {tot/1024:.1f}GB"]
io.open(f"{OUT}/06_자료/KCHS_원자료_목록.md", "w", encoding="utf8").write("\n".join(rows))
print(f"KCHS 원자료 목록: {cnt}개 · {tot/1024:.1f}GB (파일은 별도 전달)")

t = sum(os.path.getsize(os.path.join(r, x)) for r, _, fs in os.walk(OUT) for x in fs)
print(f"\n패키지 합계 {t/1048576:.0f}MB")
