# -*- coding: utf-8 -*-
"""splice 의 끝 앵커가 '다음 표 번호' 를 가리키던 것을 '다음 제목' 으로 바꾼다.

표를 재배열하면 어떤 표 뒤에 어떤 표가 오는지가 바뀐다. 끝 앵커가 다음 표 이름이면
재배열할 때마다 깨진다. 실제로 필요한 것은 '이 표 덩어리의 끝' 이므로 다음 제목까지로 잡는다.
"""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")

p = "61_sync_numbers.py"
s = io.open(p, encoding="utf8").read()

# splice 가 "\n#" 를 받으면 start 이후 첫 제목까지로 본다
OLD = '''    if start not in s or end not in s: raise SystemExit(f"[동기화 실패] 구간 앵커 없음 in {path}: {start[:50]!r}")
    i = s.index(start); j = s.index(end, i)'''
NEW = '''    if start not in s: raise SystemExit(f"[동기화 실패] 시작 앵커 없음 in {path}: {start[:50]!r}")
    i = s.index(start)
    if end == "\\n#":                      # '다음 제목까지' - 표 재배열에 흔들리지 않는다
        m = re.search(r"\\n#{1,6} ", s[i:])
        if not m: raise SystemExit(f"[동기화 실패] {path}: {start[:40]!r} 뒤에 제목이 없다")
        j = i + m.start()
    else:
        if end not in s: raise SystemExit(f"[동기화 실패] 끝 앵커 없음 in {path}: {end[:50]!r}")
        j = s.index(end, i)'''
assert OLD in s, "splice 본문 앵커 없음"
s = s.replace(OLD, NEW, 1)

# 표 덩어리를 끝내는 앵커를 모두 '다음 제목' 으로
n = 0
def sub(m):
    global n; n += 1
    return m.group(1) + '"\\n#"'
s = re.sub(r'(splice\("appendix\d_[a-z]+\.md",\s*"[^"]*",\s*)"\\n#{1,4} Table S\d+\."', sub, s)
io.open(p, "w", encoding="utf8").write(s)
compile(io.open(p, encoding="utf8").read(), p, "exec")
print(f"61_sync: splice 끝 앵커 {n}곳을 '다음 제목' 으로 · 표 재배열에 안 흔들림")

# ── 60_verify: 옮겨간 표를 찾는 문서 이름을 맞춘다 ──────────────────────
v = "60_verify_manuscript.py"
t = io.open(v, encoding="utf8").read()
k = 0
for a, b in [('("appendix4",), f"oracle table', '("appendix3",), f"oracle table'),
             ('need(f"| {r.learner} | {R(r.oracle_AUROC)} | {R(r.CI_lo)}-{R(r.CI_hi)} |" in A4',
              'need(f"| {r.learner} | {R(r.oracle_AUROC)} | {R(r.CI_lo)}-{R(r.CI_hi)} |" in A3'),
             ('f"[appendix4] 표 S3 보정 구간 불일치', 'f"[appendix3] 표 S3 보정 구간 불일치')]:
    if a in t: t = t.replace(a, b); k += 1
io.open(v, "w", encoding="utf8").write(t)
compile(io.open(v, encoding="utf8").read(), v, "exec")
print(f"60_verify: 옮겨간 표를 찾는 위치 {k}곳 수정")
