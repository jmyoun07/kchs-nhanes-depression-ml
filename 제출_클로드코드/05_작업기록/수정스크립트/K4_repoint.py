# -*- coding: utf-8 -*-
"""3단계 마무리: 표가 옮겨간 파일로 생성기·검사기의 위치를 맞춘다.

표를 전부 'Supplementary Results'(appendix3) 로 모았으므로, 천장분석 파일(appendix4)에
표를 찾으러 가는 코드가 남아 있으면 전부 실패한다. 표를 다루는 것만 옮긴다.
"""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")

# ── 61_sync: 표를 갈아끼우는 호출과 표 안 값을 고치는 patch ─────────────
p = "61_sync_numbers.py"
s = io.open(p, encoding="utf8").read()
n = 0
for a, b in [
    ('splice_table("appendix4_ceiling.md", "| Learner | Oracle AUROC (5-fold CV) | 95% CI |"',
     'splice_table("appendix3_utility.md", "| Learner | Oracle AUROC (5-fold CV) | 95% CI |"'),
    ('splice_table("appendix4_ceiling.md", "| Cohort | Probabilities | Calibration slope"',
     'splice_table("appendix3_utility.md", "| Cohort | Probabilities | Calibration slope"'),
]:
    if a in s: s = s.replace(a, b, 1); n += 1

# 'Tuned to a triage-appropriate' 는 표가 아니라 해설 문단이다.
# 끝 앵커가 '다음 표 이름' 이라 재배열에 깨졌다. 다음 제목까지로 바꾼다.
a = 'splice("appendix3_utility.md", "Tuned to a triage-appropriate", "\\n## Table S8.", interp)'
b = 'splice("appendix3_utility.md", "Tuned to a triage-appropriate", "\\n#", interp)'
if a in s: s = s.replace(a, b, 1); n += 1

# splice 가 "\n#" 를 '다음 제목까지' 로 이해하게 한다
OLD = '''    if start not in s or end not in s: raise SystemExit(f"[동기화 실패] 구간 앵커 없음 in {path}: {start[:50]!r}")
    i = s.index(start); j = s.index(end, i)'''
NEW = '''    if start not in s: raise SystemExit(f"[동기화 실패] 시작 앵커 없음 in {path}: {start[:50]!r}")
    i = s.index(start)
    if end == "\\n#":                     # 다음 제목까지 - 표 순서를 바꿔도 흔들리지 않는다
        m = re.search(r"\\n#{1,6} ", s[i + 1:])
        if not m: raise SystemExit(f"[동기화 실패] {path}: {start[:40]!r} 뒤에 제목이 없다")
        j = i + 1 + m.start()
    else:
        if end not in s: raise SystemExit(f"[동기화 실패] 끝 앵커 없음 in {path}: {end[:50]!r}")
        j = s.index(end, i)'''
if OLD in s: s = s.replace(OLD, NEW, 1); n += 1
io.open(p, "w", encoding="utf8").write(s)
compile(io.open(p, encoding="utf8").read(), p, "exec")
print(f"61_sync: {n}곳 조정")

# ── 60_verify: 표를 찾는 문서 이름 ──────────────────────────────────────
v = "60_verify_manuscript.py"
t = io.open(v, encoding="utf8").read()
k = 0
for a, b in [('("appendix4",), f"oracle table', '("appendix3",), f"oracle table'),
             ('need(_cell in A4, f"[appendix4] 표 S3 보정 구간 불일치',
              'need(_cell in A3, f"[appendix3] 표 S3 보정 구간 불일치')]:
    if a in t: t = t.replace(a, b, 1); k += 1
io.open(v, "w", encoding="utf8").write(t)
compile(io.open(v, encoding="utf8").read(), v, "exec")
print(f"60_verify: 표 위치 {k}곳 조정")
