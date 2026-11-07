# -*- coding: utf-8 -*-
"""92_crossref_audit.py 를 복구한다.

heredoc 으로 코드를 밀어 넣다가 백슬래시가 먹혀서 두 가지가 깨졌다.
  · 줄바꿈 이스케이프가 실제 줄바꿈이 되어 문자열이 끊겼다
  · \\b 가 백스페이스 문자(0x08)로 들어갔다 - 60_verify 의 제어문자 검사가 잡았다
같은 실수를 반복하지 않도록 이 파일에서 직접 고친다.
"""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")
p = "92_crossref_audit.py"
s = io.open(p, encoding="utf8").read()

ctl = sorted({hex(ord(c)) for c in s if ord(c) < 32 and c not in "\n\t"})
print(f"제어문자: {ctl or '없음'}")

lines = s.split("\n")
i = next(k for k, l in enumerate(lines) if l.startswith("# 부록 표는 모두 본문"))
j = next(k for k, l in enumerate(lines) if l.startswith('print(f"[6] 부록 구조 검사'))
print(f"깨진 구간: {i+1}~{j}행")

NEW = [
 "# 부록 표는 모두 원고 본문에서 인용돼야 한다.",
 "# JMIR 규정은 부록 항목을 '본문 등장 순서' 로 번호 매기라고 한다. 등장하지 않는 표는",
 "# 번호를 매길 근거가 없고, 심사자가 '왜 실려 있는지' 를 묻는 자리가 된다.",
 "_body_only = M.split('## Figure Legends')[0]",
 "for _n in _snum:",
 "    need(re.search(r'Table S' + str(_n) + r'(?![0-9])', _body_only) is not None,",
 "         f'Table S{_n} 이 원고 본문에서 인용되지 않는다')",
 "",
]
lines[i:j] = NEW
s = "\n".join(lines)
s = s.replace(chr(8), "\\b")

io.open(p, "w", encoding="utf8").write(s)
compile(io.open(p, encoding="utf8").read(), p, "exec")
ctl2 = sorted({hex(ord(c)) for c in io.open(p, encoding="utf8").read() if ord(c) < 32 and c not in "\n\t"})
print(f"복구 완료 · 남은 제어문자: {ctl2 or '없음'}")
