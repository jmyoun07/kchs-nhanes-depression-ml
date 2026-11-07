# -*- coding: utf-8 -*-
"""부록 표 번호가 본문 등장 순서와 어긋나면 실패하도록 검사를 추가한다.

JMIR 규정: "Each appendix must number items as S1, S2, S3... in order of appearance."
오늘 이 순서를 맞추는 데 세 단계와 한 번의 되돌리기가 들었다. 다시 어긋나면 바로 알아야 한다.
"""
import io, os, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")
p = "92_crossref_audit.py"
s = io.open(p, encoding="utf8").read()
if "등장 순서" in s and "_body_order" in s:
    print("이미 적용됨"); raise SystemExit

GUARD = '''
# 부록 표는 본문 등장 순서로 번호를 매긴다(JMIR 규정). 어긋나면 심사자가 바로 본다.
_body_order, _seen_t = [], set()
for _m in re.finditer(r"Table S(\\d+)", M.split("## Figure Legends")[0]):
    _k = int(_m.group(1))
    if _k not in _seen_t: _seen_t.add(_k); _body_order.append(_k)
need(_body_order == sorted(_body_order),
     f"부록 표 번호가 본문 등장 순서와 다름: {_body_order} (S1 부터 차례로 나와야 한다)")

# 부록 안에서도 표가 번호 순으로 놓여야 한다
_phys = [int(x) for x in re.findall(r"(?m)^#{2,4}\\s+Table S(\\d+)\\.", "\\n".join(_asrc))]
need(_phys == sorted(_phys), f"부록 안 표 배치 순서가 번호순이 아님: {_phys}")

'''
a = 'print(f"[6] 부록 구조 검사'
assert a in s, "앵커 없음"
io.open(p, "w", encoding="utf8").write(s.replace(a, GUARD.lstrip("\n") + a, 1))
compile(io.open(p, encoding="utf8").read(), p, "exec")
ctl = sorted({hex(ord(c)) for c in io.open(p, encoding="utf8").read() if ord(c) < 32 and c not in "\n\t"})
print(f"92_crossref_audit.py: 표 순서 가드 추가 · 제어문자 {ctl or '없음'}")
