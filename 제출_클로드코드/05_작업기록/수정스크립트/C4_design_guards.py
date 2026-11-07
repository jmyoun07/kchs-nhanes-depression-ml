# -*- coding: utf-8 -*-
"""사용자가 지시한 서식 규칙을 60_verify_manuscript.py 에 검사로 박는다.

눈으로 확인하면 다음 편집에서 조용히 되돌아간다. 규칙으로 만들어 두면 못 돌아간다.
지시 내용
  · 표 캡션은 한 문장, 짧게        · 표 각주는 약어 하나에 알파벳 하나
  · 초록 체크박스 금지             · 그림 이미지에 제목 금지
  · 그림 캡션도 짧게               · 패널은 문자만, 부제목 없음
"""
import io, os, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")

p = "60_verify_manuscript.py"
s = io.open(p, encoding="utf8").read()
if "서식 규칙(캡션·각주·체크박스·그림 제목)" in s:
    print("이미 적용됨"); raise SystemExit

GUARD = '''
# ══════════════════════════════════════════════════════════════════════
# 서식 규칙(캡션·각주·체크박스·그림 제목) - 예시논문 기준. 되돌아가지 못하게 막는다.
# ══════════════════════════════════════════════════════════════════════
import re as _re

# 1) 표 캡션은 한 문장 분량. 예시논문은 15-30단어이고, 설명은 각주로 내린다.
for _m in _re.finditer(r"(?m)^\\*\\*(Table \\d+)\\.\\*\\* (.+)$", M):
    _w = len(_m.group(2).split())
    need(_w <= 40, f"[manuscript] {_m.group(1)} 캡션 {_w}단어 - 40단어 이하로 (설명은 각주로)")

# 2) 표 각주: 첫 줄은 총괄 주석, 나머지는 'ABBR: 풀이.' 한 줄에 하나
_lines = M.split(chr(10))
for _i, _l in enumerate(_lines):
    if not _l.startswith("|") or _i + 1 >= len(_lines): continue
    if _lines[_i + 1].startswith("|") or _lines[_i + 1].strip(): continue
    _j = _i + 2                                   # 표 끝 다음 빈 줄 다음이 각주
    _notes = []
    while _j < len(_lines) and _lines[_j].strip() and not _lines[_j].startswith(("|", "#", "**")):
        _notes.append(_lines[_j]); _j += 1
    if len(_notes) < 2: continue
    for _n in _notes[1:]:
        need(_re.match(r"^[A-Za-z][A-Za-z0-9+\\- ]{1,20}: ", _n),
             f"[manuscript] 표 각주가 'ABBR: 풀이.' 형식이 아님: {_n[:60]}")
    _first = len(_notes[0].split())
    need(_first <= 110, f"[manuscript] 표 총괄 각주 {_first}단어 - 110단어 이하로")

# 3) 초록 체크박스 금지 - 논문 부록에 쓰는 표기가 아니다
for _n, _t in ALLDOCS.items():
    for _sym in ("\\u2705", "\\u23f3", "\\u274c", "\\u2714"):
        need(_sym not in _t, f"[{_n}] 체크박스 기호 {_sym!r} 사용 - 위치를 문장으로 적는다")

# 4) 그림 이미지에 제목을 굽지 않는다 - 캡션이 이미 말한다
_figsrc = io.open(f"{D}/73_figures_primary2.py", encoding="utf8").read()
for _bad in ("set_title(", "suptitle("):
    need(_bad not in _figsrc,
         f"[73_figures_primary2.py] {_bad} 사용 - 이미지에 제목이 박힌다(캡션이 말한다)")

# 5) 그림 캡션도 짧게. 패널은 'A: 설명; B: 설명' 형식으로 캡션이 설명한다.
_leg = M.split("## Figure Legends")[-1].split(chr(10) + "## ")[0]
for _m in _re.finditer(r"Figure (\\d+): (.+?)(?=\\s*Figure \\d+:|\\s*Tables:|$)", _leg, _re.S):
    _w = len(_m.group(2).split())
    need(_w <= 95, f"[manuscript] Figure {_m.group(1)} 캡션 {_w}단어 - 95단어 이하로")
need(_re.search(r"Figure \\d+: [^;]*?\\bA: ", _leg), "[manuscript] 패널 그림 캡션에 'A: ' 형식이 없다")
need(True, "서식 규칙 검사 완료")

'''

anchor = 'print("="*88)'
if anchor not in s:
    anchor = [l for l in s.split("\n") if l.startswith("print(") and "검증 항목" in l][0]
i = s.index(anchor)
io.open(p, "w", encoding="utf8").write(s[:i] + GUARD.replace("\\\\", "\\") + "\n" + s[i:])
compile(io.open(p, encoding="utf8").read(), p, "exec")
print("60_verify_manuscript.py: 서식 규칙 5종을 검사로 추가")
