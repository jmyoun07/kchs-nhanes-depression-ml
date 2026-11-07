# -*- coding: utf-8 -*-
"""① 패널이 제목을 갖게 됐으므로 캡션의 'A: ...; B: ...' 를 없앤다.
   ② 뒷단(Acknowledgments 이후)은 쪽을 나누지 않고 이어 붙인다.
   ③ 위 두 가지에 맞춰 서식 검사를 갱신한다.
"""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")

# ── ① 캡션에서 패널 문자 설명 제거 ───────────────────────────────────────
p = "docs/manuscript_17var.md"
s = io.open(p, encoding="utf8").read()
PAIRS = [
    ("Figure 1: Weighted sample characteristics across the three cohorts. A: prevalences and "
     "proportions; B: means. Estimates",
     "Figure 1: Weighted sample characteristics across the three cohorts. Estimates"),
    ("cycles. A: the model as transported, without any updating; B: the same predictions after "
     "intercept-and-slope recalibration applied out-of-sample by 5-fold cross-fitting, applied to all "
     "three cohorts so that the panels are comparable. Points",
     "cycles, shown as transported and after intercept-and-slope recalibration applied out-of-sample "
     "by 5-fold cross-fitting; the recalibration is applied to all three cohorts so that the panels are "
     "comparable. Points"),
    ("The deployed internal model is the one shown in A.",
     "The deployed internal model is the one shown as transported."),
    ("strategies. A: internal KCHS hold-out, scored with the original probabilities; B: NHANES "
     "pre-pandemic; C: NHANES post-pandemic, the two external cohorts being scored with probabilities "
     "recalibrated out-of-sample by 5-fold cross-fitting grouped by primary sampling unit.",
     "strategies. The internal KCHS hold-out is scored with the original probabilities, and the two "
     "external cohorts with probabilities recalibrated out-of-sample by 5-fold cross-fitting grouped by "
     "primary sampling unit."),
]
for a, b in PAIRS:
    assert a in s, f"캡션 앵커 없음: {a[:60]}"
    s = s.replace(a, b, 1)
io.open(p, "w", encoding="utf8").write(s)
print("① 캡션 3건에서 'A: ...; B: ...' 제거")

# ── ② 뒷단은 쪽을 나누지 않는다 ──────────────────────────────────────────
q = "44_make_docx.py"
d = io.open(q, encoding="utf8").read()
old = '''            _h = para(txt, bold=True, color=HEAD_COLOR)
            if lvl == 2:                     # 큰 절은 새 쪽에서 시작한다
                _h.paragraph_format.page_break_before = True'''
new = '''            _h = para(txt, bold=True, color=HEAD_COLOR)
            # 본문 큰 절만 새 쪽에서 시작한다. Acknowledgments 이후의 뒷단은
            # 절마다 한 쪽씩 쓰면 낭비이므로 이어 붙인다.
            if lvl == 2 and txt in ("Abstract", "Introduction", "Methods", "Results", "Discussion"):
                _h.paragraph_format.page_break_before = True'''
assert old in d, "쪽나눔 앵커 없음"
io.open(q, "w", encoding="utf8").write(d.replace(old, new, 1))
compile(io.open(q, encoding="utf8").read(), q, "exec")
print("② 쪽 나눔을 Abstract·Introduction·Methods·Results·Discussion 으로 한정")

# ── ③ 서식 검사 갱신 ─────────────────────────────────────────────────────
v = "60_verify_manuscript.py"
t = io.open(v, encoding="utf8").read()
o1 = '''_figsrc = io.open(f"{D}/73_figures_primary2.py", encoding="utf8").read()
for _bad in ("set_title(", "suptitle("):
    need(_bad not in _figsrc,
         f"[73_figures_primary2.py] {_bad} 사용 - 이미지에 제목이 박힌다(캡션이 말한다)")'''
n1 = '''_figsrc = io.open(f"{D}/73_figures_primary2.py", encoding="utf8").read()
# 그림 전체 제목은 금지(캡션이 말한다). 패널 제목은 panel_title() 안에서만 허용한다.
need("suptitle(" not in _figsrc,
     "[73_figures_primary2.py] suptitle 사용 - 그림 전체 제목은 캡션이 말한다")
need(_figsrc.count("set_title(") == 1 and "def panel_title" in _figsrc,
     "[73_figures_primary2.py] set_title 은 panel_title() 안에서 한 번만 써야 한다")'''
assert o1 in t, "그림 제목 검사 앵커 없음"
t = t.replace(o1, n1, 1)

o2 = '''need(_re.search(r"Figure \\d+: [^;]*?\\bA: ", _leg), "[manuscript] 패널 그림 캡션에 'A: ' 형식이 없다")'''
n2 = '''# 패널은 그림 안에서 제목을 달므로, 캡션이 'A: ...' 로 다시 설명하지 않는다
need(not _re.search(r"Figure \\d+: [^;]*?\\bA: ", _leg),
     "[manuscript] 캡션에 'A: ' 가 남아 있다 - 패널 제목과 중복된다")'''
assert o2 in t, "패널 캡션 검사 앵커 없음"
t = t.replace(o2, n2, 1)
io.open(v, "w", encoding="utf8").write(t)
compile(io.open(v, encoding="utf8").read(), v, "exec")
print("③ 서식 검사 갱신: suptitle 금지 유지, panel_title 만 허용, 캡션의 'A:' 금지로 반전")
