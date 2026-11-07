# -*- coding: utf-8 -*-
"""칸 너비 계산을 제대로 고친다.

두 가지 결함이 있었다.
 ① 글자폭을 일률 0.085인치로 봤다. Times 는 숫자가 0.5em, 마침표·괄호는 0.25-0.33em 이라
    '(0.827-0.846)' 을 13자 = 1.11인치로 과대평가했다. 실제로는 0.76인치다.
    과대평가하면 최소너비 합계가 지면을 넘고, 그 뒤 배분이 무너진다.
 ② 최소너비 합계가 지면을 넘으면 배분 루프가 중간에 break 하면서
    일부 칸이 0.10인치로 남았다. 넘칠 때는 최소치를 비례 축소해야 한다.
"""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")

NEW = '''def _adv(ch):
    """Times New Roman 의 글자 폭(em). 숫자는 0.5em 고정폭이다."""
    if ch.isdigit(): return 0.500
    if ch in ".,;:'": return 0.250
    if ch in "-()/": return 0.320
    if ch in "+=<>": return 0.564
    if ch == "%": return 0.833
    if ch == " ": return 0.250
    if ch.isupper(): return 0.722
    return 0.480

def text_width(s, pt=10.0):
    """문자열이 차지하는 폭(인치)."""
    return sum(_adv(c) for c in s) * pt / 72.0

def auto_widths(header, body, total=6.5, pad=0.17):
    """칸 너비를 내용으로 정한다. 줄바꿈 단위가 잘리지 않을 만큼은 보장한다.

    워드는 공백·'/'·하이픈 뒤에서 줄을 바꾼다. 다만 수치 구간(0.814-0.837)에서
    갈라지면 보기 흉하므로 그 하이픈은 끊기지 않는 것으로 본다.
    """
    _SEP = chr(1)
    n = len(header)
    cols = [[header[i]] + [(r[i] if i < len(r) else "") for r in body] for i in range(n)]
    wt, mins = [], []
    for c in cols:
        avg = sum(text_width(x) for x in c) / max(1, len(c))
        longest = 0.0
        for x in c:
            _x = re.sub(r"(?<=\\d)-(?=\\d)", _SEP, x)
            for w in re.split(r"[\\s/\\-]+", _x):
                if w: longest = max(longest, text_width(w.replace(_SEP, "-")))
        wt.append(max(avg * 0.75, longest, 0.1))
        mins.append(longest + pad)
    if sum(mins) > total:                      # 지면이 모자라면 최소치를 비례 축소한다
        k = total / sum(mins); mins = [m * k for m in mins]
    s = sum(wt); w = [total * x / s for x in wt]
    for _ in range(n * 2):
        small = [i for i, x in enumerate(w) if x < mins[i] - 1e-9]
        if not small: break
        need = sum(mins[i] - w[i] for i in small)
        big = [i for i in range(n) if w[i] > mins[i] + 1e-9]
        pool = sum(w[i] - mins[i] for i in big)
        if not big or pool < need - 1e-9:      # 더 뺄 곳이 없으면 최소치로 균등 정착
            return [m * total / sum(mins) for m in mins]
        for i in small: w[i] = mins[i]
        for i in big: w[i] -= need * (w[i] - mins[i]) / pool
    return w

'''

for p, ind in (("44_make_docx.py", ""), ("45_make_appendix_docx.py", "    ")):
    s = io.open(p, encoding="utf8").read()
    a = s.index(ind + "def auto_widths(")
    b = s.index(ind + "def set_col_widths(")
    blk = NEW if ind == "" else "\n".join((ind + l if l.strip() else l) for l in NEW.split("\n"))
    io.open(p, "w", encoding="utf8").write(s[:a] + blk + s[b:])
    compile(io.open(p, encoding="utf8").read(), p, "exec")
    print(f"{p}: auto_widths 재작성 (글자폭 모델 + 최소너비 초과 시 비례 축소)")
