# -*- coding: utf-8 -*-
"""수치 열은 같은 너비로 맞춘다.

작업용 파일에서 사용자가 표 2의 데이터 열 네 개를 1.083인치로 균일하게 맞춰 놓았다.
내용 기준으로만 계산하면 1.059/1.117 처럼 들쭉날쭉해서 보기 나쁘다.
같은 성격의 수치 열은 같은 폭이어야 표가 정돈돼 보인다.
"""
import io, os, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")

OLD = """        for i in small: w[i] = mins[i]
            for i in big: w[i] -= need * (w[i] - mins[i]) / pool
        return w"""

ADD = '''
    # 같은 성격의 수치 열은 같은 폭으로 맞춘다(작업용 파일에서 손으로 하신 조정).
    # 내용대로만 계산하면 1.06/1.12 처럼 들쭉날쭉해서 표가 정돈돼 보이지 않는다.
    def _numericish(col):
        vals = [x for x in col[1:] if x.strip()]
        if len(vals) < 2: return False
        hit = sum(1 for v in vals if re.fullmatch(r"[+-]?[\\d.,]+(\\s*\\([^)]*\\))?", v.strip()))
        return hit / len(vals) >= 0.7
    _num = [i for i, c in enumerate(cols) if _numericish(c)]
    if len(_num) >= 2:
        _avg = sum(w[i] for i in _num) / len(_num)
        if _avg >= max(mins[i] for i in _num):        # 균일화해도 아무 칸도 좁아지지 않을 때만
            for i in _num: w[i] = _avg
    return w
'''

for p, ind in (("44_make_docx.py", ""), ("45_make_appendix_docx.py", "    ")):
    s = io.open(p, encoding="utf8").read()
    a = s.index(ind + "def auto_widths(")
    b = s.index(ind + "def set_col_widths(")
    blk = s[a:b]
    tail = ind + "return w\n"
    assert blk.rstrip().endswith("return w"), f"{p}: auto_widths 끝이 'return w' 가 아님"
    i = blk.rindex(tail)
    add = "\n".join((ind + l[4:] if l.strip() else l) for l in ADD.split("\n"))
    io.open(p, "w", encoding="utf8").write(s[:a] + blk[:i] + add.lstrip("\n") + "\n" + s[b:])
    compile(io.open(p, encoding="utf8").read(), p, "exec")
    print(f"{p}: 수치 열 균일화 규칙 추가")
