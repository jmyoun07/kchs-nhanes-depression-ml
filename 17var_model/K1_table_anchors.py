# -*- coding: utf-8 -*-
"""1단계: 표를 갈아끼우는 splice 를 '표 자체' 로만 잡게 한다.

지금은 끝 앵커가 '다음 표 이름'(\\n## Table S5. 같은 것)이다. 표 순서를 바꾸면 전부 깨진다.
표는 머리글 행부터 파이프로 시작하는 줄이 끝나는 곳까지다. 바깥을 볼 이유가 없다.

이 단계에서는 번호도 순서도 건드리지 않는다. 검증이 그대로 통과해야 한다.
"""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")
p = "61_sync_numbers.py"
s = io.open(p, encoding="utf8").read()

HELPER = '''
def splice_table(path, header, new):
    """표 머리글 행부터 표가 끝나는 데까지만 바꾼다.

    '다음 표 이름' 같은 바깥 앵커를 쓰면 표 순서를 바꿀 때마다 전부 깨진다.
    표의 경계는 표 안에서 정해진다: 파이프로 시작하는 줄이 이어지는 동안이 표다.
    표 아래 각주는 이 구간 밖이므로 자동으로 보존된다.
    """
    s = rd(path)
    if header not in s:
        raise SystemExit(f"[동기화 실패] 표 머리글 없음 in {path}: {header[:50]!r}")
    i = s.index(header)
    rest = s[i:].split(chr(10))
    k = 0
    while k < len(rest) and rest[k].lstrip().startswith("|"): k += 1
    j = i + len(chr(10).join(rest[:k]))
    wr(path, s[:i] + new.rstrip(chr(10)) + s[j:])
    print(f"  {path}: 표 재생성 ({header[:38]}… {k}행)")

'''
anchor = "# ══════════════ [1] 원고"
assert anchor in s and "def splice_table" not in s, "앵커 없음 또는 이미 적용됨"
s = s.replace(anchor, HELPER.lstrip("\n") + anchor, 1)

# 표 머리글로 시작하는 splice 를 splice_table 로 바꾼다(끝 앵커는 버린다)
PAT = re.compile(r'splice\((("[^"]+\.md"),\s*("\|[^"]*")),\s*"(?:[^"\\]|\\.)*",\s*(.+?)\)\s*$', re.M)
n = 0
def rep(m):
    global n
    if "manuscript_17var.md" in m.group(2) or "model_card" in m.group(2):
        return m.group(0)                      # 원고·모델카드의 표는 뒤 문장까지 함께 재생성한다
    n += 1
    return f"splice_table({m.group(2)}, {m.group(3)}, {m.group(4)})"
s = PAT.sub(rep, s)
io.open(p, "w", encoding="utf8").write(s)
compile(io.open(p, encoding="utf8").read(), p, "exec")
print(f"1단계: 부록 표 splice {n}곳을 표 경계 기준으로 전환")
