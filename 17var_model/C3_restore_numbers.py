# -*- coding: utf-8 -*-
"""축소하면서 빠진 수치 문장 3건을 제자리에 돌려놓는다.

 ① 외부 코호트의 가중 유병률 - 원고에 되돌린다. 정밀도 제약을 읽는 근거이고
    61_sync_numbers.py 가 이 문장에 숫자를 주입한다(패턴이 없으면 동기화가 실패한다).
 ② 동률 판정에 실제로 쓰인 두 보정기울기 - 재현용 상세이므로 Part D 로.
    선택 '규칙' 은 본문에 남아 있다(TRIPOD 12c).
 ③ 부트스트랩 대 설계기반 표준오차 대조 - 이미 Part D 로 옮겨갔다. 검사 위치만 고친다.
"""
import io, os, sys
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")

def R(v, nd=3):
    return f"{Decimal(str(v)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP):.{nd}f}"

# ── ① 원고: 가중 유병률 복원 ─────────────────────────────────────────────
p = "docs/manuscript_17var.md"
s = io.open(p, encoding="utf8").read()
a = "The external cohorts (728 events pre, 661 post) are the binding constraint on precision;"
b = ("The external cohorts (728 events pre, 661 post; weighted prevalence 8.4% and 12.2%) are the binding "
     "constraint on precision;")
assert a in s, "가중 유병률 앵커 없음"
io.open(p, "w", encoding="utf8").write(s.replace(a, b, 1))
print("원고: 외부 코호트 가중 유병률 복원 (61_sync 주입 대상)")

# ── ② Part D: 동률 판정에 쓰인 실제 기울기 ───────────────────────────────
tb = pd.read_csv("results/sap_tiebreak.csv").set_index("algorithm")
lg, rf = R(tb.loc["LightGBM", "cal_slope"]), R(tb.loc["Random forest", "cal_slope"])
q = "docs/appendix5_methods.md"
d = io.open(q, encoding="utf8").read()
anchor = "### Model development, tuning, and resampling detail"
assert anchor in d, "Part D 모델개발 절 없음"
sent = (f"\n\nApplying the tie-break rule, the two candidate slopes differ by {R(abs(tb.loc['Random forest','cal_slope'] - tb.loc['LightGBM','cal_slope']), 2)} "
        f"({lg} for LightGBM against {rf} for random forest), which is not a close call, and the rule selects "
        f"LightGBM.")
i = d.index(anchor) + len(anchor)
j = d.find("\n### ", i)
j = len(d) if j == -1 else j
io.open(q, "w", encoding="utf8").write(d[:j].rstrip() + sent + d[j:])
print(f"Part D: 동률 판정 기울기 {lg} / {rf} 기록")

# ── ③ 검사 위치를 옮긴 곳으로 ────────────────────────────────────────────
v = "60_verify_manuscript.py"
t = io.open(v, encoding="utf8").read()
for old, new, why in [
    ('("manuscript",), "bootstrap validation pre")',
     '("appendix5",), "bootstrap validation pre")',
     "부트스트랩 대 설계기반 SE 대조는 Part D 로 이전"),
    ('       ("manuscript",), "SAP tie-break slopes")',
     '       ("appendix5",), "SAP tie-break slopes")',
     "동률 판정에 쓰인 실제 기울기는 Part D 로 이전(규칙은 본문에 남음)"),
]:
    assert old in t, f"검사 앵커 없음: {old[:50]}"
    t = t.replace(old, new, 1); print(f"60_verify: {why}")
io.open(v, "w", encoding="utf8").write(t)
compile(io.open(v, encoding="utf8").read(), v, "exec")
