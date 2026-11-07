# -*- coding: utf-8 -*-
"""패널 문자(A·B·C)를 각 패널의 제목으로 바꾼다.

예시논문(Figure 3)은 패널마다 무엇을 그린 것인지 작은 제목으로 적는다.
그림 전체 제목은 여전히 넣지 않는다 - 그건 캡션이 말한다.
"""
import io, os, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")

p = "73_figures_primary2.py"
s = io.open(p, encoding="utf8").read()

OLD = '''def panel_letter(ax, ch):
    """패널 문자만 축 바깥 왼쪽 위에 찍는다(부제목 없음 - 캡션이 설명한다)."""
    ax.text(-0.09, 1.05, ch, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="bottom", ha="left")'''
NEW = '''def panel_title(ax, text):
    """패널마다 무엇을 그린 것인지 작은 제목을 단다(예시논문 Figure 3 방식).

    그림 전체 제목은 넣지 않는다. 그건 캡션이 말한다.
    """
    ax.set_title(text, fontsize=9.5, pad=6)'''
assert OLD in s, "panel_letter 정의 앵커 없음"
s = s.replace(OLD, NEW, 1)

REPL = [
    # Figure 1 (표본 특성)
    ('    panel_letter(ax, title); ax.set_xlabel(xlabel, fontsize=9)',
     '    panel_title(ax, title); ax.set_xlabel(xlabel, fontsize=9)'),
    ('grouped(axes[0], pct, "A", "Weighted %")',
     'grouped(axes[0], pct, "Prevalence and proportion", "Weighted %")'),
    ('grouped(axes[1], cont, "B", "Weighted mean")',
     'grouped(axes[1], cont, "Means", "Weighted mean")'),
    # Figure 2 (보정곡선)
    ('for ax, (title, use_recal) in zip(axes, [("A", False), ("B", True)]):',
     'for ax, (title, use_recal) in zip(axes, [("As transported", False),\n'
     '                                         ("After intercept and slope recalibration", True)]):'),
    ('    panel_letter(ax, title); ax.set_xlabel("Predicted probability", fontsize=9)',
     '    panel_title(ax, title); ax.set_xlabel("Predicted probability", fontsize=9)'),
    # Figure 7 (결정곡선) - 패널 이름은 코호트명이다
    ('for ax, _ch, (nm, d, yv, p, w, c) in zip(axes, "ABC", SETS):',
     'for ax, (nm, d, yv, p, w, c) in zip(axes, SETS):'),
    ('    panel_letter(ax, _ch); ax.set_xlabel("Threshold probability", fontsize=9)',
     '    panel_title(ax, nm); ax.set_xlabel("Threshold probability", fontsize=9)'),
]
for a, b in REPL:
    assert a in s, f"앵커 없음: {a[:70]}"
    s = s.replace(a, b, 1)

io.open(p, "w", encoding="utf8").write(s)
compile(io.open(p, encoding="utf8").read(), p, "exec")
print("73_figures_primary2.py: 패널 문자 -> 패널 제목")
print("  Figure 1: 'Prevalence and proportion' / 'Means'")
print("  Figure 2: 'As transported' / 'After intercept and slope recalibration'")
print("  Figure 7: 코호트명 3개")
print(f"  suptitle 잔존: {s.count('suptitle(')}건 (0이어야 함)")
