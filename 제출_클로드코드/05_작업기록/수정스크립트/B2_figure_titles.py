# -*- coding: utf-8 -*-
"""73_figures_primary2.py 에서 이미지에 박힌 제목을 걷어낸다.

캡션이 이미 그림 제목을 말하므로 이미지 안에 다시 쓰지 않는다(예시파일과 동일).
패널이 여럿인 그림은 부제목을 버리고 왼쪽 위에 문자만 남긴다; 그 문자가 무엇을
가리키는지는 캡션이 'A: ...; B: ...' 형식으로 말한다.
"""
import io, os, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")
p = "73_figures_primary2.py"
s = io.open(p, encoding="utf8").read()

# 패널 문자를 축 바깥 왼쪽 위에 찍는 도우미를 save_fig 앞에 심는다
helper = '''def panel_letter(ax, ch):
    """패널 문자만 축 바깥 왼쪽 위에 찍는다(부제목 없음 - 캡션이 설명한다)."""
    ax.text(-0.09, 1.05, ch, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="bottom", ha="left")


'''
anchor = "def save_fig("
assert anchor in s, "save_fig 앵커 없음"
s = s.replace(anchor, helper + anchor, 1)

REPL = [
    # ── Figure 1. 표본 특성 (2패널) ──
    ('    ax.set_title(title, fontsize=10, fontweight="bold"); ax.set_xlabel(xlabel, fontsize=9)',
     '    panel_letter(ax, title); ax.set_xlabel(xlabel, fontsize=9)'),
    ('grouped(axes[0], pct, "A. Prevalence / proportion", "Weighted %")',
     'grouped(axes[0], pct, "A", "Weighted %")'),
    ('grouped(axes[1], cont, "B. Means", "Weighted mean")',
     'grouped(axes[1], cont, "B", "Weighted mean")'),
    ('fig.suptitle("Weighted sample characteristics across cohorts (available cases)", fontsize=11, fontweight="bold", y=1.0)\n',
     ''),
    # ── Figure 2. 보정곡선 (2패널) ──
    ('for ax, (title, use_recal) in zip(axes, [("A. As transported", False), ("B. After intercept/slope recalibration", True)]):',
     'for ax, (title, use_recal) in zip(axes, [("A", False), ("B", True)]):'),
    ('    ax.set_title(title, fontsize=10, fontweight="bold"); ax.set_xlabel("Predicted probability", fontsize=9)',
     '    panel_letter(ax, title); ax.set_xlabel("Predicted probability", fontsize=9)'),
    ('fig.suptitle("Calibration, weighted (deciles of predicted risk)", fontsize=11, fontweight="bold")\n',
     ''),
    # ── Figure 3. ROC (단일 패널) ──
    ('ax.set_title("Weighted ROC curves", fontsize=11, fontweight="bold"); ax.grid(alpha=0.25)',
     'ax.grid(alpha=0.25)'),
    # ── Figure 4. 민감도 (단일 패널) ──
    ('ax.set_title("Sensitivity analyses - weighted AUROC across specifications", fontsize=10.5, fontweight="bold")\n',
     ''),
    # ── Figure 5. SHAP (단일 패널) ──
    ('ax.set_title("Global predictor importance (development fitting partition)", fontsize=10.5, fontweight="bold")\n',
     ''),
    # ── Figure 6. 하위군 (단일 패널) ──
    ('ax.set_title("Subgroup discrimination with cluster-bootstrap intervals", fontsize=10.5, fontweight="bold")\n',
     ''),
    # ── Figure 7. 결정곡선 (3패널) ──
    ('for ax, (nm, d, yv, p, w, c) in zip(axes, SETS):',
     'for ax, _ch, (nm, d, yv, p, w, c) in zip(axes, "ABC", SETS):'),
    ('    ax.set_title(nm, fontsize=10, fontweight="bold"); ax.set_xlabel("Threshold probability", fontsize=9)',
     '    panel_letter(ax, _ch); ax.set_xlabel("Threshold probability", fontsize=9)'),
    ('fig.suptitle("Decision-curve analysis (weighted; external cohorts after cross-fit recalibration)",\n             fontsize=11, fontweight="bold")\n',
     ''),
]
for a, b in REPL:
    assert a in s, f"앵커 없음: {a[:70]}"
    s = s.replace(a, b, 1)

io.open(p, "w", encoding="utf8").write(s)
compile(io.open(p, encoding="utf8").read(), p, "exec")

left = [l for l in s.split("\n") if "set_title(" in l or "suptitle(" in l]
print(f"제목 제거 완료 - 남은 set_title/suptitle: {len(left)}건")
for l in left:
    print("   ", l.strip()[:100])
