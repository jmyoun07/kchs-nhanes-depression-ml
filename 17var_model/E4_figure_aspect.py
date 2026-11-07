# -*- coding: utf-8 -*-
"""작업용 파일에서 손으로 맞추신 그림 비율을 원본 도형 비율로 옮긴다.

작업용에서는 가로·세로를 따로 늘여서 이미지가 왜곡돼 있었다(예: 그림 5 를
가로 1.04배·세로 0.65배). 같은 겉모양을 왜곡 없이 내려면 그리는 단계에서
figsize 비율 자체를 바꿔야 한다.

목표 가로:세로 비 (작업용에서 측정)
  1  2.13   유지        5  2.06   1.29 에서
  2  2.23   유지        6  1.19   0.85 에서
  3  1.36   1.11 에서   7  1.96   3.09 에서
  4  1.96   1.69 에서
삽입 폭은 6.1 에서 6.3 인치로(작업용 평균 6.31).
"""
import io, os, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")

p = "73_figures_primary2.py"
s = io.open(p, encoding="utf8").read()

REPL = [
    # 그림 3 ROC: 1.11 -> 1.36
    ("fig, ax = plt.subplots(figsize=(6.2, 5.6))",
     "fig, ax = plt.subplots(figsize=(6.8, 5.0))"),
    # 그림 4 민감도: 1.69 -> 1.96
    ("fig, ax = plt.subplots(figsize=(8.4, 5.0))",
     "fig, ax = plt.subplots(figsize=(8.4, 4.3))"),
    # 그림 5 SHAP: 1.29 -> 2.06
    ("fig, ax = plt.subplots(figsize=(7.2, 5.6))",
     "fig, ax = plt.subplots(figsize=(8.2, 4.0))"),
    # 그림 6 하위군: 0.85 -> 1.19. 행 간격과 글자를 함께 줄여야 이름이 겹치지 않는다
    ("fig, ax = plt.subplots(figsize=(8.0, 0.30*len(fa)+1.6))",
     "fig, ax = plt.subplots(figsize=(8.0, 0.185*len(fa)+1.6))"),
    ('ax.set_yticklabels([f"{r.cohort} - {str(r.subgroup).split(\': \')[-1]}" for _, r in fa.iterrows()], fontsize=7.6)',
     'ax.set_yticklabels([f"{r.cohort} - {str(r.subgroup).split(\': \')[-1]}" for _, r in fa.iterrows()], fontsize=6.4)'),
    # 그림 7 결정곡선: 3.09 -> 1.96
    ("fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)",
     "fig, axes = plt.subplots(1, 3, figsize=(13, 6.6), sharey=True)"),
]
for a, b in REPL:
    assert a in s, f"앵커 없음: {a[:66]}"
    s = s.replace(a, b, 1)
io.open(p, "w", encoding="utf8").write(s)
compile(io.open(p, encoding="utf8").read(), p, "exec")
print("73_figures_primary2.py: 그림 3·4·5·6·7 비율 조정")

# 삽입 폭
q = "44_make_docx.py"
d = io.open(q, encoding="utf8").read()
a = "pic.add_run().add_picture(path, width=Inches(6.1))"
b = "pic.add_run().add_picture(path, width=Inches(6.3))   # 작업용 파일에서 맞추신 폭"
assert a in d, "삽입 폭 앵커 없음"
io.open(q, "w", encoding="utf8").write(d.replace(a, b, 1))
compile(io.open(q, encoding="utf8").read(), q, "exec")
print("44_make_docx.py: 그림 삽입 폭 6.1 -> 6.3 인치")
