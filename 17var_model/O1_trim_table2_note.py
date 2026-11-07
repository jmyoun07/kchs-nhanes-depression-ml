# -*- coding: utf-8 -*-
"""표 2 각주 a 를 줄인다. 마크다운과 작업용_v3.docx 양쪽에 같은 내용을 넣는다.

각주가 긴 이유는 재보정 절차를 Methods 대신 여기서 설명하고 있어서였다.
확인해 보니 Methods 에 이미 다 있다 - 'out-of-sample intercept and slope updating,
which corrects calibration without re-estimating predictor effects' 와
'cross-fitted over five folds grouped by primary sampling unit'. 각주는 그것을 가리키면 된다.

v3 는 사용자가 쪽 나눔과 표 2 행 높이를 손으로 맞춰 놓은 파일이다. 재생성하면 그 작업이
사라지므로 문단 텍스트만 바꾸고 서식은 건드리지 않는다.
"""
import io, os, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")

OLD = ("CIs are from a rescaled stratified cluster bootstrap with 1000 replicates; a cell shown without an "
       "interval was not resampled. Recalibration is intercept-and-slope updating applied out-of-sample by "
       "5-fold cross-fitting, and predictor effects are not re-estimated; since each fold uses a separate "
       "calibration map, the recalibrated AUROC may differ from the original by up to 0.002 and is not the "
       "target of recalibration.")
NEW = ("CIs are from a rescaled stratified cluster bootstrap with 1000 replicates; a cell shown without an "
       "interval was not resampled. Recalibration is the cross-fitted intercept-and-slope update described in "
       "the Methods; each fold carries its own calibration map, hence the recalibrated AUROC may shift by up "
       "to 0.002 and is not what the update targets.")
print(f"각주 a: {len(OLD.split())} → {len(NEW.split())}단어")

# ── ① 마크다운 ──────────────────────────────────────────────────────────
p = "docs/manuscript_17var.md"
s = io.open(p, encoding="utf8").read()
assert OLD in s, "마크다운 앵커 없음"
io.open(p, "w", encoding="utf8").write(s.replace(OLD, NEW, 1))
print("① docs/manuscript_17var.md 수정")

# ── ② 사용자가 배치를 맞춰 둔 v3 는 문단 텍스트만 바꾼다 ────────────────
from docx import Document
V = r"C:/Users/rudtn/Desktop/claude code/지역사회건강/제출본/01_원고/작업용_v3.docx"
d = Document(V)
hit = 0
for par in d.paragraphs:
    if " ".join(par.text.split()).startswith("CIs are from a rescaled stratified cluster bootstrap"):
        runs = par.runs
        if not runs: continue
        runs[0].text = NEW                     # 첫 run 에 전체를 넣고
        for r in runs[1:]: r.text = ""         # 나머지는 비운다 - 서식은 그대로 남는다
        hit += 1
assert hit == 1, f"v3 에서 각주 문단 {hit}개 발견(1개여야 함)"
d.save(V)
print("② 작업용_v3.docx 수정 (쪽 나눔·행 높이는 그대로)")

# ── 확인 ────────────────────────────────────────────────────────────────
from docx.oxml.ns import qn
d2 = Document(V)
brk = sum(1 for x in d2.paragraphs if x._p.findall(".//" + qn("w:br") + '[@' + qn("w:type") + '="page"]'))
t2 = d2.tables[1]
hs = []
for r in t2.rows:
    tp = r._tr.find(qn("w:trPr")); h = tp.find(qn("w:trHeight")) if tp is not None else None
    hs.append(round(int(h.get(qn("w:val"))) / 1440, 3) if h is not None else None)
ok = any(" ".join(x.text.split()).startswith("CIs are from a rescaled stratified cluster bootstrap with 1000 replicates; a cell") for x in d2.paragraphs)
print(f"\nv3 확인 · 각주 반영 {ok} · 쪽 나눔 {brk}개 · 표2 행 높이 {hs}")
