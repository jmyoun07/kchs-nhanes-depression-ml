# -*- coding: utf-8 -*-
"""수정을 제출본 원고 docx 에 반영하고, 네 파일의 문서 속성에서 개인정보를 지운다.

제출본 원고는 손으로 쪽 나눔과 표 행 높이를 맞춰 둔 파일이라 재생성하지 않는다.
생성본(논문원고_우울ML_17변수.docx)과 문단을 대조해 달라진 것만 갈아끼운다.

문서 속성: 저자란을 비워 둔 논문인데 워드 속성에 실명이 박혀 있으면 익명 심사에서 신원이 샌다.
"""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")
from docx import Document
from docx.oxml.ns import qn

SUB = r"C:/Users/rudtn/Desktop/claude code/지역사회건강/제출본"
TARGET = f"{SUB}/01_원고/Manuscript.docx"

# ── ① 생성본과 대조해 달라진 문단만 반영 ────────────────────────────────
G = Document("논문원고_우울ML_17변수.docx")     # 방금 다시 만든 것 = 정답
T = Document(TARGET)                            # 배치를 맞춰 둔 것
gp = [" ".join(p.text.split()) for p in G.paragraphs]
tp = [" ".join(p.text.split()) for p in T.paragraphs]

import difflib
sm = difflib.SequenceMatcher(None, [x for x in tp if x], [x for x in gp if x])
ft = [i for i, x in enumerate(tp) if x]
fg = [i for i, x in enumerate(gp) if x]
edits = []
for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag == "equal": continue
    edits.append((tag, [ft[k] for k in range(i1, i2)], [fg[k] for k in range(j1, j2)]))

print(f"바뀐 곳 {len(edits)}군데")
for tag, ti, gi in edits:
    print(f"  [{tag}] 제출본 문단 {ti} ← 생성본 {gi}")
    for k in ti[:1]: print(f"      전: {tp[k][:120]}")
    for k in gi[:1]: print(f"      후: {gp[k][:120]}")

# 문단 개수가 같은 replace 만 자동 적용한다(삽입·삭제는 배치를 흔든다)
applied = manual = 0
for tag, ti, gi in edits:
    if tag == "replace" and len(ti) == len(gi):
        for a, b in zip(ti, gi):
            par = T.paragraphs[a]
            body = [r for r in par.runs if not r.font.superscript]
            if not body: manual += 1; continue
            body[0].text = gp[b] if not par.runs[0].font.superscript else gp[b].lstrip(par.runs[0].text)
            for r in body[1:]: r.text = ""
            applied += 1
    else:
        manual += len(ti) + len(gi)
print(f"\n자동 반영 {applied}문단 · 수동 확인 필요 {manual}")

# ── ② 문서 속성 정리 ────────────────────────────────────────────────────
T.save(TARGET)
FILES = [TARGET,
         f"{SUB}/02_부록/Multimedia_Appendix_1_supplementary_methods_results.docx",
         f"{SUB}/02_부록/Multimedia_Appendix_2_TRIPOD-AI_checklist.docx",
         f"{SUB}/04_커버레터/Cover_Letter.docx"]
for f in FILES:
    d = Document(f); cp = d.core_properties
    for a in ("author", "last_modified_by", "comments", "category", "keywords",
              "subject", "title", "identifier", "content_status"):
        try: setattr(cp, a, "")
        except Exception: pass
    cp.revision = 1
    d.save(f)
print("\n=== 문서 속성 ===")
for f in FILES:
    cp = Document(f).core_properties
    print(f"  {os.path.basename(f)[:44]:<46} author={cp.author!r} lastBy={cp.last_modified_by!r}")
