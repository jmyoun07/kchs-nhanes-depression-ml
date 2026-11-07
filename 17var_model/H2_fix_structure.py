# -*- coding: utf-8 -*-
"""Part 껍데기를 없애면서 어긋난 구조 세 곳을 바로잡는다.

 ① 크로스워크: Part A 제목을 지우면서 그 아래 있던 범위 설명이 파일 맨 위로 떠올랐다.
    'Supplementary Methods' 제목 뒤로 옮긴다.
 ② 천장분석: Part C 제목을 지우면서 범위 설명이 제목 없이 남았다. 제목을 준다.
 ③ 보충방법: Part D 머리말을 지우면서 가중치 규약 문장이 함께 사라졌다.
    그 절에도 다시 명시한다(표마다 가중 여부를 읽는 근거다).
"""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")
D = "docs"
def rd(f): return io.open(f"{D}/{f}", encoding="utf8").read()
def wr(f, s): io.open(f"{D}/{f}", "w", encoding="utf8").write(s)

# ── ① 크로스워크: 떠 있는 범위 설명을 제목 아래로 ────────────────────────
s = rd("appendix2_crosswalk.md")
m = re.match(r"(Supports the Methods sections.*?)\n\n---\n\n(## Supplementary Methods\n\n)(.*?\n)\n",
             s, re.S)
assert m, "① 앵커 없음"
s = m.group(2) + m.group(3) + "\n" + m.group(1).strip() + "\n\n" + s[m.end():]
wr("appendix2_crosswalk.md", s)
print("① 크로스워크: 범위 설명을 'Supplementary Methods' 제목 아래로 이동")

# ── ② 천장분석: 제목 없이 남은 범위 설명에 제목을 준다 ──────────────────
s = rd("appendix4_ceiling.md")
assert s.lstrip().startswith("Supports Results Testing"), "② 앵커 없음"
s = "### Transportability-ceiling and calibration-interval analyses\n\n" + s.lstrip()
wr("appendix4_ceiling.md", s)
print("② 천장분석: 'Transportability-ceiling and calibration-interval analyses' 제목 부여")

# ── ③ 보충방법: 사라진 가중치 규약 복원 ─────────────────────────────────
s = rd("appendix5_methods.md")
a = "### Study design, model development, and analysis detail\n"
b = ("### Study design, model development, and analysis detail\n\n"
     "Detail relocated from the manuscript Methods so that the main text states what a reader needs in order "
     "to judge the study and this section states what a reader needs in order to reproduce it. Estimates are "
     "complex-survey weighted unless a row or column is labeled unweighted; nothing here restates a result "
     "reported in the manuscript, and where a number appears in both, the manuscript is the reporting site "
     "and this section gives the derivation.\n")
assert a in s and "complex-survey weighted unless" not in s, "③ 앵커 없음 또는 이미 있음"
wr("appendix5_methods.md", s.replace(a, b, 1))
print("③ 보충방법: 가중치 규약 문장 복원 (Part D 머리말과 함께 사라졌던 것)")
