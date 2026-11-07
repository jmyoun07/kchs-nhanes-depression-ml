# -*- coding: utf-8 -*-
"""표 캡션·각주를 예시파일 형식으로 줄이고, 각주가 본문을 삼키던 버그를 고친다.

 ① 캡션은 한 문장. 설명은 캡션이 아니라 각주로 내린다(예시파일과 동일).
 ② 각주는 약어 하나에 문자 하나. 'ABBR: 풀이.' 한 줄씩.
 ③ 44_make_docx.py 의 collect_notes 는 표 뒤 빈 줄을 건너뛰며 계속 읽어서
    Methods 본문 문단까지 각주로 만들고 있었다. 표 1 밑에 1000단어가 붙은 원인이다.
    표 바로 뒤 '첫 문단 하나'만 각주로 본다.
 ④ 각주 문자를 다는 anchor 가 본문 행의 첫 칸만 뒤져서, 표 1처럼 첫 칸이 번호인
    표에서는 BMI·PHQ-9 에 문자가 붙지 않았다. 모든 칸을 뒤지게 한다.
"""
import io, os, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")

# ── ① 표 1 캡션 ───────────────────────────────────────────────────────────
p = "docs/manuscript_17var.md"
s = io.open(p, encoding="utf8").read()
lines = s.split("\n")

t1_old = lines[59]
assert t1_old.startswith("**Table 1.**"), "표1 캡션 줄 위치가 다름"
t1_new = ("**Table 1.** Harmonized 17-predictor set shared by the Korea Community Health Survey "
          "(KCHS) and the National Health and Nutrition Examination Survey (NHANES).")
lines[59] = t1_new

# 표 1 각주 (표 바로 뒤 첫 문단)
n1_old = lines[83]
assert n1_old.startswith("The full year-specific harmonization crosswalk"), f"표1 각주 위치가 다름: {n1_old[:60]}"
lines[83] = "\n".join([
    "One representative variable name is shown per concept; several concepts, physical activity in "
    "particular, derive from more codes than are shown, and the full year- and cycle-specific crosswalk "
    "is in Multimedia Appendix 1. Predictors 15 to 17 entered the set through the documented protocol "
    "amendment. Living alone is absent from the pre-pandemic NHANES file. Race and ethnicity is not a "
    "predictor and is used only for the NHANES subgroup analysis.",
    "BMI: body mass index.",
    "PHQ-9: 9-item Patient Health Questionnaire.",
])

s = "\n".join(lines)

# ── ② 표 2 캡션·각주 ──────────────────────────────────────────────────────
t2_old = [l for l in s.split("\n") if l.startswith("**Table 2.**")][0]
t2_new = ("**Table 2.** Discrimination and calibration of the primary model (light gradient-boosting "
          "machine, 17 predictors) in the internal KCHS hold-out and the two NHANES cycles, before and "
          "after recalibration.")
s = s.replace(t2_old, t2_new, 1)

n2_old = """CIs are from a rescaled stratified cluster bootstrap (1000 replicates). Recalibration is intercept/slope updating applied out-of-sample (5-fold cross-fit); since each fold uses a separate monotone calibration map, the recalibrated AUROC may differ trivially (by \u22640.002) from the original and is not the target of recalibration (calibration slope and CITL are). Internal calibration uses the original probabilities. Race/ethnicity is used only for the NHANES fairness analysis, not as a predictor.
AUROC: area under the receiver operating characteristic curve.
AUPRC: area under the precision-recall curve.
ECE: expected calibration error; excellent internally (0.003) with drift on external application (0.030 pre, 0.068 post), recoverable by recalibration.
CITL: calibration-in-the-large, the offset-model calibration intercept; hence positive values denote under-prediction of observed risk.
Brier score: mean squared difference between predicted probability and observed outcome, smaller being better.
Intervals were generated for AUROC and AUPRC as transported and, in the two external cohorts only, for the calibration slope and CITL both as transported and after recalibration; a cell shown without an interval was not resampled."""
assert n2_old in s, "표2 각주 블록 앵커 없음"
n2_new = """All estimates are complex-survey weighted, and the internal hold-out shares no primary sampling unit with the training data. CIs are from a rescaled stratified cluster bootstrap with 1000 replicates; a cell shown without an interval was not resampled. Recalibration is intercept-and-slope updating applied out-of-sample by 5-fold cross-fitting, and predictor effects are not re-estimated; since each fold uses a separate calibration map, the recalibrated AUROC may differ from the original by up to 0.002 and is not the target of recalibration.
AUROC: area under the receiver operating characteristic curve.
AUPRC: area under the precision-recall curve.
ECE: expected calibration error.
CITL: calibration-in-the-large; positive values denote under-prediction of observed risk.
Brier score: mean squared difference between predicted probability and observed outcome, smaller being better."""
s = s.replace(n2_old, n2_new, 1)
io.open(p, "w", encoding="utf8").write(s)

# ── ③④ 생성기 수정 ───────────────────────────────────────────────────────
q = "44_make_docx.py"
d = io.open(q, encoding="utf8").read()

c_old = '''def collect_notes(lines, i):
    """표 바로 뒤에 붙는 주석 문단을 모은다(제목·표·굵은 라벨을 만나면 멈춘다)."""
    notes = []
    while i < len(lines):
        t = lines[i].strip()
        if not t: i += 1; continue
        if t.startswith("#") or t.startswith("|") or t.startswith("**") or t == "---": break
        notes.append(t); i += 1
    return notes, i'''
c_new = '''def collect_notes(lines, i):
    """표 바로 뒤 '첫 문단 하나'만 주석으로 본다.

    예전에는 빈 줄을 건너뛰며 계속 읽어서 표 뒤에 오는 Methods 본문 문단까지
    주석으로 만들었다. 표 1 밑에 본문 1000단어가 붙어 있던 원인이다.
    """
    notes = []
    while i < len(lines) and not lines[i].strip():   # 표와 주석 사이의 빈 줄
        i += 1
    while i < len(lines):
        t = lines[i].strip()
        if not t: break                              # 빈 줄 = 주석 끝, 이후는 본문
        if t.startswith("#") or t.startswith("|") or t.startswith("**") or t == "---": break
        notes.append(t); i += 1
    return notes, i'''
assert c_old in d, "collect_notes 앵커 없음"
d = d.replace(c_old, c_new, 1)

a_old = '''        for r in body:
            if r and tok.lower() in r[0].lower(): r[0] = r[0] + f"^{ch}"; return True
        return False'''
a_new = '''        for r in body:                       # 첫 칸만이 아니라 모든 칸을 뒤진다
            for ci, c in enumerate(r):       # (표 1은 첫 칸이 번호라 못 찾았다)
                if tok.lower() in c.lower(): r[ci] = c + f"^{ch}"; return True
        return False'''
assert a_old in d, "anchor 앵커 없음"
d = d.replace(a_old, a_new, 1)

io.open(q, "w", encoding="utf8").write(d)
compile(io.open(q, encoding="utf8").read(), q, "exec")

print(f"표1 캡션 {len(t1_old.split())} -> {len(t1_new.split())} 단어")
print(f"표2 캡션 {len(t2_old.split())} -> {len(t2_new.split())} 단어")
print(f"표2 각주 {len(n2_old.split())} -> {len(n2_new.split())} 단어")
print("44_make_docx.py: 각주가 본문을 삼키던 버그 수정 · 각주 문자 앵커를 전 칸으로 확장")
