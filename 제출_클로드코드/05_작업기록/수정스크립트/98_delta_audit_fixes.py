# -*- coding: utf-8 -*-
"""98_delta_audit_fixes.py — 델타 감사 13건의 교정."""
import io, os, re, sys, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__)); RES = f"{D}/results"

NR = pd.read_csv(f"{RES}/primary2_dpq_nonresponse.csv")
o = lambda p_, c: NR[(NR.period == p_) & (NR.factor == "__overall__")][c].iloc[0]

MS = [
 # ① LightGBM 기제 서술을 정확히 (기본 분기가 아니라 0 으로 변환된다)
 ("the learner therefore never observes a missing value for it and never learns a missing branch, and at prediction time it routes the entirely missing pre-pandemic column to the default branch, which is the same branch as \"not living alone\"",
  "the learner therefore never observes a missing value for it and never learns a missing direction for it; at prediction time the missing values are treated as zero and follow the same branch as \"not living alone\""),
 # ② 한계의 미응답 수치에 프레임을 붙인다
 (f"Depression-module non-response was {o('pre','nr3_interview_pct'):.1f}% before the pandemic and {o('post','nr3_interview_pct'):.1f}% after it and was handled as complete-case",
  f"Depression-module non-response within the mobile examination centre estimation frame cost 7.7% of the pre-pandemic and 14.1% of the post-pandemic sample "
  f"({o('pre','nr3_interview_pct'):.1f}% and {o('post','nr3_interview_pct'):.1f}% if counted over all interviewed adults) and was handled as complete-case"),
 # ⑤ 두 수치는 정의가 다르다 — '같은 통계의 다른 분모' 로 읽히지 않게
 (f"Counted instead over all interviewed adults, the figures are {o('pre','nr3_interview_pct'):.1f}% and {o('post','nr3_interview_pct'):.1f}%, but that denominator conflates module non-response with never having been examined, which the MEC weights already account for.",
  f"The corresponding three-or-more-item rates are {o('pre','nr3_mec_pct'):.1f}% and {o('post','nr3_mec_pct'):.1f}% on the MEC frame, against {o('pre','nr3_interview_pct'):.1f}% and {o('post','nr3_interview_pct'):.1f}% when counted over all interviewed adults; that second denominator conflates module non-response with never having been examined, which the MEC weights already account for."),
 # ⑦ 이어붙이면서 소문자로 시작한 문장
 ("Table S16. the difference in external AUROC between the two cycles is also given an interval.",
  "Table S16. The difference in external AUROC between the two cycles is also given an interval."),
 # ⑩ '그 탐색 실행' 의 선행사가 틀렸다
 ("(labels unused) *reduced* external AUROC, to 0.766 and 0.758 against that exploration run's baseline of 0.772 and 0.769, showing",
  "(labels unused) *reduced* external AUROC, to 0.766 and 0.758 against the baseline of the transfer-optimization run that anchors Multimedia Appendix 1 Table S11 (0.772 pre, 0.769 post), showing"),
 # ⑪ 비중의 분모를 명시 (표 S5 의 공통 설정 대비)
 ("Removing food insecurity alone accounts for 28% of that difference pre-pandemic and 50% post-pandemic.",
  "Under the common configuration of Table S5, where the 17-, 16- and 14-predictor models share one setting, removing food insecurity alone accounts for 28% of the 17-versus-14 gap pre-pandemic and 50% post-pandemic."),
]

def edit(path, pairs):
    p = f"{D}/{path}"; s = io.open(p, encoding="utf8").read(); n = 0
    for a, b in pairs:
        if a not in s: print(f"  [미일치] {path}: {a[:80]}"); continue
        s = s.replace(a, b, 1); n += 1
    io.open(p, "w", encoding="utf8").write(s); print(f"  {path}: {n}/{len(pairs)}건")

print("델타 감사 교정")
edit("docs/manuscript_17var.md", MS)

# ③ 부록2 의 미응답 문장에도 프레임을 (부록1 한 파일 안에서 값이 갈리지 않게)
edit("docs/appendix2_crosswalk.md", [
 (f"Depression-module non-response in NHANES (three or more PHQ-9 items unanswered) was {o('pre','nr3_interview_pct'):.1f}% pre and {o('post','nr3_interview_pct'):.1f}% post",
  f"Depression-module non-response in NHANES (three or more PHQ-9 items unanswered) was {o('pre','nr3_mec_pct'):.1f}% pre and {o('post','nr3_mec_pct'):.1f}% post on the mobile examination centre estimation frame ({o('pre','nr3_interview_pct'):.1f}% and {o('post','nr3_interview_pct'):.1f}% over all interviewed adults)")])

# ⑥ 표 S8 의 행 이름이 정의와 어긋난다 (7.7/14.1 은 any-item)
edit("docs/appendix3_utility.md", [
 ("| Lost to item non-response within the MEC frame | 7.7% | 14.1% |",
  "| Lost to **any** item non-response within the MEC frame (the complete-case exclusion) | 7.7% | 14.1% |")])

# ⑫ P 값 표기를 AMA/JMIR 방식으로 (선행 0 없음)
for f in ("docs/manuscript_17var.md", "docs/appendix3_utility.md"):
    p = f"{D}/{f}"; s = io.open(p, encoding="utf8").read()
    s2 = re.sub(r"bootstrap P=0\.(\d\d)", r"bootstrap P=.\1", s)
    if s2 != s: io.open(p, "w", encoding="utf8").write(s2); print(f"  {f}: P 표기 AMA 방식")

# ⑬ 모델카드의 범위 진술을 실제 범위로
edit("docs/model_card_17var.md", [
 ("the numbers in the manuscript and appendices are injected from the result files by `61_sync_numbers.py` and checked by `60_verify_manuscript.py`",
  "the primary-analysis numbers in the manuscript and appendices are injected from `results/primary2_*.csv` by `61_sync_numbers.py` and checked by `60_verify_manuscript.py`; the separate exploration-run tables (Multimedia Appendix 1, Tables S11 to S13) are transcribed from their own run logs at the precision recorded there")])

# ⑧ 표 S4 생성기: 비율은 % 로, 전량 결측은 NaN 대신 표시
p = f"{D}/61_sync_numbers.py"; s = io.open(p, encoding="utf8").read()
a = '''    _pctv = _k in ("Depression PHQ-9 ≥10", "Female", "Current smoker", "Partnered", "Employed", "Food insecure", "Hypertension", "Diabetes")
    _lab = {"Depression PHQ-9 ≥10": "Depression PHQ-9 ≥10"}.get(_k, _k)
    _f = (lambda v: f"{R(v,1)}%") if _pctv else (lambda v: R(v, 2))'''
b = '''    # 연속변수만 소수 2자리, 나머지는 모두 비율이므로 % 를 붙인다
    _CONT = ("Age (years)", "BMI", "Sleep hours", "Self-rated health", "Oral health", "Household income quintile")
    _lab = _k
    _f = (lambda v: "—" if pd.isna(v) else R(v, 2)) if _k in _CONT else \\
         (lambda v: "not collected" if pd.isna(v) else f"{R(v,1)}%")'''
assert a in s, "표 S4 생성부 앵커 없음"
io.open(p, "w", encoding="utf8").write(s.replace(a, b, 1))
compile(io.open(p, encoding="utf8").read(), p, "exec")
print("  61_sync_numbers.py: 표 S4 서식 (비율 % · 전량결측 표기)")
