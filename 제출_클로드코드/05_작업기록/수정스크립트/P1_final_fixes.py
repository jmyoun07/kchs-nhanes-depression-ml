# -*- coding: utf-8 -*-
"""제출 직전 검토에서 확정된 15건 가운데 마크다운에 해당하는 것을 고친다.

원고 docx 는 손으로 배치를 맞춘 파일이라 재생성하지 않는다(P2 에서 문단만 교체).
부록·커버레터는 배치 조정이 없으므로 마크다운을 고치고 다시 만든다.
"""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")
D = "docs"
n_ok = n_miss = 0
def fix(f, a, b, why):
    global n_ok, n_miss
    p = f"{D}/{f}"; s = io.open(p, encoding="utf8").read()
    if a not in s:
        print(f"  [앵커없음] {f}: {why}"); n_miss += 1; return False
    io.open(p, "w", encoding="utf8").write(s.replace(a, b, 1)); n_ok += 1
    print(f"  {f.split('_')[0][:11]:<12} {why}")
    return True

print("=== 원고 ===")
# ① 같은 문장이 두 번 (내 편집 잔여물)
fix("manuscript_17var.md",
    " ECE, operating characteristics and net benefit are point estimates, and two-sided bootstrap P values "
    "reach P<.001 at the smallest attainable value (Table S3).", "",
    "① 중복 문장 삭제")
fix("manuscript_17var.md", "ECE for the reason given with Table S3,",
    "ECE for the reason given with Table S3 in Multimedia Appendix 1,", "① 남은 인용을 규정 어순으로")

# ② PHQ-9 이 대리지표라는 한계
fix("manuscript_17var.md",
    "The design is cross-sectional; the model screens current symptoms rather than predicting onset.",
    "The design is cross-sectional; the model screens current symptoms rather than predicting onset. The "
    "reference standard is a PHQ-9 sum of at least 10 rather than a diagnostic interview, hence every estimate "
    "reported here is agreement with a questionnaire-defined screening result and carries the imperfect "
    "sensitivity and specificity of that instrument against a semi-structured diagnosis [3,19].",
    "② 한계에 '대리지표' 진술 추가")

# ③ 표 2 각주를 열 읽는 순서대로 (Brier score 가 ECE·CITL 보다 위 행이다)
s = io.open(f"{D}/manuscript_17var.md", encoding="utf8").read()
a3 = ("ECE: expected calibration error.\nCITL: calibration-in-the-large; positive values denote under-prediction "
      "of observed risk.\nBrier score: mean squared difference between predicted probability and observed "
      "outcome, smaller being better.")
b3 = ("Brier score: mean squared difference between predicted probability and observed outcome, smaller being "
      "better.\nECE: expected calibration error.\nCITL: calibration-in-the-large; positive values denote "
      "under-prediction of observed risk.")
if a3 in s:
    io.open(f"{D}/manuscript_17var.md", "w", encoding="utf8").write(s.replace(a3, b3, 1)); n_ok += 1
    print("  manuscript   ③ 표2 각주를 열 순서대로 재배열 (Brier → ECE → CITL)")
else:
    print("  [앵커없음] ③ 표2 각주 순서"); n_miss += 1

# ④ PHQ-9 확장형 통일 (본문 약어표가 Patient Health Questionnaire-9)
for f in ("manuscript_17var.md", "appendix3_utility.md", "appendix2_crosswalk.md", "appendix5_methods.md"):
    p = f"{D}/{f}"; s = io.open(p, encoding="utf8").read()
    k = s.count("PHQ-9: 9-item Patient Health Questionnaire")
    if k:
        io.open(p, "w", encoding="utf8").write(s.replace("PHQ-9: 9-item Patient Health Questionnaire",
                                                         "PHQ-9: Patient Health Questionnaire-9"))
        n_ok += 1; print(f"  {f.split('_')[0][:11]:<12} ④ PHQ-9 확장형 통일 {k}곳")

# ⑤ MEC 를 정의해 놓고 안 쓰던 것
fix("manuscript_17var.md", "of the post-pandemic sample on the mobile examination center frame",
    "of the post-pandemic sample on the MEC frame", "⑤ 정의해 둔 MEC 사용")

# ⑥ 동률 규칙: 반올림 전 값을 밝힌다
fix("manuscript_17var.md", "The two leading algorithms differ by less than 0.005",
    "The two leading algorithms differ by 0.0046 before rounding", "⑥ 동률 판정 근거를 반올림 전 값으로")

print("\n=== 부록 1 ===")
# ⑦⑭ 부록이 자기 자신을 가리킨다
fix("appendix5_methods.md",
    "weight variables, which are named in Multimedia Appendix 1.",
    "weight variables, which are named in the complex-survey design-variables table above.",
    "⑦ 자기참조 → 위 표를 가리킴")
fix("appendix5_methods.md",
    "The crosswalk additionally provides the year-specific harmonization crosswalk together with an English "
    "rendition of the locked amendment.",
    "This appendix additionally provides the year-specific harmonization crosswalk together with an English "
    "rendition of the locked amendment.",
    "⑭ '크로스워크가 크로스워크를 제공' 주어 교정")
# ⑨ 세미콜론 뒤 대문자
fix("appendix5_methods.md",
    "on the external side; The crosswalk lists each exclusion with its reason.",
    "on the external side; the table of variables excluded as circular with the outcome lists each exclusion "
    "with its reason.",
    "⑨ 세미콜론 뒤 대문자 + 가리키는 대상 명확화")
# ⑩ 미정의 약어
for f, a, b, why in (("appendix3_utility.md", "| Screen-all NB |", "| Screen-all net benefit |", "⑩ NB → 풀어쓰기"),
                     ("appendix3_utility.md", "population stability index", "population stability index", "")):
    if why: fix(f, a, b, why)

print(f"\n적용 {n_ok} · 앵커 없음 {n_miss}")
