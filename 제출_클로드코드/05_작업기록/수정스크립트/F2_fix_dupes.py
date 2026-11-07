# -*- coding: utf-8 -*-
"""내가 만든 중복·조각을 고친다.

빠진 수치를 되살릴 때 유사도로 자리를 찾았고, 축소본이 같은 내용을 다른 문장으로
이미 갖고 있으면 '없음' 으로 보고 새로 끼워 넣었다. 그 결과가 아래 다섯 곳이다.
이번에는 유사도를 쓰지 않고 정확한 문자열로만 고친다.
"""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")
p = "docs/manuscript_17var.md"
s = io.open(p, encoding="utf8").read()
n = 0

# ── ① 줄 47: NHANES 문장이 두 판본으로 나란히 ────────────────────────────
a = ("External validation used NHANES, an NCHS survey combining household interview and mobile examination "
     "center assessment, with depression measured by the Depression Screener (DPQ; PHQ-9) module under the "
     "2015-2018 sample design and estimation procedures [18]. External validation used NHANES, an NCHS survey "
     "combining household interview and mobile examination center assessment, with depression measured by the "
     "Depression Screener (DPQ; PHQ-9) module; NCHS documents the 2015-2018 sample design and estimation "
     "procedures [18].")
b = ("External validation used NHANES, an NCHS survey combining household interview and mobile examination "
     "center assessment, with depression measured by the Depression Screener (DPQ; PHQ-9) module; NCHS "
     "documents the 2015-2018 sample design and estimation procedures [18].")
assert a in s, "① 앵커 없음"
s = s.replace(a, b, 1); n += 1
print("① 연구설계: NHANES 문장 두 판본 -> 하나 (DPQ 약어 중복 정의도 해소)")

# ── ② 줄 129: 무응답 문장 중복 + 분모 정의가 섞임 ────────────────────────
a = ("Non-response was differential, higher among respondents with less than high-school education and among "
     "Asian and Black respondents; hence the external cohorts, and particularly the post-pandemic fairness "
     "estimates, should be read as complete-case results (Multimedia Appendix 1, Table S8). Non-response was "
     "differential: higher among respondents with less than high-school education (12.9% pre, 26.6% post, MEC "
     "frame) and among Asian and Black respondents; hence the external cohorts, and particularly the "
     "post-pandemic fairness estimates, should be read as complete-case results.")
b = ("Non-response was differential: on the same examination-center frame, three or more items were unanswered "
     "by 12.9% of respondents with less than high-school education pre-pandemic and 26.6% post-pandemic, "
     "against 7.4% and 13.3% overall on that definition, and rates were also higher among Asian and Black "
     "respondents; hence the external cohorts, and particularly the post-pandemic fairness estimates, should "
     "be read as complete-case results (Multimedia Appendix 1, Table S8).")
assert a in s, "② 앵커 없음"
s = s.replace(a, b, 1); n += 1
print("② 표본특성: 무응답 문장 중복 제거 + 3문항 기준 전체값(7.4/13.3)을 함께 적어 분모 혼동 해소")

# ── ③ 줄 163: 닫는 괄호 뒤에 남은 조각 ──────────────────────────────────
a = "(Multimedia Appendix 1, Table S11).003 in either direction. Deep tabular learners"
b = "(Multimedia Appendix 1, Table S11). Deep tabular learners"
assert a in s, "③ 앵커 없음"
s = s.replace(a, b, 1); n += 1
print("③ 천장 절: 문장 끝에 남아 있던 '.003 in either direction.' 조각 제거")

# ── ④⑤ 문단에 안 붙고 떠 있던 두 줄 ────────────────────────────────────
L = s.split("\n")
out, drop = [], 0
for i, t in enumerate(L):
    if t[:1] == " " and t.strip() and not t.startswith("|"):
        st = t.strip()
        if st.startswith("The five-item set falls 0.0002"):
            drop += 1; print("④ 경량모형: 앞 문단에 이미 있는 5항목 문장(중복 줄) 삭제"); continue
        if st.startswith("Under the common configuration"):
            # 앞 문단 끝에 붙인다(내용은 맞고 자리가 틀렸다)
            for k in range(len(out) - 1, -1, -1):
                if out[k].strip() and not out[k].startswith(("#", "|")):
                    out[k] = out[k].rstrip() + " " + st; break
            drop += 1; print("⑤ 민감도: 떠 있던 식품불안정 문장을 앞 문단에 붙임")
            continue
    out.append(t)
s = re.sub(r"\n{3,}", "\n\n", "\n".join(out))
io.open(p, "w", encoding="utf8").write(s)
print(f"\n총 {n + drop}곳 수정")
