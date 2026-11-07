# -*- coding: utf-8 -*-
"""축소가 지운 검사 대상 수치 문장 3건을 되살리고, 대소문자 때문에 빗나간 동기화 정규식 1건을 고친다."""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")

B = io.open("work_notes/manuscript_pre_final.bak.md", encoding="utf8").read()
p = "docs/manuscript_17var.md"
s = io.open(p, encoding="utf8").read()

def sent(anchor):
    """이전본에서 anchor 를 품은 한 문장을 통째로 꺼낸다."""
    i = B.index(anchor)
    a = max(B.rfind(". ", 0, i), B.rfind("\n", 0, i)) + 1
    j = B.find(". ", i)
    return " ".join(B[a:j + 1].split())

# ── ① 저학력 DPQ 미응답률(MEC 프레임) ────────────────────────────────────
if "MEC frame)" not in s:
    t = sent("(12.9% pre, 26.6% post, MEC frame)")
    m = re.search(r"Non-response was differential[^.]*\.", s)
    if m:
        s = s[:m.end()] + " " + t + s[m.end():]
    else:
        m = re.search(r"(?s)(### Participant Characteristics.*?)(\n\n)", s)
        s = s[:m.end(1)] + " " + t + s[m.end(1):]
    print(f"① 저학력 미응답률 복원: {t[:96]}...")

# ── ② 경량모형 k=5 외부값 ────────────────────────────────────────────────
if not re.search(r"reaches 0\.\d{3} pre-pandemic and 0\.\d{3} post-pandemic", s):
    t = sent("the five-item model reaches 0.737 pre-pandemic and 0.751 post-pandemic externally")
    a = s.index("### Parsimonious Deployment Model")
    b = s.index("\n### ", a)
    s = s[:b] + " " + t + s[b:]
    print(f"② k=5 외부값 복원: {t[:96]}...")

# ── ③ 식품불안정 기여 비중 ───────────────────────────────────────────────
if "of the 17-versus-14 gap" not in s:
    t = sent("accounts for 28% of the 17-versus-14 gap pre-pandemic and 50% post-pandemic")
    a = s.index("### Sensitivity Analyses")
    b = s.index("\n### ", a)
    s = s[:b] + " " + t + s[b:]
    print(f"③ 식품불안정 기여 비중 복원: {t[:96]}...")

io.open(p, "w", encoding="utf8").write(s)

# ── ④ 문장 첫 글자가 대문자가 되어 빗나간 동기화 정규식 ──────────────────
q = "61_sync_numbers.py"
d = io.open(q, encoding="utf8").read()
o1 = r'(r"external discrimination was 0\.\d{3}'
n1 = r'(r"[Ee]xternal discrimination was 0\.\d{3}'
o2 = 'f"external discrimination was {col(PRE,\'AUROC\')}'
n2 = 'f"External discrimination was {col(PRE,\'AUROC\')}'
if o1 in d:
    d = d.replace(o1, n1, 1)
    assert o2 in d, "동기화 대체문 앵커 없음"
    d = d.replace(o2, n2, 1)
    io.open(q, "w", encoding="utf8").write(d)
    compile(io.open(q, encoding="utf8").read(), q, "exec")
    print("④ 61_sync: 문장 첫 글자 대문자를 받아들이도록 수정")
