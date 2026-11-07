# -*- coding: utf-8 -*-
"""부록2(TRIPOD+AI)를 예시파일 구조로 다시 만든다.

바뀌는 것
 ① ✅/⏳ 초록 체크박스를 없앤다. 논문 부록에 쓰는 표기가 아니다.
    예시파일은 마지막 열에 '어디에 보고했는지'를 문장으로 적는다. 그 방식을 따른다.
 ② 항목 번호를 공식 TRIPOD+AI(2024) 체계로 바꾼다. 지금은 1~33 임의 번호였다.
    공식은 1~27 에 하위항목이 붙어 52개다.
 ③ 열 구성을 예시와 같게: Section/Topic · Item · Development/evaluation · Checklist item · Reported in
 ④ 'Table A.2' 는 표가 아니라 절 번호다(부록 Part A 의 A.2). 참조를 바로잡는다.
 ⑤ 92_crossref_audit.py 의 TRIPOD 열 검사는 4열 표를 가정한다. 5열이 되면 정규식이
    한 줄도 못 잡아 검사가 조용히 죽는다. 마지막 열을 보도록 고친다.
"""
import io, json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")

# ── ④ 우리 문서의 잘못된 참조부터 ────────────────────────────────────────
p = "docs/appendix2_crosswalk.md"
s = io.open(p, encoding="utf8").read()
old = "Harmonized definitions are given in Table A.2."
new = "Harmonized definitions are given in section A.2 above."
if old in s:                      # 다시 실행해도 멈추지 않게 (아래 단계가 본체다)
    io.open(p, "w", encoding="utf8").write(s.replace(old, new, 1))
    print("appendix2_crosswalk.md: 'Table A.2' -> 'section A.2' (A.2 는 표가 아니라 절)")
else:
    assert new in s, "Table A.2 참조가 고쳐지지도, 남아있지도 않음"
    print("appendix2_crosswalk.md: 'section A.2' 이미 적용됨")

# ── 자료 ─────────────────────────────────────────────────────────────────
OFF = json.loads(io.open("work_notes/tripod_official.json", encoding="utf8").read())
MAP = json.loads(io.open("work_notes/tripod_map.json", encoding="utf8").read())

def norm(t):
    """'Table A.N' 은 절 번호다. 그리고 Part A/B/C 는 Multimedia Appendix 1 에만 있다."""
    t = re.sub(r"\bTable (A\.\d+)", r"section \1", t)
    t = re.sub(r"Multimedia Appendix 2([, ]+Part [A-C])", r"Multimedia Appendix 1\1", t)
    return " ".join(t.split())

byitem = {}
for e in MAP:
    k = e["item"]
    if k in byitem and len(byitem[k]["location"]) >= len(e["location"]):
        continue                       # 같은 항목이 둘이면 더 자세한 쪽을 남긴다
    byitem[k] = e
print(f"매핑 {len(MAP)}건 -> 항목 {len(byitem)}개 (중복 정리)")

missing = [o["item"] for o in OFF if "item" in o and o["item"] != "Item" and o["item"] not in byitem]
assert not missing, f"공식 항목 누락: {missing}"

# ── 약어 목록은 원고의 Abbreviations 절에서 가져온다 ──────────────────────
MAN = io.open("docs/manuscript_17var.md", encoding="utf8").read()
ab = MAN.split("## Abbreviations")[-1].strip()
pairs = re.findall(r"^\**([A-Za-z0-9+\-]{2,14})\**\s*[::]\s*(.+?)\s*$", ab, re.M)
abbr = "; ".join(f"{k}, {v.rstrip('.')}" for k, v in pairs)

# ── 본문 조립 ────────────────────────────────────────────────────────────
L = []
L.append("# Multimedia Appendix 2. TRIPOD+AI Checklist.")
L.append("")
# TRIPOD+AI 는 원고 약어 목록에도 있다. 여기서 또 쓰면 한 문서에 두 가지 확장형이 생긴다.
L.append(f"Abbreviations: {abbr}.")
L.append("")
L.append("* D=items relevant only to the development of a prediction model; E=items relating solely "
         "to the evaluation of a prediction model; D;E=items applicable to both the development and "
         "evaluation of a prediction model.")
L.append("")
L.append("Item text reproduced from the TRIPOD+AI checklist (Collins GS, Moons KGM, Dhiman P, et al. "
         "BMJ. 2024;385:e078378).")
L.append("")
L.append("| Section/Topic | Item | Development/evaluation* | Checklist item | Reported in |")
L.append("|---|---|---|---|---|")

def esc(t):
    return " ".join(str(t).replace("|", "/").split())

n_head = n_row = 0
for o in OFF:
    if "section_row" in o:
        if o["section_row"] in ("Section/Topic",):  # 머리글 행은 이미 넣었다
            continue
        L.append(f"| **{esc(o['section_row'])}** | | | | |"); n_head += 1
        continue
    if o.get("item") == "Item":
        continue
    e = byitem[o["item"]]
    L.append(f"| {esc(o['topic'])} | {esc(o['item'])} | {esc(o['de'])} | {esc(o['text'])} | "
             f"{esc(norm(e['location']))} |")
    n_row += 1

io.open("docs/TRIPOD_AI_checklist.md", "w", encoding="utf8").write("\n".join(L) + "\n")
print(f"TRIPOD_AI_checklist.md 재작성: 항목 {n_row}행 · 절 머리 {n_head}행 · 5열")

# ── ⑤ 검사기의 열 가정 ───────────────────────────────────────────────────
q = "92_crossref_audit.py"
d = io.open(q, encoding="utf8").read()
g_old = r'''        for _row in re.findall(r"^\|[^|]*\|[^|]*\|[^|]*\|\s*([^|]+?)\s*\|\s*$", _t, re.M):'''
g_new = ('''        # 마지막 열('Reported in')을 본다. 열 수를 고정하면 표가 바뀔 때 검사가 조용히 죽는다.\n'''
         '''        for _row in [x.rsplit("|", 2)[-2].strip() for x in _t.split("\\n")\n'''
         '''                     if x.strip().startswith("|") and x.count("|") >= 5]:''')
if g_old in d:                    # 다시 실행해도 멈추지 않게
    d = d.replace(g_old, g_new, 1)
    io.open(q, "w", encoding="utf8").write(d)
    compile(io.open(q, encoding="utf8").read(), q, "exec")
    print("92_crossref_audit.py: TRIPOD 열 검사를 마지막 열 기준으로 (5열에서도 살아있게)")
else:
    assert 'rsplit("|", 2)[-2]' in d, "TRIPOD 열 검사가 고쳐지지도, 원래대로도 아님"
    print("92_crossref_audit.py: 열 검사 이미 적용됨")

st = {}
for e in byitem.values(): st[e["status"]] = st.get(e["status"], 0) + 1
print("\n상태:", st)
