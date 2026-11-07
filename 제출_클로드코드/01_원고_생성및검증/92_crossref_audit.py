# -*- coding: utf-8 -*-
"""92_crossref_audit.py - 문서 사이의 상호참조를 기계적으로 검사한다.

코덱스가 라운드마다 찾아낸 결함 가운데 상당수가 (1) 없어진 절 이름, (2) 존재하지 않는 표 번호,
(3) 첫 사용 전에 정의되지 않은 약어였다. 사람이 읽어서 잡을 게 아니라 검사로 막는다.
"""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__)); DOCS = f"{D}/docs"
rd = lambda n: io.open(f"{DOCS}/{n}", encoding="utf8").read()

M = rd("manuscript_17var.md")
OTHERS = {n: rd(n) for n in ["appendix2_crosswalk.md", "appendix3_utility.md", "appendix4_ceiling.md",
                             "TRIPOD_AI_checklist.md", "model_card_17var.md", "cover_letter.md"]}
fails = []
def need(c, msg):
    if not c: fails.append(msg)

# ── 1. 원고의 실제 제목 목록 ────────────────────────────────────────────────
HEADS = set(re.findall(r"^#{2,4}\s+(.+?)\s*$", M, re.M))
BOLD_SUBHEADS = set(h.strip(" .") for h in re.findall(r"\*\*([A-Z][^*]{3,60}?)\.\*\*", M))
# 굵은 라벨을 없애면서 사라진 이름은 면제하지 않는다. 남겨 두면 그 이름을 가리키는
# 새 오참조가 검사를 그대로 통과한다.
EXTRA = {
    "Harmonization cautions", "Multimedia Appendix 1", "Multimedia Appendix 2",
    "Results", "Methods", "Discussion", "Introduction", "Abstract", "Data Availability",
    "Predictors and Harmonization",
}
KNOWN = HEADS | BOLD_SUBHEADS | EXTRA
# 면제 목록 자체를 검사한다. 검사하지 않으면 어느 문서에도 없는 절 이름이 그대로 통과한다.
# 절이 실제로 존재할 수 있는 문서(원고·부록·모델카드)만 대상으로 한다. 인용만 하는 문서를
# 포함하면 유령 절이 자기 인용에 기대어 통과하므로 체크리스트·표지서한은 뺀다.
_SECTION_SRC = "\n".join([M] + [OTHERS[_n] for _n in (
    "appendix2_crosswalk.md", "appendix3_utility.md",
    "appendix4_ceiling.md", "model_card_17var.md")]).lower()
for _e in sorted(EXTRA):
    need(_e.lower() in _SECTION_SRC, f"[KNOWN] 어느 문서에도 없는 절 이름을 면제하고 있음(유령 절): {_e}")
# KNOWN 을 실제로 쓴다. 쓰지 않으면 유령 절 이름을 가리키는 새 상호참조가 그대로 통과한다.
_CONN = {"and", "of", "the", "for", "in", "to", "a", "or", "vs", "with", "by"}
def _sectionish(p):
    ws = p.split()
    if not ws or re.search(r"\d", p): return False
    if ws[0] in ("Table", "Tables", "Figure", "Figures", "Multimedia", "Panel", "Section"): return False
    return (len(ws) >= 3 and all(re.match(r"^[A-Z]", w) or w in _CONN for w in ws)
            and len([w for w in ws if re.match(r"^[A-Z]", w)]) >= 3)
def _resolves(p):
    if p in KNOWN: return True
    parts = [x.strip() for x in re.split(r",\s+", p)]
    return len(parts) > 1 and all(x in KNOWN for x in parts)
for _nm, _t in [("manuscript", M)] + list(OTHERS.items()):
    for _m in re.finditer(r"\((?:see )?([A-Z][^()]{6,120})\)", _t):
        _p = _m.group(1).strip().rstrip(".")
        need(not _sectionish(_p) or _resolves(_p), f"[{_nm}] 존재하지 않는 절 이름 참조: {_p}")
    if _nm == "TRIPOD_AI_checklist.md":
        # 마지막 열('Reported in')을 본다. 열 수를 고정하면 표가 바뀔 때 검사가 조용히 죽는다.
        for _row in [x.rsplit("|", 2)[-2].strip() for x in _t.split("\n")
                     if x.strip().startswith("|") and x.count("|") >= 5]:
            for _p in [x.strip() for x in re.split(r"[;.(]", _row)[0].split(",")]:
                need(not _sectionish(_p) or _p in KNOWN, f"[{_nm}] 존재하지 않는 절 이름 참조: {_p}")
print(f"[1] 원고 제목 {len(HEADS)}개 · 굵은 소제목 {len(BOLD_SUBHEADS)}개")

# 폐기된 절 이름이 어디에도 남지 않아야 한다
DEAD = ["External Validation Strategy", "Study Design and Reporting Standards",
        "Candidate Predictors and Harmonization", "Geographic and Temporal External Validation",
        "Structured Abstract", "Sections the Methods", "Intended use and users",
        "Provenance of the Harmonized Predictor Set", "Performance Evaluation",
        "the Methods and the Methods", "in Methods the Methods"]
for d in DEAD:
    for nm, t in [("manuscript", M)] + list(OTHERS.items()):
        need(d not in t, f"[{nm}] 폐기된 절 이름 잔존: {d}")

# ── 2. 표·그림 번호가 실제로 있는지 ────────────────────────────────────────
S_DEFINED = set(int(x) for x in re.findall(r"^#{2,3} Table S(\d+)\.", "\n".join(OTHERS.values()), re.M))
# 인용 말뭉치에서 '표 제목 줄' 을 뺀다. 빼지 않으면 제목 자체가 인용으로 세어져
# S_DEFINED ⊆ S_CITED 가 구조적으로 성립하고, 미인용 가드가 영원히 발화하지 못한다.
_corpus = re.sub(r"^#{2,3} Table S\d+\..*$", "", M + "\n".join(OTHERS.values()), flags=re.M)
_corpus = re.sub(r"^## Multimedia Appendix \d\b.*?(?=^## )", "", _corpus, flags=re.M | re.S)
S_CITED = set(int(x) for x in re.findall(r"Table S(\d+)", _corpus))
# 'Tables S8, S6 and S10' · 'Tables S2, S7, S11, S9 and S3' 같은 복수·범위 인용도 센다
for _m in re.finditer(r"Tables? S(\d+)\s*(?:-\s*S?(\d+))?((?:\s*(?:,|and)\s*S\d+)*)", _corpus):
    a_, b_, rest = int(_m.group(1)), _m.group(2), _m.group(3) or ""
    S_CITED.add(a_)
    if b_: S_CITED |= set(range(a_, int(b_) + 1))
    S_CITED |= set(int(x) for x in re.findall(r"S(\d+)", rest))
need(not (S_CITED - S_DEFINED), f"[전체] 존재하지 않는 부록 표 인용: {sorted(S_CITED - S_DEFINED)}")
need(not (S_DEFINED - S_CITED), f"[전체] 인용되지 않은 부록 표: {sorted(S_DEFINED - S_CITED)}")
need(S_DEFINED == set(range(1, max(S_DEFINED) + 1)), f"[부록] 표 번호가 연속이 아님: {sorted(S_DEFINED)}")
print(f"[2] 부록 표 S1-S{max(S_DEFINED)} 정의·인용 일치")

T_CITED = set(int(x) for x in re.findall(r"Table (\d)\b", M))
T_DEF = set(int(x) for x in re.findall(r"\*\*Table (\d)\.\*\*", M))
need(T_CITED <= T_DEF, f"[manuscript] 없는 표 인용: {sorted(T_CITED - T_DEF)}")
F_DEF = set(int(x) for x in re.findall(r"Figure (\d):", M))
F_CITED = set(int(x) for x in re.findall(r"Figure (\d)\b", M)) - F_DEF
need(F_CITED <= F_DEF, f"[manuscript] 없는 그림 인용: {sorted(F_CITED - F_DEF)}")
print(f"[3] 본문 표 {sorted(T_DEF)} · 그림 {sorted(F_DEF)}")

# ── 3. 참고문헌: 번호 연속, 모두 인용, 첫 등장 순서 ─────────────────────────
refsec = M.split("## References")[1].split("## Abbreviations")[0]
nums = [int(m.group(1)) for m in re.finditer(r"^(\d+)\.\s", refsec, re.M)]
need(nums == list(range(1, len(nums) + 1)), "[references] 번호가 1..N 연속이 아님")
body = M.split("## References")[0]
cited = []
for m in re.finditer(r"\[(\d+(?:[,\-]\d+)*)\]", body):
    for part in m.group(1).split(","):
        if "-" in part:
            a, b = part.split("-"); cited += list(range(int(a), int(b) + 1))
        else: cited.append(int(part))
need(set(cited) == set(nums), f"[references] 인용/목록 불일치: 미인용 {sorted(set(nums)-set(cited))}, 미등재 {sorted(set(cited)-set(nums))}")
first, seen = [], set()
for c in cited:
    if c not in seen: seen.add(c); first.append(c)
need(first == sorted(first), f"[references] 첫 등장 순서가 오름차순이 아님: {first[:12]}…")
print(f"[4] 참고문헌 {len(nums)}편 · 첫 등장 순서 정렬 확인")

# ── 4. 약어: 목록의 모든 항목이 본문에 쓰이고, 쓰이는 약어가 목록에 있는지 ──
abbr = M.split("## Abbreviations")[1]
listed = dict(re.findall(r"^([A-Za-z][A-Za-z0-9+\-]{1,12}):\s*(.+)$", abbr, re.M))
for a in listed:
    need(len(re.findall(rf"\b{re.escape(a)}\b", body)) > 0, f"[abbreviations] 본문에 쓰이지 않는 항목: {a}")
CAND = set(re.findall(r"\b([A-Z]{2,6}(?:\+[A-Z]{2})?)\b", body))
# 약어표에 실린 것은 여기서 면제하지 않는다 - 면제하면 감시 대상이 사라진다.
# 남기는 것은 일반명사·로마숫자·본문에서 풀어 쓸 필요가 없는 고유표기뿐이다.
SKIP = {"US", "AI", "OR", "SD", "II", "III", "IV", "COVID", "LOYO", "PSI", "WoE",
        "MLP", "FT", "GBM", "JMIR", "USPSTF", "PDF", "CSV", "DOI", "IRB", "NNS", "PHQ"}
# 약어표에 있는 항목은 '본문 첫 등장 시 풀어쓰기가 선행했는가' 로 검사한다
for a, full in listed.items():
    m0 = re.search(rf"\b{re.escape(a)}\b", body)
    if not m0: continue
    head_ = body[:m0.start()]
    key = full.split("(")[0].strip().rstrip(",.").lower()
    need(key[:18] in head_.lower() or f"({a})" in body[:m0.end() + 2] or a in ("PHQ-9",),
         f"[abbreviations] 본문 첫 사용 전에 풀어쓰지 않음: {a} ({full})")
# NHANES·KCHS 변수 코드는 약어가 아니다
VARCODE = re.compile(r"^(FSD|DMD|OCD|OHQ|HUQ|BPQ|DIQ|SMQ|ALQ|PAQ|PAD|SLD|BMX|RID|RIA|IND|SDMV|MCQ|HIQ|DPQ|SEQN|HSD|HSQ)")
missing = sorted(c for c in CAND - SKIP - set(listed)
                 if not VARCODE.match(c) and len(re.findall(rf"\b{c}\b", body)) >= 2)
need(not missing, f"[abbreviations] 목록에 없는 약어(2회 이상 사용): {missing}")
print(f"[5] 약어 {len(listed)}개 검사")

# ── 부록 구조 검사 ──────────────────────────────────────────────────────
# 이 검사는 예전에 결과 출력·sys.exit 뒤에 있었다. 그래서 실패가 집계되지 않고
# 늘 '통과' 로 보였다. 요약보다 앞으로 옮긴다.
_APPX = ["docs/appendix2_crosswalk.md", "docs/appendix3_utility.md",
         "docs/appendix4_ceiling.md", "docs/appendix5_methods.md"]
_asrc = [io.open(f"{D}/{v}", encoding="utf8").read() for v in _APPX]
_files = [M] + _asrc + [io.open(f"{D}/docs/TRIPOD_AI_checklist.md", encoding="utf8").read()]

# 'Part X' 표기를 쓸 거면 실제로 그 제목이 있어야 하고, 안 쓸 거면 어디에도 없어야 한다.
_declared = set(re.findall(r"(?m)^##\s+Part ([A-Z])\.", "\n".join(_asrc)))
for _t in _files:
    for _m in re.finditer(r"Multimedia Appendix (\d)[, ]+Part ([A-Z])", _t):
        _n, _pt = _m.group(1), _m.group(2)
        need(_n == "1", f"'{_m.group(0)}' - Part 는 Multimedia Appendix 1 에만 있다")
        need(_pt in _declared, f"'{_m.group(0)}' - 부록에 'Part {_pt}.' 제목이 없다")

# 부록 안의 표 번호는 S4 부터 빠짐없이 이어져야 한다
_snum = sorted({int(x) for x in re.findall(r"(?m)^#{2,4}\s+Table S(\d+)\.", "\n".join(_asrc))})
need(_snum == list(range(1, len(_snum) + 1)), f"부록 표 번호가 S4 부터 연속이 아님: {_snum}")

# 가중 여부 규약이 부록 어딘가에 한 번은 있어야 한다(없으면 수치의 해석 근거가 사라진다)
need(any("complex-survey weighted unless" in t for t in _asrc),
     "부록에 가중치 규약 문장이 없다 ('complex-survey weighted unless ...')")

# 부록 각 파일은 제목으로 시작해야 한다(제목 없이 뜬 문단은 구조가 어긋난 것이다)
for _fn, _src in zip(_APPX, _asrc):
    need(_src.lstrip().startswith("#"), f"{_fn} 이 제목이 아니라 본문으로 시작한다")
# JMIR 규정 인용 문법: 항목이 먼저, 부록 번호가 뒤 ("Table S14 in Multimedia Appendix 1").
# support.jmir.org 의 supplementary-files 안내가 이 어순만 예시로 든다.
for _t, _nm in [(M, "manuscript")] + list(zip(_asrc, _APPX)):
    for _m in re.finditer(r"Multimedia Appendix \d, (?:Table|Figure)", _t):
        need(False, f"[{_nm}] 인용 어순이 규정과 반대: '{_m.group(0)}' - '<항목> in Multimedia Appendix N' 로 쓴다")

# 부록 표는 모두 원고 본문에서 인용돼야 한다.
# JMIR 규정은 부록 항목을 '본문 등장 순서' 로 번호 매기라고 한다. 등장하지 않는 표는
# 번호를 매길 근거가 없고, 심사자가 '왜 실려 있는지' 를 묻는 자리가 된다.
_body_only = M.split('## Figure Legends')[0]
for _n in _snum:
    need(re.search(r'Table S' + str(_n) + r'(?![0-9])', _body_only) is not None,
         f'Table S{_n} 이 원고 본문에서 인용되지 않는다')

# 부록 표는 본문 등장 순서로 번호를 매긴다(JMIR 규정). 어긋나면 심사자가 바로 본다.
_body_order, _seen_t = [], set()
for _m in re.finditer(r"Table S(\d+)", M.split("## Figure Legends")[0]):
    _k = int(_m.group(1))
    if _k not in _seen_t: _seen_t.add(_k); _body_order.append(_k)
need(_body_order == sorted(_body_order),
     f"부록 표 번호가 본문 등장 순서와 다름: {_body_order} (S1 부터 차례로 나와야 한다)")

# 부록 안에서도 표가 번호 순으로 놓여야 한다
_phys = [int(x) for x in re.findall(r"(?m)^#{2,4}\s+Table S(\d+)\.", "\n".join(_asrc))]
need(_phys == sorted(_phys), f"부록 안 표 배치 순서가 번호순이 아님: {_phys}")

# 범위 표기가 뒤집히지 않았는지(재번호할 때 양끝만 바꾸면 'S4 to S3' 가 된다)
for _t, _nm in [(M, "manuscript")] + list(zip(_asrc, _APPX)):
    for _m in re.finditer(r"Tables? S(\d+)\s*(?:to|-)\s*S?(\d+)", _t):
        need(int(_m.group(1)) < int(_m.group(2)), f"[{_nm}] 표 범위가 뒤집힘: {_m.group(0)}")

# 편집 이력을 드러내는 표현은 제출본에 있으면 안 된다
for _src2, _nm2 in list(zip(_asrc, _APPX)) + [(M, "manuscript")]:
    for _w in ("relocated", "trimmed Methods", "condensed out of", "in revision", "preserved here"):
        need(_w.lower() not in _src2.lower(), f"[{_nm2}] 편집 이력 표현 잔존: {_w}")

# 한 문장에 'Multimedia Appendix N' 이 두 번 나오면 치환이 앞말을 남긴 것이다.
# 죽은 라벨만 갈아끼우고 그것을 소개하던 구절을 그대로 둬서 실제로 두 번 생겼다.
for _t, _nm in [(M, "manuscript")] + list(zip(_asrc, _APPX)) + [(rd("TRIPOD_AI_checklist.md"), "TRIPOD")]:
    for _sent in re.split(r"(?<=[.;|])\s+", _t):
        if _sent.count("Multimedia Appendix") > 1 and len(_sent) < 400:
            need(False, f"[{_nm}] 한 문장에 부록 이름이 두 번: {' '.join(_sent.split())[:96]}")

# 목록 항목이 그대로 반복되면 안 된다(모델카드에서 같은 줄이 20번 나온 적이 있다)
for _f2 in ("model_card_17var.md", "cover_letter.md"):
    _b = [" ".join(x.split()) for x in rd(_f2).split(chr(10)) if x.strip().startswith("- ") and len(x) > 40]
    need(len(_b) == len(set(_b)), f"[{_f2}] 같은 목록 항목이 반복됨 ({len(_b) - len(set(_b))}줄)")

# 사라진 절 라벨을 가리키는 포인터
for _t2, _nm2 in [(M, "manuscript")] + list(zip(_asrc, _APPX)) + [(rd("TRIPOD_AI_checklist.md"), "TRIPOD")]:
    for _m2 in re.finditer(r"(?:§|[Ss]ections?\s)[AB]\.\d", _t2):
        need(False, f"[{_nm2}] 사라진 절 라벨 참조: {_m2.group(0)}")

# 괄호 짝. 표 인용을 문장에 끼워 넣으면서 여는 괄호를 문장으로 바꾸고 닫는 괄호를 남긴 적이 있다.
for _t3, _nm3 in [(M, "manuscript")] + list(zip(_asrc, _APPX)):
    for _i3, _l3 in enumerate(_t3.split(chr(10)), 1):
        if _l3.startswith("|"): continue
        _d3 = 0; _bad3 = False
        for _c3 in _l3:
            if _c3 == "(": _d3 += 1
            elif _c3 == ")":
                _d3 -= 1
                if _d3 < 0: _bad3 = True; break
        need(not _bad3 and _d3 == 0, f"[{_nm3}] {_i3}행 괄호가 맞지 않는다: {' '.join(_l3.split())[:88]}")

print(f"[6] 부록 구조 검사 (Part 선언 {sorted(_declared) or '없음'} · 표 S1-S{len(_snum)})")

print("=" * 80)
if fails:
    for f in fails: print("  FAIL:", f)
    print(f"상호참조 검사 실패 {len(fails)}건"); sys.exit(1)
print("상호참조 검사 통과 - 절 이름·표·그림·참고문헌·약어가 모두 맞습니다.")
