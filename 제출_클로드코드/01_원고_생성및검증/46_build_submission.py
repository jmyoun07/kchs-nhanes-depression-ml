# -*- coding: utf-8 -*-
"""46_build_submission.py - 제출에 필요한 파일만 모아 '제출본' 폴더를 만든다.

17var_model 에는 분석 스크립트·중간 결과가 함께 있어 제출물을 찾기 어렵다.
이 스크립트는 원고·부록·그림·커버레터만 복사해 깨끗한 폴더로 정리한다.
폴더는 매번 새로 만들어지므로, 이 폴더를 직접 고치지 말고 항상 docs/*.md 를 고친 뒤 다시 실행할 것.

실행 순서: 61_sync_numbers → 44/45_make_*.py → 46_build_submission.py
"""
import io, os, re, shutil, sys
sys.stdout.reconfigure(encoding="utf-8")
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH as AL
from docx.oxml.ns import qn

def _clear_props(doc):
    """문서 속성을 비운다. 저자란을 비워 둔 논문인데 워드 속성에 이름이 남으면
    익명 심사에서 신원이 샌다. python-docx 는 기본값으로 자기 이름을 넣는다."""
    cp = doc.core_properties
    for _a in ("author", "last_modified_by", "comments", "category", "keywords",
               "subject", "title", "identifier", "content_status"):
        try: setattr(cp, _a, "")
        except Exception: pass
    cp.revision = 1


D = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(D)
OUT = f"{ROOT}/제출본"
FIG = f"{D}/results/figures"

# 원고는 여기서 복사하지 않는다. 제출할 원고는 01_원고 안의 작업용 파일이고,
# 거기에는 손으로 맞춘 쪽 나눔과 표 행 높이가 들어 있다. 이 스크립트가 덮어쓰면 그 작업이 사라진다.
# 생성 결과물은 17var_model/논문원고_우울ML_17변수.docx 에 그대로 있고, 본문 대조는 그것으로 한다.
DOCX = [("부록1_보충방법및결과.docx",  "02_부록", "Multimedia_Appendix_1_supplementary_methods_results.docx"),
        ("부록2_TRIPOD체크리스트.docx", "02_부록", "Multimedia_Appendix_2_TRIPOD-AI_checklist.docx")]
FIGS = [("characteristics.png", "Figure_1.png"), ("calibration.png", "Figure_2.png"),
        ("roc_curves.png", "Figure_3.png"), ("sensitivity.png", "Figure_4.png"),
        ("shap_importance.png", "Figure_5.png"), ("subgroup_forest.png", "Figure_6.png"),
        ("dca.png", "Figure_7.png")]

# 이 스크립트가 만드는 파일만 지운다.
# 예전에는 제출본 폴더를 통째로 rmtree 했다. 그 폴더에 사용자가 따로 만들어 둔
# 작업용 파일이 있어도 함께 사라졌고, 실제로 한 번 그렇게 지워졌다.
OWNED = ({f"{OUT}/{sub}/{dst}" for _, sub, dst in DOCX}
         | {f"{OUT}/03_그림/{dst}" for _, dst in FIGS}
         | {f"{OUT}/04_커버레터/Cover_Letter.docx", f"{OUT}/README.md"})
for sub in ["01_원고", "02_부록", "03_그림", "04_커버레터"]:
    os.makedirs(f"{OUT}/{sub}", exist_ok=True)
_locked = []
for _p in sorted(OWNED):
    if not os.path.exists(_p): continue
    try:
        os.remove(_p)
    except PermissionError:          # 워드가 열어 두면 지워지지도, 덮어써지지도 않는다
        _locked.append(os.path.relpath(_p, OUT))
if _locked:
    raise SystemExit("워드에서 열려 있어 갱신할 수 없는 파일이 있습니다. 닫고 다시 실행하세요:\n  · "
                     + "\n  · ".join(_locked))

# 남의 파일이 있으면 지우지 말고 알린다
_kept = [os.path.relpath(os.path.join(r, fn), OUT)
         for r, _, fs in os.walk(OUT) for fn in fs
         if os.path.join(r, fn).replace("\\", "/") not in OWNED]
if _kept:
    print("제출본 폴더에 이 스크립트가 만들지 않은 파일이 있습니다(그대로 둡니다):")
    for _k in _kept: print(f"  · {_k}")

missing = []
for src, sub, dst in DOCX:
    p = f"{D}/{src}"
    if not os.path.exists(p): missing.append(src); continue
    shutil.copy2(p, f"{OUT}/{sub}/{dst}")
for src, dst in FIGS:
    p = f"{FIG}/{src}"
    if not os.path.exists(p): missing.append(src); continue
    shutil.copy2(p, f"{OUT}/03_그림/{dst}")

# ── 커버레터를 원고와 같은 서식의 docx 로 ──
doc = Document()
st = doc.styles["Normal"]
st.font.name = "Times New Roman"; st.font.size = Pt(11)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
pf = st.paragraph_format; pf.line_spacing = 2.0; pf.space_before = Pt(0); pf.space_after = Pt(0)
for sec in doc.sections:
    sec.top_margin = sec.bottom_margin = sec.left_margin = sec.right_margin = Inches(1)
TOKEN = re.compile(r"(\*\*.+?\*\*|(?<![\w*])\*[^*\n]+?\*(?![\w*]))")
for ln in io.open(f"{D}/docs/cover_letter.md", encoding="utf8").read().split("\n"):
    t = ln.strip()
    if not t or t == "---": continue
    p = doc.add_paragraph(); p.paragraph_format.alignment = AL.JUSTIFY
    if t.startswith("#"):
        p.add_run(t.lstrip("# ").strip()).bold = True
        continue
    for piece in TOKEN.split(t):
        if not piece: continue
        if piece.startswith("**") and piece.endswith("**"): p.add_run(piece[2:-2]).bold = True
        elif piece.startswith("*") and piece.endswith("*"): p.add_run(piece[1:-1]).italic = True
        else: p.add_run(piece)
_clear_props(doc); doc.save(f"{OUT}/04_커버레터/Cover_Letter.docx")

NREF = len(re.findall(r"^\d+\. ", io.open(f"{D}/docs/manuscript_17var.md", encoding="utf8").read().split("## References")[1].split("## Abbreviations")[0], re.M))
readme = f"""# 제출본 - JMIR 투고 파일

이 폴더는 `17var_model/46_build_submission.py` 가 자동으로 만든다.
**여기 있는 파일을 직접 고치지 말 것.** 내용은 `17var_model/docs/*.md` 를 고친 뒤

    python 61_sync_numbers.py && python 44_make_docx.py && python 45_make_appendix_docx.py && python 46_build_submission.py

순서로 다시 만든다. 수치가 정본 결과와 맞는지는 `python 60_verify_manuscript.py` 가 검사한다.
분석을 다시 돌릴 때는 **70 → 72 → 71 → 73 → 74** 순서를 지킨다. 70 이 primary2_sensitivity.csv 의 앞 네 행을 쓰고
72 가 같은 파일을 전체로 다시 쓰므로, 72 를 70 보다 먼저 돌리면 뒷행이 사라진다.

| 폴더 | 파일 | 내용 |
|---|---|---|
| 01_원고 | 작업용_v3.docx | **제출할 원고.** 본문(초록·서론·방법·결과·고찰·참고문헌 {NREF}편·약어), 표 2개와 그림 7개 삽입. 쪽 나눔과 표 행 높이는 손으로 맞춘 것이므로 이 스크립트가 건드리지 않는다 |
| 02_부록 | Multimedia_Appendix_1_supplementary_methods_results.docx | 보충 방법·결과 - 조화 크로스워크·예측변수 개정·방법 상세, 이어서 Table S1-S16 |
| | Multimedia_Appendix_2_TRIPOD-AI_checklist.docx | TRIPOD+AI 보고 체크리스트 |
| 03_그림 | Figure_1.png ~ Figure_7.png | 본문 그림 원본(제출 시 개별 업로드용) |
| 04_커버레터 | Cover_Letter.docx | 편집장 앞 커버레터 |

서식은 `윤수 예시파일/JMIR Adolescent Depressed Mood and Suicidality.docx` 를 그대로 따랐다
(Times New Roman 11pt · 줄간격 2.0 · 양쪽맞춤 · 첫줄 들여쓰기 0.25" · 문단 앞뒤 공백 0 ·
표 캡션은 표 위, 주석은 표 아래 · 그림 캡션은 그림 위 · 참고문헌 내어쓰기 0.45"·10pt·줄간격 1.5).

## 제출 전 확인
- Data Availability 는 자료·코드·모델객체를 교신저자 요청 시 제공으로 서술했다. 공개 저장소(OSF·Zenodo)에
  코드와 문서를 올려 DOI 를 받으면 그 문장을 DOI 로 바꿔 넣으면 된다. (KCHS 원자료는 자료이용협약상 올릴 수 없다.)
- 그림은 JMIR 규정대로 300 dpi, 가로 300-1200 px, 세로 225-900 px, 5 MB 이하로 저장돼 있다.
- TRIPOD+AI 공식 체크리스트 양식은 저널 제출 화면에서 별도로 올린다(부록 2 는 항목별 대응표).
"""
# 안내문은 제출 파일이 아니므로 제출본 폴더 밖에 쓴다(폴더에 남으면 같이 올라간다).
io.open(f"{D}/제출본_안내.md", "w", encoding="utf8").write(readme)

n = sum(len(f) for _, _, f in os.walk(OUT) for f in [f])
print(f"제출본 폴더 생성: {OUT}")
for sub in sorted(os.listdir(OUT)):
    p = f"{OUT}/{sub}"
    if os.path.isdir(p):
        for f in sorted(os.listdir(p)): print(f"  {sub}/{f}")
    else: print(f"  {sub}")
if missing: print(f"\n[주의] 원본을 찾지 못한 파일: {missing}")
