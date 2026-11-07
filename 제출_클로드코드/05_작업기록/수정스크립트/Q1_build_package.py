# -*- coding: utf-8 -*-
"""받는 사람이 열어 보고 재현할 수 있는 패키지를 만든다.

원자료(KCHS 마이크로데이터)는 자료이용협약상 재배포할 수 없으므로 넣지 않는다.
대신 '무엇을 어디서 받아 어떻게 돌리는지' 를 README 에 적는다.
"""
import io, os, re, shutil, sys
sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"C:/Users/rudtn/Desktop/claude code/지역사회건강"
M = f"{ROOT}/17var_model"
OUT = f"{ROOT}/제출_클로드코드"
os.chdir(M)

if os.path.isdir(OUT): shutil.rmtree(OUT)
def mk(p): os.makedirs(f"{OUT}/{p}", exist_ok=True); return f"{OUT}/{p}"

# ── 01 원고·부록 원본과 생성·검사 도구 ─────────────────────────────────
d = mk("01_원고_생성및검증")
mk("01_원고_생성및검증/docs")
for f in os.listdir("docs"):
    if f.endswith(".md"): shutil.copy2(f"docs/{f}", f"{d}/docs/{f}")
for f in ("44_make_docx.py", "45_make_appendix_docx.py", "46_build_submission.py",
          "61_sync_numbers.py", "60_verify_manuscript.py", "92_crossref_audit.py",
          "87_make_ledger.py", "B7_pagecount.py"):
    if os.path.exists(f): shutil.copy2(f, f"{d}/{f}")

# ── 02 분석 ────────────────────────────────────────────────────────────
d = mk("02_분석")
mk("02_분석/src"); mk("02_분석/탐색실험")
for f in os.listdir(f"{ROOT}/src"):
    if f.endswith(".py"): shutil.copy2(f"{ROOT}/src/{f}", f"{d}/src/{f}")
KEEP = re.compile(r"^(20|21|22|23|24|30|31|32|33|34|35|36|37|38|41|42|43|47|48|49|50|51|52|"
                  r"53|54|55|56|70_primary|70_refs|71_utility|72_sensitivity|73_figures|74_shap|"
                  r"90_delta|91_add|93_living|94_dpq)")
for f in sorted(os.listdir(".")):
    if f.endswith(".py") and KEEP.match(f): shutil.copy2(f, f"{d}/{f}")
for f in sorted(os.listdir("wf")):
    if f.endswith(".py"): shutil.copy2(f"wf/{f}", f"{d}/탐색실험/{f}")

# ── 03 결과 ────────────────────────────────────────────────────────────
d = mk("03_결과")
mk("03_결과/그림")
n_csv = 0
for f in sorted(os.listdir("results")):
    if f.endswith((".csv", ".md")): shutil.copy2(f"results/{f}", f"{d}/{f}"); n_csv += 1
for f in sorted(os.listdir("results/figures")):
    if f.endswith(".png"): shutil.copy2(f"results/figures/{f}", f"{d}/그림/{f}")

# ── 04 제출본 ──────────────────────────────────────────────────────────
shutil.copytree(f"{ROOT}/제출본", f"{OUT}/04_제출본")

# ── 05 작업 기록 ───────────────────────────────────────────────────────
d = mk("05_작업기록")
mk("05_작업기록/수정스크립트"); mk("05_작업기록/검토기록")
ONE = re.compile(r"^[A-Q]\d_")
for f in sorted(os.listdir(".")):
    if f.endswith(".py") and ONE.match(f): shutil.copy2(f, f"{d}/수정스크립트/{f}")
for f in sorted(os.listdir(".")):
    if f.endswith(".py") and re.match(r"^(75|76|77|78|79|8\d|89|95|96|97|98|99)_", f):
        shutil.copy2(f, f"{d}/수정스크립트/{f}")
for f in sorted(os.listdir("work_notes")):
    if f.endswith((".json", ".md", ".txt")): shutil.copy2(f"work_notes/{f}", f"{d}/검토기록/{f}")

tot = sum(os.path.getsize(os.path.join(r, x)) for r, _, fs in os.walk(OUT) for x in fs)
cnt = sum(len(fs) for _, _, fs in os.walk(OUT))
print(f"패키지: {OUT}")
for sub in sorted(os.listdir(OUT)):
    p = f"{OUT}/{sub}"
    n = sum(len(fs) for _, _, fs in os.walk(p)) if os.path.isdir(p) else 1
    s = sum(os.path.getsize(os.path.join(r, x)) for r, _, fs in os.walk(p) for x in fs) if os.path.isdir(p) else os.path.getsize(p)
    print(f"  {sub:<24} 파일 {n:>4} · {s/1048576:.1f}MB")
print(f"\n합계 {cnt}개 파일 · {tot/1048576:.1f}MB")
