# -*- coding: utf-8 -*-
"""Methods 2차 축소본을 기계적으로 감사한다(감사 에이전트가 한도로 죽어서 대신한다).

에이전트가 하던 검사 네 가지를 스크립트로 한다. 토큰이 들지 않고, 사람 눈보다 정확하다.
  1) 동결 문자열 - 60_verify 가 원고에 요구하는 토큰이 축소본에 그대로 있는가
  2) 수치 누락   - 현재 Methods 의 모든 수치가 축소본이나 Part D 추가분에 남아 있는가
  3) 표 1        - 캡션·표·각주가 글자 단위로 같은가
  4) TRIPOD      - Methods 에 남아 있어야 할 항목이 실제로 남았는가
"""
import io, json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")

J = json.loads(io.open("work_notes/trim_methods2.json", encoding="utf8").read())
NEW = "\n\n".join(f"{s['heading']}\n\n{s['replacement']}" for s in J["subsections"])
PD = J.get("part_d_additions", "")
M = io.open("docs/manuscript_17var.md", encoding="utf8").read()
CUR = M.split("\n## Methods\n")[-1].split("\n## Results")[0]
fails = []

# ── 1) 동결 문자열 ───────────────────────────────────────────────────────
src = io.open("60_verify_manuscript.py", encoding="utf8").read()
src = src.replace('    lab = label or token\n', '    lab = label or token\n    _REC.append((token, tuple(where)))\n', 1)
src = re.sub(r"(?m)^sys\.exit\(.*$", "pass", src)
g = {"__name__": "rec", "_REC": [], "__file__": os.path.abspath("60_verify_manuscript.py")}
try: exec(compile(src, "60_rec", "exec"), g)
except SystemExit: pass
frozen = [t for t, w in g["_REC"] if "manuscript" in w and t in CUR]
for t in frozen:
    if t not in NEW: fails.append(("동결 문자열 사라짐", t[:120]))
print(f"1) 동결 문자열 {len(frozen)}개 중 사라진 것 {sum(1 for f in fails if f[0].startswith('동결'))}건")

# ── 2) 수치 누락 ─────────────────────────────────────────────────────────
NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?%?")
def nums(t): return [m.group(0) for m in NUM.finditer(t)]
from collections import Counter
cur_n, new_n = Counter(nums(CUR)), Counter(nums(NEW) + nums(PD))
lost = {k: v - new_n.get(k, 0) for k, v in cur_n.items() if v > new_n.get(k, 0)}
# 인용번호([21] 등)와 표 안 숫자는 제외 대상이 아니므로 그대로 본다
for k, v in sorted(lost.items(), key=lambda x: -x[1])[:25]:
    ctx = re.search(r".{55}(?<![\w.])" + re.escape(k) + r"(?![\d.]).{55}", CUR)
    fails.append(("수치 누락 x%d" % v, f"{k}  ...{ctx.group(0).strip() if ctx else ''}..."))
print(f"2) 수치 {len(cur_n)}종 중 줄어든 것 {len(lost)}종")

# ── 3) 표 1 ──────────────────────────────────────────────────────────────
def tbl(t):
    cap = re.search(r"(?m)^\*\*Table 1\.\*\*.*$", t)
    rows = [l for l in t.split("\n") if l.startswith("|")]
    j = t.split("\n")
    notes = []
    for i, l in enumerate(j):
        if l.startswith("|") and i + 1 < len(j) and not j[i + 1].startswith("|") and not j[i + 1].strip():
            k = i + 2
            while k < len(j) and j[k].strip() and not j[k].startswith(("|", "#", "**")):
                notes.append(j[k]); k += 1
            break
    return (cap.group(0) if cap else None), rows, notes
c1, r1, n1 = tbl(CUR); c2, r2, n2 = tbl(NEW)
if c1 != c2: fails.append(("표1 캡션 불일치", f"{str(c1)[:70]} != {str(c2)[:70]}"))
if r1 != r2: fails.append(("표1 행 불일치", f"{len(r1)}행 vs {len(r2)}행"))
if n1 != n2: fails.append(("표1 각주 불일치", f"{n1[:1]} != {n2[:1]}"))
print(f"3) 표 1 캡션 {'같음' if c1==c2 else '다름'} · 행 {len(r1)}/{len(r2)} · 각주 {len(n1)}/{len(n2)}")

# ── 4) TRIPOD 필수 항목 ──────────────────────────────────────────────────
MUST = [
 ("자료원과 기간",   r"KCHS 2017-2025|2017-March 2020"),
 ("적격 기준",       r"aged 20 years or older with all nine PHQ-9 items"),
 ("결과 정의",       r"PHQ-9 total of at least 10"),
 ("예측변수 정의",   r"Table 1|17 predictors"),
 ("표본크기·분할",   r"500,000|100,001"),
 ("결측 처리",       r"missing|imputed|median"),
 ("알고리즘",        r"elastic net|random forest|LightGBM|gradient boosting"),
 ("주모형 선택규칙", r"highest cross-validated AUROC|tie|calibration slope"),
 ("성능 지표",       r"AUROC|AUPRC|calibration"),
 ("외부검증·재보정", r"without refitting|recalibrat"),
 ("윤리",            r"exempt|Institutional Review Board|IRB|ethic"),
 ("14+개정 이력",    r"protocol had specified the other fourteen|amendment"),
 ("living alone 단서", r"living alone|Living alone"),
 ("NHANES 미사용",   r"No NHANES data were used|no NHANES outcome association"),
 ("오라클 상한",     r"diagnostic upper bound"),
]
miss = [nm for nm, pt in MUST if not re.search(pt, NEW, re.I)]
for nm in miss: fails.append(("TRIPOD 항목 사라짐", nm))
print(f"4) 필수 항목 {len(MUST)}개 중 사라진 것 {len(miss)}개")

print(f"\n{'='*74}\n총 지적 {len(fails)}건")
for k, v in fails: print(f"  [{k}] {v}")
io.open("work_notes/methods2_audit.txt", "w", encoding="utf8").write(
    "\n".join(f"[{k}] {v}" for k, v in fails))
