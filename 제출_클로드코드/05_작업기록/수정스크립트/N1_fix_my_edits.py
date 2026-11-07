# -*- coding: utf-8 -*-
"""내 문자열 치환이 만든 결함 5건을 고친다.

전부 같은 실수다. 죽은 라벨만 골라 바꾸면서 그것을 소개하던 앞말을 그대로 뒀다.
'in Multimedia Appendix 1, sections B.2-B.3' 에서 뒷부분만 갈아끼우니
'in Multimedia Appendix 1, the predictor-set amendment in Multimedia Appendix 1' 이 됐다.
이번에는 문장 전체를 다시 쓴다.
"""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")
D = "docs"
n = 0
def fix(f, a, b, why):
    global n
    p = f"{D}/{f}"; s = io.open(p, encoding="utf8").read()
    if a not in s: print(f"  [앵커없음] {f}: {why}"); return
    io.open(p, "w", encoding="utf8").write(s.replace(a, b, 1)); n += 1
    print(f"  {f.split('_')[0]:<14} {why}")

# ── ①② TRIPOD: 부록 이름이 두 번 나오고 문장이 깨졌다 ──────────────────
fix("TRIPOD_AI_checklist.md",
    "are in Multimedia Appendix 1, the predictor-set amendment in Multimedia Appendix 1.",
    "are in the predictor-set amendment in Multimedia Appendix 1.",
    "9a: 부록 이름 중복 제거")
fix("TRIPOD_AI_checklist.md",
    "with Multimedia Appendix 1 the analytic sample derivation in Multimedia Appendix 1 and S1",
    "with the analytic sample derivation and Table S1 in Multimedia Appendix 1",
    "20a: 부록 이름 중복 + 'and S1' 이 표 표기를 잃은 것")

# ── ③ 살아남은 죽은 라벨 §B.5 ───────────────────────────────────────────
fix("appendix2_crosswalk.md", "(see \u00a7B.5).", "(see the expectations declared in advance, below).",
    "§B.5 - Part 라벨 제거 때 살아남은 마지막 하나")

# ── ④ 문장 첫 글자가 소문자가 됐다 ──────────────────────────────────────
fix("appendix2_crosswalk.md",
    "in each NHANES cycle. the amendment section is an annotated English summary of the predictor-set amendment,",
    "in each NHANES cycle. The amendment section that follows is an annotated English summary of the locked "
    "amendment document,",
    "문장 첫 글자 소문자 + 자기 자신을 요약한다는 순환 표현")

# ── ⑤ 모델카드 중복이 3개 남았다 ────────────────────────────────────────
p = f"{D}/model_card_17var.md"
L = io.open(p, encoding="utf8").read().split("\n")
seen, out, dropped = set(), [], 0
for i, l in enumerate(L):
    k = " ".join(l.split())
    if k.startswith("- ") and len(k) > 40:
        if k in seen:
            dropped += 1
            if out and not out[-1].strip(): out.pop()      # 앞에 남은 빈 줄도 정리
            continue
        seen.add(k)
    out.append(l)
if dropped:
    io.open(p, "w", encoding="utf8").write(re.sub(r"\n{3,}", "\n\n", "\n".join(out)))
    n += 1; print(f"  model_card     같은 항목 {dropped}줄 추가 제거(공백 정규화 기준)")

print(f"\n{n}건 수정")
# 재확인
for f, pat, lab in (("TRIPOD_AI_checklist.md", r"Multimedia Appendix 1[^.|]*Multimedia Appendix 1", "부록 이름 중복"),
                    ("appendix2_crosswalk.md", r"\u00a7[AB]\.\d", "§ 라벨"),
                    ("appendix2_crosswalk.md", r"\. [a-z]", "소문자로 시작하는 문장")):
    hits = re.findall(pat, io.open(f"{D}/{f}", encoding="utf8").read())
    print(f"  {f.split('_')[0]:<14} {lab}: {len(hits)}건")
mc = io.open(p, encoding="utf8").read().split("\n")
bul = [" ".join(x.split()) for x in mc if x.strip().startswith("- ") and len(x) > 40]
print(f"  model_card     중복 항목: {len(bul) - len(set(bul))}건")
