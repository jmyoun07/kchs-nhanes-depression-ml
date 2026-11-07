# -*- coding: utf-8 -*-
"""97_audit_fixes_C.py — 독립감사의 나머지 지적 (탐색 실행 표기 · P값 · 경량모델 경계 · 잔여 문구)."""
import io, os, sys, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__)); RES = f"{D}/results"
PAR = pd.read_csv(f"{RES}/primary2_parsimonious.csv").set_index("k")
SENS = pd.read_csv(f"{RES}/primary2_sensitivity.csv").set_index("cond")

k5, k6, k17 = PAR.loc[5], PAR.loc[6], PAR.loc[17]
thr = k17.selection_AUROC - 0.02
print(f"경량모델 경계: 문턱 {thr:.6f} · k=5 {k5.selection_AUROC:.6f} · k=6 {k6.selection_AUROC:.6f} "
      f"· 여유 {k6.selection_AUROC - thr:.6f}")

d16 = SENS.loc["Food insecurity excluded (16)"]; d14 = SENS.loc["Three added predictors excluded (14)"]
d17 = SENS.loc["Primary (17 predictors, corrected)"]
frac_pre = (d17.ext_pre - d16.ext_pre) / (d17.ext_pre - d14.ext_pre)
frac_post = (d17.ext_post - d16.ext_post) / (d17.ext_post - d14.ext_post)
print(f"식품불안정 단독 제거가 차지하는 비중: pre {frac_pre*100:.0f}% · post {frac_post*100:.0f}%")

MS = [
 # 탐색 실행 수치를 주분석 옆에 그대로 두면 자기모순으로 읽힌다 — 기준선 대비 변화로 쓴다
 ("public-assistance receipt alone left external AUROC unchanged (0.772/0.769), weekend sleep duration moved it by 0.001 (0.771/0.769), and entering the six remaining items together (alcohol frequency, alcohol quantity in two codings, binge drinking and two public-assistance indicators) slightly *reduced* it (−0.003 pre, −0.001 post in the exploration run).",
  "public-assistance receipt alone left external AUROC unchanged, weekend sleep duration moved it by 0.001, and entering the six remaining items together (alcohol frequency, alcohol quantity in two codings, binge drinking and two public-assistance indicators) slightly *reduced* it (−0.003 pre, −0.001 post). These are changes against the in-run baseline of a separate exploration run (0.772 pre, 0.769 post; Multimedia Appendix 1, Table S11) and are not comparable in level with the primary estimates."),
 ("(labels unused) *reduced* external AUROC (to 0.766/0.758), showing",
  "(labels unused) *reduced* external AUROC, to 0.766 and 0.758 against that exploration run's baseline of 0.772 and 0.769, showing"),
 ("still yielded 0.823 at n=7,888 (the pre-pandemic cohort size) and 0.810 at n=13,098 (the pooled oracle size), rising to 0.830 only at n=200,000 (best of three learners, 5-fold weighted AUROC).",
  "still yielded 0.823 at n=7,888 (the pre-pandemic cohort size) and 0.810 at n=13,098 (the pooled oracle size), rising to 0.830 only at n=200,000 (best of three learners, 5-fold weighted AUROC, an exploration run separate from the primary analysis)."),
 # P 값: 1000회 절차의 하한을 등호로 쓰면 안 된다
 ("(*P*=0.001 and 0.001; paired rescaled cluster bootstrap, each model given its own hyperparameter search)",
  "(*P*<.001 for both; paired rescaled cluster bootstrap, each model given its own hyperparameter search)"),
 # 경량모델 경계가 아슬아슬함을 밝힌다
 ("Applying the margin set in the analysis plan (the smallest set within 0.02 AUROC of the full model, searched over every count below 17 without imposing the plan's preferred 5–10 item range) selects six items:",
  f"Applying the margin set in the analysis plan (the smallest set within 0.02 AUROC of the full model, searched over every count below 17 without imposing the plan's preferred 5-10 item range) selects six items. The five-item set falls {k17.selection_AUROC - 0.02 - k5.selection_AUROC:.4f} below the threshold and the six-item set clears it by {k6.selection_AUROC - thr:.4f}, so the five-versus-six boundary is decided at the fourth decimal and should not be read as robust; the five-item model reaches {k5.ext_pre:.3f} pre-pandemic and {k5.ext_post:.3f} post-pandemic externally. The selected six items are:"),
 # '대부분' 은 사실이 아니다
 ("Removing food insecurity alone leaves most of that difference intact.",
  f"Removing food insecurity alone accounts for {frac_pre*100:.0f}% of that difference pre-pandemic and {frac_post*100:.0f}% post-pandemic."),
 # '동일하게 측정' 은 조화 주의사항과 어긋난다
 ("the general-health, demographic and sleep items that both surveys measure identically (Figure 5)",
  "the general-health, demographic and sleep items, which carry the closest correspondence between the two surveys, though sleep required a weekday realignment and age a common top-code (Figure 5)"),
]

def edit(path, pairs):
    p = f"{D}/{path}"; s = io.open(p, encoding="utf8").read(); n = 0
    for a, b in pairs:
        if a not in s: print(f"  [미일치] {path}: {a[:80]}"); continue
        s = s.replace(a, b, 1); n += 1
    io.open(p, "w", encoding="utf8").write(s); print(f"  {path}: {n}/{len(pairs)}건")

print("독립감사 교정 C")
edit("docs/manuscript_17var.md", MS)

# 부트스트랩 P 값의 정의를 방법에 한 번 적는다
p = f"{D}/docs/manuscript_17var.md"; s = io.open(p, encoding="utf8").read()
a = "Operating characteristics and net benefit are point estimates."
b = ("Operating characteristics and net benefit are point estimates. Bootstrap P values are two-sided and are read off "
     "the replicate distribution as twice the smaller tail proportion; with 1000 replicates the smallest attainable "
     "value is reported as P<.001 rather than as an equality.")
if a in s and "Bootstrap P values are two-sided" not in s:
    io.open(p, "w", encoding="utf8").write(s.replace(a, b, 1)); print("  부트스트랩 P 값 정의 추가")
