# -*- coding: utf-8 -*-
"""A7_trim.py — 본문을 조금 줄인다.

원칙:
  · 부록에 같은 내용이 있는 '근거 설명'만 압축하고, 방법론적 '선택' 자체는 남긴다.
  · 부록에 없는 것(단독층 병합·연도×층 정의·도메인 추정·설계기반 SE 검증)은 손대지 않는다.
  · 감사에서 넣은 공개 서술(예측변수 개정 경위·MEC 프레임·living alone·반복 CV 미실시·
    사후 계산 표기)은 손대지 않는다.
"""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__))
P = f"{D}/docs/manuscript_17var.md"

TRIM = [
 # ① ECE 구간 미보고의 근거 — 표 S16 주석에 같은 설명이 있다
 ("Expected calibration error is reported as a point estimate only: it is a non-negative absolute-deviation statistic that the replicate scheme biases upward, which makes a percentile interval for it untrustworthy; in one cell such an interval did not even contain its own point estimate.",
  "Expected calibration error is reported as a point estimate only, for reasons given with Table S16."),
 # ② 중첩 대 고정 구간의 근거 — 표 S16 주석에 두 행으로 나란히 실려 있다
 ("For the recalibrated columns the interval is obtained by refitting the intercept and slope inside every replicate, so that the uncertainty of the update itself is carried; the alternative, holding the update fixed at the one estimated on the full cohort, gives a wider interval and is reported alongside it in Multimedia Appendix 1, Table S16.",
  "For the recalibrated columns the interval is obtained by refitting the intercept and slope inside every replicate, so that the uncertainty of the update itself is carried; the wider fixed-update alternative is tabulated alongside it in Multimedia Appendix 1, Table S16."),
 # ③ 두 주기 차이 구간의 산출 방식 — 표 S16 아래에 같은 설명이 있다
 ("Since the two cycles are disjoint samples, that interval is formed by differencing independently drawn replicate statistics from the two cohorts rather than by pairing replicate indices, which would impose a correspondence the samples do not have.",
  "Since the two cycles are disjoint samples, that interval differences independently drawn replicate statistics rather than pairing replicate indices."),
 # ④ 사후 후보 선별 — 부록 Part A 에 같은 내용이 표로 있다(중복 95%)
 ("A later exhaustive extraction, reported here for completeness, surfaced further harmonizable items: public-assistance receipt, alcohol frequency, alcohol quantity, binge drinking and weekend sleep duration. After the primary analysis every candidate was screened by comparing the direction of its adjusted odds ratio with depression across the three cohorts and then testing it in the model. Alcohol quantity, binge drinking and weekend sleep reversed direction between countries; public-assistance receipt was directionally consistent but added no external discrimination (ΔAUROC <0.001).",
  "A later exhaustive extraction surfaced five further harmonizable items. Each was screened after the primary analysis by comparing the direction of its adjusted association with depression across the three cohorts and then testing it in the model: three reversed direction between countries and the remainder added no external discrimination (ΔAUROC <0.001; Multimedia Appendix 1, Part A)."),
 # ⑤ 소프트웨어 판본 나열 — 모델카드와 코드에 같은 목록이 있다
 ("All analyses were performed in Python 3.14 (scikit-learn 1.9.0, lightgbm 4.6.0, shap 0.52.0, numpy 2.4.5, pandas 3.0.3, scipy 1.18.0).",
  "All analyses were performed in Python 3.14; package versions are listed in the model card."),
 # ⑥ 조화 주의사항의 도입부 반복
 ("Three harmonization cautions attach to the set. A direct Korea-US comparison",
  "Three cautions attach to the set. A direct Korea-US comparison"),
]

s = io.open(P, encoding="utf8").read()
def wc(t):
    b = t[t.index("## Introduction"):t.index("## Acknowledgments")]
    L = [l for l in b.split("\n") if not l.strip().startswith("|") and not l.strip().startswith("**Table")]
    return len(re.sub(r"[*#\[\]]", "", re.sub(r"## Figure Legends.*", "", "\n".join(L), flags=re.S)).split())

before = wc(s)
n = 0
for a, b in TRIM:
    if a not in s:
        print(f"  [미일치] {a[:80]}"); continue
    s = s.replace(a, b, 1); n += 1
io.open(P, "w", encoding="utf8").write(s)
after = wc(s)
print(f"\n{n}/{len(TRIM)}건 압축 · 서술 본문 {before} -> {after} ({before-after}단어 감소)")

# 보호 대상이 그대로 남아 있는지 확인
for k in ["dated amendment", "mobile examination center estimation frame", "living alone",
          "repeated five times", "computed post hoc", "pseudo-stratum",
          "closed-form design-based", "year crossed with the design stratum"]:
    print(f"  보존 확인 {k[:40]:<42} {k.lower() in s.lower()}")
