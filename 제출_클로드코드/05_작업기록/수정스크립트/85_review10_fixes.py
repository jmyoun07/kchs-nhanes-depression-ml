# -*- coding: utf-8 -*-
"""85_review10_fixes.py — 코덱스 10차 교차검증 지적의 교정 (텍스트분).

수치가 걸린 두 건은 스크립트를 고쳐 다시 계산한다:
  · 표본특성 CSV 정밀도 2→6자리 (73_figures_primary2.py) — 8.45→8.5 같은 이중반올림 제거
  · 오라클과의 격차는 코호트별로 (61_sync_numbers.py)
"""
import io, os, sys
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__))

MS = [
 # ② 그림 2 패널 B 는 내부 코호트도 재보정한다 (9차에서 내가 쓴 문장이 틀렸다)
 ("; the internal hold-out is shown in panel B for reference, being already well calibrated as transported, and Table 2 therefore reports no recalibrated column for it.",
  ". The same recalibration is applied to all three cohorts in panel B, the internal hold-out included, so that the panels are directly comparable; Table 2 reports no recalibrated column for the internal hold-out because the deployed internal model is the one shown in panel A."),
 # ⑥ 내부 판별력은 0.836 이다
 ("Two deliberate design choices plausibly account for the lower internal discrimination (0.83),",
  "Two deliberate design choices plausibly account for the lower internal discrimination (0.836),"),
 ("That the same predictors carry substantially more signal within Korea (~0.83) than in the US (~0.77)",
  "That the same predictors carry substantially more signal within Korea (0.836 internally, 0.81-0.82 at matched sample size) than in the US (~0.77)"),
]
A2 = [
 # ⑧ 절 이름 치환이 남긴 깨진 상호참조
 ("was differential by education and race/ethnicity; see Results the Results and Limitations the Discussion.",
  "was differential by education and race/ethnicity; see Participant Characteristics in the Results and Strengths and Limitations in the Discussion."),
 ("These comprise: the post-hoc candidate screening described below;",
  "These comprise: the post-hoc candidate screening described below and reported under Predictors and Harmonization in the Methods;"),
]
A4 = [
 # ③ 표 S13 기준행은 표 S12 의 300k 기준선과 같은 실행이다
 ("The gradient-boosting reference row is the same exploration run that anchors Table S11, and its external values (0.769/0.769) therefore differ slightly from the primary model's 0.765/0.771;",
  "The gradient-boosting reference row is the 300,000-record baseline that anchors Table S12, and its external values (0.769/0.769) therefore differ slightly from the primary model's 0.765/0.771;"),
]
MC = [
 # ④ 분할은 군집 우선이고 결과 층화 표본추출이 아니다
 ("The model was fitted on an outcome-stratified random 500,000-record training set, with a separate 100,000-record hold-out",
  "Clusters (year x design stratum x primary sampling unit) were assigned to the hold-out first, giving 100,001 records; the model was then fitted on 500,000 records drawn at random from the remaining clusters"),
]
CK = [
 # ⑨ 항목 3a 의 위치와 낡은 절 이름
 ("| 3a | Background, rationale, references to existing models (refs 5-8) | ✅ | Introduction (¶1–2; refs 1,3,11,12) |",
  "| 3a | Background, rationale, references to existing models | ✅ | Introduction (existing models in ¶2, refs 5-8; dataset shift and the pandemic in ¶3, refs 10-12) |"),
 ("| 12 | External validation approach (geographic/temporal), no refitting | ✅ | External Validation Strategy |",
  "| 12 | External validation approach (geographic/temporal), no refitting | ✅ | External Validation and Sensitivity Analyses |"),
 ("| 13 | Recalibration/updating strategy | ✅ | External Validation Strategy |",
  "| 13 | Recalibration/updating strategy | ✅ | External Validation and Sensitivity Analyses |"),
 ("| 16 | Protocol/registration; amendments | ✅ | Study Design and Reporting Standards,",
  "| 16 | Protocol/registration; amendments | ✅ | Study Design and Data Sources,"),
]

def edit(path, pairs):
    p = f"{D}/{path}"; s = io.open(p, encoding="utf8").read(); n = 0
    for a, b in pairs:
        if a not in s: print(f"  [미일치] {path}: {a[:76]}"); continue
        s = s.replace(a, b, 1); n += 1
    io.open(p, "w", encoding="utf8").write(s); print(f"  {path}: {n}/{len(pairs)}건")

print("코덱스 10차 지적 교정")
for f, pairs in [("docs/manuscript_17var.md", MS), ("docs/appendix2_crosswalk.md", A2),
                 ("docs/appendix4_ceiling.md", A4), ("docs/model_card_17var.md", MC),
                 ("docs/TRIPOD_AI_checklist.md", CK)]:
    edit(f, pairs)

# ① 오라클과의 격차는 pre·post 각각으로 (61_sync_numbers.py 가 생성하는 문장)
p = f"{D}/61_sync_numbers.py"; s = io.open(p, encoding="utf8").read()
a = ''' (r"about 0\\.\\d{3} below its point estimate", f"about {R(orc.oracle_AUROC.max()-PM.loc[PRE,'AUROC'])} below its point estimate"),'''
b = ''' (r"about 0\\.\\d{3} and 0\\.\\d{3} below its point estimate",
  f"about {R(orc.oracle_AUROC.max()-PM.loc[PRE,'AUROC'])} and {R(orc.oracle_AUROC.max()-PM.loc[POST,'AUROC'])} below its point estimate"),'''
if a in s:
    io.open(p, "w", encoding="utf8").write(s.replace(a, b, 1)); print("  61_sync_numbers.py: 오라클 격차를 코호트별로")
else:
    print("  [미일치] 61_sync_numbers.py 오라클 격차 패턴")
q = f"{D}/docs/manuscript_17var.md"; t = io.open(q, encoding="utf8").read()
a2 = "about 0.011 below its point estimate."
b2 = "about 0.011 and 0.006 below its point estimate."
if a2 in t:
    io.open(q, "w", encoding="utf8").write(t.replace(a2, b2, 1)); print("  docs/manuscript_17var.md: 오라클 격차 문장")
