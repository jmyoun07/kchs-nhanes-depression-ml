# -*- coding: utf-8 -*-
"""리뷰어 패널 확정 지적을 반영한다(중복 5건은 F2 에서 이미 처리).

방침: 꼭 필요한 것만 본문에 두고 나머지는 부록을 가리킨다.
TRIPOD 행은 docs/ 가 아니라 work_notes/tripod_map.json 을 고친다. 그 파일이 원본이고
B8_tripod_rebuild.py 가 체크리스트를 다시 만들기 때문이다.
"""
import io, json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model")

def edit(path, pairs, label):
    s = io.open(path, encoding="utf8").read()
    done = 0
    for a, b in pairs:
        if b in s and a not in s:
            done += 1; continue          # 이미 적용됨
        assert a in s, f"[{path}] 앵커 없음: {a[:80]}"
        s = s.replace(a, b, 1); done += 1
    io.open(path, "w", encoding="utf8").write(s)
    print(f"{label}: {done}건")

# ══ 원고 ═══════════════════════════════════════════════════════════════
edit("docs/manuscript_17var.md", [
 # [9] 결정곡선 범위는 Figure 7 에 있다. Table S3 은 두 임계값만 싣는다
 ("weighted decision-curve analysis [35] over threshold probabilities of 0.02 to 0.30 (Table S3).",
  "weighted decision-curve analysis [35] over threshold probabilities of 0.02 to 0.30 (Figure 7), with net "
  "benefit at 0.10 and 0.15 tabulated in Multimedia Appendix 1, Table S3."),
 # [26] 알고리즘 비교의 교차검증 AUROC 는 비가중이다. 나머지는 전부 가중이므로 표시한다
 ("Under the identical tuning protocol, cross-validated AUROC in the training set was 0.840 for LightGBM",
  "Under the identical tuning protocol, unweighted cross-validated AUROC in the training set was 0.840 for LightGBM"),
 # [25] 접힘 수 이탈을 정확히
 ("that departure and single-pass rather than repeated 5-fold cross-validation are recorded in Table S9.",
  "that departure, and the use of single-pass 4-fold cross-validation for the hyperparameter and "
  "feature-representation search and single-pass 5-fold cross-validation for the algorithm comparison in place "
  "of 5-fold cross-validation repeated five times, are recorded in Multimedia Appendix 1, Table S9."),
 # [29] 'stricter-margin seven-item variant' 의 선행사가 잘려나가 있었다
 ("The stricter-margin seven-item variant is exploratory only, and no external information entered the "
  "item-count decision (Multimedia Appendix 1, Tables S2, S3 and S6).",
  "A variant selected under a stricter 0.01 margin adds smoking as a seventh item and is exploratory only, that "
  "margin being post hoc while the plan fixed 0.02; on external data the seventh item does not help "
  "consistently, raising pre-pandemic AUROC to 0.757 while lowering post-pandemic AUROC to 0.752. No external "
  "information entered the item-count decision (Multimedia Appendix 1, Tables S2, S3 and S6)."),
 # [21] 윤리 면제 근거가 Data Availability 와 충돌했다(KCHS 는 접근통제 자료)
 ("on the grounds that the study uses publicly available de-identified data, records no personal identifiers",
  "on the grounds that the study analyzes de-identified secondary survey data obtained under the "
  "data-collecting agencies' release conditions, records no personal identifiers"),
], "원고")

# [22] 공정성: 소득분위 누락과 근거 부재
s = io.open("docs/manuscript_17var.md", encoding="utf8").read()
m = re.search(r"fairness, through weighted subgroup AUROC and calibration slope across [^;]+;", s)
assert m, "공정성 문장 앵커 없음"
s = s[:m.start()] + ("fairness, through weighted subgroup AUROC and calibration slope across sex, age, education, "
                     "household income quintile and, in NHANES, race and ethnicity, undertaken since algorithmic "
                     "performance can differ systematically across groups even where overall accuracy is high "
                     "[38], any subgroup differing from the cohort value by more than 0.05 AUROC being flagged;") + s[m.end():]

# [14] 환자·대중 참여 진술이 어디에도 없었다. 윤리 절에 한 문장
a14 = "poses no more than minimal risk"
i = s.index(a14); j = s.index(".", i) + 1
s = s[:j] + (" Patients and members of the public were not involved in the design, conduct, reporting or "
             "dissemination plans of this secondary analysis of existing survey data.") + s[j:]
io.open("docs/manuscript_17var.md", "w", encoding="utf8").write(s)
print("원고: 공정성 문장 보강 · 환자·대중 참여 진술 추가")

# ══ 부록 ═══════════════════════════════════════════════════════════════
edit("docs/appendix3_utility.md", [
 # [10] 내부 hold-out 을 재보정하지 않는 이유가 자기모순이었다
 ("the internal hold-out uses the original probabilities, since recalibrating on its own labels would be optimistic",
  "the internal hold-out uses the original probabilities, being the deployed model in its own data and already "
  "well calibrated there (calibration slope 1.03, CITL +0.10), so that no update is called for"),
], "부록 Part B")

edit("docs/appendix5_methods.md", [
 ("on the grounds that the study uses publicly available and de-identified data, collects and records no personal identifiers",
  "on the grounds that the study analyzes de-identified secondary survey data obtained under the "
  "data-collecting agencies' release conditions, collects and records no personal identifiers"),
], "부록 Part D")

# [11] 온도 스케일링이 '두 갱신 사이' 가 아니다(사후 1.17 은 1 을 넘어선다)
d = io.open("docs/appendix4_ceiling.md", encoding="utf8").read()
a = "sat between intercept-only and intercept-plus-slope updating: it left the calibration slope at 0.83 pre-pandemic and 1.17 post-pandemic"
if a in d:
    d = d.replace(a, "left less residual miscalibration than intercept-only updating: the calibration slope was "
                     "0.83 pre-pandemic and 1.17 post-pandemic, the post-pandemic value overshooting 1 rather "
                     "than lying between the intercept-only (0.68) and intercept-plus-slope (1.00) results", 1)
    io.open("docs/appendix4_ceiling.md", "w", encoding="utf8").write(d)
    print("부록 Part C: 온도 스케일링 서술 정정('사이' -> 1 을 넘어섬)")

# ══ 모델카드 [27] ═══════════════════════════════════════════════════════
edit("docs/model_card_17var.md", [
 ("Evaluated across sex, age band, education (KCHS) and race/ethnicity (NHANES).",
  "Evaluated across sex, age band, education, household income quintile (KCHS) and race/ethnicity (NHANES)."),
], "모델카드")
