# -*- coding: utf-8 -*-
"""81_review7_fixes.py — 코덱스 7차 교차검증 지적의 교정 (11건)."""
import io, os, sys
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__))

MS = [
 # ⑪ 괄호 짝이 맞지 않음
 ("0.822 internal, 0.748 pre, 0.757 post; H4 supported)),", "0.822 internal, 0.748 pre, 0.757 post; H4 supported),"),
 # ① 보정 회복은 점추정에서만 확인된다 — 초록 결론의 'fully' 는 과함
 ("transported to the US and across the pandemic with clinically useful discrimination and fully recoverable calibration.",
  "transported to the US and across the pandemic with clinically useful discrimination and with calibration that intercept/slope updating restored on point estimates."),
 # ② 결론의 단정을 결과·부록의 유보와 맞춘다
 ("The cross-national discrimination ceiling (~0.77) is a property of the transportable predictor set, most consistent with shift",
  "The cross-national discrimination ceiling (~0.77) is best read as a property of the transportable predictor set over the learners tested rather than as a proven maximum, most consistent with shift"),
 # ③ '해마다 수십만 명' 은 KCHS 에 해당하고 NHANES 에는 해당하지 않는다
 ("National health-examination and community-health surveys already collect the sociodemographic, behavioural and clinical items such a tool would need, and are administered to hundreds of thousands of respondents each year, which makes such surveys an attractive development base",
  "National health-examination and community-health surveys already collect the sociodemographic, behavioural and clinical items such a tool would need; community-health surveillance systems in particular reach hundreds of thousands of respondents each year, which makes such surveys an attractive development base"),
 # ⑦ KNHANES 를 처음 쓰는 자리에서 정의
 ("including the KCHS and the Korea National Health and Nutrition Examination Survey [5,6]",
  "including the KCHS and the Korea National Health and Nutrition Examination Survey (KNHANES) [5,6]"),
 # ④ 모형에 들어간 나이는 80 세에서 절단된다
 ("| 1 | Age | age | RIDAGEYR | Continuous |", "| 1 | Age | age | RIDAGEYR | Continuous, top-coded at 80 years in both surveys |"),
 # ⑤ NHANES 칸도 대표 코드일 뿐이다 (신체활동은 여러 문항에서 파생)
 ("The KCHS column gives one representative variable name per concept, and the full year-specific crosswalk is provided in Multimedia Appendix 1.",
  "Both survey columns give representative variable names per concept; several concepts, physical activity in particular, are derived from more than the codes shown, and the full year- and cycle-specific crosswalk is provided in Multimedia Appendix 1."),
 # ⑥ 결측 비율은 가중되지 않은 레코드 비율이다
 ("All estimates are complex-survey weighted and computed on available cases, that is, respondents with the item recorded; the share missing in each cohort and the exact values are given in Multimedia Appendix 1, Table S4.",
  "Characteristic estimates are complex-survey weighted and computed on available cases, that is, respondents with the item recorded; the accompanying missing shares are unweighted record proportions. Both, with the exact values, are given in Multimedia Appendix 1, Table S4."),
 # ⑩ 물음표 뒤의 마침표
 ("what, why, how, when and where?. Clin Kidney J", "what, why, how, when and where? Clin Kidney J"),
 # ⑧⑨ 약어표: 본문에 없는 항목 제거, 쓰이는 항목 추가
 ("AMA: American Medical Association\n", ""),
 ("CV: cross-validation\n", ""),
 ("DCA: decision curve analysis\n", ""),
 ("OSF: Open Science Framework\n", ""),
 ("PPV: positive predictive value\n", ""),
 ("DPQ: depression screener module (NHANES)\n", "DPQ: depression screener module (NHANES)\nECE: expected calibration error\n"),
 ("KCHS: Korea Community Health Survey\n", "KCHS: Korea Community Health Survey\nKDCA: Korea Disease Control and Prevention Agency\nKNHANES: Korea National Health and Nutrition Examination Survey\n"),
 ("KDCA: Korea Disease Control and Prevention Agency\nLightGBM", "LightGBM"),
]

def edit(path, pairs):
    p = f"{D}/{path}"; s = io.open(p, encoding="utf8").read(); n = 0
    for a, b in pairs:
        if a not in s: print(f"  [미일치] {path}: {a[:76]}"); continue
        s = s.replace(a, b, 1); n += 1
    io.open(p, "w", encoding="utf8").write(s); print(f"  {path}: {n}/{len(pairs)}건")

print("코덱스 7차 지적 교정")
edit("docs/manuscript_17var.md", MS)

# 약어표는 알파벳 순을 유지한다
P = f"{D}/docs/manuscript_17var.md"; s = io.open(P, encoding="utf8").read()
h, ab = s.split("## Abbreviations\n", 1)
lines = [l for l in ab.strip().split("\n") if l.strip()]
lines.sort(key=lambda l: l.split(":")[0].upper())
io.open(P, "w", encoding="utf8").write(h + "## Abbreviations\n" + "\n".join(lines) + "\n")
print(f"  약어표 {len(lines)}항목 · 알파벳 순 정렬")
