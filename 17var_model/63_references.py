# -*- coding: utf-8 -*-
"""63_references.py — 참고문헌 확장(16 → 30)과 등장순 재번호.

규칙(B1~B4): 본문 번호는 첫 등장 순서, 연속 3개 이상은 범위(4–7)로, PubMed 등재 SCI급 저널만,
논문이 아닌 자료는 제목과 링크. 모든 서지정보는 PubMed 메타데이터와 대조해 확정했다.

동작: 본문 인용을 키(@KEY)로 바꾸고 → 신규 인용을 삽입 → 첫 등장 순서로 번호를 매기고
→ References 절을 다시 쓴다. 한 번 실행하면 되고, 이후 검증은 60_verify_manuscript.py 가 한다.
"""
import io, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__)); P = f"{D}/docs/manuscript_17var.md"

# ── 서지정보 (PMID 는 대조 근거로 남긴다) ──
REFS = {
 "GBD19":     ("GBD 2019 Diseases and Injuries Collaborators. Global burden of 369 diseases and injuries in 204 countries and territories, 1990-2019: a systematic analysis for the Global Burden of Disease Study 2019. Lancet. 2020;396(10258):1204-1222.", "33069326"),
 "COVIDMD":   ("COVID-19 Mental Disorders Collaborators. Global prevalence and burden of depressive and anxiety disorders in 204 countries and territories in 2020 due to the COVID-19 pandemic. Lancet. 2021;398(10312):1700-1712.", "34634250"),
 "KROENKE":   ("Kroenke K, Spitzer RL, Williams JBW. The PHQ-9: validity of a brief depression severity measure. J Gen Intern Med. 2001;16(9):606-613.", "11556941"),
 "LEVIS":     ("Levis B, Benedetti A, Thombs BD. Accuracy of Patient Health Questionnaire-9 (PHQ-9) for screening to detect major depression: individual participant data meta-analysis. BMJ. 2019;365:l1476.", "30967483"),
 "HE":        ("He C, Levis B, Riehm KE, et al. The accuracy of the Patient Health Questionnaire-9 algorithm for screening to detect major depression: an individual participant data meta-analysis. Psychother Psychosom. 2020;89(1):25-37.", "31593971"),
 "NICKSON":   ("Nickson D, Meyer C, Walasek L, Toro C. Prediction and diagnosis of depression using machine learning with electronic health records data: a systematic review. BMC Med Inform Decis Mak. 2023;23(1):271.", "38012655"),
 "WYNANTS":   ("Wynants L, Van Calster B, Collins GS, et al. Prediction models for diagnosis and prognosis of covid-19: systematic review and critical appraisal. BMJ. 2020;369:m1328.", "32265220"),
 "RAMSPEK":   ("Ramspek CL, Jager KJ, Dekker FW, Zoccali C, van Diepen M. External validation of prognostic models: what, why, how, when and where? Clin Kidney J. 2021;14(1):49-58.", "33564405"),
 "TRIPOD":    ("Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. 2024;385:e078378.", "38626948"),
 "PROBAST":   ("Moons KGM, Wolff RF, Riley RD, et al. PROBAST: a tool to assess risk of bias and applicability of prediction model studies: explanation and elaboration. Ann Intern Med. 2019;170(1):W1-W33.", "30596876"),
 "STEYERBERG":("Steyerberg EW, Vergouwe Y. Towards better clinical prediction models: seven steps for development and an ABCD for validation. Eur Heart J. 2014;35(29):1925-1931.", "24898551"),
 "VANCALSTER":("Van Calster B, McLernon DJ, van Smeden M, Wynants L, Steyerberg EW. Calibration: the Achilles heel of predictive analytics. BMC Med. 2019;17(1):230.", "31842878"),
 "VICKERS":   ("Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. Med Decis Making. 2006;26(6):565-574.", "17099194"),
 "SHAP":      ("Lundberg SM, Lee SI. A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems 30. 2017:4765-4774. URL: https://papers.nips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html", None),
 "FINLAYSON": ("Finlayson SG, Subbaswamy A, Singh K, et al. The clinician and dataset shift in artificial intelligence. N Engl J Med. 2021;385(3):283-286.", "34260843"),
 "FUTOMA":    ("Futoma J, Simons M, Panch T, Doshi-Velez F, Celi LA. The myth of generalisability in clinical research and machine learning in health care. Lancet Digit Health. 2020;2(9):e489-e492.", "32864600"),
 "KIMGM":     ("Kim GM, Cha S, Jung M, Kim SK. Machine learning prediction of depression in culturally diverse families: findings from the Korea Community Health Survey. Front Public Health. 2025;13:1666084.", "41048299"),
 "KIMNH":     ("Kim NH, Kim M, Han JS, et al. Machine-learning model for predicting depression in second-hand smokers in cross-sectional data using the Korea National Health and Nutrition Examination Survey. Digit Health. 2024;10:20552076241257046.", "38784054"),
 "LORANT":    ("Lorant V, Deliège D, Eaton W, Robert A, Philippot P, Ansseau M. Socioeconomic inequalities in depression: a meta-analysis. Am J Epidemiol. 2003;157(2):98-112.", "12522017"),
 "ZHAI":      ("Zhai L, Zhang H, Zhang D. Sleep duration and depression among adults: a meta-analysis of prospective studies. Depress Anxiety. 2015;32(9):664-670.", "26047492"),
 "SCHUCH":    ("Schuch FB, Vancampfort D, Firth J, et al. Physical activity and incident depression: a meta-analysis of prospective cohort studies. Am J Psychiatry. 2018;175(7):631-648.", "29690792"),
 "LUPPINO":   ("Luppino FS, de Wit LM, Bouvy PF, et al. Overweight, obesity, and depression: a systematic review and meta-analysis of longitudinal studies. Arch Gen Psychiatry. 2010;67(3):220-229.", "20194822"),
 "FLUHARTY":  ("Fluharty M, Taylor AE, Grabski M, Munafò MR. The association of cigarette smoking with depression and anxiety: a systematic review. Nicotine Tob Res. 2017;19(1):3-13.", "27199385"),
 "WU":        ("Wu D, Liu F, Huang S. Assessment of the relationship between living alone and the risk of depression based on longitudinal studies: a systematic review and meta-analysis. Front Psychiatry. 2022;13:954857.", "36111305"),
 "POURMOT":   ("Pourmotabbed A, Moradi S, Babaei A, et al. Food insecurity and mental health: a systematic review and meta-analysis. Public Health Nutr. 2020;23(10):1778-1790.", "32174292"),
 "AMIRI":     ("Amiri S. Unemployment associated with major depression disorder and depressive symptoms: a systematic review and meta-analysis. Int J Occup Saf Ergon. 2022;28(4):2080-2092.", "34259616"),
 "KARIMI":    ("Karimi P, Zojaji S, Abolghasemi Fard A, Nateghi MN, Mansouri Z, Zojaji R. The impact of oral health on depression: a systematic review. Spec Care Dentist. 2025;45(1):e13079.", "39501933"),
 "OBERMEYER": ("Obermeyer Z, Powers B, Vogeli C, Mullainathan S. Dissecting racial bias in an algorithm used to manage the health of populations. Science. 2019;366(6464):447-453.", "31649194"),
 "RILEY":     ("Riley RD, Debray TPA, Collins GS, et al. Minimum sample size for external validation of a clinical prediction model with a binary outcome. Stat Med. 2021;40(19):4230-4251.", "34031906"),
 "LUMLEY":    ("Lumley T. Analysis of complex survey samples. J Stat Softw. 2004;9(8):1-19. URL: https://doi.org/10.18637/jss.v009.i08", None),
}
OLD2KEY = {1:"GBD19",2:"COVIDMD",3:"KROENKE",4:"TRIPOD",5:"PROBAST",6:"STEYERBERG",7:"VANCALSTER",
           8:"VICKERS",9:"SHAP",10:"FINLAYSON",11:"KIMGM",12:"KIMNH",13:"POURMOT",14:"AMIRI",
           15:"KARIMI",16:"LUMLEY"}

s = io.open(P, encoding="utf8").read()
body, refsec = s.split("\n## References", 1)

# ── 1. 기존 숫자 인용 → 키 ──
def to_key(m):
    nums = [int(x) for x in m.group(1).replace(" ", "").split(",")]
    return "[" + ",".join("@" + OLD2KEY[n] for n in nums) + "]"
body, n_conv = re.subn(r"\[(\d+(?:\s*,\s*\d+)*)\]", to_key, body)
print(f"[1] 기존 인용 {n_conv}건 → 키 변환")

# ── 2. 신규 인용 삽입 ──
INSERT = [
 ("enable population-scale screening [@KROENKE]",
  "enable population-scale screening [@KROENKE,@LEVIS]"),
 ("However, most published models are assessed only through internal validation.",
  "However, most published models are assessed only through internal validation [@NICKSON,@WYNANTS]."),
 ("it does not establish that a model transports to a different sampling frame, health system, or population",
  "it does not establish that a model transports to a different sampling frame, health system, or population [@RAMSPEK]"),
 ("Current depression = PHQ-9 total ≥10, computed identically in both surveys (nine items, 0–3).",
  "Current depression = PHQ-9 total ≥10, computed identically in both surveys (nine items, 0–3); the ≥10 cut-off is the conventional screening threshold and has been evaluated against semi-structured diagnostic interviews in individual-participant-data meta-analyses [@LEVIS,@HE]."),
 ("self-rated health was harmonized to NHANES HUQ010 (correcting a Table 1 file-code error for HSD010/HSQ).",
  "self-rated health was harmonized to NHANES HUQ010 (correcting a Table 1 file-code error for HSD010/HSQ). "
  "Every harmonized predictor has an established association with depression in the epidemiological literature — socioeconomic position [@LORANT], sleep duration [@ZHAI], "
  "physical activity [@SCHUCH], body mass index [@LUPPINO], smoking [@FLUHARTY] and living alone [@WU] — and the three predictors covered by the protocol amendment are documented in §2.5.1."),
 ("fairness by weighted subgroup AUROC and calibration slope across sex, age, education, and (NHANES) race/ethnicity",
  "fairness by weighted subgroup AUROC and calibration slope across sex, age, education, and (NHANES) race/ethnicity, "
  "since algorithmic performance can differ systematically across groups even when overall accuracy is high [@OBERMEYER]"),
 ("is a concrete instance of the dataset-shift problem described for deployed clinical AI [@FINLAYSON]",
  "is a concrete instance of the dataset-shift problem described for deployed clinical AI [@FINLAYSON,@FUTOMA]"),
 ("but internal estimates are hold-out rather than fully cross-validated.",
  "but internal estimates are hold-out rather than fully cross-validated. The external cycles are also modest in size for a validation study: "
  "the calibration slope is typically the binding precision constraint when the minimum sample size for external validation of a binary-outcome model is derived [@RILEY], "
  "which is consistent with the width of the external intervals reported here."),
]
for a, b in INSERT:
    if a not in body: raise SystemExit(f"[삽입 실패] 앵커 없음: {a[:90]}")
    body = body.replace(a, b, 1)
print(f"[2] 신규 인용 {len(INSERT)}개 지점 삽입")

# ── 3. 첫 등장 순서로 번호 매기기 ──
order, seen = [], set()
for m in re.finditer(r"@([A-Z0-9]+)", body):
    k = m.group(1)
    if k not in seen: seen.add(k); order.append(k)
unused = [k for k in REFS if k not in seen]
if unused: raise SystemExit(f"[번호 실패] 본문에서 인용되지 않은 문헌: {unused}")
num = {k: i + 1 for i, k in enumerate(order)}

def fmt(m):
    ns = sorted(num[k] for k in re.findall(r"@([A-Z0-9]+)", m.group(0)))
    out, i = [], 0
    while i < len(ns):                       # 연속 3개 이상은 범위로 (규칙 B1)
        j = i
        while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1: j += 1
        out.append(f"{ns[i]}-{ns[j]}" if j - i >= 2 else ",".join(str(x) for x in ns[i:j + 1]))
        i = j + 1
    return "[" + ",".join(out) + "]"
body = re.sub(r"\[@[A-Z0-9]+(?:,@[A-Z0-9]+)*\]", fmt, body)
print(f"[3] 참고문헌 {len(order)}개, 등장순 재번호 완료")

# ── 4. References 절 재작성 ──
lines = ["\n## References\n"]
for k in order:
    txt, pmid = REFS[k]
    lines.append(f"{num[k]}. {txt}" + (f" PMID:{pmid}." if pmid else ""))
tail = ""
m = re.search(r"\n\n(?=\*\*|##|---)", refsec)
if m: tail = refsec[m.start():]
io.open(P, "w", encoding="utf8").write(body + "\n".join(lines) + ("\n" + tail.lstrip("\n") if tail.strip() else "\n"))
print(f"[4] References 절 재작성 — 총 {len(order)}개")
