# -*- coding: utf-8 -*-
"""66_refs_jmir_style.py — 참고문헌을 JMIR 예시파일 표기로 변환.

예시 표기:
  Li R, Yue Y, Gu X, Xiong L, Luo M, Li L. Title. Psychiatry Res. May 2025;347:116405.
  [doi: 10.1016/j.psychres.2025.116405] [Medline: 39987588]

  · 저자 6명까지 나열, 초과하면 ', et al'
  · 학술지 ISO 약어 · 'Mon DD, YYYY' 또는 'Mon YYYY' 또는 'YYYY'
  · 권(호):페이지 · [doi: ...] [Medline: PMID]
  · PubMed 비등재 자료는 제목과 링크(URL)

모든 서지정보는 PubMed 메타데이터에서 가져오며, 본문 인용 번호(등장순)는 그대로 둔다.
"""
import html, io, json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__))
META = r"C:\Users\rudtn\.claude\projects\C--Users-rudtn-Desktop-claude-code-------\dffa918b-2ff7-4931-9dc0-4af59c5ddb26\tool-results\mcp-plugin_bio-research_pubmed-get_article_metadata-1785686886935.txt"

# 집단저자는 PubMed 저자 배열이 비어 있어 직접 지정한다.
GROUP = {"33069326": "GBD 2019 Diseases and Injuries Collaborators",
         "34634250": "COVID-19 Mental Disorders Collaborators"}
# PubMed 비등재 자료(규칙 B3: 제목과 링크)
NONPUBMED = {
 "SHAP": "Lundberg SM, Lee SI. A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems 30. 2017:4765-4774. URL: https://papers.nips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html",
 "LUMLEY": "Lumley T. Analysis of complex survey samples. J Stat Softw. 2004;9(8):1-19. URL: https://doi.org/10.18637/jss.v009.i08",
}
# 배치 조회에서 빠진 건은 개별 조회 결과를 직접 넣는다.
EXTRA = {
 "36111305": "Wu D, Liu F, Huang S. Assessment of the relationship between living alone and the risk of depression based on longitudinal studies: A systematic review and meta-analysis. Front Psychiatry. Aug 30, 2022;13:954857. [doi: 10.3389/fpsyt.2022.954857] [Medline: 36111305]",
 "32174292": "Pourmotabbed A, Moradi S, Babaei A, Ghavami A, Mohammadi H, Jalili C, et al. Food insecurity and mental health: a systematic review and meta-analysis. Public Health Nutr. Mar 16, 2020;23(10):1778-1790. [doi: 10.1017/S136898001900435X] [Medline: 32174292]",
 "34259616": "Amiri S. Unemployment associated with major depression disorder and depressive symptoms: a systematic review and meta-analysis. Int J Occup Saf Ergon. Aug 05, 2021;28(4):2080-2092. [doi: 10.1080/10803548.2021.1954793] [Medline: 34259616]",
 "39501933": "Karimi P, Zojaji S, Abolghasemi Fard A, Nateghi MN, Mansouri Z, Zojaji R. The impact of oral health on depression: A systematic review. Spec Care Dentist. Nov 06, 2024;45(1):e13079. [doi: 10.1111/scd.13079] [Medline: 39501933]",
 "17099194": "Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. Med Decis Making. 2006;26(6):565-74. [doi: 10.1177/0272989X06295361] [Medline: 17099194]",
 "31649194": "Obermeyer Z, Powers B, Vogeli C, Mullainathan S. Dissecting racial bias in an algorithm used to manage the health of populations. Science. 2019;366(6464):447-453. [doi: 10.1126/science.aax2342] [Medline: 31649194]",
 "32864600": "Futoma J, Simons M, Panch T, Doshi-Velez F, Celi LA. The myth of generalisability in clinical research and machine learning in health care. Lancet Digit Health. Aug 24, 2020;2(9):e489-e492. [doi: 10.1016/S2589-7500(20)30186-2] [Medline: 32864600]",
 "34031906": "Riley RD, Debray TPA, Collins GS, Archer L, Ensor J, van Smeden M, et al. Minimum sample size for external validation of a clinical prediction model with a binary outcome. Stat Med. May 24, 2021;40(19):4230-4251. [doi: 10.1002/sim.9025] [Medline: 34031906]",
}
MON = {"01":"Jan","02":"Feb","03":"Mar","04":"Apr","05":"May","06":"Jun",
       "07":"Jul","08":"Aug","09":"Sep","10":"Oct","11":"Nov","12":"Dec"}

def fmt(a):
    pmid = a["identifiers"]["pmid"]
    au = [x for x in a.get("authors", []) if x.get("last_name")]
    names = [f"{x['last_name']} {x.get('initials','')}".strip() for x in au]
    who = GROUP.get(pmid) or (", ".join(names[:6]) + (", et al" if len(names) > 6 else ""))
    title = html.unescape(a["title"]).strip()
    if title.endswith("."): title = title[:-1]
    j = a["journal"]["iso_abbreviation"]
    pd_ = a.get("publication_date", {}) or {}
    mon = MON.get(pd_.get("month", ""), pd_.get("month", "")); day = pd_.get("day", ""); yr = pd_.get("year", "")
    date = f"{mon} {day}, {yr}" if mon and day else (f"{mon} {yr}" if mon else yr)
    c = a.get("citation", {}) or {}
    loc = c.get("volume", "")
    if c.get("issue"): loc += f"({c['issue']})"
    if c.get("pages"): loc += f":{c['pages']}"
    s = f"{who}. {title}. {j}. {date};{loc}."
    if a.get("doi"): s += f" [doi: {a['doi']}]"
    return s + f" [Medline: {pmid}]"

meta = {a["identifiers"]["pmid"]: fmt(a) for a in json.load(io.open(META, encoding="utf8"))["articles"]}
meta.update(EXTRA)

P = f"{D}/docs/manuscript_17var.md"
s = io.open(P, encoding="utf8").read()
head, refsec = s.split("\n## References", 1)
lines = refsec.split("\n")
out, n_pm, n_url = [], 0, 0
for ln in lines:
    m = re.match(r"^(\d+)\.\s+(.*)$", ln.strip())
    if not m: out.append(ln); continue
    num, body = m.group(1), m.group(2)
    pm = re.search(r"PMID:(\d+)", body)
    if pm and pm.group(1) in meta:
        out.append(f"{num}. {meta[pm.group(1)]}"); n_pm += 1
    elif "papers.nips.cc" in body:
        out.append(f"{num}. {NONPUBMED['SHAP']}"); n_url += 1
    elif "jss.v009" in body:
        out.append(f"{num}. {NONPUBMED['LUMLEY']}"); n_url += 1
    else:
        raise SystemExit(f"[실패] 서지정보를 찾지 못함: {body[:90]}")
io.open(P, "w", encoding="utf8").write(head + "\n## References" + "\n".join(out))
print(f"참고문헌 변환 완료 — PubMed {n_pm}건 · 링크 표기 {n_url}건 (총 {n_pm + n_url})")
for ln in out:
    if re.match(r"^\d+\.", ln.strip()): print("  ", ln[:150])
