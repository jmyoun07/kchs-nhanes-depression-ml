# -*- coding: utf-8 -*-
"""
01_harmonize.py (v2, 프로토콜 준거) — KCHS 개발셋 = PHQ-9 전체연도 2017-2025.
- 연도별 변수명 변화를 '라벨 기반' 자동 매핑(크로스워크 QC 출력).
- Table 1 전 예측변수 포함(신체활동='유산소 권장 충족' 보수적 공통정의).
- 자가건강/독거는 NHANES 부재 → 조화모델 제외, KCHS엔 국내모델용으로 보존.
- 복합표본 가중치(W)·strata·psu 보존(가중 평가용).
출력: data/processed/{kchs_dev,nhanes_pre,nhanes_post}.parquet, results/qc_crosswalk.txt
"""
import os, re, sys, numpy as np, pandas as pd, pyreadstat
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
ROOT = r"C:/Users/rudtn/Desktop/claude code/지역사회건강"
KB = os.path.join(ROOT, "data/raw/kchs"); NB = os.path.join(ROOT, "data/raw/nhanes")
PROC = os.path.join(ROOT, "data/processed"); os.makedirs(PROC, exist_ok=True)
QC = []; XW = []
def qc(s): QC.append(str(s)); print(s)
def wl(p):
    p = os.path.abspath(p).replace("/", "\\"); return "\\\\?\\" + p if not p.startswith("\\\\?\\") else p
def clean(s, miss=(), lo=None, hi=None):
    s = pd.to_numeric(s, errors="coerce")
    if miss: s = s.mask(s.isin(list(miss)))
    if lo is not None: s = s.mask(s < lo)
    if hi is not None: s = s.mask(s > hi)
    return s

DEV_YEARS = list(range(2017, 2026))
PHQ_K = [f"mtb_07{c}1" for c in "abcdefghi"]
# 개념 → (우선 시도 변수명, 라벨 정규식 대체탐색)
C = {
 "age": (["age"], r"만나이|연령"), "sex": (["sex"], r"^성별$"),
 "edu": (["sob_01z1"], r"교육\s*수준"),
 "income": (["fma_14z1", "fma_13z1"], r"가구소득.*금액|월.*소득|가구.*소득"),
 "marital": (["sod_02z3"], r"결혼\s*상태|혼인|배우자"),
 "hh": (["mbhld_co"], r"총가구원수|가구원\s*수|세대원\s*수"),
 "smk": (["sma_03z2"], r"현재흡연.*여부"),
 "alc_ever": (["dra_01z1"], r"평생\s*음주\s*여부"), "alc_freq": (["drb_01z3"], r"연간\s*음주\s*빈도"),
 "pa_vd": (["pha_04z1"], r"격렬.*신체활동\s*일수"), "pa_vh": (["pha_05z1"], r"격렬.*시간\(시\)"), "pa_vm": (["pha_06z1"], r"격렬.*시간\(분\)"),
 "pa_md": (["pha_07z1"], r"중등도.*신체활동\s*일수"), "pa_mh": (["pha_08z1"], r"중등도.*시간\(시\)"), "pa_mm": (["pha_09z1"], r"중등도.*시간\(분\)"),
 "pa_wd": (["phb_01z1"], r"걷기.*실천\s*일수"),
 "slp_wd": (["mtc_17z1", "mtc_01z1"], r"평균\s*수면시간.*주중|^수면시간$"), "slp_we": (["mtc_18z1"], r"평균\s*수면시간.*주말"),
 "bmi": (["oba_bmi"], r"BMI\(자가보고\)"), "ht": (["oba_02z1"], r"^키$"), "wtkg": (["oba_03z1"], r"몸무게"),
 "srh": (["qoa_01z1"], r"주관적\s*건강수준"), "stress": (["mta_01z1"], r"주관적\s*스트레스\s*수준"),
 "htn": (["hya_04z1"], r"고혈압\s*진단\s*경험"), "dm": (["dia_04z1"], r"당뇨병\s*진단\s*경험"),
 "strata": (["kstrata"], r"층화"), "wt": (["wt_p"], r"개인가중치|개인\s*가중"), "psu": (["SPOT_NO"], r"표본지점"),
}

def resolve(names, rx, n2l, avail):
    for n in names:
        if n in avail: return n
    r = re.compile(rx)
    for n, l in n2l.items():
        if l and r.search(l): return n
    return None

def build_kchs_year(year):
    dd = os.path.join(KB, str(year), "data")
    f = [x for x in sorted(os.listdir(dd)) if x.lower().endswith(".sas7bdat") and "ctv" not in x.lower()][0]
    path = os.path.join(dd, f)
    _, meta = pyreadstat.read_sas7bdat(wl(path), metadataonly=True)
    n2l = dict(zip(meta.column_names, meta.column_labels or [""] * len(meta.column_names))); A = set(meta.column_names)
    pick = {k: resolve(v[0], v[1], n2l, A) for k, v in C.items()}
    XW.append(f"{year}: " + ", ".join(f"{k}={pick[k]}" for k in ["income","marital","hh","slp_wd","bmi","ht","wt","psu","pa_md"]))
    need = [v for v in pick.values() if v] + [c for c in PHQ_K if c in A]
    df, _ = pyreadstat.read_sas7bdat(wl(path), usecols=list(dict.fromkeys(need)))
    g = lambda k: df[pick[k]] if pick[k] and pick[k] in df.columns else pd.Series(np.nan, index=df.index)
    o = pd.DataFrame(index=df.index); o["year"] = year
    phq = df[[c for c in PHQ_K if c in df.columns]].apply(pd.to_numeric, errors="coerce").mask(lambda x: x.isin([7, 8, 9])) - 1
    _pv = phq.to_numpy(); _pv = _pv[~np.isnan(_pv)]  # 코덱스 #8: 재코딩 범위 QC(1-4→0-3 가정 검증)
    if _pv.size and not ((_pv >= 0) & (_pv <= 3)).all():
        qc(f"[경고] KCHS {year} PHQ 재코딩 범위 이탈 min={_pv.min():.0f} max={_pv.max():.0f} — 코딩 확인 필요")
    comp = phq.notna().all(axis=1) & (phq.shape[1] == 9)
    o["phq_sum"] = phq.sum(axis=1).where(comp); o["dep"] = (o["phq_sum"] >= 10).astype(float).where(comp)
    o["age"] = clean(g("age"), lo=1, hi=120); o["sex"] = clean(g("sex")).map({1: "M", 2: "F"})
    o["edu"] = clean(g("edu"), (77, 99)).map({1: "elem_low", 2: "elem_low", 3: "elem_low", 4: "middle", 5: "high", 6: "college", 7: "college", 8: "college"})
    # [버그수정#3] 77777(응답거부) 누락 → 17,354건이 최상위 소득분위로 편입되던 결함. 7/9 계열 센티넬 전체 보강.
    o["income_raw"] = clean(g("income"), (7777, 9999, 77777, 88888, 99999, 777777, 888888, 999999, 8888))
    o["income_q"] = pd.qcut(o["income_raw"], 5, labels=False, duplicates="drop")  # 연도별 상대분위(척도 상이 보정)
    o["partnered"] = clean(g("marital"), (7, 9)).map({1: 1, 2: 0, 3: 0, 4: 0, 5: 0})
    hh = pd.to_numeric(g("hh"), errors="coerce"); o["living_alone"] = (hh == 1).astype(float).where(hh.notna())
    o["smoking"] = clean(g("smk"), (7, 9)).map({1: "current", 2: "current", 3: "former", 8: "never"})
    ae, af = clean(g("alc_ever"), (7, 9)), clean(g("alc_freq"), (7, 9))
    o["current_drinker"] = np.where((ae == 1) & (af.between(1, 6)), 1.0, np.where(ae.notna(), 0.0, np.nan))
    # 신체활동: 유산소 권장 충족(주당 중등도>=150분 or 격렬>=75분). 분 변수 없으면 일수 근사.
    def mins(dv, hv, mv):
        d = clean(g(dv), (77, 88, 99), lo=0, hi=7); h = clean(g(hv), (77, 88, 99), lo=0, hi=24); mm = clean(g(mv), (77, 88, 99), lo=0, hi=60)
        dur = np.where(h.notna() | mm.notna(), h.fillna(0) * 60 + mm.fillna(0), np.nan)  # 지속시간 완전결측→NaN(0 아님)
        return d * pd.Series(dur, index=d.index)
    vig = mins("pa_vd", "pa_vh", "pa_vm"); mod = mins("pa_md", "pa_mh", "pa_mm")
    meets = (mod >= 150) | (vig >= 75)
    if pick["pa_mh"] is None:  # 분 없으면 일수 근사
        vd = clean(g("pa_vd"), (77, 88, 99), lo=0, hi=7); md = clean(g("pa_md"), (77, 88, 99), lo=0, hi=7); wd = clean(g("pa_wd"), (77, 88, 99), lo=0, hi=7)
        meets = (md >= 5) | (vd >= 3) | (wd >= 5); any_pa = md.notna() | vd.notna() | wd.notna()
    else:
        any_pa = clean(g("pa_md"), (88, 99)).notna() | clean(g("pa_vd"), (88, 99)).notna()
    o["active"] = meets.astype(float).where(any_pa)
    swd = clean(g("slp_wd"), (88, 99), 0, 24); swe = clean(g("slp_we"), (88, 99), 0, 24)
    o["sleep_hrs"] = np.where(swe.notna() & swd.notna(), (5 * swd + 2 * swe) / 7, swd)
    bmi = clean(g("bmi"), lo=10, hi=70); ht = clean(g("ht"), lo=100, hi=220); wt = clean(g("wtkg"), lo=20, hi=250)
    comp_b = wt / (ht / 100) ** 2; o["bmi"] = bmi.where(bmi.notna(), comp_b.where((comp_b >= 10) & (comp_b <= 70)))
    o["htn"] = clean(g("htn"), (7, 9)).map({1: 1, 2: 0}); o["dm"] = clean(g("dm"), (7, 9)).map({1: 1, 2: 0})
    o["srh"] = clean(g("srh"), (7, 9), 1, 5); o["stress"] = clean(g("stress"), (7, 9), 1, 5); o["race"] = np.nan
    o["W"] = pd.to_numeric(g("wt"), errors="coerce"); o["strata"] = g("strata").astype(str); o["psu"] = g("psu").astype(str)
    return o

def pa_nhanes_pre(m):
    def c(x): return clean(m.get(x), (7, 9, 7777, 9999, 77, 99))
    vig = np.where(c("PAQ650") == 1, c("PAQ655").fillna(0) * c("PAD660").fillna(0), 0)
    mod = np.where(c("PAQ665") == 1, c("PAQ670").fillna(0) * c("PAD675").fillna(0), 0)
    meets = (mod >= 150) | (vig >= 75)
    any_pa = c("PAQ650").notna() | c("PAQ665").notna()
    return pd.Series(np.where(any_pa, meets.astype(float), np.nan), index=m.index)

def pa_nhanes_post(m):
    def c(x): return clean(m.get(x), (7, 9, 7777, 9999))
    def unit_factor(s):
        """[버그수정#2] PAQ_L의 PAD790U/PAD810U는 **문자형**('D','W','M','Y').
        기존 코드는 pd.to_numeric으로 파싱해 전량 NaN→fillna(0)→post 신체활동 100% '0분'(전원 비활동)이 됐다.
        문자 코드를 주당 환산계수로 직접 매핑한다. 숫자 코드(1~4)로 오는 경우도 함께 허용."""
        u = m.get(s)
        if u is None: return pd.Series(np.nan, index=m.index)
        us = pd.Series(u, index=m.index).astype(str).str.strip().str.upper()
        fac = us.map({"D": 7.0, "W": 1.0, "M": 1 / 4.3, "Y": 1 / 52,
                      "1": 7.0, "2": 1.0, "3": 1 / 4.3, "4": 1 / 52})
        return pd.to_numeric(fac, errors="coerce")
    def wk(qs, us, ms):
        q = c(qs); mn = c(ms); f = unit_factor(us)
        return q * f * mn   # (빈도 × 주당환산) × 1회 분 = 주당 분
    mod = wk("PAD790Q", "PAD790U", "PAD800"); vig = wk("PAD810Q", "PAD810U", "PAD820")
    meets = (mod.fillna(0) >= 150) | (vig.fillna(0) >= 75)
    any_pa = c("PAD790Q").notna() | c("PAD810Q").notna()
    return pd.Series(np.where(any_pa, meets.astype(float), np.nan), index=m.index)

def build_nhanes(period, folder, namer):
    def rd(mod, cols):
        """[버그수정#1-a] pyreadstat은 usecols에 없는 컬럼을 요청해도 **에러 없이 조용히 누락**시킨다.
        DMDHHSIZ가 P_DEMO에 부재한 것을 이 침묵 때문에 놓쳐 living_alone이 pre에서 전량 날조됐다.
        요청 대비 실제 반환 컬럼을 대조해 누락을 QC에 명시 기록한다."""
        path = os.path.join(NB, folder, namer(mod))
        df = pyreadstat.read_xport(path, usecols=["SEQN"] + cols, encoding="latin1")[0]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            qc(f"  [!] [{period}/{mod}] 원자료에 부재하여 누락된 컬럼: {missing}")
        return df
    wtmec = "WTMECPRP" if period == "pre" else "WTMEC2YR"
    demo = rd("DEMO", ["RIAGENDR", "RIDAGEYR", "DMDEDUC2", "DMDMARTZ", "DMDHHSIZ", "INDFMPIR", "RIDRETH3", wtmec, "SDMVSTRA", "SDMVPSU"])
    parts = [rd("DPQ", [f"DPQ0{i}0" for i in range(1, 10)]), rd("SMQ", ["SMQ020", "SMQ040"]),
             rd("ALQ", ["ALQ111", "ALQ121", "ALQ130"]), rd("BMX", ["BMXBMI"]), rd("BPQ", ["BPQ020"]),
             rd("DIQ", ["DIQ010"]), rd("SLQ", ["SLD012"])]
    if period == "pre":
        parts.append(rd("PAQ", ["PAQ650", "PAQ655", "PAD660", "PAQ665", "PAQ670", "PAD675"]))
    else:
        parts.append(rd("PAQ", ["PAD790Q", "PAD790U", "PAD800", "PAD810Q", "PAD810U", "PAD820"]))
    try:
        parts.append(rd("HUQ", ["HUQ010"]))   # 자가건강: NHANES=HUQ·HUQ010 (Table1의 HSQ·HSD010 오류 정정). 미다운로드면 스킵
    except Exception:
        pass
    m = demo
    for d in parts: m = m.merge(d, on="SEQN", how="left")
    for col in (["DMDHHSIZ", "HUQ010", wtmec] + [f"DPQ0{i}0" for i in range(1, 10)]):
        if col not in m.columns: m[col] = np.nan
    o = pd.DataFrame(index=m.index); o["period"] = period
    phq = m[[f"DPQ0{i}0" for i in range(1, 10)]].apply(pd.to_numeric, errors="coerce").mask(lambda x: x.isin([7, 9]))
    comp = phq.notna().all(axis=1); o["phq_sum"] = phq.sum(axis=1).where(comp); o["dep"] = (o["phq_sum"] >= 10).astype(float).where(comp)
    o["age"] = m["RIDAGEYR"]; o["sex"] = m["RIAGENDR"].map({1: "M", 2: "F"})
    o["edu"] = clean(m["DMDEDUC2"], (7, 9)).map({1: "elem_low", 2: "middle", 3: "high", 4: "college", 5: "college"})
    o["income_raw"] = m["INDFMPIR"]
    o["income_q"] = pd.qcut(o["income_raw"], 5, labels=False, duplicates="drop")
    o["partnered"] = clean(m["DMDMARTZ"], (77, 99)).map({1: 1, 2: 0, 3: 0})
    # [버그수정#1-b] P_DEMO(pre)에는 DMDHHSIZ가 아예 없다(29컬럼 전수확인). 기존엔 전량 NaN이 된 뒤
    # 학습 파이프라인의 최빈값 대체가 걸려 pre 7,888명 전원이 '비독거(0)'로 **날조**됐다.
    # 여기서는 NaN을 그대로 보존하고(대체는 하지 않음), 결측 사실을 QC에 명시한다.
    o["living_alone"] = (m["DMDHHSIZ"] == 1).astype(float).where(m["DMDHHSIZ"].notna())
    if o["living_alone"].isna().all():
        qc(f"  [!] [{period}] living_alone 100% 결측 — DMDHHSIZ 부재. 최빈값 대체 시 상수 날조가 되므로 "
           f"모델링 단계에서 결측 유지 또는 변수 제외 필요(민감도로 정량화).")
    s2, s4 = clean(m["SMQ020"], (7, 9)), clean(m["SMQ040"], (7, 9))
    sm = pd.Series(np.nan, index=m.index, dtype=object)
    sm[s2 == 2] = "never"; sm[(s2 == 1) & (s4 == 3)] = "former"; sm[(s2 == 1) & (s4.isin([1, 2]))] = "current"; o["smoking"] = sm
    a1, a2 = clean(m["ALQ111"], (7, 9)), clean(m["ALQ121"], (77, 99))
    o["current_drinker"] = np.where((a1 == 1) & (a2.between(1, 10)), 1.0, np.where(a1.notna(), 0.0, np.nan))
    o["active"] = pa_nhanes_pre(m) if period == "pre" else pa_nhanes_post(m)
    o["sleep_hrs"] = clean(m["SLD012"], lo=0, hi=24); o["bmi"] = clean(m["BMXBMI"], lo=8, hi=80)
    o["htn"] = clean(m["BPQ020"], (7, 9)).map({1: 1, 2: 0}); o["dm"] = clean(m["DIQ010"], (7, 9)).map({1: 1, 2: 0, 3: 0})
    o["srh"] = clean(m["HUQ010"], (7, 9), 1, 5); o["stress"] = np.nan; o["race"] = m["RIDRETH3"]
    o["W"] = pd.to_numeric(m[wtmec], errors="coerce"); o["strata"] = m["SDMVSTRA"].astype(str); o["psu"] = m["SDMVPSU"].astype(str)
    return o

def finalize(df, name):
    n0 = len(df); df = df[df["age"] >= 20]; df = df[df["dep"].notna()]
    qc(f"[{name}] 성인20+·PHQ완결 n={len(df)} (원 {n0}) | 우울 유병률 {df['dep'].mean():.3%}")
    pcols = ["age", "sex", "edu", "income_raw", "partnered", "smoking", "current_drinker", "active", "sleep_hrs", "bmi", "htn", "dm"]
    qc("  결측%: " + ", ".join(f"{c}={df[c].isna().mean()*100:.1f}" for c in pcols))
    return df

kfr = [build_kchs_year(y) for y in DEV_YEARS]
for d in kfr: qc(f"KCHS {int(d['year'].iloc[0])}: n={len(d)} 완결={int(d['dep'].notna().sum())} 유병={d['dep'].mean():.3%} active결측%={d['active'].isna().mean()*100:.0f} bmi결측%={d['bmi'].isna().mean()*100:.0f}")
kchs = finalize(pd.concat(kfr, ignore_index=True), "KCHS dev 2017-2025")
kchs["W"] = kchs.groupby("year")["W"].transform(lambda w: w / w.sum())  # #15: 연도별 가중치 정규화(연도 동일 기여, person-period 추정)
kchs.to_parquet(os.path.join(PROC, "kchs_dev.parquet"))
npre = finalize(build_nhanes("pre", "pre_2017_2020", lambda x: f"P_{x}.xpt"), "NHANES pre"); npre.to_parquet(os.path.join(PROC, "nhanes_pre.parquet"))
npost = finalize(build_nhanes("post", "post_2021_2023", lambda x: f"{x}_L.xpt"), "NHANES post"); npost.to_parquet(os.path.join(PROC, "nhanes_post.parquet"))
qc("\n=== 연도별 크로스워크(핵심 변수) ===\n" + "\n".join(XW))
open(os.path.join(ROOT, "results", "qc_crosswalk.txt"), "w", encoding="utf-8").write("\n".join(QC + [""] + XW))
qc("\n저장 완료.")
