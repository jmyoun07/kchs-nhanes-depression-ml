# -*- coding: utf-8 -*-
"""
15_ext_harmonize.py — 확장 후보 8변수를 KCHS / NHANES pre / NHANES post 에서 동일 정의로 추출.
기존 data/processed/{kchs_dev,nhanes_pre,nhanes_post}.parquet 에 병합 → ext_{kchs,pre,post}.parquet

정렬 근거:
 - NHANES: 기존 parquet의 index는 DEMO 파일의 위치 index. 동일 순서로 DEMO를 재독해 SEQN을 복원하고
   SEQN 기준으로 후보 모듈을 merge한 뒤 index로 재정렬. (age 일치 검증 포함)
 - KCHS: 기존 parquet의 index는 연도순 concat(ignore_index)의 위치 index. 동일 순서로 재구성 후 reindex.
   (year 일치 검증 포함)

조화 정의(검증된 코딩 근거는 results/ext_qc.txt 참조):
 employed        KCHS soa_01z1 1→1, 2→0 (7/9 결측)            | NHANES OCD150 {1,2}→1, {3,4}→0 (7/9 결측)
 oral_health     KCHS ora_01z1 1..5 그대로 (7/9 결측)          | NHANES OHQ845 1..5 그대로 (7/9 결측)
 alcohol_freq    '주 2회 이상 음주' 이진. 경계가 양국에서 정확히 일치하는 유일한 지점.
                 KCHS 2017-19 drb_01z2 {4,5}=1 / 2020-25 drb_01z3 {5,6}=1, 그 외 유효값·비음주(8)=0
                 NHANES ALQ121 {1,2,3,4}=1 (주2회 이상), {5..10}=0, 0(연간비음주)=0
 alcohol_qty     '1회 음주량 5잔 이상' 이진. KCHS drb_03z1 {3,4,5}=1, {1,2}=0, 8(비음주)=0
                 NHANES ALQ130 >=5 →1, <5 →0, 비음주(스킵)=0
                 (+ alcohol_qty_drinks 연속: KCHS 범주중앙값 1.5/3.5/5.5/8/12잔, NHANES ALQ130 원값)
 binge           '월 1회 이상 폭음' 이진. KCHS drb_04z1(남)/drb_05z1(여) {3,4,5}=1, {1,2}=0, 8(비음주)=0
                 NHANES ALQ142 {1..7}=1, {8,9,10}=0, 0=0, 비음주(스킵)=0
 welfare         KCHS fma_04z1 1(현재수급)=1, {2(과거),3(비수급)}=0
                 NHANES [정정] FSQ165는 라벨 확인 결과 'HH FS benefit: **ever** received'(평생)이며
                 12개월/현재 수급이 아니다(가중 35%, 비가중 49% — 현재수급으로 볼 수 없는 값).
                 KCHS '현재 수급자'와 맞는 것은 FSD230('currently receive'). 스킵패턴 반영:
                 FSD230=1→1 / FSD230=2 or FSQ012=2 or FSQ165=2 →0. (+ welfare_12mo = FSQ012 기준 보조지표)
 sleep_weekend   KCHS mtc_18z1 (시간, 77/88/99 결측)           | NHANES SLD013 (시간)
 food_insecure   KCHS nue_01z1 {3,4}=1, {1,2}=0               | NHANES FSDAD {3,4}=1, {1,2}=0
"""
import os, re, sys, numpy as np, pandas as pd, pyreadstat
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

ROOT = r"C:/Users/rudtn/Desktop/claude code/지역사회건강"
KB = os.path.join(ROOT, "data/raw/kchs"); NB = os.path.join(ROOT, "data/raw/nhanes")
PROC = os.path.join(ROOT, "data/processed"); RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
QC = []
def qc(s): QC.append(str(s)); print(s)
def wl(p):
    p = os.path.abspath(p).replace("/", "\\"); return "\\\\?\\" + p if not p.startswith("\\\\?\\") else p
def clean(s, miss=(), lo=None, hi=None):
    s = pd.to_numeric(s, errors="coerce")
    if miss: s = s.mask(s.isin(list(miss)))
    if lo is not None: s = s.mask(s < lo)
    if hi is not None: s = s.mask(s > hi)
    return s

EXT = ["employed", "oral_health", "alcohol_freq", "alcohol_qty", "alcohol_qty_drinks",
       "binge", "welfare", "welfare_12mo", "sleep_weekend", "food_insecure"]
DEV_YEARS = list(range(2017, 2026))

# ---------------------------------------------------------------- KCHS
KC = {  # 개념 → (우선 변수명 후보, 라벨 정규식)
 "emp":  (["soa_01z1"], r"경제활동\s*여부"),
 "oral": (["ora_01z1"], r"주관적\s*구강건강"),
 "afq":  (["drb_01z3", "drb_01z2"], r"연간\s*음주\s*빈도"),
 "aqty": (["drb_03z1"], r"음주량$|섭취\s*시\s*음주량"),
 "bnM":  (["drb_04z1"], r"월간\s*폭음.*_남"),
 "bnF":  (["drb_05z1"], r"월간\s*폭음.*_여"),
 "wel":  (["fma_04z1"], r"기초생활수급자\s*여부"),
 "slwe": (["mtc_18z1"], r"평균\s*수면시간.*주말"),
 "food": (["nue_01z1"], r"가구\s*식품\s*안정성"),
 "age":  (["age"], r"만나이|연령"),
}
def resolve(names, rx, n2l, avail):
    for n in names:
        if n in avail: return n
    r = re.compile(rx)
    for n, l in n2l.items():
        if l and r.search(l): return n
    return None

def kchs_path(year):
    dd = os.path.join(KB, str(year), "data")
    f = [x for x in sorted(os.listdir(dd)) if x.lower().endswith(".sas7bdat") and "ctv" not in x.lower()][0]
    return os.path.join(dd, f)

XW = []
def build_kchs_year(year):
    path = kchs_path(year)
    _, meta = pyreadstat.read_sas7bdat(wl(path), metadataonly=True)
    n2l = dict(zip(meta.column_names, meta.column_labels or [""] * len(meta.column_names)))
    A = set(meta.column_names)
    pick = {k: resolve(v[0], v[1], n2l, A) for k, v in KC.items()}
    XW.append(f"{year}: " + ", ".join(f"{k}={pick[k]}" for k in KC))
    need = [v for v in pick.values() if v]
    df, _ = pyreadstat.read_sas7bdat(wl(path), usecols=list(dict.fromkeys(need)))
    got = [c for c in need if c in df.columns]
    if set(need) - set(got): qc(f"  [!] KCHS {year} 반환누락: {sorted(set(need)-set(got))}")
    g = lambda k: df[pick[k]] if pick[k] and pick[k] in df.columns else pd.Series(np.nan, index=df.index)

    o = pd.DataFrame(index=df.index); o["year"] = year; o["_age"] = clean(g("age"), lo=1, hi=120)
    o["employed"] = clean(g("emp"), (7, 9)).map({1: 1.0, 2: 0.0})
    o["oral_health"] = clean(g("oral"), (7, 9), 1, 5)

    # ---- 음주빈도: 2017-19 drb_01z2(1..5, 8=비해당) vs 2020-25 drb_01z3(1=연간비음주,2..6, 8=비해당)
    af = clean(g("afq"), (7, 9))
    if pick["afq"] == "drb_01z2":   # 1=월1회미만 2=월1회 3=월2-4회 4=주2-3회 5=주4회이상
        wk2 = af.isin([4, 5])
    else:                           # drb_01z3: 1=연간비음주 2=월1회미만 3=월1회 4=월2-4회 5=주2-3회 6=주4회이상
        wk2 = af.isin([5, 6])
    valid_af = af.isin([1, 2, 3, 4, 5, 6, 8])
    o["alcohol_freq"] = wk2.astype(float).where(valid_af)

    # ---- 1회 음주량: drb_03z1 1=1~2잔 2=3~4잔 3=5~6잔 4=7~9잔 5=10잔+ 8=비해당(비음주)
    aq = clean(g("aqty"), (7, 9))
    valid_aq = aq.isin([1, 2, 3, 4, 5, 8])
    o["alcohol_qty"] = aq.isin([3, 4, 5]).astype(float).where(valid_aq)
    o["alcohol_qty_drinks"] = aq.map({1: 1.5, 2: 3.5, 3: 5.5, 4: 8.0, 5: 12.0, 8: 0.0}).where(valid_aq)

    # ---- 폭음: drb_04z1(남)/drb_05z1(여) 1=전혀 2=월1회미만 3=월1회 4=주1회 5=거의매일 8=비해당
    bm, bf = clean(g("bnM"), (7, 9)), clean(g("bnF"), (7, 9))
    bn = pd.Series(np.where(bm.isin([1, 2, 3, 4, 5]), bm, np.where(bf.isin([1, 2, 3, 4, 5]), bf, np.nan)), index=df.index)
    skipped = (bm == 8) & ((bf == 8) | bf.isna())      # 남녀 모두 비해당(=연간 비음주)
    o["binge"] = np.where(bn.isin([3, 4, 5]), 1.0, np.where(bn.isin([1, 2]) | skipped, 0.0, np.nan))
    if pick["bnM"] is None and pick["bnF"] is None: o["binge"] = np.nan

    o["welfare"] = clean(g("wel"), (7, 9)).map({1: 1.0, 2: 0.0, 3: 0.0})
    o["welfare_12mo"] = o["welfare"]   # KCHS는 현재 수급만 조사 → 동일값(대조용)
    o["sleep_weekend"] = clean(g("slwe"), (77, 88, 99), 0, 24)
    fi = clean(g("food"), (7, 9))
    o["food_insecure"] = fi.isin([3, 4]).astype(float).where(fi.isin([1, 2, 3, 4]))
    return o

qc("=== KCHS 후보변수 추출 ===")
kparts = [build_kchs_year(y) for y in DEV_YEARS]
for d, y in zip(kparts, DEV_YEARS):
    qc(f"KCHS {y} n={len(d)} | " + ", ".join(f"{c}결측%={d[c].isna().mean()*100:.0f}" for c in EXT))
kext = pd.concat(kparts, ignore_index=True)
del kparts

kchs = pd.read_parquet(os.path.join(PROC, "kchs_dev.parquet"))
qc(f"\nKCHS 재구성 총행={len(kext)} / 기존 parquet index max={kchs.index.max()} n={len(kchs)}")
assert kext.index.max() >= kchs.index.max(), "KCHS 행수 불일치 — 정렬 근거 붕괴"
kal = kext.reindex(kchs.index)
# 정렬 검증
chk_y = (kal["year"].to_numpy() == kchs["year"].to_numpy()).mean()
a1, a2 = kal["_age"].to_numpy(), kchs["age"].to_numpy()
chk_a = np.nanmean((a1 == a2) | (np.isnan(a1) & np.isnan(a2)))
qc(f"KCHS 정렬검증: year 일치율={chk_y:.6f}, age 일치율={chk_a:.6f}")
assert chk_y > 0.9999 and chk_a > 0.9999, "KCHS 정렬 검증 실패"
kout = pd.concat([kchs, kal[EXT]], axis=1)
kout.to_parquet(os.path.join(PROC, "ext_kchs.parquet"))
qc(f"저장: data/processed/ext_kchs.parquet {kout.shape}")

# ---------------------------------------------------------------- NHANES
FOLD = {"pre": ("pre_2017_2020", lambda m: f"P_{m}.xpt"), "post": ("post_2021_2023", lambda m: f"{m}_L.xpt")}
def build_nhanes(period):
    folder, namer = FOLD[period]
    def rd(mod, cols):
        p = os.path.join(NB, folder, namer(mod))
        df = pyreadstat.read_xport(p, usecols=["SEQN"] + cols, encoding="latin1")[0]
        miss = [c for c in cols if c not in df.columns]
        if miss: qc(f"  [!] [{period}/{mod}] 원자료 부재 컬럼: {miss}")
        for c in miss: df[c] = np.nan
        return df
    demo = rd("DEMO", ["RIDAGEYR"])          # 기존 parquet index 복원 기준 (동일 파일·동일 순서)
    m = demo.copy()
    for mod, cols in [("OCQ", ["OCD150"]), ("OHQ", ["OHQ845"]),
                      ("ALQ", ["ALQ111", "ALQ121", "ALQ130", "ALQ142"]),
                      ("FSQ", ["FSQ165", "FSQ012", "FSD230", "FSDAD"]), ("SLQ", ["SLD013"])]:
        m = m.merge(rd(mod, cols), on="SEQN", how="left")
    o = pd.DataFrame(index=m.index); o["_age"] = m["RIDAGEYR"]

    ocd = clean(m["OCD150"], (7, 9))
    o["employed"] = np.where(ocd.isin([1, 2]), 1.0, np.where(ocd.isin([3, 4]), 0.0, np.nan))
    o["oral_health"] = clean(m["OHQ845"], (7, 9), 1, 5)

    a111 = clean(m["ALQ111"], (7, 9))          # 평생 12잔+ 음주 1=예 2=아니오
    a121 = clean(m["ALQ121"], (77, 99))        # 0=연간 비음주, 1=매일 … 10=연1-2회
    never = (a111 == 2)
    # 주 2회 이상 = ALQ121 {1,2,3,4} (매일/거의매일/주3-4회/주2회). 5=주1회는 제외(KCHS 경계와 일치)
    o["alcohol_freq"] = np.where(a121.isin([1, 2, 3, 4]), 1.0,
                        np.where(a121.isin([0, 5, 6, 7, 8, 9, 10]) | never, 0.0, np.nan))
    a130 = clean(m["ALQ130"], (777, 999), 1, 90)
    nondrink = never | (a121 == 0)
    o["alcohol_qty"] = np.where(a130.notna(), (a130 >= 5).astype(float), np.where(nondrink, 0.0, np.nan))
    o["alcohol_qty_drinks"] = np.where(a130.notna(), a130, np.where(nondrink, 0.0, np.nan))
    a142 = clean(m["ALQ142"], (77, 99))
    o["binge"] = np.where(a142.isin([1, 2, 3, 4, 5, 6, 7]), 1.0,
                 np.where(a142.isin([0, 8, 9, 10]) | nondrink, 0.0, np.nan))

    # SNAP: 스킵패턴 FSQ165(ever) → FSQ012(12개월) → FSD230(현재). KCHS '현재 수급자'와 맞추려면 FSD230.
    f165, f012, f230 = clean(m["FSQ165"], (7, 9)), clean(m["FSQ012"], (7, 9)), clean(m["FSD230"], (7, 9))
    o["welfare"] = np.where(f230 == 1, 1.0,
                   np.where((f230 == 2) | (f012 == 2) | (f165 == 2), 0.0, np.nan))
    o["welfare_12mo"] = np.where(f012 == 1, 1.0, np.where((f012 == 2) | (f165 == 2), 0.0, np.nan))
    o["sleep_weekend"] = clean(m["SLD013"], (77, 99), 0, 24)
    fd = clean(m["FSDAD"], (7, 9))
    o["food_insecure"] = fd.isin([3, 4]).astype(float).where(fd.isin([1, 2, 3, 4]))
    return o

qc("\n=== NHANES 후보변수 추출 ===")
outs = {}
for per, pq in [("pre", "nhanes_pre"), ("post", "nhanes_post")]:
    base = pd.read_parquet(os.path.join(PROC, f"{pq}.parquet"))
    ne = build_nhanes(per)
    qc(f"NHANES {per}: DEMO행={len(ne)} / 기존 parquet index max={base.index.max()} n={len(base)}")
    al = ne.reindex(base.index)
    a1, a2 = al["_age"].to_numpy(dtype=float), base["age"].to_numpy(dtype=float)
    chk = np.mean((a1 == a2) | (np.isnan(a1) & np.isnan(a2)))
    qc(f"NHANES {per} 정렬검증: age 일치율={chk:.6f}")
    assert chk > 0.9999, f"NHANES {per} 정렬 검증 실패"
    out = pd.concat([base, al[EXT]], axis=1)
    out.to_parquet(os.path.join(PROC, f"ext_{per}.parquet"))
    outs[per] = out
    qc(f"저장: data/processed/ext_{per}.parquet {out.shape}")

# ---------------------------------------------------------------- 요약 비교
BIN = ["employed", "alcohol_freq", "alcohol_qty", "binge", "welfare", "welfare_12mo", "food_insecure"]
CON = ["oral_health", "sleep_weekend", "alcohol_qty_drinks"]
def stats(df, v):
    s = pd.to_numeric(df[v], errors="coerce"); w = pd.to_numeric(df["W"], errors="coerce")
    ok = s.notna() & w.notna() & (w > 0)
    val = np.nan if ok.sum() == 0 else float(np.average(s[ok], weights=w[ok]))
    return dict(miss=float(s.isna().mean() * 100), val=val, unw=float(s.mean()) if s.notna().any() else np.nan,
                n=int(s.notna().sum()))
rows = []
for v in BIN + CON:
    k, p1, p2 = stats(kout, v), stats(outs["pre"], v), stats(outs["post"], v)
    nh = np.nanmean([p1["val"], p2["val"]])
    if v in BIN:
        lo, hi = min(k["val"], nh), max(k["val"], nh)
        ratio = hi / lo if lo and lo > 0 else np.inf
        match = "good" if ratio < 1.5 else ("fair" if ratio < 2.0 else "poor")
    else:
        ratio = abs(k["val"] - nh) / nh if nh else np.inf
        match = "good" if ratio < 0.10 else ("fair" if ratio < 0.25 else "poor")
    rows.append(dict(var=v, kind="binary" if v in BIN else "cont",
                     kchs=k["val"], pre=p1["val"], post=p2["val"],
                     kchs_miss=k["miss"], pre_miss=p1["miss"], post_miss=p2["miss"],
                     kchs_n=k["n"], pre_n=p1["n"], post_n=p2["n"],
                     ratio=ratio, match=match))
summ = pd.DataFrame(rows)
pd.set_option("display.width", 250)
qc("\n=== 가중 유병률/평균 비교 (binary=비율, cont=평균) ===")
qc(summ.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
summ.to_csv(os.path.join(RES, "ext_candidate_distributions.csv"), index=False, encoding="utf-8-sig")

# ---- 전이 가능성 진단: 변수↔우울 관계가 양국에서 같은 방향·유사 강도인가 (concept shift 검사)
def wm(s, w):
    ok = s.notna() & w.notna() & (w > 0)
    return float(np.average(s[ok], weights=w[ok])) if ok.sum() else np.nan
def logor(df, v):
    """가중 로그오즈비(이진) 또는 SD당 로그오즈비 근사(연속). 부호·크기 비교용."""
    d, w, x = df["dep"], df["W"], pd.to_numeric(df[v], errors="coerce")
    ok = d.notna() & w.notna() & (w > 0) & x.notna()
    if ok.sum() < 50: return np.nan
    d, w, x = d[ok], w[ok], x[ok]
    if v in BIN:
        p1, p0 = wm(d.where(x == 1), w), wm(d.where(x == 0), w)
        if not (0 < p1 < 1 and 0 < p0 < 1): return np.nan
        return float(np.log((p1 / (1 - p1)) / (p0 / (1 - p0))))
    # 연속: 가중 최소제곱 로지스틱 대신 표준화 후 단순 로지스틱 1스텝(부호/스케일 비교 목적)
    z = (x - np.average(x, weights=w)) / (np.sqrt(np.average((x - np.average(x, weights=w)) ** 2, weights=w)) or 1)
    hi, lo = wm(d.where(z >= 0.5), w), wm(d.where(z <= -0.5), w)
    if not (0 < hi < 1 and 0 < lo < 1): return np.nan
    return float(np.log((hi / (1 - hi)) / (lo / (1 - lo))))
tr = []
for v in BIN + CON:
    lk, lp, lq = logor(kout, v), logor(outs["pre"], v), logor(outs["post"], v)
    lnh = np.nanmean([lp, lq])
    same = (np.sign(lk) == np.sign(lp)) and (np.sign(lk) == np.sign(lq))
    rat = abs(lnh) / abs(lk) if lk and abs(lk) > 1e-9 else np.nan
    tr.append(dict(var=v, logOR_kchs=lk, logOR_pre=lp, logOR_post=lq,
                   same_direction=bool(same), strength_ratio_nh_over_kchs=rat))
trans = pd.DataFrame(tr)
qc("\n=== 전이 가능성 진단: 우울과의 (로그)연관 부호·강도 ===")
qc(trans.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
trans.to_csv(os.path.join(RES, "ext_candidate_transferability.csv"), index=False, encoding="utf-8-sig")

qc("\n=== KCHS 연도 커버리지 (비결측 연도) ===")
for v in EXT:
    yrs = kout.groupby("year")[v].apply(lambda s: s.notna().mean() > 0.5)
    qc(f"  {v}: {sorted(yrs[yrs].index.tolist())} ({int(yrs.sum())}개 연도)")

qc("\n=== 연도별 크로스워크 ===\n" + "\n".join(XW))
open(os.path.join(RES, "ext_qc.txt"), "w", encoding="utf-8").write("\n".join(QC))
qc("\n완료.")
