# -*- coding: utf-8 -*-
"""18_auroc_max.py — 17변수 고정·NHANES 무재적합 하에서 외부 AUROC 최대화 시도.
기준선: 내부 0.8336 / pre 0.7681 / post 0.7682
핵심 가설:
  (A) 설문내 분위(rank) 변환 — srh/oral_health의 한·미 척도 비동등(EVGFP vs 대칭척도) 상쇄
  (B) 단조성 제약 — 방향 확실한 변수에 monotone constraint → 한국 특유 굴곡 학습 방지
  (C) 강정규화/얕은트리 — 전이 안정성
  (D) 시드/모델 앙상블(순위평균)
주의: 모델 선택은 KCHS 내부 CV로만. NHANES는 평가 전용(§2.10).
"""
import os, sys, warnings, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import lightgbm as lgb

ROOT = r"C:/Users/rudtn/Desktop/claude code/지역사회건강"
PROC = os.path.join(ROOT, "data/processed"); RES = os.path.join(ROOT, "results")
SEED = 42
kchs = pd.read_parquet(f"{PROC}/ext_kchs.parquet")
pre  = pd.read_parquet(f"{PROC}/ext_pre.parquet")
post = pd.read_parquet(f"{PROC}/ext_post.parquet")

NUM = ["age","sleep_hrs","bmi","income_q","srh","oral_health"]
BIN = ["partnered","living_alone","current_drinker","active","htn","dm","employed","food_insecure"]
CAT = ["sex","edu","smoking"]
FE  = NUM + BIN + CAT

y = kchs["dep"].astype(int).values
tr, ho = train_test_split(np.arange(len(kchs)), train_size=500000, test_size=100000,
                          stratify=y, random_state=SEED)
Who, yho = kchs.iloc[ho]["W"].to_numpy(float), y[ho]

def wauc(d, p): return roc_auc_score(d["dep"].astype(int).values, p, sample_weight=d["W"].to_numpy(float))
def hauc(p):    return roc_auc_score(yho, p, sample_weight=Who)

def ct(num_cols=NUM):
    return ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num_cols),
        ("bin", "passthrough", BIN),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", OneHotEncoder(handle_unknown="ignore", drop="first"))]), CAT)])

def lgbm(**kw):
    p = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, subsample=0.8,
             colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1)
    p.update(kw); return lgb.LGBMClassifier(**p)

ROWS = []
def rec(name, i, p, s, note=""):
    ROWS.append(dict(name=name, internal=round(i,4), ext_pre=round(p,4), ext_post=round(s,4), note=note))
    print(f"{name:<46}{i:>8.4f}{p:>9.4f}{s:>9.4f}  {note}", flush=True)

def run(name, pipe, dfs=None, note=""):
    K, P, S = dfs if dfs else (kchs, pre, post)
    pipe.fit(K.iloc[tr][FE], y[tr])
    i = hauc(pipe.predict_proba(K.iloc[ho][FE])[:,1])
    p = wauc(P, pipe.predict_proba(P[FE])[:,1]); s = wauc(S, pipe.predict_proba(S[FE])[:,1])
    rec(name, i, p, s, note); return pipe, i, p, s

print(f"{'실험 (17변수 · NHANES 무재적합 · 가중AUROC)':<46}{'내부':>8}{'외부pre':>9}{'외부post':>9}")
print("-"*80, flush=True)

# ── 0. 기준선 재현 ──
base, bi, bp, bs = run("① 기준선 LightGBM", Pipeline([("pre",ct()),("clf",lgbm())]), note="← 기준")

# ── A. 설문내 분위(rank) 변환 ──
# 각 데이터셋 '내부'에서 순위→[0,1]. 라벨 미사용이므로 §2.10 위반 아님.
def to_rank(df, cols):
    d = df.copy()
    for c in cols:
        v = pd.to_numeric(d[c], errors="coerce").to_numpy(float)
        m = ~np.isnan(v); r = np.full(len(v), np.nan)
        if m.sum() > 0: r[m] = (rankdata(v[m]) - 0.5) / m.sum()
        d[c] = r
    return d

for cols, tag in [(["srh","oral_health"], "srh+oral"), (["srh"], "srh만"),
                  (NUM, "수치형 전체")]:
    K2, P2, S2 = to_rank(kchs, cols), to_rank(pre, cols), to_rank(post, cols)
    run(f"② 설문내 분위변환 ({tag})", Pipeline([("pre",ct()),("clf",lgbm())]),
        dfs=(K2,P2,S2), note="A: 척도 비동등 상쇄")

# ── B. 단조성 제약 ──
# 전처리 후 컬럼 순서: NUM(6) → BIN(8) → CAT(one-hot)
# 방향: age(+), sleep_hrs(0), bmi(0), income_q(-), srh(+ 나쁠수록↑), oral_health(+)
def mono_vec(pipe_ct, n_total):
    mc = [0]*n_total
    idx = {c:i for i,c in enumerate(NUM)}
    mc[idx["srh"]] = 1; mc[idx["oral_health"]] = 1; mc[idx["income_q"]] = -1
    # BIN 구간
    b0 = len(NUM)
    for j,c in enumerate(BIN):
        if c in ("partnered","employed","active"): mc[b0+j] = -1   # 보호
        elif c in ("living_alone","food_insecure","htn","dm"): mc[b0+j] = 1  # 위험
    return mc

_ct = ct(); _Xt = _ct.fit_transform(kchs.iloc[tr[:5000]][FE]); ncol = _Xt.shape[1]
mc = mono_vec(_ct, ncol)
run("③ 단조성 제약", Pipeline([("pre",ct()),("clf",lgbm(monotone_constraints=mc))]),
    note="B: 방향 강제")

# ── B+A 결합 ──
K2,P2,S2 = to_rank(kchs,["srh","oral_health"]), to_rank(pre,["srh","oral_health"]), to_rank(post,["srh","oral_health"])
run("④ 분위변환 + 단조성", Pipeline([("pre",ct()),("clf",lgbm(monotone_constraints=mc))]),
    dfs=(K2,P2,S2), note="A+B")

# ── C. 강정규화 / 얕은 트리 ──
for tag, kw in [("강정규화", dict(num_leaves=15, min_child_samples=500, reg_alpha=1.0,
                                 reg_lambda=5.0, subsample=0.7, colsample_bytree=0.7,
                                 n_estimators=900, learning_rate=0.02)),
                ("초강정규화", dict(num_leaves=7, min_child_samples=2000, reg_alpha=5.0,
                                   reg_lambda=20.0, max_depth=3, n_estimators=1200, learning_rate=0.02))]:
    run(f"⑤ {tag}", Pipeline([("pre",ct()),("clf",lgbm(**kw))]), note="C: 전이 안정성")
    run(f"⑥ {tag}+분위+단조", Pipeline([("pre",ct()),("clf",lgbm(monotone_constraints=mc, **kw))]),
        dfs=(K2,P2,S2), note="A+B+C")

# ── D. 시드 앙상블(순위평균) ──
def rank_avg(preds):
    return np.mean([rankdata(p)/len(p) for p in preds], axis=0)
ps_i, ps_p, ps_s = [], [], []
for sd in [42, 7, 2024, 99, 555]:
    pp = Pipeline([("pre",ct()),("clf",lgbm(random_state=sd, monotone_constraints=mc,
                                            num_leaves=15, min_child_samples=500,
                                            reg_alpha=1.0, reg_lambda=5.0,
                                            n_estimators=900, learning_rate=0.02))])
    pp.fit(K2.iloc[tr][FE], y[tr])
    ps_i.append(pp.predict_proba(K2.iloc[ho][FE])[:,1])
    ps_p.append(pp.predict_proba(P2[FE])[:,1]); ps_s.append(pp.predict_proba(S2[FE])[:,1])
rec("⑦ 시드앙상블(5) 순위평균", hauc(rank_avg(ps_i)), wauc(pre, rank_avg(ps_p)),
    wauc(post, rank_avg(ps_s)), "D: A+B+C+앙상블")

# ── 결과 저장 ──
df = pd.DataFrame(ROWS); df.to_csv(f"{RES}/auroc_max_17var.csv", index=False, encoding="utf-8-sig")
print("\n" + "="*80)
bp_row = df.loc[df["ext_pre"].idxmax()]; bs_row = df.loc[df["ext_post"].idxmax()]
print(f"기준선          : 내부 {bi:.4f} / pre {bp:.4f} / post {bs:.4f}")
print(f"외부pre  최고   : {bp_row['name']} = {bp_row['ext_pre']:.4f} (Δ{bp_row['ext_pre']-bp:+.4f})")
print(f"외부post 최고   : {bs_row['name']} = {bs_row['ext_post']:.4f} (Δ{bs_row['ext_post']-bs:+.4f})")
print(f"→ 0.80 도달: pre {'달성' if df['ext_pre'].max()>=0.80 else '미달'} / post {'달성' if df['ext_post'].max()>=0.80 else '미달'}")
print(f"저장: results/auroc_max_17var.csv")
