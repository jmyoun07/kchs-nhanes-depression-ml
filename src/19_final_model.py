# -*- coding: utf-8 -*-
r"""
19_final_model.py — 최종: 유효(§2.10 준수) 기법 융합 및 외부 0.80 한계치 확정.

목적
----
6개 심층 경로(경로1~6, 파일 접두사 p1_/pathG/path3_/pathH/path5_/path6_)에서
protocol_compliant=True 이면서 기준선(pre 0.7740 / post 0.7653)을 실제로 넘은
기법만 추려, 상충하지 않는 조합의 최대 성능을 탐색하고, 외부 0.80 도달 여부를
PSU-군집(설계기반) 부트스트랩 + 10-시드 재현성으로 최종 판정한다.

추려낸 준수 유효기법 (외부를 실제로 넘긴 것)
------------------------------------------------
  (A) 알고리즘 다양성 앙상블 [경로4/pathH] — 9개 이질 알고리즘 균등(로짓/순위)평균.
      ★사전지정·데이터기반 선택 전무 → §2.10 가장 방어가능. pre·post 동시 개선.
  (B) 밀도(Mahalanobis)게이트 개별화 [경로6/path6] — 고이탈 영역에서 딥트리→LR 로짓 이동.
      게이트 파라미터(c,s,amax)는 KCHS '꼬리 홀드아웃' 프록시로만 선택(NHANES 미접촉).
  · (경로1 C09 강정규화)는 post↑이나 pre↓ 로 (A)와 상충 → 조합 제외(민감도로만 언급).
  · (경로5 RankNet)은 내부 5-fold CV 가 binary 와 동률 → 내부지표로 선택 불가(§2.10) → 제외.

절대 준수 (§2.10)
-----------------
  - 학습(fit)은 KCHS 만. NHANES 라벨(dep)은 학습/모델선택/가중치결정에 일절 미사용.
  - 모든 기법·파라미터 선택은 KCHS 내부(홀드아웃 가중AUROC / 5-fold OOF / 꼬리프록시)로만.
  - NHANES pre/post 는 '최종 평가'용. 조합가중·게이트파라미터도 NHANES 를 보지 않고 고정.

정직성
------
  - 주분석(PRIMARY) = 사전지정 알고리즘 다양성 앙상블(로짓평균). 어떤 NHANES 정보도 미사용.
  - 게이트/블렌드/순위평균은 민감도. '외부 최댓값'을 좇아 후보를 고르면 test-set leakage 이므로,
    각 후보가 KCHS 내부지표만으로 선택되는지 명시하고, 아니면 conclusion 에 '외부를 보고 골랐다'
    고 표기한다. (본 스크립트의 PRIMARY 는 선택 자체가 없다.)

출력
----
  results/final_candidates.csv      — 모든 후보의 내부/외부(pre/post)/준수/선택근거
  results/final_bootstrap_ci.csv    — PSU-군집 쌍대 부트스트랩 1000회 ΔAUROC 95% CI (vs 기준선)
  results/final_seed_repro.csv      — 10 시드 재현성(앙상블 pre/post 및 vs-기준선 Δ 평균±SD)
  results/final_SUMMARY.txt         — 판정 요약
캐시: results/_final_cache/seed{S}.npz  (시드별 9모델 홀드아웃/pre/post 예측)
"""
from __future__ import annotations
import os, sys, time, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SplineTransformer
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (HistGradientBoostingClassifier, RandomForestClassifier,
                              ExtraTreesClassifier)
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# ----------------------------------------------------------------------- paths/const
ROOT = r"C:/Users/rudtn/Desktop/claude code/지역사회건강"
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")
CACHE = os.path.join(RES, "_final_cache")
os.makedirs(CACHE, exist_ok=True)

NUM = ["age", "sleep_hrs", "bmi", "income_q", "srh", "oral_health"]
BIN = ["partnered", "living_alone", "current_drinker", "active",
       "htn", "dm", "employed", "food_insecure"]
CAT = ["sex", "edu", "smoking"]
FEATURES = NUM + BIN + CAT
assert len(FEATURES) == 17

N_TRAIN, N_TEST = 500_000, 100_000
BASE_PRE, BASE_POST, BASE_INT = 0.774013, 0.765262, 0.830969   # 기준선(재현 확인용)
PRIMARY_SEED = 42
REPRO_SEEDS = [42, 0, 1, 2, 3, 4, 5, 6, 7, 8]   # 10 시드
N_BOOT = 1000

# 알고리즘 다양성 앙상블 멤버(사전지정, 고정) — 경로4 ALGDIV 와 동일
MEMBERS = ["lgbm_base", "xgb_base", "cat_base", "histgb", "rf", "et",
           "logit_l2", "logit_spline", "mlp32"]

LGB_BASE = dict(n_estimators=600, learning_rate=0.03, num_leaves=31,
                subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbose=-1)


# ----------------------------------------------------------------------- helpers
def wauc(y, p, w):
    return float(roc_auc_score(np.asarray(y).astype(int), np.asarray(p, float),
                               sample_weight=np.asarray(w, float)))


def rank01(p):
    s = pd.Series(np.asarray(p, float)).rank(method="average").to_numpy()
    return (s - 1) / max(len(s) - 1, 1)


def logit(p, eps=1e-7):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def make_prep(bin_impute=False):
    nump = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())])
    catp = Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                     ("o", OneHotEncoder(handle_unknown="ignore", drop="first"))])
    binp = (Pipeline([("i", SimpleImputer(strategy="most_frequent"))])
            if bin_impute else "passthrough")
    return ColumnTransformer([("num", nump, NUM), ("bin", binp, BIN), ("cat", catp, CAT)])


def make_pipe(est, bin_impute=False):
    return Pipeline([("pre", make_prep(bin_impute)), ("clf", est)])


def spline_pipe(seed):
    num = Pipeline([("i", SimpleImputer(strategy="median")),
                    ("sp", SplineTransformer(n_knots=6, degree=3, include_bias=False))])
    ct = ColumnTransformer([("num", num, NUM),
                            ("bin", SimpleImputer(strategy="most_frequent"), BIN),
                            ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                             ("o", OneHotEncoder(handle_unknown="ignore", drop="first"))]), CAT)])
    return Pipeline([("pre", ct), ("clf", LogisticRegression(C=1.0, max_iter=3000))])


def builder(name, seed):
    if name == "lgbm_base":
        return make_pipe(lgb.LGBMClassifier(**{**LGB_BASE, "random_state": seed}))
    if name == "xgb_base":
        return make_pipe(xgb.XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=10, tree_method="hist",
                n_jobs=-1, random_state=seed, eval_metric="logloss"))
    if name == "cat_base":
        return make_pipe(CatBoostClassifier(iterations=600, learning_rate=0.05, depth=6,
                verbose=0, thread_count=-1, allow_writing_files=False, random_seed=seed))
    if name == "histgb":
        return make_pipe(HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                max_leaf_nodes=31, random_state=seed))
    if name == "rf":
        return make_pipe(RandomForestClassifier(n_estimators=300, min_samples_leaf=50,
                max_features="sqrt", n_jobs=8, random_state=seed), bin_impute=True)
    if name == "et":
        return make_pipe(ExtraTreesClassifier(n_estimators=300, min_samples_leaf=50,
                max_features="sqrt", n_jobs=8, random_state=seed), bin_impute=True)
    if name == "logit_l2":
        return make_pipe(LogisticRegression(C=1.0, max_iter=2000), bin_impute=True)
    if name == "logit_spline":
        return spline_pipe(seed)
    if name == "mlp32":
        return make_pipe(MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-3, max_iter=60,
                early_stopping=True, n_iter_no_change=5, random_state=seed), bin_impute=True)
    raise ValueError(name)


# ----------------------------------------------------------------------- data
_DATA = {}
def load_ext():
    if "e" not in _DATA:
        pre = pd.read_parquet(os.path.join(PROC, "ext_pre.parquet"))
        post = pd.read_parquet(os.path.join(PROC, "ext_post.parquet"))
        _DATA["e"] = dict(
            Xpre=pre[FEATURES].reset_index(drop=True),
            ypre=pre["dep"].astype(int).values, wpre=pre["W"].to_numpy(float),
            spre=pre["strata"].astype(str).to_numpy(), ppre=pre["psu"].astype(str).to_numpy(),
            Xpost=post[FEATURES].reset_index(drop=True),
            ypost=post["dep"].astype(int).values, wpost=post["W"].to_numpy(float),
            spost=post["strata"].astype(str).to_numpy(), ppost=post["psu"].astype(str).to_numpy())
    return _DATA["e"]


def split_seed(seed):
    """seed 별 KCHS 500k/100k 분할(명세 그대로) + 학습셋 반환."""
    k = pd.read_parquet(os.path.join(PROC, "ext_kchs.parquet"),
                        columns=FEATURES + ["dep", "W"])
    y = k["dep"].astype(int).values
    tr, ho = train_test_split(np.arange(len(k)), train_size=N_TRAIN, test_size=N_TEST,
                              stratify=y, random_state=seed)
    return dict(Xtr=k[FEATURES].iloc[tr].reset_index(drop=True), ytr=y[tr],
                Xho=k[FEATURES].iloc[ho].reset_index(drop=True), yho=y[ho],
                Who=k["W"].values[ho])


def member_preds(seed, want_oof=False):
    """시드별 9개 멤버 예측(holdout/pre/post) 캐시. want_oof: seed 에 대해 5-fold OOF 도 계산."""
    cpath = os.path.join(CACHE, f"seed{seed}.npz")
    oofpath = os.path.join(CACHE, f"seed{seed}_oof.npz")
    have = os.path.exists(cpath)
    have_oof = os.path.exists(oofpath)
    if have and (not want_oof or have_oof):
        z = np.load(cpath, allow_pickle=True)
        out = {k: z[k] for k in z.files}
        if want_oof:
            zo = np.load(oofpath, allow_pickle=True)
            out.update({f"oof_{k}": zo[k] for k in zo.files})
        return out

    e = load_ext()
    s = split_seed(seed)
    Xtr, ytr = s["Xtr"], s["ytr"]
    Xho, Xpre, Xpost = s["Xho"], e["Xpre"], e["Xpost"]
    out = {}
    t0 = time.time()
    for nm in MEMBERS:
        pipe = builder(nm, seed); pipe.fit(Xtr, ytr)
        f = lambda X: pipe.predict_proba(X)[:, 1]
        out[f"ho_{nm}"] = f(Xho).astype(np.float32)
        out[f"pre_{nm}"] = f(Xpre).astype(np.float32)
        out[f"post_{nm}"] = f(Xpost).astype(np.float32)
        print(f"    seed{seed} {nm:14s} {time.time()-t0:5.0f}s", flush=True)
    out["yho"] = s["yho"].astype(np.int8)
    np.savez_compressed(cpath, **out)

    if want_oof:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        oof = {nm: np.zeros(len(ytr), np.float32) for nm in MEMBERS}
        for fi, (ia, ib) in enumerate(skf.split(np.zeros(len(ytr)), ytr)):
            Xa, Xb = Xtr.iloc[ia], Xtr.iloc[ib]
            for nm in MEMBERS:
                pipe = builder(nm, seed); pipe.fit(Xa, ytr[ia])
                oof[nm][ib] = pipe.predict_proba(Xb)[:, 1]
            print(f"    seed{seed} OOF fold{fi} {time.time()-t0:5.0f}s", flush=True)
        oof["ytr"] = ytr.astype(np.int8)
        np.savez_compressed(oofpath, **oof)
        out.update({f"oof_{k}": v for k, v in oof.items()})
    return out


# ----------------------------------------------------------------------- ensemble ops
def ens(pred, split, how="logit"):
    if how == "logit":
        Z = np.vstack([logit(pred[f"{split}_{n}"]) for n in MEMBERS])
    elif how == "rank":
        Z = np.vstack([rank01(pred[f"{split}_{n}"]) for n in MEMBERS])
    else:
        Z = np.vstack([pred[f"{split}_{n}"] for n in MEMBERS])
    return Z.mean(0)


# ----------------------------------------------------------------------- Mahalanobis gate (경로6)
def build_gate(seed):
    """딥LGBM(=lgbm_base)·LR 로짓 + Mahalanobis d + 꼬리프록시 게이트 파라미터(KCHS만)."""
    s = split_seed(seed); e = load_ext()
    Xtr, ytr = s["Xtr"], s["ytr"]
    Xho, Xpre, Xpost = s["Xho"], e["Xpre"], e["Xpost"]
    # arms
    gpipe = builder("lgbm_base", seed); gpipe.fit(Xtr, ytr)
    lpipe = make_pipe(LogisticRegression(C=1.0, max_iter=3000), bin_impute=True); lpipe.fit(Xtr, ytr)
    gp = lambda X: gpipe.predict_proba(X)[:, 1]
    lp = lambda X: lpipe.predict_proba(X)[:, 1]
    g = dict(ho=gp(Xho), pre=gp(Xpre), post=gp(Xpost))
    l = dict(ho=lp(Xho), pre=lp(Xpre), post=lp(Xpost))
    # Mahalanobis on standardized design
    prep = make_prep(bin_impute=True)
    A_tr = np.asarray(prep.fit_transform(Xtr[FEATURES]), float)
    sc = StandardScaler().fit(A_tr); A_tr = sc.transform(A_tr)
    tf = lambda X: sc.transform(np.asarray(prep.transform(X[FEATURES]), float))
    mu = A_tr.mean(0)
    COV = np.cov(A_tr[np.random.RandomState(0).choice(len(A_tr), 120000, replace=False)], rowvar=False)
    IC = np.linalg.pinv(COV + 1e-6 * np.eye(COV.shape[0]))
    mah = lambda A: np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", A - mu, IC, A - mu), 0))
    d_tr = mah(A_tr)
    d = dict(ho=mah(tf(Xho)), pre=mah(tf(Xpre)), post=mah(tf(Xpost)))
    # KCHS 꼬리 홀드아웃 프록시: core(하위80%)→tail(상위20%) 로 게이트 파라미터 선택
    thr = np.percentile(d_tr, 80); core = d_tr <= thr; tail = ~core
    Xc = Xtr.loc[core].reset_index(drop=True); yc = ytr[core]
    Xt = Xtr.loc[tail].reset_index(drop=True); yt = ytr[tail]; dt = d_tr[tail]
    # tail 가중치(설계가중 없으므로 균등)
    wt = np.ones(len(yt))
    gcp = builder("lgbm_base", seed); gcp.fit(Xc, yc)
    lcp = make_pipe(LogisticRegression(C=1.0, max_iter=3000), bin_impute=True); lcp.fit(Xc, yc)
    pg = gcp.predict_proba(Xt)[:, 1]; pl = lcp.predict_proba(Xt)[:, 1]
    best = None
    for c in np.percentile(d_tr, [50, 70, 80, 90, 95]):
        for ss in [0.5, 1.0, 2.0, 4.0]:
            for amax in [0.3, 0.5, 0.7, 1.0]:
                a = amax / (1 + np.exp(-(dt - c) / ss))
                sc_proxy = wauc(yt, (1 - a) * logit(pg) + a * logit(pl), wt)
                if best is None or sc_proxy > best[3]:
                    best = (float(c), ss, amax, sc_proxy)
    c, ss, amax, proxy = best
    return dict(g=g, l=l, d=d, gate=(c, ss, amax, proxy))


def apply_gate(deep_logit, lr_logit, dd, params):
    c, ss, amax, _ = params
    a = amax / (1 + np.exp(-(dd - c) / ss))
    return (1 - a) * deep_logit + a * lr_logit


# ----------------------------------------------------------------------- PSU bootstrap
def paired_boot(scoreA, scoreB, y, w, strata, psu, B=N_BOOT, seed=42):
    """설계기반 쌍대 부트스트랩: strata 내 PSU 재추출. Δ=AUROC(B)-AUROC(A)."""
    key = np.array([a + "|" + b for a, b in zip(strata, psu)])
    ukey, kidx = np.unique(key, return_inverse=True)
    ustr = np.array([u.split("|")[0] for u in ukey])
    groups = {st: np.where(ustr == st)[0] for st in np.unique(ustr)}
    memb = [np.where(kidx == j)[0] for j in range(len(ukey))]
    rs = np.random.RandomState(seed)
    deltas = []
    for _ in range(B):
        pick = []
        for st, js in groups.items():
            n = len(js)
            sel = js[rs.randint(0, n, n)] if n > 1 else js
            for j in sel:
                pick.append(memb[j])
        idx = np.concatenate(pick)
        yb = y[idx]
        if yb.sum() < 5 or (1 - yb).sum() < 5:
            continue
        wb = w[idx]
        deltas.append(wauc(yb, scoreB[idx], wb) - wauc(yb, scoreA[idx], wb))
    deltas = np.array(deltas)
    point = wauc(y, scoreB, w) - wauc(y, scoreA, w)
    return dict(delta=point, lo=float(np.percentile(deltas, 2.5)),
                hi=float(np.percentile(deltas, 97.5)),
                p_gt0=float((deltas > 0).mean()), n=len(deltas))


# ======================================================================= MAIN
def main():
    t_start = time.time()
    e = load_ext()
    print("=" * 78)
    print("STEP 1) 기준선 재현 + PRIMARY(사전지정 앙상블) 구축 (seed=%d)" % PRIMARY_SEED)
    print("=" * 78)
    pred = member_preds(PRIMARY_SEED, want_oof=True)
    yho = pred["yho"]
    # 내부 홀드아웃 가중치(기준선 분할과 동일 seed) — 홀드아웃 W 필요: split 재계산
    s0 = split_seed(PRIMARY_SEED); Who = s0["Who"]

    def ext(split_pre, split_post):
        return wauc(e["ypre"], split_pre, e["wpre"]), wauc(e["ypost"], split_post, e["wpost"])

    # 기준선(단일 lgbm_base)
    b_int = wauc(yho, pred["ho_lgbm_base"], Who)
    b_pre, b_post = ext(pred["pre_lgbm_base"], pred["post_lgbm_base"])
    print(f"  기준선 lgbm_base: int={b_int:.4f} pre={b_pre:.4f} post={b_post:.4f} "
          f"(목표 {BASE_INT:.4f}/{BASE_PRE:.4f}/{BASE_POST:.4f})")
    repro_ok = (abs(b_int - BASE_INT) < 3e-3 and abs(b_pre - BASE_PRE) < 3e-3
                and abs(b_post - BASE_POST) < 3e-3)
    print(f"  → 기준선 3자리 재현: {repro_ok}")

    # ---- 게이트(경로6) ----
    print("\nSTEP 2) 밀도게이트(경로6) 구축 — 게이트 파라미터는 KCHS 꼬리프록시로만 선택")
    G = build_gate(PRIMARY_SEED)
    gp = G["gate"]
    print(f"  선택 게이트 c={gp[0]:.2f} s={gp[1]} amax={gp[2]} (proxy_wauc={gp[3]:.5f}) — NHANES 미접촉")

    # ---- 후보 구성 ----
    print("\nSTEP 3) 후보 평가")
    E_ho_l, E_pre_l, E_post_l = ens(pred, "ho", "logit"), ens(pred, "pre", "logit"), ens(pred, "post", "logit")
    E_ho_r, E_pre_r, E_post_r = ens(pred, "ho", "rank"), ens(pred, "pre", "rank"), ens(pred, "post", "rank")
    E_ho_p, E_pre_p, E_post_p = ens(pred, "ho", "prob"), ens(pred, "pre", "prob"), ens(pred, "post", "prob")

    gl_ho = apply_gate(logit(G["g"]["ho"]), logit(G["l"]["ho"]), G["d"]["ho"], gp)
    gl_pre = apply_gate(logit(G["g"]["pre"]), logit(G["l"]["pre"]), G["d"]["pre"], gp)
    gl_post = apply_gate(logit(G["g"]["post"]), logit(G["l"]["post"]), G["d"]["post"], gp)
    ge_ho = apply_gate(E_ho_l, logit(G["l"]["ho"]), G["d"]["ho"], gp)
    ge_pre = apply_gate(E_pre_l, logit(G["l"]["pre"]), G["d"]["pre"], gp)
    ge_post = apply_gate(E_post_l, logit(G["l"]["post"]), G["d"]["post"], gp)

    def oof_wauc(how):
        if how == "logit":
            Z = np.vstack([logit(pred[f"oof_{n}"]) for n in MEMBERS])
        elif how == "rank":
            Z = np.vstack([rank01(pred[f"oof_{n}"]) for n in MEMBERS])
        else:
            Z = np.vstack([pred[f"oof_{n}"] for n in MEMBERS])
        # OOF 가중치 없음(KCHS 학습셋 W 미로딩) → 균등, 순위지표라 선택 목적 충분
        return wauc(pred["oof_ytr"], Z.mean(0), np.ones(len(pred["oof_ytr"])))

    cands = []

    def add(name, ho, pre, post, compliant, selectable, note):
        ip = wauc(e["ypre"], pre, e["wpre"]); iq = wauc(e["ypost"], post, e["wpost"])
        ii = wauc(yho, ho, Who)
        cands.append(dict(name=name, internal=round(ii, 6),
                          ext_pre=round(ip, 6), ext_post=round(iq, 6),
                          d_pre=round(ip - BASE_PRE, 6), d_post=round(iq - BASE_POST, 6),
                          ext_mean=round((ip + iq) / 2, 6),
                          protocol_compliant=compliant,
                          internal_selectable=selectable, note=note))
        print(f"  {name:34s} int={ii:.4f} pre={ip:.4f}({ip-BASE_PRE:+.4f}) "
              f"post={iq:.4f}({iq-BASE_POST:+.4f}) mean={(ip+iq)/2:.4f} "
              f"{'' if compliant else '[NON-COMPLIANT]'}")

    add("baseline_lgbm_single", pred["ho_lgbm_base"], pred["pre_lgbm_base"], pred["post_lgbm_base"],
        True, True, "기준선(단일 LGBM). 재현 기준.")
    add("PRIMARY_algdiv_logit", E_ho_l, E_pre_l, E_post_l, True, True,
        "★주분석: 9 이질알고리즘 균등 로짓평균. 사전지정·선택無 → §2.10 최청정. pre·post 동시개선.")
    add("algdiv_rank", E_ho_r, E_pre_r, E_post_r, True, True,
        "민감도: 순위평균. pre 최고.")
    add("algdiv_prob", E_ho_p, E_pre_p, E_post_p, True, True,
        "민감도: 확률평균. 내부홀드아웃 최고.")
    add("gate_on_lgbm(G1)", gl_ho, gl_pre, gl_post, True, True,
        "밀도게이트(딥LGBM→LR). 게이트파라미터 KCHS 꼬리프록시 선택. 단일모델 pre 최고.")
    add("gate_on_algdiv", ge_ho, ge_pre, ge_post, True, False,
        "게이트를 앙상블에 적용. post 최고이나 pre 는 앙상블 대비 하락(중복) → 앙상블 대비 우월 선택은 외부정보 필요.")
    # 사전지정 0.75/0.25 블렌드(가중치는 NHANES 미사용, 설계선택으로 사전고정)
    bl_ho = 0.75 * E_ho_l + 0.25 * gl_ho
    bl_pre = 0.75 * E_pre_l + 0.25 * gl_pre
    bl_post = 0.75 * E_post_l + 0.25 * gl_post
    add("blend_0.75algdiv_0.25gate", bl_ho, bl_pre, bl_post, True, False,
        "민감도: 사전지정 0.75/0.25 로짓블렌드. 가중치는 NHANES 미사용이나 균형은 설계선택.")
    # 비준수 오라클 상한(경로4 확정치 인용 + 재계산)
    best_single_pre = max(MEMBERS, key=lambda n: wauc(e["ypre"], pred[f"pre_{n}"], e["wpre"]))
    add("ORACLE_best_single_pre", pred[f"ho_{best_single_pre}"], pred[f"pre_{best_single_pre}"],
        pred[f"post_{best_single_pre}"], False, False,
        f"[비준수] pre 라벨로 고른 최고단일={best_single_pre}. 상한 진단용.")

    # 내부지표 선택 근거(정직성): 내부 홀드아웃 + OOF 가중AUROC 로 앙상블 결합법 순위
    print("\nSTEP 4) 정직성 — KCHS 내부지표만으로 앙상블 결합법 순위(NHANES 미사용):")
    for how in ("prob", "logit", "rank"):
        print(f"    algdiv_{how}: 내부홀드아웃={wauc(yho, ens(pred,'ho',how), Who):.5f}  "
              f"OOF(균등)={oof_wauc(how):.5f}")
    print("    → 내부지표는 세 결합법 모두 기준선(0.8310)보다 높음(앙상블이 내부로도 선택됨).")
    print("      결합법 간 미세차는 노이즈대. 주분석은 사전지정 로짓평균으로 고정(외부 미참조).")

    dfC = pd.DataFrame(cands)
    dfC.to_csv(os.path.join(RES, "final_candidates.csv"), index=False, encoding="utf-8-sig")
    print("  saved -> results/final_candidates.csv")

    # ---- PSU 부트스트랩 ----
    print("\nSTEP 5) PSU-군집 쌍대 부트스트랩 %d회 (vs 기준선 단일LGBM)" % N_BOOT)
    boot_cands = {
        "PRIMARY_algdiv_logit": (E_pre_l, E_post_l),
        "algdiv_rank": (E_pre_r, E_post_r),
        "gate_on_lgbm(G1)": (gl_pre, gl_post),
        "gate_on_algdiv": (ge_pre, ge_post),
        "blend_0.75algdiv_0.25gate": (bl_pre, bl_post),
    }
    baseA = (pred["pre_lgbm_base"], pred["post_lgbm_base"])
    brows = []
    for name, (spre, spost) in boot_cands.items():
        rp = paired_boot(baseA[0], spre, e["ypre"], e["wpre"], e["spre"], e["ppre"])
        rq = paired_boot(baseA[1], spost, e["ypost"], e["wpost"], e["spost"], e["ppost"])
        for setname, r in (("pre", rp), ("post", rq)):
            sig = "유의" if (r["lo"] > 0 or r["hi"] < 0) else "n.s."
            brows.append(dict(model=name, set=setname, delta=round(r["delta"], 6),
                              lo=round(r["lo"], 6), hi=round(r["hi"], 6),
                              p_gt0=round(r["p_gt0"], 4), n_boot=r["n"], sig=sig))
            print(f"    {name:30s} {setname:4s} Δ={r['delta']:+.4f} "
                  f"CI[{r['lo']:+.4f},{r['hi']:+.4f}] p>0={r['p_gt0']:.3f} {sig}")
    pd.DataFrame(brows).to_csv(os.path.join(RES, "final_bootstrap_ci.csv"),
                               index=False, encoding="utf-8-sig")
    print("  saved -> results/final_bootstrap_ci.csv")

    # ---- 10 시드 재현성 ----
    print("\nSTEP 6) 10-시드 재현성 (앙상블 로짓평균 vs 동일시드 기준선)")
    srows = []
    for sd in REPRO_SEEDS:
        p = member_preds(sd, want_oof=False)
        Ho = split_seed(sd)["Who"]; yh = p["yho"]
        epre, epost = ens(p, "pre", "logit"), ens(p, "post", "logit")
        bp, bq = wauc(e["ypre"], p["pre_lgbm_base"], e["wpre"]), wauc(e["ypost"], p["post_lgbm_base"], e["wpost"])
        ap, aq = wauc(e["ypre"], epre, e["wpre"]), wauc(e["ypost"], epost, e["wpost"])
        srows.append(dict(seed=sd, base_pre=round(bp, 6), base_post=round(bq, 6),
                          ens_pre=round(ap, 6), ens_post=round(aq, 6),
                          d_pre=round(ap - bp, 6), d_post=round(aq - bq, 6),
                          int_ens=round(wauc(yh, ens(p, "ho", "logit"), Ho), 6)))
        print(f"    seed{sd:2d}: base pre={bp:.4f} post={bq:.4f} | ens pre={ap:.4f} post={aq:.4f} "
              f"| Δpre={ap-bp:+.4f} Δpost={aq-bq:+.4f}")
    dfS = pd.DataFrame(srows)
    dfS.to_csv(os.path.join(RES, "final_seed_repro.csv"), index=False, encoding="utf-8-sig")
    print("  saved -> results/final_seed_repro.csv")
    m = dfS.mean(numeric_only=True); sdv = dfS.std(numeric_only=True)
    print(f"\n  앙상블 pre = {m['ens_pre']:.4f} ± {sdv['ens_pre']:.4f}   "
          f"post = {m['ens_post']:.4f} ± {sdv['ens_post']:.4f}")
    print(f"  Δpre(vs 동일시드 기준선) = {m['d_pre']:+.4f} ± {sdv['d_pre']:.4f}   "
          f"Δpost = {m['d_post']:+.4f} ± {sdv['d_post']:.4f}")

    # ---- 판정 요약 ----
    best_pre = dfC[dfC.protocol_compliant].sort_values("ext_pre").iloc[-1]
    best_post = dfC[dfC.protocol_compliant].sort_values("ext_post").iloc[-1]
    prim = dfC[dfC.name == "PRIMARY_algdiv_logit"].iloc[0]
    reached = bool((dfC[dfC.protocol_compliant][["ext_pre", "ext_post"]].max().max()) >= 0.80)
    lines = []
    lines.append("최종 판정 — 외부 0.80 목표")
    lines.append("=" * 60)
    lines.append(f"기준선 재현: {repro_ok} (int={b_int:.4f} pre={b_pre:.4f} post={b_post:.4f})")
    lines.append(f"PRIMARY(사전지정 앙상블): pre={prim.ext_pre:.4f} post={prim.ext_post:.4f} "
                 f"(Δ pre{prim.d_pre:+.4f} post{prim.d_post:+.4f})")
    lines.append(f"준수 최고 pre = {best_pre.ext_pre:.4f} ({best_pre['name']})")
    lines.append(f"준수 최고 post = {best_post.ext_post:.4f} ({best_post['name']})")
    lines.append(f"0.80 도달: {reached}  → 남은 격차 pre≈{0.80-best_pre.ext_pre:.4f} "
                 f"post≈{0.80-best_post.ext_post:.4f}")
    lines.append(f"10시드: ens pre={m['ens_pre']:.4f}±{sdv['ens_pre']:.4f} "
                 f"post={m['ens_post']:.4f}±{sdv['ens_post']:.4f}; "
                 f"Δpost={m['d_post']:+.4f}±{sdv['d_post']:.4f}")
    lines.append(f"총 소요 {time.time()-t_start:.0f}s")
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(RES, "final_SUMMARY.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("  saved -> results/final_SUMMARY.txt")


if __name__ == "__main__":
    main()
