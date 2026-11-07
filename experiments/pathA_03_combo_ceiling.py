# -*- coding: utf-8 -*-
"""경로A 단계3: 조합 탐색 + 표현 강건화 + (명시적 비준수) 상한 진단.

주의: C9x_* 로 시작하는 실험은 NHANES 라벨을 사용하는 '진단용 상한'이며
      §2.10 비준수(protocol_compliant=False)로 기록된다. 모델 선택에 쓰지 않는다.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathA_common import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

OUT = os.path.join(RES, "pathA_combo_ceiling.csv")
ROWS = []


def rec(res, compliant, note=""):
    r = dict(res); r["protocol_compliant"] = bool(compliant); r["note"] = note
    ROWS.append(r)
    pd.DataFrame(ROWS).to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"{r['name']:38s} int={r['internal']:.4f} pre={r['ext_pre']:.4f} "
          f"post={r['ext_post']:.4f} ok={compliant} [{r.get('seconds')}s] {note}", flush=True)
    return r


def LR(C=1.0):
    return LogisticRegression(max_iter=3000, C=C, solver="lbfgs")


# ============================================================ 1) within-survey z 표준화
def zwithin(cols):
    Kc, Pc, Qc = K.copy(), PRE.copy(), POST.copy()
    refK = K.iloc[ITR]
    for c in cols:
        for d, src, w in ((Kc, refK, refK["W"]), (Pc, PRE, PRE["W"]), (Qc, POST, POST["W"])):
            v = src[c].to_numpy(float); ww = np.asarray(w, float)
            m = ~np.isnan(v)
            mu = np.average(v[m], weights=ww[m])
            sd = np.sqrt(np.average((v[m] - mu) ** 2, weights=ww[m]))
            d[c] = (d[c].to_numpy(float) - mu) / (sd if sd > 0 else 1.0)
    return Kc, Pc, Qc


for cols, tag in ((NUM, "C01_zwithin_all6"), (["srh", "oral_health"], "C02_zwithin_srh_oh")):
    a, b, c = zwithin(cols)
    r, _ = evaluate(a, b, c, NUM, BIN, CAT, tag=tag)
    rec(r, True, f"설문내 가중 z표준화 cols={','.join(cols)}")

# ============================================================ 2) rank 부분적용(shrinkage)
def blend(cols, lam):
    """z(원값, KCHS train 기준) 와 z(설문내 rank) 의 가중 평균."""
    Kr, Pr, Qr = rank_transform(cols, weighted=True, normal=True)
    Kz, Pz, Qz = K.copy(), PRE.copy(), POST.copy()
    refK = K.iloc[ITR]
    outs = []
    for (dz, dr) in ((Kz, Kr), (Pz, Pr), (Qz, Qr)):
        dd = dz.copy()
        for c in cols:
            v = refK[c].to_numpy(float); ww = refK["W"].to_numpy(float)
            m = ~np.isnan(v)
            mu = np.average(v[m], weights=ww[m])
            sd = np.sqrt(np.average((v[m] - mu) ** 2, weights=ww[m])) or 1.0
            zr = (dz[c].to_numpy(float) - mu) / sd
            dd[c] = (1 - lam) * zr + lam * dr[c].to_numpy(float)
        outs.append(dd)
    return outs


for lam in (0.25, 0.5, 0.75):
    a, b, c = blend(NUM, lam)
    r, _ = evaluate(a, b, c, NUM, BIN, CAT, tag=f"C03_blend_all6_lam{lam}")
    rec(r, True, f"원값z와 설문내 정규점수 혼합 lambda={lam}")

# ============================================================ 3) 지지구간 정렬 (age topcode 80)
Kc, Pc, Qc = K.copy(), PRE.copy(), POST.copy()
for d in (Kc, Pc, Qc):
    d["age"] = np.minimum(d["age"].to_numpy(float), 80.0)
    d["bmi"] = np.clip(d["bmi"].to_numpy(float), 15.0, 45.0)
    d["sleep_hrs"] = np.clip(d["sleep_hrs"].to_numpy(float), 3.0, 12.0)
r, _ = evaluate(Kc, Pc, Qc, NUM, BIN, CAT, tag="C04_clip_support")
rec(r, True, "age<=80, bmi 15-45, sleep 3-12 로 절단(양 설문 공통 지지구간)")

# ============================================================ 4) 가법 표현(트리 깊이1) — 표현 단순화
import lightgbm as lgb
ADD = dict(LGBM); ADD.update(max_depth=1, num_leaves=2, n_estimators=2000, learning_rate=0.05)
r, _ = evaluate(K, PRE, POST, NUM, BIN, CAT, model=lgb.LGBMClassifier(**ADD), tag="C05_lgbm_additive")
rec(r, True, "깊이1 스텀프(가법 GAM 근사)")

D2 = dict(LGBM); D2.update(max_depth=2, num_leaves=4, n_estimators=1200, learning_rate=0.05)
r, _ = evaluate(K, PRE, POST, NUM, BIN, CAT, model=lgb.LGBMClassifier(**D2), tag="C06_lgbm_depth2")
rec(r, True, "깊이2(2차 상호작용까지만)")

# 가법 + rank
Ka, Pa, Qa = rank_transform(NUM, weighted=True, normal=True)
r, _ = evaluate(Ka, Pa, Qa, NUM, BIN, CAT, model=lgb.LGBMClassifier(**ADD), tag="C07_additive_rank")
rec(r, True, "깊이1 + 설문내 정규점수")

# ============================================================ 5) 단조 제약
MONO = {"age": 0, "sleep_hrs": 0, "bmi": 0, "income_q": -1, "srh": 1, "oral_health": 1}
prep_cols = NUM + BIN + CAT
# ColumnTransformer 출력 순서: num(6) -> bin(8) -> cat(onehot)
mc = [MONO[c] for c in NUM] + [0] * len(BIN)
n_oh = 1 + 3 + 2  # sex(2-1) edu(4-1) smoking(3-1)
mc += [0] * n_oh
M = dict(LGBM); M.update(monotone_constraints=mc)
r, _ = evaluate(K, PRE, POST, NUM, BIN, CAT, model=lgb.LGBMClassifier(**M), tag="C08_monotone")
rec(r, True, "srh/oral_health 단조증가, income_q 단조감소 제약")

Km, Pm, Qm = rank_transform(NUM, weighted=True, normal=True)
r, _ = evaluate(Km, Pm, Qm, NUM, BIN, CAT, model=lgb.LGBMClassifier(**M), tag="C09_monotone_rank")
rec(r, True, "단조제약 + 설문내 정규점수")

# ============================================================ 9) 상한 진단 (§2.10 비준수, 선택 미사용)
def nhanes_cv(D, tag, folds=5):
    y = D["dep"].astype(int).to_numpy(); w = D["W"].to_numpy(float)
    X = D[NUM + BIN + CAT].reset_index(drop=True)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    oof = np.zeros(len(D))
    for tr, va in skf.split(X, y):
        pipe = Pipeline([("pre", make_prep(NUM, BIN, CAT)), ("clf", lgb.LGBMClassifier(**ADD))])
        pipe.fit(X.iloc[tr], y[tr])
        oof[va] = pipe.predict_proba(X.iloc[va])[:, 1]
    return wauc(y, oof, w)


a_pre = nhanes_cv(PRE, "pre")
a_post = nhanes_cv(POST, "post")
rec({"name": "C90_ORACLE_nhanes_selffit_cv", "internal": np.nan,
     "ext_pre": a_pre, "ext_post": a_post, "seconds": None}, False,
    "★비준수 진단: NHANES 자체 5-fold CV(가법LGBM). 17변수의 미국내 달성가능 상한 추정")

# 로지스틱 버전 상한
def nhanes_cv_lr(D, folds=5):
    y = D["dep"].astype(int).to_numpy(); w = D["W"].to_numpy(float)
    X = D[NUM + BIN + CAT].reset_index(drop=True)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    oof = np.zeros(len(D))
    for tr, va in skf.split(X, y):
        pipe = Pipeline([("pre", make_prep(NUM, BIN, CAT, True)), ("clf", LR())])
        pipe.fit(X.iloc[tr], y[tr])
        oof[va] = pipe.predict_proba(X.iloc[va])[:, 1]
    return wauc(y, oof, w)


rec({"name": "C91_ORACLE_nhanes_selffit_logit", "internal": np.nan,
     "ext_pre": nhanes_cv_lr(PRE), "ext_post": nhanes_cv_lr(POST), "seconds": None}, False,
    "★비준수 진단: NHANES 자체 5-fold CV(로지스틱). 상한 추정")

print("saved:", OUT)
