# -*- coding: utf-8 -*-
"""경로A 단계6: 상한(ceiling) 정밀 추정 + 개념이동(concept shift) 진단 실험.

C90/C91에서 'NHANES 자체적합 CV'가 전이모델과 비슷하다는 신호가 나왔다.
표본이 작아 CV가 비관적일 수 있으므로 (a) pre+post 통합 적합, (b) 정규화 강도 탐색,
(c) 표본크기 곡선으로 상한을 견고하게 추정한다.

D9x_* = §2.10 비준수 진단(선택에 미사용). D0x_* = 준수.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathA_common import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

OUT = os.path.join(RES, "pathA_ceiling2.csv")
ROWS = []


def rec(name, internal, pre, post, ok, note, sec=None):
    r = dict(name=name, internal=internal, ext_pre=pre, ext_post=post,
             seconds=sec, protocol_compliant=bool(ok), note=note)
    ROWS.append(r); pd.DataFrame(ROWS).to_csv(OUT, index=False, encoding="utf-8-sig")
    f = lambda x: "  nan " if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"
    print(f"{name:40s} int={f(internal)} pre={f(pre)} post={f(post)} ok={ok} {note}", flush=True)


NH = pd.concat([PRE.assign(_src="pre"), POST.assign(_src="post")], ignore_index=True)
FE = NUM + BIN + CAT


def selffit_cv(D, mk, folds=5, seed=SEED, train_on=None):
    """D 안에서 교차적합 OOF 예측 → 가중 AUROC. train_on=별도 학습 프레임(통합적합용)."""
    y = D["dep"].astype(int).to_numpy(); w = D["W"].to_numpy(float)
    X = D[FE].reset_index(drop=True)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(D))
    for tr, va in skf.split(X, y):
        clf = mk()
        bi = not isinstance(clf, lgb.LGBMClassifier)
        pipe = Pipeline([("pre", make_prep(NUM, BIN, CAT, bi)), ("clf", clf)])
        pipe.fit(X.iloc[tr], y[tr])
        oof[va] = pipe.predict_proba(X.iloc[va])[:, 1]
    return wauc(y, oof, w), oof


# ---------------- (a) pre+post 통합 자체적합 (표본 2배) → 평가는 각 기간별로 분해
for cname, mk in (("logit_C1", lambda: LogisticRegression(max_iter=3000, C=1.0)),
                  ("logit_C0.1", lambda: LogisticRegression(max_iter=3000, C=0.1)),
                  ("logit_C0.01", lambda: LogisticRegression(max_iter=3000, C=0.01)),
                  ("lgbm_add", lambda: lgb.LGBMClassifier(**{**LGBM, "max_depth": 1,
                                                             "num_leaves": 2,
                                                             "n_estimators": 1500,
                                                             "learning_rate": 0.05})),
                  ("lgbm_d3_reg", lambda: lgb.LGBMClassifier(**{**LGBM, "max_depth": 3,
                                                                "num_leaves": 8,
                                                                "n_estimators": 800,
                                                                "learning_rate": 0.03,
                                                                "min_child_samples": 200}))):
    a_all, oof = selffit_cv(NH, mk)
    m = (NH["_src"] == "pre").to_numpy()
    ap = wauc(NH["dep"][m].astype(int), oof[m], NH["W"][m])
    ao = wauc(NH["dep"][~m].astype(int), oof[~m], NH["W"][~m])
    rec(f"D90_ORACLE_nhanes_pooled_{cname}", np.nan, ap, ao, False,
        f"★비준수 상한진단: NHANES pre+post 통합 5-fold 자체적합(n={len(NH)}), all={a_all:.4f}")

# ---------------- (b) 학습표본 크기 곡선 (상한이 표본부족 때문인지 확인)
y_nh = NH["dep"].astype(int).to_numpy(); X_nh = NH[FE].reset_index(drop=True)
rng = np.random.default_rng(SEED)
for frac in (0.25, 0.5, 1.0):
    aucs = []
    for rep in range(3):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + rep)
        oof = np.zeros(len(NH))
        for tr, va in skf.split(X_nh, y_nh):
            if frac < 1.0:
                tr = rng.choice(tr, int(len(tr) * frac), replace=False)
            pipe = Pipeline([("pre", make_prep(NUM, BIN, CAT, True)),
                             ("clf", LogisticRegression(max_iter=3000, C=0.1))])
            pipe.fit(X_nh.iloc[tr], y_nh[tr])
            oof[va] = pipe.predict_proba(X_nh.iloc[va])[:, 1]
        aucs.append(wauc(y_nh, oof, NH["W"]))
    rec(f"D91_ORACLE_learncurve_frac{frac}", np.nan, float(np.mean(aucs)), np.nan, False,
        f"★비준수: NHANES 자체적합 학습표본 {frac:.0%} (pre+post 통합 AUROC), sd={np.std(aucs):.4f}")

# ---------------- (c) 개념이동 진단: age 문제 확인 (age 방향 반전)
Kd, Pd, Qd = K.copy(), PRE.copy(), POST.copy()
NUM_noage = [c for c in NUM if c != "age"]
r, _ = evaluate(Kd, Pd, Qd, NUM_noage, BIN, CAT, tag="D92_drop_age")
rec("D92_DIAG_drop_age", r["internal"], r["ext_pre"], r["ext_post"], False,
    "★선택근거가 외부성능이면 비준수. age 제외 시 외부 변화 진단용(개념이동 검증)", r["seconds"])

r, _ = evaluate(Kd, Pd, Qd, ["srh", "oral_health"], BIN, CAT, tag="D93_srh_only_core")
rec("D93_DIAG_srh_oh_only", r["internal"], r["ext_pre"], r["ext_post"], False,
    "★진단: 연속형 중 srh/oral_health만 사용(개념이동 큰 age/sleep/bmi 제외)", r["seconds"])

# ---------------- (d) 준수 최적안: 내부 CV로만 고른 표현
# 내부 CV 상위 후보 3개를 KCHS train에서 3-fold로 비교 (NHANES 미접촉)
print("\n[내부 CV 기반 선택 — NHANES 미접촉]", flush=True)
Krk, Prk, Qrk = rank_transform(NUM, weighted=True, normal=True)
cands = {
    "baseline": (K, PRE, POST, NUM, BIN, CAT, None),
    "rankN_all6": (Krk, Prk, Qrk, NUM, BIN, CAT, None),
    "additive": (K, PRE, POST, NUM, BIN, CAT,
                 lambda: lgb.LGBMClassifier(**{**LGBM, "max_depth": 1, "num_leaves": 2,
                                               "n_estimators": 2000, "learning_rate": 0.05})),
    "depth2": (K, PRE, POST, NUM, BIN, CAT,
               lambda: lgb.LGBMClassifier(**{**LGBM, "max_depth": 2, "num_leaves": 4,
                                             "n_estimators": 1200, "learning_rate": 0.05})),
}
cvres = {}
for nm, (a, b, c, n_, bi_, ca_, mf) in cands.items():
    mu, sd = cv_internal(a, n_, bi_, ca_, model_fn=mf, folds=3, sub=200_000)
    cvres[nm] = mu
    print(f"  CV {nm:12s} = {mu:.4f} (sd {sd:.4f})", flush=True)
pd.Series(cvres).to_csv(os.path.join(RES, "pathA_internal_cv_selection.csv"),
                        encoding="utf-8-sig")
best = max(cvres, key=cvres.get)
print(f"  -> 내부 CV 선택: {best}", flush=True)
a, b, c, n_, bi_, ca_, mf = cands[best]
r, _ = evaluate(a, b, c, n_, bi_, ca_, model=(mf() if mf else None), tag="D01_cvselected")
rec("D01_CVSELECTED_" + best, r["internal"], r["ext_pre"], r["ext_post"], True,
    f"내부 CV(KCHS train 3-fold)로만 선택 후 NHANES 1회 평가. CV={cvres}", r["seconds"])

print("saved:", OUT)
