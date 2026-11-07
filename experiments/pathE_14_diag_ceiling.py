# -*- coding: utf-8 -*-
"""경로E-14: 진단 + 상한(oracle). 선택에는 사용하지 않는 사후 진단용.

- 각 변수의 단변량 가중 AUROC를 KCHS/pre/post에서 비교 → concept shift 정량화
- NHANES에 직접 적합한 oracle 모델의 성능 = 17변수로 도달 가능한 상한
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathE_10_common import *
from sklearn.model_selection import StratifiedKFold

t0 = time.time()
k, pre, post = load()
y = k["dep"].astype(int).values

rows = []
# ---- 단변량 가중 AUROC ----------------------------------------------------
def uni(df, col):
    v = df[col]
    if not pd.api.types.is_numeric_dtype(v):
        # 범주형: 각 수준의 가중 우울률로 점수화
        g = df.groupby(col, observed=True).apply(
            lambda d: np.average(d["dep"].astype(float), weights=d["W"]))
        s = v.map(g).astype(float)
    else:
        s = v.astype(float)
    m = s.notna()
    if m.sum() < 50 or df.loc[m, "dep"].nunique() < 2:
        return np.nan
    a = roc_auc_score(df.loc[m, "dep"].astype(int), s[m], sample_weight=df.loc[m, "W"])
    return max(a, 1 - a) * (1 if a >= .5 else -1) if False else a


for c in FEATURES:
    rows.append(dict(feature=c,
                     kchs=uni(k, c), pre=uni(pre, c), post=uni(post, c)))
uni_df = pd.DataFrame(rows)
uni_df["gap_pre"] = (uni_df.kchs - .5).abs() - (uni_df.pre - .5).abs()
uni_df["gap_post"] = (uni_df.kchs - .5).abs() - (uni_df.post - .5).abs()
print(uni_df.round(4).to_string())
uni_df.round(6).to_csv(os.path.join(RES, "pathE_14_univariate_auc.csv"),
                       index=False, encoding="utf-8-sig")

# ---- oracle 상한: NHANES에 직접 적합 -------------------------------------
def cv_oracle(df, tag):
    Xa = df[FEATURES]; ya = df["dep"].astype(int).values; Wa = df["W"].to_numpy(float)
    oof = np.zeros(len(df))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xa, ya):
        pipe = Pipeline([("pre", make_pre()),
                         ("clf", lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03,
                                                    num_leaves=15, subsample=0.8,
                                                    colsample_bytree=0.8, random_state=SEED,
                                                    n_jobs=-1, verbose=-1))])
        pipe.fit(Xa.iloc[tr], ya[tr])
        oof[te] = pipe.predict_proba(Xa.iloc[te])[:, 1]
    return wauc(ya, oof, Wa)


a_pre = cv_oracle(pre, "pre")
a_post = cv_oracle(post, "post")
# pre에 적합 → post 평가
pipe = Pipeline([("pre", make_pre()),
                 ("clf", lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03,
                                            num_leaves=15, subsample=0.8,
                                            colsample_bytree=0.8, random_state=SEED,
                                            n_jobs=-1, verbose=-1))])
pipe.fit(pre[FEATURES], pre["dep"].astype(int))
a_p2p = wauc(post["dep"], pipe.predict_proba(post[FEATURES])[:, 1], post["W"])

rec("E14_oracle_nhanes_pre_5foldCV", np.nan if False else a_pre, a_pre, a_post, False,
    "NHANES 자체 적합 상한(§2.10 위반, 진단 전용). internal칸=pre CV")
rec("E14_oracle_pre_fit_to_post", a_pre, a_pre, a_p2p, False,
    "NHANES pre 적합 → post 예측 (미국 내 시간이전 상한, §2.10 위반, 진단 전용)")
print(f"oracle: pre-CV={a_pre:.4f} post-CV={a_post:.4f} pre->post={a_p2p:.4f}")
dump("pathE_14_ceiling.csv")
print(f"[{time.time()-t0:.0f}s]")
