# -*- coding: utf-8 -*-
"""74_shap_cross.py — 교차조사 SHAP 순위 일치도를 '교정된 주분석' 모델로 재계산.

원고는 KCHS 와 두 NHANES 주기 사이의 예측변수 중요도 순위 일치도(Spearman)를 근거로
'전이의 한계는 관계의 구조가 아니라 강도에 있다'고 말한다. 그런데 그 값의 근거파일
(shap_cross.csv)은 정정 이전 세대였다. 여기서 정본 모델로 다시 계산한다.

출력: results/primary2_shap_cross.csv (변수별 |SHAP| 평균), primary2_shap_rho.csv (상관)
"""
import io, os, sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__)); RES = f"{D}/results"

src = io.open(f"{D}/70_primary_corrected.py", encoding="utf8").read()
g = {"__file__": f"{D}/70_primary_corrected.py", "__name__": "__prep__"}
exec(compile(src[: src.index("p_tr = model.predict_proba")], "70_prep", "exec"), g)
K, P, S, FE = g["K"], g["P"], g["S"], g["FE"]
model, SEED = g["model"], g["SEED"]
import shap
from scipy.stats import spearmanr

pre_step = model.named_steps["pre"]; clf = model.named_steps["clf"]
names = [n.split("__")[-1] for n in pre_step.get_feature_names_out()]
expl = shap.TreeExplainer(clf)

def imp(df, n=20000, seed=SEED):
    idx = np.random.RandomState(seed).choice(len(df), min(n, len(df)), replace=False)
    X = pre_step.transform(df.iloc[idx][FE])
    sv = expl.shap_values(X)
    if isinstance(sv, list): sv = sv[1]
    a = pd.Series(np.abs(sv).mean(0), index=names)
    # 원-핫으로 쪼개진 열은 원래 변수로 합산
    return pd.Series({v: a[[i for i in a.index if i == v or i.startswith(v + "_")]].sum() for v in FE})

print("[1] KCHS 개발 학습셋", flush=True)
kc = imp(K.iloc[g["tr"]])
print("[2] NHANES pre", flush=True); pr = imp(P)
print("[3] NHANES post", flush=True); po = imp(S)

out = pd.DataFrame({"KCHS": kc, "NHANES pre": pr, "NHANES post": po}).round(4)
out.index.name = "predictor"
out.to_csv(f"{RES}/primary2_shap_cross.csv", encoding="utf-8-sig")

rows = []
for lab, v in [("NHANES pre", pr), ("NHANES post", po)]:
    rho, p = spearmanr(kc.rank(ascending=False), v.rank(ascending=False))
    rows.append(dict(comparison=f"KCHS vs {lab}", spearman_rho=round(float(rho), 3), p_value=float(p)))
    print(f"   KCHS vs {lab}: rho = {rho:.3f}", flush=True)
pd.DataFrame(rows).to_csv(f"{RES}/primary2_shap_rho.csv", index=False, encoding="utf-8-sig")
print("\n저장: primary2_shap_cross.csv · primary2_shap_rho.csv")
