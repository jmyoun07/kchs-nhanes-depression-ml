# -*- coding: utf-8 -*-
"""09_tables.py — Table 3 기저특성(데이터셋별). 결과: results/table3_characteristics.csv"""
import os, numpy as np, pandas as pd
ROOT = r"C:/Users/rudtn/Desktop/claude code/지역사회건강"; PROC = os.path.join(ROOT, "data/processed"); RES = os.path.join(ROOT, "results")

def desc(df, label):  # 복합표본 가중(비율·평균), 비가중 n 별도
    n = len(df); w = df["W"].to_numpy(float)
    def pw(mask):
        m = mask.to_numpy().astype(float); return f"{np.average(m, weights=w)*100:.1f}% (n={int(mask.sum()):,})"
    def mw(col):
        x = df[col].to_numpy(float); ok = ~np.isnan(x)
        return f"{np.average(x[ok], weights=w[ok]):.1f}"
    return {
        "특성": label, "N (비가중)": f"{n:,}",
        "우울 PHQ-9≥10 (가중)": pw(df["dep"] == 1),
        "연령(가중평균)": mw("age"),
        "여성 (가중)": pw(df["sex"] == "F"),
        "현재흡연 (가중)": pw(df["smoking"] == "current"),
        "유배우 (가중)": pw(df["partnered"] == 1),
        "BMI (가중평균)": mw("bmi"),
        "수면시간 (가중평균)": mw("sleep_hrs"),
        "자가건강 1-5 (가중평균)": mw("srh"),
        "고혈압 (가중)": pw(df["htn"] == 1),
        "당뇨 (가중)": pw(df["dm"] == 1),
    }

rows = [desc(pd.read_parquet(os.path.join(PROC, f"{k}.parquet")), lab)
        for k, lab in [("kchs_dev", "KCHS 개발(2017-25)"), ("nhanes_pre", "NHANES pre"), ("nhanes_post", "NHANES post")]]
t3 = pd.DataFrame(rows).set_index("특성").T
t3.to_csv(os.path.join(RES, "table3_characteristics.csv"), encoding="utf-8-sig")
print(t3.to_string())
