# -*- coding: utf-8 -*-
"""90_delta_and_calib_ci.py — 리뷰어가 요구할 두 가지 구간을 계산한다.

 (1) 사후−사전 AUROC 차의 구간. 주변구간이 겹친다는 말은 동일성의 근거가 못 된다.
 (2) 외부 코호트 보정지표(기울기·CITL·ECE)의 구간. 보정은 1차 주장인데 점추정만 있었다.

중요: 재표본 가중치와 지표 정의를 여기서 다시 구현하지 않는다. 70_primary_corrected.py 를
그대로 실행해 그 안의 RaoWuBoot·reps_for·cal_slope·citl·ece·recal 를 쓴다. 별도 구현이면
표 2 의 AUROC 구간과 다른 재표본에서 나온 값이 되어 "같은 재표본" 이라는 서술이 거짓이 된다.

차이의 구간은 두 코호트가 서로 다른 표본이므로 재표본 인덱스를 짝지어서는 안 된다.
독립으로 뽑은 두 인덱스의 쌍에서 차를 만든다(합성곱 방식).

출력: results/primary2_delta_auroc.csv, results/primary2_calibration_ci.csv
"""
import io, os, sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__)); RES = f"{D}/results"

src = io.open(f"{D}/70_primary_corrected.py", encoding="utf8").read()
g = {"__file__": f"{D}/70_primary_corrected.py", "__name__": "__prep__"}
exec(compile(src[: src.index("mrows=[]")], "70_prep", "exec"), g)   # REPS 까지 포함
P, S = g["P"], g["S"]; yp, ys = g["yp"], g["ys"]; Wp, Ws = g["Wp"], g["Ws"]
p_pre, p_post = g["p_pre"], g["p_post"]; REPS = g["REPS"]; SEED = g["SEED"]
cal_slope, citl, ece, recal = g["cal_slope"], g["citl"], g["ece"], g["recal"]
from sklearn.metrics import roc_auc_score
grp = lambda d: (d["strata"].astype(str) + "|" + d["psu"].astype(str)).values

def boot(y, p, reps, fn):
    """70 의 ci() 와 같은 유효성 규칙으로 재표본 통계를 모은다."""
    out = []
    for w in reps:
        m = w > 0
        if m.sum() < 10 or y[m].min() == y[m].max(): continue
        try: out.append(fn(y, p, w))
        except Exception: pass
    return np.array(out)

AUC = lambda y, p, w: roc_auc_score(y, p, sample_weight=w)
a_pre = boot(yp, p_pre, REPS["NHANES pre"], AUC)
a_post = boot(ys, p_post, REPS["NHANES post"], AUC)
obs = AUC(ys, p_post, Ws) - AUC(yp, p_pre, Wp)

# 두 표본은 서로 겹치지 않으므로 재표본 인덱스를 짝지으면 없는 상관이 생긴다.
# 독립으로 뽑은 인덱스 쌍에서 차를 만든다.
rs = np.random.RandomState(SEED + 1)
K = 20000
diff = a_post[rs.randint(0, len(a_post), K)] - a_pre[rs.randint(0, len(a_pre), K)]
lo, hi = np.percentile(diff, [2.5, 97.5])
pval = 2 * min((diff <= 0).mean(), (diff >= 0).mean())
pd.DataFrame([dict(contrast="post − pre (external AUROC)", delta=round(float(obs), 6),
                   CI_lo=round(float(lo), 6), CI_hi=round(float(hi), 6),
                   P=round(float(max(pval, 1 / K)), 4),
                   replicates_pre=len(a_pre), replicates_post=len(a_post), pairs=K)]
             ).to_csv(f"{RES}/primary2_delta_auroc.csv", index=False, encoding="utf-8-sig")
print(f"[1] 사후−사전 AUROC 차 {obs:+.4f} (95% CI {lo:+.4f}–{hi:+.4f}, P={max(pval,1/K):.3f})", flush=True)

rows = []
for nm, d, y, p, w in [("NHANES pre", P, yp, p_pre, Wp), ("NHANES post", S, ys, p_post, Ws)]:
    R = REPS[nm]
    pr = recal(y, p, w, "slope", groups=grp(d))          # 표 2 와 같은 재보정
    for lab, q in [("as transported", p), ("after intercept/slope recalibration", pr)]:
        sl = boot(y, q, R, cal_slope); ct = boot(y, q, R, citl); ec = boot(y, q, R, ece)
        rows.append(dict(set=nm, probabilities=lab,
                         cal_slope=round(cal_slope(y, q, w), 6),
                         slope_lo=round(float(np.percentile(sl, 2.5)), 6), slope_hi=round(float(np.percentile(sl, 97.5)), 6),
                         CITL=round(citl(y, q, w), 6),
                         CITL_lo=round(float(np.percentile(ct, 2.5)), 6), CITL_hi=round(float(np.percentile(ct, 97.5)), 6),
                         ECE=round(ece(y, q, w), 6),
                         ECE_lo=round(float(np.percentile(ec, 2.5)), 6), ECE_hi=round(float(np.percentile(ec, 97.5)), 6)))
        print(f"   {nm:<12} {lab:<38} slope {rows[-1]['cal_slope']:.3f} [{rows[-1]['slope_lo']:.3f},{rows[-1]['slope_hi']:.3f}]"
              f"  CITL {rows[-1]['CITL']:+.3f} [{rows[-1]['CITL_lo']:+.3f},{rows[-1]['CITL_hi']:+.3f}]", flush=True)
pd.DataFrame(rows).to_csv(f"{RES}/primary2_calibration_ci.csv", index=False, encoding="utf-8-sig")
print("저장: primary2_delta_auroc.csv · primary2_calibration_ci.csv")
