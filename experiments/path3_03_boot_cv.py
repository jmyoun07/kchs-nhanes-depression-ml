# -*- coding: utf-8 -*-
"""
path3_03_boot_cv.py — 경로3 불확실성·내부선택기준.

A) 외부 AUROC의 복합표본(PSU) 부트스트랩 CI — '얼마나 움직여야 진짜 개선인가'
B) 내부 전용 시간전이 CV — 어느 시기 학습이 '다른 시기'로 잘 전이되는지
   (KCHS 홀드아웃의 특정 연도만 평가, 학습은 다른 연도에서)
   → NHANES를 전혀 쓰지 않는 연도설계 선택 근거.

출력: results/path3_03_bootci.csv, results/path3_03_temporalcv.csv
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from path3_common import Bench, Recorder, RES, SEED, FEATURES, wauc

B = Bench()
LOG = []
def qc(s):
    print(s); sys.stdout.flush(); LOG.append(str(s))

# ============================================================
# A) 외부 AUROC의 PSU 부트스트랩 CI
# ============================================================
qc("########## A) 외부 AUROC 복합표본 부트스트랩 CI ##########")
base, (pi, pp, po), _ = B.run("BASE_spec_500k", B.tr)
qc(f"기준선: int={base['internal']:.4f} pre={base['ext_pre']:.4f} post={base['ext_post']:.4f}")


def psu_boot(df, y, p, w, nboot=1000, seed=SEED):
    """층 내 PSU 복원추출 부트스트랩 (Rao-Wu 계열 단순형)."""
    st = df["strata"].astype(str).to_numpy()
    ps = df["psu"].astype(str).to_numpy()
    key = np.char.add(np.char.add(st, "_"), ps)
    strata_u = np.unique(st)
    idx_by = {}
    psus_by = {}
    for s in strata_u:
        m = st == s
        u = np.unique(key[m])
        psus_by[s] = u
        for kk in u:
            idx_by[kk] = np.where(key == kk)[0]
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(nboot):
        take = []
        for s in strata_u:
            u = psus_by[s]
            sel = rng.choice(u, size=len(u), replace=True)
            for kk in sel:
                take.append(idx_by[kk])
        ii = np.concatenate(take)
        try:
            out.append(wauc(y[ii], p[ii], w[ii]))
        except Exception:
            pass
    out = np.array(out)
    return out


rows = []
for nm, df, y, p, w in [("NHANES pre", B.pre, B.ypre, pp, B.wpre),
                        ("NHANES post", B.post, B.ypost, po, B.wpost)]:
    t0 = time.time()
    bs = psu_boot(df, y, p, w, nboot=1000)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    pt = wauc(y, p, w)
    rows.append(dict(set=nm, auroc=pt, ci_lo=lo, ci_hi=hi, se=bs.std(),
                     n_strata=df["strata"].nunique(), nboot=len(bs),
                     secs=round(time.time() - t0, 1),
                     p_ge_080=float((bs >= 0.80).mean())))
    qc(f"  {nm}: AUROC={pt:.4f} 95%CI=[{lo:.4f},{hi:.4f}] SE={bs.std():.4f} "
       f"P(boot>=0.80)={float((bs>=0.80).mean()):.3f}")
pd.DataFrame(rows).to_csv(os.path.join(RES, "path3_03_bootci.csv"),
                          index=False, encoding="utf-8-sig")

# ============================================================
# B) 내부 전용 시간전이 CV
# ============================================================
qc("\n########## B) 내부 전용 시간전이 CV (NHANES 미사용) ##########")
year_pool = B.year[B.pool]
year_ho = B.year[B.ho]
NTR = 400000
res = []

HOLDS = [("H_2017_2018", [2017, 2018]),
         ("H_2020_2021", [2020, 2021]),
         ("H_2024_2025", [2024, 2025])]


def draw(elig, n, seed=SEED):
    rng = np.random.RandomState(seed)
    return np.array(elig) if len(elig) <= n else rng.choice(elig, n, replace=False)


for hname, hy in HOLDS:
    ho_m = np.isin(year_ho, hy)
    yv = B.yho[ho_m]; wv = B.who[ho_m]
    Xho_h = B.Xho[ho_m]
    others = [y for y in range(2017, 2026) if y not in hy]
    DESIGNS = [("all_other_years", others),
               ("nearest3", sorted(others, key=lambda y: min(abs(y - h) for h in hy))[:3]),
               ("farthest3", sorted(others, key=lambda y: -min(abs(y - h) for h in hy))[:3]),
               ("postpandemic_only", [y for y in others if y >= 2021] or others),
               ("prepandemic_only", [y for y in others if y <= 2020] or others)]
    for dname, dy in DESIGNS:
        elig = B.pool[np.isin(year_pool, dy)]
        idx = draw(elig, NTR)
        t0 = time.time()
        _, _, pipe = B.run("tmp", idx)
        p = pipe.predict_proba(Xho_h)[:, 1]
        a = wauc(yv, p, wv)
        res.append(dict(holdout=hname, design=dname, years=",".join(map(str, dy)),
                        train_n=len(idx), n_eval=int(ho_m.sum()), auroc_heldout_year=a,
                        secs=round(time.time() - t0, 1)))
        qc(f"  [{hname}] {dname:20s} n={len(idx):>7,d} 내부(해당연도)AUROC={a:.4f}")
    pd.DataFrame(res).to_csv(os.path.join(RES, "path3_03_temporalcv.csv"),
                             index=False, encoding="utf-8-sig")

d = pd.DataFrame(res)
qc("\n--- 설계별 평균(내부 시간전이) ---")
for dn, g in d.groupby("design"):
    qc(f"  {dn:20s} mean={g['auroc_heldout_year'].mean():.4f} min={g['auroc_heldout_year'].min():.4f}")
qc(f"\n저장: {os.path.join(RES,'path3_03_bootci.csv')} / {os.path.join(RES,'path3_03_temporalcv.csv')}")
