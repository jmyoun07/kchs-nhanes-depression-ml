# -*- coding: utf-8 -*-
"""
pathC_02_oof.py — 스태킹/가중치 학습용 KCHS 내부 OOF 예측 생성.

설계(§2.10 엄수):
  KCHS train(1,638,808) → base_tr(80%) / meta(20%)  [stratify=dep, seed=42]
  기저모델은 base_tr에서만 fit → meta·holdout·pre·post 예측.
  메타러너(가중치·스태킹·보정)는 meta 예측만으로 학습 → holdout/pre/post는 순수 평가.
  NHANES 라벨은 어떤 fit에도 사용되지 않음.
출력: experiments/cache/oof/<model>.npz, results/pathC_oof_models.csv
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from pathC_common import load, wauc, RES, SEED
from pathC_01_base_library import mk

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "oof")
os.makedirs(CACHE, exist_ok=True)

# 스태킹용 다양성 확보 세트(시드복제 제외 — 스태킹엔 정보중복)
NAMES = ["lgbm_s42", "lgbm_reg", "lgbm_shallow", "lgbm_deep", "lgbm_goss",
         "rf", "et", "histgb", "enet", "logit_plain", "gnb"]

if __name__ == "__main__":
    d = load("full02")
    ib, im = train_test_split(np.arange(len(d.Xtr)), test_size=0.2,
                              stratify=d.ytr, random_state=SEED)
    Xb, yb = d.Xtr.iloc[ib], d.ytr[ib]
    Xm, ym, Wm = d.Xtr.iloc[im], d.ytr[im], d.Wtr[im]
    print(f"base_tr={len(Xb)} meta={len(Xm)} holdout={len(d.Xho)}")
    sys.stdout.flush()
    np.savez_compressed(os.path.join(CACHE, "_labels.npz"),
                        ym=ym, Wm=Wm, yho=d.yho, Who=d.Who,
                        ypre=d.ypre, wpre=d.wpre, ypost=d.ypost, wpost=d.wpost)
    rows = []
    for nm in NAMES:
        f = os.path.join(CACHE, nm + ".npz")
        t0 = time.time()
        if os.path.exists(f):
            z = np.load(f)
            pm, pho, ppre, ppost = z["meta"], z["ho"], z["pre"], z["post"]
            secs = float(z["secs"])
        else:
            try:
                m = mk(nm)
                m.fit(Xb, yb)
                pm = m.predict_proba(Xm)[:, 1]
                pho = m.predict_proba(d.Xho)[:, 1]
                ppre = m.predict_proba(d.Xpre)[:, 1]
                ppost = m.predict_proba(d.Xpost)[:, 1]
            except Exception as e:
                print(f"[{nm}] 실패: {type(e).__name__}: {e}")
                sys.stdout.flush()
                continue
            secs = time.time() - t0
            np.savez_compressed(f, meta=pm, ho=pho, pre=ppre, post=ppost, secs=secs)
        rows.append(dict(model=nm, meta_auc=wauc(ym, pm, Wm),
                         internal=wauc(d.yho, pho, d.Who),
                         ext_pre=wauc(d.ypre, ppre, d.wpre),
                         ext_post=wauc(d.ypost, ppost, d.wpost), secs=secs))
        r = rows[-1]
        print(f"{nm:14s} meta={r['meta_auc']:.4f} 내부={r['internal']:.4f} "
              f"pre={r['ext_pre']:.4f} post={r['ext_post']:.4f} [{secs:.0f}s]")
        sys.stdout.flush()
    pd.DataFrame(rows).to_csv(os.path.join(RES, "pathC_oof_models.csv"),
                              index=False, encoding="utf-8-sig")
    print("완료:", CACHE)
