# -*- coding: utf-8 -*-
"""
pathF_00_baseline.py — 기준선 재현 확인.
목표: 내부 0.8336 / pre 0.7681 / post 0.7682
두 가지 분할 사양을 모두 시험:
  (a) test_size=0.2 (기존 17_ext_transport_combo.py A_core3 사양)
  (b) train_size=500000, test_size=100000 (과업 지시문 사양)
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from pathF_common import (RES, SEED, FEATURES, load, income_q_trainonly,
                          make_pipe, wauc)

QC = []
def qc(s):
    QC.append(str(s)); print(s); sys.stdout.flush()

k, pre, post = load()
y = k["dep"].astype(int).values
ypre = pre["dep"].astype(int).values; wpre = pre["W"].to_numpy(float)
ypost = post["dep"].astype(int).values; wpost = post["W"].to_numpy(float)
rows = []

def run(tag, itr, iho, kk, W):
    t0 = time.time()
    pipe = make_pipe()
    X = kk[FEATURES]
    pipe.fit(X.iloc[itr], y[itr])
    pi = pipe.predict_proba(X.iloc[iho])[:, 1]
    pp = pipe.predict_proba(pre[FEATURES])[:, 1]
    po = pipe.predict_proba(post[FEATURES])[:, 1]
    r = dict(name=tag, train_n=len(itr), holdout_n=len(iho),
             internal=wauc(y[iho], pi, W[iho]),
             ext_pre=wauc(ypre, pp, wpre), ext_post=wauc(ypost, po, wpost))
    rows.append(r)
    qc(f"{tag:40s} tr={len(itr):>9,d} ho={len(iho):>9,d} "
       f"내부={r['internal']:.4f} pre={r['ext_pre']:.4f} post={r['ext_post']:.4f} [{time.time()-t0:.0f}s]")
    return r

# (a) test_size=0.2
itr, iho = train_test_split(np.arange(len(k)), test_size=0.2, stratify=y, random_state=SEED)
is_tr = np.zeros(len(k), bool); is_tr[itr] = True
ka = k.copy(); ka["income_q"] = income_q_trainonly(ka, is_tr)
Wa = ka["W"].to_numpy(float)
run("(a) test_size=0.2", itr, iho, ka, Wa)

# (b) train_size=500000, test_size=100000
itr2, iho2 = train_test_split(np.arange(len(k)), train_size=500000, test_size=100000,
                              stratify=y, random_state=SEED)
is_tr2 = np.zeros(len(k), bool); is_tr2[itr2] = True
kb = k.copy(); kb["income_q"] = income_q_trainonly(kb, is_tr2)
Wb = kb["W"].to_numpy(float)
run("(b) train=500k test=100k", itr2, iho2, kb, Wb)

# (b') 동일 분할이되 income_q를 파일 내장값 그대로 사용
kb2 = k.copy(); Wb2 = kb2["W"].to_numpy(float)
run("(b') train=500k/test=100k, income_q as-is", itr2, iho2, kb2, Wb2)

pd.DataFrame(rows).to_csv(os.path.join(RES, "pathF_00_baseline.csv"), index=False, encoding="utf-8-sig")
open(os.path.join(RES, "pathF_00_baseline_qc.txt"), "w", encoding="utf-8").write("\n".join(QC))
qc("완료")
