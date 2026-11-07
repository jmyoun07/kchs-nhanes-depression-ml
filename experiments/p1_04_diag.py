# -*- coding: utf-8 -*-
"""경로1-04: 사후 진단 — 내부 CV 지표가 외부 전이성능을 예측하는가?

★ 사전선언 선택(p1_02/p1_03)이 **먼저 확정·기록된 뒤**에만 실행한다.
★ 여기서 산출하는 전 설정의 외부 AUROC는 **선택에 사용하지 않는다**(사후 상관 진단 전용).
   따라서 이 표에서 '외부 최고'를 고르는 것은 §2.10 위반이며 protocol_compliant=False로만 보고한다.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_common import *
from p1_01_grid import G
import numpy as np
import pandas as pd

NJOBS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
SHARD = int(sys.argv[2]) if len(sys.argv) > 2 else 0
NSHARD = int(sys.argv[3]) if len(sys.argv) > 3 else 1

if __name__ == "__main__":
    k, pre, post = load()
    y = k["dep"].astype(int).to_numpy()
    itr, iho = base_split(len(k), y)
    Wi, Wp, Ws = k["W"].to_numpy(float)[iho], pre["W"].values, post["W"].values
    rows = []
    t0 = time.time()
    mine = [(i, n, s) for i, (n, s, _) in enumerate(G) if i % NSHARD == SHARD]
    OUT = os.path.join(RES, f"p1_04_ext_all_s{SHARD}.csv")
    for j, (i, name, spec) in enumerate(mine, 1):
        t = time.time()
        p = dict(LGB_BASE)
        p.update({kk: vv for kk, vv in spec.get("params", {}).items() if kk != "n_jobs"})
        p["n_jobs"] = NJOBS
        pipe = Pipeline([("pre", make_pre()), ("clf", lgb.LGBMClassifier(**p))])
        kw = {}
        if spec.get("sw") == "survey":
            kw["clf__sample_weight"] = k["W"].to_numpy(float)[itr]
        try:
            pipe.fit(k.iloc[itr][FEATURES], y[itr], **kw)
            a = (wauc(y[iho], pipe.predict_proba(k.iloc[iho][FEATURES])[:, 1], Wi),
                 wauc(pre["dep"], pipe.predict_proba(pre[FEATURES])[:, 1], Wp),
                 wauc(post["dep"], pipe.predict_proba(post[FEATURES])[:, 1], Ws))
        except Exception as e:
            print(f"[{j}] {name} FAILED {e}", flush=True)
            continue
        rows.append(dict(idx=i, name=name, internal=round(a[0], 6),
                         ext_pre=round(a[1], 6), ext_post=round(a[2], 6),
                         seconds=round(time.time() - t, 1)))
        print(f"[{j}/{len(mine)}] {name} int={a[0]:.5f} pre={a[1]:.5f} post={a[2]:.5f} "
              f"[{time.time()-t:.0f}s tot={time.time()-t0:.0f}s]", flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
    print("done", OUT)
