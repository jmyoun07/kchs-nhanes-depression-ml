# -*- coding: utf-8 -*-
"""경로1-03: 사전선언 규칙이 뽑은 설정을 기준선 프로토콜로 최종 1회 외부검증.

- 학습/평가 프로토콜은 기준선과 완전 동일:
    train_test_split(n, train_size=500000, test_size=100000, stratify=dep, seed=42)
    파이프라인(중앙값대치+표준화 / passthrough / 최빈대치+OHE)을 학습폴드에서 적합
- NHANES는 여기서 처음이자 마지막으로 사용된다.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_common import *
from p1_01_grid import G
import numpy as np
import pandas as pd

SEL = json.load(open(os.path.join(RES, "p1_selection.json")))
SPEC = {n: s for n, s, _ in G}


def eval_spec(k, pre, post, y, itr, iho, spec):
    p = dict(LGB_BASE)
    p.update({kk: vv for kk, vv in spec.get("params", {}).items() if kk != "n_jobs"})
    pipe = Pipeline([("pre", make_pre()), ("clf", lgb.LGBMClassifier(**p))])
    kw = {}
    if spec.get("sw") == "survey":
        kw["clf__sample_weight"] = k["W"].to_numpy(float)[itr]
    pipe.fit(k.iloc[itr][FEATURES], y[itr], **kw)
    return (pipe.predict_proba(k.iloc[iho][FEATURES])[:, 1],
            pipe.predict_proba(pre[FEATURES])[:, 1],
            pipe.predict_proba(post[FEATURES])[:, 1])


if __name__ == "__main__":
    k, pre, post = load()
    y = k["dep"].astype(int).to_numpy()
    itr, iho = base_split(len(k), y)
    Wi, Wp, Ws = k["W"].to_numpy(float)[iho], pre["W"].values, post["W"].values
    rows = []
    t0 = time.time()

    # 기준선 재기록
    pi, pp, ps = eval_spec(k, pre, post, y, itr, iho, {"params": {}})
    base = (wauc(y[iho], pi, Wi), wauc(pre["dep"], pp, Wp), wauc(post["dep"], ps, Ws))
    rec(rows, name="baseline_literal", internal=round(base[0], 6),
        ext_pre=round(base[1], 6), ext_post=round(base[2], 6),
        protocol_compliant=True, rule="", note="기준선 재현")

    cache = {}

    def get(nm):
        if nm not in cache:
            cache[nm] = eval_spec(k, pre, post, y, itr, iho, SPEC[nm])
        return cache[nm]

    for rule, nm in SEL.items():
        if "," in nm:  # 앙상블
            names = nm.split(",")
            P = [get(x) for x in names]
            from scipy.stats import rankdata
            pi = np.mean([rankdata(p[0]) / len(p[0]) for p in P], axis=0)
            pp = np.mean([rankdata(p[1]) / len(p[1]) for p in P], axis=0)
            ps = np.mean([rankdata(p[2]) / len(p[2]) for p in P], axis=0)
        else:
            pi, pp, ps = get(nm)
        a = (wauc(y[iho], pi, Wi), wauc(pre["dep"], pp, Wp), wauc(post["dep"], ps, Ws))
        rec(rows, name=nm, internal=round(a[0], 6), ext_pre=round(a[1], 6),
            ext_post=round(a[2], 6), protocol_compliant=True, rule=rule,
            note=f"KCHS 내부 CV 규칙 {rule}이 선택. dpre={a[1]-base[1]:+.4f} "
                 f"dpost={a[2]-base[2]:+.4f}")
        pd.DataFrame(rows).to_csv(os.path.join(RES, "p1_03_final.csv"),
                                  index=False, encoding="utf-8-sig")
    dump(rows, "p1_03_final.csv")
    print(f"[{time.time()-t0:.0f}s]")
