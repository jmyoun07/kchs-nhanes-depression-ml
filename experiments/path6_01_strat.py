# -*- coding: utf-8 -*-
"""
경로6 STAGE1 — (2) 연령대별 분리모델, (3) 성별 분리모델, (7) 캘리브레이션 인지 결합.

분리모델의 핵심 난점: AUROC는 '전체 집단 하나의 순위'로 계산되므로
층별 모델 점수를 층 간 비교 가능하게 만드는 결합규칙이 성능을 좌우한다.
결합규칙 4종을 모두 실측한다.
  R1 raw      : 층별 모델 예측확률 그대로 (KCHS 내에서 층내 보정되어 있다고 가정)
  R2 platt    : 층별로 KCHS 학습셋에서 Platt(로지스틱) 재보정 후 확률 결합  (§2.10 준수)
  R3 iso      : 층별 isotonic 재보정                                        (§2.10 준수)
  R4 blend    : 전역모델과 로짓 혼합(α는 KCHS 내부 CV로 결정)               (§2.10 준수)
  R5 tgt_rank : 대상 데이터 층내 순위를 KCHS 층별 점수분포로 사상            (회색: 대상 공변량분포 사용)
"""
import time, numpy as np, pandas as pd
from path6_common import *
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression as LR

rows = []
T0 = time.time()

# ---------------------------------------------------------------- 전역 기준선 점수
t0 = time.time()
(g_te, g_pre, g_post), _ = fit_predict(XTR, YTR, [XTE, XP, XQ])
# 학습셋 자기 예측(보정 학습용) — 과적합 방지 위해 2-fold OOF
from sklearn.model_selection import StratifiedKFold
oof_g = np.zeros(len(XTR))
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
for tr, va in skf.split(XTR, YTR):
    (o,), _ = fit_predict(XTR.iloc[tr].reset_index(drop=True), YTR[tr],
                          [XTR.iloc[va].reset_index(drop=True)])
    oof_g[va] = o
rows.append(report("S0_global_baseline", g_te, g_pre, g_post,
                   note="전역 단일 LGBM (기준선)", secs=time.time() - t0))
print(rows[-1])


def build_strat(gtr, gte, gp, gq, tag, note_prefix):
    """gtr/gte/gp/gq: 층 인덱스 배열(정수). 층별 모델 학습 후 5가지 결합규칙 평가."""
    out = []
    levels = sorted(set(gtr.tolist()))
    pte_raw = np.zeros(len(XTE)); ppre_raw = np.zeros(len(XP)); ppost_raw = np.zeros(len(XQ))
    pte_pl = np.zeros(len(XTE)); ppre_pl = np.zeros(len(XP)); ppost_pl = np.zeros(len(XQ))
    pte_is = np.zeros(len(XTE)); ppre_is = np.zeros(len(XP)); ppost_is = np.zeros(len(XQ))
    oof_s = np.zeros(len(XTR))
    band_diag = []
    t0 = time.time()
    for L in levels:
        mtr = gtr == L
        if mtr.sum() < 2000:
            print("  skip level", L, mtr.sum()); continue
        Xl = XTR.loc[mtr].reset_index(drop=True); yl = YTR[mtr]; wl = WTR[mtr]
        mte, mp, mq = gte == L, gp == L, gq == L
        Xs = [XTE.loc[mte], XP.loc[mp], XQ.loc[mq]]
        (a, b, c), _ = fit_predict(Xl, yl, Xs)
        pte_raw[mte] = a; ppre_raw[mp] = b; ppost_raw[mq] = c
        # 층내 OOF (혼합/보정 학습용)
        oo = np.zeros(mtr.sum())
        for tr, va in StratifiedKFold(3, shuffle=True, random_state=SEED).split(Xl, yl):
            (z,), _ = fit_predict(Xl.iloc[tr].reset_index(drop=True), yl[tr],
                                  [Xl.iloc[va].reset_index(drop=True)])
            oo[va] = z
        oof_s[np.where(mtr)[0]] = oo
        # R2 Platt: 층 내 KCHS OOF 로 로짓 재보정 (가중)
        pl = LR(max_iter=1000)
        pl.fit(logit(oo).reshape(-1, 1), yl, sample_weight=wl)
        f = lambda v: pl.predict_proba(logit(v).reshape(-1, 1))[:, 1]
        pte_pl[mte] = f(a); ppre_pl[mp] = f(b); ppost_pl[mq] = f(c)
        # R3 isotonic
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        iso.fit(oo, yl, sample_weight=wl)
        pte_is[mte] = iso.predict(a); ppre_is[mp] = iso.predict(b); ppost_is[mq] = iso.predict(c)
        # 층내 진단
        try:
            band_diag.append(dict(level=L, n_tr=int(mtr.sum()),
                                  in_int=round(wauc(YTE[mte], a, WTE[mte]), 4),
                                  in_pre=round(wauc(YP[mp], b, WP[mp]), 4),
                                  in_post=round(wauc(YQ[mq], c, WQ[mq]), 4),
                                  glob_int=round(wauc(YTE[mte], g_te[mte], WTE[mte]), 4),
                                  glob_pre=round(wauc(YP[mp], g_pre[mp], WP[mp]), 4),
                                  glob_post=round(wauc(YQ[mq], g_post[mq], WQ[mq]), 4)))
        except Exception as e:
            band_diag.append(dict(level=L, err=str(e)))
    secs = time.time() - t0
    out.append(report(f"{tag}_R1_raw", pte_raw, ppre_raw, ppost_raw,
                      note=note_prefix + " | 층별모델 확률 그대로", secs=secs))
    out.append(report(f"{tag}_R2_platt", pte_pl, ppre_pl, ppost_pl,
                      note=note_prefix + " | 층별 Platt 재보정(KCHS OOF)"))
    out.append(report(f"{tag}_R3_iso", pte_is, ppre_is, ppost_is,
                      note=note_prefix + " | 층별 isotonic 재보정(KCHS OOF)"))
    return out, pte_raw, ppre_raw, ppost_raw, oof_s, band_diag


# ---------------------------------------------------------------- (2) 연령대별
gtr = age_band_idx(AGE_TR); gte = age_band_idx(AGE_TE)
gp = age_band_idx(AGE_P); gq = age_band_idx(AGE_Q)
o, a_te, a_pre, a_post, a_oof, a_diag = build_strat(gtr, gte, gp, gq, "S2_age", "연령대 20-39/40-59/60+")
rows += o
for r in o: print(r)
pd.DataFrame(a_diag).to_csv(os.path.join(RES, "path6_01_ageband_diag.csv"),
                            index=False, encoding="utf-8-sig")
print(pd.DataFrame(a_diag))

# ---------------------------------------------------------------- (3) 성별
smap = {v: i for i, v in enumerate(sorted(pd.unique(SEX_TR)))}
str_ = np.array([smap.get(v, -1) for v in SEX_TR])
ste = np.array([smap.get(v, -1) for v in SEX_TE])
sp = np.array([smap.get(v, -1) for v in SEX_P])
sq = np.array([smap.get(v, -1) for v in SEX_Q])
o, s_te, s_pre, s_post, s_oof, s_diag = build_strat(str_, ste, sp, sq, "S3_sex", "성별 분리")
rows += o
for r in o: print(r)
pd.DataFrame(s_diag).to_csv(os.path.join(RES, "path6_01_sexband_diag.csv"),
                            index=False, encoding="utf-8-sig")
print(pd.DataFrame(s_diag))

# ---------------------------------------------------------------- R4 전역-층별 로짓 혼합 (α 내부 CV)
def blend_alpha_cv(oof_strat, name, ste_, spre_, spost_):
    """α는 KCHS 학습셋 OOF로만 결정(§2.10 준수)."""
    best, tab = None, []
    for al in np.arange(0, 1.01, 0.1):
        s = al * logit(oof_g) + (1 - al) * logit(oof_strat)
        v = wauc(YTR, s, WTR)
        tab.append((round(al, 2), round(v, 6)))
        if best is None or v > best[1]:
            best = (al, v)
    print(f"  [{name}] KCHS OOF α curve:", tab)
    al = best[0]
    return al, best[1], (al * logit(g_te) + (1 - al) * logit(ste_),
                         al * logit(g_pre) + (1 - al) * logit(spre_),
                         al * logit(g_post) + (1 - al) * logit(spost_)), tab


for nm, oofs, tri in [("S2_age", a_oof, (a_te, a_pre, a_post)),
                      ("S3_sex", s_oof, (s_te, s_pre, s_post))]:
    al, cv, (bt, bp, bq), tab = blend_alpha_cv(oofs, nm, *tri)
    rows.append(report(f"{nm}_R4_blend_a{al:.1f}", bt, bp, bq,
                       note=f"전역·층별 로짓혼합, α={al:.1f} (KCHS OOF 선택, OOF AUROC={cv:.5f})"))
    print(rows[-1])
    # 진단: 전 α 외부곡선 (선택근거로 쓰면 비준수 → 진단만)
    for al2 in [0.0, 0.3, 0.5, 0.7, 1.0]:
        bt2 = al2 * logit(g_te) + (1 - al2) * logit(tri[0])
        bp2 = al2 * logit(g_pre) + (1 - al2) * logit(tri[1])
        bq2 = al2 * logit(g_post) + (1 - al2) * logit(tri[2])
        rows.append(report(f"DIAG_{nm}_blend_a{al2:.1f}", bt2, bp2, bq2,
                           compliant=False,
                           note="★진단전용: α 외부곡선. 외부로 α 고르면 §2.10 위반"))

# ---------------------------------------------------------------- R5 대상 층내 순위 사상 (회색)
def tgt_rank_map(strat_scores_tr, gtr_, scores, gvec):
    """대상 데이터에서 층내 순위 -> KCHS 동일층 점수분포의 분위수로 사상."""
    out = np.zeros(len(scores))
    for L in sorted(set(gvec.tolist())):
        m = gvec == L
        if m.sum() == 0: continue
        ref = strat_scores_tr[gtr_ == L]
        if len(ref) == 0: continue
        r = zrank(scores[m])
        out[m] = np.quantile(ref, np.clip(r, 0, 1))
    return out


for nm, oofs, gv, tri in [("S2_age", a_oof, (gte, gp, gq), (a_te, a_pre, a_post)),
                          ("S3_sex", s_oof, (ste, sp, sq), (s_te, s_pre, s_post))]:
    mt = tgt_rank_map(oofs, gtr if nm == "S2_age" else str_, tri[0], gv[0])
    mp_ = tgt_rank_map(oofs, gtr if nm == "S2_age" else str_, tri[1], gv[1])
    mq_ = tgt_rank_map(oofs, gtr if nm == "S2_age" else str_, tri[2], gv[2])
    rows.append(report(f"{nm}_R5_tgtrank", mt, mp_, mq_, compliant=False,
                       note="★회색: 대상 데이터 층내 순위분포 사용(라벨 미사용이나 transductive)"))
    print(rows[-1])

save(rows, "path6_01_strat.csv")
print("TOTAL", round(time.time() - T0, 1), "s")
