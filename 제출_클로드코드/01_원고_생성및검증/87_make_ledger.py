# -*- coding: utf-8 -*-
"""87_make_ledger.py - 근거대장(docs/evidence_ledger.md)을 정본 CSV 에서 통째로 생성한다.

이 문서는 손으로 관리하다 세대가 갈렸다(k*=5, slope 0.99, AUPRC 0.233 등 폐기값 잔존).
이제 생성물로 바꾼다. 값은 반올림하지 않고 저장된 정밀도를 그대로 싣는다 - 이중 반올림을 없애려는 것이다.
"""
import io, os, sys, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__)); RES = f"{D}/results"
rd = lambda n: pd.read_csv(f"{RES}/{n}")

PM = rd("primary2_metrics.csv").set_index("set")
GAIN = rd("primary2_gain.csv"); SENS = rd("primary2_sensitivity.csv")
PARS = rd("primary2_parsimonious.csv"); FAIR = rd("primary2_fairness.csv")
OPR = rd("primary2_operating_ref.csv"); NB = rd("primary2_netbenefit_models.csv")
RHO = rd("primary2_shap_rho.csv"); XS = rd("primary2_shap_cross.csv")
CH = rd("primary2_characteristics.csv").set_index("characteristic")
NR = rd("primary2_dpq_nonresponse.csv")
DSG = rd("primary2_design.csv") if os.path.exists(f"{RES}/primary2_design.csv") else None

L = ["# Evidence ledger - 정본 결과 파일에서 생성 (internal QC, not for submission)", "",
     "이 문서는 `87_make_ledger.py` 가 `results/primary2_*.csv` 에서 자동 생성한다. **직접 고치지 말 것.**",
     "값은 CSV 에 저장된 정밀도 그대로 싣는다(문서용 반올림은 `61_sync_numbers.py` 가 원고에만 적용한다).", ""]

def table(title, header, rows, note=None):
    L.append(f"## {title}"); L.append("")
    if note: L.append(note); L.append("")
    L.append("| " + " | ".join(header) + " |")
    L.append("|" + "---|" * len(header))
    for r in rows: L.append("| " + " | ".join(str(x) for x in r) + " |")
    L.append("")

if DSG is not None:
    table("개발 설계 (primary2_design.csv)", list(DSG.columns), DSG.values.tolist())

table("주분석 성능 (primary2_metrics.csv)", ["set"] + list(PM.columns),
      [[i] + list(r) for i, r in PM.iterrows()],
      "재보정 3단계: `recal_int_*`(절편만), `recal_is_*`(절편+기울기, 표 2 게재), `recal_platt_*`(온도).")
table("14→17 예측변수 이득 (primary2_gain.csv)", list(GAIN.columns), GAIN.values.tolist())
table("민감도 (primary2_sensitivity.csv, 표 S5)", list(SENS.columns), SENS.values.tolist())
table("경량모델 궤적 (primary2_parsimonious.csv, 표 S6)",
      ["k", "added", "selection_AUROC", "holdout_AUROC", "ext_pre", "ext_post"],
      PARS[["k", "added", "selection_AUROC", "holdout_AUROC", "ext_pre", "ext_post"]].values.tolist(),
      f"배치 모형은 계획의 0.02 마진이 고르는 k*={int(PARS[(PARS.k<17)&(PARS.selection_AUROC>=PARS.loc[PARS.k==17,'selection_AUROC'].iloc[0]-0.02)].k.min())} 이다.")
table("하위군 판별력·보정 (primary2_fairness.csv, 표 S7)", list(FAIR.columns), FAIR.values.tolist())
table("운영점 (primary2_operating_ref.csv, 표 S1)", list(OPR.columns), OPR.values.tolist())
table("순편익 (primary2_netbenefit_models.csv, 표 S3)", list(NB.columns), NB.values.tolist())
table("SHAP 순위 일치 (primary2_shap_rho.csv)", list(RHO.columns), RHO.values.tolist())
table("SHAP 조사별 순위 (primary2_shap_cross.csv)", list(XS.columns), XS.values.tolist())
table("표본특성 (primary2_characteristics.csv, 그림 1·표 S4)", ["characteristic"] + list(CH.columns),
      [[i] + list(r) for i, r in CH.iterrows()])
table("DPQ 미응답 (primary2_dpq_nonresponse.csv, 표 S8)", list(NR.columns), NR.values.tolist(),
      "`nr3_*` 열이 ≥3문항 결측률(면접·MEC·MEC가중 세 프레임), `anymiss_interview_pct` 는 1문항 이상 결측률이다. "
      "표 S8 의 완전례 제외 행(7.7%/14.1%)은 MEC 프레임의 1문항 이상 결측이다.")

L += ["## 탐색 실행 (별도 세대)", "",
      "`results/ceiling_experiments.csv`(표 S11), `results/methods_explore.csv`(표 S12), "
      "`results/deeplearning_explore.csv`(표 S13)는 주분석과 다른 실행이다. 저장 정밀도가 3~4자리라 "
      "부록에서는 저장값을 그대로 싣는다. 주분석 수치와 직접 비교하지 않는다.", ""]

io.open(f"{D}/docs/evidence_ledger.md", "w", encoding="utf8").write("\n".join(L) + "\n")
print(f"근거대장 생성: docs/evidence_ledger.md ({len(L)}줄)")
