# 폐기된(superseded) 결과 파일 — 인용 금지

아래 파일들은 초기 세대 산출물이며, 같은 양을 더 올바른 방법으로 다시 계산한 파일이 있습니다.
문서(원고·부록·모델카드·커버레터)에서 이 파일들을 인용하면 `60_verify_manuscript.py`가 실패합니다.

| 폐기 파일 | 대체 파일 | 폐기 이유 |
|---|---|---|
| `final_metrics.csv` | `review2_metrics.csv`, `review3_metrics.csv` | CI가 재척도 없는 군집 부트스트랩(층당 PSU=2에서 하향편향), CITL이 기울기 자유추정 절편 |
| `final_parsimonious.csv` | `parsimonious_clean.csv` | SHAP 순서·문항수를 hold-out에서 선택(선택 후 성능) |
| `gain_14to17_bootstrap.csv` | `review2_gain.csv` | 미재척도 부트스트랩, P값 중심화 미적용 |
| `fairness.csv`, `fairness_ci.csv`, `review2_fairness.csv`, `review3_fairness.csv` | `review3_fairness_domain.csv` | 순서대로: CI 없음 / 미재척도 / 부분집합 후 설계 재구성 / 단일층 통합했으나 여전히 부분집합 기반. 정본은 전체 설계 복제가중치를 도메인에 적용 |
| `oracle_ci.csv` | `review3_oracle_ci.csv` | 미재척도 부트스트랩 |
| `table_s1_operating.csv` | `review2_operating.csv` | Youden 지점을 0.01 격자에서 탐색 |
| `table3_characteristics.csv` | `characteristics_availablecase.csv` | 결측을 '해당 없음'으로 계산(식품불안정 2.9% vs 실제 3.7%) |
| `shap_importance.csv` | `shap_importance_train.csv` | hold-out 표본에서 산출 |
| `audit_recompute_*.csv`, `audit_gain_retuned.csv` | `review2_*.csv` | 중간 세대 |

`20_final_pipeline.py`, `41_audit_recompute.py`, `43_consolidate.py`, `46_review_fixes.py`는 이 파일들을 만든 **이력 스크립트**입니다.
현재 정본 수치는 `47_review2_recompute.py`, `48_parsimonious_clean.py`, `51_sap_compliance.py`, `52_subgroup_ci_fix.py`, `53_review3_fixes.py`, `54_review3_ci.py`, `36_utility.py`가 만듭니다.

## 2세대 폐기 — 교정 전(pre-correction) 산출물

아래는 소득분위·2017 소득 거부코드·음주 오코딩·2018 신체활동·수면 정의·연령 상한 교정 **이전**,
그리고 단순무작위 분할(군집 공유) 위에서 계산된 파일들이다. 정본은 모두 `primary2_*.csv` 다.

| 폐기 파일 | 대체 파일 |
|---|---|
| `review2_metrics.csv`, `review3_metrics.csv` | `primary2_metrics.csv` |
| `review2_gain.csv` | `primary2_gain.csv` |
| `review3_fairness_domain.csv` | `primary2_fairness.csv` |
| `parsimonious_clean.csv` | `primary2_parsimonious.csv` |
| `final_sensitivity.csv`, `sensitivity_full.csv` | `primary2_sensitivity.csv` |
| `review2_operating.csv`, `table_s4_operating.csv` | `primary2_operating.csv`(개발셋 확정 임계값), `primary2_operating_ref.csv`(Youden·유병률), `primary2_triage.csv` |
| `table_s4_netbenefit.csv` | `primary2_netbenefit_models.csv` |
| `characteristics_availablecase.csv` | `primary2_characteristics.csv` |
| `shap_importance_train.csv` | `primary2_shap.csv` |
| `income_fix_sensitivity.csv`, `age_topcode_sensitivity.csv` | 교정이 주분석에 반영됨 — 단일교정 효과는 부록3 Table S10(미교정 기준선 대비)에만 남김 |

생성 스크립트: `70_primary_corrected.py`(주분석) · `71_utility_primary2.py`(운영점·순편익) ·
`72_sensitivity_primary2.py`(나머지 민감도) · `73_figures_primary2.py`(표본특성·그림 7종).
문서 주입은 `61_sync_numbers.py`, 기계 검증은 `60_verify_manuscript.py` 가 담당한다.
`results/figures/*.png` 도 모두 `73_figures_primary2.py` 재생성분이다.
