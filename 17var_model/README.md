# 17변수 모델 (17-predictor model)

> **이 폴더는 조화 예측변수를 17개로 확장한 모델 전용입니다.**
> 원 프로토콜(14변수) 작업물은 프로젝트 루트에 **그대로 보존**되어 있습니다 (`../docs/manuscript.md`, `../논문원고_우울ML.docx`).

## 14변수 vs 17변수
| | 14변수 (원 프로토콜) | **17변수 (이 폴더)** |
|---|---|---|
| 예측변수 | Table 1의 14개 | 14 + **employment · oral_health · food_insecurity** |
| 근거 | 원 사전지정 | 사전지정 **amendment**(분석 착수 전 잠금, 2026-07-15) |
| 지위 | 원 primary (보존) | amendment primary / 확장분석 |

## 확정 성능 (leakage-free, 내부 CV로만 선택·튜닝)
```
주모델: 17변수 LightGBM  (동일 프로토콜 튜닝 시 CV-AUROC 0.840 > RF 0.835 > enet 0.822)
개발표본: 풀링 2,048,510 중 학습 500,000 / hold-out 100,000 (결과층화 분할)

내부(hold-out)  AUROC 0.832 [0.822–0.841]  (층=연도×설계층)   slope 1.00  CITL +0.10
외부 NHANES pre  AUROC 0.772 [0.735–0.808]   slope 0.66  CITL +0.49 → 재보정 0.99 / 0.00
외부 NHANES post AUROC 0.769 [0.739–0.799]   slope 0.68  CITL +1.11 → 재보정 0.99 / 0.00

14(자체 튜닝)→17 이득: pre +0.023 [0.013,0.033] · post +0.016 [0.008,0.023]  (P=.0005, 중심화 부트스트랩)
경량 5문항 = 배포모델(사전지정 0.02 여유, 학습셋 내부 선택): 내부 0.809 · pre 0.742 · post 0.746   (손실 0.023/0.031/0.023)
7문항 변형(0.01 엄격기준, 외부정보 개입 → 사후·탐색적): 내부 0.819 · pre 0.760 · post 0.758
외부 상한 ~0.77: PSU 분리 오라클 0.764–0.776 · 카탈로그된 2,199회 실험
```
H1·H2·H3 지지, H4는 부분 지지(프로토콜이 허용 손실을 정량화하지 않았고, 외부 손실이 내부보다 큼).
CI는 **재척도 층별 군집 부트스트랩**(층당 n_h−1개 PSU 복원추출 + 가중치 재척도) 1000회. NHANES는 층당 PSU가 2개뿐이라 재척도 없이는 CI가 약 29% 좁아진다. CITL은 offset(기울기=1) 모델 절편이며 양수 = 과소예측.

## 원고·문서
| 파일 | 내용 |
|---|---|
| `논문원고_우울ML_17변수.docx` | **JMIR 규격 제출 원고** (구조화 초록, 표 2개, 그림 7개 — 그림은 본문 최초 인용 순서로 번호 부여) |
| `docs/manuscript_17var.md` | 원고 **원본**(docx는 `44_make_docx.py`로만 재생성) |
| `docs/TRIPOD_AI_checklist.md` → `부록1_TRIPOD체크리스트.docx` | Multimedia Appendix 1 |
| `docs/appendix2_crosswalk.md` → `부록2_조화크로스워크.docx` | Multimedia Appendix 2 — 연도별 조화 크로스워크 + 사전지정 amendment(영문) |
| `docs/appendix3_utility.md` → `부록3_임상유용성.docx` | Multimedia Appendix 3 — 임상유용성·특성·민감도(Table S1·S4–S8) |
| `docs/appendix4_ceiling.md` → `부록4_전이천장분석.docx` | Multimedia Appendix 4 — 전이 천장 분석(Table S2–S3) |
| `docs/prespecification_17var.md` | 사전지정서 원본(한국어, 2026-07-15 잠금) |
| `docs/model_card_17var.md` | 배포용 모델 카드 |
| `docs/evidence_ledger.md` | 내부 QC — 원고의 모든 수치 → 근거 파일 대조표 |
| `docs/cover_letter.md` | 커버레터 |

## 코드
| 스크립트 | 역할 |
|---|---|
| `20_final_pipeline.py` | 최종 파이프라인(개발→내부→외부→경량→민감도). 정본 `final_*.csv` 생성 |
| `21_figures_tables.py` | SHAP·공정성·DCA·보정곡선 |
| `22_supplement.py` · `23_extra_metrics.py` | operating characteristics·ECE·확장 민감도 |
| `37_gain_boot.py` | 14→17 짝지음 부트스트랩 |
| `38_table_figs.py` | Figure 1(특성)·Figure 4(민감도) — 값을 결과 CSV에서 읽음 |
| `46_review_fixes.py`~`50_figures_update.py` | 2차 심사 재계산 — 재척도 부트스트랩·정확 Youden·available-case 유병률·하위군 CI/CITL(pre+post)·연속점수 민감도·경량모델 재선택·그림 갱신 |
| `41_audit_recompute.py` | 감사 재계산 — CITL(offset)·층별 부트스트랩·컷오프≥5 외부·연령≥30·SHAP순서 경량모델 |
| `42_audit_tuning.py` | 감사 재계산 — 3알고리즘 동일 프로토콜 튜닝·14변수 자체 튜닝 후 이득 재추정 |
| `43_consolidate.py` | 감사 결과를 정본 `final_*.csv`로 반영 |
| `44_make_docx.py` · `45_make_appendix_docx.py` | 원고·부록 docx 생성 |

> 데이터 경로는 모두 `17var_model/data`, 결과는 `17var_model/results`로 통일되어 있습니다.

## 검증 (반복 실행 가능)

```bash
python 61_sync_numbers.py   # 결과 파일 → 문서로 수치 주입
python 60_verify_manuscript.py  # 353항목 기계 검증
```

정본 결과 파일에서 기대 문자열을 만들어 원고·부록·모델카드·커버레터에 실제로 그 값이 있는지 304항목 기계 검증합니다. 폐기 파일 인용, 그림 인용 순서, 부록표 누락, 초록 단어수, 금지 표현도 함께 검사합니다. 수치를 바꿀 때는 **결과 파일을 먼저 갱신하고 이 스크립트를 통과시킨 뒤** docx를 재생성하세요. 폐기 파일 목록은 `results/_SUPERSEDED.md` 참조.

## 제출 전 채울 것
- 저자·소속·교신저자·CRediT author contributions (`[to be completed]`)
- IRB 승인번호 (`[to be inserted]`, §2.14)
- 코드·변수사전·모델카드·프로토콜/사전지정서를 공개 저장소에 올리고 DOI 삽입 (`[to be inserted]` — §2.1·Data Availability는 현재 "수락 시 공개 예정"으로 기술)
- **공개 레지스트리 등록은 하지 않았음**을 전제로 문구가 작성되어 있음. 실제로 OSF 등에 등록했다면 §2.1에 등록번호를 넣어 "pre-registered"로 바꿀 것
- TRIPOD+AI 공식 서식 업로드
