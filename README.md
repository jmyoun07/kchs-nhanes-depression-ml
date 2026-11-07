# Cross-National & Cross-Pandemic Depression Screening ML

우울증 스크리닝 머신러닝 모델의 개발(KCHS)과 국가 간·COVID 전후 외부검증(NHANES) 연구.

> **투고 대상:** JMIR (대안: JMIR Public Health & Surveillance / Medical Informatics / Mental Health)
> **보고 기준:** TRIPOD+AI (2024) · 편향평가 PROBAST · OSF 사전등록 후 분석 시작

원 프로토콜: [`JMIR_Protocol_KCHS_NHANES_Depression_ML.docx`](JMIR_Protocol_KCHS_NHANES_Depression_ML.docx)

---

## ⚠️ 데이터 취급 원칙

- **데이터 다운로드는 연구자 본인이 직접 수행한다.** (자동 다운로드 스크립트는 실행하지 않음)
- **KCHS 원시자료는 재배포 금지**(질병관리청 예규 제124호) → 저장소·공개물에 절대 포함하지 않는다.
- NHANES는 공개 데이터라 재다운로드 가능 → 역시 저장소에 커밋하지 않는다(`.gitignore` 처리).
- 최종 공개물: **분석 코드 + 변수 조화 사전**만 (원시 데이터 제외).

## 데이터 배치 위치 (연구자가 직접 넣는 곳)

```
data/raw/kchs/     ← KDCA 승인 후 받은 KCHS 원시자료 + 이용지침서
data/raw/nhanes/   ← NHANES .XPT 파일
                     pre  : P_DEMO, P_DPQ, P_SMQ, P_ALQ, P_PAQ, P_SLQ, P_BMX, P_HSQ, P_BPQ, P_DIQ
                     post : DEMO_L, DPQ_L, SMQ_L, ALQ_L, PAQ_L, SLQ_L, BMX_L, HSQ_L, BPQ_L, DIQ_L
```

## 폴더 구조

```
├── protocol(.docx)            원 연구 프로토콜
├── docs/                      SAP(분석계획서), OSF 사전등록, 변수 조화 사전
├── data/
│   ├── raw/{kchs,nhanes}/     원시 데이터 (연구자가 직접 배치, git 제외)
│   ├── interim/               병합·정제 중간 산출물 (git 제외)
│   └── processed/             분석용 최종 데이터셋 (git 제외)
├── src/                       분석 파이프라인 (00~08 순서 실행)
├── models/                    학습된 모델 (git 제외)
└── results/{tables,figures}/  결과 표·그림
```

## 파이프라인 실행 순서 (데이터 확보 후)

| 스크립트 | 역할 | 입력 |
|---|---|---|
| `src/01_harmonize.py` | Table 1 기반 변수 조화 → 분석용 변수 생성 | raw → interim |
| `src/02_preprocess.py` | 결측 대체·전처리 (개발셋에서만 학습) | interim → processed |
| `src/03_train.py` | KCHS에서 모델 개발(elastic net/RF/GBM) | processed |
| `src/04_internal_validation.py` | KCHS hold-out 내부검증 | processed, models |
| `src/05_external_validation.py` | NHANES 지리적·COVID 전후 외부검증 | processed, models |
| `src/06_recalibration.py` | 보정 저하 시 절편/기울기 재보정 | models |
| `src/07_interpret_fairness.py` | SHAP·공정성 하위그룹 분석 | models |
| `src/08_parsimonious.py` | 상위 5–10 변수 경량 모델 | processed, models |

> 데이터 다운로드/병합 절차는 [`docs/data_acquisition_guide.md`](docs/data_acquisition_guide.md) 참고.

## 환경 설정

```bash
# Python
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt

# R (일부 분석: rms, dcurves, survey)
Rscript install_R_packages.R
```

## 재현성

seed 고정, 패키지 버전 고정(`requirements.txt`), 전처리 파이프라인은 개발셋에서만 학습해 외부셋에 그대로 적용(누수 방지).
