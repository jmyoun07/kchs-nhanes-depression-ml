# RUN MANIFEST — 재현성

실행순서: 01_harmonize → 03_train → 05_external → 06_decision_curve → 07_interpret_fairness → 08_parsimonious → 09_tables → 10_domestic_model

## 패키지 버전
- numpy: 2.4.5
- pandas: 3.0.3
- sklearn: 1.9.0
- lightgbm: 4.6.0
- shap: 0.52.0
- pyreadstat: 1.3.5
- scipy: 1.18.0
- matplotlib: 3.10.9

## 알려진 재현성 한계(#23)
- 스크립트가 절대경로(C:/Users/rudtn/...) 하드코딩 → 향후 config/CLI화 필요
- seed=42 고정. LightGBM n_jobs=-1은 환경 따라 미세 변동 가능
- 원시데이터(KCHS 재배포금지/NHANES 공개)는 미포함; 코드+변수사전으로 재현