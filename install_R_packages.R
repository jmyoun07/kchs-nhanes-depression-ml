# 이 연구에서 R로 수행하는 분석용 패키지 설치
# (보정·의사결정곡선·복합표본 설계 처리는 R 생태계가 성숙)
pkgs <- c(
  "rms",       # 회귀·보정(calibration) 진단
  "dcurves",   # 의사결정곡선분석(net benefit)
  "survey",    # 복합표본 설계(strata/PSU/weight)
  "pROC",      # ROC·AUROC 및 CI
  "CalibrationCurves",  # 외부검증 보정곡선
  "tidyverse"  # 데이터 처리·시각화
)
to_install <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(to_install) > 0) {
  install.packages(to_install, repos = "https://cloud.r-project.org")
}
invisible(lapply(pkgs, require, character.only = TRUE))
cat("R 패키지 준비 완료\n")
