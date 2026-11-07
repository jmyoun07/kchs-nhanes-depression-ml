# 프로토콜 정오표 (Erratum) — 구현 정정 기록

Codex(read-only) 검증에 따라, 프로토콜 대비 **정당한 데이터-코드 정정**을 투명하게 기록한다. (프로토콜은 "NHANES variable codes are exact"이라 명시하므로 침묵 변경이 아니라 erratum으로 남김.)

## Erratum #1 — 자가건강(Self-rated health) NHANES 매핑
- **프로토콜 Table 1 원문:** Self-rated health · Subjective health level · **NHANES `HSQ · HSD010`** · 5-level ordinal.
- **실측 문제:** NHANES 2017–2023 `HSQ`(P_HSQ / HSQ_L)에는 `HSQ590`(HIV검사)만 있고 **`HSD010` 없음**.
- **정정:** 자가건강은 NHANES **`HUQ · HUQ010`**("General health condition", 1=Excellent…5=Poor)에 존재. **P_HUQ(2017–2020, 결측 0%)·HUQ_L(2021–2023) 모두 확인**. KCHS `qoa_01z1`(주관적 건강수준, 1=매우좋음…5=매우나쁨)과 **동일 5단계·동일 방향**.
- **결론(Codex COMPLIANT):** 동일 사전지정 개념("Subjective health level, 5-level ordinal")의 **파일 참조만 수정** → §2.5 "양 설문 교집합" 충족 → **프로토콜대로 조화모델에 포함**. HARKing 아님(사전지정 변수 복원).
- **Caveat:** ① 범주 앵커 의미차(매우좋음/좋음/보통/나쁨/매우나쁨 vs Excellent/Very good/Good/Fair/Poor) — 서열형(1–5)으로 처리, 잔여 문화·의미차 명시. ② 주기별 7/9 결측코드 확인(HUQ010 7=Refused/9=Don't know → 결측 처리).

## Erratum #2 — Table 1 예측변수 복원 (편의 제외 → 프로토콜대로)
이전 라운드에서 rigor 목적으로 제외했던 아래 Table 1 변수를 **프로토콜 원칙대로 재포함**(harmonization cautions: "PA·alcohol·income은 보수적 공통정의로 포함, 잔차는 민감도"):
- **신체활동(active)**, **음주(current_drinker)**, **독거(living_alone)** → 조화모델 재포함, 비교불가는 민감도로 정량화(§2.12).

## 영향
- **프로토콜의 full model = Table 1 전체 14변수(자가건강 포함).** 내부 AUROC: **LightGBM 0.824**(slope 0.995), elastic net 0.802, RF 0.821 — **셋 다 0.80 초과 → H1 충족.**
- 자가건강 빠진 0.76 모델은 "사전지정 변수 누락한 구현오류/ablation"으로 기술.

## 남은 작업
- 사용자: **P_HUQ·HUQ_L 다운로드** → NHANES 자가건강(HUQ010) 확보.
- 그 후: 하모나이저 재실행(자가건강 포함) → **외부검증·재보정·경량·공정성·Table 재생성**(현 외부 수치는 자가건강 미포함 구버전).
