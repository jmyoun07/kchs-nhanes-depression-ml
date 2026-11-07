# -*- coding: utf-8 -*-
# docx → hwp 변환 스크립트 (한글 Office 설치 필요)
# 사용법:  python docx를_hwp로_변환.py  [원본.docx]  [결과.hwp]
#   인자 없으면 기본값: IRB_우울ML_심의용_v2.docx → IRB_우울ML_심의용_v2.hwp
# ※ 실행 중 한글 '보안 승인' 대화상자가 뜨면 "모두 허용"을 클릭하세요(1회).
#    이 승인은 사용자가 대화형으로 눌러야 하며 자동화로는 통과되지 않습니다.

import sys, os
try:
    import win32com.client
except ImportError:
    print("pywin32 미설치 → 먼저: pip install pywin32"); sys.exit(1)

src = sys.argv[1] if len(sys.argv) > 1 else "IRB_우울ML_심의용_v2.docx"
dst = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(src)[0] + ".hwp"
src, dst = os.path.abspath(src), os.path.abspath(dst)

hwp = win32com.client.Dispatch("HWPFrame.HwpObject")
try:
    hwp.RegisterModule("FilePathCheckDLL", "FilePathCheckerModule")  # 보안 모듈(있으면 대화상자 생략)
except Exception:
    pass
print("한글에서 여는 중... (보안 대화상자가 뜨면 '모두 허용' 클릭)")
hwp.Open(src, "", "")
hwp.SaveAs(dst, "HWP", "")
hwp.Quit()
print("완료:", dst)
