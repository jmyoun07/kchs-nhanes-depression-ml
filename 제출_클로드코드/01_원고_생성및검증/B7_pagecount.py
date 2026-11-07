# -*- coding: utf-8 -*-
"""제출본 docx 의 실제 쪽수를 Word 로 센다(섹션별 시작 쪽 포함).

추정이 아니라 Word 가 매기는 실제 쪽번호다. 인용용으로 쓰려면 이 값이어야 한다.
"""
import os, sys
sys.stdout.reconfigure(encoding="utf-8")
import win32com.client as win32

D = r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model"
DOC = os.path.abspath(f"{D}/논문원고_우울ML_17변수.docx")
HEADS = ["Abstract", "Introduction", "Methods", "Results", "Discussion",
         "Figure Legends", "Acknowledgments", "References", "Abbreviations"]
WD_ACTIVE_END_PAGE = 3   # wdActiveEndPageNumber

word = win32.gencache.EnsureDispatch("Word.Application")
word.Visible = False
word.DisplayAlerts = False
try:
    doc = word.Documents.Open(DOC, ReadOnly=True)
    try:
        doc.Repaginate()
        total = doc.ComputeStatistics(2)          # wdStatisticPages
        words = doc.ComputeStatistics(0)          # wdStatisticWords
        starts = {}
        for p in doc.Paragraphs:
            t = p.Range.Text.strip().rstrip("\r")
            if t in HEADS and t not in starts:
                starts[t] = p.Range.Information(WD_ACTIVE_END_PAGE)
        print(f"전체 {total}쪽 · {words:,}단어  ({os.path.basename(DOC)})\n")
        order = [h for h in HEADS if h in starts]
        print(f"{'섹션':<18}{'시작 쪽':>8}{'분량(쪽)':>10}")
        for i, h in enumerate(order):
            nxt = starts[order[i + 1]] if i + 1 < len(order) else total + 1
            print(f"{h:<18}{starts[h]:>8}{nxt - starts[h]:>10}")
    finally:
        doc.Close(False)
finally:
    word.Quit()
