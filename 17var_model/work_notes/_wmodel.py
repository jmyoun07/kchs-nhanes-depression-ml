def _adv(ch):
    """Times New Roman 의 글자 폭(em). 숫자는 0.5em 고정폭이다."""
    if ch.isdigit(): return 0.500
    if ch in ".,;:'": return 0.250
    if ch in "-()/": return 0.320
    if ch in "+=<>": return 0.564
    if ch == "%": return 0.833
    if ch == " ": return 0.250
    if ch.isupper(): return 0.722
    return 0.480

def text_width(s, pt=10.0):
    """문자열이 차지하는 폭(인치)."""
    return sum(_adv(c) for c in s) * pt / 72.0

def auto_widths(header, body, total=6.5, pad=0.17):
    """칸 너비를 내용으로 정한다. 줄바꿈 단위가 잘리지 않을 만큼은 보장한다.

    워드는 공백·'/'·하이픈 뒤에서 줄을 바꾼다. 다만 수치 구간(0.814-0.837)에서
    갈라지면 보기 흉하므로 그 하이픈은 끊기지 않는 것으로 본다.
    """
    _SEP = chr(1)
    n = len(header)
    cols = [[header[i]] + [(r[i] if i < len(r) else "") for r in body] for i in range(n)]
    wt, mins = [], []
    for c in cols:
        avg = sum(text_width(x) for x in c) / max(1, len(c))
        longest = 0.0
        for x in c:
            _x = re.sub(r"(?<=\d)-(?=\d)", _SEP, x)
            for w in re.split(r"[\s/\-]+", _x):
                if w: longest = max(longest, text_width(w.replace(_SEP, "-")))
        wt.append(max(avg * 0.75, longest, 0.1))
        mins.append(longest + pad)
    if sum(mins) > total:                      # 지면이 모자라면 최소치를 비례 축소한다
        k = total / sum(mins); mins = [m * k for m in mins]
    s = sum(wt); w = [total * x / s for x in wt]
    for _ in range(n * 2):
        small = [i for i, x in enumerate(w) if x < mins[i] - 1e-9]
        if not small: break
        need = sum(mins[i] - w[i] for i in small)
        big = [i for i in range(n) if w[i] > mins[i] + 1e-9]
        pool = sum(w[i] - mins[i] for i in big)
        if not big or pool < need - 1e-9:      # 더 뺄 곳이 없으면 최소치로 균등 정착
            return [m * total / sum(mins) for m in mins]
        for i in small: w[i] = mins[i]
        for i in big: w[i] -= need * (w[i] - mins[i]) / pool
    return w

