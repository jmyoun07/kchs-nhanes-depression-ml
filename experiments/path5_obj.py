# -*- coding: utf-8 -*-
"""
path5_obj.py — 경로5 커스텀 목적함수(손실) 라이브러리.

모든 함수는 LightGBM native custom objective 시그니처를 따른다:
    f(preds, dataset) -> (grad, hess)
preds 는 raw score(로짓). init_score 를 쓰면 preds 에 이미 포함되어 들어온다.

주의(공정 비교):
  * LightGBM 리프값 = -sum(g)/(sum(h)+lambda) 이므로 g,h 를 같은 상수로 스케일하면
    (reg_lambda=0) 거의 상쇄된다. 다만 min_sum_hessian_in_leaf 제약이 달라지므로
    hess_norm=True 옵션으로 총 헤시안을 logloss 와 맞출 수 있게 했다.
"""
import numpy as np

EPS = 1e-12
HMIN = 1e-6


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _norm(g, h, y, p_ref=None, on=False):
    """총 헤시안을 logloss 기준과 맞춘다(옵션)."""
    if not on:
        return g, np.maximum(h, HMIN)
    h = np.maximum(h, HMIN)
    if p_ref is None:
        return g, h
    href = np.sum(np.maximum(p_ref * (1 - p_ref), HMIN))
    c = href / max(np.sum(h), EPS)
    return g * c, h * c


# --------------------------------------------------------------- 0. 대조군(=binary logloss)
def make_logloss():
    def obj(pred, ds):
        y = ds.get_label()
        p = _sig(pred)
        return p - y, np.maximum(p * (1 - p), HMIN)
    return obj


# --------------------------------------------------------------- 1. focal loss
def make_focal(gamma=2.0, alpha=None, hess_norm=False):
    """FL = -a_t (1-p_t)^g log(p_t).  alpha=None -> 클래스 가중 없음(a=0.5 대칭)."""
    g_ = float(gamma)

    def obj(pred, ds):
        y = ds.get_label()
        p = _sig(pred)
        pt = np.where(y > 0.5, p, 1 - p)
        pt = np.clip(pt, 1e-9, 1 - 1e-9)
        at = 1.0 if alpha is None else np.where(y > 0.5, alpha, 1 - alpha)
        s = np.where(y > 0.5, 1.0, -1.0)          # dpt/dx = s*p*(1-p)
        pp = p * (1 - p)
        omp = 1 - pt
        logpt = np.log(pt)
        # dL/dpt
        dL = -at * (-g_ * omp ** (g_ - 1) * logpt + omp ** g_ / pt)
        # d2L/dpt2
        d2L = -at * (g_ * (g_ - 1) * omp ** (g_ - 2) * logpt
                     - 2 * g_ * omp ** (g_ - 1) / pt
                     - omp ** g_ / pt ** 2)
        grad = dL * s * pp
        hess = d2L * (pp ** 2) + dL * s * pp * (1 - 2 * p)
        hess = np.abs(hess)
        return _norm(grad, hess, y, p, hess_norm)
    return obj


# --------------------------------------------------------------- 2. label smoothing
def make_label_smoothing(eps=0.05):
    e = float(eps)

    def obj(pred, ds):
        y = ds.get_label()
        ys = y * (1 - e) + (1 - y) * e
        p = _sig(pred)
        return p - ys, np.maximum(p * (1 - p), HMIN)
    return obj


# --------------------------------------------------------------- 3. robust losses
def make_gce(q=0.5, hess_norm=False):
    """Generalized Cross Entropy (Zhang & Sabuncu 2018): L=(1-pt^q)/q. q->0 = CE, q=1 = MAE."""
    qq = float(q)

    def obj(pred, ds):
        y = ds.get_label()
        p = _sig(pred)
        pt = np.clip(np.where(y > 0.5, p, 1 - p), 1e-9, 1 - 1e-9)
        s = np.where(y > 0.5, 1.0, -1.0)
        pp = p * (1 - p)
        dL = -pt ** (qq - 1)                      # dL/dpt
        d2L = -(qq - 1) * pt ** (qq - 2)
        grad = dL * s * pp
        hess = np.abs(d2L * pp ** 2 + dL * s * pp * (1 - 2 * p))
        return _norm(grad, hess, y, p, hess_norm)
    return obj


def make_savage(hess_norm=False):
    """Savage loss (Masnadi-Shirazi & Vasconcelos): 유계 손실, 라벨잡음 강건."""
    def obj(pred, ds):
        y = ds.get_label()
        z = np.where(y > 0.5, 1.0, -1.0) * pred          # margin
        s = _sig(z)
        # L = (1-s)^2 ;  dL/dz = -2 s (1-s)^2 ;  d2L/dz2 = 2 s (1-s)^2 (3s-1)
        dLdz = -2 * s * (1 - s) ** 2
        d2Ldz2 = 2 * s * (1 - s) ** 2 * (3 * s - 1)
        sgn = np.where(y > 0.5, 1.0, -1.0)
        grad = dLdz * sgn
        hess = np.abs(d2Ldz2)
        return _norm(grad, hess, y, _sig(pred), hess_norm)
    return obj


def make_gradclip(c=0.5):
    """Huber 화(化) 로지스틱: 그래디언트 크기를 c 로 절단(이상치/라벨잡음 영향 제한)."""
    cc = float(c)

    def obj(pred, ds):
        y = ds.get_label()
        p = _sig(pred)
        g = p - y
        gc = np.clip(g, -cc, cc)
        h = np.where(np.abs(g) >= cc, HMIN * 10, np.maximum(p * (1 - p), HMIN))
        return gc, h
    return obj


def make_brier():
    """Brier(제곱) 손실: L=(p-y)^2."""
    def obj(pred, ds):
        y = ds.get_label()
        p = _sig(pred)
        pp = p * (1 - p)
        grad = 2 * (p - y) * pp
        hess = np.abs(2 * pp * pp + 2 * (p - y) * pp * (1 - 2 * p))
        return grad, np.maximum(hess, HMIN)
    return obj


# --------------------------------------------------------------- 4. RankNet (AUROC 직접 근사)
def make_ranknet(y, n_neg_per_pos=5, lam_ce=0.0, seed=42, sigma=1.0):
    """
    쌍(pairwise) 랭킹 손실: L = sum_{i in P, j in N} log(1+exp(-sigma(f_i-f_j)))
    AUROC = P(f_pos > f_neg) 의 매끄러운 상계 -> 순위지표 직접 최적화.
    lam_ce>0 이면 logloss 를 lam_ce 비중으로 섞는다.
    매 부스팅 반복마다 음성표본을 재추출(확률적 근사).
    """
    P = np.where(y > 0.5)[0]
    N = np.where(y <= 0.5)[0]
    rng = np.random.RandomState(seed)
    K = int(n_neg_per_pos)
    n = len(y)

    pi = np.repeat(P, K)

    def obj(pred, ds):
        yy = ds.get_label()
        nj = N[rng.randint(0, len(N), size=len(pi))]
        d = sigma * (pred[pi] - pred[nj])
        s = _sig(-d)                     # dL/dd = -sigma * s
        gg = -sigma * s
        hh = (sigma ** 2) * s * (1 - s)
        g = (np.bincount(pi, weights=gg, minlength=n)
             - np.bincount(nj, weights=gg, minlength=n))
        h = (np.bincount(pi, weights=hh, minlength=n)
             + np.bincount(nj, weights=hh, minlength=n))
        # 스케일: 쌍 개수로 정규화하여 n 스케일로 맞춤
        sc = n / max(len(pi), 1)
        g *= sc
        h *= sc
        if lam_ce > 0:
            p = _sig(pred)
            g = (1 - lam_ce) * g + lam_ce * (p - yy)
            h = (1 - lam_ce) * h + lam_ce * np.maximum(p * (1 - p), HMIN)
        return g, np.maximum(h, HMIN)
    return obj


# --------------------------------------------------------------- 5. Group DRO
def make_group_dro(gid, n_groups, eta=0.01, mode="exp", floor=0.0):
    """
    Group DRO 근사(온라인 지수가중, Sagawa et al. 2020).
      각 부스팅 반복에서 그룹별 평균 logloss L_g 계산 -> q_g <- q_g*exp(eta*L_g), 정규화
      -> 표본가중 w_i = q_{g(i)} / (n_g/n)  (그룹 크기 보정)
    KCHS 내부 그룹만 사용(연도/연령대 등). 외부 데이터 미사용 -> 프로토콜 준수.
    """
    gid = np.asarray(gid, dtype=np.int64)
    cnt = np.bincount(gid, minlength=n_groups).astype(float)
    cnt[cnt == 0] = 1.0
    n = len(gid)
    state = dict(q=np.full(n_groups, 1.0 / n_groups))

    def obj(pred, ds):
        y = ds.get_label()
        p = np.clip(_sig(pred), 1e-9, 1 - 1e-9)
        li = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        Lg = np.bincount(gid, weights=li, minlength=n_groups) / cnt
        q = state["q"] * np.exp(eta * (Lg - Lg.mean()))
        q = np.maximum(q, floor)
        q = q / q.sum()
        state["q"] = q
        w = (q / (cnt / n))[gid]
        g = (p - y) * w
        h = np.maximum(p * (1 - p) * w, HMIN)
        return g, h
    obj.state = state
    return obj


# --------------------------------------------------------------- 6. 전이 정규화(그룹 예측분포 정렬)
def make_transfer_reg(gid, n_groups, lam=1.0, moment=1):
    """
    L = logloss + lam * sum_g n_g/n * (mean_g(p) - mean(p))^2   (moment=1)
    moment=2 이면 그룹별 분산도 전체 분산에 맞추는 항을 추가.
    KCHS 내부 그룹(연도 등) 간 예측분포 차이를 벌점화 -> 그룹불변 표현 유도.
    """
    gid = np.asarray(gid, dtype=np.int64)
    cnt = np.bincount(gid, minlength=n_groups).astype(float)
    cnt[cnt == 0] = 1.0
    n = len(gid)

    def obj(pred, ds):
        y = ds.get_label()
        p = _sig(pred)
        pp = p * (1 - p)
        g = p - y
        h = np.maximum(pp, HMIN)
        mg = np.bincount(gid, weights=p, minlength=n_groups) / cnt
        mall = p.mean()
        dmean = mg - mall
        # d/dp_i [ sum_g (n_g/n)(m_g-mbar)^2 ] = (2/n)(m_{g(i)} - mbar)  (mbar 항 상쇄)
        dpen = (2.0 / n) * dmean[gid]
        g = g + lam * n * dpen * pp        # n 스케일 맞춤
        if moment == 2:
            vg = np.bincount(gid, weights=(p - mg[gid]) ** 2, minlength=n_groups) / cnt
            vall = p.var()
            dv = vg - vall
            dpen2 = (2.0 / n) * dv[gid] * 2 * (p - mg[gid])
            g = g + lam * n * dpen2 * pp
        return g, h
    return obj
