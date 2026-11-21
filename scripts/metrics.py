"""Forecast metrics and the Diebold-Mariano test (numpy only).

Conventions: the target y and forecasts yhat are LOG volatility (log of the Garman-Klass
standard deviation). Variances are exp(2 * y) and exp(2 * yhat).
"""
import math

import numpy as np


def mse_logvol(y, yhat):
    return float(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2))


def qlike_loss(var_true, var_pred):
    """Per-observation QLIKE: v/h - log(v/h) - 1. Zero iff v == h, always >= 0.

    Under-predicting variance (h < v) costs more than over-predicting by the same ratio.
    """
    r = np.asarray(var_true) / np.asarray(var_pred)
    return r - np.log(r) - 1.0


def qlike(y, yhat, smear=1.0):
    """Mean QLIKE from log-vol target and forecast; `smear` rescales the forecast variance."""
    return float(np.mean(qlike_loss(np.exp(2 * np.asarray(y)), np.exp(2 * np.asarray(yhat)) * smear)))


def smearing_factor(y_train, yhat_train):
    """Duan smearing for variance: exp(2*yhat) is the MEDIAN variance of a log-model, below
    its mean (Jensen). Rescaling by mean(exp(2 * residual)) from training data fixes the
    bias without any assumption about the error distribution."""
    return float(np.mean(np.exp(2 * (np.asarray(y_train) - np.asarray(yhat_train)))))


def _norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


def dm_test(loss_a, loss_b, lag=None):
    """Diebold-Mariano test on two aligned loss series (one entry per period).

    d_t = loss_a - loss_b. Negative mean => A has the lower loss (A is better).
    Long-run variance is Newey-West (Bartlett kernel), `lag` defaults to floor(T^(1/3)).
    Returns (mean_d, statistic, two_sided_p). Normal reference distribution.
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    t = len(d)
    if t < 10:
        raise ValueError("need at least 10 periods")
    lag = int(t ** (1 / 3)) if lag is None else lag
    dc = d - d.mean()
    lrv = float(dc @ dc) / t
    for k in range(1, lag + 1):
        lrv += 2 * (1 - k / (lag + 1)) * float(dc[k:] @ dc[:-k]) / t
    if lrv <= 1e-18:                       # identical (or constant-difference) loss series
        if abs(d.mean()) < 1e-12:
            return float(d.mean()), 0.0, 1.0
        return float(d.mean()), math.copysign(math.inf, d.mean()), 0.0
    se = math.sqrt(lrv / t)
    stat = d.mean() / se
    return float(d.mean()), float(stat), float(2 * _norm_sf(abs(stat)))
