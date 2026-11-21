import numpy as np
import pytest

from metrics import dm_test, mse_logvol, qlike, qlike_loss, smearing_factor


def test_qlike_zero_when_perfect_and_nonnegative():
    y = np.log([0.01, 0.02, 0.05])
    assert qlike(y, y) == pytest.approx(0.0)
    assert (qlike_loss(np.exp(2 * y), np.exp(2 * (y + 0.3))) >= 0).all()


def test_qlike_known_value_and_asymmetry():
    # true variance 4, forecast variance 2: r=2 -> 2 - ln2 - 1
    assert qlike_loss(4.0, 2.0) == pytest.approx(1 - np.log(2))
    # under-predicting by a factor of 2 costs more than over-predicting by a factor of 2
    assert qlike_loss(1.0, 0.5) > qlike_loss(1.0, 2.0)


def test_mse_logvol():
    assert mse_logvol([0, 1], [1, 3]) == pytest.approx((1 + 4) / 2)


def test_smearing_restores_mean_variance():
    rng = np.random.default_rng(0)
    yhat = rng.normal(-4, 0.3, 50_000)
    y = yhat + rng.normal(0, 0.5, 50_000)         # log-normal noise: mean variance > median
    s = smearing_factor(y, yhat)
    assert s > 1.5                                  # exp(2 * 0.25) = 1.65
    assert np.mean(np.exp(2 * y)) == pytest.approx(np.mean(np.exp(2 * yhat)) * s, rel=0.01)
    assert qlike(y, yhat, smear=s) < qlike(y, yhat)


def test_dm_zero_lag_matches_paired_t_statistic():
    rng = np.random.default_rng(1)
    a, b = rng.normal(1, 1, 400), rng.normal(1.1, 1, 400)
    mean_d, stat, p = dm_test(a, b, lag=0)
    d = a - b
    assert stat == pytest.approx(d.mean() / (d.std(ddof=0) / np.sqrt(len(d))))
    assert mean_d == pytest.approx(d.mean())


def test_dm_detects_a_real_difference_and_not_noise():
    rng = np.random.default_rng(2)
    base = rng.exponential(1, 600)
    better = base - 0.15 + rng.normal(0, 0.1, 600)
    mean_d, stat, p = dm_test(better, base)
    assert mean_d < 0 and stat < -5 and p < 1e-6          # A (better) has lower loss
    _, stat_n, p_n = dm_test(base + rng.normal(0, 0.1, 600), base + rng.normal(0, 0.1, 600))
    assert p_n > 0.01


def test_dm_hac_widens_the_interval_for_autocorrelated_differences():
    rng = np.random.default_rng(3)
    e = rng.normal(0, 1, 1000)
    d = np.convolve(e, np.ones(10) / 10, mode="valid")     # strongly autocorrelated, mean 0
    z = np.zeros(len(d))
    _, stat_naive, _ = dm_test(d, z, lag=0)
    _, stat_hac, _ = dm_test(d, z, lag=10)
    assert abs(stat_hac) < abs(stat_naive) or abs(stat_naive) < 1   # HAC is more conservative


def test_dm_antisymmetric():
    rng = np.random.default_rng(4)
    a, b = rng.normal(0, 1, 200), rng.normal(0.2, 1, 200)
    m1, s1, p1 = dm_test(a, b)
    m2, s2, p2 = dm_test(b, a)
    assert m1 == pytest.approx(-m2) and s1 == pytest.approx(-s2) and p1 == pytest.approx(p2)


def test_dm_identical_series_is_not_significant_and_does_not_warn(recwarn):
    a = np.random.default_rng(5).normal(1, 1, 100)
    assert dm_test(a, a.copy()) == (0.0, 0.0, 1.0)
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]
