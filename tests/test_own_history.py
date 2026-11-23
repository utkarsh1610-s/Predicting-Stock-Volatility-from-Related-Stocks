import numpy as np
import pandas as pd
import pytest

import own_history_test as oh
from baselines import COR, HAR, SEC
from placebo import MKT


def synth(sector_matters=True, seed=0, n_stocks=30, n_days=900, own_noise=0.02):
    """y = stock level + market + sector factor + noise. Own HAR inputs reveal y almost exactly
    (so own history makes the sector redundant); the market and sector inputs are clean factors."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2014-01-02", periods=n_days)
    m = rng.normal(0, 0.4, n_days)
    secs = {s: rng.normal(0, 0.4, n_days) * (1.0 if sector_matters else 0.0) for s in "ABC"}
    recs = []
    for i in range(n_stocks):
        s = "ABC"[i % 3]
        level = rng.normal(-4.5, 0.4)
        y = level + m + secs[s] + rng.normal(0, 0.15, n_days)
        d = pd.DataFrame({"ticker": f"T{i}", "sector": s, "date": dates, "y_date": dates + pd.offsets.BDay(1),
                          "y": y, "lv": y - m - secs[s]})
        for c in HAR:
            d[c] = y + rng.normal(0, own_noise, n_days)               # own history ~ reveals y when noise is small
        for c in MKT:
            d[c] = m + rng.normal(0, 0.02, n_days)
        for c in SEC + COR:
            d[c] = secs[s] + m + rng.normal(0, 0.02, n_days)          # sector average includes the market
        recs.append(d)
    df = pd.concat(recs, ignore_index=True)
    df["split"] = np.where(df["date"] < "2016-06-01", "train", "val")
    return df


def test_stock_level_is_fit_on_training_rows_only():
    df = synth(n_stocks=6, n_days=900)
    assert (df["split"] == "val").any() and (df["split"] == "train").any()      # otherwise the test is vacuous
    a = oh.add_stock_level(df)
    wrecked = df.copy()
    wrecked.loc[wrecked["split"] == "val", "lv"] += 50.0
    b = oh.add_stock_level(wrecked)
    np.testing.assert_array_equal(a["stock_level"].to_numpy(), b["stock_level"].to_numpy())


def test_stocks_without_training_rows_are_dropped():
    df = synth(n_stocks=3, n_days=300)
    df.loc[df["ticker"] == "T0", "split"] = "val"
    assert "T0" not in set(oh.add_stock_level(df)["ticker"])


def test_the_2x2_shows_sector_help_without_own_history_and_none_with_it():
    df = oh.add_stock_level(synth(sector_matters=True))
    out = pd.DataFrame(oh.comparisons(oh.fit_all(df), "ALL"))
    r = out[out["comparison"] == "sector beyond market"].set_index("own_history")
    assert r.loc["without", "qlike_change"] < -0.05 and r.loc["without", "dm_p"] < 1e-6
    assert r.loc["with", "qlike_change"] > -0.01
    assert oh.verdict(out, "sector") == "redundant"


def test_no_sector_factor_means_the_sector_is_absent_even_without_own_history():
    df = oh.add_stock_level(synth(sector_matters=False, seed=2))
    out = pd.DataFrame(oh.comparisons(oh.fit_all(df), "ALL"))
    assert oh.verdict(out, "sector") == "absent"


def test_own_history_row_is_reported_and_helps():
    df = oh.add_stock_level(synth())
    out = pd.DataFrame(oh.comparisons(oh.fit_all(df), "ALL"))
    own = out[out["comparison"] == "own history beyond market + stock level"].iloc[0]
    assert own["qlike_change"] < -0.1                                 # own history is very informative here


def test_flag_requires_the_strict_threshold():
    df = oh.add_stock_level(synth(sector_matters=True))
    L = oh.fit_all(df)
    loose = pd.DataFrame(oh.comparisons(L, "S", strict_p=0.05))
    strict = pd.DataFrame(oh.comparisons(L, "S", strict_p=1e-300))
    assert loose["flag"].any() and not strict["flag"].any()


def test_verdict_is_mixed_when_the_sector_helps_even_with_own_history():
    df = oh.add_stock_level(synth(sector_matters=True, own_noise=0.6, seed=4))     # own history is noisy
    out = pd.DataFrame(oh.comparisons(oh.fit_all(df), "ALL"))
    r = out[out["comparison"] == "sector beyond market"].set_index("own_history")
    assert r.loc["without", "qlike_change"] < -0.01 and r.loc["with", "qlike_change"] < -0.005
    assert oh.verdict(out, "sector") == "mixed"
