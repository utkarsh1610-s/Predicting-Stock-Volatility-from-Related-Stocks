import numpy as np
import pandas as pd
import pytest

import sector_clustering as sc


def simulate(sector_strength=1.0, seed=0, n_days=1500, per_sector=15, sectors=("A", "B", "C")):
    """lv = stock level + market factor + sector factor + noise, with persistent factors."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n_days)

    def ar1(phi, scale):
        x = np.zeros(n_days)
        for t in range(1, n_days):
            x[t] = phi * x[t - 1] + rng.normal(0, scale)
        return x

    market = ar1(0.9, 0.2)
    cols, sector = {}, {}
    for s in sectors:
        factor = sector_strength * ar1(0.9, 0.2)
        for i in range(per_sector):
            t = f"{s}{i}"
            cols[t] = -4.5 + rng.normal(0, 0.3) + market + factor + rng.normal(0, 0.4, n_days)
            sector[t] = s
    return pd.DataFrame(cols, index=dates), pd.Series(sector)


def test_sector_adds_explained_variance_only_when_a_sector_factor_exists():
    lv, sec = simulate(sector_strength=1.0)
    with_f = sc.build_tables(lv, sec)[0].set_index("scope").loc["ALL"]
    lv0, sec0 = simulate(sector_strength=0.0)
    without = sc.build_tables(lv0, sec0)[0].set_index("scope").loc["ALL"]
    assert with_f["sector_adds"] > 0.10
    assert without["sector_adds"] < 0.01
    assert with_f["r2_market_plus_sector"] > with_f["r2_market"]


def test_same_sector_residual_correlation_is_higher_only_with_a_sector_factor():
    lv, sec = simulate(sector_strength=1.0)
    t = sc.build_tables(lv, sec)[0].set_index("scope").loc["ALL"]
    assert t["resid_corr_same_sector"] > t["resid_corr_other_sectors"] + 0.1
    lv0, sec0 = simulate(sector_strength=0.0)
    t0 = sc.build_tables(lv0, sec0)[0].set_index("scope").loc["ALL"]
    assert abs(t0["resid_corr_same_sector"] - t0["resid_corr_other_sectors"]) < 0.03
    # with the market removed and no sector factor, residuals are essentially uncorrelated everywhere
    assert abs(t0["resid_corr_same_sector"]) < 0.05 and abs(t0["resid_corr_other_sectors"]) < 0.05
    # the market alone makes raw correlations high everywhere
    assert t0["raw_corr_other_sectors"] > 0.3


def test_own_series_is_excluded_from_the_market_factor():
    lv, sec = simulate(n_days=600, per_sector=5, sectors=("A", "B"))
    long = sc.long_panel(lv, sec)
    d = long[(long["ticker"] == "A0")].set_index("date")
    others = lv.drop(columns="A0").mean(axis=1)
    expected = (others - others[d.index].mean()).loc[d.index]
    np.testing.assert_allclose(d["m"].to_numpy(), expected.to_numpy(), atol=1e-9)


def test_a_planted_sector_episode_is_the_largest_month():
    lv, sec = simulate(sector_strength=0.0, n_days=1200)
    cols = [c for c in lv.columns if c.startswith("B")]
    bump = (lv.index >= "2012-03-01") & (lv.index < "2012-04-01")
    lv.loc[bump, cols] += np.log(2.0)                       # sector B doubles for one month
    _, episodes = sc.build_tables(lv, sec)
    top = episodes.iloc[0]
    assert top["sector"] == "B" and top["month"] == "2012-03"
    assert 1.6 < top["ratio"] < 2.2


def test_excess_persistence_is_high_for_a_persistent_sector_factor():
    lv, sec = simulate(sector_strength=1.0)
    t = sc.build_tables(lv, sec)[0].set_index("scope")
    assert (t.loc[["A", "B", "C"], "excess_autocorr_lag1"] > 0.7).all()


def test_test_years_are_dropped_before_anything_is_computed():
    stocks = pd.DataFrame({"date": pd.to_datetime(["2021-12-30", "2021-12-31", "2022-01-03", "2023-06-01"]),
                           "ticker": "X"})
    vix = pd.DataFrame({"date": stocks["date"], "vix_close": 20.0})
    s, v = sc.restrict_to_train_val(stocks, vix)
    assert s["date"].max() == pd.Timestamp("2021-12-31") and v["date"].max() == pd.Timestamp("2021-12-31")
    assert len(s) == 2
