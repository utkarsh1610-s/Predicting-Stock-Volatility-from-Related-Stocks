import numpy as np
import pandas as pd

import group_analysis as ga


def world(n_stocks=12, seed=0, stress_gain=0.0, volatile_gain=0.0):
    """Panel 2019-2021 plus predictions for har / har_sector on the validation split.
    har_sector's loss is lower by `stress_gain` on stressed days and by `volatile_gain` in each
    stock's top recent-vol quartile; otherwise the two models have the same loss distribution."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", "2021-12-31")
    vix = pd.Series(np.where(rng.random(len(dates)) < 0.3, 35.0, 15.0), index=dates)
    rows = []
    for t in range(n_stocks):
        har22 = rng.normal(-4, 0.4, len(dates))
        rows.append(pd.DataFrame({"ticker": f"T{t}", "date": dates, "y_date": dates + pd.offsets.BDay(1),
                                  "vix": vix.to_numpy(), "har_22": har22,
                                  "sector": ["Financials", "Energy", "Utilities"][t % 3]}))
    panel = pd.concat(rows, ignore_index=True)
    panel["split"] = np.where(panel["y_date"] < "2020-01-01", "train", "val")
    val = panel[panel["split"] == "val"].copy()
    q = ga.vol_quartiles(panel, "val")
    base = rng.gamma(2.0, 0.1, len(val))
    gain = stress_gain * (val["vix"].to_numpy() > 25)
    gain += volatile_gain * (val["har_22"].to_numpy() >= q.loc[val["ticker"], "q75"].to_numpy())
    keys = val[["ticker", "date", "y_date"]].reset_index(drop=True)
    preds = {"har": keys.assign(loss=base + 0.05 * rng.standard_normal(len(val))),
             "har_sector": keys.assign(loss=base + 0.05 * rng.standard_normal(len(val)) - gain)}
    return panel, preds


def test_nw_dummy_test_returns_the_difference_in_means():
    rng = np.random.default_rng(0)
    dummy = rng.random(2000) < 0.3
    d = rng.normal(0, 1, 2000) + 0.5 * dummy
    coef, p = ga.nw_dummy_test(d, dummy)
    assert np.isclose(coef, d[dummy].mean() - d[~dummy].mean())
    assert p < 1e-6
    coef0, p0 = ga.nw_dummy_test(rng.normal(0, 1, 2000), dummy)
    assert p0 > 0.01


def test_quartiles_use_only_rows_before_the_split():
    panel, _ = world()
    q = ga.vol_quartiles(panel, "val")
    before = panel[panel["y_date"] < "2020-01-01"]
    assert np.allclose(q["q75"], before.groupby("ticker")["har_22"].quantile(0.75).loc[q.index])
    shifted = panel.copy()
    shifted.loc[shifted["y_date"] >= "2020-01-01", "har_22"] += 10     # change only validation-period values
    assert np.allclose(ga.vol_quartiles(shifted, "val")["q75"], q["q75"])


def test_stocks_with_short_history_get_no_volatility_group():
    panel, preds = world()
    late = (panel["ticker"] == "T0") & (panel["y_date"] < "2020-01-01")
    panel = panel[~late | (panel["date"] > "2019-11-01")]               # T0: only ~40 rows before validation
    out = ga.add_groups(preds["har"], panel, "val")
    assert out.loc[out["ticker"] == "T0", "recent_vol"].isna().all()
    assert out.loc[out["ticker"] != "T0", "recent_vol"].notna().all()


def test_groups_come_from_the_forecast_date_not_the_predicted_date():
    panel, preds = world()
    out = ga.add_groups(preds["har"], panel, "val")
    m = out.merge(panel[["ticker", "date", "vix"]], on=["ticker", "date"], suffixes=("", "_panel"))
    assert (m["vix"] == m["vix_panel"]).all()


def test_finds_a_planted_stress_effect_and_not_an_absent_one():
    ga_pairs = [("har_sector", "har")]
    for gain, expect in [(0.05, True), (0.0, False)]:
        panel, preds = world(stress_gain=gain, seed=1)
        old = ga.PAIRS
        try:
            ga.PAIRS = ga_pairs
            res, diff = ga.analyse(preds, panel, "val")
        finally:
            ga.PAIRS = old
        s = diff[diff["test"] == "stressed minus calm"].iloc[0]
        if expect:
            assert s.diff_in_mean_loss_gap < 0 and s.p < 0.01
            g = res[res["grouping"] == "stress"].set_index("group")["qlike_change"]
            assert g["stressed (VIX>25)"] < g["calm (VIX<=25)"] - 0.1
        else:
            assert s.p > 0.01


def test_finds_a_planted_volatile_stock_effect():
    panel, preds = world(volatile_gain=0.05, seed=2)
    old = ga.PAIRS
    try:
        ga.PAIRS = [("har_sector", "har")]
        res, diff = ga.analyse(preds, panel, "val")
    finally:
        ga.PAIRS = old
    v = diff[diff["test"] == "top minus bottom quartile"].iloc[0]
    assert v.diff_in_mean_loss_gap < 0 and v.p < 0.01
    stress = diff[diff["test"] == "stressed minus calm"].iloc[0]
    assert stress.p > 0.01                                               # no stress effect was planted
