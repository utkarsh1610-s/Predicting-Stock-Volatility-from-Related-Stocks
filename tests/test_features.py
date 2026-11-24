import numpy as np
import pandas as pd
import pytest

from features import assign_split, build_panel, garman_klass_logvol

CAL = pd.bdate_range("2019-09-02", "2020-04-30")


def synth(tickers=("AAA", "BBB"), seed=0, cal=CAL):
    rng = np.random.default_rng(seed)
    frames = []
    for t in tickers:
        n = len(cal)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        open_ = close * np.exp(rng.normal(0, 0.003, n))
        hi = np.maximum(open_, close) * np.exp(rng.uniform(0.001, 0.01, n))
        lo = np.minimum(open_, close) * np.exp(-rng.uniform(0.001, 0.01, n))
        frames.append(pd.DataFrame({
            "date": cal, "ticker": t, "open": open_, "high": hi, "low": lo, "close": close,
            "volume": rng.integers(1_000, 5_000, n), "sector": "S"}))
    stocks = pd.concat(frames, ignore_index=True)
    vix = pd.DataFrame({"date": cal, "vix_close": rng.uniform(12, 30, len(cal))})
    return stocks, vix


def test_garman_klass_known_value():
    df = pd.DataFrame({"open": [100.0], "high": [110.0], "low": [90.0], "close": [100.0]})
    expected = 0.5 * np.log(0.5 * np.log(110 / 90) ** 2)
    assert garman_klass_logvol(df).iloc[0] == pytest.approx(expected)


def test_y_is_next_day_value():
    panel, _ = build_panel(*synth())
    for _, g in panel.groupby("ticker"):
        g = g.set_index("date")
        nxt = g.index.to_series().shift(-1)
        ok = g["y_date"] == nxt                      # consecutive rows are consecutive days
        assert (g.loc[ok, "y"].iloc[:-1].values == g["lv"].shift(-1).loc[ok].iloc[:-1].values).all()
        assert (g["y_date"] > g.index.to_series()).all()


def test_no_lookahead_and_train_only_winsor():
    """Wrecking prices after the train period must not move train rows or the bounds."""
    stocks, vix = synth()
    base, b0 = build_panel(stocks, vix)

    cutoff = pd.Timestamp("2019-12-31")
    wrecked = stocks.copy()
    late = wrecked["date"] > cutoff
    wrecked.loc[late, "high"] *= 1.6                     # much wider ranges => volatility changes
    wrecked.loc[late, "low"] /= 1.6
    wrecked.loc[late, "volume"] *= 50
    alt, b1 = build_panel(wrecked, vix)

    assert b0 == b1
    # rows whose feature date is before the cutoff: identical except y at the boundary day
    feats = ["lv", "har_1", "har_5", "har_22", "ret", "dlogvol", "vix"]
    a = base[base["date"] < cutoff].set_index(["ticker", "date"])
    b = alt[alt["date"] < cutoff].set_index(["ticker", "date"])
    pd.testing.assert_frame_equal(a[feats + ["y"]], b[feats + ["y"]])


def test_each_feature_uses_only_past_values(monkeypatch):
    """Changing day t+1 onward changes nothing at day t, across all windows.

    Winsor bounds are pinned so this isolates lookahead in the features; the
    train-only property of the bounds is covered by the test above.
    """
    import features
    monkeypatch.setattr(features, "winsor_bounds", lambda lv, dates, q=None: (-9.0, 0.0))
    stocks, vix = synth(("AAA",))
    base, _ = build_panel(stocks, vix)
    t = pd.Timestamp("2019-12-10")
    mod = stocks.copy()
    after = mod["date"] > t
    mod.loc[after, "high"] *= 1.3
    mod.loc[after, "low"] /= 1.3
    mod.loc[after, "close"] *= 1.02
    mod.loc[after, "volume"] *= 7
    alt, _ = build_panel(mod, vix)
    feats = ["lv", "har_1", "har_5", "har_22", "ret", "dlogvol", "vix"]
    pd.testing.assert_frame_equal(base.set_index("date").loc[:t, feats],
                                  alt.set_index("date").loc[:t, feats])


def test_gap_blanks_windows_and_label():
    stocks, vix = synth(("AAA",))
    gap_day = pd.Timestamp("2019-11-14")
    stocks = stocks[stocks["date"] != gap_day]
    panel, _ = build_panel(stocks, vix)
    dates = set(panel["date"])
    day_before = CAL[CAL.get_loc(gap_day) - 1]
    assert gap_day not in dates
    assert day_before not in dates                       # its label day is missing
    # the next 22 calendar days contain the gap, so har_22 is undefined there
    window = CAL[CAL.get_loc(gap_day):CAL.get_loc(gap_day) + 22]
    assert not dates & set(window)
    assert CAL[CAL.get_loc(gap_day) + 22] in dates


def test_split_is_by_label_date():
    s = assign_split(pd.Series(pd.to_datetime(
        ["2019-12-31", "2020-01-02", "2021-12-31", "2022-01-03", "2009-12-31"])))
    assert list(s.iloc[:4]) == ["train", "val", "val", "test"]
    assert pd.isna(s.iloc[4])
    panel, _ = build_panel(*synth())
    row = panel[(panel["date"] == "2019-12-31") & (panel["ticker"] == "AAA")]
    assert row["split"].iloc[0] == "val"                 # features end 2019, label is 2020
