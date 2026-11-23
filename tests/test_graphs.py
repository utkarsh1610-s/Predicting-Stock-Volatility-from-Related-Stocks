import numpy as np
import pandas as pd
import pytest

from graphs import corr_edges, neighbor_features, return_matrix, sector_edges

CAL = pd.bdate_range("2019-01-01", "2019-08-30")
TICKERS = [f"T{i}" for i in range(12)]


def rand_returns(seed=0, tickers=TICKERS, cal=CAL):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0, 0.01, (len(cal), len(tickers))), index=cal, columns=tickers)


def test_sector_edges_same_sector_no_self_symmetric():
    stocks = pd.DataFrame({"ticker": list("ABCDE"), "sector": ["x", "x", "x", "y", "y"]})
    e = sector_edges(stocks)
    assert not (e["ticker"] == e["neighbor"]).any()
    pairs = set(zip(e["ticker"], e["neighbor"]))
    assert pairs == {(a, b) for a, b in [("A", "B"), ("A", "C"), ("B", "A"), ("B", "C"),
                                         ("C", "A"), ("C", "B"), ("D", "E"), ("E", "D")]}


def test_corr_edges_picks_the_correlated_twin_and_respects_k():
    r = rand_returns()
    r["T1"] = r["T0"] + np.random.default_rng(1).normal(0, 1e-4, len(r))   # near copy
    e = corr_edges(r, k=3)
    assert not (e["ticker"] == e["neighbor"]).any()
    assert e.groupby(["month", "ticker"]).size().max() == 3
    top = e[(e["ticker"] == "T0") & (e["rank"] == 1)]
    assert (top["neighbor"] == "T1").all() and len(top) > 0
    assert (e[e["rank"] == 1]["corr"] >= e[e["rank"] == 2]["corr"].min()).all()


def test_corr_graph_for_a_month_uses_only_earlier_days():
    r = rand_returns()
    base = corr_edges(r)
    month = pd.Timestamp("2019-06-01")
    d0 = r.index[r.index >= month][0]

    # wreck every day from the month's first trading day onward
    wrecked = r.copy()
    wrecked.loc[wrecked.index >= d0] = np.random.default_rng(9).normal(0, 0.05, wrecked[wrecked.index >= d0].shape)
    alt = corr_edges(wrecked)
    key = ["month", "ticker", "neighbor"]
    same = lambda df: df[df["month"] == month].sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(same(base), same(alt))

    # control: changing the day just before d0 does change June's graph
    prev = r.index[r.index.get_loc(d0) - 1]
    nudged = r.copy()
    nudged.loc[prev] = np.random.default_rng(3).normal(0, 0.05, r.shape[1])
    assert not same(base).equals(same(corr_edges(nudged)))


def test_min_common_days_and_pairwise_complete():
    r = rand_returns()
    d0 = pd.Timestamp("2019-06-03")
    pos = r.index.get_loc(d0)
    window = r.index[pos - 60:pos]
    r.loc[window[:31], "T5"] = np.nan          # only 29 days left in window: too few
    r.loc[window[:10], "T6"] = np.nan          # 50 days left: fine
    e = corr_edges(r)
    june = e[e["month"] == pd.Timestamp("2019-06-01")]
    assert "T5" not in set(june["ticker"]) | set(june["neighbor"])
    assert "T6" in set(june["ticker"])


def test_no_graph_when_history_too_short():
    r = rand_returns(cal=pd.bdate_range("2019-01-01", "2019-02-28"))
    assert corr_edges(r).empty


def _rows(vals):
    """vals: dict ticker -> list of har_1 values over CAL_SHORT (also used for har_5/22)."""
    cal = pd.bdate_range("2019-05-29", periods=len(next(iter(vals.values()))))
    recs = []
    for t, v in vals.items():
        for d, x in zip(cal, v):
            recs.append({"date": d, "ticker": t, "sector": "s", "har_1": x, "har_5": x + 10, "har_22": x + 20})
    return pd.DataFrame(recs)


def test_neighbor_mean_excludes_self_and_ignores_missing():
    rows = _rows({"A": [1.0, 1.0, 1.0], "B": [3.0, np.nan, 3.0], "C": [5.0, 5.0, 5.0]})
    sec = pd.DataFrame({"ticker": ["A", "A", "B", "B", "C", "C"],
                        "neighbor": ["B", "C", "A", "C", "A", "B"]})
    cor = pd.DataFrame({"month": pd.to_datetime(["2019-05-01"] * 2),
                        "ticker": ["A", "A"], "neighbor": ["B", "C"]})
    out = neighbor_features(rows, sec, cor).set_index(["ticker", "date"])
    d = rows["date"].unique()
    # A's sector peers are B and C: mean of (3, 5) = 4 on day 0; B missing on day 1 -> C only
    assert out.loc[("A", d[0]), "sec_har_1"] == 4.0 and out.loc[("A", d[0]), "sec_n"] == 2
    assert out.loc[("A", d[1]), "sec_har_1"] == 5.0 and out.loc[("A", d[1]), "sec_n"] == 1
    assert out.loc[("A", d[0]), "sec_har_22"] == 4.0 + 20
    # never includes the stock itself
    assert out.loc[("C", d[0]), "sec_har_1"] == 2.0
    # correlation neighbors of A in May are {B, C}; B has no value on d[1]
    assert out.loc[("A", d[0]), "cor_har_1"] == 4.0
    assert out.loc[("A", d[1]), "cor_n"] == 1


def test_months_without_a_graph_give_nan_and_zero_neighbors():
    rows = _rows({"A": [1.0, 1.0], "B": [2.0, 2.0]})
    sec = pd.DataFrame({"ticker": ["A", "B"], "neighbor": ["B", "A"]})
    cor = pd.DataFrame({"month": pd.to_datetime(["2019-04-01"]), "ticker": ["A"], "neighbor": ["B"]})
    out = neighbor_features(rows, sec, cor)
    assert out["cor_har_1"].isna().all() and (out["cor_n"] == 0).all()


def test_neighbor_value_is_same_day_not_future():
    rows = _rows({"A": [1.0, 1.0, 1.0], "B": [3.0, 100.0, 100.0]})
    sec = pd.DataFrame({"ticker": ["A", "B"], "neighbor": ["B", "A"]})
    cor = pd.DataFrame({"month": pd.to_datetime(["2019-05-01"]), "ticker": ["A"], "neighbor": ["B"]})
    out = neighbor_features(rows, sec, cor).set_index(["ticker", "date"])
    d0 = rows["date"].min()
    assert out.loc[("A", d0), "sec_har_1"] == 3.0     # B's day-0 value, not its later 100s


def test_return_matrix_blanks_across_gaps():
    cal = pd.bdate_range("2019-01-01", periods=5)
    stocks = pd.DataFrame({"date": cal.delete(2), "ticker": "A", "close": [1.0, 2.0, 3.0, 4.0]})
    r = return_matrix(stocks, cal)["A"]
    assert r.isna().tolist() == [True, False, True, True, False]
