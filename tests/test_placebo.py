import numpy as np
import pandas as pd

from placebo import market_edges, random_corr_edges, random_sector_edges


def sec_edges():
    s = pd.DataFrame({"ticker": list("ABCDEF"), "sector": ["x", "x", "x", "y", "y", "y"]})
    from graphs import sector_edges
    return sector_edges(s)


def cor_edges():
    rows = []
    for m in pd.to_datetime(["2019-01-01", "2019-02-01"]):
        for t, ns in {"A": "BC", "B": "AC", "C": "AB"}.items():
            rows += [(m, t, n, 0.9, i + 1) for i, n in enumerate(ns)]
    rows += [(pd.Timestamp("2019-02-01"), "D", "E", 0.5, 1)]
    return pd.DataFrame(rows, columns=["month", "ticker", "neighbor", "corr", "rank"])


def test_market_edges_link_every_ordered_pair():
    e = market_edges(list("ABCD"))
    assert len(e) == 12 and not (e["ticker"] == e["neighbor"]).any()


def test_random_sector_twin_keeps_degrees_not_identity():
    real = sec_edges()
    twin = random_sector_edges(real, seed=0)
    assert (twin.groupby("ticker").size() == real.groupby("ticker").size()).all()
    assert not (twin["ticker"] == twin["neighbor"]).any()
    assert twin.duplicated().sum() == 0
    # with 6 stocks and 2 peers each, some seed must pick someone outside the sector
    assert any(set(random_sector_edges(real, s).neighbor[random_sector_edges(real, s).ticker == "A"]) != {"B", "C"}
               for s in range(10))


def test_random_corr_twin_keeps_months_degrees_and_candidate_pool():
    real = cor_edges()
    twin = random_corr_edges(real, seed=1)
    key = ["month", "ticker"]
    assert (twin.groupby(key).size() == real.groupby(key).size()).all()
    assert not (twin["ticker"] == twin["neighbor"]).any()
    for month, g in real.groupby("month"):
        pool = set(g["ticker"]) | set(g["neighbor"])
        assert set(twin[twin["month"] == month]["neighbor"]) <= pool       # only stocks with a graph that month


def test_twins_are_reproducible_and_depend_on_the_seed():
    real = cor_edges()
    a, b = random_corr_edges(real, 3), random_corr_edges(real, 3)
    pd.testing.assert_frame_equal(a, b)
    assert not any(random_corr_edges(real, s).equals(a) for s in (4, 5, 6, 7))


def test_tree_losses_are_keyed_deterministic_and_reward_a_useful_feature(monkeypatch):
    import placebo_trees as pt
    import trees
    monkeypatch.setattr(trees, "MAX_ROUNDS", 100)
    monkeypatch.setattr(trees, "PATIENCE", 10)
    monkeypatch.setattr(pt, "PARAMS", dict(num_leaves=7, learning_rate=0.1, min_child_samples=50))
    rng = np.random.default_rng(0)
    n = 6000
    df = pd.DataFrame({"ticker": rng.choice(list("ABCDE"), n), "y_date": pd.bdate_range("2016-01-04", periods=n),
                       "split": np.where(np.arange(n) < 4000, "train", "val"),
                       "f": rng.normal(size=n), "noise": rng.normal(size=n)})
    df["y"] = -4.5 + 0.5 * df["f"] + rng.normal(0, 0.3, n)
    a, b = pt.tree_losses(df, ["f"]), pt.tree_losses(df, ["f"])
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 2000 and set(a.columns) == {"ticker", "y_date", "loss"}
    assert pt.tree_losses(df, ["noise"])["loss"].mean() > a["loss"].mean()
