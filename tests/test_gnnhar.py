import numpy as np
import pandas as pd
import torch

import gnnhar
from metrics import qlike


def synthetic(n_stocks=40, n_days=700, effect=0.0, seed=0):
    """Panel with own har_*, neighbor means sec_har_*, and a label that depends on the own
    features linearly and, if `effect` > 0, on the neighbor mean through a kink."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2017-01-02", periods=n_days)
    rows = []
    for t in range(n_stocks):
        base = rng.normal(-4.0, 0.3)
        own = base + 0.3 * rng.standard_normal((n_days, 3))
        msg = -4.0 + 0.3 * rng.standard_normal((n_days, 3))
        y = base + 0.5 * (own[:, 0] - base) + effect * np.maximum(msg[:, 0] + 4.0, 0) + 0.2 * rng.standard_normal(n_days)
        rows.append(pd.DataFrame({"ticker": f"T{t}", "date": dates, "y": y,
                                  "har_1": own[:, 0], "har_5": own[:, 1], "har_22": own[:, 2],
                                  "sec_har_1": msg[:, 0], "sec_har_5": msg[:, 1], "sec_har_22": msg[:, 2]}))
    df = pd.concat(rows, ignore_index=True)
    df["y_date"] = df["date"] + pd.offsets.BDay(1)
    df["split"] = np.where(df["y_date"] < "2019-07-01", "train", "val")
    return df


def test_objective_is_qlike_up_to_a_constant():
    rng = np.random.default_rng(1)
    y = rng.normal(-4, 0.5, 1000)
    z1, z2 = y + rng.normal(0, 0.3, 1000), y + rng.normal(0.1, 0.4, 1000)
    obj = lambda z: float(gnnhar.qlike_objective(torch.tensor(y), torch.tensor(z)))
    const = float(np.mean(2 * y + 1))
    assert np.isclose(obj(z1) - const, qlike(y, z1), atol=1e-9)
    assert np.isclose(obj(z2) - obj(z1), qlike(y, z2) - qlike(y, z1), atol=1e-9)


def test_network_starts_at_the_linear_fit():
    init = np.array([0.5, 1.0, -2.0, 0.25])
    model, _ = gnnhar._fit(np.zeros((10, 6)), np.zeros(10), init, seed=0, epochs=0)
    x = torch.randn(50, 6)
    linear = 0.5 + x[:, :3] @ torch.tensor([1.0, -2.0, 0.25])
    assert torch.allclose(model(x), linear, atol=0.05)     # hidden layer starts near zero


def test_validation_labels_cannot_change_the_fit():
    df = synthetic(n_stocks=10, n_days=700)
    z1, e1 = gnnhar.fit_network(df, ["sec_har_1", "sec_har_5", "sec_har_22"], seed=0)
    df2 = df.copy()
    val = df2["split"] == "val"
    df2.loc[val, "y"] = df2.loc[val, "y"] + 5.0                    # scramble every validation label
    df2.loc[val, ["sec_har_1", "har_1"]] *= 3.0                    # and validation inputs' scale
    z2, e2 = gnnhar.fit_network(df2, ["sec_har_1", "sec_har_5", "sec_har_22"], seed=0)
    train = (df["split"] == "train").to_numpy()
    assert e1 == e2
    np.testing.assert_allclose(z1[train], z2[train], atol=1e-5)


def test_finds_a_planted_neighbor_effect_and_not_an_absent_one():
    msg = ["sec_har_1", "sec_har_5", "sec_har_22"]
    for effect, should_help in [(1.0, True), (0.0, False)]:
        df = synthetic(effect=effect, seed=2)
        val = (df["split"] == "val").to_numpy()
        z_own, _ = gnnhar.fit_network(df, [], seed=0)
        z_gnn, _ = gnnhar.fit_network(df, msg, seed=0)
        gain = qlike(df["y"][val], z_gnn[val]) / qlike(df["y"][val], z_own[val]) - 1
        if should_help:
            assert gain < -0.05, gain
        else:
            assert abs(gain) < 0.02, gain


def test_early_stopping_uses_only_the_last_training_year(monkeypatch):
    df = synthetic(n_stocks=5, n_days=700)
    seen = {}
    real = gnnhar._fit

    def spy(x, y, init, seed, epochs, x_stop=None, y_stop=None):
        if x_stop is not None:
            seen["fit"], seen["stop"] = len(x), len(x_stop)
        return real(x, y, init, seed, epochs, x_stop, y_stop)

    monkeypatch.setattr(gnnhar, "_fit", spy)
    gnnhar.fit_network(df, [], seed=0)
    train = df["split"] == "train"
    assert seen["stop"] == int((train & (df["y_date"] >= gnnhar.ES_START)).sum())
    assert seen["fit"] == int((train & (df["y_date"] < gnnhar.ES_START)).sum())


def test_missing_early_stopping_year_is_an_error():
    import pytest
    df = synthetic(n_stocks=3, n_days=400)          # ends in 2018: no 2019 rows to stop on
    with pytest.raises(ValueError, match="early-stopping"):
        gnnhar.fit_network(df, [], seed=0)
