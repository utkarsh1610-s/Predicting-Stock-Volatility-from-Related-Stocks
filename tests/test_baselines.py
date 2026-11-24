import numpy as np
import pandas as pd
import pytest

import baselines as bl


def synth(seed=0, n_tickers=40, neighbor_signal=0.5, noise=0.3):
    """y depends on own HAR and (strongly) on neighbors' har_1, so the neighbor models must win."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2016-01-04", "2018-12-31")
    recs = []
    for t in range(n_tickers):
        har1 = rng.normal(-4.5, 0.5, len(days))
        har5 = 0.7 * har1 + 0.3 * rng.normal(-4.5, 0.4, len(days))
        har22 = 0.5 * har5 + 0.5 * rng.normal(-4.5, 0.3, len(days))
        nb = rng.normal(-4.5, 0.5, len(days))
        y = -0.5 + 0.3 * har1 + 0.3 * har5 + 0.3 * har22 + neighbor_signal * (nb + 4.5) \
            + rng.normal(0, noise, len(days))
        recs.append(pd.DataFrame({
            "ticker": f"T{t}", "date": days, "y_date": days + pd.offsets.BDay(1), "lv": har1,
            "har_1": har1, "har_5": har5, "har_22": har22,
            "sec_har_1": nb, "sec_har_5": nb, "sec_har_22": nb,
            "cor_har_1": nb, "cor_har_5": nb, "cor_har_22": nb, "y": y}))
    df = pd.concat(recs, ignore_index=True)
    df["split"] = np.select([df["y_date"] < "2017-01-01", df["y_date"] < "2018-01-01"],
                            ["train", "val"], "test")
    return df


def test_fit_ols_recovers_coefficients():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5000, 3))
    y = 2.0 + x @ np.array([1.0, -0.5, 0.25]) + rng.normal(0, 0.01, 5000)
    assert bl.fit_ols(x, y) == pytest.approx([2.0, 1.0, -0.5, 0.25], abs=0.01)


def test_common_sample_drops_rows_missing_any_neighbor_feature():
    df = synth(n_tickers=2)
    df.loc[0, "cor_har_5"] = np.nan
    df.loc[1, "sec_har_22"] = np.nan
    assert len(bl.common_sample(df)) == len(df) - 2


def test_naive_forecast_is_todays_log_vol():
    out = bl.run(synth())
    assert (out["yhat"]["naive"] == out["df"]["lv"].to_numpy()).all()


def test_fit_uses_train_rows_only():
    df = synth()
    base = bl.run(df)
    wrecked = df.copy()
    wrecked.loc[wrecked["split"] != "train", "y"] += 3.0         # val/test labels destroyed
    alt = bl.run(wrecked)
    for k in base["coefs"]:
        assert base["coefs"][k] == pytest.approx(alt["coefs"][k])
    assert base["smear"] == pytest.approx(alt["smear"])          # smearing is fit on train too


def test_test_split_is_untouched_unless_requested():
    df = synth()
    sabotaged = df.copy()
    sabotaged.loc[sabotaged["split"] == "test", ["y", "har_1", "cor_har_1"]] = np.nan
    a = bl.run(df)
    b = bl.run(sabotaged)                                        # would blow up if test were scored
    assert set(a["metrics"]["split"]) == {"train", "val"}
    pd.testing.assert_frame_equal(a["metrics"], b["metrics"])
    assert "test" in set(bl.run(df, include_test=True)["metrics"]["split"])


def test_neighbor_models_win_when_neighbors_carry_signal_and_dm_agrees():
    out = bl.run(synth(neighbor_signal=0.6))
    m = out["metrics"].query("split == 'val'").set_index("model")
    assert m.loc["naive", "qlike"] > m.loc["har", "qlike"] > m.loc["har_sector", "qlike"]
    assert m.loc["har_corr", "mse_logvol"] < m.loc["har", "mse_logvol"]
    dm = out["dm"].query("split == 'val'").set_index(["model", "vs"])
    assert dm.loc[("har_sector", "har"), "model_better"] and dm.loc[("har_sector", "har"), "p_value"] < 1e-6
    assert dm.loc[("har", "naive"), "model_better"]


def test_neighbors_do_not_win_when_they_carry_no_signal():
    out = bl.run(synth(neighbor_signal=0.0, seed=3))
    dm = out["dm"].query("split == 'val'").set_index(["model", "vs"])
    assert dm.loc[("har_sector", "har"), "p_value"] > 0.01       # no real difference to find


def test_every_model_is_scored_on_identical_rows():
    out = bl.run(synth())
    n = out["metrics"].groupby("split")["n"].nunique()
    assert (n == 1).all()
