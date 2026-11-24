import numpy as np
import pandas as pd

import disagreement as dg


def fake_predictions(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2022-01-03", periods=60)
    rows = []
    for t in range(20):
        y = rng.normal(-4, 0.3, len(dates))
        rows.append(pd.DataFrame({"ticker": f"T{t}", "date": dates, "y_date": dates + pd.offsets.BDay(1),
                                  "split": "test", "y": y}))
    base = pd.concat(rows, ignore_index=True)
    own = base.assign(yhat=base["y"] + rng.normal(0, 0.3, len(base)), smear=np.exp(0.1))
    peer = base.assign(yhat=base["y"] + rng.normal(0, 0.1, len(base)), smear=1.0)   # peers: closer forecasts
    monkeypatch.setattr(dg, "PREDICTIONS_DIR", tmp_path)
    own.to_parquet(tmp_path / "har_with_test.parquet")
    peer.to_parquet(tmp_path / "har_sector_with_test.parquet")
    return base


def test_smearing_is_applied_and_disagreement_is_peer_minus_own(tmp_path, monkeypatch):
    base = fake_predictions(tmp_path, monkeypatch)
    j = dg.pair("test")
    own = pd.read_parquet(tmp_path / "har_with_test.parquet")
    np.testing.assert_allclose(j["f_own"].to_numpy(), own["yhat"].to_numpy() + 0.05)
    np.testing.assert_allclose(j["disagreement"], j["f_peer"] - j["f_own"])
    assert len(j) == len(base)


def test_deciles_cover_every_row_and_credit_the_better_model(tmp_path, monkeypatch):
    fake_predictions(tmp_path, monkeypatch)
    j = dg.pair("test")
    d = dg.deciles(j)
    assert d["rows"].sum() == len(j) and len(d) == 10
    assert (d["peer_closer_share"] > 0.5).mean() >= 0.8          # the planted better model wins in most bins
