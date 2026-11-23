import numpy as np
import pandas as pd
import pytest

import baselines as bl
import trees
from test_baselines import synth

SMALL_GRID = [dict(num_leaves=7, learning_rate=0.1, min_child_samples=100),
              dict(num_leaves=31, learning_rate=0.1, min_child_samples=100)]


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(trees, "MAX_ROUNDS", 150)
    monkeypatch.setattr(trees, "PATIENCE", 15)


def make(seed=0, neighbor_signal=0.5, nonlinear=0.0):
    """synth() plus the tree-only inputs; optional nonlinear term in y that HAR cannot fit."""
    df = synth(seed=seed, n_tickers=25, neighbor_signal=neighbor_signal)
    rng = np.random.default_rng(seed + 100)
    df["ret"] = rng.normal(0, 0.01, len(df))
    df["dlogvol"] = rng.normal(0, 0.3, len(df))
    df["vix"] = rng.uniform(10, 40, len(df))
    df["y"] = df["y"] + nonlinear * ((df["vix"] > 25) & (df["har_1"] > -4.5)).astype(float)
    return df


def run(df, **kw):
    return trees.run(bl.common_sample(df), grid=SMALL_GRID, **kw)


def test_trees_beat_har_on_a_nonlinear_target():
    out = run(make(nonlinear=1.0))
    m = out["metrics"].query("split == 'val'").set_index("model")
    assert m.loc["gbt", "mse_logvol"] < 0.8 * m.loc["har", "mse_logvol"]
    dm = out["dm"].query("split == 'val'").set_index(["model", "vs"])
    assert dm.loc[("gbt", "har"), "model_better"] and dm.loc[("gbt", "har"), "p_value"] < 1e-6


def test_neighbor_trees_win_only_when_neighbors_carry_signal():
    with_sig = run(make(neighbor_signal=0.6)).get("dm").query("split == 'val'").set_index(["model", "vs"])
    assert with_sig.loc[("gbt_sector", "gbt"), "model_better"]
    assert with_sig.loc[("gbt_sector", "gbt"), "p_value"] < 1e-6
    no_sig = run(make(neighbor_signal=0.0, seed=3)).get("dm").query("split == 'val'").set_index(["model", "vs"])
    assert no_sig.loc[("gbt_sector", "gbt"), "p_value"] > 0.01


def test_test_years_are_not_touched_by_default():
    df = make()
    sab = df.copy()
    sab.loc[sab["split"] == "test", ["y", "har_1", "vix", "cor_har_1"]] = np.nan
    a, b = run(df), run(sab)                                     # would crash if test were used
    assert set(a["metrics"]["split"]) == {"train", "val"}
    pd.testing.assert_frame_equal(a["metrics"], b["metrics"])


def test_test_labels_never_influence_fit_or_selection_even_when_scored():
    df = make()
    wrecked = df.copy()
    wrecked.loc[wrecked["split"] == "test", "y"] += 5.0
    a, b = run(df, include_test=True), run(wrecked, include_test=True)
    assert a["params"] == b["params"]
    keep = (a["df"]["split"] != "test").to_numpy()
    for name in trees.GBT_MODELS:
        np.testing.assert_array_equal(a["yhat"][name][keep], b["yhat"][name][keep])
    assert a["smear"] == b["smear"]


def test_val_labels_are_used_only_through_selection_not_smearing():
    """Smearing is estimated on train; wrecking val labels may change the chosen
    model but the smearing factor must be computed from train residuals only."""
    df = make()
    out = run(df)
    train = (out["df"]["split"] == "train").to_numpy()
    from metrics import smearing_factor
    for name in trees.GBT_MODELS:
        assert out["smear"][name] == pytest.approx(
            smearing_factor(out["df"]["y"].to_numpy()[train], out["yhat"][name][train]))


def test_deterministic():
    df = make()
    a, b = run(df), run(df)
    for name in trees.GBT_MODELS:
        np.testing.assert_array_equal(a["yhat"][name], b["yhat"][name])


def test_selection_picks_lowest_validation_mse():
    df = bl.common_sample(make())
    tr, va = df[df["split"] == "train"], df[df["split"] == "val"]
    cols = trees.GBT_MODELS["gbt"]
    scores = [trees.fit_one(tr[cols], tr["y"], va[cols], va["y"], p)[0].best_score["val"]["l2"]
              for p in SMALL_GRID]
    _, chosen, curve = trees.tune(df, cols, SMALL_GRID)
    best = SMALL_GRID[int(np.argmin(scores))]
    assert {k: chosen[k] for k in best} == best
    # the curve's validation minimum is at (or within patience of) the chosen iteration
    assert abs(int(curve["val_mse"].idxmin()) + 1 - chosen["best_iteration"]) <= 1
    assert {"round", "train_mse", "val_mse"} <= set(curve.columns)


def test_training_loss_keeps_falling_while_validation_stops():
    df = bl.common_sample(make(nonlinear=1.0))
    _, _, curve = trees.tune(df, trees.GBT_MODELS["gbt"], SMALL_GRID[1:])
    assert curve["train_mse"].iloc[-1] < curve["train_mse"].iloc[0]


def test_every_model_scored_on_identical_rows():
    out = run(make())
    assert (out["metrics"].groupby("split")["n"].nunique() == 1).all()
    assert {"naive", "har", "gbt", "gbt_sector", "gbt_corr"} <= set(out["metrics"]["model"])
