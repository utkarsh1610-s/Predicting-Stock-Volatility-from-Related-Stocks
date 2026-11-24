"""Stage 6: gradient-boosted trees (LightGBM), with and without neighbor features.

  python scripts/trees.py                 # tune on validation, report train + validation
  python scripts/trees.py --include-test  # ALSO score the test years (once, at the end)

Models forecast next-day log volatility y, on the same rows as the baselines:
  gbt          har_1/5/22, ret, dlogvol, vix
  gbt_sector   + sec_har_1/5/22
  gbt_corr     + cor_har_1/5/22

Tuning: each model gets the same small grid (leaves, learning rate, min leaf size). For every
config, trees are added until validation MSE stops improving (early stopping, patience 100);
the config with the lowest validation MSE wins. So, unlike the baselines, VALIDATION SCORES
HERE ARE TUNED ON VALIDATION and mildly optimistic; the test years are the clean number.
Trees never see validation labels in their fit, only through that selection.

Outputs: results/trees_{metrics,dm,params}.csv, results/trees_curves.csv (train vs validation
loss per boosting round, for the over/underfitting plots), data/predictions/gbt*.parquet.
The metrics and DM tables include the baselines so the comparison is in one place.
"""
import argparse
import itertools

import lightgbm as lgb
import numpy as np
import pandas as pd

import baselines as bl
from config import PREDICTIONS_DIR, RESULTS_DIR
from metrics import smearing_factor

BASE = ["har_1", "har_5", "har_22", "ret", "dlogvol", "vix"]
GBT_MODELS = {"gbt": BASE, "gbt_sector": BASE + bl.SEC, "gbt_corr": BASE + bl.COR}
GRID = [dict(num_leaves=nl, learning_rate=lr, min_child_samples=mcs)
        for nl, lr, mcs in itertools.product([7, 31, 127], [0.03, 0.1], [500, 5000])]
MAX_ROUNDS, PATIENCE, SEED = 3000, 100, 0
PAIRS = bl.PAIRS + [("gbt", "har"), ("gbt_sector", "gbt"), ("gbt_corr", "gbt"),
                    ("gbt_corr", "gbt_sector"), ("gbt_sector", "har_sector"),
                    ("gbt_corr", "har_corr")]


def fit_one(xtr, ytr, xva, yva, params):
    """Train with early stopping on validation. Returns (booster, history)."""
    cfg = dict(objective="l2", metric="l2", verbose=-1, seed=SEED, deterministic=True,
               force_row_wise=True, num_threads=8, **params)
    dtr = lgb.Dataset(xtr, ytr)
    dva = lgb.Dataset(xva, yva, reference=dtr)
    hist = {}
    booster = lgb.train(cfg, dtr, MAX_ROUNDS, valid_sets=[dtr, dva], valid_names=["train", "val"],
                        callbacks=[lgb.early_stopping(PATIENCE, verbose=False),
                                   lgb.record_evaluation(hist)])
    return booster, hist


def tune(df, cols, grid=GRID):
    """Pick the grid config with the lowest validation MSE. Returns (booster, params, curve)."""
    tr, va = df[df["split"] == "train"], df[df["split"] == "val"]
    best = None
    for params in grid:
        booster, hist = fit_one(tr[cols], tr["y"], va[cols], va["y"], params)
        score = booster.best_score["val"]["l2"]
        if best is None or score < best[0]:
            best = (score, booster, params, hist)
    _, booster, params, hist = best
    n = booster.best_iteration
    curve = pd.DataFrame({"round": np.arange(1, len(hist["val"]["l2"]) + 1),
                          "train_mse": hist["train"]["l2"], "val_mse": hist["val"]["l2"]})
    return booster, {**params, "best_iteration": n}, curve


def fit_trees(df, models=GBT_MODELS, grid=GRID):
    """Tune and fit every model. df: common sample restricted to the splits in use."""
    yhat, smear, params, curves = {}, {}, {}, []
    train = (df["split"] == "train").to_numpy()
    for name, cols in models.items():
        booster, p, curve = tune(df, cols, grid)
        yhat[name] = booster.predict(df[cols], num_iteration=p["best_iteration"])
        smear[name] = smearing_factor(df["y"].to_numpy()[train], yhat[name][train])
        params[name] = p
        curves.append(curve.assign(model=name))
    return yhat, smear, params, pd.concat(curves, ignore_index=True)


def run(df, include_test=False, models=GBT_MODELS, grid=GRID):
    base = bl.run(df, include_test)               # also restricts df to the splits in use
    df = base["df"]
    yhat, smear, params, curves = fit_trees(df, models, grid)
    yhat_all, smear_all = {**base["yhat"], **yhat}, {**base["smear"], **smear}
    pairs = [(a, b) for a, b in PAIRS if a in yhat_all and b in yhat_all]
    eval_splits = [s for s in base["splits"] if s != "train"]
    return {"df": df, "yhat": yhat_all, "smear": smear_all, "params": params, "curves": curves,
            "splits": base["splits"],
            "metrics": bl.score(df, yhat_all, smear_all, base["splits"]),
            "dm": bl.dm_table(df, yhat_all, smear_all, eval_splits, pairs)}


def main(include_test):
    raw = bl.load()
    res = run(bl.common_sample(raw), include_test)
    df, metrics, dm = res["df"], res["metrics"], res["dm"]

    RESULTS_DIR.mkdir(exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    tag = "_with_test" if include_test else ""
    metrics.to_csv(RESULTS_DIR / f"trees_metrics{tag}.csv", index=False, float_format=bl.CSV_FMT)
    dm.to_csv(RESULTS_DIR / f"trees_dm{tag}.csv", index=False, float_format=bl.CSV_FMT)
    pd.DataFrame(res["params"]).T.rename_axis("model").to_csv(RESULTS_DIR / f"trees_params{tag}.csv")
    res["curves"].to_csv(RESULTS_DIR / f"trees_curves{tag}.csv", index=False, float_format=bl.CSV_FMT)
    for name in GBT_MODELS:
        out = df[["ticker", "date", "y_date", "split", "y"]].copy()
        out["yhat"], out["smear"] = res["yhat"][name], res["smear"][name]
        out.to_parquet(PREDICTIONS_DIR / f"{name}{tag}.parquet", index=False)

    print(f"Rows: {len(df):,} ({', '.join(res['splits'])}); grid of {len(GRID)} configs per model")
    print("\nChosen hyper-parameters:")
    print(pd.DataFrame(res["params"]).T.to_string())
    print("\nSmearing factors:", {k: round(v, 3) for k, v in res["smear"].items() if k != "naive"})
    show = metrics.copy()
    base = show[show["model"] == "har"].set_index("split")
    show["qlike_vs_har"] = show.apply(lambda r: f"{r.qlike / base.loc[r.split, 'qlike'] - 1:+.2%}", axis=1)
    show["mse_vs_har"] = show.apply(lambda r: f"{r.mse_logvol / base.loc[r.split, 'mse_logvol'] - 1:+.2%}", axis=1)
    print("\n" + show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nDiebold-Mariano on daily mean QLIKE (negative diff = `model` has lower loss):")
    print(dm.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-test", action="store_true")
    main(ap.parse_args().include_test)
