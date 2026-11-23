"""Stage 5: naive and HAR baselines, with and without neighbor features.

  python scripts/baselines.py                 # fit on train, report train + validation
  python scripts/baselines.py --include-test  # ALSO score the test years (do this once, at the end)

Models (all forecast next-day log volatility y):
  naive          yhat = lv                      tomorrow = today
  har            OLS on har_1, har_5, har_22    pooled across stocks
  har_sector     har + sec_har_1/5/22           sector peers' averaged volatility
  har_corr       har + cor_har_1/5/22           correlation neighbors' averaged volatility

Fair-comparison rule: every model is fit and scored on the SAME rows, those where both
neighbor feature sets exist (drops ~1% of rows: early 2010, new listings). Coefficients
are fit on the training split only; nothing here is tuned, so validation is a clean
out-of-sample score and the test years stay untouched unless --include-test.

Metrics: MSE on log volatility, QLIKE (variance forecast = exp(2*yhat) rescaled by the
training-set smearing factor, since a log-model gives the median variance), and
Diebold-Mariano tests on the cross-sectional mean loss per day.
Writes results/baseline_metrics.csv, results/baseline_dm.csv and
data/predictions/<model>.parquet (for the per-group analysis later).
"""
import argparse

import numpy as np
import pandas as pd

from config import FEATURES_DIR, PREDICTIONS_DIR, RESULTS_DIR
from metrics import dm_test, mse_logvol, qlike, qlike_loss, smearing_factor

HAR = ["har_1", "har_5", "har_22"]
SEC = ["sec_har_1", "sec_har_5", "sec_har_22"]
COR = ["cor_har_1", "cor_har_5", "cor_har_22"]
MODELS = {"naive": None, "har": HAR, "har_sector": HAR + SEC, "har_corr": HAR + COR}
CSV_FMT = "%.8g"    # results are committed; 8 digits keeps them identical across machines
# (treatment, control) pairs for the Diebold-Mariano tests
PAIRS = [("har", "naive"), ("har_sector", "har"), ("har_corr", "har"), ("har_corr", "har_sector")]


def load():
    panel = pd.read_parquet(FEATURES_DIR / "panel.parquet")
    nbr = pd.read_parquet(FEATURES_DIR / "neighbors.parquet")
    return panel.merge(nbr, on=["ticker", "date"], how="left", validate="1:1")


def common_sample(df):
    """Rows where every model's inputs exist."""
    return df.dropna(subset=HAR + SEC + COR).reset_index(drop=True)


def fit_ols(x, y):
    a = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    return coef


def predict_ols(coef, x):
    return coef[0] + np.asarray(x) @ coef[1:]


def fit_and_predict(df):
    """Fit every model on the train split. Returns (yhat dict over all rows, smear dict, coefs)."""
    train = (df["split"] == "train").to_numpy()
    yhat, smear, coefs = {}, {}, {}
    for name, cols in MODELS.items():
        if cols is None:
            yhat[name], smear[name] = df["lv"].to_numpy(), 1.0
            continue
        x = df[cols].to_numpy()
        coefs[name] = fit_ols(x[train], df["y"].to_numpy()[train])
        yhat[name] = predict_ols(coefs[name], x)
        smear[name] = smearing_factor(df["y"].to_numpy()[train], yhat[name][train])
    return yhat, smear, coefs


def score(df, yhat, smear, splits):
    rows = []
    for split in splits:
        m = (df["split"] == split).to_numpy()
        y = df["y"].to_numpy()[m]
        for name in MODELS:
            p = yhat[name][m]
            rows.append({"model": name, "split": split, "n": int(m.sum()),
                         "mse_logvol": mse_logvol(y, p),
                         "qlike": qlike(y, p, smear[name]),
                         "qlike_raw": qlike(y, p)})
    return pd.DataFrame(rows)


def daily_loss(df, yhat, smear, name, split):
    """Cross-sectional mean QLIKE per day (one number per day), for DM tests."""
    m = (df["split"] == split).to_numpy()
    loss = qlike_loss(np.exp(2 * df["y"].to_numpy()[m]), np.exp(2 * yhat[name][m]) * smear[name])
    return pd.Series(loss, index=df["y_date"][m]).groupby(level=0).mean()


def dm_table(df, yhat, smear, splits):
    rows = []
    for split in splits:
        for a, b in PAIRS:
            la, lb = daily_loss(df, yhat, smear, a, split), daily_loss(df, yhat, smear, b, split)
            mean_d, stat, p = dm_test(la.to_numpy(), lb.to_numpy())
            rows.append({"split": split, "model": a, "vs": b, "days": len(la),
                         "mean_qlike_diff": mean_d, "dm_stat": stat, "p_value": p,
                         "model_better": mean_d < 0})
    return pd.DataFrame(rows)


def run(df, include_test=False):
    """Fit on train, score the requested splits. `df` must already be the common sample.

    Test rows are dropped before anything is fit or scored unless include_test is set.
    """
    splits = ["train", "val"] + (["test"] if include_test else [])
    df = df[df["split"].isin(splits)].reset_index(drop=True)
    yhat, smear, coefs = fit_and_predict(df)
    return {"df": df, "yhat": yhat, "smear": smear, "coefs": coefs, "splits": splits,
            "metrics": score(df, yhat, smear, splits),
            "dm": dm_table(df, yhat, smear, [s for s in splits if s != "train"])}


def main(include_test):
    raw = load()
    res = run(common_sample(raw), include_test)
    df, yhat, smear, metrics, dm = res["df"], res["yhat"], res["smear"], res["metrics"], res["dm"]

    RESULTS_DIR.mkdir(exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    tag = "_with_test" if include_test else ""
    metrics.to_csv(RESULTS_DIR / f"baseline_metrics{tag}.csv", index=False, float_format=CSV_FMT)
    dm.to_csv(RESULTS_DIR / f"baseline_dm{tag}.csv", index=False, float_format=CSV_FMT)
    for name in MODELS:
        out = df[["ticker", "date", "y_date", "split", "y"]].copy()
        out["yhat"], out["smear"] = yhat[name], smear[name]
        out.to_parquet(PREDICTIONS_DIR / f"{name}{tag}.parquet", index=False)

    print(f"Common sample: {len(df):,} rows ({', '.join(res['splits'])}); "
          f"{len(raw) - len(common_sample(raw)):,} of {len(raw):,} panel rows dropped for missing neighbor features")
    print("\nOLS coefficients (intercept, then inputs in order):")
    for name, c in res["coefs"].items():
        print(f"  {name:11s}" + "  ".join(f"{v:+.3f}" for v in c))
    print("\nSmearing factors:", {k: round(v, 3) for k, v in smear.items() if k != "naive"})
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
