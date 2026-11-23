"""Stage 3: build the modelling panel (features + next-day target) from data/processed.

Output: data/features/panel.parquet, one row per ticker per trading day.

  lv        today's log Garman-Klass volatility, winsorized
  har_1/5/22  log of the mean volatility over the last 1/5/22 trading days (HAR inputs)
  ret       close-to-close log return
  dlogvol   log change in share volume vs the previous day
  vix       VIX close, same day
  y         TARGET: lv on the NEXT trading day
  split     train / val / test, by the date of y

Design rules (each is covered by tests/test_features.py):
  * No lookahead: every feature at day t uses data up to and including day t only.
    y is the only column that looks forward (exactly one trading day).
  * Gaps: each ticker is laid on the full market calendar. Rolling windows need
    every day present, so a missing day blanks the features that window touches,
    and y is NaN unless the next calendar day exists. Those rows are dropped.
  * Split by the date of y, not of the features, so a row whose label falls in
    validation never trains the model (the 2019-12-31 row has a 2020 label).
  * Winsorization bounds come from the training split only.
"""
import json

import numpy as np
import pandas as pd

from config import FEATURES_DIR, HAR_WINDOWS, PROCESSED_DIR, SPLITS, WINSOR_Q

GK_FLOOR = 1e-12   # variance floor so log() is defined; unused on the current snapshot


def garman_klass_logvol(df):
    """Log of the Garman-Klass daily volatility (sqrt of the variance estimate)."""
    var = (0.5 * np.log(df["high"] / df["low"]) ** 2
           - (2 * np.log(2) - 1) * np.log(df["close"] / df["open"]) ** 2)
    return 0.5 * np.log(var.clip(lower=GK_FLOOR))


def assign_split(dates):
    out = pd.Series(pd.NA, index=dates.index, dtype="object")
    for name, (a, b) in SPLITS.items():
        out[(dates >= a) & (dates <= b)] = name
    return out


def winsor_bounds(lv, dates):
    train = lv[(dates >= SPLITS["train"][0]) & (dates <= SPLITS["train"][1])]
    return tuple(float(train.quantile(q)) for q in WINSOR_Q)


def _ticker_features(g, calendar, lo, hi):
    """Features for one ticker, laid on the market calendar (missing days = NaN)."""
    g = g.set_index("date").reindex(calendar[(calendar >= g["date"].min())
                                             & (calendar <= g["date"].max())])
    lv = g["lv"].clip(lo, hi)
    vol = np.exp(lv)
    out = pd.DataFrame({"lv": lv}, index=g.index)
    for w in HAR_WINDOWS:
        out[f"har_{w}"] = np.log(vol.rolling(w, min_periods=w).mean())
    out["ret"] = np.log(g["close"] / g["close"].shift(1))
    out["dlogvol"] = np.log(g["volume"] / g["volume"].shift(1))
    out["y"] = lv.shift(-1)
    out["y_date"] = pd.Series(g.index, index=g.index).shift(-1)
    return out


def compute_features(stocks, vix):
    """Features for every ticker-day, before any label-based filtering.

    Returns (rows, bounds). Rows still containing NaN (warm-up, gaps, last day) are kept
    so that other stages (graphs) can use a ticker's har_* values on days where tomorrow's
    label is unknown or missing, without letting label availability decide who is visible.
    """
    calendar = pd.DatetimeIndex(sorted(vix["date"]))
    stocks = stocks.copy()
    stocks["lv"] = garman_klass_logvol(stocks)
    lo, hi = winsor_bounds(stocks["lv"], stocks["date"])

    parts = []
    for tkr, g in stocks.groupby("ticker", sort=True):
        f = _ticker_features(g, calendar, lo, hi)
        f["ticker"] = tkr
        parts.append(f.rename_axis("date").reset_index())
    rows = pd.concat(parts, ignore_index=True)
    rows = rows.merge(vix.rename(columns={"vix_close": "vix"}), on="date", how="left")
    rows = rows.merge(stocks[["ticker", "sector"]].drop_duplicates(), on="ticker")
    return rows, {"winsor_lo": lo, "winsor_hi": hi, "quantiles": list(WINSOR_Q)}


def build_panel(stocks, vix):
    """stocks: processed stocks table; vix: processed vix table. Returns (panel, bounds)."""
    rows, bounds = compute_features(stocks, vix)
    # a row needs all features and a label; the label is the next calendar day
    # (guaranteed by reindexing: y_date is NaT across a gap because lv is NaN there)
    panel = rows.dropna(subset=["lv", "har_22", "ret", "dlogvol", "y"]).copy()
    panel["split"] = assign_split(panel["y_date"])
    panel = panel.dropna(subset=["split"])
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    cols = ["date", "ticker", "sector", "split", "y_date", "lv", "har_1", "har_5", "har_22",
            "ret", "dlogvol", "vix", "y"]
    return panel[cols], bounds


def main():
    stocks = pd.read_parquet(PROCESSED_DIR / "stocks.parquet")
    vix = pd.read_parquet(PROCESSED_DIR / "vix.parquet")
    panel, bounds = build_panel(stocks, vix)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(FEATURES_DIR / "panel.parquet", index=False)
    (FEATURES_DIR / "winsor_bounds.json").write_text(json.dumps(bounds, indent=2))

    print(f"Panel: {panel['ticker'].nunique()} tickers, {len(panel):,} rows "
          f"(from {len(stocks):,} cleaned rows)")
    print(f"Winsor bounds on log-vol (train only): {bounds['winsor_lo']:.3f}, {bounds['winsor_hi']:.3f}")
    print(panel.groupby("split").agg(rows=("y", "size"), stocks=("ticker", "nunique"),
                                     first=("y_date", "min"), last=("y_date", "max")).to_string())
    print("\nNaN count per column:", int(panel.isna().sum().sum()))


if __name__ == "__main__":
    main()
