"""Stage 2: clean the raw price CSVs and combine them into analysis-ready tables.

Outputs (data/processed/):
  stocks.parquet       one row per ticker per trading day, with sector labels
  vix.parquet          VIX daily close
  cleaning_log.csv     per-ticker row counts and the reason for every drop
  wide_range_days.csv  kept rows with high/low > 1.5, for manual review

Cleaning is limited to data validity. Modeling choices (winsorizing the
volatility target, filling small gaps) belong in the feature step.
"""
import pandas as pd

from config import (CONSTITUENTS, LISTING_STARTS, MIN_ROWS, PROCESSED_DIR, PX, VIX,
                    price_path)

WIDE_RANGE = 1.5   # high/low ratio worth a manual look


def clean_ohlc(df, tol=1e-6, snap_tol=1e-4):
    """Snap tiny float noise; drop invalid rows. Returns (clean_df, per-rule counts)."""
    df = df.copy()
    has_nan = df.isna().any(axis=1)     # before snapping: max/min below skip NaN and would mask it
    hi, lo = df[PX].max(axis=1), df[PX].min(axis=1)

    # how far the reported High/Low fall outside the range of the four prices
    excess = pd.concat([hi - df["High"], df["Low"] - lo], axis=1).max(axis=1) / df["Close"]
    df["High"], df["Low"] = hi, lo

    rules = {
        "nonpositive": (df[PX] <= 0).any(axis=1),
        "nan": has_nan,
        "zero_volume": df["Volume"] == 0,
        "zero_range": (df["High"] - df["Low"]) / df["Close"] < tol,   # Garman-Klass would be 0
        "bad_tick": excess > snap_tol,          # violation bigger than float noise
    }
    drop = pd.concat(rules.values(), axis=1).any(axis=1)
    counts = {f"drop_{k}": int(v.sum()) for k, v in rules.items()}
    counts["dropped_total"] = int(drop.sum())
    return df[~drop], counts


def load_vix():
    vix = pd.read_csv(price_path(VIX), index_col=0, parse_dates=True)[["Close"]]
    vix = vix.rename_axis("date").rename(columns={"Close": "vix_close"}).reset_index()
    assert vix["vix_close"].notna().all() and vix["date"].is_unique
    return vix


def build():
    meta = pd.read_csv(CONSTITUENTS)
    vix = load_vix()
    calendar = pd.DatetimeIndex(vix["date"])     # market trading days

    frames, log = [], []
    for _, row in meta.iterrows():
        tkr = row["ticker"]
        p = price_path(tkr)
        if not p.exists():
            log.append({"ticker": tkr, "status": "no_data"})
            continue
        raw = pd.read_csv(p, index_col=0, parse_dates=True)
        entry = {"ticker": tkr, "rows_raw": len(raw)}

        # 1. start at the real US listing where Yahoo stitched in other history
        if tkr in LISTING_STARTS:
            start, _reason = LISTING_STARTS[tkr]
            entry["listing_start"] = start
            entry["drop_pre_listing"] = int((raw.index < start).sum())
            raw = raw[raw.index >= start]

        # 2. row-level validity rules
        clean, counts = clean_ohlc(raw)
        entry.update(counts)
        entry["rows_clean"] = len(clean)
        entry["first_date"] = clean.index.min().date()
        entry["last_date"] = clean.index.max().date()
        span = calendar[(calendar >= clean.index.min()) & (calendar <= clean.index.max())]
        entry["missing_trading_days"] = len(span) - len(clean)
        entry["wide_range_days"] = int((clean["High"] / clean["Low"] > WIDE_RANGE).sum())

        # 3. minimum history
        if len(clean) < MIN_ROWS:
            entry["status"] = f"dropped_short_history(<{MIN_ROWS})"
            log.append(entry)
            continue
        entry["status"] = "kept"
        log.append(entry)

        clean = clean.rename_axis("date").reset_index()
        clean.columns = [c.lower() for c in clean.columns]
        clean["ticker"] = tkr
        frames.append(clean)

    stocks = pd.concat(frames, ignore_index=True)
    stocks = stocks.merge(meta[["ticker", "company", "sector", "sub_industry"]], on="ticker")
    stocks = stocks.sort_values(["ticker", "date"]).reset_index(drop=True)

    # integrity checks
    assert not stocks.duplicated(["ticker", "date"]).any()
    assert stocks["date"].isin(calendar).all(), "stock dates outside the VIX calendar"
    assert stocks[["sector", "sub_industry"]].notna().all().all()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    stocks.to_parquet(PROCESSED_DIR / "stocks.parquet", index=False)
    vix.to_parquet(PROCESSED_DIR / "vix.parquet", index=False)
    log = pd.DataFrame(log)
    log.to_csv(PROCESSED_DIR / "cleaning_log.csv", index=False)
    wide = stocks[stocks["high"] / stocks["low"] > WIDE_RANGE]
    wide.to_csv(PROCESSED_DIR / "wide_range_days.csv", index=False)

    # summary
    kept = log[log["status"] == "kept"]
    print(f"Stocks: {stocks['ticker'].nunique()} tickers, {len(stocks):,} rows, "
          f"{stocks['date'].min().date()} to {stocks['date'].max().date()}")
    print(f"VIX: {len(vix):,} rows")
    print(f"Rows dropped: {int(log['drop_pre_listing'].sum()):,} pre-listing, "
          f"{int(log['dropped_total'].sum()):,} by validity rules")
    print("\nNot kept:")
    print(log[log["status"] != "kept"][["ticker", "status", "rows_clean"]].to_string(index=False))
    print("\nMost missing trading days among kept tickers:")
    print(kept.nlargest(8, "missing_trading_days")[
        ["ticker", "first_date", "rows_clean", "missing_trading_days"]].to_string(index=False))
    print(f"\nWide-range days kept for review: {len(wide)}")
    for name, (a, b) in {"train": ("2010", "2019"), "val": ("2020", "2021"),
                         "test": ("2022", "2025")}.items():
        sub = stocks[(stocks["date"] >= a) & (stocks["date"] <= f"{b}-12-31")]
        print(f"{name} {a}-{b}: {sub['ticker'].nunique()} stocks")


if __name__ == "__main__":
    build()
