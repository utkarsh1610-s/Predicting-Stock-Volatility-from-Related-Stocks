"""Stage 1: download raw daily OHLCV from Yahoo Finance for every constituent plus the VIX.

Resumable: tickers already on disk are skipped, so rerun after a rate-limit block.
Raw files are written exactly as Yahoo returns them; all cleaning happens in clean.py.
Note: auto_adjust=True means Yahoo rescales past prices after every dividend or
split, so a later re-download will not match this snapshot exactly.
"""
import time
from datetime import date

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

from config import CONSTITUENTS, END, RAW_PRICE_DIR, RUN_LOG, START, VIX, price_path

PAUSE = 3              # seconds between tickers
BATCH_SIZE = 50        # long break after this many downloads
BATCH_PAUSE = 60
MAX_RETRIES = 3
MAX_CONSECUTIVE_FAILS = 5


def fetch(ticker):
    """Download daily bars for one ticker. Retries only on rate limits / empty results."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = yf.Ticker(ticker).history(start=START, end=END, interval="1d", auto_adjust=True)
            if df.empty:
                raise ValueError("empty result")
            df.index = df.index.tz_localize(None)
            return df[["Open", "High", "Low", "Close", "Volume"]]
        except (YFRateLimitError, ValueError) as e:
            print(f"  [{ticker}] attempt {attempt}/{MAX_RETRIES} failed: {e!r}")
            if attempt < MAX_RETRIES:
                wait = 30 * 2 ** (attempt - 1)
                print(f"  waiting {wait}s...")
                time.sleep(wait)
    return None


def download_all(tickers):
    RAW_PRICE_DIR.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tickers if not price_path(t).exists()]
    print(f"{len(tickers) - len(todo)} already on disk, {len(todo)} to download")
    consecutive, done = 0, 0
    t0 = time.time()

    for i, tkr in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {tkr}")
        df = fetch(tkr)
        if df is None:
            consecutive += 1
            if consecutive >= MAX_CONSECUTIVE_FAILS:
                print("\nToo many failures in a row; probably blocked. "
                      "Wait a while (or switch network) and rerun; it will resume.")
                break
        else:
            tmp = price_path(tkr).with_suffix(".tmp")   # write-then-rename: no half-written files
            df.to_csv(tmp)
            tmp.replace(price_path(tkr))
            consecutive = 0
            done += 1
            if done % BATCH_SIZE == 0:
                print(f"  --- batch break ({BATCH_PAUSE}s) ---")
                time.sleep(BATCH_PAUSE)
        time.sleep(PAUSE)

    print(f"\nDownload pass finished in {(time.time() - t0) / 60:.1f} min.")
    if done:
        with open(RUN_LOG, "a") as f:       # append, so every session is recorded
            f.write(f"run date: {date.today()} | yfinance {yf.__version__} | "
                    f"range {START} to {END} | auto_adjust=True | downloaded {done}\n")


if __name__ == "__main__":
    meta = pd.read_csv(CONSTITUENTS)
    tickers = [VIX] + meta["ticker"].tolist()
    download_all(tickers)

    missing = [t for t in tickers if not price_path(t).exists()]
    (RAW_PRICE_DIR.parent / "failed_tickers.txt").write_text("\n".join(missing))
    print(f"Missing tickers: {missing or 'none'}")
    if missing:
        print("Constituents with no data in the window (e.g. 2026 spin-offs) are expected "
              "to stay missing; anything else: wait and rerun to resume.")
