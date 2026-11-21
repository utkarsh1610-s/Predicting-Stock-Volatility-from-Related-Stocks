"""Stage 0: snapshot the S&P 500 constituent list (ticker, sector, sub-industry) from Wikipedia.

Saves the raw HTML next to the parsed CSV. Refuses to overwrite an existing
snapshot unless --force is passed, because the price download was built from
that exact list (re-scraping later gives a different membership).
"""
import argparse
import io
import sys
from datetime import date

import pandas as pd
import requests

from config import CONSTITUENTS, RAW_WIKI_DIR

URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# Wikipedia can return 403 for the default python-requests User-Agent
HEADERS = {"User-Agent": "CS6220-class-project (student research)"}


def main(force):
    if CONSTITUENTS.exists() and not force:
        sys.exit(f"{CONSTITUENTS} already exists; pass --force to re-scrape "
                 "(this changes the stock universe).")

    resp = requests.get(URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    RAW_WIKI_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_WIKI_DIR / f"sp500_{date.today()}.html").write_text(resp.text)

    # The first table on the page is the current constituents list
    df = pd.read_html(io.StringIO(resp.text))[0]
    df = df[["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]].rename(columns={
        "Symbol": "wiki_ticker", "Security": "company",
        "GICS Sector": "sector", "GICS Sub-Industry": "sub_industry"})
    # Wikipedia writes BRK.B / BF.B; Yahoo expects BRK-B / BF-B
    df["ticker"] = df["wiki_ticker"].str.replace(".", "-", regex=False)

    print(f"Rows: {len(df)}, unique tickers: {df['ticker'].nunique()}, "
          f"sectors: {df['sector'].nunique()} (expect 11)")
    print(f"Missing values: {int(df.isna().sum().sum())}")
    CONSTITUENTS.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CONSTITUENTS, index=False)
    print(f"Saved {CONSTITUENTS}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)
