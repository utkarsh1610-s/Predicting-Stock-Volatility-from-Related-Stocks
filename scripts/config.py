"""Shared settings for the data pipeline. Run all scripts from the project root."""
from pathlib import Path

# Study window (END is exclusive, so this includes all of 2025)
START, END = "2010-01-01", "2026-01-01"

CONSTITUENTS = Path("data/sp500_constituents.csv")
RAW_WIKI_DIR = Path("data/raw_wiki")
RAW_PRICE_DIR = Path("data/raw_full")
PROCESSED_DIR = Path("data/processed")
FEATURES_DIR = Path("data/features")
RUN_LOG = Path("data/run_info.txt")

VIX = "^VIX"
PX = ["Open", "High", "Low", "Close"]

# A ticker needs this many clean rows to enter the model: 22-day HAR lag
# + 60-day correlation window + margin. Drops brand-new listings such as Q.
MIN_ROWS = 126

# Yahoo stitches some tickers' early history from a different security
# (foreign/OTC quotes, a SPAC shell, pre-relisting trading). We start each of
# these at its real US exchange listing, checked against the volume data.
LISTING_STARTS = {
    "SW":   ("2024-07-08", "Smurfit Westrock NYSE debut; earlier rows are mostly zero-volume foreign quotes"),
    "FERG": ("2022-03-08", "US liquidity jumps ~10x in Mar 2022; 2021 secondary listing traded ~$3M/day"),
    "AMCR": ("2019-06-11", "Amcor NYSE listing; earlier rows are an OTC shadow of the ASX listing"),
    "VRT":  ("2020-02-07", "Vertiv de-SPAC; earlier rows are the SPAC shell trading near $10"),
    "VST":  ("2017-05-10", "Vistra NYSE listing; earlier rows are thin OTC trading"),
    "CHTR": ("2010-09-14", "Charter Nasdaq relisting after bankruptcy; earlier rows OTC with zero-volume days"),
    "HWM":  ("2016-12-23", "End of zero-volume when-issued period after the Alcoa/Arconic split"),
}


def price_path(ticker):
    return RAW_PRICE_DIR / f"{ticker.replace('^', '')}.csv"


# Time splits, assigned by the date of the value being predicted (see features.py)
SPLITS = {"train": ("2010-01-01", "2019-12-31"),
          "val":   ("2020-01-01", "2021-12-31"),
          "test":  ("2022-01-01", "2025-12-31")}

# HAR lookback windows (trading days) and winsorization of daily log-volatility,
# with bounds estimated on the training split only
HAR_WINDOWS = (1, 5, 22)
WINSOR_Q = (0.001, 0.999)
