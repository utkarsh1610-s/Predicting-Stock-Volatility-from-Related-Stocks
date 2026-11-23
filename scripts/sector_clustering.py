"""Does volatility cluster within sectors beyond the market? (same-day explanation, not forecasting)

  python scripts/sector_clustering.py

Descriptive only, and restricted to the training and validation years (2010-2021) by construction, so the
test years are never loaded. For every stock and day, with the stock's own series removed from the averages:
  market factor   mean daily log-volatility of all OTHER stocks that day
  sector excess   mean of the stock's sector peers minus the market factor
Reported (results/sector_clustering.csv, results/sector_episodes.csv):
  * R2 of a stock's log-volatility (stock fixed effects) from the market factor alone, and with the
    sector excess added. The difference is how much more the sector explains.
  * average pairwise correlation of daily volatility residuals after the market is removed, for
    stocks in the same sector vs different sectors (and the same before removing the market).
  * how persistent a sector's excess volatility is (autocorrelation at 1, 5, 22 days).
  * the months where a sector's volatility was furthest from the market's.
This is exploratory: it names sectors only after looking, so any follow-up test on them is a hypothesis
to pre-register, not a result.
"""
import numpy as np
import pandas as pd

from config import PROCESSED_DIR, RESULTS_DIR, SPLITS
from features import compute_features

MIN_OBS = 500          # a stock needs this many days to enter
CSV_FMT = "%.8g"


def restrict_to_train_val(stocks, vix):
    """Drop everything after the validation period, so the test years are never used."""
    end = SPLITS["val"][1]
    return stocks[stocks["date"] <= end], vix[vix["date"] <= end]


def lv_matrix(rows):
    """Dates x tickers of daily log-volatility, plus each ticker's sector."""
    sector = rows.drop_duplicates("ticker").set_index("ticker")["sector"]
    lv = rows.pivot(index="date", columns="ticker", values="lv")
    return lv, sector.reindex(lv.columns)


def long_panel(lv, sector):
    """One row per stock-day with the market and sector-excess factors, own series excluded, each
    column demeaned within the stock (the fixed-effects transformation)."""
    valid = lv.notna()
    tot_sum, tot_cnt = lv.sum(axis=1), valid.sum(axis=1)
    sec_sum = {s: lv[sector.index[sector == s]].sum(axis=1) for s in sector.unique()}
    sec_cnt = {s: valid[sector.index[sector == s]].sum(axis=1) for s in sector.unique()}
    parts = []
    for t in lv.columns:
        v, ok, s = lv[t], valid[t], sector[t]
        v0, n = v.fillna(0), ok.astype(int)
        d = pd.DataFrame({"y": v, "m": (tot_sum - v0) / (tot_cnt - n).replace(0, np.nan),
                          "s": (sec_sum[s] - v0) / (sec_cnt[s] - n).replace(0, np.nan)})[ok].dropna()
        if len(d) < MIN_OBS:
            continue
        d = d - d.mean()
        d["sx"] = d["s"] - d["m"]                          # sector excess over the market (demeaned)
        d["ticker"], d["sector"] = t, s
        d.index.name = "date"
        parts.append(d.drop(columns="s").reset_index())
    return pd.concat(parts, ignore_index=True)


def r2(cols, df):
    x = df[cols].to_numpy()
    y = df["y"].to_numpy()
    b, *_ = np.linalg.lstsq(x, y, rcond=None)
    e = y - x @ b
    return float(1 - (e @ e) / (y @ y))


def residual_corr_matrix(long):
    """Daily volatility residuals after removing each stock's own market loading, then correlations."""
    res = {}
    for t, d in long.groupby("ticker"):
        beta = (d["m"] * d["y"]).sum() / (d["m"] ** 2).sum()
        res[t] = pd.Series((d["y"] - beta * d["m"]).to_numpy(), index=d["date"].to_numpy())
    resid = pd.DataFrame(res)
    raw = long.pivot(index="date", columns="ticker", values="y")
    return resid.corr(min_periods=MIN_OBS), raw.corr(min_periods=MIN_OBS)


def mean_same_diff(corr, sector, scope=None):
    """Mean off-diagonal correlation among same-sector pairs and among different-sector pairs
    (restricted to pairs involving `scope` if given)."""
    names = list(corr.columns)
    sec = np.array([sector[t] for t in names])
    c = corr.to_numpy()
    off = ~np.eye(len(names), dtype=bool)
    same = sec[:, None] == sec[None, :]
    if scope is not None:
        k = sec == scope
        sel = np.outer(k, np.ones(len(k), bool))                # rows in scope
        return float(np.nanmean(c[sel & same & off])), float(np.nanmean(c[sel & ~same & off]))
    return float(np.nanmean(c[same & off])), float(np.nanmean(c[~same & off]))


def sector_excess_daily(lv, sector):
    """Daily mean log-volatility of each sector minus the all-stock mean."""
    market = lv.mean(axis=1)
    return pd.DataFrame({s: lv[sector.index[sector == s]].mean(axis=1) - market for s in sector.unique()})


def build_tables(lv, sector):
    long = long_panel(lv, sector)
    resid_c, raw_c = residual_corr_matrix(long)
    kept = pd.Series({t: sector[t] for t in resid_c.columns})
    rows = []
    for scope in ["ALL"] + sorted(kept.unique()):
        d = long if scope == "ALL" else long[long["sector"] == scope]
        a, b = r2(["m"], d), r2(["m", "sx"], d)
        sec_scope = None if scope == "ALL" else scope
        rs, rd = mean_same_diff(resid_c, kept, sec_scope)
        ws, wd = mean_same_diff(raw_c, kept, sec_scope)
        rows.append({"scope": scope, "stocks": d["ticker"].nunique(), "r2_market": a, "r2_market_plus_sector": b,
                     "sector_adds": b - a, "resid_corr_same_sector": rs, "resid_corr_other_sectors": rd,
                     "raw_corr_same_sector": ws, "raw_corr_other_sectors": wd})
    table = pd.DataFrame(rows).set_index("scope")
    excess = sector_excess_daily(lv, kept).dropna(how="all")
    for k in (1, 5, 22):
        table.loc[excess.columns, f"excess_autocorr_lag{k}"] = [excess[s].dropna().autocorr(k) for s in excess.columns]
    monthly = excess.groupby(excess.index.to_period("M")).mean().stack().rename("excess").reset_index()
    monthly.columns = ["month", "sector", "excess"]
    monthly["month"] = monthly["month"].astype(str)
    monthly["ratio"] = np.exp(monthly["excess"])
    return table.reset_index(), monthly.sort_values("excess", key=abs, ascending=False)


def main():
    stocks = pd.read_parquet(PROCESSED_DIR / "stocks.parquet")
    vix = pd.read_parquet(PROCESSED_DIR / "vix.parquet")
    stocks, vix = restrict_to_train_val(stocks, vix)
    rows, _ = compute_features(stocks, vix)
    assert rows["date"].max() <= pd.Timestamp(SPLITS["val"][1])
    lv, sector = lv_matrix(rows)
    table, episodes = build_tables(lv, sector)

    RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(RESULTS_DIR / "sector_clustering.csv", index=False, float_format=CSV_FMT)
    episodes.to_csv(RESULTS_DIR / "sector_episodes.csv", index=False, float_format=CSV_FMT)

    pd.set_option("display.width", 250)
    print(f"Stocks: {lv.shape[1]}, days: {lv.shape[0]} ({lv.index.min():%Y-%m-%d} to {lv.index.max():%Y-%m-%d})\n")
    show = table[["scope", "stocks", "r2_market", "sector_adds", "resid_corr_same_sector",
                  "resid_corr_other_sectors", "excess_autocorr_lag1", "excess_autocorr_lag22"]]
    print(show.round(3).to_string(index=False))
    print("\nLargest months a sector's volatility was far from the market (ratio sector / market):")
    print(episodes.head(10)[["sector", "month", "ratio"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
