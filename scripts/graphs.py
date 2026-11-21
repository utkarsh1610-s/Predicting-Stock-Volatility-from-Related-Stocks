"""Stage 4: build the sector and correlation graphs and the neighbor features.

Outputs:
  data/graphs/sector_edges.parquet   ticker, neighbor            (fixed)
  data/graphs/corr_edges.parquet     month, ticker, neighbor, corr, rank   (rebuilt monthly)
  data/features/neighbors.parquet    one row per panel row (ticker, date) with
      sec_har_{1,5,22}, sec_n    mean of the sector peers' har_* on the same day, and count
      cor_har_{1,5,22}, cor_n    same for the correlation neighbors of that month

Rules (each is covered by tests/test_graphs.py):
  * Correlation graph for month M uses only returns from trading days strictly before
    M's first trading day, so it never sees any day it is applied to.
  * Pairwise-complete correlation: a pair needs CORR_MIN_COMMON shared days in the
    window. A stock with fewer has no neighbors that month and its cor_* features are NaN
    (true for the first months of 2010 and for new listings).
  * A neighbor contributes on day t only if it has har_* that day. cor_n / sec_n say how
    many did; the mean is NaN when none did.
  * Neighbor values are same-day har_* (known at close of t), never anything later.
"""
import numpy as np
import pandas as pd

from config import (CORR_K, CORR_MIN_COMMON, CORR_WINDOW, FEATURES_DIR, GRAPHS_DIR,
                    HAR_WINDOWS, PROCESSED_DIR)
from features import build_panel, compute_features

HAR_COLS = [f"har_{w}" for w in HAR_WINDOWS]


def return_matrix(stocks, calendar):
    """Close-to-close log returns, dates x tickers, NaN where a day or its predecessor is missing."""
    close = stocks.pivot(index="date", columns="ticker", values="close").reindex(calendar)
    return np.log(close / close.shift(1))


def sector_edges(stocks):
    sec = stocks.drop_duplicates("ticker").set_index("ticker")["sector"].sort_index()
    pairs = sec.reset_index().merge(sec.reset_index(), on="sector", suffixes=("", "_n"))
    pairs = pairs[pairs["ticker"] != pairs["ticker_n"]]
    return (pairs[["ticker", "ticker_n"]].rename(columns={"ticker_n": "neighbor"})
            .sort_values(["ticker", "neighbor"]).reset_index(drop=True))


def corr_edges(returns, k=CORR_K, window=CORR_WINDOW, min_common=CORR_MIN_COMMON):
    """Top-k correlation neighbors per ticker for each calendar month.

    The graph for a month is estimated on the `window` trading days before the
    month's first trading day.
    """
    calendar = returns.index
    tickers = np.array(returns.columns)
    first_days = calendar.to_series().groupby(calendar.to_period("M")).first()
    out = []
    for month, d0 in first_days.items():
        pos = calendar.get_loc(d0)
        past = returns.iloc[max(0, pos - window):pos]
        if len(past) < min_common:
            continue
        c = past.corr(min_periods=min_common).to_numpy(copy=True)
        np.fill_diagonal(c, np.nan)
        score = np.where(np.isnan(c), -np.inf, c)
        order = np.argsort(-score, axis=1, kind="stable")[:, :k]
        for i, row in enumerate(order):
            for rank, j in enumerate(row, 1):
                if np.isfinite(score[i, j]):
                    out.append((month.to_timestamp(), tickers[i], tickers[j], score[i, j], rank))
    return pd.DataFrame(out, columns=["month", "ticker", "neighbor", "corr", "rank"])


def _adjacency(edges, tickers):
    idx = {t: i for i, t in enumerate(tickers)}
    a = np.zeros((len(tickers), len(tickers)))
    for t, n in zip(edges["ticker"], edges["neighbor"]):
        if t in idx and n in idx:
            a[idx[t], idx[n]] = 1.0
    return a


def _neighbor_means(wide, adj, prefix):
    """wide: dict col -> (dates x tickers) values. Mean over neighbors that have a value."""
    avail = (~np.isnan(wide[HAR_COLS[0]].to_numpy())).astype(float)
    den = avail @ adj.T
    out = {}
    for col in HAR_COLS:
        vals = np.nan_to_num(wide[col].to_numpy())
        num = vals @ adj.T
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"{prefix}_{col}"] = np.where(den > 0, num / den, np.nan)
    out[f"{prefix}_n"] = den
    return out


def neighbor_features(rows, sec_edges, cor_edges):
    """rows: compute_features output. Returns a (date, ticker)-keyed frame of neighbor means."""
    tickers = sorted(rows["ticker"].unique())
    wide = {c: rows.pivot(index="date", columns="ticker", values=c)[tickers] for c in HAR_COLS}
    dates = wide[HAR_COLS[0]].index

    sec = _neighbor_means(wide, _adjacency(sec_edges, tickers), "sec")
    sec_df = {k: pd.DataFrame(v, index=dates, columns=tickers).stack(future_stack=True)
              for k, v in sec.items()}

    # correlation features: start from "no graph" (NaN, zero neighbors), fill in month by month
    cor_cols = [f"cor_{c}" for c in HAR_COLS] + ["cor_n"]
    cor = {c: np.full((len(dates), len(tickers)), np.nan) for c in cor_cols}
    cor["cor_n"][:] = 0.0
    for month, e in cor_edges.groupby("month"):
        in_month = (dates >= month) & (dates < month + pd.offsets.MonthBegin(1))
        if not in_month.any():
            continue
        sub = {c: w[in_month] for c, w in wide.items()}
        for k, v in _neighbor_means(sub, _adjacency(e, tickers), "cor").items():
            cor[k][in_month] = v
    cor_df = {k: pd.DataFrame(v, index=dates, columns=tickers).stack(future_stack=True)
              for k, v in cor.items()}

    return pd.DataFrame({**sec_df, **cor_df}).rename_axis(["date", "ticker"]).reset_index()


def main():
    stocks = pd.read_parquet(PROCESSED_DIR / "stocks.parquet")
    vix = pd.read_parquet(PROCESSED_DIR / "vix.parquet")
    calendar = pd.DatetimeIndex(sorted(vix["date"]))

    sec = sector_edges(stocks)
    cor = corr_edges(return_matrix(stocks, calendar))
    rows, _ = compute_features(stocks, vix)
    panel, _ = build_panel(stocks, vix)
    nbr = neighbor_features(rows, sec, cor)
    nbr = panel[["date", "ticker"]].merge(nbr, on=["date", "ticker"], how="left")

    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    sec.to_parquet(GRAPHS_DIR / "sector_edges.parquet", index=False)
    cor.to_parquet(GRAPHS_DIR / "corr_edges.parquet", index=False)
    nbr.to_parquet(FEATURES_DIR / "neighbors.parquet", index=False)

    deg = sec.groupby("ticker").size()
    print(f"Sector graph: {len(sec):,} directed edges; peers per stock {deg.min()}-{deg.max()}")
    per_month = cor.groupby("month").size()
    print(f"Correlation graph: {cor['month'].nunique()} months "
          f"({cor['month'].min():%Y-%m} to {cor['month'].max():%Y-%m}), "
          f"{len(cor):,} edges, mean corr of neighbors {cor['corr'].mean():.2f}")
    same = cor.merge(stocks.drop_duplicates("ticker")[["ticker", "sector"]], on="ticker").merge(
        stocks.drop_duplicates("ticker")[["ticker", "sector"]].rename(
            columns={"ticker": "neighbor", "sector": "sector_n"}), on="neighbor")
    print(f"Share of correlation neighbors in the same sector: {(same['sector'] == same['sector_n']).mean():.0%}")
    print(f"Neighbor rows: {len(nbr):,}; with sector features {nbr['sec_har_1'].notna().mean():.1%}, "
          f"with correlation features {nbr['cor_har_1'].notna().mean():.1%}")


if __name__ == "__main__":
    main()
