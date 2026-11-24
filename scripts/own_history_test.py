"""Is the sector's information redundant once a stock's own history is known?

  python scripts/own_history_test.py        # validation only; the test years are never scored

sector_clustering.py shows that volatility clusters by sector on the same day, yet adding sector peers to
a forecast that already has the stock's own history and the market gains almost nothing. Hypothesis: a stock
hit by a sector shock already shows it in its own recent volatility, so the sector information is real but
redundant. Test: does the sector help when the stock's own recent history is NOT available?

QLIKE-trained linear models (as baselines.har_q), fit on the training years, scored on validation:
  WITHOUT own history   stock_level (the stock's mean log-vol over the TRAINING years only, a constant per
                        stock) + market average            [+ sector peers] [+ correlation neighbors]
  WITH own history      own HAR (har_1/5/22) + market average [+ sector peers] [+ correlation neighbors]
For each graph we compare "+graph" against "market only" within each setting.

READING THE RESULT (written before the test was run), pooled over all stocks, for each graph separately:
  Redundant: the graph's gain WITHOUT own history is more than 1.0% QLIKE (DM p < 0.05, still more than 0.5%
    without Feb-Apr 2020) AND its gain WITH own history is under 0.5%.
  Absent: the gain without own history is under 0.5%. Then the sector adds nothing even by itself.
  Anything else: mixed, reported as such.
Per sector (each sector gets its own fit) is hypothesis-generating: 11 sectors x 2 graphs x 2 settings = 44
tests, so a pair is only flagged at DM p < 0.001 with a gain over 1.0%. Utilities, Energy and Real Estate
are the sectors sector_clustering.py singled out AFTER looking, so a result there is a hypothesis, not a finding.

Outputs: results/own_history.csv.
"""
import numpy as np
import pandas as pd

import baselines as bl
from config import PROCESSED_DIR, RESULTS_DIR
from features import compute_features, panel_from_rows
from graphs import corr_edges, neighbor_features, return_matrix, sector_edges
from placebo import MKT, market_edges, renamed
from sector_check import losses_q
from sensitivity import compare

GRAPHS = {"sector": bl.SEC, "corr": bl.COR}
MIN_SECTOR_STOCKS = 10


def add_stock_level(df):
    """A constant per stock: its mean log-volatility over the TRAINING rows only. Stocks with no training
    rows (listed after 2019) get none and are dropped."""
    level = df[df["split"] == "train"].groupby("ticker")["lv"].mean()
    out = df.assign(stock_level=df["ticker"].map(level))
    return out.dropna(subset=["stock_level"]).reset_index(drop=True)


def model_specs(graphs=GRAPHS):
    specs = {"without/market": ["stock_level"] + MKT, "with/market": bl.HAR + MKT}
    for g, cols in graphs.items():
        specs[f"without/market+{g}"] = ["stock_level"] + MKT + cols
        specs[f"with/market+{g}"] = bl.HAR + MKT + cols
    return specs


def fit_all(df, graphs=GRAPHS):
    return {name: losses_q(df, cols) for name, cols in model_specs(graphs).items()}


def comparisons(L, scope, graphs=GRAPHS, strict_p=0.001):
    out = []
    for setting in ("without", "with"):
        for g in graphs:
            chg, p, cov, n = compare(L[f"{setting}/market+{g}"], L[f"{setting}/market"], "loss")
            out.append({"scope": scope, "comparison": f"{g} beyond market", "own_history": setting,
                        "qlike_change": chg, "dm_p": p, "qlike_change_exCOVID": cov, "rows": n,
                        "flag": bool(chg < -0.01 and p < strict_p and cov < -0.005)})
    chg, p, cov, n = compare(L["with/market"], L["without/market"], "loss")
    out.append({"scope": scope, "comparison": "own history beyond market + stock level", "own_history": "-",
                "qlike_change": chg, "dm_p": p, "qlike_change_exCOVID": cov, "rows": n, "flag": False})
    return out


def verdict(pooled, graph):
    """Apply the pre-registered reading to the pooled rows for one graph."""
    r = pooled[pooled["comparison"] == f"{graph} beyond market"].set_index("own_history")
    wo, wi = r.loc["without"], r.loc["with"]
    if wo["qlike_change"] > -0.005:
        return "absent"
    if wo["qlike_change"] < -0.01 and wo["dm_p"] < 0.05 and wo["qlike_change_exCOVID"] < -0.005 \
            and wi["qlike_change"] > -0.005:
        return "redundant"
    return "mixed"


def build_base(stocks, vix):
    calendar = pd.DatetimeIndex(sorted(vix["date"]))
    rows, _ = compute_features(stocks, vix)
    panel = panel_from_rows(rows)
    sec = sector_edges(stocks)
    cor = corr_edges(return_matrix(stocks, calendar))
    nbr = neighbor_features(rows, sec, cor)
    mkt = renamed(neighbor_features(rows, market_edges(stocks["ticker"].unique()), cor), "sec", "mkt")
    base = panel.merge(nbr, on=["ticker", "date"], how="left").merge(mkt, on=["ticker", "date"], how="left")
    base = base[base["split"].isin(["train", "val"])]
    return base.dropna(subset=bl.HAR + bl.SEC + bl.COR + MKT).reset_index(drop=True)


def main():
    stocks = pd.read_parquet(PROCESSED_DIR / "stocks.parquet")
    vix = pd.read_parquet(PROCESSED_DIR / "vix.parquet")
    base = add_stock_level(build_base(stocks, vix))
    print(f"Base sample: {len(base):,} rows, {base['ticker'].nunique()} stocks", flush=True)

    out = comparisons(fit_all(base), "ALL")
    pooled = pd.DataFrame(out)
    for sector, d in base.groupby("sector"):
        if d["ticker"].nunique() >= MIN_SECTOR_STOCKS:
            out += comparisons(fit_all(d), sector)
            print(f"  {sector} done", flush=True)
    res = pd.DataFrame(out)
    RESULTS_DIR.mkdir(exist_ok=True)
    res.to_csv(RESULTS_DIR / "own_history.csv", index=False, float_format=bl.CSV_FMT)

    fmt = lambda r: f"{r.qlike_change:+.2%}  p={r.dm_p:.3g}  exCOVID {r.qlike_change_exCOVID:+.2%}"
    print("\nPOOLED (QLIKE change, negative = better)")
    for _, r in pooled.iterrows():
        label = f"{r.comparison}, {r.own_history} own history" if r.own_history != "-" else r.comparison
        print(f"  {label:58s} {fmt(r)}")
    print("\nVERDICT (pre-registered reading):", {g: verdict(pooled, g) for g in GRAPHS})
    sectors = res[res["scope"] != "ALL"]
    flagged = sectors[sectors["flag"]]
    print(f"\nPER SECTOR: {len(flagged)} of {len(sectors) - sectors['comparison'].eq('own history beyond market + stock level').sum()} "
          "sector/graph/setting tests flagged (p<0.001, gain>1%)")
    show = sectors[sectors["comparison"] != "own history beyond market + stock level"]
    wide = show.pivot_table(index="scope", columns=["comparison", "own_history"], values="qlike_change")
    pd.set_option("display.width", 250)
    print((wide * 100).round(2).to_string())


if __name__ == "__main__":
    main()
