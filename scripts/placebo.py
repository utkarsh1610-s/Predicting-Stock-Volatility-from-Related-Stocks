"""Is the neighbor gain about specific links, or just the market-wide volatility level?

  python scripts/placebo.py        # validation only; the test years are never scored

Every model is HAR plus one extra feature set, fit on the training years, scored on validation:
  market        mean har_* of ALL other stocks that day (no graph at all)
  sector, corr  the real graphs (as in the baselines)
  market+sector, market+corr   the graph ON TOP of the market average
  rand_sector   per stock, as many random peers as its sector has (fixed over time), 5 seeds
  rand_corr     per stock and month, as many random stocks as its correlation neighbors, 5 seeds
The random twins keep every stock's neighbor COUNT and the monthly availability of the real graph,
so they remove only the information in WHO the neighbors are.

READING THE RESULT (written before the experiment was run):
  Market-wide only: har+market is within 1.0 percentage point of har+sector and har+corr (QLIKE vs
    har) AND adding the graph on top of the market improves QLIKE by less than 0.5%. Then graph
    structure adds ~nothing in a linear HAR, and GNNHAR has little to learn from the graph.
  Graph-specific information: each real graph beats its random twin by more than 1% QLIKE with DM
    p < 0.05 in every seed AND market+graph beats market by more than 0.5% (p < 0.05, and still
    positive with Feb-Apr 2020 removed).
  Anything in between is reported as mixed. This is a LINEAR test: a nonlinear model could in
  principle find structure that an average hides, so a "market-wide" verdict is about HAR only.

Outputs: results/placebo.csv.
"""
import numpy as np
import pandas as pd

import baselines as bl
from config import PROCESSED_DIR, RESULTS_DIR
from features import compute_features, panel_from_rows
from graphs import corr_edges, neighbor_features, return_matrix, sector_edges
from metrics import qlike_loss, smearing_factor
from sensitivity import compare, excl_covid, rel

SEEDS = range(5)
MKT = [f"mkt_{c}" for c in bl.HAR]
RSEC = [f"rsec_{c}" for c in bl.HAR]
RCOR = [f"rcor_{c}" for c in bl.HAR]


def market_edges(tickers):
    """Every stock linked to every other stock."""
    t = np.array(sorted(tickers))
    a, b = np.meshgrid(t, t, indexing="ij")
    keep = a != b
    return pd.DataFrame({"ticker": a[keep], "neighbor": b[keep]})


def random_sector_edges(sec_edges, seed):
    """Same peer COUNT per stock as the sector graph, but random peers; fixed over time."""
    rng = np.random.default_rng(seed)
    tickers = np.array(sorted(set(sec_edges["ticker"]) | set(sec_edges["neighbor"])))
    rows = []
    for t, c in sec_edges.groupby("ticker").size().items():
        pool = tickers[tickers != t]
        rows += [(t, n) for n in rng.choice(pool, size=min(c, len(pool)), replace=False)]
    return pd.DataFrame(rows, columns=["ticker", "neighbor"])


def random_corr_edges(cor_edges, seed):
    """Per stock and month, as many neighbors as the real graph gave it, drawn at random from the
    stocks that have a graph that month."""
    rng = np.random.default_rng(seed)
    rows = []
    for month, g in cor_edges.groupby("month"):
        pool_all = np.array(sorted(set(g["ticker"]) | set(g["neighbor"])))
        for t, c in g.groupby("ticker").size().items():
            pool = pool_all[pool_all != t]
            rows += [(month, t, n) for n in rng.choice(pool, size=min(c, len(pool)), replace=False)]
    return pd.DataFrame(rows, columns=["month", "ticker", "neighbor"])


def losses_from(df, cols):
    """Fit HAR + `cols` on train rows; per-row QLIKE on validation rows, keyed (ticker, y_date)."""
    train = (df["split"] == "train").to_numpy()
    y = df["y"].to_numpy()
    x = df[cols].to_numpy()
    pred = bl.predict_ols(bl.fit_ols(x[train], y[train]), x)
    smear = smearing_factor(y[train], pred[train])
    m = (df["split"] == "val").to_numpy()
    out = df.loc[m, ["ticker", "y_date"]].reset_index(drop=True)
    out["loss"] = qlike_loss(np.exp(2 * y[m]), np.exp(2 * pred[m]) * smear)
    return out


def renamed(nbr, prefix_from, prefix_to):
    keep = {f"{prefix_from}_{c}": f"{prefix_to}_{c}" for c in bl.HAR}
    return nbr[["date", "ticker", *keep]].rename(columns=keep)


def main():
    stocks = pd.read_parquet(PROCESSED_DIR / "stocks.parquet")
    vix = pd.read_parquet(PROCESSED_DIR / "vix.parquet")
    calendar = pd.DatetimeIndex(sorted(vix["date"]))
    rows, _ = compute_features(stocks, vix)
    panel = panel_from_rows(rows)
    sec = sector_edges(stocks)
    cor = corr_edges(return_matrix(stocks, calendar))

    nbr = neighbor_features(rows, sec, cor)
    mkt = renamed(neighbor_features(rows, market_edges(stocks["ticker"].unique()), cor), "sec", "mkt")
    base = panel.merge(nbr, on=["ticker", "date"], how="left").merge(mkt, on=["ticker", "date"], how="left")
    base = base[base["split"].isin(["train", "val"])]
    base = base.dropna(subset=bl.HAR + bl.SEC + bl.COR + MKT).reset_index(drop=True)
    print(f"Base sample: {len(base):,} rows (train+val)", flush=True)

    L = {"har": losses_from(base, bl.HAR),
         "market": losses_from(base, bl.HAR + MKT),
         "sector": losses_from(base, bl.HAR + bl.SEC),
         "corr": losses_from(base, bl.HAR + bl.COR),
         "market+sector": losses_from(base, bl.HAR + MKT + bl.SEC),
         "market+corr": losses_from(base, bl.HAR + MKT + bl.COR)}
    for s in SEEDS:
        rn = neighbor_features(rows, random_sector_edges(sec, s), random_corr_edges(cor, s))
        d = base[["ticker", "date", "split", "y", "y_date"] + bl.HAR].merge(
            renamed(rn, "sec", "rsec"), on=["ticker", "date"], how="left").merge(
            renamed(rn, "cor", "rcor"), on=["ticker", "date"], how="left")
        L[f"rand_sector_s{s}"] = losses_from(d.dropna(subset=RSEC), bl.HAR + RSEC)
        L[f"rand_corr_s{s}"] = losses_from(d.dropna(subset=RCOR), bl.HAR + RCOR)
        print(f"  seed {s} done", flush=True)

    out = []

    def add(label, a, b):
        chg, p, chg_cov, n = compare(L[a], L[b], "loss")
        out.append({"comparison": label, "model": a, "versus": b, "qlike_change": chg, "dm_p": p,
                    "qlike_change_exCOVID": chg_cov, "rows": n})

    for m in ["market", "sector", "corr", "market+sector", "market+corr"]:
        add(f"{m} vs har", m, "har")
    for g in ["sector", "corr"]:
        add(f"{g} vs market", g, "market")
        add(f"market+{g} vs market", f"market+{g}", "market")
    for s in SEEDS:
        add(f"rand_sector_s{s} vs har", f"rand_sector_s{s}", "har")
        add(f"rand_corr_s{s} vs har", f"rand_corr_s{s}", "har")
        add(f"sector vs rand_sector_s{s}", "sector", f"rand_sector_s{s}")
        add(f"corr vs rand_corr_s{s}", "corr", f"rand_corr_s{s}")
    res = pd.DataFrame(out)
    RESULTS_DIR.mkdir(exist_ok=True)
    res.to_csv(RESULTS_DIR / "placebo.csv", index=False, float_format=bl.CSV_FMT)

    fmt = lambda r: f"{r.qlike_change:+.2%}  p={r.dm_p:.3g}  exCOVID {r.qlike_change_exCOVID:+.2%}"
    print("\nGAIN OVER PLAIN HAR (QLIKE, validation; negative = better)")
    for _, r in res[res["versus"] == "har"].iterrows():
        if "_s" not in r["model"]:
            print(f"  {r['model']:15s} {fmt(r)}")
    for kind in ["rand_sector", "rand_corr"]:
        sub = res[res["comparison"].str.fullmatch(f"{kind}_s\\d vs har")]
        print(f"  {kind:15s} mean {sub.qlike_change.mean():+.2%}  range [{sub.qlike_change.min():+.2%}, "
              f"{sub.qlike_change.max():+.2%}]  exCOVID mean {sub.qlike_change_exCOVID.mean():+.2%}")
    print("\nDOES THE GRAPH ADD ANYTHING BEYOND THE MARKET AVERAGE?")
    for _, r in res[res["comparison"].str.contains(r"^(sector|corr|market\+sector|market\+corr) vs market")].iterrows():
        print(f"  {r['comparison']:24s} {fmt(r)}")
    print("\nREAL GRAPH VS ITS RANDOM TWIN (same degrees)")
    for kind, real in [("rand_sector", "sector"), ("rand_corr", "corr")]:
        sub = res[res["comparison"].str.fullmatch(f"{real} vs {kind}_s\\d")]
        print(f"  {real} vs {kind}: mean {sub.qlike_change.mean():+.2%}  range [{sub.qlike_change.min():+.2%}, "
              f"{sub.qlike_change.max():+.2%}]  max p {sub.dm_p.max():.3g}  exCOVID mean {sub.qlike_change_exCOVID.mean():+.2%}")


if __name__ == "__main__":
    main()
