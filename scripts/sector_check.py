"""Does any single sector show a graph effect beyond the market average?

  python scripts/sector_check.py        # validation only; the test years are never scored

The pooled placebo (placebo.py) fits one set of coefficients for all stocks, which can wash out a sector
where links matter much more than elsewhere. Here every sector gets its OWN regression (QLIKE-trained
HAR, as in baselines.har_q), trained on that sector's training rows and scored on its validation rows:
  none            HAR
  market          HAR + market average of all other stocks
  market+sector   HAR + market + the sector-peer average
  market+corr     HAR + market + the correlation-neighbor average
The question for each sector and graph: does adding the graph on top of the market help?

DECISION RULE (written before running; 11 sectors x 2 graphs = 22 tests, so p is Bonferroni-style):
  A sector/graph pair is flagged "graph matters" only if market+graph beats market by more than 1.0% QLIKE
  with DM p < 0.002 (about 0.05 / 22), AND the improvement holds, at more than 0.5%, with Feb-Apr 2020
  removed. A flagged pair would still need a random-twin check (placebo.py logic) before it is believed.
  Note the sector-peer average is, within one sector, essentially an industry factor shared by its stocks.

Outputs: results/sector_check.csv.
"""
import numpy as np
import pandas as pd

import baselines as bl
from config import PROCESSED_DIR, RESULTS_DIR
from features import compute_features, panel_from_rows
from graphs import corr_edges, neighbor_features, return_matrix, sector_edges
from metrics import qlike_loss
from placebo import MKT, market_edges, renamed
from sensitivity import compare

MIN_SECTOR_STOCKS = 10


def losses_q(df, cols):
    """QLIKE-trained linear fit on train rows; per-row val QLIKE keyed (ticker, y_date)."""
    train = (df["split"] == "train").to_numpy()
    y = df["y"].to_numpy()
    x = df[cols].to_numpy()
    pred = bl.predict_ols(bl.fit_qlike_linear(x[train], y[train]), x)
    m = (df["split"] == "val").to_numpy()
    out = df.loc[m, ["ticker", "y_date"]].reset_index(drop=True)
    out["loss"] = qlike_loss(np.exp(2 * y[m]), np.exp(2 * pred[m]))
    return out


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
    print(f"Base sample: {len(base):,} rows", flush=True)

    out = []
    for sector, d in base.groupby("sector"):
        n_stocks = d["ticker"].nunique()
        if n_stocks < MIN_SECTOR_STOCKS:
            print(f"  skipping {sector}: only {n_stocks} stocks")
            continue
        L = {"none": losses_q(d, bl.HAR), "market": losses_q(d, bl.HAR + MKT),
             "market+sector": losses_q(d, bl.HAR + MKT + bl.SEC),
             "market+corr": losses_q(d, bl.HAR + MKT + bl.COR)}
        rec = {"sector": sector, "stocks": n_stocks, "val_rows": len(L["none"])}
        chg, p, cov, _ = compare(L["market"], L["none"], "loss")
        rec |= {"market_vs_none": chg, "market_vs_none_p": p, "market_vs_none_exCOVID": cov}
        for g in ["sector", "corr"]:
            chg, p, cov, _ = compare(L[f"market+{g}"], L["market"], "loss")
            rec |= {f"{g}_beyond_market": chg, f"{g}_p": p, f"{g}_exCOVID": cov,
                    f"{g}_flag": bool(chg < -0.01 and p < 0.002 and cov < -0.005)}
        out.append(rec)
        print(f"  {sector:24s} done", flush=True)

    res = pd.DataFrame(out)
    RESULTS_DIR.mkdir(exist_ok=True)
    res.to_csv(RESULTS_DIR / "sector_check.csv", index=False, float_format=bl.CSV_FMT)
    pct = lambda c: res[c].map(lambda x: f"{x:+.2%}")
    show = pd.DataFrame({"sector": res["sector"], "stocks": res["stocks"],
                         "mkt vs none": pct("market_vs_none"),
                         "+sector": pct("sector_beyond_market"), "p": res["sector_p"].map(lambda x: f"{x:.3g}"),
                         "exCOV": pct("sector_exCOVID"), "flag": res["sector_flag"],
                         "+corr": pct("corr_beyond_market"), "p ": res["corr_p"].map(lambda x: f"{x:.3g}"),
                         "exCOV ": pct("corr_exCOVID"), "flag ": res["corr_flag"]})
    pd.set_option("display.width", 250)
    print("\nPER-SECTOR, validation, QLIKE change (negative = better)")
    print(show.to_string(index=False))
    print(f"\nFlagged sector/graph pairs: {int(res['sector_flag'].sum() + res['corr_flag'].sum())} of {2 * len(res)}")


if __name__ == "__main__":
    main()
