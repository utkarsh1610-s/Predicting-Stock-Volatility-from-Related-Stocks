"""The placebo experiment (placebo.py) with LightGBM instead of linear HAR.

  python scripts/placebo_trees.py      # validation only; the test years are never scored

Does any structure survive a nonlinear model once the market-wide level is available? Every model has
the same base inputs (har_1/5/22, ret, dlogvol, vix) plus one extra feature set:
  market | sector | corr | market+sector | market+corr | rand_sector (3 seeds) | rand_corr (3 seeds)
(definitions as in placebo.py). ONE FIXED hyper-parameter setting for all models (7 leaves, lr 0.03, 500
rows per leaf, early stopping on validation): it is what every earlier tuning run chose, and it keeps
tuning luck out of the comparison so differences come from the features.

READING THE RESULT (written before the experiment was run):
  Graph-specific structure exists for trees only if BOTH hold:
    (a) market+graph beats market alone by more than 0.5% QLIKE, with DM p < 0.05 and still positive
        with Feb-Apr 2020 removed, for at least one graph; and
    (b) that real graph beats its random twin by more than 1.0% QLIKE with p < 0.05 in every seed.
  Otherwise: no structure beyond the market level, even for trees. The verdict is about equal-weight
  neighbor averages; a model that sees individual neighbors (GNNHAR) is a different test.

Outputs: results/placebo_trees.csv.
"""
import numpy as np
import pandas as pd

import baselines as bl
import trees
from config import PROCESSED_DIR, RESULTS_DIR
from features import compute_features, panel_from_rows
from graphs import corr_edges, neighbor_features, return_matrix, sector_edges
from metrics import qlike_loss, smearing_factor
from placebo import MKT, RCOR, RSEC, market_edges, random_corr_edges, random_sector_edges, renamed
from sensitivity import compare

SEEDS = range(3)
PARAMS = dict(num_leaves=7, learning_rate=0.03, min_child_samples=500)
BASE = trees.BASE


def tree_losses(df, cols):
    """Fit LightGBM on train rows (early stopping on val); per-row val QLIKE keyed (ticker, y_date)."""
    tr, va = df[df["split"] == "train"], df[df["split"] == "val"]
    booster, _ = trees.fit_one(tr[cols], tr["y"], va[cols], va["y"], PARAMS)
    n = booster.best_iteration
    smear = smearing_factor(tr["y"].to_numpy(), booster.predict(tr[cols], num_iteration=n))
    out = va[["ticker", "y_date"]].reset_index(drop=True)
    out["loss"] = qlike_loss(np.exp(2 * va["y"].to_numpy()),
                             np.exp(2 * booster.predict(va[cols], num_iteration=n)) * smear)
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
    base = base.dropna(subset=BASE + bl.SEC + bl.COR + MKT).reset_index(drop=True)
    print(f"Base sample: {len(base):,} rows (train+val)", flush=True)

    specs = {"none": BASE, "market": BASE + MKT, "sector": BASE + bl.SEC, "corr": BASE + bl.COR,
             "market+sector": BASE + MKT + bl.SEC, "market+corr": BASE + MKT + bl.COR}
    L = {}
    for name, cols in specs.items():
        L[name] = tree_losses(base, cols)
        print(f"  {name} fitted", flush=True)
    for s in SEEDS:
        rn = neighbor_features(rows, random_sector_edges(sec, s), random_corr_edges(cor, s))
        d = base[["ticker", "date", "split", "y", "y_date"] + BASE].merge(
            renamed(rn, "sec", "rsec"), on=["ticker", "date"], how="left").merge(
            renamed(rn, "cor", "rcor"), on=["ticker", "date"], how="left")
        L[f"rand_sector_s{s}"] = tree_losses(d.dropna(subset=RSEC), BASE + RSEC)
        L[f"rand_corr_s{s}"] = tree_losses(d.dropna(subset=RCOR), BASE + RCOR)
        print(f"  seed {s} fitted", flush=True)

    out = []

    def add(label, a, b):
        chg, p, chg_cov, n = compare(L[a], L[b], "loss")
        out.append({"comparison": label, "model": a, "versus": b, "qlike_change": chg, "dm_p": p,
                    "qlike_change_exCOVID": chg_cov, "rows": n})

    for m in ["market", "sector", "corr", "market+sector", "market+corr"]:
        add(f"{m} vs none", m, "none")
    for g in ["sector", "corr"]:
        add(f"{g} vs market", g, "market")
        add(f"market+{g} vs market", f"market+{g}", "market")
    for s in SEEDS:
        add(f"rand_sector_s{s} vs none", f"rand_sector_s{s}", "none")
        add(f"rand_corr_s{s} vs none", f"rand_corr_s{s}", "none")
        add(f"sector vs rand_sector_s{s}", "sector", f"rand_sector_s{s}")
        add(f"corr vs rand_corr_s{s}", "corr", f"rand_corr_s{s}")
    res = pd.DataFrame(out)
    RESULTS_DIR.mkdir(exist_ok=True)
    res.to_csv(RESULTS_DIR / "placebo_trees.csv", index=False, float_format=bl.CSV_FMT)

    fmt = lambda r: f"{r.qlike_change:+.2%}  p={r.dm_p:.3g}  exCOVID {r.qlike_change_exCOVID:+.2%}"
    print("\nGAIN OVER TREES WITH NO NEIGHBOR FEATURE (QLIKE, validation; negative = better)")
    for _, r in res[res["versus"] == "none"].iterrows():
        if "_s" not in r["model"]:
            print(f"  {r['model']:15s} {fmt(r)}")
    for kind in ["rand_sector", "rand_corr"]:
        sub = res[res["comparison"].str.fullmatch(f"{kind}_s\\d vs none")]
        print(f"  {kind:15s} mean {sub.qlike_change.mean():+.2%}  range [{sub.qlike_change.min():+.2%}, "
              f"{sub.qlike_change.max():+.2%}]  exCOVID mean {sub.qlike_change_exCOVID.mean():+.2%}")
    print("\nDOES THE GRAPH ADD ANYTHING BEYOND THE MARKET AVERAGE?")
    for _, r in res[res["comparison"].str.contains(r"^(?:sector|corr|market\+sector|market\+corr) vs market")].iterrows():
        print(f"  {r['comparison']:24s} {fmt(r)}")
    print("\nREAL GRAPH VS ITS RANDOM TWIN (same degrees; positive = real is worse)")
    for kind, real in [("rand_sector", "sector"), ("rand_corr", "corr")]:
        sub = res[res["comparison"].str.fullmatch(f"{real} vs {kind}_s\\d")]
        print(f"  {real} vs {kind}: mean {sub.qlike_change.mean():+.2%}  range [{sub.qlike_change.min():+.2%}, "
              f"{sub.qlike_change.max():+.2%}]  max p {sub.dm_p.max():.3g}  exCOVID mean {sub.qlike_change_exCOVID.mean():+.2%}")


if __name__ == "__main__":
    main()
