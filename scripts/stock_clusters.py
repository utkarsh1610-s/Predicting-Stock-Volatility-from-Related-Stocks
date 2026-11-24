"""Clustering task: which stocks' volatility moves together beyond the market, and do the groups
match the official GICS sectors?

  python scripts/stock_clusters.py         # training years only (2010-2019)

DEFINITION (resolves DECISIONS.md #53). GNNHAR has no per-stock embedding, so we cluster what the
graphs are meant to capture: co-movement in volatility that the market does not explain.
  1. For each stock, daily log volatility `lv` (winsorized, as in the features) on training days only.
  2. Remove the market: regress each stock's lv on the same-day mean lv of all OTHER stocks
     (leave-one-out, so a stock never explains itself) and keep the residual.
  3. Similarity = correlation of two stocks' residuals over the days both trade (at least
     MIN_OVERLAP days); distance = 1 - correlation. Stocks with fewer than MIN_DAYS training days
     are left out (late listings).
  4. Average-linkage hierarchical clustering on that distance, cut into k groups for k = 5..25.
     k = 11 (the number of GICS sectors) is the main comparison with sectors.

QUALITY AND MEANING
  * Silhouette (on the same distance) for every k, and for the GICS sectors used as cluster labels.
    Sectors get a score too, so we can say whether the data-driven groups separate co-movement
    better than the official ones.
  * Adjusted Rand index (ARI) between the clusters and the sectors: 1 = identical, about 0 = no
    more agreement than chance.
  * Mean residual correlation within vs between clusters.
  * Pairs for inspection by hand: the most correlated pairs from DIFFERENT sectors that share a
    cluster, and the least correlated pairs from the SAME sector that the clustering split.
Descriptive analysis, no decision rule.

Outputs (results/): stock_clusters.csv (ticker, company, sector, cluster at k=11),
cluster_quality.csv, cluster_summary.csv, cluster_pairs.csv.
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

import baselines as bl
from config import PROCESSED_DIR, RESULTS_DIR, SPLITS
from features import compute_features

MIN_DAYS, MIN_OVERLAP = 500, 250
K_MAIN, K_RANGE = 11, range(5, 26)
N_PAIRS = 25


def adjusted_rand_index(a, b):
    """ARI between two labelings (Hubert and Arabie 1985)."""
    a, b = pd.factorize(np.asarray(a))[0], pd.factorize(np.asarray(b))[0]
    table = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(table, (a, b), 1)
    comb = lambda x: x * (x - 1) / 2
    sum_ij = comb(table).sum()
    sum_a, sum_b = comb(table.sum(axis=1)).sum(), comb(table.sum(axis=0)).sum()
    expected = sum_a * sum_b / comb(len(a))
    top = (sum_a + sum_b) / 2
    return 1.0 if top == expected else float((sum_ij - expected) / (top - expected))


def silhouette(dist, labels):
    """Mean silhouette from a square distance matrix. Points alone in their cluster score 0."""
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return np.nan
    s = np.zeros(len(labels))
    for i in range(len(labels)):
        own = labels == labels[i]
        if own.sum() == 1:
            continue
        a = dist[i, own].sum() / (own.sum() - 1)
        b = min(dist[i, labels == c].mean() for c in uniq if c != labels[i])
        s[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(s.mean())


def market_residuals(lv):
    """lv: dates x tickers. Residual of each column on the leave-one-out cross-sectional mean."""
    total, count = np.nansum(lv.to_numpy(), axis=1), lv.notna().sum(axis=1).to_numpy()
    out = {}
    for t in lv.columns:
        x = lv[t].to_numpy()
        have = ~np.isnan(x) & (count > 1)
        mkt = (total[have] - x[have]) / (count[have] - 1)
        coef = bl.fit_ols(mkt[:, None], x[have])
        r = np.full(len(x), np.nan)
        r[have] = x[have] - bl.predict_ols(coef, mkt[:, None])
        out[t] = r
    return pd.DataFrame(out, index=lv.index)


def cluster(dist, k):
    z = linkage(squareform(dist, checks=False), method="average")
    return fcluster(z, t=k, criterion="maxclust")


def pairs_table(corr, meta, labels):
    """Most correlated cross-sector pairs sharing a cluster; least correlated same-sector pairs split apart."""
    t = corr.index.to_numpy()
    sec = meta.loc[t, "sector"].to_numpy()
    iu = np.triu_indices(len(t), 1)
    c = corr.to_numpy()[iu]
    same_sector, same_cluster = sec[iu[0]] == sec[iu[1]], labels[iu[0]] == labels[iu[1]]

    def take(mask, ascending, kind):
        idx = np.where(mask)[0]
        idx = idx[np.argsort(c[idx] if ascending else -c[idx], kind="stable")][:N_PAIRS]
        return pd.DataFrame({"kind": kind,
                             "ticker_a": t[iu[0][idx]], "company_a": meta.loc[t[iu[0][idx]], "company"].to_numpy(),
                             "sector_a": sec[iu[0][idx]],
                             "ticker_b": t[iu[1][idx]], "company_b": meta.loc[t[iu[1][idx]], "company"].to_numpy(),
                             "sector_b": sec[iu[1][idx]],
                             "residual_corr": c[idx], "same_cluster": same_cluster[idx]})

    return pd.concat([take(same_cluster & ~same_sector, False, "cross-sector, same cluster"),
                      take(~same_cluster & same_sector, True, "same sector, split apart")], ignore_index=True)


def analyse(lv, meta):
    """lv: training-day log volatility, dates x tickers. Returns the four output tables."""
    lv = lv.loc[:, lv.notna().sum() >= MIN_DAYS]
    corr = market_residuals(lv).corr(min_periods=MIN_OVERLAP)
    c = corr.to_numpy(copy=True)
    c[np.isnan(c)] = 0.0                                     # too little overlap: treat as unrelated
    np.fill_diagonal(c, 1.0)
    corr = pd.DataFrame(c, index=corr.index, columns=corr.columns)
    dist = np.clip(1.0 - c, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    sectors = meta.loc[corr.index, "sector"].to_numpy()

    quality = [{"labels": "gics_sector", "k": len(np.unique(sectors)),
                "silhouette": silhouette(dist, sectors), "ari_vs_sector": 1.0}]
    for k in K_RANGE:
        lab = cluster(dist, k)
        quality.append({"labels": "clusters", "k": k, "silhouette": silhouette(dist, lab),
                        "ari_vs_sector": adjusted_rand_index(lab, sectors)})
    quality = pd.DataFrame(quality)

    labels = cluster(dist, K_MAIN)
    members = pd.DataFrame({"ticker": corr.index, "company": meta.loc[corr.index, "company"].to_numpy(),
                            "sector": sectors, "cluster": labels})
    c = corr.to_numpy()
    off = ~np.eye(len(c), dtype=bool)
    summary = []
    for k, g in members.groupby("cluster"):
        inside = (labels == k)
        top = g["sector"].value_counts()
        summary.append({"cluster": k, "size": len(g), "top_sector": top.index[0],
                        "top_sector_share": top.iloc[0] / len(g), "n_sectors": len(top),
                        "mean_corr_within": c[np.ix_(inside, inside)][off[np.ix_(inside, inside)]].mean() if len(g) > 1 else np.nan,
                        "mean_corr_to_others": c[np.ix_(inside, ~inside)].mean() if (~inside).any() else np.nan,
                        "example_tickers": " ".join(g["ticker"].head(8))})
    summary = pd.DataFrame(summary).sort_values("size", ascending=False)
    return members, quality, summary, pairs_table(corr, meta, labels)


def main():
    stocks = pd.read_parquet(PROCESSED_DIR / "stocks.parquet")
    vix = pd.read_parquet(PROCESSED_DIR / "vix.parquet")
    rows, _ = compute_features(stocks, vix)
    a, b = SPLITS["train"]
    rows = rows[(rows["date"] >= a) & (rows["date"] <= b)]
    lv = rows.pivot(index="date", columns="ticker", values="lv")
    meta = stocks.drop_duplicates("ticker").set_index("ticker")[["company", "sector"]]
    members, quality, summary, pairs = analyse(lv, meta)

    RESULTS_DIR.mkdir(exist_ok=True)
    members.to_csv(RESULTS_DIR / "stock_clusters.csv", index=False)
    quality.to_csv(RESULTS_DIR / "cluster_quality.csv", index=False, float_format=bl.CSV_FMT)
    summary.to_csv(RESULTS_DIR / "cluster_summary.csv", index=False, float_format=bl.CSV_FMT)
    pairs.to_csv(RESULTS_DIR / "cluster_pairs.csv", index=False, float_format=bl.CSV_FMT)

    print(f"{len(members)} stocks clustered on market-residual volatility, {a[:4]}-{b[:4]}")
    print("\nQUALITY (silhouette on 1 - residual correlation; ARI vs GICS sectors)")
    print(quality.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nCLUSTERS AT k = {K_MAIN}")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\nPAIRS TO INSPECT")
    print(pairs.groupby("kind").head(8).to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
